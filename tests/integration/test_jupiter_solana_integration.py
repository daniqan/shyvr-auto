"""
Integration tests for Jupiter DEX + Solana Wallet integration.

This module tests the complete end-to-end integration between
JupiterDEXClient and SolanaWallet through the DEXWalletBridge.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.dex.jupiter_client import JupiterDEXClient
from src.dex.base import DEXConfig, SwapType
from src.wallet.solana_wallet import SolanaWallet
from src.wallet.base import WalletConfig
from src.trading.dex_wallet_bridge import DEXWalletBridge, SwapExecutionConfig
from src.wallet.base import Chain, NetworkType


@pytest.fixture
def jupiter_config():
    """Create Jupiter DEX configuration for testing."""
    return DEXConfig(
        chain=Chain.SOLANA,
        name="jupiter",
        max_slippage_bps=50,
        timeout_seconds=30,
        rate_limit_per_second=10
    )


@pytest.fixture
def solana_wallet_config():
    """Create Solana wallet configuration for testing."""
    return WalletConfig(
        chain=Chain.SOLANA,
        network=NetworkType.DEVNET,
        rpc_url="https://api.devnet.solana.com",
        wallet_address="5xot9PVkphiX2adznghwrAuxGs2zeWisNSxMcq7rJB5BqJ4b",
        private_key="test_private_key_here"
    )


@pytest.fixture
def mock_jupiter_client(jupiter_config):
    """Create mocked Jupiter client for integration testing."""
    client = JupiterDEXClient(jupiter_config)
    
    # Mock the HTTP session and requests
    client._make_request = AsyncMock()
    client._connected = True
    
    return client


@pytest.fixture
def mock_solana_wallet(solana_wallet_config):
    """Create mocked Solana wallet for integration testing."""
    wallet = SolanaWallet(solana_wallet_config)
    
    # Mock the connection and keypair
    wallet._connected = True
    wallet._wallet_address = solana_wallet_config.wallet_address
    wallet.keypair = MagicMock()
    wallet.client = AsyncMock()
    
    return wallet


@pytest.fixture
def integration_bridge(mock_jupiter_client, mock_solana_wallet):
    """Create DEX-Wallet bridge with mocked components."""
    config = SwapExecutionConfig(
        enable_balance_checks=True,
        enable_account_preparation=True
    )
    return DEXWalletBridge(mock_jupiter_client, mock_solana_wallet, config)


class TestJupiterSolanaIntegration:
    """Integration test suite for Jupiter + Solana wallet."""
    
    @pytest.mark.asyncio
    async def test_complete_swap_flow_success(self, integration_bridge, mock_jupiter_client, mock_solana_wallet):
        """Test complete successful swap flow from quote to execution."""
        # Step 1: Mock Jupiter quote response
        quote_response = {
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "inAmount": "1000000000",
            "outAmount": "100000000",
            "priceImpactPct": "0.25",
            "routePlan": [
                {"swapInfo": {"outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"}}
            ]
        }
        
        # Step 2: Mock Jupiter swap transaction response
        swap_response = {
            "swapTransaction": "base64_encoded_transaction_data",
            "lastValidBlockHeight": 12345
        }
        
        # Configure Jupiter client mocks
        mock_jupiter_client._make_request.side_effect = [
            quote_response,  # For get_quote
            swap_response    # For execute_swap
        ]
        
        # Step 3: Mock wallet balance check
        mock_solana_wallet.check_token_balance_for_swap = AsyncMock(return_value=True)
        
        # Step 4: Mock wallet account preparation
        mock_solana_wallet.prepare_swap_accounts = AsyncMock(return_value={
            "input_account": None,  # SOL doesn't need token account
            "output_account": "usdc_token_account_address"
        })
        
        # Step 5: Mock wallet DEX swap execution
        from src.wallet.base import TransactionResult, TransactionStatus
        mock_solana_wallet.execute_dex_swap = AsyncMock(return_value=TransactionResult(
            transaction_hash="actual_transaction_hash_123",
            status=TransactionStatus.PENDING
        ))
        
        # Execute complete swap flow
        quote = await integration_bridge.get_quote(
            input_token="SOL",
            output_token="USDC",
            amount=Decimal("1.0")
        )
        
        swap_result = await integration_bridge.execute_swap(quote)
        
        # Verify quote properties
        assert quote.input_token == "So11111111111111111111111111111111111111112"
        assert quote.output_token == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        assert quote.input_amount == Decimal("1000000000")
        assert quote.output_amount == Decimal("100000000")
        assert quote.price_impact_bps == 25
        assert quote.dex_name == "jupiter"
        
        # Verify swap result
        assert swap_result.transaction_hash == "actual_transaction_hash_123"
        assert swap_result.input_token == quote.input_token
        assert swap_result.output_token == quote.output_token
        assert swap_result.dex_name == "jupiter"
        
        # Verify method call sequence
        assert mock_jupiter_client._make_request.call_count == 2
        mock_solana_wallet.check_token_balance_for_swap.assert_called_once()
        mock_solana_wallet.prepare_swap_accounts.assert_called_once()
        mock_solana_wallet.execute_dex_swap.assert_called_once_with(
            swap_transaction_data="base64_encoded_transaction_data",
            last_valid_block_height=12345
        )
    
    @pytest.mark.asyncio
    async def test_swap_flow_insufficient_balance(self, integration_bridge, mock_jupiter_client, mock_solana_wallet):
        """Test swap flow fails with insufficient balance."""
        # Mock Jupiter quote response
        quote_response = {
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "inAmount": "1000000000",
            "outAmount": "100000000",
            "priceImpactPct": "0.25",
            "routePlan": []
        }
        mock_jupiter_client._make_request.return_value = quote_response
        
        # Mock insufficient balance
        mock_solana_wallet.check_token_balance_for_swap = AsyncMock(return_value=False)
        
        # Get quote
        quote = await integration_bridge.get_quote("SOL", "USDC", Decimal("1.0"))
        
        # Execute swap should fail with insufficient balance
        from src.trading.dex_wallet_bridge import InsufficientBalanceError
        with pytest.raises(InsufficientBalanceError):
            await integration_bridge.execute_swap(quote)
    
    @pytest.mark.asyncio
    async def test_swap_flow_account_preparation_error(self, integration_bridge, mock_jupiter_client, mock_solana_wallet):
        """Test swap flow continues despite account preparation errors."""
        # Mock Jupiter responses
        quote_response = {
            "inputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "outputMint": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
            "inAmount": "100000000",
            "outAmount": "100000000",
            "priceImpactPct": "0.1",
            "routePlan": []
        }
        
        swap_response = {
            "swapTransaction": "transaction_data",
            "lastValidBlockHeight": 12345
        }
        
        mock_jupiter_client._make_request.side_effect = [quote_response, swap_response]
        
        # Mock successful balance check
        mock_solana_wallet.check_token_balance_for_swap = AsyncMock(return_value=True)
        
        # Mock account preparation failure (should not fail the swap)
        mock_solana_wallet.prepare_swap_accounts = AsyncMock(side_effect=Exception("Account prep failed"))
        
        # Mock successful swap execution
        from src.wallet.base import TransactionResult, TransactionStatus
        mock_solana_wallet.execute_dex_swap = AsyncMock(return_value=TransactionResult(
            transaction_hash="tx_hash_after_prep_error",
            status=TransactionStatus.PENDING
        ))
        
        # Execute swap flow
        quote = await integration_bridge.get_quote("USDC", "USDT", Decimal("100.0"))
        swap_result = await integration_bridge.execute_swap(quote)
        
        # Swap should still succeed despite account preparation error
        assert swap_result.transaction_hash == "tx_hash_after_prep_error"
        mock_solana_wallet.execute_dex_swap.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_swap_flow_jupiter_api_error(self, integration_bridge, mock_jupiter_client, mock_solana_wallet):
        """Test swap flow handles Jupiter API errors."""
        # Mock Jupiter API error
        mock_jupiter_client._make_request.side_effect = Exception("Jupiter API unavailable")
        
        # Quote generation should fail
        from src.dex.base import DEXError
        with pytest.raises(DEXError, match="Quote generation failed"):
            await integration_bridge.get_quote("SOL", "USDC", Decimal("1.0"))
    
    @pytest.mark.asyncio
    async def test_swap_flow_wallet_execution_error(self, integration_bridge, mock_jupiter_client, mock_solana_wallet):
        """Test swap flow handles wallet execution errors."""
        # Mock successful Jupiter responses
        quote_response = {
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "inAmount": "1000000000",
            "outAmount": "100000000",
            "priceImpactPct": "0.25",
            "routePlan": []
        }
        
        swap_response = {
            "swapTransaction": "transaction_data",
            "lastValidBlockHeight": 12345
        }
        
        mock_jupiter_client._make_request.side_effect = [quote_response, swap_response]
        
        # Mock successful pre-checks
        mock_solana_wallet.check_token_balance_for_swap = AsyncMock(return_value=True)
        mock_solana_wallet.prepare_swap_accounts = AsyncMock(return_value={})
        
        # Mock wallet execution failure
        from src.wallet.base import WalletTransactionError
        mock_solana_wallet.execute_dex_swap = AsyncMock(
            side_effect=WalletTransactionError("Transaction signing failed")
        )
        
        # Execute swap flow
        quote = await integration_bridge.get_quote("SOL", "USDC", Decimal("1.0"))
        
        # Swap execution should fail
        from src.trading.dex_wallet_bridge import SwapExecutionError
        with pytest.raises(SwapExecutionError, match="Swap execution failed"):
            await integration_bridge.execute_swap(quote)
    
    @pytest.mark.asyncio
    async def test_cost_estimation_integration(self, integration_bridge, mock_jupiter_client, mock_solana_wallet):
        """Test integrated cost estimation with Jupiter and Solana wallet."""
        # Mock Jupiter quote response
        quote_response = {
            "inputMint": "So11111111111111111111111111111111111111112",
            "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "inAmount": "1000000000",
            "outAmount": "100000000",
            "priceImpactPct": "0.25",
            "routePlan": [{"swapInfo": {"outputMint": "intermediate_token"}}]
        }
        mock_jupiter_client._make_request.return_value = quote_response
        
        # Mock gas estimation and price
        mock_jupiter_client.estimate_gas = AsyncMock(return_value=150000)
        mock_solana_wallet.get_gas_price.return_value = Decimal("0.000005")
        
        # Estimate swap cost
        cost_breakdown = await integration_bridge.estimate_swap_cost(
            input_token="SOL",
            output_token="USDC",
            amount=Decimal("1.0")
        )
        
        # Verify cost breakdown
        assert cost_breakdown["input_amount"] == Decimal("1000000000")
        assert cost_breakdown["output_amount"] == Decimal("100000000")
        assert cost_breakdown["estimated_gas"] == 150000
        assert cost_breakdown["gas_price"] == Decimal("0.000005")
        assert cost_breakdown["estimated_tx_fee"] == Decimal("0.75")  # 150000 * 0.000005
        assert cost_breakdown["total_cost"] == Decimal("1000000000.75")
        assert cost_breakdown["dex_name"] == "jupiter"
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, integration_bridge, mock_jupiter_client):
        """Test integrated health check across Jupiter and Solana wallet."""
        # Mock Jupiter health check
        mock_jupiter_client.health_check.return_value = {
            "connected": True,
            "dex_name": "jupiter",
            "chain": "solana",
            "supported_tokens": 1000,
            "status": "healthy"
        }
        
        # Perform health check
        health_status = await integration_bridge.health_check()
        
        # Verify health status
        assert health_status["dex_health"]["status"] == "healthy"
        assert health_status["wallet_connected"] is True
        assert health_status["wallet_address"] == "5xot9PVkphiX2adznghwrAuxGs2zeWisNSxMcq7rJB5BqJ4b"
        assert health_status["chain"] == "solana"
        assert health_status["bridge_status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_swap_status_tracking_integration(self, integration_bridge, mock_solana_wallet):
        """Test swap status tracking through wallet integration."""
        # Mock wallet transaction status
        from src.wallet.base import TransactionResult, TransactionStatus
        mock_solana_wallet.get_transaction_status.return_value = TransactionResult(
            transaction_hash="test_tx_hash",
            status=TransactionStatus.CONFIRMED,
            block_number=12345
        )
        
        # Get swap status
        swap_status = await integration_bridge.get_swap_status("test_tx_hash")
        
        # Verify status mapping
        from src.dex.base import SwapStatus
        assert swap_status.status == SwapStatus.CONFIRMED
        assert swap_status.transaction_hash == "test_tx_hash"
        assert swap_status.block_number == 12345
        assert swap_status.dex_name == "jupiter"
    
    @pytest.mark.asyncio
    async def test_context_manager_integration(self, mock_jupiter_client, mock_solana_wallet):
        """Test bridge context manager with Jupiter and Solana wallet."""
        # Mock connection states
        mock_jupiter_client.is_connected = False
        mock_solana_wallet.is_connected = False
        mock_jupiter_client.close = AsyncMock()
        
        # Use bridge as context manager
        async with DEXWalletBridge(mock_jupiter_client, mock_solana_wallet) as bridge:
            assert isinstance(bridge, DEXWalletBridge)
            assert bridge.chain == Chain.SOLANA
            
            # Verify connections were established
            mock_jupiter_client.connect.assert_called_once()
            mock_solana_wallet.connect.assert_called_once()
        
        # Verify cleanup
        mock_jupiter_client.close.assert_called_once()
        mock_solana_wallet.disconnect.assert_called_once()


class TestJupiterSolanaErrorScenarios:
    """Test error scenarios in Jupiter + Solana integration."""
    
    @pytest.mark.asyncio
    async def test_chain_mismatch_error(self, jupiter_config, solana_wallet_config):
        """Test error when Jupiter and wallet chains don't match."""
        # Create mismatched configurations
        jupiter_client = JupiterDEXClient(jupiter_config)
        
        ethereum_wallet_config = solana_wallet_config
        ethereum_wallet_config.chain = Chain.ETHEREUM
        solana_wallet = SolanaWallet(ethereum_wallet_config)
        
        # Bridge creation should fail
        from src.trading.dex_wallet_bridge import DEXWalletBridgeError
        with pytest.raises(DEXWalletBridgeError, match="Chain mismatch"):
            DEXWalletBridge(jupiter_client, solana_wallet)
    
    @pytest.mark.asyncio
    async def test_unsupported_chain_swap_execution(self, mock_jupiter_client, mock_solana_wallet):
        """Test error when trying to execute swap on unsupported chain."""
        # Force chain mismatch in bridge
        bridge = DEXWalletBridge(mock_jupiter_client, mock_solana_wallet)
        bridge.chain = Chain.ETHEREUM  # Force unsupported chain
        
        # Mock quote and swap result
        from src.dex.base import SwapQuote, SwapResult, SwapStatus
        quote = SwapQuote(
            input_token="ETH",
            output_token="USDC",
            input_amount=Decimal("1"),
            output_amount=Decimal("1000"),
            price=Decimal("1000"),
            price_impact_bps=10,
            slippage_bps=50,
            dex_name="jupiter"
        )
        
        swap_result = SwapResult(
            transaction_hash="pending",
            status=SwapStatus.PENDING,
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            dex_name="jupiter",
            transaction_data={"swapTransaction": "tx_data"}
        )
        
        # Mock successful pre-checks
        mock_solana_wallet.check_token_balance_for_swap = AsyncMock(return_value=True)
        mock_jupiter_client.execute_swap.return_value = swap_result
        
        # Swap execution should fail for unsupported chain
        from src.trading.dex_wallet_bridge import SwapExecutionError
        with pytest.raises(SwapExecutionError, match="DEX swap execution not supported for chain"):
            await bridge.execute_swap(quote, enable_pre_checks=False)
    
    @pytest.mark.asyncio
    async def test_quote_validation_failure_scenarios(self, integration_bridge, mock_jupiter_client):
        """Test various quote validation failure scenarios."""
        # Mock expired quote
        from src.dex.base import SwapQuote
        from datetime import datetime, timedelta
        
        expired_quote = SwapQuote(
            input_token="SOL",
            output_token="USDC",
            input_amount=Decimal("1"),
            output_amount=Decimal("100"),
            price=Decimal("100"),
            price_impact_bps=10,
            slippage_bps=50,
            valid_until=datetime.now() - timedelta(minutes=1),  # Expired
            dex_name="jupiter"
        )
        
        mock_jupiter_client.validate_quote.return_value = False
        
        # Swap should fail validation
        from src.trading.dex_wallet_bridge import SwapExecutionError
        with pytest.raises(SwapExecutionError, match="Quote validation failed"):
            await integration_bridge.execute_swap(expired_quote)