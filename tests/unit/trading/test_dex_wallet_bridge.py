"""
Tests for DEX-Wallet integration bridge.

This module tests the DEXWalletBridge class that orchestrates
swap execution between DEX clients and wallet implementations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.trading.dex_wallet_bridge import (
    DEXWalletBridge,
    SwapExecutionConfig,
    DEXWalletBridgeError,
    InsufficientBalanceError,
    SwapExecutionError
)
from src.dex.base import (
    SwapQuote,
    SwapResult,
    SwapStatus,
    SwapType,
    DEXBase,
    DEXConfig,
    DEXError
)
from src.wallet.base import (
    WalletBase,
    WalletConfig,
    TransactionResult,
    TransactionStatus,
    WalletBalance,
    Chain,
    NetworkType
)
from src.utils.base import Chain


@pytest.fixture
def mock_dex_client():
    """Create mock DEX client for testing."""
    dex = MagicMock(spec=DEXBase)
    dex.chain = Chain.SOLANA
    dex.name = "jupiter"
    dex.is_connected = True
    
    # Mock methods
    dex.connect = AsyncMock(return_value=True)
    dex.disconnect = AsyncMock()
    dex.validate_quote = AsyncMock(return_value=True)
    dex.health_check = AsyncMock(return_value={
        "status": "healthy",
        "connected": True,
        "dex_name": "jupiter",
        "chain": "solana"
    })
    
    return dex


@pytest.fixture
def mock_wallet():
    """Create mock wallet for testing."""
    wallet = MagicMock(spec=WalletBase)
    wallet.config = WalletConfig(
        chain=Chain.SOLANA,
        network=NetworkType.DEVNET,
        rpc_url="https://api.devnet.solana.com",
        wallet_address="5xot9PVkphiX2adznghwrAuxGs2zeWisNSxMcq7rJB5BqJ4b"
    )
    wallet.is_connected = True
    wallet.wallet_address = "5xot9PVkphiX2adznghwrAuxGs2zeWisNSxMcq7rJB5BqJ4b"
    
    # Mock methods
    wallet.connect = AsyncMock(return_value=True)
    wallet.disconnect = AsyncMock()
    wallet.get_gas_price = AsyncMock(return_value=Decimal("0.000005"))
    
    return wallet


@pytest.fixture
def sample_quote():
    """Create sample swap quote for testing."""
    return SwapQuote(
        input_token="So11111111111111111111111111111111111111112",  # SOL
        output_token="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        input_amount=Decimal("1000000000"),  # 1 SOL in lamports
        output_amount=Decimal("100000000"),  # 100 USDC in smallest units
        price=Decimal("100"),
        price_impact_bps=25,  # 0.25%
        slippage_bps=50,      # 0.5%
        valid_until=datetime.now() + timedelta(minutes=1),
        dex_name="jupiter",
        quote_id="test_quote_123"
    )


@pytest.fixture
def bridge(mock_dex_client, mock_wallet):
    """Create DEX-Wallet bridge for testing."""
    return DEXWalletBridge(mock_dex_client, mock_wallet)


class TestDEXWalletBridge:
    """Test suite for DEX-Wallet bridge integration."""
    
    def test_bridge_initialization_success(self, mock_dex_client, mock_wallet):
        """Test successful bridge initialization."""
        bridge = DEXWalletBridge(mock_dex_client, mock_wallet)
        
        assert bridge.dex_client == mock_dex_client
        assert bridge.wallet == mock_wallet
        assert bridge.chain == Chain.SOLANA
        assert isinstance(bridge.config, SwapExecutionConfig)
    
    def test_bridge_initialization_chain_mismatch(self, mock_dex_client, mock_wallet):
        """Test bridge initialization fails with chain mismatch."""
        # Change wallet chain to create mismatch
        mock_wallet.config.chain = Chain.ETHEREUM
        
        with pytest.raises(DEXWalletBridgeError, match="Chain mismatch"):
            DEXWalletBridge(mock_dex_client, mock_wallet)
    
    @pytest.mark.asyncio
    async def test_get_quote_success(self, bridge, mock_dex_client, sample_quote):
        """Test successful quote generation."""
        mock_dex_client.get_quote.return_value = sample_quote
        
        result = await bridge.get_quote(
            input_token="SOL",
            output_token="USDC",
            amount=Decimal("1.0")
        )
        
        assert result == sample_quote
        mock_dex_client.get_quote.assert_called_once_with(
            input_token="SOL",
            output_token="USDC",
            amount=Decimal("1.0"),
            swap_type=SwapType.EXACT_INPUT,
            slippage_bps=None
        )
    
    @pytest.mark.asyncio
    async def test_get_quote_failure(self, bridge, mock_dex_client):
        """Test quote generation failure."""
        mock_dex_client.get_quote.side_effect = Exception("Quote failed")
        
        with pytest.raises(DEXError, match="Quote generation failed"):
            await bridge.get_quote("SOL", "USDC", Decimal("1.0"))
    
    @pytest.mark.asyncio
    async def test_execute_swap_success(self, bridge, mock_dex_client, mock_wallet, sample_quote):
        """Test successful swap execution."""
        # Mock balance check
        mock_wallet.check_token_balance_for_swap = AsyncMock(return_value=True)
        
        # Mock account preparation
        mock_wallet.prepare_swap_accounts = AsyncMock(return_value={
            "input_account": "input_account_address",
            "output_account": "output_account_address"
        })
        
        # Mock DEX swap preparation
        swap_result = SwapResult(
            transaction_hash="pending",
            status=SwapStatus.PENDING,
            input_token=sample_quote.input_token,
            output_token=sample_quote.output_token,
            input_amount=sample_quote.input_amount,
            dex_name="jupiter",
            quote_used=sample_quote,
            transaction_data={
                "swapTransaction": "base64_encoded_transaction",
                "lastValidBlockHeight": 12345
            }
        )
        mock_dex_client.execute_swap.return_value = swap_result
        
        # Mock wallet execution
        transaction_result = TransactionResult(
            transaction_hash="actual_tx_hash",
            status=TransactionStatus.PENDING
        )
        mock_wallet.execute_dex_swap = AsyncMock(return_value=transaction_result)
        
        # Execute swap
        result = await bridge.execute_swap(sample_quote)
        
        # Verify result
        assert isinstance(result, SwapResult)
        assert result.transaction_hash == "actual_tx_hash"
        assert result.status == SwapStatus.PENDING
        
        # Verify method calls
        mock_dex_client.validate_quote.assert_called_once_with(sample_quote)
        mock_wallet.check_token_balance_for_swap.assert_called_once()
        mock_wallet.prepare_swap_accounts.assert_called_once()
        mock_dex_client.execute_swap.assert_called_once()
        mock_wallet.execute_dex_swap.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_swap_insufficient_balance(self, bridge, mock_wallet, sample_quote):
        """Test swap execution fails with insufficient balance."""
        # Mock insufficient balance
        mock_wallet.check_token_balance_for_swap = AsyncMock(return_value=False)
        
        with pytest.raises(InsufficientBalanceError):
            await bridge.execute_swap(sample_quote)
    
    @pytest.mark.asyncio
    async def test_execute_swap_quote_validation_failure(self, bridge, mock_dex_client, sample_quote):
        """Test swap execution fails with invalid quote."""
        mock_dex_client.validate_quote.return_value = False
        
        with pytest.raises(SwapExecutionError, match="Quote validation failed"):
            await bridge.execute_swap(sample_quote)
    
    @pytest.mark.asyncio
    async def test_execute_swap_no_transaction_data(self, bridge, mock_dex_client, mock_wallet, sample_quote):
        """Test swap execution fails without transaction data."""
        # Mock successful pre-checks
        mock_wallet.check_token_balance_for_swap = AsyncMock(return_value=True)
        mock_wallet.prepare_swap_accounts = AsyncMock(return_value={})
        
        # Mock swap result without transaction data
        swap_result = SwapResult(
            transaction_hash="pending",
            status=SwapStatus.PENDING,
            input_token=sample_quote.input_token,
            output_token=sample_quote.output_token,
            input_amount=sample_quote.input_amount,
            dex_name="jupiter"
        )
        mock_dex_client.execute_swap.return_value = swap_result
        
        with pytest.raises(SwapExecutionError, match="No transaction data available"):
            await bridge.execute_swap(sample_quote)
    
    @pytest.mark.asyncio
    async def test_execute_swap_wallet_execution_failure(self, bridge, mock_dex_client, mock_wallet, sample_quote):
        """Test swap execution fails during wallet execution."""
        # Mock successful pre-checks
        mock_wallet.check_token_balance_for_swap = AsyncMock(return_value=True)
        mock_wallet.prepare_swap_accounts = AsyncMock(return_value={})
        
        # Mock swap result with transaction data
        swap_result = SwapResult(
            transaction_hash="pending",
            status=SwapStatus.PENDING,
            input_token=sample_quote.input_token,
            output_token=sample_quote.output_token,
            input_amount=sample_quote.input_amount,
            dex_name="jupiter",
            transaction_data={"swapTransaction": "tx_data"}
        )
        mock_dex_client.execute_swap.return_value = swap_result
        
        # Mock wallet execution failure
        mock_wallet.execute_dex_swap = AsyncMock(side_effect=Exception("Wallet execution failed"))
        
        with pytest.raises(SwapExecutionError, match="Swap execution failed"):
            await bridge.execute_swap(sample_quote)
    
    @pytest.mark.asyncio
    async def test_execute_swap_skip_pre_checks(self, bridge, mock_dex_client, mock_wallet, sample_quote):
        """Test swap execution with pre-checks disabled."""
        # Mock successful execution
        swap_result = SwapResult(
            transaction_hash="pending",
            status=SwapStatus.PENDING,
            input_token=sample_quote.input_token,
            output_token=sample_quote.output_token,
            input_amount=sample_quote.input_amount,
            dex_name="jupiter",
            transaction_data={"swapTransaction": "tx_data"}
        )
        mock_dex_client.execute_swap.return_value = swap_result
        
        transaction_result = TransactionResult(
            transaction_hash="actual_tx_hash",
            status=TransactionStatus.PENDING
        )
        mock_wallet.execute_dex_swap = AsyncMock(return_value=transaction_result)
        
        # Execute swap without pre-checks
        result = await bridge.execute_swap(sample_quote, enable_pre_checks=False)
        
        # Verify result
        assert result.transaction_hash == "actual_tx_hash"
        
        # Verify pre-checks were skipped
        mock_dex_client.validate_quote.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_swap_status_success(self, bridge, mock_wallet):
        """Test successful swap status retrieval."""
        transaction_result = TransactionResult(
            transaction_hash="test_tx_hash",
            status=TransactionStatus.CONFIRMED,
            block_number=12345
        )
        mock_wallet.get_transaction_status.return_value = transaction_result
        
        result = await bridge.get_swap_status("test_tx_hash")
        
        assert isinstance(result, SwapResult)
        assert result.transaction_hash == "test_tx_hash"
        assert result.status == SwapStatus.CONFIRMED
        assert result.block_number == 12345
        assert result.dex_name == "jupiter"
    
    @pytest.mark.asyncio
    async def test_get_swap_status_failure(self, bridge, mock_wallet):
        """Test swap status retrieval handles errors."""
        mock_wallet.get_transaction_status.side_effect = Exception("Status check failed")
        
        result = await bridge.get_swap_status("test_tx_hash")
        
        assert result.status == SwapStatus.FAILED
        assert "Status check failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_estimate_swap_cost_success(self, bridge, mock_dex_client, mock_wallet, sample_quote):
        """Test successful swap cost estimation."""
        # Mock quote generation
        mock_dex_client.get_quote.return_value = sample_quote
        
        # Mock gas estimation
        mock_dex_client.estimate_gas.return_value = 150000
        
        result = await bridge.estimate_swap_cost("SOL", "USDC", Decimal("1.0"))
        
        assert "input_amount" in result
        assert "output_amount" in result
        assert "estimated_gas" in result
        assert "estimated_tx_fee" in result
        assert "total_cost" in result
        assert result["dex_name"] == "jupiter"
    
    @pytest.mark.asyncio
    async def test_estimate_swap_cost_failure(self, bridge, mock_dex_client):
        """Test swap cost estimation handles errors."""
        mock_dex_client.get_quote.side_effect = Exception("Cost estimation failed")
        
        with pytest.raises(DEXError, match="Cost estimation failed"):
            await bridge.estimate_swap_cost("SOL", "USDC", Decimal("1.0"))
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, bridge, mock_dex_client):
        """Test successful bridge health check."""
        result = await bridge.health_check()
        
        assert "dex_health" in result
        assert "wallet_connected" in result
        assert "wallet_address" in result
        assert "chain" in result
        assert "bridge_status" in result
        assert result["bridge_status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy_dex(self, bridge, mock_dex_client):
        """Test bridge health check with unhealthy DEX."""
        mock_dex_client.health_check.return_value = {"status": "unhealthy"}
        
        result = await bridge.health_check()
        
        assert result["bridge_status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_health_check_disconnected_wallet(self, bridge, mock_wallet):
        """Test bridge health check with disconnected wallet."""
        mock_wallet.is_connected = False
        
        result = await bridge.health_check()
        
        assert result["bridge_status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_close(self, bridge, mock_dex_client, mock_wallet):
        """Test bridge close method."""
        mock_dex_client.close = AsyncMock()
        
        await bridge.close()
        
        mock_dex_client.close.assert_called_once()
        mock_wallet.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_dex_client, mock_wallet):
        """Test bridge as async context manager."""
        mock_dex_client.is_connected = False
        mock_wallet.is_connected = False
        mock_dex_client.close = AsyncMock()
        
        async with DEXWalletBridge(mock_dex_client, mock_wallet) as bridge:
            assert isinstance(bridge, DEXWalletBridge)
            # Verify connections were established
            mock_dex_client.connect.assert_called_once()
            mock_wallet.connect.assert_called_once()
        
        # Verify cleanup on exit
        mock_dex_client.close.assert_called_once()
        mock_wallet.disconnect.assert_called_once()


class TestSwapExecutionConfig:
    """Test suite for SwapExecutionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SwapExecutionConfig()
        
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.enable_balance_checks is True
        assert config.enable_account_preparation is True
        assert config.confirmation_timeout_seconds == 60
        assert config.enable_slippage_protection is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SwapExecutionConfig(
            max_retries=5,
            retry_delay_seconds=2.0,
            enable_balance_checks=False,
            enable_account_preparation=False,
            confirmation_timeout_seconds=120,
            enable_slippage_protection=False
        )
        
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0
        assert config.enable_balance_checks is False
        assert config.enable_account_preparation is False
        assert config.confirmation_timeout_seconds == 120
        assert config.enable_slippage_protection is False


class TestDEXWalletBridgeErrors:
    """Test suite for bridge-specific errors."""
    
    def test_dex_wallet_bridge_error(self):
        """Test base DEXWalletBridgeError."""
        error = DEXWalletBridgeError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_insufficient_balance_error(self):
        """Test InsufficientBalanceError."""
        error = InsufficientBalanceError("Not enough tokens")
        assert str(error) == "Not enough tokens"
        assert isinstance(error, DEXWalletBridgeError)
    
    def test_swap_execution_error(self):
        """Test SwapExecutionError."""
        error = SwapExecutionError("Swap failed")
        assert str(error) == "Swap failed"
        assert isinstance(error, DEXWalletBridgeError)