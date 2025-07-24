"""
Tests for Solana wallet DEX integration methods.

This module tests the new DEX-specific methods added to SolanaWallet
for integration with DEX clients like Jupiter.
"""

import pytest
import asyncio
import base64
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.wallet.solana_wallet import SolanaWallet
from src.wallet.base import (
    WalletConfig,
    TransactionResult,
    TransactionStatus,
    Chain,
    NetworkType,
    WalletTransactionError,
    WalletError
)


@pytest.fixture
def solana_config():
    """Create test Solana wallet configuration."""
    return WalletConfig(
        chain=Chain.SOLANA,
        network=NetworkType.DEVNET,
        rpc_url="https://api.devnet.solana.com",
        wallet_address="5xot9PVkphiX2adznghwrAuxGs2zeWisNSxMcq7rJB5BqJ4b",
        private_key="2b5e8e9f1c3d8f9a2b5e8e9f1c3d8f9a2b5e8e9f1c3d8f9a2b5e8e9f1c3d8f9a2b5e8e9f1c3d8f9a2b5e8e9f1c3d8f9a"
    )


@pytest.fixture
def mock_solana_wallet(solana_config):
    """Create mock Solana wallet for testing."""
    wallet = SolanaWallet(solana_config)
    
    # Mock the keypair and client
    mock_keypair = MagicMock()
    mock_keypair.pubkey.return_value = MagicMock()
    
    mock_client = AsyncMock()
    
    wallet.keypair = mock_keypair
    wallet.client = mock_client
    wallet._connected = True
    wallet._wallet_address = solana_config.wallet_address
    
    return wallet


@pytest.fixture
def sample_swap_transaction():
    """Create sample swap transaction data for testing."""
    # This is a mock base64-encoded transaction string
    return "mock_base64_transaction_data"


class TestSolanaWalletDEXIntegration:
    """Test suite for Solana wallet DEX integration methods."""
    
    @pytest.mark.asyncio
    async def test_execute_dex_swap_success_versioned_transaction(self, mock_solana_wallet, sample_swap_transaction):
        """Test successful DEX swap execution with VersionedTransaction."""
        # Mock transaction parsing and execution
        with patch('base64.b64decode') as mock_decode, \
             patch('solders.transaction.VersionedTransaction') as mock_versioned_tx, \
             patch('solders.transaction.Transaction') as mock_tx:
            
            # Setup mocks
            mock_decode.return_value = b"mock_transaction_bytes"
            mock_transaction = MagicMock()
            mock_transaction.sign = MagicMock()
            mock_versioned_tx.from_bytes.return_value = mock_transaction
            
            # Mock successful submission
            mock_response = MagicMock()
            mock_response.value = "mock_transaction_hash"
            mock_solana_wallet.client.send_raw_transaction.return_value = mock_response
            
            # Execute swap
            result = await mock_solana_wallet.execute_dex_swap(
                swap_transaction_data=sample_swap_transaction,
                last_valid_block_height=12345
            )
            
            # Verify result
            assert isinstance(result, TransactionResult)
            assert result.transaction_hash == "mock_transaction_hash"
            assert result.status == TransactionStatus.PENDING
            
            # Verify transaction was signed
            mock_transaction.sign.assert_called_once_with([mock_solana_wallet.keypair])
            
            # Verify transaction was submitted
            mock_solana_wallet.client.send_raw_transaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_dex_swap_success_legacy_transaction(self, mock_solana_wallet, sample_swap_transaction):
        """Test successful DEX swap execution with legacy Transaction."""
        # Mock transaction parsing to fall back to legacy
        with patch('base64.b64decode') as mock_decode, \
             patch('solders.transaction.VersionedTransaction') as mock_versioned_tx, \
             patch('solders.transaction.Transaction') as mock_tx:
            
            # Setup mocks - VersionedTransaction fails, legacy succeeds
            mock_decode.return_value = b"mock_transaction_bytes"
            mock_versioned_tx.from_bytes.side_effect = Exception("Not a versioned transaction")
            
            mock_transaction = MagicMock()
            mock_transaction.sign = MagicMock()
            mock_tx.from_bytes.return_value = mock_transaction
            
            # Mock successful submission
            mock_response = MagicMock()
            mock_response.value = "legacy_transaction_hash"
            mock_solana_wallet.client.send_transaction.return_value = mock_response
            
            # Execute swap
            result = await mock_solana_wallet.execute_dex_swap(sample_swap_transaction)
            
            # Verify result
            assert result.transaction_hash == "legacy_transaction_hash"
            assert result.status == TransactionStatus.PENDING
            
            # Verify legacy transaction path was used
            mock_solana_wallet.client.send_transaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_dex_swap_read_only_mode(self, mock_solana_wallet, sample_swap_transaction):
        """Test DEX swap execution fails in read-only mode."""
        mock_solana_wallet.keypair = None
        
        with pytest.raises(WalletTransactionError, match="Cannot execute DEX swaps in read-only mode"):
            await mock_solana_wallet.execute_dex_swap(sample_swap_transaction)
    
    @pytest.mark.asyncio
    async def test_execute_dex_swap_not_connected(self, mock_solana_wallet, sample_swap_transaction):
        """Test DEX swap execution fails when wallet not connected."""
        mock_solana_wallet._connected = False
        
        with pytest.raises(WalletTransactionError, match="Wallet not connected"):
            await mock_solana_wallet.execute_dex_swap(sample_swap_transaction)
    
    @pytest.mark.asyncio
    async def test_execute_dex_swap_invalid_transaction_data(self, mock_solana_wallet):
        """Test DEX swap execution fails with invalid transaction data."""
        with pytest.raises(WalletTransactionError, match="Failed to decode swap transaction"):
            await mock_solana_wallet.execute_dex_swap("invalid_base64_data")
    
    @pytest.mark.asyncio
    async def test_execute_dex_swap_signing_failure(self, mock_solana_wallet, sample_swap_transaction):
        """Test DEX swap execution fails when transaction signing fails."""
        with patch('base64.b64decode') as mock_decode, \
             patch('src.wallet.solana_wallet.VersionedTransaction') as mock_versioned_tx:
            
            mock_decode.return_value = b"mock_transaction_bytes"
            mock_transaction = MagicMock()
            mock_transaction.sign.side_effect = Exception("Signing failed")
            mock_versioned_tx.from_bytes.return_value = mock_transaction
            
            with pytest.raises(WalletTransactionError, match="Failed to sign swap transaction"):
                await mock_solana_wallet.execute_dex_swap(sample_swap_transaction)
    
    @pytest.mark.asyncio
    async def test_execute_dex_swap_submission_failure(self, mock_solana_wallet, sample_swap_transaction):
        """Test DEX swap execution fails when transaction submission fails."""
        with patch('base64.b64decode') as mock_decode, \
             patch('src.wallet.solana_wallet.VersionedTransaction') as mock_versioned_tx:
            
            mock_decode.return_value = b"mock_transaction_bytes"
            mock_transaction = MagicMock()
            mock_versioned_tx.from_bytes.return_value = mock_transaction
            
            # Mock submission failure
            mock_solana_wallet.client.send_raw_transaction.side_effect = Exception("Submission failed")
            
            with pytest.raises(WalletTransactionError, match="Swap transaction submission failed"):
                await mock_solana_wallet.execute_dex_swap(sample_swap_transaction)
    
    @pytest.mark.asyncio
    async def test_check_token_balance_for_swap_sol_sufficient(self, mock_solana_wallet):
        """Test token balance check for SOL with sufficient balance."""
        # Mock SOL balance
        mock_solana_wallet.get_native_balance = AsyncMock(return_value=Decimal("1.0"))
        
        result = await mock_solana_wallet.check_token_balance_for_swap(
            token_address="So11111111111111111111111111111111111111112",  # SOL
            required_amount=Decimal("0.5")
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_token_balance_for_swap_sol_insufficient(self, mock_solana_wallet):
        """Test token balance check for SOL with insufficient balance."""
        # Mock low SOL balance
        mock_solana_wallet.get_native_balance = AsyncMock(return_value=Decimal("0.005"))
        
        result = await mock_solana_wallet.check_token_balance_for_swap(
            token_address="So11111111111111111111111111111111111111112",  # SOL
            required_amount=Decimal("0.5")
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_token_balance_for_swap_spl_token_sufficient(self, mock_solana_wallet):
        """Test token balance check for SPL token with sufficient balance."""
        # Mock SPL token balance
        mock_solana_wallet.get_token_balance = AsyncMock(return_value=Decimal("100.0"))
        
        result = await mock_solana_wallet.check_token_balance_for_swap(
            token_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            required_amount=Decimal("50.0")
        )
        
        assert result is True
        mock_solana_wallet.get_token_balance.assert_called_once_with("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
    
    @pytest.mark.asyncio
    async def test_check_token_balance_for_swap_spl_token_insufficient(self, mock_solana_wallet):
        """Test token balance check for SPL token with insufficient balance."""
        # Mock low SPL token balance
        mock_solana_wallet.get_token_balance = AsyncMock(return_value=Decimal("10.0"))
        
        result = await mock_solana_wallet.check_token_balance_for_swap(
            token_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            required_amount=Decimal("50.0")
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_token_balance_for_swap_error(self, mock_solana_wallet):
        """Test token balance check handles errors gracefully."""
        # Mock balance check failure
        mock_solana_wallet.get_token_balance = AsyncMock(side_effect=Exception("Balance check failed"))
        
        result = await mock_solana_wallet.check_token_balance_for_swap(
            token_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            required_amount=Decimal("50.0")
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_prepare_swap_accounts_success(self, mock_solana_wallet):
        """Test successful swap account preparation."""
        with patch('spl.token.instructions.get_associated_token_address') as mock_get_ata, \
             patch('solders.pubkey.Pubkey') as mock_pubkey:
            
            # Mock address generation
            mock_get_ata.return_value = MagicMock()
            mock_pubkey.from_string.return_value = MagicMock()
            
            # Mock account existence checks (both accounts exist)
            mock_account_info = MagicMock()
            mock_account_info.value = True  # Account exists
            mock_solana_wallet.client.get_account_info.return_value = mock_account_info
            
            result = await mock_solana_wallet.prepare_swap_accounts(
                input_token="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                output_token="Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"   # USDT
            )
            
            assert "input_account" in result
            assert "output_account" in result
    
    @pytest.mark.asyncio
    async def test_prepare_swap_accounts_create_missing(self, mock_solana_wallet):
        """Test swap account preparation creates missing accounts."""
        with patch('spl.token.instructions.get_associated_token_address') as mock_get_ata, \
             patch('solders.pubkey.Pubkey') as mock_pubkey:
            
            # Mock address generation
            mock_get_ata.return_value = MagicMock()
            mock_pubkey.from_string.return_value = MagicMock()
            
            # Mock account existence checks (accounts don't exist)
            mock_account_info = MagicMock()
            mock_account_info.value = None  # Account doesn't exist
            mock_solana_wallet.client.get_account_info.return_value = mock_account_info
            
            # Mock account creation
            mock_solana_wallet.create_associated_token_account = AsyncMock()
            
            result = await mock_solana_wallet.prepare_swap_accounts(
                input_token="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                output_token="Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
            )
            
            # Verify accounts were created
            assert mock_solana_wallet.create_associated_token_account.call_count == 2
            
            assert "input_account" in result
            assert "output_account" in result
    
    @pytest.mark.asyncio
    async def test_prepare_swap_accounts_sol_tokens(self, mock_solana_wallet):
        """Test swap account preparation with SOL tokens (no accounts needed)."""
        result = await mock_solana_wallet.prepare_swap_accounts(
            input_token="So11111111111111111111111111111111111111112",  # SOL
            output_token="So11111111111111111111111111111111111111112"   # SOL
        )
        
        assert result["input_account"] is None
        assert result["output_account"] is None
    
    @pytest.mark.asyncio
    async def test_prepare_swap_accounts_read_only_mode(self, mock_solana_wallet):
        """Test swap account preparation fails in read-only mode."""
        mock_solana_wallet.keypair = None
        
        with pytest.raises(WalletTransactionError, match="Cannot prepare accounts in read-only mode"):
            await mock_solana_wallet.prepare_swap_accounts(
                input_token="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                output_token="Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
            )
    
    @pytest.mark.asyncio
    async def test_prepare_swap_accounts_error_handling(self, mock_solana_wallet):
        """Test swap account preparation handles errors gracefully."""
        with patch('spl.token.instructions.get_associated_token_address') as mock_get_ata:
            # Mock error during account preparation
            mock_get_ata.side_effect = Exception("Account preparation failed")
            
            with pytest.raises(WalletTransactionError, match="Account preparation failed"):
                await mock_solana_wallet.prepare_swap_accounts(
                    input_token="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    output_token="Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
                )