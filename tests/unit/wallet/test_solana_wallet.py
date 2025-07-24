"""
Test cases for Solana wallet implementation.

Following TDD methodology - these tests will initially fail and drive implementation.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Optional

from src.wallet.base import (
    WalletConfig,
    WalletBalance,
    TransactionResult,
    TransactionStatus,
    Chain,
    NetworkType,
    WalletError,
    WalletConnectionError,
    WalletTransactionError
)


class TestSolanaWalletConnection:
    """Test Solana wallet connection functionality."""
    
    @pytest.fixture
    def sol_mainnet_config(self):
        """Create Solana mainnet wallet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87,  # Base58 encoded private key format
            rpc_url="https://api.mainnet-beta.solana.com",
            timeout_seconds=30
        )
    
    @pytest.fixture
    def sol_devnet_config(self):
        """Create Solana devnet wallet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.DEVNET,
            private_key="0x" + "b" * 64,  # Hex format private key
            rpc_url="https://api.devnet.solana.com"
        )
    
    async def test_solana_wallet_creation(self, sol_mainnet_config):
        """Test creating Solana wallet instance."""
        from src.wallet.solana_wallet import SolanaWallet
        
        wallet = SolanaWallet(sol_mainnet_config)
        assert wallet.config == sol_mainnet_config
        assert wallet.chain == Chain.SOLANA
        assert wallet.network == NetworkType.MAINNET
        assert not wallet.is_connected
        assert wallet.LAMPORTS_PER_SOL == 1_000_000_000
    
    async def test_unsupported_chain_rejection(self):
        """Test that non-Solana chains are rejected."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.ETHEREUM,  # Not supported by SolanaWallet
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64
        )
        
        with pytest.raises(WalletError, match="SolanaWallet only supports Solana chain"):
            SolanaWallet(config)
    
    @patch('src.wallet.solana_wallet.AsyncClient')
    @patch('src.wallet.solana_wallet.Keypair')
    async def test_solana_wallet_connect_success(self, mock_keypair, mock_client, sol_mainnet_config):
        """Test successful Solana wallet connection."""
        from src.wallet.solana_wallet import SolanaWallet
        
        # Mock Solana client
        mock_client_instance = AsyncMock()
        mock_client_instance.get_health.return_value = {"status": "ok"}
        mock_client_instance.get_slot.return_value = 150000000
        mock_client.return_value = mock_client_instance
        
        # Mock keypair
        mock_keypair_instance = Mock()
        mock_keypair_instance.pubkey.return_value = "11111111111111111111111111111112"
        mock_keypair.from_bytes.return_value = mock_keypair_instance
        
        wallet = SolanaWallet(sol_mainnet_config)
        result = await wallet.connect()
        
        assert result == True
        assert wallet.is_connected == True
        assert wallet.wallet_address == "11111111111111111111111111111112"
    
    @patch('src.wallet.solana_wallet.AsyncClient')
    async def test_solana_wallet_connect_failure(self, mock_client, sol_mainnet_config):
        """Test Solana wallet connection failure."""
        from src.wallet.solana_wallet import SolanaWallet
        
        # Mock client connection failure
        mock_client_instance = AsyncMock()
        mock_client_instance.get_health.side_effect = Exception("Connection failed")
        mock_client.return_value = mock_client_instance
        
        wallet = SolanaWallet(sol_mainnet_config)
        
        with pytest.raises(WalletConnectionError, match="Failed to connect to Solana RPC"):
            await wallet.connect()
    
    async def test_solana_wallet_read_only_mode(self):
        """Test Solana wallet in read-only mode."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            mock_client.return_value = mock_client_instance
            
            mock_pubkey.from_string.return_value = "validated_pubkey"
            
            wallet = SolanaWallet(config)
            await wallet.connect()
            
            assert wallet.is_connected == True
            assert wallet.wallet_address == "11111111111111111111111111111112"
    
    @patch('src.wallet.solana_wallet.base58')
    async def test_base58_private_key_handling(self, mock_base58, sol_mainnet_config):
        """Test handling of base58 encoded private keys."""
        from src.wallet.solana_wallet import SolanaWallet
        
        mock_base58.b58decode.return_value = b"decoded_key_bytes"
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Keypair') as mock_keypair:
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            mock_client.return_value = mock_client_instance
            
            mock_keypair_instance = Mock()
            mock_keypair_instance.pubkey.return_value = "test_pubkey"
            mock_keypair.from_bytes.return_value = mock_keypair_instance
            
            wallet = SolanaWallet(sol_mainnet_config)
            await wallet.connect()
            
            # Should decode base58 key since length is 88
            mock_base58.b58decode.assert_called_once()


class TestSolanaWalletBalances:
    """Test Solana wallet balance operations."""
    
    @pytest.fixture
    def connected_sol_wallet(self, sol_mainnet_config):
        """Create connected Solana wallet for testing."""
        from src.wallet.solana_wallet import SolanaWallet
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Keypair') as mock_keypair:
            
            # Setup mocks
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            mock_client_instance.get_slot.return_value = 150000000
            mock_client.return_value = mock_client_instance
            
            mock_keypair_instance = Mock()
            mock_keypair_instance.pubkey.return_value = "11111111111111111111111111111112"
            mock_keypair.from_bytes.return_value = mock_keypair_instance
            
            wallet = SolanaWallet(sol_mainnet_config)
            return wallet, mock_client_instance
    
    async def test_get_sol_balance(self, connected_sol_wallet):
        """Test getting SOL balance."""
        wallet, mock_client = connected_sol_wallet
        
        # Mock SOL balance (2.5 SOL in lamports)
        mock_response = Mock()
        mock_response.value = 2500000000  # 2.5 SOL
        mock_client.get_balance.return_value = mock_response
        
        with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            mock_pubkey.from_string.return_value = "mocked_pubkey"
            
            await wallet.connect()
            balance = await wallet.get_native_balance()
            
            assert balance == Decimal("2.5")
    
    async def test_get_spl_token_balance(self, connected_sol_wallet):
        """Test getting SPL token balance."""
        wallet, mock_client = connected_sol_wallet
        
        # Mock SPL token account response
        mock_token_account = Mock()
        mock_token_account.account.data.parsed = {
            'info': {
                'tokenAmount': {
                    'uiAmount': 1000.0,
                    'decimals': 6
                }
            }
        }
        mock_response = Mock()
        mock_response.value = [mock_token_account]
        mock_client.get_token_accounts_by_owner.return_value = mock_response
        
        with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            mock_pubkey.from_string.return_value = "mocked_pubkey"
            
            await wallet.connect()
            token_balance = await wallet.get_token_balance("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
            
            assert token_balance == Decimal("1000.0")
    
    async def test_get_spl_token_balance_no_account(self, connected_sol_wallet):
        """Test getting SPL token balance when no token account exists."""
        wallet, mock_client = connected_sol_wallet
        
        # Mock empty token account response
        mock_response = Mock()
        mock_response.value = []
        mock_client.get_token_accounts_by_owner.return_value = mock_response
        
        with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            mock_pubkey.from_string.return_value = "mocked_pubkey"
            
            await wallet.connect()
            token_balance = await wallet.get_token_balance("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
            
            assert token_balance == Decimal("0")
    
    async def test_get_full_wallet_balance(self, connected_sol_wallet):
        """Test getting full wallet balance."""
        wallet, mock_client = connected_sol_wallet
        
        # Mock SOL balance
        mock_balance_response = Mock()
        mock_balance_response.value = 1500000000  # 1.5 SOL
        mock_client.get_balance.return_value = mock_balance_response
        
        with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            mock_pubkey.from_string.return_value = "mocked_pubkey"
            
            await wallet.connect()
            balance = await wallet.get_balance()
            
            assert isinstance(balance, WalletBalance)
            assert balance.native_balance == Decimal("1.5")
            assert balance.native_symbol == "SOL"


class TestSolanaWalletTransactions:
    """Test Solana wallet transaction operations."""
    
    @pytest.fixture
    def connected_sol_wallet_with_keypair(self, sol_mainnet_config):
        """Create connected Solana wallet with keypair for testing."""
        from src.wallet.solana_wallet import SolanaWallet
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Keypair') as mock_keypair:
            
            # Setup client mock
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            mock_client_instance.get_latest_blockhash.return_value = Mock(value=Mock(blockhash="mock_blockhash"))
            mock_client_instance.send_transaction.return_value = Mock(value="mock_signature")
            mock_client.return_value = mock_client_instance
            
            # Setup keypair mock
            mock_keypair_instance = Mock()
            mock_keypair_instance.pubkey.return_value = "sender_pubkey"
            mock_keypair.from_bytes.return_value = mock_keypair_instance
            
            wallet = SolanaWallet(sol_mainnet_config)
            return wallet, mock_client_instance, mock_keypair_instance
    
    async def test_send_sol_transaction(self, connected_sol_wallet_with_keypair):
        """Test sending SOL transaction.""" 
        wallet, mock_client, mock_keypair = connected_sol_wallet_with_keypair
        
        with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey, \
             patch('src.wallet.solana_wallet.transfer') as mock_transfer, \
             patch('src.wallet.solana_wallet.Transaction') as mock_transaction:
            
            mock_pubkey.from_string.return_value = "recipient_pubkey"
            mock_transfer.return_value = "transfer_instruction"
            
            mock_tx_instance = Mock()
            mock_transaction.return_value = mock_tx_instance
            
            await wallet.connect()
            result = await wallet.send_native_token(
                to_address="11111111111111111111111111111112",
                amount=Decimal("1.5")
            )
            
            assert isinstance(result, TransactionResult)
            assert result.transaction_hash == "mock_signature"
            assert result.status == TransactionStatus.PENDING
    
    async def test_send_spl_token_transaction(self, connected_sol_wallet_with_keypair):
        """Test sending SPL token transaction."""
        wallet, mock_client, mock_keypair = connected_sol_wallet_with_keypair
        
        # Mock token info
        mock_account_info = Mock()
        mock_account_info.value = Mock()
        mock_account_info.value.data = bytearray(44)  # Minimum mint data size
        mock_account_info.value.data[4] = 6  # decimals at offset 4
        mock_client.get_account_info.return_value = mock_account_info
        
        with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey, \
             patch('src.wallet.solana_wallet.transfer_checked') as mock_transfer_checked, \
             patch('src.wallet.solana_wallet.get_associated_token_address') as mock_get_ata, \
             patch('src.wallet.solana_wallet.Transaction') as mock_transaction:
            
            mock_pubkey.from_string.return_value = "pubkey_mock"
            mock_get_ata.return_value = "associated_token_account"
            mock_transfer_checked.return_value = "transfer_instruction"
            
            mock_tx_instance = Mock()
            mock_transaction.return_value = mock_tx_instance
            
            await wallet.connect()
            result = await wallet.send_token(
                token_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                to_address="11111111111111111111111111111112",
                amount=Decimal("100.0")
            )
            
            assert isinstance(result, TransactionResult)
            assert result.status == TransactionStatus.PENDING
    
    async def test_estimate_transaction_fees(self, connected_sol_wallet_with_keypair):
        """Test transaction fee estimation."""
        wallet, mock_client, mock_keypair = connected_sol_wallet_with_keypair
        
        await wallet.connect()
        
        # Test SOL transfer fee
        sol_fee = await wallet.estimate_gas(
            to_address="11111111111111111111111111111112",
            amount=Decimal("1.0")
        )
        assert sol_fee == 5000  # Fixed SOL transfer fee
        
        # Test SPL token transfer fee
        token_fee = await wallet.estimate_gas(
            to_address="11111111111111111111111111111112",
            amount=Decimal("100.0"),
            token_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        )
        assert token_fee == 10000  # Fixed SPL token transfer fee
    
    async def test_get_gas_price(self, connected_sol_wallet_with_keypair):
        """Test getting transaction fee in SOL."""
        wallet, mock_client, mock_keypair = connected_sol_wallet_with_keypair
        
        await wallet.connect()
        gas_price = await wallet.get_gas_price()
        
        # Should return 5000 lamports converted to SOL
        assert gas_price == Decimal("0.000005")


class TestSolanaWalletTransactionStatus:
    """Test Solana wallet transaction status checking."""
    
    async def test_get_confirmed_transaction_status(self):
        """Test getting status of confirmed transaction."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Signature') as mock_signature:
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            
            # Mock confirmed transaction status
            mock_status_info = Mock()
            mock_status_info.err = None
            mock_status_info.confirmation_status = "confirmed"
            mock_status_info.slot = 150000000
            
            mock_response = Mock()
            mock_response.value = [mock_status_info]
            mock_client_instance.get_signature_statuses.return_value = mock_response
            
            mock_client.return_value = mock_client_instance
            mock_signature.from_string.return_value = "mock_signature"
            
            wallet = SolanaWallet(config)
            await wallet.connect()
            
            result = await wallet.get_transaction_status("mock_tx_hash")
            
            assert result.status == TransactionStatus.CONFIRMED
            assert result.block_number == 150000000
    
    async def test_get_failed_transaction_status(self):
        """Test getting status of failed transaction."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Signature') as mock_signature:
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            
            # Mock failed transaction status
            mock_status_info = Mock()
            mock_status_info.err = "Transaction failed"
            
            mock_response = Mock()
            mock_response.value = [mock_status_info]
            mock_client_instance.get_signature_statuses.return_value = mock_response
            
            mock_client.return_value = mock_client_instance
            mock_signature.from_string.return_value = "mock_signature"
            
            wallet = SolanaWallet(config)
            await wallet.connect()
            
            result = await wallet.get_transaction_status("mock_tx_hash")
            
            assert result.status == TransactionStatus.FAILED
            assert result.error_message == "Transaction failed"
    
    async def test_get_pending_transaction_status(self):
        """Test getting status of pending transaction."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Signature') as mock_signature:
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            
            # Mock no status found (pending)
            mock_response = Mock()
            mock_response.value = [None]
            mock_client_instance.get_signature_statuses.return_value = mock_response
            
            mock_client.return_value = mock_client_instance
            mock_signature.from_string.return_value = "mock_signature"
            
            wallet = SolanaWallet(config)
            await wallet.connect()
            
            result = await wallet.get_transaction_status("mock_tx_hash")
            
            assert result.status == TransactionStatus.PENDING


class TestSolanaWalletUtilities:
    """Test Solana wallet utility functions."""
    
    async def test_address_validation(self):
        """Test Solana address validation."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(config)
        
        with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            # Valid address
            mock_pubkey.from_string.side_effect = [None]  # No exception
            assert await wallet.validate_address("11111111111111111111111111111112") == True
            
            # Invalid address
            mock_pubkey.from_string.side_effect = Exception("Invalid pubkey")
            assert await wallet.validate_address("invalid") == False
    
    async def test_message_signing(self):
        """Test message signing functionality."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87,
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Keypair') as mock_keypair, \
             patch('src.wallet.solana_wallet.base58') as mock_base58:
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            mock_client.return_value = mock_client_instance
            
            mock_keypair_instance = Mock()
            mock_keypair_instance.pubkey.return_value = "test_pubkey"
            mock_keypair_instance.sign_message.return_value = b"signature_bytes"
            mock_keypair.from_bytes.return_value = mock_keypair_instance
            
            mock_base58.b58encode.return_value = b"base58_signature"
            
            wallet = SolanaWallet(config)
            await wallet.connect()
            
            signature = await wallet.sign_message("Hello, World!")
            assert signature == "base58_signature"
    
    async def test_read_only_mode_restrictions(self):
        """Test that read-only mode prevents signing operations."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            mock_client.return_value = mock_client_instance
            
            wallet = SolanaWallet(config)
            await wallet.connect()
            
            # Should raise error for transaction operations
            with pytest.raises(WalletTransactionError, match="Cannot send transactions in read-only mode"):
                await wallet.send_native_token("11111111111111111111111111111112", Decimal("1.0"))
            
            # Should raise error for message signing
            with pytest.raises(WalletError, match="Cannot sign messages in read-only mode"):
                await wallet.sign_message("test")
    
    async def test_get_supported_tokens(self):
        """Test getting supported SPL tokens."""
        from src.wallet.solana_wallet import SolanaWallet
        
        # Mainnet config
        mainnet_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(mainnet_config)
        tokens = await wallet.get_supported_tokens()
        
        assert "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" in tokens  # USDC
        assert "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB" in tokens  # USDT
        
        # Devnet config
        devnet_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.DEVNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.devnet.solana.com"
        )
        
        devnet_wallet = SolanaWallet(devnet_config)
        devnet_tokens = await devnet_wallet.get_supported_tokens()
        
        assert "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU" in devnet_tokens  # USDC Devnet


class TestSolanaWalletTokenOperations:
    """Test Solana wallet SPL token operations."""
    
    async def test_create_associated_token_account(self):
        """Test creating associated token account."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.DEVNET,
            private_key="5" + "a" * 87,
            rpc_url="https://api.devnet.solana.com"
        )
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Keypair') as mock_keypair, \
             patch('src.wallet.solana_wallet.get_associated_token_address') as mock_get_ata, \
             patch('src.wallet.solana_wallet.create_associated_token_account') as mock_create_ata, \
             patch('src.wallet.solana_wallet.Transaction') as mock_transaction, \
             patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            
            # Setup mocks
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            mock_client_instance.get_account_info.return_value = Mock(value=None)  # Account doesn't exist
            mock_client_instance.get_latest_blockhash.return_value = Mock(value=Mock(blockhash="mock_blockhash"))
            mock_client_instance.send_transaction.return_value = Mock(value="create_ata_signature")
            mock_client.return_value = mock_client_instance
            
            mock_keypair_instance = Mock()
            mock_keypair_instance.pubkey.return_value = "owner_pubkey"
            mock_keypair.from_bytes.return_value = mock_keypair_instance
            
            mock_get_ata.return_value = "associated_token_address"
            mock_create_ata.return_value = "create_instruction"
            mock_pubkey.from_string.return_value = "pubkey_mock"
            
            mock_tx_instance = Mock()
            mock_transaction.return_value = mock_tx_instance
            
            wallet = SolanaWallet(config)
            await wallet.connect()
            
            result = await wallet.create_associated_token_account(
                mint_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            )
            
            assert result == "associated_token_address"
    
    async def test_get_token_accounts(self):
        """Test getting all SPL token accounts."""
        from src.wallet.solana_wallet import SolanaWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client, \
             patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get_health.return_value = {"status": "ok"}
            
            # Mock token accounts response
            mock_account1 = Mock()
            mock_account1.pubkey = "token_account_1"
            mock_account1.account.data.parsed = {
                'info': {
                    'mint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
                    'tokenAmount': {
                        'uiAmount': 500.0,
                        'decimals': 6
                    }
                }
            }
            
            mock_account2 = Mock()
            mock_account2.pubkey = "token_account_2"
            mock_account2.account.data.parsed = {
                'info': {
                    'mint': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
                    'tokenAmount': {
                        'uiAmount': 100.0,
                        'decimals': 6
                    }
                }
            }
            
            mock_response = Mock()
            mock_response.value = [mock_account1, mock_account2]
            mock_client_instance.get_token_accounts_by_owner.return_value = mock_response
            
            mock_client.return_value = mock_client_instance
            mock_pubkey.from_string.return_value = "owner_pubkey"
            
            wallet = SolanaWallet(config)
            await wallet.connect()
            
            accounts = await wallet.get_token_accounts()
            
            assert len(accounts) == 2
            assert accounts[0]['address'] == "token_account_1"
            assert accounts[0]['balance'] == Decimal("500.0")
            assert accounts[1]['address'] == "token_account_2"
            assert accounts[1]['balance'] == Decimal("100.0")


# These tests are expected to fail initially - this is TDD
class TestSolanaWalletIntegration:
    """Integration tests that will initially fail but drive implementation."""
    
    async def test_solana_mainnet_integration(self):
        """Test Solana mainnet integration."""
        # This test will drive implementation of mainnet-specific features
        pytest.skip("Mainnet integration test - implement after basic functionality")
    
    async def test_spl_token_metadata_integration(self):
        """Test SPL token metadata integration."""
        # This will drive implementation of token metadata fetching
        pytest.skip("Token metadata integration - requires metadata program integration")
    
    async def test_transaction_history_integration(self):
        """Test transaction history integration."""
        # This will drive implementation of transaction history fetching
        pytest.skip("Transaction history - requires enhanced RPC integration")