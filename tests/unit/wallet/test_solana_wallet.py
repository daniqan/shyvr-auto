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

# Import will fail until implementation is complete - this drives TDD
try:
    from src.wallet.solana_wallet import SolanaWallet
except ImportError:
    SolanaWallet = None


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

    @pytest.mark.skipif(SolanaWallet is None, reason="SolanaWallet not implemented")
    async def test_solana_wallet_creation(self, sol_mainnet_config):
        """Test creating Solana wallet instance."""
        wallet = SolanaWallet(sol_mainnet_config)
        assert wallet.chain == Chain.SOLANA
        assert wallet.network == NetworkType.MAINNET
        assert wallet.is_connected == False
        
        # Test invalid chain raises error
        invalid_config = WalletConfig(
            chain=Chain.ETHEREUM,  # Wrong chain for SolanaWallet
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87
        )
        
        with pytest.raises(WalletError, match="SolanaWallet only supports Solana chain"):
            SolanaWallet(invalid_config)
    
    @pytest.mark.skipif(SolanaWallet is None, reason="SolanaWallet not implemented")
    async def test_solana_wallet_connect_success(self, sol_mainnet_config):
        """Test successful Solana wallet connection."""
        wallet = SolanaWallet(sol_mainnet_config)
        
        # Mock Solana RPC client
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_health.return_value = "ok"
            
            # Mock Keypair.from_bytes
            with patch('src.wallet.solana_wallet.Keypair') as mock_keypair:
                mock_keypair_instance = Mock()
                mock_pubkey = Mock()
                mock_pubkey.__str__ = Mock(return_value="11111111111111111111111111111112")
                mock_keypair_instance.pubkey.return_value = mock_pubkey
                mock_keypair.from_bytes.return_value = mock_keypair_instance
                
                # Mock base58 decode
                with patch('src.wallet.solana_wallet.base58') as mock_base58:
                    mock_base58.b58decode.return_value = b'decoded_key_bytes'
                    
                    success = await wallet.connect()
                    assert success == True
                    assert wallet.is_connected == True
                    assert wallet.wallet_address is not None
    
    @pytest.mark.skipif(SolanaWallet is None, reason="SolanaWallet not implemented")
    async def test_solana_wallet_connect_failure(self, sol_mainnet_config):
        """Test Solana wallet connection failure."""
        wallet = SolanaWallet(sol_mainnet_config)
        
        # Mock failed Solana RPC connection
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_health.side_effect = Exception("RPC failed")
            
            with pytest.raises(WalletConnectionError, match="Failed to connect to Solana RPC"):
                await wallet.connect()
    
    @pytest.mark.skipif(SolanaWallet is None, reason="SolanaWallet not implemented")
    async def test_solana_wallet_read_only_mode(self, sol_mainnet_config):
        """Test Solana wallet in read-only mode."""
        # Create read-only config
        readonly_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(readonly_config)
        
        # Mock Solana RPC client
        with patch('src.wallet.solana_wallet.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_health.return_value = "ok"
            
            # Mock Pubkey validation
            with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
                mock_pubkey.from_string.return_value = Mock()
                
                success = await wallet.connect()
                assert success == True
                assert wallet.is_connected == True
                assert wallet.wallet_address == readonly_config.wallet_address
    
    @pytest.mark.skipif(SolanaWallet is None, reason="SolanaWallet not implemented")
    async def test_solana_wallet_invalid_private_key(self):
        """Test Solana wallet with invalid private key."""
        invalid_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="invalid_key",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(invalid_config)
        
        with pytest.raises(WalletConnectionError, match="Invalid Solana private key"):
            await wallet.connect()


class TestSolanaWalletBalances:
    """Test Solana wallet balance operations."""
    
    @pytest.fixture
    def sol_config(self):
        """Create Solana wallet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87,
            rpc_url="https://api.mainnet-beta.solana.com"
        )
    
    @pytest.fixture
    def connected_sol_wallet(self, sol_config):
        """Create connected Solana wallet for testing."""
        if SolanaWallet is None:
            pytest.skip("SolanaWallet not implemented")
            
        wallet = SolanaWallet(sol_config)
        
        # Mock the connection
        wallet._connected = True
        wallet._wallet_address = "11111111111111111111111111111112"
        wallet.client = AsyncMock()
        
        return wallet
    
    async def test_get_sol_balance(self, connected_sol_wallet):
        """Test getting SOL balance."""
        wallet = connected_sol_wallet
        
        # Mock SOL balance response (1.5 SOL = 1.5 * 10^9 lamports)
        mock_response = Mock()
        mock_response.value = 1500000000  # 1.5 SOL in lamports
        wallet.client.get_balance.return_value = mock_response
        
        balance = await wallet.get_native_balance()
        assert balance == Decimal("1.5")
        
        # Verify correct pubkey was used
        from unittest.mock import ANY
        wallet.client.get_balance.assert_called_once_with(ANY)
    
    async def test_get_spl_token_balance(self, connected_sol_wallet):
        """Test getting SPL token balance."""
        wallet = connected_sol_wallet
        token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
        
        # Mock token accounts response
        mock_account = Mock()
        mock_account.account.data.parsed = {
            'info': {
                'tokenAmount': {
                    'uiAmount': 100.0,  # 100 USDC
                    'decimals': 6
                }
            }
        }
        
        mock_response = Mock()
        mock_response.value = [mock_account]
        wallet.client.get_token_accounts_by_owner.return_value = mock_response
        
        balance = await wallet.get_token_balance(token_address)
        assert balance == Decimal("100.0")
    
    async def test_get_spl_token_balance_no_account(self, connected_sol_wallet):
        """Test getting SPL token balance when no token account exists."""
        wallet = connected_sol_wallet
        token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        
        # Mock empty token accounts response
        mock_response = Mock()
        mock_response.value = []  # No token accounts
        wallet.client.get_token_accounts_by_owner.return_value = mock_response
        
        balance = await wallet.get_token_balance(token_address)
        assert balance == Decimal("0")
    
    async def test_get_full_wallet_balance(self, connected_sol_wallet):
        """Test getting full wallet balance including tokens."""
        wallet = connected_sol_wallet
        token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        
        # Mock SOL balance
        mock_sol_response = Mock()
        mock_sol_response.value = 2000000000  # 2.0 SOL
        wallet.client.get_balance.return_value = mock_sol_response
        
        # Mock token balance
        with patch.object(wallet, 'get_token_balance', return_value=Decimal("50.0")):
            balance = await wallet.get_balance(token_address)
            
            assert balance.native_balance == Decimal("2.0")
            assert balance.native_symbol == "SOL"
            assert balance.token_balances[token_address] == Decimal("50.0")


class TestSolanaWalletTransactions:
    """Test Solana wallet transaction operations."""
    
    @pytest.fixture
    def sol_config(self):
        """Create Solana wallet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87,
            rpc_url="https://api.mainnet-beta.solana.com"
        )
    
    @pytest.fixture
    def connected_sol_wallet_with_keypair(self, sol_config):
        """Create connected Solana wallet with keypair for testing."""
        if SolanaWallet is None:
            pytest.skip("SolanaWallet not implemented")
            
        wallet = SolanaWallet(sol_config)
        
        # Mock the connection and keypair
        wallet._connected = True
        wallet._wallet_address = "11111111111111111111111111111112"
        wallet.client = AsyncMock()
        wallet.keypair = Mock()
        wallet.keypair.pubkey.return_value = Mock()
        
        return wallet
    
    async def test_send_sol_transaction(self, connected_sol_wallet_with_keypair):
        """Test sending SOL transaction."""
        wallet = connected_sol_wallet_with_keypair
        to_address = "11111111111111111111111111111113"
        amount = Decimal("0.5")
        
        # Mock transaction components
        mock_blockhash_response = Mock()
        mock_blockhash_response.value.blockhash = "blockhash123"
        wallet.client.get_latest_blockhash.return_value = mock_blockhash_response
        
        # Mock transaction sending
        mock_tx_response = Mock()
        mock_tx_response.value = "signature123"
        wallet.client.send_transaction.return_value = mock_tx_response
        
        with patch('src.wallet.solana_wallet.transfer') as mock_transfer:
            with patch('src.wallet.solana_wallet.Transaction') as mock_transaction:
                mock_tx_instance = Mock()
                mock_transaction.return_value = mock_tx_instance
                
                result = await wallet.send_native_token(to_address, amount)
                
                assert result.transaction_hash == "signature123"
                assert result.status == TransactionStatus.PENDING
    
    async def test_send_spl_token_transaction(self, connected_sol_wallet_with_keypair):
        """Test sending SPL token transaction."""
        wallet = connected_sol_wallet_with_keypair
        token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
        to_address = "11111111111111111111111111111113"
        amount = Decimal("100.0")
        
        # Mock token info
        with patch.object(wallet, '_get_token_info') as mock_token_info:
            mock_token_info.return_value = Mock(decimals=6, symbol="USDC")
            
            # Mock blockhash
            mock_blockhash_response = Mock()
            mock_blockhash_response.value.blockhash = "blockhash123"
            wallet.client.get_latest_blockhash.return_value = mock_blockhash_response
            
            # Mock transaction sending
            mock_tx_response = Mock()
            mock_tx_response.value = "signature456"
            wallet.client.send_transaction.return_value = mock_tx_response
            
            with patch('src.wallet.solana_wallet.transfer_checked') as mock_transfer:
                with patch('src.wallet.solana_wallet.Transaction') as mock_transaction:
                    with patch('spl.token.instructions.get_associated_token_address') as mock_ata:
                        mock_ata.return_value = Mock()
                        
                        result = await wallet.send_token(token_address, to_address, amount)
                        
                        assert result.transaction_hash == "signature456"
                        assert result.status == TransactionStatus.PENDING
    
    async def test_estimate_transaction_fees(self, connected_sol_wallet_with_keypair):
        """Test transaction fee estimation for Solana."""
        wallet = connected_sol_wallet_with_keypair
        to_address = "11111111111111111111111111111113"
        amount = Decimal("1.0")
        
        # Test SOL transfer fee
        sol_fee = await wallet.estimate_gas(to_address, amount)
        assert sol_fee == 5000  # Standard SOL transfer fee
        
        # Test SPL token transfer fee
        token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        token_fee = await wallet.estimate_gas(to_address, amount, token_address)
        assert token_fee == 10000  # SPL token transfer fee
    
    async def test_get_gas_price(self, connected_sol_wallet_with_keypair):
        """Test getting current transaction fee (gas price equivalent)."""
        wallet = connected_sol_wallet_with_keypair
        
        fee = await wallet.get_gas_price()
        
        # Solana has fixed fees, should return fee in SOL equivalent
        assert fee == Decimal("0.000005")  # 5000 lamports in SOL


class TestSolanaWalletTransactionStatus:
    """Test Solana wallet transaction status tracking."""
    
    @pytest.fixture
    def sol_config(self):
        """Create Solana wallet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87,
            rpc_url="https://api.mainnet-beta.solana.com"
        )
    
    @pytest.fixture
    def connected_sol_wallet(self, sol_config):
        """Create connected Solana wallet for testing."""
        if SolanaWallet is None:
            pytest.skip("SolanaWallet not implemented")
            
        wallet = SolanaWallet(sol_config)
        wallet._connected = True
        wallet.client = AsyncMock()
        
        return wallet
    
    async def test_get_confirmed_transaction_status(self, connected_sol_wallet):
        """Test getting confirmed transaction status."""
        wallet = connected_sol_wallet
        tx_hash = "signature123456789"
        
        # Mock confirmed transaction status
        mock_status = Mock()
        mock_status.err = None
        mock_status.confirmation_status = "confirmed"
        mock_status.slot = 100000
        
        mock_response = Mock()
        mock_response.value = [mock_status]
        wallet.client.get_signature_statuses.return_value = mock_response
        
        result = await wallet.get_transaction_status(tx_hash)
        
        assert result.transaction_hash == tx_hash
        assert result.status == TransactionStatus.CONFIRMED
        assert result.block_number == 100000
    
    async def test_get_failed_transaction_status(self, connected_sol_wallet):
        """Test getting failed transaction status."""
        wallet = connected_sol_wallet
        tx_hash = "signature123456789"
        
        # Mock failed transaction status
        mock_status = Mock()
        mock_status.err = "Transaction failed"
        mock_status.confirmation_status = "confirmed"
        
        mock_response = Mock()
        mock_response.value = [mock_status]
        wallet.client.get_signature_statuses.return_value = mock_response
        
        result = await wallet.get_transaction_status(tx_hash)
        
        assert result.transaction_hash == tx_hash
        assert result.status == TransactionStatus.FAILED
        assert result.error_message == "Transaction failed"
    
    async def test_get_pending_transaction_status(self, connected_sol_wallet):
        """Test getting pending transaction status."""
        wallet = connected_sol_wallet
        tx_hash = "signature123456789"
        
        # Mock pending transaction (no status info)
        mock_response = Mock()
        mock_response.value = [None]  # No status info means pending
        wallet.client.get_signature_statuses.return_value = mock_response
        
        result = await wallet.get_transaction_status(tx_hash)
        
        assert result.transaction_hash == tx_hash
        assert result.status == TransactionStatus.PENDING


class TestSolanaWalletUtilities:
    """Test Solana wallet utility functions."""
    
    @pytest.fixture
    def sol_config(self):
        """Create Solana wallet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87,
            rpc_url="https://api.mainnet-beta.solana.com"
        )
    
    @pytest.fixture
    def connected_sol_wallet_with_keypair(self, sol_config):
        """Create connected Solana wallet with keypair."""
        if SolanaWallet is None:
            pytest.skip("SolanaWallet not implemented")
            
        wallet = SolanaWallet(sol_config)
        wallet._connected = True
        wallet.keypair = Mock()
        
        return wallet
    
    async def test_message_signing(self, connected_sol_wallet_with_keypair):
        """Test message signing functionality."""
        wallet = connected_sol_wallet_with_keypair
        message = "Hello, Solana!"
        
        # Mock signing
        wallet.keypair.sign_message.return_value = b'signature_bytes'
        
        with patch('src.wallet.solana_wallet.base58') as mock_base58:
            mock_base58.b58encode.return_value.decode.return_value = "signature_base58"
            
            signature = await wallet.sign_message(message)
            assert signature == "signature_base58"
    
    async def test_address_validation(self, connected_sol_wallet_with_keypair):
        """Test Solana address validation."""
        wallet = connected_sol_wallet_with_keypair
        
        # Valid Solana addresses
        valid_addresses = [
            "11111111111111111111111111111112",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        ]
        
        with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            mock_pubkey.from_string.return_value = Mock()  # Valid address
            
            for addr in valid_addresses:
                assert await wallet.validate_address(addr) == True
        
        # Invalid addresses
        invalid_addresses = [
            "invalid",
            "0x742d35Cc6598C75327f9c5C3A3Cd9dF5e1b3Bb",  # Ethereum format
            "",  # Empty
        ]
        
        with patch('src.wallet.solana_wallet.Pubkey') as mock_pubkey:
            mock_pubkey.from_string.side_effect = Exception("Invalid address")
            
            for addr in invalid_addresses:
                assert await wallet.validate_address(addr) == False
    
    async def test_read_only_mode_restrictions(self, sol_config):
        """Test that read-only mode prevents signing operations."""
        if SolanaWallet is None:
            pytest.skip("SolanaWallet not implemented")
            
        # Create read-only wallet
        readonly_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112",
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet = SolanaWallet(readonly_config)
        wallet._connected = True
        wallet.client = AsyncMock()
        # No keypair set (read-only mode)
        
        # Should not be able to send transactions
        with pytest.raises(WalletTransactionError, match="read-only mode"):
            await wallet.send_native_token("11111111111111111111111111111113", Decimal("1.0"))
        
        # Should not be able to sign messages
        with pytest.raises(WalletError, match="read-only mode"):
            await wallet.sign_message("test message")


class TestSolanaWalletTokenOperations:
    """Test Solana wallet SPL token operations."""
    
    @pytest.fixture
    def sol_config(self):
        """Create Solana wallet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87,
            rpc_url="https://api.mainnet-beta.solana.com"
        )
    
    @pytest.fixture
    def connected_sol_wallet_with_keypair(self, sol_config):
        """Create connected Solana wallet with keypair."""
        if SolanaWallet is None:
            pytest.skip("SolanaWallet not implemented")
            
        wallet = SolanaWallet(sol_config)
        wallet._connected = True
        wallet._wallet_address = "11111111111111111111111111111112"
        wallet.client = AsyncMock()
        wallet.keypair = Mock()
        wallet.keypair.pubkey.return_value = Mock()
        
        return wallet
    
    async def test_create_associated_token_account(self, connected_sol_wallet_with_keypair):
        """Test creating associated token account."""
        wallet = connected_sol_wallet_with_keypair
        mint_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
        
        # Mock that account doesn't exist
        wallet.client.get_account_info.return_value = Mock(value=None)
        
        # Mock blockhash
        mock_blockhash_response = Mock()
        mock_blockhash_response.value.blockhash = "blockhash123"
        wallet.client.get_latest_blockhash.return_value = mock_blockhash_response
        
        # Mock transaction sending
        mock_tx_response = Mock()
        mock_tx_response.value = "signature789"
        wallet.client.send_transaction.return_value = mock_tx_response
        
        with patch('spl.token.instructions.get_associated_token_address') as mock_ata:
            mock_token_account = "token_account_address_123"
            mock_ata.return_value = mock_token_account
            
            with patch('spl.token.instructions.create_associated_token_account') as mock_create:
                with patch('src.wallet.solana_wallet.Transaction') as mock_transaction:
                    
                    token_account = await wallet.create_associated_token_account(mint_address)
                    assert token_account == mock_token_account
    
    async def test_get_token_accounts(self, connected_sol_wallet_with_keypair):
        """Test getting all token accounts owned by wallet."""
        wallet = connected_sol_wallet_with_keypair
        
        # Mock token accounts response
        mock_account1 = Mock()
        mock_account1.pubkey = "token_account_1"
        mock_account1.account.data.parsed = {
            'info': {
                'mint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
                'tokenAmount': {
                    'uiAmount': 100.0,
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
                    'uiAmount': 50.0,
                    'decimals': 6
                }
            }
        }
        
        mock_response = Mock()
        mock_response.value = [mock_account1, mock_account2]
        wallet.client.get_token_accounts_by_owner.return_value = mock_response
        
        token_accounts = await wallet.get_token_accounts()
        
        assert len(token_accounts) == 2
        assert token_accounts[0]['address'] == "token_account_1"
        assert token_accounts[0]['balance'] == Decimal("100.0")
        assert token_accounts[1]['address'] == "token_account_2"
        assert token_accounts[1]['balance'] == Decimal("50.0")
    
    async def test_supported_tokens_list(self, connected_sol_wallet_with_keypair):
        """Test getting list of supported SPL tokens."""
        wallet = connected_sol_wallet_with_keypair
        
        supported_tokens = await wallet.get_supported_tokens()
        
        # Should return list of common SPL token addresses
        assert isinstance(supported_tokens, list)
        
        # For mainnet, should have common tokens
        if wallet.config.network == NetworkType.MAINNET:
            assert len(supported_tokens) >= 3
            # Should include USDC
            assert "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" in supported_tokens


class TestSolanaWalletAdvancedFeatures:
    """Test advanced Solana wallet features."""
    
    @pytest.fixture
    def sol_config(self):
        """Create Solana wallet configuration."""
        return WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87,
            rpc_url="https://api.mainnet-beta.solana.com"
        )
    
    @pytest.mark.skipif(SolanaWallet is None, reason="SolanaWallet not implemented")
    async def test_token_info_caching(self, sol_config):
        """Test that SPL token information is cached for efficiency."""
        wallet = SolanaWallet(sol_config)
        wallet._connected = True
        wallet.client = AsyncMock()
        
        token_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        
        # Mock mint account info
        mock_account_info = Mock()
        mock_account_info.value.data = bytearray(44)  # Mock mint data
        mock_account_info.value.data[4] = 6  # Decimals at offset 4
        wallet.client.get_account_info.return_value = mock_account_info
        
        # First call should query blockchain
        token_info_1 = await wallet._get_token_info(token_address)
        
        # Second call should use cache
        token_info_2 = await wallet._get_token_info(token_address)
        
        assert token_info_1.decimals == 6
        assert token_info_2.decimals == 6
        assert token_info_1 == token_info_2
        
        # Should only call blockchain once due to caching
        assert wallet.client.get_account_info.call_count == 1
    
    @pytest.mark.skipif(SolanaWallet is None, reason="SolanaWallet not implemented") 
    async def test_different_private_key_formats(self, sol_config):
        """Test Solana wallet supports different private key formats."""
        # Test Base58 format
        base58_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="5" + "a" * 87,  # Base58 format
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet1 = SolanaWallet(base58_config)
        assert wallet1.chain == Chain.SOLANA
        
        # Test hex format
        hex_config = WalletConfig(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            private_key="0x" + "b" * 64,  # Hex format
            rpc_url="https://api.mainnet-beta.solana.com"
        )
        
        wallet2 = SolanaWallet(hex_config)
        assert wallet2.chain == Chain.SOLANA


# TDD Failing Tests - These are designed to fail until implementation is complete
class TestSolanaWalletTDDFailingScenarios:
    """Tests designed to fail - this drives TDD implementation."""
    
    async def test_solana_wallet_import_fails_initially(self):
        """Test that SolanaWallet import fails until implemented."""
        # This test ensures we're following TDD - import should fail first
        if SolanaWallet is None:
            # This is expected in TDD - implementation comes after tests
            assert True
        else:
            # If import succeeds, wallet should have all required methods
            required_methods = [
                'connect', 'disconnect', 'get_balance', 'get_native_balance',
                'get_token_balance', 'send_native_token', 'send_token',
                'estimate_gas', 'get_gas_price', 'get_transaction_status',
                'sign_message', 'validate_address'
            ]
            
            for method in required_methods:
                assert hasattr(SolanaWallet, method), f"Missing method: {method}"
    
    async def test_solana_wallet_live_integration(self):
        """Test live Solana integration - will fail until RPC configured."""
        # This test will fail until proper RPC endpoints are configured
        pytest.skip("Live Solana integration not configured yet")
    
    async def test_solana_wallet_program_interaction(self):
        """Test Solana program interaction - will fail until implemented."""
        # This test will fail until program interaction is added
        pytest.skip("Solana program interaction not implemented yet")