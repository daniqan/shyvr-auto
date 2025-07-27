"""
Enhanced test cases for production-ready Ethereum wallet features.

This module tests the enhanced functionality added for production readiness
including retry logic, nonce management, and transaction confirmation.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock

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

from src.wallet.ethereum_wallet import EthereumWallet


class TestEthereumWalletRetryLogic:
    """Test retry logic and error handling."""
    
    @pytest.fixture
    def eth_config(self):
        """Create Ethereum wallet configuration."""
        return WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64,
            rpc_url="https://sepolia.infura.io/v3/test-key",
            timeout_seconds=30
        )
    
    @pytest.fixture
    def connected_eth_wallet(self, eth_config):
        """Create connected Ethereum wallet for testing."""
        wallet = EthereumWallet(eth_config)
        wallet._connected = True
        wallet._wallet_address = "0x742d35Cc6598C75327f9c5C3A3Cd9dF5e1b3Bb24"
        wallet.w3 = AsyncMock()
        return wallet
    
    async def test_retry_logic_on_network_failure(self, connected_eth_wallet):
        """Test that network operations retry on failure."""
        wallet = connected_eth_wallet
        
        # Mock network failure followed by success
        wallet.w3.eth.get_balance.side_effect = [
            ConnectionError("Network timeout"),
            ConnectionError("Network timeout"),
            1500000000000000000  # 1.5 ETH in wei on third attempt
        ]
        
        # Should succeed after retries
        balance = await wallet.get_native_balance()
        assert balance == Decimal("1.5")
        
        # Should have been called 3 times due to retries
        assert wallet.w3.eth.get_balance.call_count == 3
    
    async def test_retry_logic_gives_up_after_max_attempts(self, connected_eth_wallet):
        """Test that retry logic gives up after maximum attempts."""
        wallet = connected_eth_wallet
        
        # Mock persistent network failure
        wallet.w3.eth.get_balance.side_effect = ConnectionError("Persistent failure")
        
        # Should raise error after exhausting retries
        with pytest.raises(ConnectionError, match="Persistent failure"):
            await wallet.get_native_balance()
        
        # Should have tried max retries + 1 (initial attempt)
        assert wallet.w3.eth.get_balance.call_count == 3  # max_retries=2 + initial attempt
    
    async def test_non_network_errors_not_retried(self, connected_eth_wallet):
        """Test that non-network errors are not retried."""
        wallet = connected_eth_wallet
        
        # Mock non-network error
        wallet.w3.eth.get_balance.side_effect = ValueError("Invalid address")
        
        # Should raise error immediately without retries
        with pytest.raises(ValueError, match="Invalid address"):
            await wallet.get_native_balance()
        
        # Should have been called only once (no retries)
        assert wallet.w3.eth.get_balance.call_count == 1


class TestEthereumWalletNonceManagement:
    """Test nonce management for concurrent transactions."""
    
    @pytest.fixture
    def eth_config(self):
        """Create Ethereum wallet configuration."""
        return WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64,
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
    
    @pytest.fixture
    def connected_eth_wallet_with_account(self, eth_config):
        """Create connected Ethereum wallet with account."""
        wallet = EthereumWallet(eth_config)
        wallet._connected = True
        wallet._wallet_address = "0x742d35Cc6598C75327f9c5C3A3Cd9dF5e1b3Bb24"
        wallet.w3 = AsyncMock()
        wallet.account = Mock()
        wallet.account.address = wallet._wallet_address
        return wallet
    
    async def test_nonce_increments_for_concurrent_transactions(self, connected_eth_wallet_with_account):
        """Test that nonces increment properly for concurrent transactions."""
        wallet = connected_eth_wallet_with_account
        
        # Mock network nonce
        wallet.w3.eth.get_transaction_count.return_value = 5
        
        # Get multiple nonces
        nonce1 = await wallet._get_next_nonce()
        nonce2 = await wallet._get_next_nonce()
        nonce3 = await wallet._get_next_nonce()
        
        # Should increment properly
        assert nonce1 == 5
        assert nonce2 == 6
        assert nonce3 == 7
    
    async def test_nonce_resets_to_network_nonce_when_higher(self, connected_eth_wallet_with_account):
        """Test that nonce resets to network nonce when it's higher."""
        wallet = connected_eth_wallet_with_account
        
        # Start with network nonce 5
        wallet.w3.eth.get_transaction_count.return_value = 5
        nonce1 = await wallet._get_next_nonce()
        assert nonce1 == 5
        
        # Network nonce jumps to 10 (maybe other transactions confirmed)
        wallet.w3.eth.get_transaction_count.return_value = 10
        nonce2 = await wallet._get_next_nonce()
        assert nonce2 == 10  # Should use higher network nonce
        
        # Next nonce should continue from there
        nonce3 = await wallet._get_next_nonce()
        assert nonce3 == 11


class TestEthereumWalletTransactionConfirmation:
    """Test transaction confirmation waiting."""
    
    @pytest.fixture
    def eth_config(self):
        """Create Ethereum wallet configuration."""
        return WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64,
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
    
    @pytest.fixture
    def connected_eth_wallet(self, eth_config):
        """Create connected Ethereum wallet."""
        wallet = EthereumWallet(eth_config)
        wallet._connected = True
        wallet.w3 = AsyncMock()
        return wallet
    
    async def test_wait_for_transaction_confirmation_success(self, connected_eth_wallet):
        """Test successful transaction confirmation waiting."""
        wallet = connected_eth_wallet
        tx_hash = "0x123456789abcdef"
        
        # Mock transaction status progression: pending -> confirmed
        pending_result = TransactionResult(
            transaction_hash=tx_hash,
            status=TransactionStatus.PENDING
        )
        confirmed_result = TransactionResult(
            transaction_hash=tx_hash,
            status=TransactionStatus.CONFIRMED,
            gas_used=21000,
            block_number=1000000
        )
        
        with patch.object(wallet, 'get_transaction_status', side_effect=[pending_result, confirmed_result]):
            result = await wallet.wait_for_transaction_confirmation(tx_hash, timeout_seconds=10, poll_interval=0.1)
            
            assert result.status == TransactionStatus.CONFIRMED
            assert result.block_number == 1000000
    
    async def test_wait_for_transaction_confirmation_failure(self, connected_eth_wallet):
        """Test transaction confirmation waiting when transaction fails."""
        wallet = connected_eth_wallet
        tx_hash = "0x123456789abcdef"
        
        # Mock failed transaction
        failed_result = TransactionResult(
            transaction_hash=tx_hash,
            status=TransactionStatus.FAILED,
            error_message="Transaction reverted"
        )
        
        with patch.object(wallet, 'get_transaction_status', return_value=failed_result):
            with pytest.raises(WalletTransactionError, match="Transaction failed"):
                await wallet.wait_for_transaction_confirmation(tx_hash, timeout_seconds=1, poll_interval=0.1)
    
    async def test_wait_for_transaction_confirmation_timeout(self, connected_eth_wallet):
        """Test transaction confirmation waiting timeout."""
        wallet = connected_eth_wallet
        tx_hash = "0x123456789abcdef"
        
        # Mock persistent pending status
        pending_result = TransactionResult(
            transaction_hash=tx_hash,
            status=TransactionStatus.PENDING
        )
        
        with patch.object(wallet, 'get_transaction_status', return_value=pending_result):
            with pytest.raises(WalletTransactionError, match="did not confirm within"):
                await wallet.wait_for_transaction_confirmation(tx_hash, timeout_seconds=0.2, poll_interval=0.1)


class TestEthereumWalletProductionFeatures:
    """Test production-ready features."""
    
    @pytest.fixture
    def eth_config(self):
        """Create Ethereum wallet configuration."""
        return WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64,
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
    
    async def test_wallet_supports_both_ethereum_and_base(self, eth_config):
        """Test that wallet supports both Ethereum and Base chains."""
        # Test Ethereum
        eth_wallet = EthereumWallet(eth_config)
        assert eth_wallet.chain == Chain.ETHEREUM
        
        # Test Base
        base_config = WalletConfig(
            chain=Chain.BASE,
            network=NetworkType.TESTNET,
            private_key="0x" + "b" * 64,
            rpc_url="https://base-sepolia.infura.io/v3/test-key"
        )
        base_wallet = EthereumWallet(base_config)
        assert base_wallet.chain == Chain.BASE
    
    async def test_wallet_rejects_unsupported_chains(self):
        """Test that wallet rejects unsupported chains."""
        with pytest.raises(WalletError, match="EthereumWallet does not support chain"):
            WalletConfig(
                chain=Chain.SOLANA,  # Unsupported
                network=NetworkType.TESTNET,
                private_key="0x" + "a" * 64
            )
            # This should fail in WalletConfig validation, but let's test EthereumWallet too
            config = WalletConfig(
                chain=Chain.SOLANA,
                network=NetworkType.TESTNET,
                wallet_address="test_address",  # Use address to bypass private key validation
                rpc_url="https://test.com"
            )
            EthereumWallet(config)
    
    async def test_connection_validation_chain_id_mismatch(self):
        """Test that connection validates chain ID matches expected network."""
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,  # Expects Sepolia (chain ID 11155111)
            private_key="0x" + "a" * 64,
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
        
        wallet = EthereumWallet(config)
        
        with patch('src.wallet.ethereum_wallet.AsyncWeb3') as mock_web3:
            mock_web3_instance = AsyncMock()
            mock_web3.return_value = mock_web3_instance
            mock_web3_instance.is_connected.return_value = True
            mock_web3_instance.eth.chain_id = 1  # Mainnet chain ID instead of Sepolia
            
            with patch('src.wallet.ethereum_wallet.Account') as mock_account:
                mock_account_instance = Mock()
                mock_account_instance.address = "0x742d35Cc6598C75327f9c5C3A3Cd9dF5e1b3Bb24"
                mock_account.from_key.return_value = mock_account_instance
                
                with pytest.raises(WalletConnectionError, match="Network mismatch"):
                    await wallet.connect()
    
    async def test_gas_estimation_includes_safety_buffer(self):
        """Test that gas estimation includes a safety buffer."""
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64,
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
        
        wallet = EthereumWallet(config)
        wallet._connected = True
        wallet._wallet_address = "0x742d35Cc6598C75327f9c5C3A3Cd9dF5e1b3Bb24"
        wallet.w3 = AsyncMock()
        
        # Mock gas estimation returning 21000
        wallet.w3.eth.estimate_gas.return_value = 21000
        
        estimated_gas = await wallet.estimate_gas(
            to_address="0x742d35Cc6598C75327f9c5C3A3Cd9dF5e1b3AA",
            amount=Decimal("1.0")
        )
        
        # Should include 10% buffer: 21000 * 1.1 = 23100
        assert estimated_gas == 23100
    
    async def test_token_cache_optimization(self):
        """Test that token information is cached for efficiency."""
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64,
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
        
        wallet = EthereumWallet(config)
        wallet._connected = True
        wallet.w3 = AsyncMock()
        
        token_address = "0xA0b86a33E6441CcE67d6ea49e8F7a0D73c37A8c4"
        
        # Mock contract calls
        mock_contract = AsyncMock()
        mock_contract.functions.symbol.return_value.call.return_value = "USDC"
        mock_contract.functions.decimals.return_value.call.return_value = 6
        mock_contract.functions.name.return_value.call.return_value = "USD Coin"
        
        wallet.w3.eth.contract.return_value = mock_contract
        
        # First call should populate cache
        token_info_1 = await wallet._get_token_info(token_address)
        
        # Second call should use cache (contract methods shouldn't be called again)
        token_info_2 = await wallet._get_token_info(token_address)
        
        assert token_info_1.symbol == "USDC"
        assert token_info_2.symbol == "USDC"
        assert token_info_1 == token_info_2
        
        # Verify contract was only created once (due to caching)
        wallet.w3.eth.contract.assert_called_once()