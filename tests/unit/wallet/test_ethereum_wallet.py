"""
Test cases for Ethereum wallet implementation.

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


class TestEthereumWalletConnection:
    """Test Ethereum wallet connection functionality."""
    
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
    def base_config(self):
        """Create Base chain wallet configuration."""
        return WalletConfig(
            chain=Chain.BASE,
            network=NetworkType.TESTNET,
            private_key="0x" + "b" * 64,
            rpc_url="https://base-sepolia.infura.io/v3/test-key"
        )
    
    async def test_ethereum_wallet_creation(self, eth_config):
        """Test creating Ethereum wallet instance."""
        # This will fail until EthereumWallet is properly imported
        from src.wallet.ethereum_wallet import EthereumWallet
        
        wallet = EthereumWallet(eth_config)
        assert wallet.config == eth_config
        assert wallet.chain == Chain.ETHEREUM
        assert wallet.network == NetworkType.TESTNET
        assert not wallet.is_connected
    
    async def test_base_wallet_creation(self, base_config):
        """Test creating Base chain wallet instance."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        wallet = EthereumWallet(base_config)
        assert wallet.config == base_config
        assert wallet.chain == Chain.BASE
    
    async def test_unsupported_chain_rejection(self):
        """Test that unsupported chains are rejected."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        config = WalletConfig(
            chain=Chain.SOLANA,  # Not supported by EthereumWallet
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64
        )
        
        with pytest.raises(WalletError, match="EthereumWallet does not support chain"):
            EthereumWallet(config)
    
    @patch('src.wallet.ethereum_wallet.AsyncWeb3')
    @patch('src.wallet.ethereum_wallet.Account')
    async def test_ethereum_wallet_connect_success(self, mock_account, mock_web3, eth_config):
        """Test successful Ethereum wallet connection."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        # Mock Web3 setup
        mock_w3_instance = AsyncMock()
        mock_w3_instance.is_connected.return_value = True
        mock_w3_instance.eth.chain_id = 11155111  # Sepolia
        mock_w3_instance.eth.block_number = 12345
        mock_web3.return_value = mock_w3_instance
        
        # Mock account setup
        mock_account_instance = Mock()
        mock_account_instance.address = "0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A"
        mock_account.from_key.return_value = mock_account_instance
        
        wallet = EthereumWallet(eth_config)
        result = await wallet.connect()
        
        assert result == True
        assert wallet.is_connected == True
        assert wallet.wallet_address == "0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A"
    
    @patch('src.wallet.ethereum_wallet.AsyncWeb3')
    async def test_ethereum_wallet_connect_failure(self, mock_web3, eth_config):
        """Test Ethereum wallet connection failure."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        # Mock Web3 connection failure
        mock_w3_instance = AsyncMock()
        mock_w3_instance.is_connected.return_value = False
        mock_web3.return_value = mock_w3_instance
        
        wallet = EthereumWallet(eth_config)
        
        with pytest.raises(WalletConnectionError, match="Failed to connect to Ethereum network"):
            await wallet.connect()
    
    async def test_ethereum_wallet_read_only_mode(self):
        """Test Ethereum wallet in read-only mode."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.MAINNET,
            wallet_address="0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A",
            rpc_url="https://mainnet.infura.io/v3/test-key"
        )
        
        with patch('src.wallet.ethereum_wallet.AsyncWeb3') as mock_web3:
            mock_w3_instance = AsyncMock()
            mock_w3_instance.is_connected.return_value = True
            mock_web3.return_value = mock_w3_instance
            
            wallet = EthereumWallet(config)
            await wallet.connect()
            
            assert wallet.is_connected == True
            assert wallet.wallet_address == "0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A"


class TestEthereumWalletBalances:
    """Test Ethereum wallet balance operations."""
    
    @pytest.fixture
    def connected_eth_wallet(self, eth_config):
        """Create connected Ethereum wallet for testing."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        with patch('src.wallet.ethereum_wallet.AsyncWeb3') as mock_web3, \
             patch('src.wallet.ethereum_wallet.Account') as mock_account:
            
            # Setup mocks
            mock_w3_instance = AsyncMock()
            mock_w3_instance.is_connected.return_value = True
            mock_w3_instance.eth.chain_id = 11155111
            mock_w3_instance.eth.block_number = 12345
            mock_web3.return_value = mock_w3_instance
            
            mock_account_instance = Mock()
            mock_account_instance.address = "0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A"
            mock_account.from_key.return_value = mock_account_instance
            
            wallet = EthereumWallet(eth_config)
            return wallet, mock_w3_instance
    
    async def test_get_eth_balance(self, connected_eth_wallet):
        """Test getting ETH balance."""
        wallet, mock_w3 = connected_eth_wallet
        
        # Mock ETH balance (1.5 ETH in wei)
        mock_w3.eth.get_balance.return_value = 1500000000000000000
        
        await wallet.connect()
        balance = await wallet.get_native_balance()
        
        assert balance == Decimal("1.5")
    
    async def test_get_erc20_token_balance(self, connected_eth_wallet):
        """Test getting ERC-20 token balance."""
        wallet, mock_w3 = connected_eth_wallet
        
        # Mock token contract
        mock_contract = AsyncMock()
        mock_contract.functions.balanceOf.return_value.call.return_value = 100000000  # 100 tokens with 6 decimals
        mock_contract.functions.decimals.return_value.call.return_value = 6
        mock_contract.functions.symbol.return_value.call.return_value = "USDC"
        mock_contract.functions.name.return_value.call.return_value = "USD Coin"
        
        mock_w3.eth.contract.return_value = mock_contract
        
        await wallet.connect()
        token_balance = await wallet.get_token_balance("0xA0b86a33E6441CcE67d6ea49e8F7a0D73c37A8c4")
        
        assert token_balance == Decimal("100.0")
    
    async def test_get_full_wallet_balance(self, connected_eth_wallet):
        """Test getting full wallet balance with native and token balances."""
        wallet, mock_w3 = connected_eth_wallet
        
        # Mock ETH balance
        mock_w3.eth.get_balance.return_value = 2000000000000000000  # 2 ETH
        
        await wallet.connect()
        balance = await wallet.get_balance()
        
        assert isinstance(balance, WalletBalance)
        assert balance.native_balance == Decimal("2.0")
        assert balance.native_symbol == "ETH"


class TestEthereumWalletTransactions:
    """Test Ethereum wallet transaction operations."""
    
    @pytest.fixture
    def connected_eth_wallet_with_account(self, eth_config):
        """Create connected Ethereum wallet with account for testing."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        with patch('src.wallet.ethereum_wallet.AsyncWeb3') as mock_web3, \
             patch('src.wallet.ethereum_wallet.Account') as mock_account:
            
            # Setup Web3 mock
            mock_w3_instance = AsyncMock()
            mock_w3_instance.is_connected.return_value = True
            mock_w3_instance.eth.chain_id = 11155111
            mock_w3_instance.eth.gas_price = 20000000000  # 20 gwei
            mock_w3_instance.eth.get_transaction_count.return_value = 42
            mock_w3_instance.eth.estimate_gas.return_value = 21000
            mock_w3_instance.eth.send_raw_transaction.return_value = AsyncMock()
            mock_w3_instance.eth.send_raw_transaction.return_value.hex.return_value = "0xabc123"
            mock_web3.return_value = mock_w3_instance
            
            # Setup Account mock
            mock_account_instance = Mock()
            mock_account_instance.address = "0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A"
            mock_signed_txn = Mock()
            mock_signed_txn.rawTransaction = b"raw_tx_data"
            mock_account_instance.sign_transaction.return_value = mock_signed_txn
            mock_account.from_key.return_value = mock_account_instance
            
            wallet = EthereumWallet(eth_config)
            return wallet, mock_w3_instance, mock_account_instance
    
    async def test_send_eth_transaction(self, connected_eth_wallet_with_account):
        """Test sending ETH transaction."""
        wallet, mock_w3, mock_account = connected_eth_wallet_with_account
        
        await wallet.connect()
        result = await wallet.send_native_token(
            to_address="0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A",
            amount=Decimal("0.5")
        )
        
        assert isinstance(result, TransactionResult)
        assert result.transaction_hash == "0xabc123"
        assert result.status == TransactionStatus.PENDING
    
    async def test_send_erc20_token_transaction(self, connected_eth_wallet_with_account):
        """Test sending ERC-20 token transaction."""
        wallet, mock_w3, mock_account = connected_eth_wallet_with_account
        
        # Mock token contract
        mock_contract = AsyncMock()
        mock_contract.functions.decimals.return_value.call.return_value = 6
        mock_contract.functions.symbol.return_value.call.return_value = "USDC"
        mock_contract.functions.name.return_value.call.return_value = "USD Coin"
        mock_contract.functions.transfer.return_value.build_transaction.return_value = {
            'chainId': 11155111,
            'gas': 100000,
            'gasPrice': 20000000000,
            'nonce': 42
        }
        
        mock_w3.eth.contract.return_value = mock_contract
        mock_w3.eth.estimate_gas.return_value = 65000
        
        await wallet.connect()
        result = await wallet.send_token(
            token_address="0xA0b86a33E6441CcE67d6ea49e8F7a0D73c37A8c4",
            to_address="0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A", 
            amount=Decimal("100.0")
        )
        
        assert isinstance(result, TransactionResult)
        assert result.status == TransactionStatus.PENDING
    
    async def test_estimate_gas_for_eth_transfer(self, connected_eth_wallet_with_account):
        """Test gas estimation for ETH transfer."""
        wallet, mock_w3, mock_account = connected_eth_wallet_with_account
        
        mock_w3.eth.estimate_gas.return_value = 21000
        
        await wallet.connect()
        gas_estimate = await wallet.estimate_gas(
            to_address="0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A",
            amount=Decimal("1.0")
        )
        
        # Should add 10% buffer: 21000 * 1.1 = 23100
        assert gas_estimate == 23100
    
    async def test_get_current_gas_price(self, connected_eth_wallet_with_account):
        """Test getting current gas price."""
        wallet, mock_w3, mock_account = connected_eth_wallet_with_account
        
        mock_w3.eth.gas_price = 25000000000  # 25 gwei
        
        await wallet.connect()
        gas_price = await wallet.get_gas_price()
        
        assert gas_price == Decimal("25.0")  # Should convert wei to gwei


class TestEthereumWalletTransactionStatus:
    """Test Ethereum wallet transaction status checking."""
    
    async def test_get_confirmed_transaction_status(self):
        """Test getting status of confirmed transaction."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            wallet_address="0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A",
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
        
        with patch('src.wallet.ethereum_wallet.AsyncWeb3') as mock_web3:
            mock_w3_instance = AsyncMock()
            mock_w3_instance.is_connected.return_value = True
            
            # Mock transaction receipt
            mock_receipt = Mock()
            mock_receipt.status = 1  # Success
            mock_receipt.gasUsed = 21000
            mock_receipt.effectiveGasPrice = 20000000000
            mock_receipt.blockNumber = 12345
            mock_w3_instance.eth.get_transaction_receipt.return_value = mock_receipt
            
            mock_web3.return_value = mock_w3_instance
            
            wallet = EthereumWallet(config) 
            await wallet.connect()
            
            result = await wallet.get_transaction_status("0xabc123")
            
            assert result.status == TransactionStatus.CONFIRMED
            assert result.gas_used == 21000
            assert result.block_number == 12345
    
    async def test_get_failed_transaction_status(self):
        """Test getting status of failed transaction."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            wallet_address="0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A",
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
        
        with patch('src.wallet.ethereum_wallet.AsyncWeb3') as mock_web3:
            mock_w3_instance = AsyncMock()
            mock_w3_instance.is_connected.return_value = True
            
            # Mock failed transaction receipt
            mock_receipt = Mock()
            mock_receipt.status = 0  # Failed
            mock_receipt.gasUsed = 21000
            mock_receipt.effectiveGasPrice = 20000000000
            mock_receipt.blockNumber = 12345
            mock_w3_instance.eth.get_transaction_receipt.return_value = mock_receipt
            
            mock_web3.return_value = mock_w3_instance
            
            wallet = EthereumWallet(config)
            await wallet.connect()
            
            result = await wallet.get_transaction_status("0xdef456")
            
            assert result.status == TransactionStatus.FAILED


class TestEthereumWalletUtilities:
    """Test Ethereum wallet utility functions."""
    
    async def test_address_validation(self):
        """Test Ethereum address validation."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            wallet_address="0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A",
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
        
        wallet = EthereumWallet(config)
        
        # Valid Ethereum address
        assert await wallet.validate_address("0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A") == True
        
        # Invalid addresses
        assert await wallet.validate_address("invalid") == False
        assert await wallet.validate_address("0x123") == False
    
    async def test_message_signing(self):
        """Test message signing functionality."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            private_key="0x" + "a" * 64,
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
        
        with patch('src.wallet.ethereum_wallet.AsyncWeb3') as mock_web3, \
             patch('src.wallet.ethereum_wallet.Account') as mock_account:
            
            mock_w3_instance = AsyncMock()
            mock_w3_instance.is_connected.return_value = True
            mock_web3.return_value = mock_w3_instance
            
            mock_account_instance = Mock()
            mock_account_instance.address = "0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A"
            mock_signed_message = Mock()
            mock_signed_message.signature.hex.return_value = "0xsignature"
            mock_account_instance.sign_message_hash.return_value = mock_signed_message
            mock_account.from_key.return_value = mock_account_instance
            
            wallet = EthereumWallet(config)
            await wallet.connect()
            
            signature = await wallet.sign_message("Hello, World!")
            assert signature == "0xsignature"
    
    async def test_read_only_mode_restrictions(self):
        """Test that read-only mode prevents signing operations."""
        from src.wallet.ethereum_wallet import EthereumWallet
        
        config = WalletConfig(
            chain=Chain.ETHEREUM,
            network=NetworkType.TESTNET,
            wallet_address="0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A",
            rpc_url="https://sepolia.infura.io/v3/test-key"
        )
        
        with patch('src.wallet.ethereum_wallet.AsyncWeb3') as mock_web3:
            mock_w3_instance = AsyncMock()
            mock_w3_instance.is_connected.return_value = True
            mock_web3.return_value = mock_w3_instance
            
            wallet = EthereumWallet(config)
            await wallet.connect()
            
            # Should raise error for transaction operations
            with pytest.raises(WalletTransactionError, match="Cannot send transactions in read-only mode"):
                await wallet.send_native_token("0x742C1FE0C6e8D55fA92D95bE3F91Fa0b2D58cA2A", Decimal("1.0"))
            
            # Should raise error for message signing
            with pytest.raises(WalletError, match="Cannot sign messages in read-only mode"):
                await wallet.sign_message("test")


# These tests are expected to fail initially - this is TDD
class TestEthereumWalletIntegration:
    """Integration tests that will initially fail but drive implementation."""
    
    async def test_ethereum_mainnet_integration(self):
        """Test Ethereum mainnet integration."""
        # This test will drive implementation of mainnet-specific features
        pytest.skip("Mainnet integration test - implement after basic functionality")
    
    async def test_base_chain_integration(self):
        """Test Base chain integration."""
        # This test will drive implementation of Base chain-specific features  
        pytest.skip("Base chain integration test - implement after basic functionality")
    
    async def test_transaction_history_fetching(self):
        """Test transaction history fetching."""
        # This will drive implementation of block explorer integration
        pytest.skip("Transaction history - requires block explorer API integration")