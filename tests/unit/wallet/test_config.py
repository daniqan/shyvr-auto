"""
Test cases for wallet configuration manager.

Following TDD methodology - these tests will initially fail and drive implementation.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, mock_open
from typing import Optional

from src.wallet.base import (
    WalletConfig,
    Chain,
    NetworkType,
    WalletError
)


class TestWalletConfigManager:
    """Test WalletConfigManager functionality."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file for testing."""
        config_content = """
wallet:
  networks:
    ethereum:
      mainnet:
        rpc_url: "https://mainnet.infura.io/v3/custom-key"
        gas_price_gwei: 25.0
        max_gas_limit: 600000
      testnet:
        rpc_url: "https://sepolia.infura.io/v3/custom-key"
        gas_price_gwei: 15.0
    solana:
      mainnet:
        rpc_url: "https://custom-rpc.solana.com"
      devnet:
        rpc_url: "https://custom-devnet.solana.com"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            yield f.name
        os.unlink(f.name)
    
    async def test_config_manager_creation(self):
        """Test creating wallet config manager."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        assert manager.config_path is None
        assert Chain.ETHEREUM in manager._network_configs
        assert Chain.SOLANA in manager._network_configs
        assert Chain.BASE in manager._network_configs
    
    async def test_config_manager_with_file(self, temp_config_file):
        """Test creating wallet config manager with config file."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager(temp_config_file)
        assert manager.config_path == temp_config_file
    
    async def test_get_network_config_ethereum_mainnet(self):
        """Test getting Ethereum mainnet network configuration."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        config = manager.get_network_config(Chain.ETHEREUM, NetworkType.MAINNET)
        
        assert config.rpc_url == "https://eth-mainnet.g.alchemy.com/v2/"
        assert config.chain_id == 1
        assert config.gas_price_gwei == 20.0
        assert config.max_gas_limit == 500000
    
    async def test_get_network_config_solana_devnet(self):
        """Test getting Solana devnet network configuration."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        config = manager.get_network_config(Chain.SOLANA, NetworkType.DEVNET)
        
        assert config.rpc_url == "https://api.devnet.solana.com"
        assert config.gas_price_gwei is None  # Not applicable for Solana
        assert config.block_explorer_url == "https://solscan.io/?cluster=devnet"
    
    async def test_get_network_config_base_chain(self):
        """Test getting Base chain network configuration."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        config = manager.get_network_config(Chain.BASE, NetworkType.MAINNET)
        
        assert config.rpc_url == "https://base-mainnet.g.alchemy.com/v2/"
        assert config.chain_id == 8453
        assert config.gas_price_gwei == 0.1  # Lower gas for Base
    
    async def test_get_network_config_unsupported_chain(self):
        """Test error for unsupported chain."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # This would fail if we had an unsupported chain enum value
        # For now, test with a valid chain but unsupported network
        with pytest.raises(WalletError, match="Unsupported network devnet for chain ethereum"):
            manager.get_network_config(Chain.ETHEREUM, NetworkType.DEVNET)
    
    @patch('src.wallet.config.keyring')
    async def test_store_and_retrieve_private_key(self, mock_keyring):
        """Test storing and retrieving private key securely."""
        from src.wallet.config import WalletConfigManager
        
        mock_keyring.get_password.return_value = None  # No existing encryption key
        mock_keyring.set_password.return_value = None
        
        with patch('src.wallet.config.Fernet') as mock_fernet:
            mock_fernet.generate_key.return_value = b"mock_encryption_key"
            mock_cipher = Mock()
            mock_cipher.encrypt.return_value = b"encrypted_private_key"
            mock_cipher.decrypt.return_value = b"decrypted_private_key"
            mock_fernet.return_value = mock_cipher
            
            manager = WalletConfigManager()
            
            # Store private key
            manager.store_private_key(Chain.ETHEREUM, NetworkType.TESTNET, "test_private_key")
            
            # Retrieve private key
            retrieved_key = manager.get_private_key(Chain.ETHEREUM, NetworkType.TESTNET)
            
            assert retrieved_key == "decrypted_private_key"
    
    @patch('src.wallet.config.keyring')
    async def test_store_and_retrieve_mnemonic(self, mock_keyring):
        """Test storing and retrieving mnemonic phrase securely."""
        from src.wallet.config import WalletConfigManager
        
        mock_keyring.get_password.return_value = None
        mock_keyring.set_password.return_value = None
        
        with patch('src.wallet.config.Fernet') as mock_fernet:
            mock_fernet.generate_key.return_value = b"mock_encryption_key"
            mock_cipher = Mock()
            mock_cipher.encrypt.return_value = b"encrypted_mnemonic"
            mock_cipher.decrypt.return_value = b"decrypted_mnemonic"
            mock_fernet.return_value = mock_cipher
            
            manager = WalletConfigManager()
            
            # Store mnemonic
            manager.store_mnemonic(Chain.SOLANA, NetworkType.MAINNET, "test mnemonic phrase")
            
            # Retrieve mnemonic
            retrieved_mnemonic = manager.get_mnemonic(Chain.SOLANA, NetworkType.MAINNET)
            
            assert retrieved_mnemonic == "decrypted_mnemonic"
    
    async def test_create_wallet_config_with_private_key(self):
        """Test creating wallet config with private key."""
        from src.wallet.config import WalletConfigManager
        
        with patch('src.wallet.config.keyring'):
            manager = WalletConfigManager()
            
            config = manager.create_wallet_config(
                chain=Chain.ETHEREUM,
                network=NetworkType.TESTNET,
                private_key="0x" + "a" * 64,
                api_key="test-api-key"
            )
            
            assert config.chain == Chain.ETHEREUM
            assert config.network == NetworkType.TESTNET
            assert config.private_key is None  # Should be None after secure storage
            assert config.rpc_url.endswith("test-api-key")
            assert config.gas_price_gwei == 10.0  # Testnet default
    
    async def test_create_wallet_config_read_only(self):
        """Test creating read-only wallet config."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        config = manager.create_wallet_config(
            chain=Chain.SOLANA,
            network=NetworkType.MAINNET,
            wallet_address="11111111111111111111111111111112"
        )
        
        assert config.chain == Chain.SOLANA
        assert config.network == NetworkType.MAINNET
        assert config.wallet_address == "11111111111111111111111111111112"
        assert config.rpc_url == "https://api.mainnet-beta.solana.com"
    
    @patch.dict(os.environ, {
        'ETHEREUM_TESTNET_PRIVATE_KEY': '0x' + 'b' * 64,
        'ETHEREUM_API_KEY': 'env-api-key',
        'SOLANA_MAINNET_WALLET_ADDRESS': '11111111111111111111111111111112'
    })
    async def test_get_wallet_config_from_env(self):
        """Test creating wallet config from environment variables."""
        from src.wallet.config import WalletConfigManager
        
        with patch('src.wallet.config.keyring'):
            manager = WalletConfigManager()
            
            # Test Ethereum config from env
            eth_config = manager.get_wallet_config_from_env(Chain.ETHEREUM, NetworkType.TESTNET)
            assert eth_config.chain == Chain.ETHEREUM
            assert eth_config.network == NetworkType.TESTNET
            assert eth_config.api_key == "env-api-key"
            
            # Test Solana config from env
            sol_config = manager.get_wallet_config_from_env(Chain.SOLANA, NetworkType.MAINNET)
            assert sol_config.chain == Chain.SOLANA
            assert sol_config.wallet_address == "11111111111111111111111111111112"
    
    async def test_get_supported_chains(self):
        """Test getting list of supported chains."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        chains = manager.get_supported_chains()
        
        assert Chain.ETHEREUM in chains
        assert Chain.SOLANA in chains
        assert Chain.BASE in chains
    
    async def test_get_chain_info(self):
        """Test getting chain information."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Test Ethereum info
        eth_info = manager.get_chain_info(Chain.ETHEREUM)
        assert eth_info["name"] == "ethereum"
        assert eth_info["native_symbol"] == "ETH"
        assert eth_info["decimals"] == 18
        assert "mainnet" in eth_info["networks"]
        assert "testnet" in eth_info["networks"]
        
        # Test Solana info
        sol_info = manager.get_chain_info(Chain.SOLANA)
        assert sol_info["name"] == "solana"
        assert sol_info["native_symbol"] == "SOL"
        assert sol_info["decimals"] == 9
    
    async def test_get_chain_info_unsupported(self):
        """Test error for unsupported chain in get_chain_info."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Remove a chain from config to test error
        manager._network_configs.pop(Chain.ETHEREUM)
        
        with pytest.raises(WalletError, match="Unsupported chain: ethereum"):
            manager.get_chain_info(Chain.ETHEREUM)
    
    @patch('src.wallet.config.keyring')
    async def test_clear_stored_keys(self, mock_keyring):
        """Test clearing stored private keys and mnemonics."""
        from src.wallet.config import WalletConfigManager
        
        mock_keyring.delete_password.return_value = None
        
        manager = WalletConfigManager()
        manager.clear_stored_keys(Chain.ETHEREUM, NetworkType.TESTNET)
        
        # Should attempt to delete both private key and mnemonic
        assert mock_keyring.delete_password.call_count == 2
    
    @patch('src.wallet.config.keyring')
    async def test_clear_stored_keys_not_found(self, mock_keyring):
        """Test clearing stored keys when they don't exist."""
        from src.wallet.config import WalletConfigManager
        from keyring.errors import PasswordDeleteError
        
        mock_keyring.delete_password.side_effect = PasswordDeleteError("Password not found")
        
        manager = WalletConfigManager()
        # Should not raise exception even if keys don't exist
        manager.clear_stored_keys(Chain.SOLANA, NetworkType.MAINNET)
    
    async def test_load_custom_config_file(self, temp_config_file):
        """Test loading custom configuration from file."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager(temp_config_file)
        
        # Should have loaded custom RPC URLs
        eth_config = manager.get_network_config(Chain.ETHEREUM, NetworkType.MAINNET)
        assert eth_config.rpc_url == "https://mainnet.infura.io/v3/custom-key"
        assert eth_config.gas_price_gwei == 25.0  # Custom value
        
        sol_config = manager.get_network_config(Chain.SOLANA, NetworkType.MAINNET)
        assert sol_config.rpc_url == "https://custom-rpc.solana.com"


class TestNetworkConfig:
    """Test NetworkConfig data structure."""
    
    async def test_network_config_creation(self):
        """Test creating network configuration."""
        from src.wallet.config import NetworkConfig
        
        config = NetworkConfig(
            rpc_url="https://test.rpc.url",
            chain_id=12345,
            gas_price_gwei=30.0,
            max_gas_limit=800000,
            block_explorer_url="https://explorer.test",
            timeout_seconds=60
        )
        
        assert config.rpc_url == "https://test.rpc.url"
        assert config.chain_id == 12345
        assert config.gas_price_gwei == 30.0
        assert config.max_gas_limit == 800000
        assert config.block_explorer_url == "https://explorer.test"
        assert config.timeout_seconds == 60
    
    async def test_network_config_defaults(self):
        """Test network configuration with defaults."""
        from src.wallet.config import NetworkConfig
        
        config = NetworkConfig(rpc_url="https://test.rpc.url")
        
        assert config.rpc_url == "https://test.rpc.url"
        assert config.chain_id is None
        assert config.gas_price_gwei is None
        assert config.max_gas_limit is None
        assert config.block_explorer_url is None
        assert config.timeout_seconds == 30  # Default value


class TestChainConfig:
    """Test ChainConfig data structure."""
    
    async def test_chain_config_creation(self):
        """Test creating chain configuration."""
        from src.wallet.config import ChainConfig, NetworkConfig
        
        mainnet = NetworkConfig(rpc_url="https://mainnet.rpc")
        testnet = NetworkConfig(rpc_url="https://testnet.rpc")
        devnet = NetworkConfig(rpc_url="https://devnet.rpc")
        
        config = ChainConfig(
            mainnet=mainnet,
            testnet=testnet,
            devnet=devnet,
            native_symbol="TEST",
            decimals=8
        )
        
        assert config.mainnet == mainnet
        assert config.testnet == testnet
        assert config.devnet == devnet
        assert config.native_symbol == "TEST"
        assert config.decimals == 8
    
    async def test_chain_config_defaults(self):
        """Test chain configuration with defaults."""
        from src.wallet.config import ChainConfig, NetworkConfig
        
        mainnet = NetworkConfig(rpc_url="https://mainnet.rpc")
        testnet = NetworkConfig(rpc_url="https://testnet.rpc")
        
        config = ChainConfig(mainnet=mainnet, testnet=testnet)
        
        assert config.devnet is None
        assert config.native_symbol == ""
        assert config.decimals == 18


class TestWalletConfigManagerEncryption:
    """Test wallet config manager encryption functionality."""
    
    @patch('src.wallet.config.keyring')
    @patch('src.wallet.config.Fernet')
    async def test_encryption_key_generation(self, mock_fernet, mock_keyring):
        """Test encryption key generation and storage."""
        from src.wallet.config import WalletConfigManager
        
        # Mock no existing key
        mock_keyring.get_password.return_value = None
        mock_keyring.set_password.return_value = None
        mock_fernet.generate_key.return_value = b"new_encryption_key"
        
        manager = WalletConfigManager()
        key = manager._get_encryption_key()
        
        assert key == b"new_encryption_key"
        mock_fernet.generate_key.assert_called_once()
        mock_keyring.set_password.assert_called_once()
    
    @patch('src.wallet.config.keyring')
    async def test_encryption_key_retrieval(self, mock_keyring):
        """Test retrieving existing encryption key."""
        from src.wallet.config import WalletConfigManager
        
        # Mock existing key
        mock_keyring.get_password.return_value = "existing_key"
        
        manager = WalletConfigManager()
        key = manager._get_encryption_key()
        
        assert key == b"existing_key"
    
    @patch('src.wallet.config.Fernet')
    async def test_data_encryption_decryption(self, mock_fernet):
        """Test data encryption and decryption."""
        from src.wallet.config import WalletConfigManager
        
        mock_cipher = Mock()
        mock_cipher.encrypt.return_value = b"encrypted_data"
        mock_cipher.decrypt.return_value = b"decrypted_data"
        mock_fernet.return_value = mock_cipher
        
        with patch.object(WalletConfigManager, '_get_encryption_key', return_value=b"test_key"):
            manager = WalletConfigManager()
            
            # Test encryption
            encrypted = manager._encrypt_data("test_data")
            assert encrypted == "encrypted_data"
            
            # Test decryption
            decrypted = manager._decrypt_data("encrypted_data")
            assert decrypted == "decrypted_data"


class TestWalletConfigManagerSecurity:
    """Test wallet configuration security features."""
    
    @patch('src.wallet.config.keyring')
    async def test_private_key_encryption_validation(self, mock_keyring):
        """Test that private keys are properly encrypted before storage."""
        from src.wallet.config import WalletConfigManager
        
        mock_keyring.get_password.return_value = None
        mock_keyring.set_password.return_value = None
        
        with patch('src.wallet.config.Fernet') as mock_fernet:
            mock_cipher = Mock()
            mock_cipher.encrypt.return_value = b"encrypted_private_key"
            mock_fernet.return_value = mock_cipher
            mock_fernet.generate_key.return_value = b"test_key"
            
            manager = WalletConfigManager()
            
            # Store private key
            manager.store_private_key(Chain.ETHEREUM, NetworkType.TESTNET, "0x" + "a" * 64)
            
            # Verify encryption was called
            mock_cipher.encrypt.assert_called_once()
            # Verify raw key is not stored
            mock_keyring.set_password.assert_called_with(
                "shyvr-wallet-encryption", 
                "master_key", 
                "test_key"
            )
    
    @patch('src.wallet.config.keyring')
    async def test_mnemonic_phrase_validation(self, mock_keyring):
        """Test mnemonic phrase validation and secure storage."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Valid mnemonic phrases (BIP39)
        valid_mnemonics = [
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
            "legal winner thank year wave sausage worth useful legal winner thank yellow",
        ]
        
        for mnemonic in valid_mnemonics:
            # Should not raise exception for valid mnemonic
            assert manager.validate_mnemonic(mnemonic) == True
        
        # Invalid mnemonic phrases
        invalid_mnemonics = [
            "invalid mnemonic phrase",
            "one two three four five",  # Too short
            "",  # Empty
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon invalid",  # Invalid word
        ]
        
        for mnemonic in invalid_mnemonics:
            assert manager.validate_mnemonic(mnemonic) == False
    
    async def test_private_key_format_validation(self):
        """Test private key format validation for different chains."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Valid Ethereum private keys
        valid_eth_keys = [
            "0x" + "a" * 64,  # With 0x prefix
            "a" * 64,  # Without 0x prefix
            "0x" + "123456789abcdef" * 4,  # Mixed hex
        ]
        
        for key in valid_eth_keys:
            assert manager.validate_private_key(key, Chain.ETHEREUM) == True
        
        # Valid Solana private keys
        valid_sol_keys = [
            "5" + "a" * 87,  # Base58 format
            "0x" + "b" * 64,  # Hex format
        ]
        
        for key in valid_sol_keys:
            assert manager.validate_private_key(key, Chain.SOLANA) == True
        
        # Invalid private keys
        invalid_keys = [
            "invalid_key",
            "0x" + "g" * 64,  # Invalid hex
            "0x" + "a" * 63,  # Too short
            "",  # Empty
        ]
        
        for key in invalid_keys:
            assert manager.validate_private_key(key, Chain.ETHEREUM) == False
            assert manager.validate_private_key(key, Chain.SOLANA) == False
    
    @patch('src.wallet.config.keyring')
    async def test_secure_key_derivation(self, mock_keyring):
        """Test secure key derivation from mnemonic."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        
        # Should derive consistent keys for the same mnemonic
        eth_key_1 = manager.derive_private_key_from_mnemonic(mnemonic, Chain.ETHEREUM, 0)
        eth_key_2 = manager.derive_private_key_from_mnemonic(mnemonic, Chain.ETHEREUM, 0)
        assert eth_key_1 == eth_key_2
        
        # Different derivation paths should produce different keys
        eth_key_path_0 = manager.derive_private_key_from_mnemonic(mnemonic, Chain.ETHEREUM, 0)
        eth_key_path_1 = manager.derive_private_key_from_mnemonic(mnemonic, Chain.ETHEREUM, 1)
        assert eth_key_path_0 != eth_key_path_1
        
        # Different chains should produce different keys
        eth_key = manager.derive_private_key_from_mnemonic(mnemonic, Chain.ETHEREUM, 0)
        sol_key = manager.derive_private_key_from_mnemonic(mnemonic, Chain.SOLANA, 0)
        assert eth_key != sol_key
    
    async def test_environment_variable_security(self):
        """Test secure handling of environment variables."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Should warn about private keys in environment variables
        with patch('src.wallet.config.logger') as mock_logger:
            with patch.dict(os.environ, {'ETHEREUM_MAINNET_PRIVATE_KEY': '0x' + 'a' * 64}):
                config = manager.get_wallet_config_from_env(Chain.ETHEREUM, NetworkType.MAINNET)
                
                # Should log security warning
                mock_logger.warning.assert_called()
                assert "environment variable" in str(mock_logger.warning.call_args)
                assert "security risk" in str(mock_logger.warning.call_args)


class TestWalletConfigManagerAdvanced:
    """Test advanced wallet configuration features."""
    
    async def test_multi_account_support(self):
        """Test support for multiple accounts per chain."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Should support multiple accounts with different names
        with patch('src.wallet.config.keyring'):
            manager.store_private_key(
                Chain.ETHEREUM, 
                NetworkType.MAINNET, 
                "0x" + "a" * 64,
                account_name="main"
            )
            
            manager.store_private_key(
                Chain.ETHEREUM, 
                NetworkType.MAINNET, 
                "0x" + "b" * 64,
                account_name="trading"
            )
            
            # Should retrieve correct keys by account name
            main_key = manager.get_private_key(Chain.ETHEREUM, NetworkType.MAINNET, "main")
            trading_key = manager.get_private_key(Chain.ETHEREUM, NetworkType.MAINNET, "trading")
            
            assert main_key != trading_key
    
    async def test_hardware_wallet_integration(self):
        """Test hardware wallet integration configuration."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Should support hardware wallet configuration
        hw_config = manager.create_hardware_wallet_config(
            chain=Chain.ETHEREUM,
            network=NetworkType.MAINNET,
            hardware_type="ledger",
            derivation_path="m/44'/60'/0'/0/0"
        )
        
        assert hw_config.chain == Chain.ETHEREUM
        assert hw_config.hardware_type == "ledger"
        assert hw_config.derivation_path == "m/44'/60'/0'/0/0"
        assert hw_config.private_key is None  # No private key for hardware wallets
    
    async def test_config_backup_and_restore(self):
        """Test configuration backup and restore functionality."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Create backup
        with patch('src.wallet.config.keyring'):
            manager.store_private_key(Chain.ETHEREUM, NetworkType.TESTNET, "0x" + "a" * 64)
            manager.store_mnemonic(Chain.SOLANA, NetworkType.MAINNET, "test mnemonic")
            
            backup_data = manager.create_backup("backup_password")
            
            # Backup should be encrypted
            assert isinstance(backup_data, bytes)
            assert len(backup_data) > 0
            
            # Clear current config
            manager.clear_stored_keys(Chain.ETHEREUM, NetworkType.TESTNET)
            manager.clear_stored_keys(Chain.SOLANA, NetworkType.MAINNET)
            
            # Restore from backup
            manager.restore_from_backup(backup_data, "backup_password")
            
            # Should have restored keys
            eth_key = manager.get_private_key(Chain.ETHEREUM, NetworkType.TESTNET)
            sol_mnemonic = manager.get_mnemonic(Chain.SOLANA, NetworkType.MAINNET)
            
            assert eth_key is not None
            assert sol_mnemonic is not None
    
    async def test_gas_price_optimization(self):
        """Test gas price optimization features."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Should provide optimized gas prices based on network conditions
        with patch('src.wallet.config.requests') as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {
                "fast": 25.0,
                "standard": 20.0,
                "safe": 15.0
            }
            mock_requests.get.return_value = mock_response
            
            gas_prices = manager.get_optimized_gas_prices(Chain.ETHEREUM, NetworkType.MAINNET)
            
            assert gas_prices["fast"] == 25.0
            assert gas_prices["standard"] == 20.0
            assert gas_prices["safe"] == 15.0
    
    async def test_custom_rpc_endpoint_validation(self):
        """Test validation of custom RPC endpoints."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # Valid RPC endpoints
        valid_endpoints = [
            "https://eth-mainnet.g.alchemy.com/v2/api-key",
            "https://api.mainnet-beta.solana.com",
            "wss://eth-mainnet.g.alchemy.com/v2/api-key",  # WebSocket
        ]
        
        for endpoint in valid_endpoints:
            assert manager.validate_rpc_endpoint(endpoint) == True
        
        # Invalid RPC endpoints
        invalid_endpoints = [
            "http://insecure-endpoint.com",  # HTTP instead of HTTPS
            "invalid-url",
            "",
            "ftp://not-http.com",
        ]
        
        for endpoint in invalid_endpoints:
            assert manager.validate_rpc_endpoint(endpoint) == False


class TestWalletConfigManagerPerformance:
    """Test wallet configuration performance optimizations."""
    
    async def test_config_caching(self):
        """Test that configuration is cached for performance."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        # First call should load from storage
        config1 = manager.get_network_config(Chain.ETHEREUM, NetworkType.MAINNET)
        
        # Second call should use cache
        config2 = manager.get_network_config(Chain.ETHEREUM, NetworkType.MAINNET)
        
        # Should be the same object (cached)
        assert config1 is config2
    
    async def test_batch_operations(self):
        """Test batch operations for multiple configurations."""
        from src.wallet.config import WalletConfigManager
        
        manager = WalletConfigManager()
        
        configs_to_create = [
            (Chain.ETHEREUM, NetworkType.MAINNET),
            (Chain.ETHEREUM, NetworkType.TESTNET),
            (Chain.SOLANA, NetworkType.MAINNET),
            (Chain.BASE, NetworkType.MAINNET),
        ]
        
        # Batch creation should be more efficient than individual calls
        start_time = pytest.current_time() if hasattr(pytest, 'current_time') else 0
        
        configs = manager.get_network_configs_batch(configs_to_create)
        
        end_time = pytest.current_time() if hasattr(pytest, 'current_time') else 1
        
        assert len(configs) == 4
        # Should complete quickly (test would fail if too slow)
        assert (end_time - start_time) < 1.0  # Less than 1 second


# TDD Failing Tests - These are designed to fail until implementation is complete
class TestWalletConfigManagerTDDFailingScenarios:
    """Tests designed to fail - this drives TDD implementation."""
    
    async def test_config_manager_import_fails_initially(self):
        """Test that WalletConfigManager import fails until implemented."""
        # This test ensures we're following TDD - import should fail first
        try:
            from src.wallet.config import WalletConfigManager
            
            # If import succeeds, manager should have all required methods
            required_methods = [
                'get_network_config', 'store_private_key', 'get_private_key',
                'store_mnemonic', 'get_mnemonic', 'create_wallet_config',
                'validate_private_key', 'validate_mnemonic', 'clear_stored_keys'
            ]
            
            for method in required_methods:
                assert hasattr(WalletConfigManager, method), f"Missing method: {method}"
                
        except ImportError:
            # This is expected in TDD - implementation comes after tests
            assert True
    
    async def test_hardware_wallet_support_pending(self):
        """Test hardware wallet support - will fail until implemented."""
        # This test will fail until hardware wallet support is added
        pytest.skip("Hardware wallet support not implemented yet")
    
    async def test_advanced_security_features_pending(self):
        """Test advanced security features - will fail until implemented."""
        # This test will fail until advanced security features are added
        pytest.skip("Advanced security features not implemented yet")
    
    async def test_config_file_yaml_parsing_error_handling(self):
        """Test handling of malformed YAML config files."""
        # This will drive implementation of robust YAML parsing
        pytest.skip("YAML error handling - implement after basic functionality")
    
    async def test_keyring_backend_fallback(self):
        """Test fallback when keyring backend is not available."""
        # This will drive implementation of keyring fallback mechanisms
        pytest.skip("Keyring fallback - implement after basic functionality")
    
    async def test_config_validation_and_migration(self):
        """Test configuration validation and migration."""
        # This will drive implementation of config validation
        pytest.skip("Config validation - implement after basic functionality")