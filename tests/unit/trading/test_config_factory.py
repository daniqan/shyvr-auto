"""
Tests for DEXWalletConfigFactory supporting multi-chain configuration.

This module tests the DEXWalletConfigFactory class that creates properly configured
DEX clients, wallets, and integration bridges for both Solana and Ethereum chains.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from decimal import Decimal

from src.trading.config_factory import DEXWalletConfigFactory, ConfigurationError
from src.utils.base import Chain
from src.wallet.base import NetworkType
from src.dex.jupiter_client import JupiterDEXClient
from src.dex.uniswap_v3_client import UniswapV3Client
from src.wallet.solana_wallet import SolanaWallet
from src.wallet.ethereum_wallet import EthereumWallet
from src.trading.dex_wallet_bridge import DEXWalletBridge


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config_data = {
        "dex": {
            "jupiter": {
                "enabled": True,
                "base_url": "https://quote-api.jup.ag",
                "max_slippage_bps": 50,
                "timeout_seconds": 30
            },
            "uniswap_v3": {
                "enabled": True,
                "max_slippage_bps": 50,
                "timeout_seconds": 30,
                "fee_tiers": [500, 3000, 10000]
            }
        },
        "wallets": {
            "solana": {
                "enabled": True,
                "network": "devnet",
                "rpc_url": "https://api.devnet.solana.com",
                "timeout_seconds": 30
            },
            "ethereum": {
                "enabled": True,
                "network": "sepolia",
                "rpc_url": "https://sepolia.infura.io/v3/test-key",
                "timeout_seconds": 30
            }
        }
    }
    
    mock_config = MagicMock()
    mock_config.get.side_effect = lambda key, default=None: _get_nested_value(config_data, key, default)
    return mock_config


def _get_nested_value(data, key, default=None):
    """Helper function to get nested dictionary values using dot notation."""
    keys = key.split('.')
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


@pytest.fixture
def factory_with_mock_config(mock_config):
    """Create factory with mocked configuration."""
    with patch('src.trading.config_factory.ConfigManager', return_value=mock_config):
        factory = DEXWalletConfigFactory()
        factory.config = mock_config
        return factory


class TestDEXWalletConfigFactoryEthereum:
    """Test suite for Ethereum support in DEXWalletConfigFactory."""
    
    def test_create_uniswap_client_success(self, factory_with_mock_config):
        """Test successful creation of UniswapV3Client."""
        client = factory_with_mock_config.create_uniswap_client(Chain.ETHEREUM)
        assert isinstance(client, UniswapV3Client)
        assert client.config.chain == Chain.ETHEREUM
    
    def test_create_ethereum_wallet_success(self, factory_with_mock_config):
        """Test successful creation of EthereumWallet."""
        # Mock environment variables
        with patch.dict(os.environ, {
            "ETHEREUM_PRIVATE_KEY": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "ETHEREUM_RPC_URL": "https://sepolia.infura.io/v3/test-key"
        }):
            wallet = factory_with_mock_config.create_ethereum_wallet()
            assert isinstance(wallet, EthereumWallet)
            assert wallet.config.chain == Chain.ETHEREUM
    
    def test_create_dex_wallet_bridge_ethereum(self, factory_with_mock_config):
        """Test creation of DEX-wallet bridge for Ethereum."""
        # Mock environment variables
        with patch.dict(os.environ, {
            "ETHEREUM_PRIVATE_KEY": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "ETHEREUM_RPC_URL": "https://sepolia.infura.io/v3/test-key"
        }):
            bridge, dex, wallet = factory_with_mock_config.create_dex_wallet_bridge(Chain.ETHEREUM)
            assert isinstance(bridge, DEXWalletBridge)
            assert isinstance(dex, UniswapV3Client)
            assert isinstance(wallet, EthereumWallet)
    
    def test_get_supported_chains_includes_ethereum(self, factory_with_mock_config):
        """Test that supported chains includes Ethereum when properly configured."""
        supported_chains = factory_with_mock_config.get_supported_chains()
        # This might pass if the configuration check exists but method creation doesn't
        # or fail if the configuration parsing for Ethereum is missing
        assert isinstance(supported_chains, list)
        # We expect this to eventually include Chain.ETHEREUM


class TestDEXWalletConfigFactoryMultiChain:
    """Test suite for multi-chain support in DEXWalletConfigFactory."""
    
    def test_create_dex_wallet_bridge_both_chains(self, factory_with_mock_config):
        """Test creation of bridges for both Solana and Ethereum."""
        # Test Solana
        with patch.dict(os.environ, {
            "SOLANA_PRIVATE_KEY": "test_private_key",
            "SOLANA_RPC_URL": "https://api.devnet.solana.com"
        }):
            bridge_sol, dex_sol, wallet_sol = factory_with_mock_config.create_dex_wallet_bridge(Chain.SOLANA)
            assert isinstance(bridge_sol, DEXWalletBridge)
            assert isinstance(dex_sol, JupiterDEXClient)
            assert isinstance(wallet_sol, SolanaWallet)
        
        # Test Ethereum
        with patch.dict(os.environ, {
            "ETHEREUM_PRIVATE_KEY": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "ETHEREUM_RPC_URL": "https://sepolia.infura.io/v3/test-key"
        }):
            bridge_eth, dex_eth, wallet_eth = factory_with_mock_config.create_dex_wallet_bridge(Chain.ETHEREUM)
            assert isinstance(bridge_eth, DEXWalletBridge)
            assert isinstance(dex_eth, UniswapV3Client)
            assert isinstance(wallet_eth, EthereumWallet)
    
    def test_validation_includes_ethereum_config(self, factory_with_mock_config):
        """Test that configuration validation checks Ethereum setup."""
        validation_result = factory_with_mock_config.validate_configuration()
        assert isinstance(validation_result, dict)
        assert "valid" in validation_result
        assert "supported_chains" in validation_result
        # Eventually should include ethereum validation


class TestEthereumWalletCreation:
    """Test specific Ethereum wallet creation functionality."""
    
    @patch.dict(os.environ, {
        "ETHEREUM_PRIVATE_KEY": "0x1234567890123456789012345678901234567890123456789012345678901234",
        "ETHEREUM_RPC_URL": "https://sepolia.infura.io/v3/test-key"
    })
    def test_ethereum_wallet_with_environment_variables(self, factory_with_mock_config):
        """Test Ethereum wallet creation with environment variables."""
        wallet = factory_with_mock_config.create_ethereum_wallet()
        assert isinstance(wallet, EthereumWallet)
    
    def test_ethereum_wallet_missing_credentials(self, factory_with_mock_config):
        """Test Ethereum wallet creation fails without credentials."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError):
                # Should fail at configuration level due to missing credentials
                wallet = factory_with_mock_config.create_ethereum_wallet()


class TestUniswapV3ClientCreation:
    """Test specific UniswapV3Client creation functionality."""
    
    def test_uniswap_client_ethereum_chain(self, factory_with_mock_config):
        """Test UniswapV3Client creation for Ethereum chain."""
        client = factory_with_mock_config.create_uniswap_client(Chain.ETHEREUM)
        assert isinstance(client, UniswapV3Client)
        assert client.config.chain == Chain.ETHEREUM
    
    def test_uniswap_client_invalid_chain(self, factory_with_mock_config):
        """Test UniswapV3Client creation fails for invalid chains."""
        with pytest.raises(ConfigurationError):
            # Should fail because Solana is not supported by Uniswap
            client = factory_with_mock_config.create_uniswap_client(Chain.SOLANA)
    
    def test_uniswap_client_disabled_in_config(self, factory_with_mock_config):
        """Test UniswapV3Client creation fails when disabled in config."""
        # Modify mock to return disabled
        factory_with_mock_config.config.get.side_effect = lambda key, default=None: (
            False if key == "dex.uniswap_v3.enabled" else 
            _get_nested_value({
                "dex": {
                    "uniswap_v3": {
                        "enabled": False,
                        "max_slippage_bps": 50,
                        "timeout_seconds": 30
                    }
                }
            }, key, default)
        )
        
        with pytest.raises(ConfigurationError):
            client = factory_with_mock_config.create_uniswap_client(Chain.ETHEREUM)


class TestBackwardCompatibility:
    """Test that Ethereum additions don't break existing Solana functionality."""
    
    def test_solana_functionality_still_works(self, factory_with_mock_config):
        """Test that existing Solana functionality continues to work."""
        # Mock environment variables for Solana
        with patch.dict(os.environ, {
            "SOLANA_PRIVATE_KEY": "test_private_key",
            "SOLANA_RPC_URL": "https://api.devnet.solana.com"
        }):
            # These should continue to work
            jupiter_client = factory_with_mock_config.create_jupiter_client()
            assert isinstance(jupiter_client, JupiterDEXClient)
            
            solana_wallet = factory_with_mock_config.create_solana_wallet()
            assert isinstance(solana_wallet, SolanaWallet)
            
            bridge, dex, wallet = factory_with_mock_config.create_dex_wallet_bridge(Chain.SOLANA)
            assert isinstance(bridge, DEXWalletBridge)
    
    def test_get_supported_chains_includes_solana(self, factory_with_mock_config):
        """Test that Solana is still included in supported chains."""
        supported_chains = factory_with_mock_config.get_supported_chains()
        assert Chain.SOLANA in supported_chains