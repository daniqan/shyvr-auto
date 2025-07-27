"""
End-to-end integration tests for Ethereum support in DEXWalletConfigFactory.

This module tests the complete integration from configuration to working
wallet/DEX pairs for Ethereum chains.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from src.trading.config_factory import DEXWalletConfigFactory, ConfigurationError
from src.utils.base import Chain
from src.wallet.base import NetworkType
from src.dex.uniswap_v3_client import UniswapV3Client
from src.wallet.ethereum_wallet import EthereumWallet
from src.trading.dex_wallet_bridge import DEXWalletBridge


@pytest.fixture
def ethereum_config():
    """Create Ethereum configuration for testing."""
    config_data = {
        "dex": {
            "uniswap_v3": {
                "enabled": True,
                "max_slippage_bps": 50,
                "timeout_seconds": 30,
                "fee_tiers": [500, 3000, 10000]
            }
        },
        "wallets": {
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


class TestEthereumFactoryIntegration:
    """End-to-end integration tests for Ethereum factory support."""
    
    @patch.dict(os.environ, {
        "ETHEREUM_PRIVATE_KEY": "0x1234567890123456789012345678901234567890123456789012345678901234",
        "ETHEREUM_RPC_URL": "https://sepolia.infura.io/v3/test-key"
    })
    def test_ethereum_end_to_end_integration(self, ethereum_config):
        """Test complete end-to-end Ethereum integration."""
        with patch('src.trading.config_factory.ConfigManager', return_value=ethereum_config):
            # Create factory
            factory = DEXWalletConfigFactory()
            factory.config = ethereum_config
            
            # Test individual component creation
            uniswap_client = factory.create_uniswap_client(Chain.ETHEREUM)
            assert isinstance(uniswap_client, UniswapV3Client)
            assert uniswap_client.config.chain == Chain.ETHEREUM
            
            ethereum_wallet = factory.create_ethereum_wallet(Chain.ETHEREUM)
            assert isinstance(ethereum_wallet, EthereumWallet)
            assert ethereum_wallet.config.chain == Chain.ETHEREUM
            assert ethereum_wallet.config.network == NetworkType.TESTNET
            
            # Test integrated bridge creation
            bridge, dex, wallet = factory.create_dex_wallet_bridge(Chain.ETHEREUM)
            assert isinstance(bridge, DEXWalletBridge)
            assert isinstance(dex, UniswapV3Client)
            assert isinstance(wallet, EthereumWallet)
            
            # Verify chain consistency
            assert bridge.chain == Chain.ETHEREUM
            assert dex.config.chain == Chain.ETHEREUM
            assert wallet.config.chain == Chain.ETHEREUM
    
    @patch.dict(os.environ, {
        "ETHEREUM_PRIVATE_KEY": "0x1234567890123456789012345678901234567890123456789012345678901234",
        "ETHEREUM_RPC_URL": "https://sepolia.infura.io/v3/test-key"
    })
    def test_ethereum_base_chain_support(self, ethereum_config):
        """Test that the factory supports Base chain (EVM compatible)."""
        with patch('src.trading.config_factory.ConfigManager', return_value=ethereum_config):
            factory = DEXWalletConfigFactory()
            factory.config = ethereum_config
            
            # Test Base chain support
            uniswap_client = factory.create_uniswap_client(Chain.BASE)
            assert isinstance(uniswap_client, UniswapV3Client)
            assert uniswap_client.config.chain == Chain.BASE
            
            ethereum_wallet = factory.create_ethereum_wallet(Chain.BASE)
            assert isinstance(ethereum_wallet, EthereumWallet)
            assert ethereum_wallet.config.chain == Chain.BASE
            
            # Test integrated bridge creation for Base
            bridge, dex, wallet = factory.create_dex_wallet_bridge(Chain.BASE)
            assert isinstance(bridge, DEXWalletBridge)
            assert bridge.chain == Chain.BASE
    
    @patch.dict(os.environ, {
        "SOLANA_PRIVATE_KEY": "test_private_key",
        "SOLANA_RPC_URL": "https://api.devnet.solana.com",
        "ETHEREUM_PRIVATE_KEY": "0x1234567890123456789012345678901234567890123456789012345678901234",
        "ETHEREUM_RPC_URL": "https://sepolia.infura.io/v3/test-key"
    })
    def test_multi_chain_configuration_integrity(self, ethereum_config):
        """Test that multi-chain support maintains configuration integrity."""
        # Add Solana config to existing Ethereum config
        multi_chain_config_data = {
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
        
        multi_chain_config = MagicMock()
        multi_chain_config.get.side_effect = lambda key, default=None: _get_nested_value(multi_chain_config_data, key, default)
        
        with patch('src.trading.config_factory.ConfigManager', return_value=multi_chain_config):
            factory = DEXWalletConfigFactory()
            factory.config = multi_chain_config
            
            # Test that both chains work independently
            from src.dex.jupiter_client import JupiterDEXClient
            from src.wallet.solana_wallet import SolanaWallet
            
            # Solana bridge
            bridge_sol, dex_sol, wallet_sol = factory.create_dex_wallet_bridge(Chain.SOLANA)
            assert isinstance(bridge_sol, DEXWalletBridge)
            assert isinstance(dex_sol, JupiterDEXClient)
            assert isinstance(wallet_sol, SolanaWallet)
            assert bridge_sol.chain == Chain.SOLANA
            
            # Ethereum bridge
            bridge_eth, dex_eth, wallet_eth = factory.create_dex_wallet_bridge(Chain.ETHEREUM)
            assert isinstance(bridge_eth, DEXWalletBridge)
            assert isinstance(dex_eth, UniswapV3Client)
            assert isinstance(wallet_eth, EthereumWallet)
            assert bridge_eth.chain == Chain.ETHEREUM
            
            # Verify no cross-contamination
            assert bridge_sol.chain != bridge_eth.chain
            assert type(dex_sol) != type(dex_eth)
            assert type(wallet_sol) != type(wallet_eth)
    
    def test_configuration_validation_with_ethereum(self, ethereum_config):
        """Test that configuration validation works with Ethereum setup."""
        with patch('src.trading.config_factory.ConfigManager', return_value=ethereum_config):
            factory = DEXWalletConfigFactory()
            factory.config = ethereum_config
            
            # Test validation
            validation_result = factory.validate_configuration()
            assert isinstance(validation_result, dict)
            assert "valid" in validation_result
            assert "supported_chains" in validation_result
            
            # Test supported chains detection
            supported_chains = factory.get_supported_chains()
            assert isinstance(supported_chains, list)
    
    def test_error_handling_ethereum_disabled(self, ethereum_config):
        """Test error handling when Ethereum components are disabled."""
        # Modify config to disable Ethereum
        ethereum_config.get.side_effect = lambda key, default=None: (
            False if key in ["dex.uniswap_v3.enabled", "wallets.ethereum.enabled"] else
            _get_nested_value({
                "dex": {"uniswap_v3": {"enabled": False}},
                "wallets": {"ethereum": {"enabled": False}}
            }, key, default)
        )
        
        with patch('src.trading.config_factory.ConfigManager', return_value=ethereum_config):
            factory = DEXWalletConfigFactory()
            factory.config = ethereum_config
            
            # Test that disabled components raise appropriate errors
            with pytest.raises(ConfigurationError, match="Uniswap V3 DEX is disabled"):
                factory.create_uniswap_client(Chain.ETHEREUM)
            
            with pytest.raises(ConfigurationError, match="Ethereum wallet is disabled"):
                factory.create_ethereum_wallet(Chain.ETHEREUM)
    
    def test_invalid_chain_combinations(self, ethereum_config):
        """Test error handling for invalid chain combinations."""
        with patch('src.trading.config_factory.ConfigManager', return_value=ethereum_config):
            factory = DEXWalletConfigFactory()
            factory.config = ethereum_config
            
            # Test invalid chain for Uniswap
            with pytest.raises(ConfigurationError, match="Uniswap V3 only supports Ethereum and Base chains"):
                factory.create_uniswap_client(Chain.SOLANA)
            
            # Test invalid chain for Ethereum wallet
            with pytest.raises(ConfigurationError, match="Ethereum wallet only supports EVM chains"):
                factory.create_ethereum_wallet(Chain.SOLANA)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials_handling(self, ethereum_config):
        """Test handling of missing credentials."""
        with patch('src.trading.config_factory.ConfigManager', return_value=ethereum_config):
            factory = DEXWalletConfigFactory()
            factory.config = ethereum_config
            
            # Test missing Ethereum credentials
            with pytest.raises(ConfigurationError, match="Either ETHEREUM_PRIVATE_KEY or ETHEREUM_WALLET_ADDRESS must be provided"):
                factory.create_ethereum_wallet(Chain.ETHEREUM)