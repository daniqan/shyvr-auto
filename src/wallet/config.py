"""
Wallet configuration management with secure key storage and network settings.

This module handles wallet configuration, including secure private key management,
network configurations, and environment-based settings.
"""

import os
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import yaml
from cryptography.fernet import Fernet
import keyring

from .base import WalletConfig, Chain, NetworkType, WalletError

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Network-specific configuration."""
    rpc_url: str
    chain_id: Optional[int] = None
    gas_price_gwei: Optional[float] = None
    max_gas_limit: Optional[int] = None
    block_explorer_url: Optional[str] = None
    timeout_seconds: int = 30


@dataclass  
class ChainConfig:
    """Chain-specific configuration with network settings."""
    mainnet: NetworkConfig
    testnet: NetworkConfig
    devnet: Optional[NetworkConfig] = None
    native_symbol: str = ""
    decimals: int = 18


class WalletConfigManager:
    """
    Manages wallet configuration and secure key storage.
    
    Handles loading configuration from environment variables and config files,
    with secure private key storage using keyring and encryption.
    """
    
    DEFAULT_NETWORKS = {
        Chain.ETHEREUM: ChainConfig(
            mainnet=NetworkConfig(
                rpc_url="https://eth-mainnet.g.alchemy.com/v2/",
                chain_id=1,
                gas_price_gwei=20.0,
                max_gas_limit=500000,
                block_explorer_url="https://etherscan.io"
            ),
            testnet=NetworkConfig(
                rpc_url="https://eth-sepolia.g.alchemy.com/v2/",
                chain_id=11155111,
                gas_price_gwei=10.0,
                max_gas_limit=500000,
                block_explorer_url="https://sepolia.etherscan.io"
            ),
            native_symbol="ETH",
            decimals=18
        ),
        Chain.BASE: ChainConfig(
            mainnet=NetworkConfig(
                rpc_url="https://base-mainnet.g.alchemy.com/v2/",
                chain_id=8453,
                gas_price_gwei=0.1,
                max_gas_limit=500000,
                block_explorer_url="https://basescan.org"
            ),
            testnet=NetworkConfig(
                rpc_url="https://base-sepolia.g.alchemy.com/v2/",
                chain_id=84532,
                gas_price_gwei=0.1,
                max_gas_limit=500000,
                block_explorer_url="https://sepolia.basescan.org"
            ),
            native_symbol="ETH",
            decimals=18
        ),
        Chain.SOLANA: ChainConfig(
            mainnet=NetworkConfig(
                rpc_url="https://api.mainnet-beta.solana.com",
                gas_price_gwei=None,  # Not applicable for Solana
                max_gas_limit=None,   # Not applicable for Solana
                block_explorer_url="https://solscan.io"
            ),
            testnet=NetworkConfig(
                rpc_url="https://api.testnet.solana.com",
                gas_price_gwei=None,
                max_gas_limit=None,
                block_explorer_url="https://solscan.io/?cluster=testnet"
            ),
            devnet=NetworkConfig(
                rpc_url="https://api.devnet.solana.com",
                gas_price_gwei=None,
                max_gas_limit=None,
                block_explorer_url="https://solscan.io/?cluster=devnet"
            ),
            native_symbol="SOL",
            decimals=9
        )
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize wallet configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self._encryption_key: Optional[bytes] = None
        self._network_configs = self.DEFAULT_NETWORKS.copy()
        
        # Load custom configurations if provided
        if config_path and os.path.exists(config_path):
            self._load_config_file(config_path)
    
    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data."""
        if self._encryption_key is None:
            # Try to get existing key from keyring
            key_str = keyring.get_password("shyvr-rlte", "encryption_key")
            if key_str:
                self._encryption_key = key_str.encode()
            else:
                # Generate new key and store it
                self._encryption_key = Fernet.generate_key()
                keyring.set_password("shyvr-rlte", "encryption_key", 
                                   self._encryption_key.decode())
        return self._encryption_key
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        key = self._get_encryption_key()
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        return encrypted.decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        key = self._get_encryption_key()
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_data.encode())
        return decrypted.decode()
    
    def store_private_key(self, chain: Chain, network: NetworkType, private_key: str) -> None:
        """
        Securely store private key in keyring.
        
        Args:
            chain: Blockchain chain
            network: Network type
            private_key: Private key to store
        """
        key_id = f"{chain.value}_{network.value}_private_key"
        encrypted_key = self._encrypt_data(private_key)
        keyring.set_password("shyvr-rlte-wallets", key_id, encrypted_key)
        logger.info(f"Stored private key for {chain.value} {network.value}")
    
    def get_private_key(self, chain: Chain, network: NetworkType) -> Optional[str]:
        """
        Retrieve private key from secure storage.
        
        Args:
            chain: Blockchain chain
            network: Network type
            
        Returns:
            Decrypted private key or None if not found
        """
        key_id = f"{chain.value}_{network.value}_private_key"
        encrypted_key = keyring.get_password("shyvr-rlte-wallets", key_id)
        if encrypted_key:
            try:
                return self._decrypt_data(encrypted_key)
            except Exception as e:
                logger.error(f"Failed to decrypt private key for {key_id}: {e}")
                return None
        return None
    
    def store_mnemonic(self, chain: Chain, network: NetworkType, mnemonic: str) -> None:
        """
        Securely store mnemonic phrase in keyring.
        
        Args:
            chain: Blockchain chain
            network: Network type
            mnemonic: Mnemonic phrase to store
        """
        key_id = f"{chain.value}_{network.value}_mnemonic"
        encrypted_mnemonic = self._encrypt_data(mnemonic)
        keyring.set_password("shyvr-rlte-wallets", key_id, encrypted_mnemonic)
        logger.info(f"Stored mnemonic for {chain.value} {network.value}")
    
    def get_mnemonic(self, chain: Chain, network: NetworkType) -> Optional[str]:
        """
        Retrieve mnemonic phrase from secure storage.
        
        Args:
            chain: Blockchain chain
            network: Network type
            
        Returns:
            Decrypted mnemonic phrase or None if not found
        """
        key_id = f"{chain.value}_{network.value}_mnemonic"
        encrypted_mnemonic = keyring.get_password("shyvr-rlte-wallets", key_id)
        if encrypted_mnemonic:
            try:
                return self._decrypt_data(encrypted_mnemonic)
            except Exception as e:
                logger.error(f"Failed to decrypt mnemonic for {key_id}: {e}")
                return None
        return None
    
    def _load_config_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            wallet_config = config_data.get('wallet', {})
            
            # Load custom network configurations
            for chain_name, chain_data in wallet_config.get('networks', {}).items():
                try:
                    chain = Chain(chain_name.lower())
                    self._load_chain_config(chain, chain_data)
                except ValueError:
                    logger.warning(f"Unknown chain in config: {chain_name}")
                    
        except Exception as e:
            logger.error(f"Failed to load wallet config from {config_path}: {e}")
    
    def _load_chain_config(self, chain: Chain, chain_data: Dict[str, Any]) -> None:
        """Load configuration for a specific chain."""
        try:
            mainnet_data = chain_data.get('mainnet', {})
            testnet_data = chain_data.get('testnet', {})
            devnet_data = chain_data.get('devnet', {})
            
            mainnet = NetworkConfig(**mainnet_data) if mainnet_data else self._network_configs[chain].mainnet
            testnet = NetworkConfig(**testnet_data) if testnet_data else self._network_configs[chain].testnet
            devnet = NetworkConfig(**devnet_data) if devnet_data else self._network_configs[chain].devnet
            
            self._network_configs[chain] = ChainConfig(
                mainnet=mainnet,
                testnet=testnet,
                devnet=devnet,
                native_symbol=chain_data.get('native_symbol', self._network_configs[chain].native_symbol),
                decimals=chain_data.get('decimals', self._network_configs[chain].decimals)
            )
            
        except Exception as e:
            logger.error(f"Failed to load config for {chain.value}: {e}")
    
    def get_network_config(self, chain: Chain, network: NetworkType) -> NetworkConfig:
        """
        Get network configuration for chain and network type.
        
        Args:
            chain: Blockchain chain
            network: Network type
            
        Returns:
            NetworkConfig for the specified chain and network
            
        Raises:
            WalletError: If configuration not found
        """
        if chain not in self._network_configs:
            raise WalletError(f"Unsupported chain: {chain.value}")
        
        chain_config = self._network_configs[chain]
        
        if network == NetworkType.MAINNET:
            return chain_config.mainnet
        elif network == NetworkType.TESTNET:
            return chain_config.testnet
        elif network == NetworkType.DEVNET and chain_config.devnet:
            return chain_config.devnet
        else:
            raise WalletError(f"Unsupported network {network.value} for chain {chain.value}")
    
    def create_wallet_config(
        self,
        chain: Chain,
        network: NetworkType = NetworkType.TESTNET,
        private_key: Optional[str] = None,
        mnemonic: Optional[str] = None,
        wallet_address: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> WalletConfig:
        """
        Create wallet configuration with network settings.
        
        Args:
            chain: Blockchain chain
            network: Network type
            private_key: Private key (will be stored securely if provided)
            mnemonic: Mnemonic phrase (will be stored securely if provided)
            wallet_address: Wallet address for read-only access
            api_key: API key for RPC services
            
        Returns:
            WalletConfig object
            
        Raises:
            WalletError: If configuration is invalid
        """
        # Get network configuration
        network_config = self.get_network_config(chain, network)
        
        # Store sensitive data securely if provided
        if private_key:
            self.store_private_key(chain, network, private_key)
            private_key = None  # Don't keep in memory
            
        if mnemonic:
            self.store_mnemonic(chain, network, mnemonic)
            mnemonic = None  # Don't keep in memory
        
        # Build RPC URL with API key if provided
        rpc_url = network_config.rpc_url
        if api_key and not rpc_url.endswith('/'):
            rpc_url += api_key
        
        return WalletConfig(
            chain=chain,
            network=network,
            private_key=private_key,
            mnemonic=mnemonic,
            wallet_address=wallet_address,
            rpc_url=rpc_url,
            api_key=api_key,
            gas_price_gwei=network_config.gas_price_gwei,
            max_gas_limit=network_config.max_gas_limit,
            timeout_seconds=network_config.timeout_seconds
        )
    
    def get_wallet_config_from_env(
        self,
        chain: Chain,
        network: NetworkType = NetworkType.TESTNET
    ) -> WalletConfig:
        """
        Create wallet configuration from environment variables.
        
        Args:
            chain: Blockchain chain
            network: Network type
            
        Returns:
            WalletConfig loaded from environment
        """
        chain_prefix = chain.value.upper()
        network_suffix = network.value.upper()
        
        # Try to get private key from environment or keyring
        private_key = os.getenv(f"{chain_prefix}_{network_suffix}_PRIVATE_KEY")
        if not private_key:
            private_key = self.get_private_key(chain, network)
        
        # Try to get mnemonic from environment or keyring
        mnemonic = os.getenv(f"{chain_prefix}_{network_suffix}_MNEMONIC")
        if not mnemonic:
            mnemonic = self.get_mnemonic(chain, network)
        
        # Get other configuration from environment
        wallet_address = os.getenv(f"{chain_prefix}_{network_suffix}_WALLET_ADDRESS")
        api_key = os.getenv(f"{chain_prefix}_API_KEY")
        
        return self.create_wallet_config(
            chain=chain,
            network=network,
            private_key=private_key,
            mnemonic=mnemonic,
            wallet_address=wallet_address,
            api_key=api_key
        )
    
    def get_supported_chains(self) -> List[Chain]:
        """Get list of supported chains."""
        return list(self._network_configs.keys())
    
    def get_chain_info(self, chain: Chain) -> Dict[str, Any]:
        """
        Get information about a supported chain.
        
        Args:
            chain: Blockchain chain
            
        Returns:
            Dictionary with chain information
        """
        if chain not in self._network_configs:
            raise WalletError(f"Unsupported chain: {chain.value}")
        
        chain_config = self._network_configs[chain]
        return {
            "name": chain.value,
            "native_symbol": chain_config.native_symbol,
            "decimals": chain_config.decimals,
            "networks": {
                "mainnet": {
                    "rpc_url": chain_config.mainnet.rpc_url,
                    "chain_id": getattr(chain_config.mainnet, 'chain_id', None),
                    "explorer": chain_config.mainnet.block_explorer_url
                },
                "testnet": {
                    "rpc_url": chain_config.testnet.rpc_url,
                    "chain_id": getattr(chain_config.testnet, 'chain_id', None),
                    "explorer": chain_config.testnet.block_explorer_url
                }
            }
        }
    
    def clear_stored_keys(self, chain: Chain, network: NetworkType) -> None:
        """
        Clear stored private keys and mnemonics for a chain/network.
        
        Args:
            chain: Blockchain chain
            network: Network type
        """
        private_key_id = f"{chain.value}_{network.value}_private_key"
        mnemonic_id = f"{chain.value}_{network.value}_mnemonic"
        
        try:
            keyring.delete_password("shyvr-rlte-wallets", private_key_id)
        except keyring.errors.PasswordDeleteError:
            pass  # Key didn't exist
            
        try:
            keyring.delete_password("shyvr-rlte-wallets", mnemonic_id)
        except keyring.errors.PasswordDeleteError:
            pass  # Key didn't exist
            
        logger.info(f"Cleared stored keys for {chain.value} {network.value}")