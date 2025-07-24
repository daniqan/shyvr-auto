"""
Configuration Factory for DEX-Wallet Integration

This module provides factory methods to create properly configured
DEX clients, wallets, and integration bridges from the application
configuration file and environment variables.
"""

import os
from typing import Dict, Optional, Tuple, Any
from decimal import Decimal
import structlog

from src.utils.config import Config
from src.utils.base import Chain, NetworkType
from src.dex.base import DEXConfig
from src.dex.jupiter_client import JupiterDEXClient
from src.wallet.base import WalletConfig
from src.wallet.solana_wallet import SolanaWallet
from src.trading.dex_wallet_bridge import DEXWalletBridge, SwapExecutionConfig

logger = structlog.get_logger()


class ConfigurationError(Exception):
    """Error in configuration factory operations."""
    pass


class DEXWalletConfigFactory:
    """
    Factory class for creating configured DEX and wallet instances.
    
    This factory reads from the application configuration file and
    environment variables to create properly configured instances
    for DEX-wallet integration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration factory.
        
        Args:
            config_path: Path to configuration file (defaults to standard location)
        """
        self.config = Config(config_path)
        self.logger = logger.bind(factory="DEXWalletConfigFactory")
    
    def create_jupiter_client(self, chain: Chain = Chain.SOLANA) -> JupiterDEXClient:
        """
        Create configured Jupiter DEX client.
        
        Args:
            chain: Blockchain network (must be SOLANA for Jupiter)
            
        Returns:
            Configured JupiterDEXClient instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if chain != Chain.SOLANA:
            raise ConfigurationError(f"Jupiter DEX only supports Solana, got: {chain}")
        
        try:
            # Get Jupiter-specific configuration
            jupiter_config = self.config.get("dex.jupiter", {})
            
            if not jupiter_config.get("enabled", True):
                raise ConfigurationError("Jupiter DEX is disabled in configuration")
            
            # Create DEX configuration
            dex_config = DEXConfig(
                chain=Chain.SOLANA,
                name="jupiter",
                api_key=jupiter_config.get("api_key"),
                base_url=jupiter_config.get("base_url", "https://quote-api.jup.ag"),
                max_slippage_bps=jupiter_config.get("max_slippage_bps", 50),
                timeout_seconds=jupiter_config.get("timeout_seconds", 30),
                rate_limit_per_second=jupiter_config.get("rate_limit_per_second", 10),
                enable_price_impact_warnings=jupiter_config.get("enable_price_impact_warnings", True),
                max_price_impact_bps=jupiter_config.get("max_price_impact_bps", 1000)
            )
            
            client = JupiterDEXClient(dex_config)
            
            self.logger.info(
                "Created Jupiter DEX client",
                max_slippage_bps=dex_config.max_slippage_bps,
                timeout_seconds=dex_config.timeout_seconds
            )
            
            return client
            
        except Exception as e:
            self.logger.error("Failed to create Jupiter DEX client", error=str(e))
            raise ConfigurationError(f"Jupiter DEX configuration error: {e}")
    
    def create_solana_wallet(self) -> SolanaWallet:
        """
        Create configured Solana wallet.
        
        Returns:
            Configured SolanaWallet instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Get Solana wallet configuration
            solana_config = self.config.get("wallets.solana", {})
            
            if not solana_config.get("enabled", True):
                raise ConfigurationError("Solana wallet is disabled in configuration")
            
            # Parse network type
            network_str = solana_config.get("network", "devnet").lower()
            if network_str == "mainnet-beta" or network_str == "mainnet":
                network = NetworkType.MAINNET
            elif network_str == "devnet":
                network = NetworkType.DEVNET
            elif network_str == "testnet":
                network = NetworkType.TESTNET
            else:
                raise ConfigurationError(f"Invalid Solana network: {network_str}")
            
            # Get credentials from environment or config
            private_key = os.getenv("SOLANA_PRIVATE_KEY") or solana_config.get("private_key")
            wallet_address = os.getenv("SOLANA_WALLET_ADDRESS") or solana_config.get("wallet_address")
            rpc_url = os.getenv("SOLANA_RPC_URL") or solana_config.get("rpc_url", "https://api.devnet.solana.com")
            
            # Validate that we have either private key or wallet address
            if not private_key and not wallet_address:
                raise ConfigurationError(
                    "Either SOLANA_PRIVATE_KEY or SOLANA_WALLET_ADDRESS must be provided"
                )
            
            # Create wallet configuration
            wallet_config = WalletConfig(
                chain=Chain.SOLANA,
                network=network,
                rpc_url=rpc_url,
                private_key=private_key,
                wallet_address=wallet_address,
                timeout_seconds=solana_config.get("timeout_seconds", 30),
                max_retries=solana_config.get("max_retries", 3)
            )
            
            wallet = SolanaWallet(wallet_config)
            
            self.logger.info(
                "Created Solana wallet",
                network=network.value,
                read_only=private_key is None,
                rpc_url=rpc_url
            )
            
            return wallet
            
        except Exception as e:
            self.logger.error("Failed to create Solana wallet", error=str(e))
            raise ConfigurationError(f"Solana wallet configuration error: {e}")
    
    def create_dex_wallet_bridge(
        self,
        chain: Chain = Chain.SOLANA
    ) -> Tuple[DEXWalletBridge, Any, Any]:
        """
        Create configured DEX-wallet bridge with appropriate clients.
        
        Args:
            chain: Blockchain network to create bridge for
            
        Returns:
            Tuple of (bridge, dex_client, wallet)
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            if chain == Chain.SOLANA:
                # Create Jupiter client and Solana wallet
                dex_client = self.create_jupiter_client(chain)
                wallet = self.create_solana_wallet()
            else:
                raise ConfigurationError(f"Unsupported chain for DEX-wallet bridge: {chain}")
            
            # Get integration configuration
            integration_config = self.create_swap_execution_config()
            
            # Create bridge
            bridge = DEXWalletBridge(dex_client, wallet, integration_config)
            
            self.logger.info(
                "Created DEX-wallet bridge",
                chain=chain.value,
                dex=dex_client.__class__.__name__,
                wallet=wallet.__class__.__name__
            )
            
            return bridge, dex_client, wallet
            
        except Exception as e:
            self.logger.error("Failed to create DEX-wallet bridge", chain=chain.value, error=str(e))
            raise ConfigurationError(f"Bridge configuration error: {e}")
    
    def create_swap_execution_config(self) -> SwapExecutionConfig:
        """
        Create swap execution configuration from config file.
        
        Returns:
            Configured SwapExecutionConfig instance
        """
        try:
            # Get integration configuration
            swap_config = self.config.get("integration.swap_execution", {})
            global_wallet_config = self.config.get("wallets.global", {})
            
            config = SwapExecutionConfig(
                max_retries=swap_config.get("max_retries", 3),
                retry_delay_seconds=float(swap_config.get("retry_delay_seconds", 1.0)),
                enable_balance_checks=swap_config.get("enable_balance_checks", True),
                enable_account_preparation=swap_config.get("enable_account_preparation", True),
                confirmation_timeout_seconds=swap_config.get("confirmation_timeout_seconds", 60),
                enable_slippage_protection=swap_config.get("enable_slippage_protection", True)
            )
            
            self.logger.debug(
                "Created swap execution config",
                max_retries=config.max_retries,
                enable_balance_checks=config.enable_balance_checks
            )
            
            return config
            
        except Exception as e:
            self.logger.error("Failed to create swap execution config", error=str(e))
            # Return default configuration if config parsing fails
            return SwapExecutionConfig()
    
    def get_supported_chains(self) -> list[Chain]:
        """
        Get list of supported chains based on configuration.
        
        Returns:
            List of supported Chain enums
        """
        supported_chains = []
        
        try:
            # Check DEX configurations
            if self.config.get("dex.jupiter.enabled", True):
                supported_chains.append(Chain.SOLANA)
            
            if self.config.get("dex.hyperliquid.enabled", False):
                supported_chains.append(Chain.ETHEREUM)
            
            if self.config.get("dex.uniswap_v3.enabled", False):
                supported_chains.extend([Chain.ETHEREUM, Chain.BASE])
            
            # Filter by wallet availability
            enabled_chains = []
            for chain in supported_chains:
                if chain == Chain.SOLANA and self.config.get("wallets.solana.enabled", True):
                    enabled_chains.append(chain)
                elif chain == Chain.ETHEREUM and self.config.get("wallets.ethereum.enabled", False):
                    enabled_chains.append(chain)
                elif chain == Chain.BASE and self.config.get("wallets.base.enabled", False):
                    enabled_chains.append(chain)
            
            return list(set(enabled_chains))  # Remove duplicates
            
        except Exception as e:
            self.logger.error("Failed to get supported chains", error=str(e))
            return [Chain.SOLANA]  # Default fallback
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration for DEX-wallet integration.
        
        Returns:
            Dictionary with validation results and any issues found
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "supported_chains": []
        }
        
        try:
            supported_chains = self.get_supported_chains()
            validation_result["supported_chains"] = [chain.value for chain in supported_chains]
            
            if not supported_chains:
                validation_result["valid"] = False
                validation_result["errors"].append("No supported chains configured")
            
            # Validate Solana configuration if enabled
            if Chain.SOLANA in supported_chains:
                solana_validation = self._validate_solana_config()
                validation_result["errors"].extend(solana_validation["errors"])
                validation_result["warnings"].extend(solana_validation["warnings"])
                if not solana_validation["valid"]:
                    validation_result["valid"] = False
            
            # Check for required environment variables
            env_checks = self._validate_environment_variables()
            validation_result["errors"].extend(env_checks["errors"])
            validation_result["warnings"].extend(env_checks["warnings"])
            if not env_checks["valid"]:
                validation_result["valid"] = False
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Configuration validation failed: {e}")
        
        return validation_result
    
    def _validate_solana_config(self) -> Dict[str, Any]:
        """Validate Solana-specific configuration."""
        result = {"valid": True, "errors": [], "warnings": []}
        
        try:
            solana_config = self.config.get("wallets.solana", {})
            
            # Check RPC URL
            rpc_url = os.getenv("SOLANA_RPC_URL") or solana_config.get("rpc_url")
            if not rpc_url:
                result["errors"].append("SOLANA_RPC_URL not configured")
                result["valid"] = False
            
            # Check credentials
            private_key = os.getenv("SOLANA_PRIVATE_KEY") or solana_config.get("private_key")
            wallet_address = os.getenv("SOLANA_WALLET_ADDRESS") or solana_config.get("wallet_address")
            
            if not private_key and not wallet_address:
                result["errors"].append("Either SOLANA_PRIVATE_KEY or SOLANA_WALLET_ADDRESS required")
                result["valid"] = False
            elif not private_key:
                result["warnings"].append("Running in read-only mode (no private key)")
            
            # Check network configuration
            network = solana_config.get("network", "devnet")
            if network not in ["mainnet-beta", "mainnet", "devnet", "testnet"]:
                result["errors"].append(f"Invalid Solana network: {network}")
                result["valid"] = False
            
        except Exception as e:
            result["errors"].append(f"Solana config validation error: {e}")
            result["valid"] = False
        
        return result
    
    def _validate_environment_variables(self) -> Dict[str, Any]:
        """Validate required environment variables."""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Optional but recommended environment variables
        recommended_vars = [
            "SOLANA_PRIVATE_KEY",
            "SOLANA_RPC_URL"
        ]
        
        for var in recommended_vars:
            if not os.getenv(var):
                result["warnings"].append(f"Environment variable {var} not set")
        
        return result
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        try:
            supported_chains = self.get_supported_chains()
            validation = self.validate_configuration()
            
            return {
                "supported_chains": [chain.value for chain in supported_chains],
                "enabled_dexs": self._get_enabled_dexs(),
                "enabled_wallets": self._get_enabled_wallets(),
                "validation": validation,
                "integration_settings": {
                    "swap_execution": self.config.get("integration.swap_execution", {}),
                    "monitoring": self.config.get("integration.monitoring", {})
                }
            }
            
        except Exception as e:
            self.logger.error("Failed to get configuration summary", error=str(e))
            return {"error": str(e)}
    
    def _get_enabled_dexs(self) -> list[str]:
        """Get list of enabled DEX names."""
        enabled_dexs = []
        
        if self.config.get("dex.jupiter.enabled", True):
            enabled_dexs.append("jupiter")
        if self.config.get("dex.hyperliquid.enabled", False):
            enabled_dexs.append("hyperliquid")
        if self.config.get("dex.uniswap_v3.enabled", False):
            enabled_dexs.append("uniswap_v3")
        
        return enabled_dexs
    
    def _get_enabled_wallets(self) -> list[str]:
        """Get list of enabled wallet chains."""
        enabled_wallets = []
        
        if self.config.get("wallets.solana.enabled", True):
            enabled_wallets.append("solana")
        if self.config.get("wallets.ethereum.enabled", False):
            enabled_wallets.append("ethereum")
        if self.config.get("wallets.base.enabled", False):
            enabled_wallets.append("base")
        
        return enabled_wallets