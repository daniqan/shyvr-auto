"""
DEX Integration Manager

This module contains DEX client initialization and management for live trading.
Extracted from live_mode.py for better maintainability.
"""

import asyncio
import os
from typing import Dict, List, Optional
import structlog

from src.utils.base import Chain
from src.dex.base import DEXBase, DEXConfig, DEXConnectionError
from src.dex import JupiterDEXClient, UniswapV3Client, HyperliquidDEXClient
from src.wallet.solana_wallet import SolanaWallet
from src.wallet.ethereum_wallet import EthereumWallet
from src.wallet.base import WalletConfig, NetworkType

logger = structlog.get_logger()


class DEXIntegrationManager:
    """Manages DEX client initialization and health monitoring for live trading."""
    
    def __init__(self):
        self.dex_clients: Dict[str, DEXBase] = {}
        self.dex_connection_errors: Dict[str, str] = {}
        self.dex_config_errors: List[str] = []
        self.dex_health_monitor: Optional[asyncio.Task] = None
        self.logger = logger.bind(component="DEXIntegrationManager")
    
    async def initialize_dex_clients(self) -> Dict[str, DEXBase]:
        """Initialize DEX clients for live trading with wallet integration and safety checks."""
        self.logger.info("Initializing DEX clients for live trading")
        
        # Reset client state
        self.dex_clients.clear()
        self.dex_connection_errors.clear()
        self.dex_config_errors.clear()
        
        # Validate required environment variables
        required_env_vars = {
            'SOLANA_PRIVATE_KEY': 'Solana private key for Jupiter DEX',
            'ETHEREUM_PRIVATE_KEY': 'Ethereum private key for Uniswap V3',
            'HYPERLIQUID_PRIVATE_KEY': 'Hyperliquid private key',
            'HYPERLIQUID_WALLET_ADDRESS': 'Hyperliquid wallet address',
            'INFURA_PROJECT_ID': 'Infura project ID for Ethereum connections'
        }
        
        for env_var, description in required_env_vars.items():
            if not os.getenv(env_var):
                error_msg = f"Missing required environment variable: {env_var} ({description})"
                self.dex_config_errors.append(error_msg)
                self.logger.error(error_msg)
        
        if self.dex_config_errors:
            raise ValueError(f"Missing required environment variables: {', '.join(required_env_vars.keys())}")
        
        # Initialize Jupiter DEX Client (Solana)
        await self._initialize_jupiter_client()
        
        # Initialize Uniswap V3 Client (Ethereum)
        await self._initialize_uniswap_client()
        
        # Initialize Hyperliquid Client
        await self._initialize_hyperliquid_client()
        
        # Check if we have enough working clients for safe trading
        working_clients = len(self.dex_clients)
        if working_clients == 0:
            raise RuntimeError("No DEX clients available - all connections failed")
        elif working_clients < 2:
            self.logger.warning(f"Only {working_clients} DEX client(s) available - reduced redundancy")
        
        # Start health monitoring for initialized clients
        if self.dex_clients:
            await self._start_dex_health_monitoring()
        
        self.logger.info(f"DEX client initialization complete - {working_clients} clients available")
        return self.dex_clients.copy()

    async def _initialize_jupiter_client(self) -> None:
        """Initialize Jupiter DEX client with Solana wallet integration."""
        try:
            # Create Solana wallet config
            solana_private_key = os.getenv('SOLANA_PRIVATE_KEY')
            solana_network = os.getenv('SOLANA_NETWORK', 'devnet')
            solana_rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.devnet.solana.com')
            
            wallet_config = WalletConfig(
                chain=Chain.SOLANA,
                network=NetworkType.DEVNET if solana_network == 'devnet' else NetworkType.MAINNET,
                private_key=solana_private_key,
                rpc_url=solana_rpc_url
            )
            
            solana_wallet = SolanaWallet(config=wallet_config)
            
            # Configure Jupiter client
            jupiter_config = DEXConfig(
                chain=Chain.SOLANA,
                name="jupiter",
                wallet_address=solana_wallet.wallet_address,
                max_slippage_bps=50,  # 0.5%
                timeout_seconds=30,
                rate_limit_per_second=10
            )
            
            # Initialize Jupiter client
            jupiter_client = JupiterDEXClient(config=jupiter_config)
            
            # Connect to Jupiter
            if await jupiter_client.connect():
                self.dex_clients['jupiter'] = jupiter_client
                self.logger.info("Jupiter DEX client initialized successfully")
            else:
                raise DEXConnectionError("Jupiter connection failed")
                
        except Exception as e:
            error_msg = f"Jupiter client initialization failed: {str(e)}"
            self.dex_connection_errors['jupiter'] = error_msg
            self.logger.error(error_msg)

    async def _initialize_uniswap_client(self) -> None:
        """Initialize Uniswap V3 client with Ethereum wallet integration."""
        try:
            # Create Ethereum wallet config
            ethereum_private_key = os.getenv('ETHEREUM_PRIVATE_KEY')
            infura_project_id = os.getenv('INFURA_PROJECT_ID')
            ethereum_network = os.getenv('ETHEREUM_NETWORK', 'mainnet')
            
            # Construct RPC URL with Infura
            rpc_url = f"https://{ethereum_network}.infura.io/v3/{infura_project_id}"
            
            wallet_config = WalletConfig(
                chain=Chain.ETHEREUM,
                network=NetworkType.MAINNET if ethereum_network == 'mainnet' else NetworkType.TESTNET,
                private_key=ethereum_private_key,
                rpc_url=rpc_url,
                api_key=infura_project_id
            )
            
            ethereum_wallet = EthereumWallet(config=wallet_config)
            
            # Configure Uniswap V3 client
            uniswap_config = DEXConfig(
                chain=Chain.ETHEREUM,
                name="uniswap_v3",
                wallet_address=ethereum_wallet.wallet_address,
                max_slippage_bps=50,  # 0.5%
                timeout_seconds=30,
                rate_limit_per_second=5  # More conservative for Ethereum
            )
            
            # Initialize Uniswap V3 client
            uniswap_client = UniswapV3Client(config=uniswap_config)
            
            # Connect to Uniswap V3
            if await uniswap_client.connect():
                self.dex_clients['uniswap_v3'] = uniswap_client
                self.logger.info("Uniswap V3 client initialized successfully")
            else:
                raise DEXConnectionError("Uniswap V3 connection failed")
                
        except Exception as e:
            error_msg = f"Uniswap V3 client initialization failed: {str(e)}"
            self.dex_connection_errors['uniswap_v3'] = error_msg
            self.logger.error(error_msg)

    async def _initialize_hyperliquid_client(self) -> None:
        """Initialize Hyperliquid client with wallet integration."""
        try:
            # Get Hyperliquid credentials
            hyperliquid_private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
            hyperliquid_wallet_address = os.getenv('HYPERLIQUID_WALLET_ADDRESS')
            hyperliquid_api_key = os.getenv('HYPERLIQUID_API_KEY', '')
            
            # Configure Hyperliquid client  
            hyperliquid_config = DEXConfig(
                chain=Chain.ARBITRUM,  # Hyperliquid runs on Arbitrum
                name="hyperliquid",
                api_key=hyperliquid_api_key,
                wallet_address=hyperliquid_wallet_address,
                max_slippage_bps=30,  # 0.3% - tighter for perps
                timeout_seconds=20,
                rate_limit_per_second=15
            )
            
            # Initialize Hyperliquid client
            hyperliquid_client = HyperliquidDEXClient(config=hyperliquid_config)
            
            # Connect to Hyperliquid
            if await hyperliquid_client.connect():
                self.dex_clients['hyperliquid'] = hyperliquid_client
                self.logger.info("Hyperliquid client initialized successfully")
            else:
                raise DEXConnectionError("Hyperliquid connection failed")
                
        except Exception as e:
            error_msg = f"Hyperliquid client initialization failed: {str(e)}"
            self.dex_connection_errors['hyperliquid'] = error_msg
            self.logger.error(error_msg)

    async def _start_dex_health_monitoring(self) -> None:
        """Start background health monitoring for DEX clients."""
        if self.dex_health_monitor:
            self.dex_health_monitor.cancel()
        
        self.dex_health_monitor = asyncio.create_task(self._dex_health_monitoring_loop())
        self.logger.info("DEX health monitoring started")

    async def _dex_health_monitoring_loop(self) -> None:
        """Background loop for monitoring DEX client health."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_dex_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"DEX health monitoring error: {e}")

    async def _check_dex_health(self) -> None:
        """Check health of all DEX clients."""
        failed_clients = []
        
        for dex_name, client in self.dex_clients.items():
            try:
                # Use is_connected property as a simple health check
                if not client.is_connected:
                    failed_clients.append(dex_name)
                    self.logger.warning(f"DEX client {dex_name} is not connected")
            except Exception as e:
                failed_clients.append(dex_name)
                self.logger.error(f"DEX client {dex_name} health check error: {e}")
        
        # Remove failed clients and attempt reconnection
        for dex_name in failed_clients:
            if dex_name in self.dex_clients:
                del self.dex_clients[dex_name]
                self.dex_connection_errors[dex_name] = "Health check failed - client removed"
        
        # Check if we still have enough working clients
        if len(self.dex_clients) == 0:
            self.logger.error("All DEX clients failed - no trading possible")

    async def stop_health_monitoring(self) -> None:
        """Stop DEX health monitoring."""
        if self.dex_health_monitor:
            self.dex_health_monitor.cancel()
            try:
                await self.dex_health_monitor
            except asyncio.CancelledError:
                pass
            self.dex_health_monitor = None
            self.logger.info("DEX health monitoring stopped")

    async def disconnect_all_clients(self) -> None:
        """Disconnect all DEX clients."""
        for dex_name, client in self.dex_clients.items():
            try:
                await client.disconnect()
                self.logger.info(f"Disconnected from {dex_name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from {dex_name}: {e}")
        
        self.dex_clients.clear()

    def get_client_status(self) -> Dict[str, Dict[str, str]]:
        """Get status of all DEX clients."""
        status = {}
        
        # Add connected clients
        for dex_name, client in self.dex_clients.items():
            status[dex_name] = {
                "status": "connected" if client.is_connected else "disconnected",
                "chain": client.config.chain.value,
                "error": None
            }
        
        # Add failed clients
        for dex_name, error_msg in self.dex_connection_errors.items():
            if dex_name not in status:
                status[dex_name] = {
                    "status": "failed",
                    "chain": "unknown",
                    "error": error_msg
                }
        
        return status

    def get_health_summary(self) -> Dict[str, any]:
        """Get overall health summary of DEX integration."""
        total_configured = len(self.dex_clients) + len(self.dex_connection_errors)
        connected = len([c for c in self.dex_clients.values() if c.is_connected])
        
        return {
            "total_configured": total_configured,
            "connected": connected,
            "failed": len(self.dex_connection_errors),
            "health_percentage": (connected / total_configured * 100) if total_configured > 0 else 0,
            "monitoring_active": self.dex_health_monitor is not None and not self.dex_health_monitor.done(),
            "config_errors": self.dex_config_errors
        }

    async def reconnect_failed_clients(self) -> Dict[str, bool]:
        """Attempt to reconnect failed DEX clients."""
        reconnection_results = {}
        
        for dex_name in list(self.dex_connection_errors.keys()):
            try:
                if dex_name == "jupiter":
                    await self._initialize_jupiter_client()
                elif dex_name == "uniswap_v3":
                    await self._initialize_uniswap_client()
                elif dex_name == "hyperliquid":
                    await self._initialize_hyperliquid_client()
                
                if dex_name in self.dex_clients:
                    del self.dex_connection_errors[dex_name]
                    reconnection_results[dex_name] = True
                    self.logger.info(f"Successfully reconnected to {dex_name}")
                else:
                    reconnection_results[dex_name] = False
                    
            except Exception as e:
                reconnection_results[dex_name] = False
                self.logger.error(f"Failed to reconnect to {dex_name}: {e}")
        
        return reconnection_results