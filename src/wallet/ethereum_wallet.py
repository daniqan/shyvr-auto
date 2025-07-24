"""
Ethereum wallet integration using Web3.py for Ethereum and EVM-compatible chains.

This module provides wallet functionality for Ethereum, Base, and other EVM chains
including balance queries, transaction signing, and contract interactions.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from web3 import Web3, AsyncWeb3
from web3.exceptions import Web3Exception, TransactionNotFound, BlockNotFound
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_utils import to_checksum_address, is_address
import aiohttp

from .base import (
    WalletBase, 
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

logger = logging.getLogger(__name__)


@dataclass
class ERC20Token:
    """ERC-20 token information."""
    address: str
    symbol: str
    decimals: int
    name: str = ""


class EthereumWallet(WalletBase):
    """
    Ethereum wallet implementation using Web3.py.
    
    Supports Ethereum mainnet, testnets, and EVM-compatible chains like Base.
    Provides functionality for ETH and ERC-20 token operations.
    """
    
    # Standard ERC-20 ABI for token operations
    ERC20_ABI = [
        {
            "constant": True,
            "inputs": [{"name": "_owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "balance", "type": "uint256"}],
            "type": "function"
        },
        {
            "constant": False,
            "inputs": [
                {"name": "_to", "type": "address"},
                {"name": "_value", "type": "uint256"}
            ],
            "name": "transfer",
            "outputs": [{"name": "", "type": "bool"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [],
            "name": "decimals",
            "outputs": [{"name": "", "type": "uint8"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [],
            "name": "symbol",
            "outputs": [{"name": "", "type": "string"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [],
            "name": "name",
            "outputs": [{"name": "", "type": "string"}],
            "type": "function"
        }
    ]
    
    def __init__(self, config: WalletConfig):
        """
        Initialize Ethereum wallet.
        
        Args:
            config: Wallet configuration
            
        Raises:
            WalletError: If configuration is invalid for Ethereum
        """
        super().__init__(config)
        
        if config.chain not in [Chain.ETHEREUM, Chain.BASE]:
            raise WalletError(f"EthereumWallet does not support chain: {config.chain}")
        
        self.w3: Optional[AsyncWeb3] = None
        self.account: Optional[LocalAccount] = None
        self._token_cache: Dict[str, ERC20Token] = {}
        
    async def connect(self) -> bool:
        """
        Connect to Ethereum network and initialize wallet.
        
        Returns:
            True if connection successful
            
        Raises:
            WalletConnectionError: If connection fails
        """
        try:
            # Initialize Web3 connection
            if not self.config.rpc_url:
                raise WalletConnectionError("RPC URL not configured")
            
            # Create async Web3 instance
            self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(
                self.config.rpc_url,
                request_kwargs={'timeout': self.config.timeout_seconds}
            ))
            
            # Note: PoA middleware may be needed for some testnets
            # For now, using standard middleware setup
            
            # Test connection
            if not await self.w3.is_connected():
                raise WalletConnectionError("Failed to connect to Ethereum network")
            
            # Initialize account or set read-only mode
            if self.config.wallet_address:
                # Read-only mode (prioritize this over private key lookup)
                if not is_address(self.config.wallet_address):
                    raise WalletConnectionError("Invalid wallet address")
                self._wallet_address = to_checksum_address(self.config.wallet_address)
                logger.info(f"Read-only mode for address: {self._wallet_address}")
            else:
                # Try to get private key for full wallet functionality
                from .config import WalletConfigManager
                config_manager = WalletConfigManager()
                
                private_key = self.config.private_key
                if not private_key:
                    private_key = config_manager.get_private_key(self.config.chain, self.config.network)
                
                if private_key:
                    try:
                        self.account = Account.from_key(private_key)
                        self._wallet_address = self.account.address
                        logger.info(f"Loaded account: {self.account.address}")
                    except Exception as e:
                        raise WalletConnectionError(f"Invalid private key: {e}")
                else:
                    raise WalletConnectionError("No private key or wallet address provided")
            
            self._connected = True
            
            # Validate network chain ID
            chain_id = self.w3.eth.chain_id
            expected_chain_ids = self._get_expected_chain_ids()
            
            if expected_chain_ids and chain_id not in expected_chain_ids:
                expected_list = ", ".join(map(str, expected_chain_ids))
                raise WalletConnectionError(
                    f"Network mismatch: expected chain ID {expected_list} for {self.config.chain.value} "
                    f"{self.config.network.value}, but got {chain_id}"
                )
            
            # Log connection info
            block_number = await self.w3.eth.get_block_number()
            logger.info(f"Connected to {self.config.chain.value} (Chain ID: {chain_id}, Block: {block_number})")
            
            return True
            
        except asyncio.TimeoutError as e:
            logger.error(f"Ethereum connection timeout: {e}")
            raise WalletConnectionError(f"Connection timeout: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle specific RPC errors
            if "429" in error_msg or "too many requests" in error_msg:
                raise WalletConnectionError(f"Rate limit exceeded: {e}")
            elif "503" in error_msg or "service unavailable" in error_msg:
                raise WalletConnectionError(f"Service unavailable: {e}")
            elif "401" in error_msg or "unauthorized" in error_msg:
                raise WalletConnectionError(f"Invalid API key or unauthorized: {e}")
            elif "timeout" in error_msg:
                raise WalletConnectionError(f"Connection timeout: {e}")
            else:
                logger.error(f"Failed to connect to Ethereum wallet: {e}")
                raise WalletConnectionError(f"Connection failed: {e}")
    
    def _get_expected_chain_ids(self) -> List[int]:
        """Get expected chain IDs for the current chain and network configuration."""
        chain_id_map = {
            (Chain.ETHEREUM, NetworkType.MAINNET): [1],
            (Chain.ETHEREUM, NetworkType.TESTNET): [11155111],  # Sepolia
            (Chain.BASE, NetworkType.MAINNET): [8453],
            (Chain.BASE, NetworkType.TESTNET): [84532],  # Base Sepolia
        }
        
        return chain_id_map.get((self.config.chain, self.config.network), [])
    
    async def disconnect(self) -> None:
        """Disconnect from Ethereum network."""
        self._connected = False
        self.w3 = None
        self.account = None
        self._wallet_address = None
        logger.info("Disconnected from Ethereum wallet")
    
    async def get_balance(self, token_address: Optional[str] = None) -> WalletBalance:
        """
        Get wallet balance for ETH and ERC-20 tokens.
        
        Args:
            token_address: ERC-20 token address, None for ETH balance
            
        Returns:
            WalletBalance with native and token balances
        """
        if not self.is_connected or not self.w3 or not self.wallet_address:
            raise WalletError("Wallet not connected")
        
        try:
            # Get ETH balance
            eth_balance_wei = await self.w3.eth.get_balance(self.wallet_address)
            eth_balance = Decimal(eth_balance_wei) / Decimal(10**18)
            
            token_balances = {}
            
            if token_address:
                # Get specific token balance
                token_balance = await self.get_token_balance(token_address)
                token_info = await self._get_token_info(token_address)
                token_balances[token_address] = token_balance
            
            native_symbol = "ETH" if self.config.chain == Chain.ETHEREUM else "ETH"
            
            return WalletBalance(
                native_balance=eth_balance,
                native_symbol=native_symbol,
                token_balances=token_balances
            )
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise WalletError(f"Balance query failed: {e}")
    
    async def get_native_balance(self) -> Decimal:
        """
        Get ETH balance.
        
        Returns:
            ETH balance as Decimal
        """
        if not self.is_connected or not self.w3 or not self.wallet_address:
            raise WalletError("Wallet not connected")
        
        try:
            balance_wei = await self.w3.eth.get_balance(self.wallet_address)
            return Decimal(balance_wei) / Decimal(10**18)
        except Exception as e:
            logger.error(f"Failed to get ETH balance: {e}")
            raise WalletError(f"ETH balance query failed: {e}")
    
    async def get_token_balance(self, token_address: str) -> Decimal:
        """
        Get ERC-20 token balance.
        
        Args:
            token_address: Token contract address
            
        Returns:
            Token balance as Decimal
        """
        if not self.is_connected or not self.w3 or not self.wallet_address:
            raise WalletError("Wallet not connected")
        
        try:
            token_address = to_checksum_address(token_address)
            contract = self.w3.eth.contract(address=token_address, abi=self.ERC20_ABI)
            
            # Get token balance
            balance = await contract.functions.balanceOf(self.wallet_address).call()
            
            # Get token decimals
            token_info = await self._get_token_info(token_address)
            decimals = token_info.decimals
            
            return Decimal(balance) / Decimal(10**decimals)
            
        except Exception as e:
            logger.error(f"Failed to get token balance for {token_address}: {e}")
            raise WalletError(f"Token balance query failed: {e}")
    
    async def send_native_token(
        self,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[Decimal] = None
    ) -> TransactionResult:
        """
        Send ETH to another address.
        
        Args:
            to_address: Recipient address
            amount: Amount of ETH to send
            gas_price: Optional gas price in gwei
            
        Returns:
            TransactionResult with transaction details
        """
        if not self.account:
            raise WalletTransactionError("Cannot send transactions in read-only mode")
        
        if not self.is_connected or not self.w3:
            raise WalletTransactionError("Wallet not connected")
        
        try:
            to_address = to_checksum_address(to_address)
            amount_wei = int(amount * Decimal(10**18))
            
            # Get current gas price if not provided
            if gas_price is None:
                gas_price_wei = await self.w3.eth.gas_price
            else:
                gas_price_wei = int(gas_price * Decimal(10**9))  # Convert gwei to wei
            
            # Get nonce
            nonce = await self.w3.eth.get_transaction_count(self.account.address)
            
            # Estimate gas
            gas_limit = await self.estimate_gas(to_address, amount)
            
            # Build transaction
            transaction = {
                'to': to_address,
                'value': amount_wei,
                'gas': gas_limit,
                'gasPrice': gas_price_wei,
                'nonce': nonce,
                'chainId': await self.w3.eth.chain_id
            }
            
            # Sign and send transaction
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = await self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"Sent {amount} ETH to {to_address}, tx: {tx_hash.hex()}")
            
            return TransactionResult(
                transaction_hash=tx_hash.hex(),
                status=TransactionStatus.PENDING,
                gas_used=None,  # Will be set when confirmed
                gas_price=Decimal(gas_price_wei) / Decimal(10**9)
            )
            
        except Exception as e:
            logger.error(f"Failed to send ETH: {e}")
            raise WalletTransactionError(f"ETH transfer failed: {e}")
    
    async def send_token(
        self,
        token_address: str,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[Decimal] = None
    ) -> TransactionResult:
        """
        Send ERC-20 tokens to another address.
        
        Args:
            token_address: Token contract address
            to_address: Recipient address
            amount: Amount of tokens to send
            gas_price: Optional gas price in gwei
            
        Returns:
            TransactionResult with transaction details
        """
        if not self.account:
            raise WalletTransactionError("Cannot send transactions in read-only mode")
        
        if not self.is_connected or not self.w3:
            raise WalletTransactionError("Wallet not connected")
        
        try:
            token_address = to_checksum_address(token_address)
            to_address = to_checksum_address(to_address)
            
            # Get token info
            token_info = await self._get_token_info(token_address)
            amount_units = int(amount * Decimal(10**token_info.decimals))
            
            # Create contract instance
            contract = self.w3.eth.contract(address=token_address, abi=self.ERC20_ABI)
            
            # Get current gas price if not provided
            if gas_price is None:
                gas_price_wei = await self.w3.eth.gas_price
            else:
                gas_price_wei = int(gas_price * Decimal(10**9))
            
            # Get nonce
            nonce = await self.w3.eth.get_transaction_count(self.account.address)
            
            # Build transaction
            transaction = await contract.functions.transfer(
                to_address, amount_units
            ).build_transaction({
                'chainId': await self.w3.eth.chain_id,
                'gas': 100000,  # Will be estimated
                'gasPrice': gas_price_wei,
                'nonce': nonce
            })
            
            # Estimate gas
            transaction['gas'] = await self.w3.eth.estimate_gas(transaction)
            
            # Sign and send transaction
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = await self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"Sent {amount} {token_info.symbol} to {to_address}, tx: {tx_hash.hex()}")
            
            return TransactionResult(
                transaction_hash=tx_hash.hex(),
                status=TransactionStatus.PENDING,
                gas_used=None,  # Will be set when confirmed
                gas_price=Decimal(gas_price_wei) / Decimal(10**9)
            )
            
        except Exception as e:
            logger.error(f"Failed to send token: {e}")
            raise WalletTransactionError(f"Token transfer failed: {e}")
    
    async def estimate_gas(
        self,
        to_address: str,
        amount: Decimal,
        token_address: Optional[str] = None,
        data: Optional[bytes] = None
    ) -> int:
        """
        Estimate gas for transaction.
        
        Args:
            to_address: Recipient address
            amount: Transaction amount
            token_address: Token address for ERC-20 transfers
            data: Optional transaction data
            
        Returns:
            Estimated gas units
        """
        if not self.is_connected or not self.w3 or not self.wallet_address:
            raise WalletError("Wallet not connected")
        
        try:
            to_address = to_checksum_address(to_address)
            
            if token_address:
                # Estimate gas for token transfer
                token_address = to_checksum_address(token_address)
                token_info = await self._get_token_info(token_address)
                amount_units = int(amount * Decimal(10**token_info.decimals))
                
                contract = self.w3.eth.contract(address=token_address, abi=self.ERC20_ABI)
                gas_estimate = await contract.functions.transfer(
                    to_address, amount_units
                ).estimate_gas({'from': self.wallet_address})
                
            else:
                # Estimate gas for ETH transfer
                amount_wei = int(amount * Decimal(10**18))
                transaction = {
                    'from': self.wallet_address,
                    'to': to_address,
                    'value': amount_wei
                }
                if data:
                    transaction['data'] = data
                    
                gas_estimate = await self.w3.eth.estimate_gas(transaction)
            
            # Add 10% buffer for safety
            return int(gas_estimate * 1.1)
            
        except Exception as e:
            logger.error(f"Failed to estimate gas: {e}")
            raise WalletError(f"Gas estimation failed: {e}")
    
    async def get_gas_price(self) -> Decimal:
        """
        Get current gas price in gwei.
        
        Returns:
            Current gas price in gwei
        """
        if not self.is_connected or not self.w3:
            raise WalletError("Wallet not connected")
        
        try:
            gas_price_wei = await self.w3.eth.gas_price
            return Decimal(gas_price_wei) / Decimal(10**9)  # Convert wei to gwei
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            raise WalletError(f"Gas price query failed: {e}")
    
    async def get_transaction_status(self, transaction_hash: str) -> TransactionResult:
        """
        Get transaction status and details.
        
        Args:
            transaction_hash: Transaction hash to query
            
        Returns:
            TransactionResult with current status
        """
        if not self.is_connected or not self.w3:
            raise WalletError("Wallet not connected")
        
        try:
            # Get transaction receipt
            try:
                receipt = await self.w3.eth.get_transaction_receipt(transaction_hash)
                
                # Transaction is confirmed
                status = TransactionStatus.CONFIRMED if receipt.status == 1 else TransactionStatus.FAILED
                
                return TransactionResult(
                    transaction_hash=transaction_hash,
                    status=status,
                    gas_used=receipt.gasUsed,
                    gas_price=Decimal(receipt.effectiveGasPrice) / Decimal(10**9) if receipt.effectiveGasPrice else None,
                    block_number=receipt.blockNumber,
                    timestamp=None  # Would need to fetch block for timestamp
                )
                
            except TransactionNotFound:
                # Check if transaction exists in mempool
                try:
                    await self.w3.eth.get_transaction(transaction_hash)  
                    return TransactionResult(
                        transaction_hash=transaction_hash,
                        status=TransactionStatus.PENDING
                    )
                except TransactionNotFound:
                    return TransactionResult(
                        transaction_hash=transaction_hash,
                        status=TransactionStatus.FAILED,
                        error_message="Transaction not found"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            raise WalletError(f"Transaction status query failed: {e}")
    
    async def sign_message(self, message: str) -> str:
        """
        Sign message with wallet private key.
        
        Args:
            message: Message to sign
            
        Returns:
            Signed message as hex string
        """
        if not self.account:
            raise WalletError("Cannot sign messages in read-only mode")
        
        try:
            # Sign message
            signed_message = self.account.sign_message_hash(
                Web3.keccak(text=message)
            )
            return signed_message.signature.hex()
            
        except Exception as e:
            logger.error(f"Failed to sign message: {e}")
            raise WalletError(f"Message signing failed: {e}")
    
    async def validate_address(self, address: str) -> bool:
        """
        Validate Ethereum address format.
        
        Args:
            address: Address to validate
            
        Returns:
            True if address is valid
        """
        return is_address(address)
    
    async def _get_token_info(self, token_address: str) -> ERC20Token:
        """
        Get ERC-20 token information.
        
        Args:
            token_address: Token contract address
            
        Returns:
            ERC20Token with token details
        """
        token_address = to_checksum_address(token_address)
        
        # Check cache first
        if token_address in self._token_cache:
            return self._token_cache[token_address]
        
        if not self.is_connected or not self.w3:
            raise WalletError("Wallet not connected")
        
        try:
            contract = self.w3.eth.contract(address=token_address, abi=self.ERC20_ABI)
            
            # Get token info
            symbol = await contract.functions.symbol().call()
            decimals = await contract.functions.decimals().call()
            name = ""
            
            try:
                name = await contract.functions.name().call()
            except:
                pass  # Some tokens don't have name
            
            token_info = ERC20Token(
                address=token_address,
                symbol=symbol,
                decimals=decimals,
                name=name
            )
            
            # Cache token info
            self._token_cache[token_address] = token_info
            
            return token_info
            
        except Exception as e:
            logger.error(f"Failed to get token info for {token_address}: {e}")
            raise WalletError(f"Token info query failed: {e}")
    
    async def get_supported_tokens(self) -> List[str]:
        """
        Get list of commonly used token addresses for this network.
        
        Returns:
            List of token contract addresses
        """
        # Common tokens by network - these could be loaded from config
        if self.config.chain == Chain.ETHEREUM:
            if self.config.network == NetworkType.MAINNET:
                return [
                    "0xA0b86a33E6441CcE67d6ea49e8F7a0D73c37A8c4",  # USDC
                    "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
                ]
            else:  # Testnet
                return [
                    "0x07865c6E87B9F70255377e024ace6630C1Eaa37F",  # USDC (Goerli)
                ]
        elif self.config.chain == Chain.BASE:
            if self.config.network == NetworkType.MAINNET:
                return [
                    "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC
                    "0x4200000000000000000000000000000000000006",  # WETH
                ]
        
        return []
    
    async def get_transaction_history(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get transaction history for wallet address.
        
        Note: This requires integration with block explorer APIs
        as Web3 doesn't provide transaction history directly.
        
        Args:
            limit: Maximum number of transactions
            offset: Number of transactions to skip
            
        Returns:
            List of transaction dictionaries
        """
        if not self.wallet_address:
            raise WalletError("Wallet not connected")
        
        # This would typically integrate with Etherscan API or similar
        # For now, return empty list as this requires external API keys
        logger.warning("Transaction history requires block explorer API integration")
        return []