"""
Solana wallet integration using solana-py for SOL and SPL token operations.

This module provides wallet functionality for Solana blockchain including
balance queries, transaction signing, and SPL token interactions.
"""

import asyncio
import base58
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from solana.rpc.async_api import AsyncClient
from solana.rpc.core import RPCException
from solana.rpc.types import TxOpts
from solders.transaction import Transaction
from solders.system_program import transfer, TransferParams
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.signature import Signature
from spl.token.instructions import transfer_checked, TransferCheckedParams
from spl.token.client import Token
import json

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
class SPLToken:
    """SPL token information."""
    mint_address: str
    symbol: str
    decimals: int
    name: str = ""
    
    
class SolanaWallet(WalletBase):
    """
    Solana wallet implementation using solana-py.
    
    Provides functionality for SOL and SPL token operations including
    balance queries, transfers, and token interactions.
    """
    
    def __init__(self, config: WalletConfig):
        """
        Initialize Solana wallet.
        
        Args:
            config: Wallet configuration
            
        Raises:
            WalletError: If configuration is invalid for Solana
        """
        super().__init__(config)
        
        if config.chain != Chain.SOLANA:
            raise WalletError(f"SolanaWallet only supports Solana chain, got: {config.chain}")
        
        self.client: Optional[AsyncClient] = None
        self.keypair: Optional[Keypair] = None
        self._token_cache: Dict[str, SPLToken] = {}
        
        # Solana uses lamports (1 SOL = 1e9 lamports)
        self.LAMPORTS_PER_SOL = 1_000_000_000
        
    async def connect(self) -> bool:
        """
        Connect to Solana network and initialize wallet.
        
        Returns:
            True if connection successful
            
        Raises:
            WalletConnectionError: If connection fails
        """
        try:
            # Initialize Solana RPC client
            if not self.config.rpc_url:
                raise WalletConnectionError("RPC URL not configured")
            
            self.client = AsyncClient(
                self.config.rpc_url,
                timeout=self.config.timeout_seconds
            )
            
            # Test connection
            try:
                health = await self.client.get_health()
                logger.info(f"Solana RPC health: {health}")
            except Exception as e:
                raise WalletConnectionError(f"Failed to connect to Solana RPC: {e}")
            
            # Initialize keypair if private key provided
            from .config import WalletConfigManager
            config_manager = WalletConfigManager()
            
            private_key = self.config.private_key
            if not private_key:
                private_key = config_manager.get_private_key(self.config.chain, self.config.network)
            
            if private_key:
                try:
                    # Solana private keys can be base58 encoded or raw bytes
                    if len(private_key) == 88:  # Base58 encoded
                        private_key_bytes = base58.b58decode(private_key)
                    else:  # Assume hex
                        private_key_bytes = bytes.fromhex(private_key.replace('0x', ''))
                    
                    self.keypair = Keypair.from_bytes(private_key_bytes)
                    self._wallet_address = str(self.keypair.pubkey())
                    logger.info(f"Loaded Solana keypair: {self._wallet_address}")
                    
                except Exception as e:
                    raise WalletConnectionError(f"Invalid Solana private key: {e}")
                    
            elif self.config.wallet_address:
                # Read-only mode
                try:
                    Pubkey.from_string(self.config.wallet_address)  # Validate address
                    self._wallet_address = self.config.wallet_address
                    logger.info(f"Read-only mode for Solana address: {self._wallet_address}")
                except Exception as e:
                    raise WalletConnectionError(f"Invalid Solana address: {e}")
            else:
                raise WalletConnectionError("No private key or wallet address provided")
            
            self._connected = True
            
            # Log connection info
            try:
                slot = await self.client.get_slot()
                logger.info(f"Connected to Solana {self.config.network.value} (Slot: {slot})")
            except Exception as e:
                logger.warning(f"Could not get slot info: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Solana wallet: {e}")
            raise WalletConnectionError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Solana network."""
        if self.client:
            await self.client.close()
        
        self._connected = False
        self.client = None
        self.keypair = None
        self._wallet_address = None
        logger.info("Disconnected from Solana wallet")
    
    async def get_balance(self, token_address: Optional[str] = None) -> WalletBalance:
        """
        Get wallet balance for SOL and SPL tokens.
        
        Args:
            token_address: SPL token mint address, None for SOL balance
            
        Returns:
            WalletBalance with native and token balances
        """
        if not self.is_connected or not self.client or not self.wallet_address:
            raise WalletError("Wallet not connected")
        
        try:
            # Get SOL balance
            sol_balance = await self.get_native_balance()
            
            token_balances = {}
            
            if token_address:
                # Get specific token balance
                token_balance = await self.get_token_balance(token_address)
                token_balances[token_address] = token_balance
            
            return WalletBalance(
                native_balance=sol_balance,
                native_symbol="SOL",
                token_balances=token_balances
            )
            
        except Exception as e:
            logger.error(f"Failed to get Solana balance: {e}")
            raise WalletError(f"Balance query failed: {e}")
    
    async def get_native_balance(self) -> Decimal:
        """
        Get SOL balance.
        
        Returns:
            SOL balance as Decimal
        """
        if not self.is_connected or not self.client or not self.wallet_address:
            raise WalletError("Wallet not connected")
        
        try:
            pubkey = Pubkey.from_string(self.wallet_address)
            response = await self.client.get_balance(pubkey)
            
            if response.value is None:
                raise WalletError("Failed to get SOL balance")
                
            balance_lamports = response.value
            return Decimal(balance_lamports) / Decimal(self.LAMPORTS_PER_SOL)
            
        except Exception as e:
            logger.error(f"Failed to get SOL balance: {e}")
            raise WalletError(f"SOL balance query failed: {e}")
    
    async def get_token_balance(self, token_address: str) -> Decimal:
        """
        Get SPL token balance.
        
        Args:
            token_address: SPL token mint address
            
        Returns:
            Token balance as Decimal
        """
        if not self.is_connected or not self.client or not self.wallet_address:
            raise WalletError("Wallet not connected")
        
        try:
            from spl.token.client import Token as SPLToken
            from spl.token.core import _TokenCore
            
            # Get token mint
            mint_pubkey = Pubkey.from_string(token_address)
            owner_pubkey = Pubkey.from_string(self.wallet_address)
            
            # Find associated token account
            token_accounts = await self.client.get_token_accounts_by_owner(
                owner_pubkey,
                {"mint": mint_pubkey}
            )
            
            if not token_accounts.value:
                return Decimal(0)  # No token account exists
            
            # Get balance from first token account
            token_account = token_accounts.value[0]
            balance_info = token_account.account.data.parsed['info']
            
            token_amount = balance_info['tokenAmount']
            return Decimal(token_amount['uiAmount'] or 0)
            
        except Exception as e:
            logger.error(f"Failed to get SPL token balance for {token_address}: {e}")
            raise WalletError(f"Token balance query failed: {e}")
    
    async def send_native_token(
        self,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[Decimal] = None
    ) -> TransactionResult:
        """
        Send SOL to another address.
        
        Args:
            to_address: Recipient address
            amount: Amount of SOL to send
            gas_price: Not used for Solana (fees are fixed)
            
        Returns:
            TransactionResult with transaction details
        """
        if not self.keypair:
            raise WalletTransactionError("Cannot send transactions in read-only mode")
        
        if not self.is_connected or not self.client:
            raise WalletTransactionError("Wallet not connected")
        
        try:
            # Convert addresses to Pubkey
            from_pubkey = self.keypair.pubkey()
            to_pubkey = Pubkey.from_string(to_address)
            
            # Convert SOL to lamports
            lamports = int(amount * Decimal(self.LAMPORTS_PER_SOL))
            
            # Create transfer instruction
            transfer_ix = transfer(
                TransferParams(
                    from_pubkey=from_pubkey,
                    to_pubkey=to_pubkey,
                    lamports=lamports
                )
            )
            
            # Get recent blockhash
            recent_blockhash = await self.client.get_latest_blockhash()
            
            # Create and sign transaction
            transaction = Transaction(
                recent_blockhash=recent_blockhash.value.blockhash,
                fee_payer=from_pubkey
            )
            transaction.add(transfer_ix)
            transaction.sign(self.keypair)
            
            # Send transaction
            response = await self.client.send_transaction(
                transaction,
                opts=TxOpts(skip_confirmation=False, preflight_commitment="confirmed")
            )
            
            tx_hash = str(response.value)
            
            logger.info(f"Sent {amount} SOL to {to_address}, tx: {tx_hash}")
            
            return TransactionResult(
                transaction_hash=tx_hash,
                status=TransactionStatus.PENDING
            )
            
        except Exception as e:
            logger.error(f"Failed to send SOL: {e}")
            raise WalletTransactionError(f"SOL transfer failed: {e}")
    
    async def send_token(
        self,
        token_address: str,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[Decimal] = None
    ) -> TransactionResult:
        """
        Send SPL tokens to another address.
        
        Args:
            token_address: SPL token mint address
            to_address: Recipient address
            amount: Amount of tokens to send
            gas_price: Not used for Solana
            
        Returns:
            TransactionResult with transaction details
        """
        if not self.keypair:
            raise WalletTransactionError("Cannot send transactions in read-only mode")
        
        if not self.is_connected or not self.client:
            raise WalletTransactionError("Wallet not connected")
        
        try:
            from spl.token.client import Token as SPLToken
            from spl.token.instructions import get_associated_token_address
            
            # Convert addresses to Pubkey
            mint_pubkey = Pubkey.from_string(token_address)
            owner_pubkey = self.keypair.pubkey()
            dest_pubkey = Pubkey.from_string(to_address)
            
            # Get token info
            token_info = await self._get_token_info(token_address)
            
            # Convert amount to token units
            token_amount = int(amount * Decimal(10**token_info.decimals))
            
            # Get associated token addresses
            source_token_account = get_associated_token_address(owner_pubkey, mint_pubkey)
            dest_token_account = get_associated_token_address(dest_pubkey, mint_pubkey)
            
            # Create transfer instruction
            transfer_ix = transfer_checked(
                TransferCheckedParams(
                    program_id=spl.token.constants.TOKEN_PROGRAM_ID,
                    source=source_token_account,
                    mint=mint_pubkey,
                    dest=dest_token_account,
                    owner=owner_pubkey,
                    amount=token_amount,
                    decimals=token_info.decimals
                )
            )
            
            # Get recent blockhash
            recent_blockhash = await self.client.get_latest_blockhash()
            
            # Create and sign transaction
            transaction = Transaction(
                recent_blockhash=recent_blockhash.value.blockhash,
                fee_payer=owner_pubkey
            )
            transaction.add(transfer_ix)
            transaction.sign(self.keypair)
            
            # Send transaction
            response = await self.client.send_transaction(
                transaction,
                opts=TxOpts(skip_confirmation=False, preflight_commitment="confirmed")
            )
            
            tx_hash = str(response.value)
            
            logger.info(f"Sent {amount} {token_info.symbol} to {to_address}, tx: {tx_hash}")
            
            return TransactionResult(
                transaction_hash=tx_hash,
                status=TransactionStatus.PENDING
            )
            
        except Exception as e:
            logger.error(f"Failed to send SPL token: {e}")
            raise WalletTransactionError(f"Token transfer failed: {e}")
    
    async def estimate_gas(
        self,
        to_address: str,
        amount: Decimal,
        token_address: Optional[str] = None,
        data: Optional[bytes] = None
    ) -> int:
        """
        Estimate transaction fees for Solana.
        
        Note: Solana has fixed fees per signature, typically 5000 lamports.
        
        Args:
            to_address: Recipient address
            amount: Transaction amount
            token_address: Token mint for SPL transfers
            data: Not used for Solana
            
        Returns:
            Estimated fee in lamports
        """
        # Solana has fixed fees per signature
        # Basic transactions: 5000 lamports
        # Complex transactions (like token transfers): 10000 lamports
        
        if token_address:
            return 10000  # SPL token transfer
        else:
            return 5000   # SOL transfer
    
    async def get_gas_price(self) -> Decimal:
        """
        Get current transaction fee in lamports.
        
        Returns:
            Transaction fee in lamports (converted to SOL equivalent)
        """
        # Solana has fixed fees, return typical fee in SOL
        fee_lamports = 5000
        return Decimal(fee_lamports) / Decimal(self.LAMPORTS_PER_SOL)
    
    async def get_transaction_status(self, transaction_hash: str) -> TransactionResult:
        """
        Get transaction status and details.
        
        Args:
            transaction_hash: Transaction signature to query
            
        Returns:
            TransactionResult with current status
        """
        if not self.is_connected or not self.client:
            raise WalletError("Wallet not connected")
        
        try:
            signature = Signature.from_string(transaction_hash)
            
            # Get transaction status
            response = await self.client.get_signature_statuses([signature])
            
            if not response.value or not response.value[0]:
                return TransactionResult(
                    transaction_hash=transaction_hash,
                    status=TransactionStatus.PENDING
                )
            
            status_info = response.value[0]
            
            if status_info.err:
                return TransactionResult(
                    transaction_hash=transaction_hash,
                    status=TransactionStatus.FAILED,
                    error_message=str(status_info.err)
                )
            elif status_info.confirmation_status == "confirmed" or status_info.confirmation_status == "finalized":
                return TransactionResult(
                    transaction_hash=transaction_hash,
                    status=TransactionStatus.CONFIRMED,
                    block_number=status_info.slot
                )
            else:
                return TransactionResult(
                    transaction_hash=transaction_hash,
                    status=TransactionStatus.PENDING
                )
                
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            raise WalletError(f"Transaction status query failed: {e}")
    
    async def sign_message(self, message: str) -> str:
        """
        Sign message with wallet keypair.
        
        Args:
            message: Message to sign
            
        Returns:
            Signed message as base58 string
        """
        if not self.keypair:
            raise WalletError("Cannot sign messages in read-only mode")
        
        try:
            message_bytes = message.encode('utf-8')
            signature = self.keypair.sign_message(message_bytes)
            return base58.b58encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to sign message: {e}")
            raise WalletError(f"Message signing failed: {e}")
    
    async def validate_address(self, address: str) -> bool:
        """
        Validate Solana address format.
        
        Args:
            address: Address to validate
            
        Returns:
            True if address is valid
        """
        try:
            Pubkey.from_string(address)
            return True
        except:
            return False
    
    async def _get_token_info(self, token_address: str) -> SPLToken:
        """
        Get SPL token information.
        
        Args:
            token_address: Token mint address
            
        Returns:
            SPLToken with token details
        """
        # Check cache first
        if token_address in self._token_cache:
            return self._token_cache[token_address]
        
        if not self.is_connected or not self.client:
            raise WalletError("Wallet not connected")
        
        try:
            mint_pubkey = Pubkey.from_string(token_address)
            
            # Get mint account info
            response = await self.client.get_account_info(mint_pubkey)
            
            if not response.value or not response.value.data:
                raise WalletError(f"Token mint not found: {token_address}")
            
            # Parse mint data (first 44 bytes contain mint info)
            data = response.value.data
            if len(data) < 44:
                raise WalletError("Invalid mint data")
            
            # Decimals is at offset 4
            decimals = data[4]
            
            # For now, we'll use generic symbol/name
            # In production, you'd want to integrate with token lists or metadata
            token_info = SPLToken(
                mint_address=token_address,
                symbol=f"TOKEN_{token_address[:8]}",
                decimals=decimals,
                name=f"SPL Token {token_address[:8]}"
            )
            
            # Cache token info
            self._token_cache[token_address] = token_info
            
            return token_info
            
        except Exception as e:
            logger.error(f"Failed to get token info for {token_address}: {e}")
            raise WalletError(f"Token info query failed: {e}")
    
    async def get_supported_tokens(self) -> List[str]:
        """
        Get list of commonly used SPL token addresses.
        
        Returns:
            List of SPL token mint addresses
        """
        # Common SPL tokens - these could be loaded from config or token lists
        if self.config.network == NetworkType.MAINNET:
            return [
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
                "So11111111111111111111111111111111111111112",   # Wrapped SOL
                "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",   # Marinade SOL
            ]
        else:  # Devnet/Testnet
            return [
                "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU",  # USDC (Devnet)
            ]
    
    async def create_associated_token_account(
        self,
        mint_address: str,
        owner_address: Optional[str] = None
    ) -> str:
        """
        Create associated token account for SPL token.
        
        Args:
            mint_address: SPL token mint address
            owner_address: Owner address (defaults to wallet address)
            
        Returns:
            Created token account address
        """
        if not self.keypair:
            raise WalletTransactionError("Cannot create accounts in read-only mode")
        
        if not self.is_connected or not self.client:
            raise WalletTransactionError("Wallet not connected")
        
        try:
            from spl.token.instructions import (
                create_associated_token_account,
                get_associated_token_address
            )
            
            mint_pubkey = Pubkey.from_string(mint_address)
            owner_pubkey = Pubkey.from_string(owner_address or self.wallet_address)
            
            # Get associated token address
            token_account = get_associated_token_address(owner_pubkey, mint_pubkey)
            
            # Check if account already exists
            account_info = await self.client.get_account_info(token_account)
            if account_info.value:
                return str(token_account)  # Account already exists
            
            # Create associated token account instruction
            create_ix = create_associated_token_account(
                payer=self.keypair.pubkey(),
                owner=owner_pubkey,
                mint=mint_pubkey
            )
            
            # Get recent blockhash
            recent_blockhash = await self.client.get_latest_blockhash()
            
            # Create and sign transaction
            transaction = Transaction(
                recent_blockhash=recent_blockhash.value.blockhash,
                fee_payer=self.keypair.pubkey()
            )
            transaction.add(create_ix)
            transaction.sign(self.keypair)
            
            # Send transaction
            response = await self.client.send_transaction(
                transaction,
                opts=TxOpts(skip_confirmation=False, preflight_commitment="confirmed")
            )
            
            logger.info(f"Created associated token account: {token_account}")
            return str(token_account)
            
        except Exception as e:
            logger.error(f"Failed to create associated token account: {e}")
            raise WalletTransactionError(f"Token account creation failed: {e}")
    
    async def get_token_accounts(self) -> List[Dict[str, Any]]:
        """
        Get all SPL token accounts owned by wallet.
        
        Returns:
            List of token account information
        """
        if not self.is_connected or not self.client or not self.wallet_address:
            raise WalletError("Wallet not connected")
        
        try:
            owner_pubkey = Pubkey.from_string(self.wallet_address)
            
            # Get all token accounts
            response = await self.client.get_token_accounts_by_owner(owner_pubkey, {})
            
            token_accounts = []
            for account in response.value:
                account_data = account.account.data.parsed['info']
                token_accounts.append({
                    'address': str(account.pubkey),
                    'mint': account_data['mint'],
                    'balance': Decimal(account_data['tokenAmount']['uiAmount'] or 0),
                    'decimals': account_data['tokenAmount']['decimals']
                })
            
            return token_accounts
            
        except Exception as e:
            logger.error(f"Failed to get token accounts: {e}")
            raise WalletError(f"Token accounts query failed: {e}")