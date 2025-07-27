"""
DEX-Wallet Integration Bridge

This module provides the integration layer between DEX clients (like Jupiter)
and wallet implementations, enabling seamless execution of token swaps.

The bridge handles:
- Quote validation and execution
- Transaction preparation and signing
- Balance checks and account preparation
- Error handling and retry logic
- Comprehensive logging and monitoring
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import structlog

from src.dex.base import (
    DEXBase,
    SwapQuote,
    SwapResult,
    SwapStatus,
    SwapType,
    DEXError,
    DEXTransactionError
)
from src.wallet.base import (
    WalletBase,
    TransactionResult,
    TransactionStatus,
    WalletError,
    WalletTransactionError
)
from src.utils.base import Chain
from src.logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    ChainType, performance_tracker
)

logger = structlog.get_logger()


@dataclass
class SwapExecutionConfig:
    """Configuration for swap execution parameters."""
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_balance_checks: bool = True
    enable_account_preparation: bool = True
    confirmation_timeout_seconds: int = 60
    enable_slippage_protection: bool = True


class DEXWalletBridgeError(Exception):
    """Base error for DEX-Wallet bridge operations."""
    pass


class InsufficientBalanceError(DEXWalletBridgeError):
    """Error when insufficient balance for swap."""
    pass


class SwapExecutionError(DEXWalletBridgeError):
    """Error during swap execution."""
    pass


class DEXWalletBridge:
    """
    Integration bridge between DEX clients and wallet implementations.
    
    This class orchestrates the complete swap process from quote generation
    to transaction execution, providing a unified interface for DEX operations
    across different blockchain networks.
    """
    
    def __init__(
        self,
        dex_client: DEXBase,
        wallet: WalletBase,
        config: Optional[SwapExecutionConfig] = None
    ):
        """
        Initialize DEX-Wallet bridge.
        
        Args:
            dex_client: DEX client implementation (e.g., JupiterDEXClient)
            wallet: Wallet implementation (e.g., SolanaWallet)
            config: Swap execution configuration
            
        Raises:
            DEXWalletBridgeError: If DEX and wallet chains don't match
        """
        if dex_client.chain != wallet.config.chain:
            raise DEXWalletBridgeError(
                f"Chain mismatch: DEX ({dex_client.chain}) != Wallet ({wallet.config.chain})"
            )
        
        self.dex_client = dex_client
        self.wallet = wallet
        self.config = config or SwapExecutionConfig()
        self.chain = dex_client.chain
        
        self.logger = logger.bind(
            dex=dex_client.name,
            wallet=wallet.__class__.__name__,
            chain=self.chain.value
        )
        
    async def get_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal,
        swap_type: SwapType = SwapType.EXACT_INPUT,
        slippage_bps: Optional[int] = None
    ) -> SwapQuote:
        """
        Get a swap quote from the DEX.
        
        Args:
            input_token: Input token address or symbol
            output_token: Output token address or symbol
            amount: Amount to swap
            swap_type: Type of swap (exact input/output)
            slippage_bps: Slippage tolerance in basis points
            
        Returns:
            SwapQuote with pricing and routing information
            
        Raises:
            DEXError: If quote generation fails
        """
        # Map Chain to ChainType for logging
        chain_map = {
            Chain.SOLANA: ChainType.SOLANA,
            Chain.ETHEREUM: ChainType.ETHEREUM,
            Chain.BASE: ChainType.BASE
        }
        
        async with performance_tracker(
            source="dex_wallet_bridge",
            operation="get_quote",
            category=ActivityCategory.TRADING
        ) as tracker:
            try:
                # Log quote request
                await activity_logger.log_activity(
                    category=ActivityCategory.TRADING,
                    action=ActivityAction.READ,
                    source="dex_wallet_bridge",
                    event_type="quote_request",
                    title=f"Requesting swap quote: {input_token} -> {output_token}",
                    severity=ActivitySeverity.INFO,
                    token_address=input_token,
                    chain=chain_map.get(self.chain, ChainType.SOLANA),
                    amount_usd=amount,
                    metadata={
                        "input_token": input_token,
                        "output_token": output_token,
                        "amount": str(amount),
                        "swap_type": swap_type.value,
                        "slippage_bps": slippage_bps,
                        "dex_client": self.dex_client.name
                    }
                )
                
                self.logger.info(
                    "Requesting swap quote",
                    input_token=input_token,
                    output_token=output_token,
                    amount=str(amount),
                    swap_type=swap_type.value
                )
                
                quote = await self.dex_client.get_quote(
                    input_token=input_token,
                    output_token=output_token,
                    amount=amount,
                    swap_type=swap_type,
                    slippage_bps=slippage_bps
                )
                
                # Log successful quote generation
                await activity_logger.log_activity(
                    category=ActivityCategory.TRADING,
                    action=ActivityAction.SUCCESS,
                    source="dex_wallet_bridge",
                    event_type="quote_generated",
                    title=f"Quote generated for {input_token} -> {output_token}",
                    severity=ActivitySeverity.INFO,
                    token_address=input_token,
                    chain=chain_map.get(self.chain, ChainType.SOLANA),
                    amount_usd=amount,
                    metadata={
                        "quote_id": quote.quote_id,
                        "price": str(quote.price),
                        "price_impact_bps": quote.price_impact_bps,
                        "output_amount": str(quote.output_amount),
                        "route": quote.route[:100] if hasattr(quote, 'route') and quote.route else None  # Truncate route data
                    }
                )
                
                self.logger.info(
                    "Generated swap quote",
                    quote_id=quote.quote_id,
                    price=str(quote.price),
                    price_impact_bps=quote.price_impact_bps,
                    output_amount=str(quote.output_amount)
                )
                
                return quote
                
            except Exception as e:
                # Log quote generation failure
                await activity_logger.log_error(
                    category=ActivityCategory.TRADING,
                    source="dex_wallet_bridge",
                    event_type="quote_generation_failed",
                    title=f"Failed to generate quote for {input_token} -> {output_token}",
                    error_message=str(e),
                    exception=e,
                    severity=ActivitySeverity.ERROR,
                    token_address=input_token,
                    chain=chain_map.get(self.chain, ChainType.SOLANA),
                    amount_usd=amount,
                    metadata={
                        "input_token": input_token,
                        "output_token": output_token,
                        "amount": str(amount),
                        "dex_client": self.dex_client.name
                    }
                )
                self.logger.error("Failed to get swap quote", error=str(e))
                raise DEXError(f"Quote generation failed: {e}")
    
    async def execute_swap(
        self,
        quote: SwapQuote,
        enable_pre_checks: bool = True
    ) -> SwapResult:
        """
        Execute a token swap using the provided quote.
        
        This method orchestrates the complete swap process:
        1. Pre-execution validation and balance checks
        2. Account preparation (if needed)
        3. Transaction preparation via DEX
        4. Transaction signing and submission via wallet
        5. Transaction confirmation monitoring
        
        Args:
            quote: SwapQuote to execute
            enable_pre_checks: Whether to perform pre-execution validation
            
        Returns:
            SwapResult with execution details
            
        Raises:
            InsufficientBalanceError: If insufficient balance
            SwapExecutionError: If swap execution fails
        """
        # Map Chain to ChainType for logging
        chain_map = {
            Chain.SOLANA: ChainType.SOLANA,
            Chain.ETHEREUM: ChainType.ETHEREUM,
            Chain.BASE: ChainType.BASE
        }
        
        start_time = datetime.now()
        
        # Log swap execution initiation
        await activity_logger.log_activity(
            category=ActivityCategory.TRADING,
            action=ActivityAction.EXECUTE,
            source="dex_wallet_bridge",
            event_type="swap_execution_started",
            title=f"Starting swap execution: {quote.input_token} -> {quote.output_token}",
            severity=ActivitySeverity.INFO,
            token_address=quote.input_token,
            chain=chain_map.get(self.chain, ChainType.SOLANA),
            amount_usd=quote.input_amount,
            metadata={
                "quote_id": quote.quote_id,
                "input_token": quote.input_token,
                "output_token": quote.output_token,
                "input_amount": str(quote.input_amount),
                "expected_output": str(quote.output_amount),
                "price": str(quote.price),
                "price_impact_bps": quote.price_impact_bps,
                "enable_pre_checks": enable_pre_checks
            }
        )
        
        async with performance_tracker(
            source="dex_wallet_bridge",
            operation="execute_swap",
            category=ActivityCategory.TRADING,
            metadata={"quote_id": quote.quote_id}
        ) as tracker:
            try:
                self.logger.info(
                    "Starting swap execution",
                    quote_id=quote.quote_id,
                    input_token=quote.input_token,
                    output_token=quote.output_token,
                    input_amount=str(quote.input_amount)
                )
                
                # Step 1: Pre-execution validation
                if enable_pre_checks:
                    await self._perform_pre_execution_checks(quote)
                
                # Step 2: Account preparation (Solana-specific)
                if self.chain == Chain.SOLANA and self.config.enable_account_preparation:
                    await self._prepare_swap_accounts(quote)
                
                # Step 3: Prepare swap transaction via DEX
                swap_result = await self._prepare_swap_transaction(quote)
                
                # Step 4: Execute transaction via wallet
                transaction_result = await self._execute_swap_transaction(swap_result)
                
                # Step 5: Update swap result with transaction details
                swap_result.transaction_hash = transaction_result.transaction_hash
                swap_result.status = SwapStatus.PENDING if transaction_result.status == TransactionStatus.PENDING else SwapStatus.CONFIRMED
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Log successful swap execution
                await activity_logger.log_activity(
                    category=ActivityCategory.TRADING,
                    action=ActivityAction.SUCCESS,
                    source="dex_wallet_bridge",
                    event_type="swap_executed",
                    title=f"Swap executed successfully: {quote.input_token} -> {quote.output_token}",
                    severity=ActivitySeverity.INFO,
                    token_address=quote.input_token,
                    chain=chain_map.get(self.chain, ChainType.SOLANA),
                    amount_usd=quote.input_amount,
                    execution_time_ms=int(execution_time * 1000),
                    metadata={
                        "quote_id": quote.quote_id,
                        "transaction_hash": swap_result.transaction_hash,
                        "status": swap_result.status.value,
                        "execution_time_seconds": execution_time,
                        "input_amount": str(quote.input_amount),
                        "output_amount": str(quote.output_amount)
                    }
                )
                
                self.logger.info(
                    "Swap execution completed",
                    quote_id=quote.quote_id,
                    transaction_hash=swap_result.transaction_hash,
                    execution_time=execution_time,
                    status=swap_result.status.value
                )
                
                return swap_result
                
            except InsufficientBalanceError as e:
                # Log insufficient balance error
                await activity_logger.log_error(
                    category=ActivityCategory.TRADING,
                    source="dex_wallet_bridge",
                    event_type="swap_insufficient_balance",
                    title=f"Insufficient balance for swap: {quote.input_token} -> {quote.output_token}",
                    error_message=str(e),
                    exception=e,
                    severity=ActivitySeverity.WARNING,
                    token_address=quote.input_token,
                    chain=chain_map.get(self.chain, ChainType.SOLANA),
                    amount_usd=quote.input_amount,
                    metadata={
                        "quote_id": quote.quote_id,
                        "required_amount": str(quote.input_amount)
                    }
                )
                raise
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Log swap execution failure
                await activity_logger.log_error(
                    category=ActivityCategory.TRADING,
                    source="dex_wallet_bridge",
                    event_type="swap_execution_failed",
                    title=f"Swap execution failed: {quote.input_token} -> {quote.output_token}",
                    error_message=str(e),
                    exception=e,
                    severity=ActivitySeverity.ERROR,
                    token_address=quote.input_token,
                    chain=chain_map.get(self.chain, ChainType.SOLANA),
                    amount_usd=quote.input_amount,
                    execution_time_ms=int(execution_time * 1000),
                    metadata={
                        "quote_id": quote.quote_id,
                        "execution_time_seconds": execution_time,
                        "failure_stage": "execution"
                    }
                )
                
                self.logger.error(
                    "Swap execution failed",
                    quote_id=quote.quote_id,
                    execution_time=execution_time,
                    error=str(e)
                )
                raise SwapExecutionError(f"Swap execution failed: {e}")
    
    async def _perform_pre_execution_checks(self, quote: SwapQuote) -> None:
        """
        Perform pre-execution validation checks.
        
        Args:
            quote: SwapQuote to validate
            
        Raises:
            DEXError: If quote validation fails
            InsufficientBalanceError: If insufficient balance
        """
        # Validate quote
        if not await self.dex_client.validate_quote(quote):
            raise DEXError("Quote validation failed")
        
        # Check balance if enabled
        if self.config.enable_balance_checks:
            await self._check_sufficient_balance(quote)
    
    async def _check_sufficient_balance(self, quote: SwapQuote) -> None:
        """
        Check if wallet has sufficient balance for the swap.
        
        Args:
            quote: SwapQuote to check balance for
            
        Raises:
            InsufficientBalanceError: If insufficient balance
        """
        try:
            # For Solana wallets, use the specialized balance check method
            if hasattr(self.wallet, 'check_token_balance_for_swap'):
                has_sufficient_balance = await self.wallet.check_token_balance_for_swap(
                    quote.input_token,
                    quote.input_amount
                )
                
                if not has_sufficient_balance:
                    raise InsufficientBalanceError(
                        f"Insufficient balance for swap: need {quote.input_amount} of {quote.input_token}"
                    )
            else:
                # Fallback to generic balance check
                if quote.input_token == self.wallet.get_native_symbol():
                    current_balance = await self.wallet.get_native_balance()
                else:
                    current_balance = await self.wallet.get_balance(quote.input_token)
                    current_balance = current_balance.token_balances.get(quote.input_token, Decimal(0))
                
                if current_balance < quote.input_amount:
                    raise InsufficientBalanceError(
                        f"Insufficient balance: have {current_balance}, need {quote.input_amount}"
                    )
            
            self.logger.debug("Balance check passed", input_token=quote.input_token)
            
        except InsufficientBalanceError:
            raise
        except Exception as e:
            self.logger.warning(f"Balance check failed: {e}")
            # Don't fail the swap for balance check errors, just log them
    
    async def _prepare_swap_accounts(self, quote: SwapQuote) -> None:
        """
        Prepare necessary token accounts for the swap (Solana-specific).
        
        Args:
            quote: SwapQuote requiring account preparation
            
        Raises:
            SwapExecutionError: If account preparation fails
        """
        try:
            if hasattr(self.wallet, 'prepare_swap_accounts'):
                accounts = await self.wallet.prepare_swap_accounts(
                    quote.input_token,
                    quote.output_token
                )
                self.logger.debug("Swap accounts prepared", accounts=accounts)
            
        except Exception as e:
            self.logger.warning(f"Account preparation failed: {e}")
            # Don't fail the swap, Jupiter can handle account creation
    
    async def _prepare_swap_transaction(self, quote: SwapQuote) -> SwapResult:
        """
        Prepare swap transaction via DEX client.
        
        Args:
            quote: SwapQuote to prepare transaction for
            
        Returns:
            SwapResult with transaction preparation details
            
        Raises:
            DEXTransactionError: If transaction preparation fails
        """
        wallet_address = self.wallet.wallet_address
        if not wallet_address:
            raise SwapExecutionError("Wallet address not available")
        
        return await self.dex_client.execute_swap(
            quote=quote,
            wallet_address=wallet_address
        )
    
    async def _execute_swap_transaction(self, swap_result: SwapResult) -> TransactionResult:
        """
        Execute the prepared swap transaction via wallet.
        
        Args:
            swap_result: SwapResult with prepared transaction data
            
        Returns:
            TransactionResult with execution details
            
        Raises:
            SwapExecutionError: If transaction execution fails
        """
        if not hasattr(swap_result, 'transaction_data') or not swap_result.transaction_data:
            raise SwapExecutionError("No transaction data available for execution")
        
        transaction_data = swap_result.transaction_data
        
        # Handle different wallet types and transaction formats
        if self.chain == Chain.SOLANA and hasattr(self.wallet, 'execute_dex_swap'):
            # Solana-specific DEX swap execution
            return await self.wallet.execute_dex_swap(
                swap_transaction_data=transaction_data.get("swapTransaction"),
                last_valid_block_height=transaction_data.get("lastValidBlockHeight")
            )
        else:
            raise SwapExecutionError(f"DEX swap execution not supported for chain: {self.chain}")
    
    async def get_swap_status(self, transaction_hash: str) -> SwapResult:
        """
        Get the status of a swap transaction.
        
        Args:
            transaction_hash: Transaction hash to query
            
        Returns:
            SwapResult with current status
        """
        try:
            # Try wallet first for transaction status
            tx_result = await self.wallet.get_transaction_status(transaction_hash)
            
            # Map wallet transaction status to swap status
            if tx_result.status == TransactionStatus.CONFIRMED:
                status = SwapStatus.CONFIRMED
            elif tx_result.status == TransactionStatus.FAILED:
                status = SwapStatus.FAILED
            else:
                status = SwapStatus.PENDING
            
            return SwapResult(
                transaction_hash=transaction_hash,
                status=status,
                input_token="",  # Would need to be stored separately
                output_token="",  # Would need to be stored separately
                input_amount=Decimal(0),  # Would need to be stored separately
                block_number=tx_result.block_number,
                error_message=tx_result.error_message,
                dex_name=self.dex_client.name
            )
            
        except Exception as e:
            self.logger.error("Failed to get swap status", transaction_hash=transaction_hash, error=str(e))
            return SwapResult(
                transaction_hash=transaction_hash,
                status=SwapStatus.FAILED,
                input_token="",
                output_token="",
                input_amount=Decimal(0),
                error_message=str(e),
                dex_name=self.dex_client.name
            )
    
    async def estimate_swap_cost(
        self,
        input_token: str,
        output_token: str,
        amount: Decimal
    ) -> Dict[str, Any]:
        """
        Estimate the total cost of a swap including fees.
        
        Args:
            input_token: Input token address
            output_token: Output token address
            amount: Amount to swap
            
        Returns:
            Dictionary with cost breakdown
        """
        try:
            # Get quote for fee estimation
            quote = await self.get_quote(input_token, output_token, amount)
            
            # Estimate transaction fee
            estimated_gas = await self.dex_client.estimate_gas(
                input_token, output_token, amount, self.wallet.wallet_address
            )
            
            gas_price = await self.wallet.get_gas_price()
            estimated_tx_fee = Decimal(estimated_gas) * gas_price
            
            # Calculate total cost
            total_cost = quote.input_amount + estimated_tx_fee
            
            return {
                "input_amount": quote.input_amount,
                "output_amount": quote.output_amount,
                "price": quote.price,
                "price_impact_bps": quote.price_impact_bps,
                "estimated_gas": estimated_gas,
                "gas_price": gas_price,
                "estimated_tx_fee": estimated_tx_fee,
                "total_cost": total_cost,
                "route": quote.route,
                "dex_name": self.dex_client.name
            }
            
        except Exception as e:
            self.logger.error("Failed to estimate swap cost", error=str(e))
            raise DEXError(f"Cost estimation failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on DEX and wallet components.
        
        Returns:
            Dictionary with health status
        """
        dex_health = await self.dex_client.health_check()
        
        wallet_connected = self.wallet.is_connected
        wallet_address = self.wallet.wallet_address
        
        return {
            "dex_health": dex_health,
            "wallet_connected": wallet_connected,
            "wallet_address": wallet_address,
            "chain": self.chain.value,
            "bridge_status": "healthy" if dex_health.get("status") == "healthy" and wallet_connected else "unhealthy"
        }
    
    async def close(self) -> None:
        """Close DEX and wallet connections."""
        if hasattr(self.dex_client, 'close'):
            await self.dex_client.close()
        
        if hasattr(self.wallet, 'disconnect'):
            await self.wallet.disconnect()
        
        self.logger.info("DEX-Wallet bridge closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        # Ensure both DEX and wallet are connected
        if not self.dex_client.is_connected:
            await self.dex_client.connect()
        
        if not self.wallet.is_connected:
            await self.wallet.connect()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()