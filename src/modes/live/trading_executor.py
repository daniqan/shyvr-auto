"""
Live Trading Executor

This module contains the live trading execution system for DEX integration.
Extracted from live_mode.py for better maintainability.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional
import structlog

from src.portfolio.base import Portfolio
from src.dex.base import DEXBase, SwapResult, SwapStatus, DEXError

logger = structlog.get_logger()


class TradingSessionManager:
    """Manages trading sessions and state."""
    
    def __init__(self, config):
        self.config = config
        from uuid import uuid4
        self.session_id = str(uuid4())
        self.session_start_time = datetime.now()
        self.last_trade_time: Optional[datetime] = None
        self.trades_this_session = 0
        self.volume_this_session = Decimal("0")
        self.logger = logger.bind(component="TradingSessionManager")
    
    def can_trade_now(self) -> bool:
        """Check if trading is allowed based on timing constraints."""
        now = datetime.now()
        
        # Check minimum interval between trades
        if (self.last_trade_time and 
            (now - self.last_trade_time).total_seconds() < self.config.min_trade_interval_seconds):
            return False
        
        return True
    
    def record_trade(self, amount_usd: Decimal) -> None:
        """Record a completed trade."""
        self.last_trade_time = datetime.now()
        self.trades_this_session += 1
        self.volume_this_session += amount_usd
        
        self.logger.info("Trade recorded",
                        session_trades=self.trades_this_session,
                        session_volume=str(self.volume_this_session))


class LiveTradingExecutor:
    """Live trading executor with real DEX integration."""
    
    def __init__(self, portfolio: Portfolio, dex_clients: Dict[str, DEXBase], 
                 enable_real_trading: bool = True, max_slippage_bps: int = 100,
                 order_timeout_seconds: int = 30):
        self.portfolio = portfolio
        self.dex_clients = dex_clients
        self.enable_real_trading = enable_real_trading
        self.max_slippage_bps = max_slippage_bps
        self.order_timeout_seconds = order_timeout_seconds
        self.is_active = False
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.logger = logger.bind(component="LiveTradingExecutor")
    
    async def start(self) -> None:
        """Start the trading executor."""
        self.is_active = True
        self.logger.info("Live trading executor started", 
                        enable_real_trading=self.enable_real_trading)
    
    async def stop(self) -> None:
        """Stop the trading executor."""
        self.is_active = False
        
        # Cancel any pending orders
        for order_id in list(self.pending_orders.keys()):
            await self._cancel_order(order_id)
        
        self.logger.info("Live trading executor stopped")
    
    async def execute_buy_order(self, token_address: str, amount_usd: Decimal, 
                               dex_preference: List[str] = None) -> SwapResult:
        """Execute real buy order through DEX."""
        if not self.is_active:
            return SwapResult(
                transaction_hash="EXECUTOR_INACTIVE",
                status=SwapStatus.FAILED,
                input_token="USDC",
                output_token=token_address,
                input_amount=amount_usd,
                error_message="Trading executor not active"
            )
        
        if not self.enable_real_trading:
            # Return simulated result for testing
            return SwapResult(
                transaction_hash="SIMULATED_BUY",
                status=SwapStatus.CONFIRMED,
                input_token="USDC",
                output_token=token_address,
                input_amount=amount_usd,
                actual_output_amount=amount_usd / Decimal("1.5"),  # Mock price
                timestamp=datetime.now(),
                dex_name="simulated"
            )
        
        dex_order = dex_preference or ["jupiter", "uniswap_v3", "hyperliquid"]
        
        for dex_name in dex_order:
            if dex_name not in self.dex_clients:
                continue
            
            dex_client = self.dex_clients[dex_name]
            
            try:
                # Get quote
                quote = await dex_client.get_quote(
                    input_token="USDC",
                    output_token=token_address,
                    amount=amount_usd,
                    slippage_bps=self.max_slippage_bps
                )
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    dex_client.execute_swap(quote),
                    timeout=self.order_timeout_seconds
                )
                
                # Record execution
                self._record_execution("BUY", token_address, amount_usd, result)
                
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning("Order timeout", dex=dex_name, token=token_address)
                continue
            except DEXError as e:
                self.logger.warning("DEX error", dex=dex_name, error=str(e))
                continue
            except Exception as e:
                self.logger.error("Unexpected error", dex=dex_name, error=str(e))
                continue
        
        # All DEXs failed
        return SwapResult(
            transaction_hash="ALL_DEX_FAILED",
            status=SwapStatus.FAILED,
            input_token="USDC",
            output_token=token_address,
            input_amount=amount_usd,
            error_message="All DEX clients failed"
        )
    
    async def execute_sell_order(self, token_address: str, amount: Decimal,
                                dex_preference: List[str] = None) -> SwapResult:
        """Execute real sell order through DEX."""
        if not self.is_active:
            return SwapResult(
                transaction_hash="EXECUTOR_INACTIVE",
                status=SwapStatus.FAILED,
                input_token=token_address,
                output_token="USDC",
                input_amount=amount,
                error_message="Trading executor not active"
            )
        
        if not self.enable_real_trading:
            # Return simulated result for testing
            return SwapResult(
                transaction_hash="SIMULATED_SELL",
                status=SwapStatus.CONFIRMED,
                input_token=token_address,
                output_token="USDC",
                input_amount=amount,
                actual_output_amount=amount * Decimal("1.5"),  # Mock price
                timestamp=datetime.now(),
                dex_name="simulated"
            )
        
        dex_order = dex_preference or ["jupiter", "uniswap_v3", "hyperliquid"]
        
        for dex_name in dex_order:
            if dex_name not in self.dex_clients:
                continue
            
            dex_client = self.dex_clients[dex_name]
            
            try:
                # Get quote
                quote = await dex_client.get_quote(
                    input_token=token_address,
                    output_token="USDC",
                    amount=amount,
                    slippage_bps=self.max_slippage_bps
                )
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    dex_client.execute_swap(quote),
                    timeout=self.order_timeout_seconds
                )
                
                # Record execution
                self._record_execution("SELL", token_address, amount, result)
                
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning("Order timeout", dex=dex_name, token=token_address)
                continue
            except DEXError as e:
                self.logger.warning("DEX error", dex=dex_name, error=str(e))
                continue
            except Exception as e:
                self.logger.error("Unexpected error", dex=dex_name, error=str(e))
                continue
        
        # All DEXs failed
        return SwapResult(
            transaction_hash="ALL_DEX_FAILED",
            status=SwapStatus.FAILED,
            input_token=token_address,
            output_token="USDC",
            input_amount=amount,
            error_message="All DEX clients failed"
        )
    
    def _record_execution(self, action: str, token: str, amount: Decimal, result: SwapResult) -> None:
        """Record trade execution for analysis."""
        execution_record = {
            "timestamp": datetime.now(),
            "action": action,
            "token": token,
            "amount": amount,
            "result": result,
            "success": result.status == SwapStatus.CONFIRMED,
            "dex_used": result.dex_name
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    async def _cancel_order(self, order_id: str) -> None:
        """Cancel pending order."""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            self.logger.info("Order cancelled", order_id=order_id)

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
                "pending_orders": len(self.pending_orders),
                "is_active": self.is_active
            }
        
        successful = sum(1 for record in self.execution_history if record["success"])
        total = len(self.execution_history)
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "pending_orders": len(self.pending_orders),
            "is_active": self.is_active,
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None
        }

    def get_dex_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics by DEX."""
        dex_stats = {}
        
        for record in self.execution_history:
            dex_name = record["result"].dex_name
            if dex_name not in dex_stats:
                dex_stats[dex_name] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "success_rate": 0.0
                }
            
            dex_stats[dex_name]["total"] += 1
            if record["success"]:
                dex_stats[dex_name]["successful"] += 1
            else:
                dex_stats[dex_name]["failed"] += 1
        
        # Calculate success rates
        for dex_name, stats in dex_stats.items():
            if stats["total"] > 0:
                stats["success_rate"] = stats["successful"] / stats["total"]
        
        return dex_stats

    async def get_pending_order_details(self) -> List[Dict[str, Any]]:
        """Get details of pending orders."""
        return [
            {
                "order_id": order_id,
                "details": order_details,
                "age_seconds": (datetime.now() - order_details.get("created_at", datetime.now())).total_seconds()
            }
            for order_id, order_details in self.pending_orders.items()
        ]