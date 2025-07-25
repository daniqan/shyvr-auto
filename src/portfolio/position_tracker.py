"""
Position Tracker for Portfolio Management

This module provides comprehensive position tracking functionality
across different DEXs and blockchain networks. It handles position
lifecycle management, P&L tracking, and risk monitoring.

Key Features:
- Multi-chain position tracking (Solana, Ethereum, Hyperliquid)
- Real-time price updates and P&L calculations
- Stop loss and take profit monitoring
- Position sizing and risk management
- Support for spot and perpetual futures positions
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID
import structlog

from .base import (
    Position,
    PositionType,
    PositionStatus,
    PortfolioConfig,
    PortfolioError,
    InvalidPositionError,
)
from src.utils.base import Chain


logger = structlog.get_logger()


@dataclass
class PositionUpdateResult:
    """Result of position update operation."""
    success: bool
    position_id: UUID
    message: str = ""
    old_price: Optional[Decimal] = None
    new_price: Optional[Decimal] = None
    price_change: Optional[Decimal] = None
    pnl_change: Optional[Decimal] = None
    
    # Risk monitoring flags
    stop_loss_triggered: bool = False
    take_profit_triggered: bool = False
    liquidation_risk: bool = False
    
    # Price levels
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    liquidation_price: Optional[Decimal] = None
    
    # Funding updates (for perpetuals)
    funding_payment: Optional[Decimal] = None


@dataclass
class PositionCloseResult:
    """Result of position close operation."""
    success: bool
    position_id: UUID
    message: str = ""
    close_size: Optional[Decimal] = None
    close_price: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    remaining_size: Optional[Decimal] = None
    fees: Optional[Decimal] = None
    reason: Optional[str] = None


class PositionTracker:
    """
    Position tracker for managing individual trading positions.
    
    This class handles the complete lifecycle of trading positions,
    from opening to closing, including real-time price updates,
    P&L calculations, and risk monitoring.
    """
    
    def __init__(self, config: PortfolioConfig):
        """
        Initialize position tracker.
        
        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.positions: Dict[UUID, Position] = {}
        self.logger = logger.bind(component="position_tracker")
        
    @property
    def total_positions(self) -> int:
        """Get total number of positions."""
        return len(self.positions)
    
    @property
    def open_positions_count(self) -> int:
        """Get count of open positions (including partial positions)."""
        return len([p for p in self.positions.values() if p.status in [PositionStatus.OPEN, PositionStatus.PARTIAL]])
    
    @property
    def closed_positions_count(self) -> int:
        """Get count of closed positions."""
        return len([p for p in self.positions.values() if p.status == PositionStatus.CLOSED])
    
    @property
    def partial_positions_count(self) -> int:
        """Get count of partially closed positions."""
        return len([p for p in self.positions.values() if p.status == PositionStatus.PARTIAL])
    
    def add_position(self, position: Position) -> PositionUpdateResult:
        """
        Add a new position to tracking.
        
        Args:
            position: Position to add
            
        Returns:
            PositionUpdateResult with operation status
        """
        try:
            if position.position_id in self.positions:
                return PositionUpdateResult(
                    success=False,
                    position_id=position.position_id,
                    message=f"Position {position.position_id} already exists"
                )
            
            # Set stop loss and take profit levels if not already set
            if position.stop_loss_price is None:
                position.stop_loss_price = self._calculate_stop_loss_price(position)
            
            if position.take_profit_price is None:
                position.take_profit_price = self._calculate_take_profit_price(position)
            
            # Add position to tracking
            self.positions[position.position_id] = position
            
            self.logger.info(
                "Position added to tracker",
                position_id=str(position.position_id),
                symbol=position.symbol,
                size=str(position.size),
                entry_price=str(position.entry_price),
                dex=position.dex_name,
                chain=position.chain.value
            )
            
            return PositionUpdateResult(
                success=True,
                position_id=position.position_id,
                message="Position added successfully"
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to add position",
                position_id=str(position.position_id),
                error=str(e)
            )
            return PositionUpdateResult(
                success=False,
                position_id=position.position_id,
                message=f"Failed to add position: {e}"
            )
    
    def get_position(self, position_id: UUID) -> Optional[Position]:
        """
        Get position by ID.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Position if found, None otherwise
        """
        return self.positions.get(position_id)
    
    def update_position_price(
        self,
        position_id: UUID,
        new_price: Decimal
    ) -> PositionUpdateResult:
        """
        Update position with new market price.
        
        Args:
            position_id: Position identifier
            new_price: New market price
            
        Returns:
            PositionUpdateResult with update details
        """
        try:
            position = self.positions.get(position_id)
            if position is None:
                return PositionUpdateResult(
                    success=False,
                    position_id=position_id,
                    message="Position not found"
                )
            
            old_price = position.current_price
            old_pnl = position.unrealized_pnl
            
            # Update position price
            position.update_price(new_price)
            
            # Calculate changes
            price_change = new_price - old_price
            new_pnl = position.unrealized_pnl
            pnl_change = new_pnl - old_pnl
            
            # Check risk conditions
            stop_loss_triggered = self._check_stop_loss(position)
            take_profit_triggered = self._check_take_profit(position)
            liquidation_risk = self._check_liquidation_risk(position)
            
            self.logger.debug(
                "Position price updated",
                position_id=str(position_id),
                symbol=position.symbol,
                old_price=str(old_price),
                new_price=str(new_price),
                price_change=str(price_change),
                pnl=str(new_pnl),
                pnl_change=str(pnl_change),
                stop_loss_triggered=stop_loss_triggered,
                take_profit_triggered=take_profit_triggered
            )
            
            return PositionUpdateResult(
                success=True,
                position_id=position_id,
                message="Position price updated",
                old_price=old_price,
                new_price=new_price,
                price_change=price_change,
                pnl_change=pnl_change,
                stop_loss_triggered=stop_loss_triggered,
                take_profit_triggered=take_profit_triggered,
                liquidation_risk=liquidation_risk,
                stop_loss_price=position.stop_loss_price,
                take_profit_price=position.take_profit_price,
                liquidation_price=position.liquidation_price
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to update position price",
                position_id=str(position_id),
                error=str(e)
            )
            return PositionUpdateResult(
                success=False,
                position_id=position_id,
                message=f"Failed to update price: {e}"
            )
    
    def close_position(
        self,
        position_id: UUID,
        close_size: Decimal,
        close_price: Decimal,
        reason: str = "manual_close",
        fees: Decimal = Decimal("0")
    ) -> PositionCloseResult:
        """
        Close a position (fully or partially).
        
        Args:
            position_id: Position identifier
            close_size: Size to close
            close_price: Price at which to close
            reason: Reason for closing
            fees: Transaction fees
            
        Returns:
            PositionCloseResult with close details
        """
        try:
            position = self.positions.get(position_id)
            if position is None:
                return PositionCloseResult(
                    success=False,
                    position_id=position_id,
                    message="Position not found"
                )
            
            if close_size > position.size:
                return PositionCloseResult(
                    success=False,
                    position_id=position_id,
                    message=f"Close size {close_size} exceeds position size {position.size}"
                )
            
            if close_size <= 0:
                return PositionCloseResult(
                    success=False,
                    position_id=position_id,
                    message="Close size must be positive"
                )
            
            # Calculate realized P&L
            price_diff = close_price - position.entry_price
            
            # Apply position-specific calculations
            if position.position_type == PositionType.SPOT or position.side == "LONG":
                realized_pnl = price_diff * close_size
            elif position.side == "SHORT":
                realized_pnl = -price_diff * close_size
            else:
                realized_pnl = price_diff * close_size  # Default to long
            
            # Apply leverage for leveraged positions
            if position.leverage > 1:
                realized_pnl *= position.leverage
            
            # Subtract fees
            net_realized_pnl = realized_pnl - fees
            
            # Update position size
            remaining_size = position.size - close_size
            position.size = remaining_size
            
            # Update position status
            if remaining_size == 0:
                position.status = PositionStatus.CLOSED
            else:
                position.status = PositionStatus.PARTIAL
            
            position.updated_at = datetime.now()
            
            self.logger.info(
                "Position closed",
                position_id=str(position_id),
                symbol=position.symbol,
                close_size=str(close_size),
                close_price=str(close_price),
                realized_pnl=str(net_realized_pnl),
                remaining_size=str(remaining_size),
                reason=reason
            )
            
            return PositionCloseResult(
                success=True,
                position_id=position_id,
                message="Position closed successfully",
                close_size=close_size,
                close_price=close_price,
                realized_pnl=net_realized_pnl,
                remaining_size=remaining_size,
                fees=fees,
                reason=reason
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to close position",
                position_id=str(position_id),
                error=str(e)
            )
            return PositionCloseResult(
                success=False,
                position_id=position_id,
                message=f"Failed to close position: {e}"
            )
    
    def update_funding(
        self,
        position_id: UUID,
        funding_payment: Decimal
    ) -> PositionUpdateResult:
        """
        Update funding payment for perpetual positions.
        
        Args:
            position_id: Position identifier
            funding_payment: Funding payment amount
            
        Returns:
            PositionUpdateResult with update status
        """
        try:
            position = self.positions.get(position_id)
            if position is None:
                return PositionUpdateResult(
                    success=False,
                    position_id=position_id,
                    message="Position not found"
                )
            
            if position.position_type != PositionType.PERPETUAL:
                return PositionUpdateResult(
                    success=False,
                    position_id=position_id,
                    message="Funding updates only apply to perpetual positions"
                )
            
            # Update funding
            position.update_funding(funding_payment)
            
            self.logger.debug(
                "Position funding updated",
                position_id=str(position_id),
                funding_payment=str(funding_payment),
                total_funding=str(position.unrealized_funding)
            )
            
            return PositionUpdateResult(
                success=True,
                position_id=position_id,
                message="Funding updated",
                funding_payment=funding_payment
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to update funding",
                position_id=str(position_id),
                error=str(e)
            )
            return PositionUpdateResult(
                success=False,
                position_id=position_id,
                message=f"Failed to update funding: {e}"
            )
    
    def get_positions_by_status(self, status: PositionStatus) -> Dict[UUID, Position]:
        """Get positions filtered by status."""
        return {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.status == status
        }
    
    def get_positions_by_chain(self, chain: Chain) -> Dict[UUID, Position]:
        """Get positions filtered by blockchain chain."""
        return {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.chain == chain
        }
    
    def get_positions_by_dex(self, dex_name: str) -> Dict[UUID, Position]:
        """Get positions filtered by DEX."""
        return {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.dex_name == dex_name
        }
    
    def get_positions_by_type(self, position_type: PositionType) -> Dict[UUID, Position]:
        """Get positions filtered by position type."""
        return {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.position_type == position_type
        }
    
    @property
    def total_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L across all open positions."""
        return sum(
            pos.unrealized_pnl for pos in self.positions.values()
            if pos.status in [PositionStatus.OPEN, PositionStatus.PARTIAL]
        )
    
    @property 
    def total_market_value(self) -> Decimal:
        """Calculate total market value of all positions."""
        return sum(
            pos.market_value for pos in self.positions.values()
            if pos.status in [PositionStatus.OPEN, PositionStatus.PARTIAL]
        )
    
    @property
    def total_cost_basis(self) -> Decimal:
        """Calculate total cost basis of all positions."""
        return sum(
            pos.cost_basis for pos in self.positions.values()
            if pos.status in [PositionStatus.OPEN, PositionStatus.PARTIAL]
        )
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all positions."""
        chain_counts = {}
        dex_counts = {}
        type_counts = {}
        
        for position in self.positions.values():
            # Chain counts
            chain_name = position.chain.value.upper()
            chain_counts[chain_name] = chain_counts.get(chain_name, 0) + 1
            
            # DEX counts
            dex_counts[position.dex_name] = dex_counts.get(position.dex_name, 0) + 1
            
            # Position type counts
            type_name = position.position_type.value.upper()
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_positions": self.total_positions,
            "open_positions": self.open_positions_count,
            "closed_positions": self.closed_positions_count,
            "partial_positions": self.partial_positions_count,
            "chains": chain_counts,
            "dexs": dex_counts,
            "position_types": type_counts,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_market_value": self.total_market_value,
            "total_cost_basis": self.total_cost_basis
        }
    
    def _calculate_stop_loss_price(self, position: Position) -> Optional[Decimal]:
        """Calculate stop loss price based on configuration."""
        if not self.config.enable_risk_management:
            return None
        
        stop_loss_multiplier = Decimal("1") - self.config.stop_loss_pct
        
        # For long positions (spot or long perpetuals)
        if position.position_type == PositionType.SPOT or position.side == "LONG":
            return position.entry_price * stop_loss_multiplier
        # For short positions
        elif position.side == "SHORT":
            return position.entry_price * (Decimal("2") - stop_loss_multiplier)
        else:
            return position.entry_price * stop_loss_multiplier
    
    def _calculate_take_profit_price(self, position: Position) -> Optional[Decimal]:
        """Calculate take profit price based on configuration."""
        if not self.config.enable_risk_management:
            return None
        
        take_profit_multiplier = Decimal("1") + self.config.take_profit_pct
        
        # For long positions (spot or long perpetuals)
        if position.position_type == PositionType.SPOT or position.side == "LONG":
            return position.entry_price * take_profit_multiplier
        # For short positions
        elif position.side == "SHORT":
            return position.entry_price * (Decimal("2") - take_profit_multiplier)
        else:
            return position.entry_price * take_profit_multiplier
    
    def _check_stop_loss(self, position: Position) -> bool:
        """Check if stop loss should be triggered."""
        if position.stop_loss_price is None:
            return False
        
        # For long positions
        if position.position_type == PositionType.SPOT or position.side == "LONG":
            return position.current_price <= position.stop_loss_price
        # For short positions
        elif position.side == "SHORT":
            return position.current_price >= position.stop_loss_price
        else:
            return position.current_price <= position.stop_loss_price
    
    def _check_take_profit(self, position: Position) -> bool:
        """Check if take profit should be triggered."""
        if position.take_profit_price is None:
            return False
        
        # For long positions
        if position.position_type == PositionType.SPOT or position.side == "LONG":
            return position.current_price >= position.take_profit_price
        # For short positions
        elif position.side == "SHORT":
            return position.current_price <= position.take_profit_price
        else:
            return position.current_price >= position.take_profit_price
    
    def _check_liquidation_risk(self, position: Position) -> bool:
        """Check if position is at risk of liquidation."""
        if position.liquidation_price is None:
            return False
        
        # Add buffer for liquidation warning (e.g., 5% away from liquidation)
        buffer_pct = Decimal("0.05")
        
        # For long positions
        if position.side == "LONG":
            warning_price = position.liquidation_price * (Decimal("1") + buffer_pct)
            return position.current_price <= warning_price
        # For short positions
        elif position.side == "SHORT":
            warning_price = position.liquidation_price * (Decimal("1") - buffer_pct)
            return position.current_price >= warning_price
        else:
            return False