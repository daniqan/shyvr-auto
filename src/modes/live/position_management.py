"""
Position Management Utilities

This module contains position and order management utilities for live trading.
Extracted from live_mode.py for better maintainability.
"""

from decimal import Decimal
from typing import Dict, List
from uuid import UUID
import structlog

from src.portfolio.base import Portfolio, Position

logger = structlog.get_logger()


class PositionManager:
    """Manages live trading positions with real-time updates."""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.position_cache: Dict[str, Position] = {}
        self.logger = logger.bind(component="PositionManager")
    
    async def get_live_positions(self) -> List[Position]:
        """Get current live positions with real-time updates."""
        return list(self.portfolio.positions.values())
    
    async def update_position_price(self, position_id: UUID, new_price: Decimal) -> None:
        """Update position with real-time price data."""
        positions = list(self.portfolio.positions.values())
        for position in positions:
            if position.position_id == position_id:
                position.update_price(new_price)
                self.logger.debug("Position price updated", 
                                position_id=str(position_id), 
                                new_price=str(new_price))
                break
    
    async def get_position_by_symbol(self, symbol: str) -> List[Position]:
        """Get positions by symbol."""
        return [
            position for position in self.portfolio.positions.values()
            if position.symbol == symbol
        ]
    
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        from src.portfolio.base import PositionStatus
        return [
            position for position in self.portfolio.positions.values()
            if position.status == PositionStatus.OPEN
        ]
    
    async def get_positions_by_dex(self, dex_name: str) -> List[Position]:
        """Get positions by DEX."""
        return [
            position for position in self.portfolio.positions.values()
            if position.dex_name == dex_name
        ]
    
    async def get_position_summary(self) -> Dict[str, any]:
        """Get summary of all positions."""
        positions = list(self.portfolio.positions.values())
        
        from src.portfolio.base import PositionStatus
        open_positions = [p for p in positions if p.status == PositionStatus.OPEN]
        closed_positions = [p for p in positions if p.status == PositionStatus.CLOSED]
        
        total_value = sum(pos.market_value for pos in open_positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in open_positions)
        total_realized_pnl = sum(pos.realized_pnl for pos in closed_positions)
        
        return {
            "total_positions": len(positions),
            "open_positions": len(open_positions),
            "closed_positions": len(closed_positions),
            "total_market_value": float(total_value),
            "total_unrealized_pnl": float(total_unrealized_pnl),
            "total_realized_pnl": float(total_realized_pnl),
            "largest_position": max(open_positions, key=lambda p: p.market_value) if open_positions else None,
            "smallest_position": min(open_positions, key=lambda p: p.market_value) if open_positions else None
        }
    
    async def get_positions_requiring_attention(self) -> List[Dict[str, any]]:
        """Get positions that may require manual attention."""
        attention_positions = []
        
        from src.portfolio.base import PositionStatus
        open_positions = [p for p in self.portfolio.positions.values() if p.status == PositionStatus.OPEN]
        
        for position in open_positions:
            issues = []
            
            # Check for large unrealized losses
            if position.unrealized_pnl < position.market_value * Decimal("-0.10"):  # 10% loss
                issues.append("large_unrealized_loss")
            
            # Check for stale positions (would need proper timestamp comparison)
            # if position is older than some threshold:
            #     issues.append("stale_position")
            
            # Check for low liquidity (would need liquidity data)
            # if position has low liquidity:
            #     issues.append("low_liquidity")
            
            if issues:
                attention_positions.append({
                    "position_id": str(position.position_id),
                    "symbol": position.symbol,
                    "issues": issues,
                    "market_value": float(position.market_value),
                    "unrealized_pnl": float(position.unrealized_pnl),
                    "unrealized_pnl_pct": float(position.unrealized_pnl / position.market_value) if position.market_value > 0 else 0
                })
        
        return attention_positions
    
    async def close_position(self, position_id: UUID, close_price: Decimal) -> bool:
        """Close a position with the given price."""
        try:
            for position in self.portfolio.positions.values():
                if position.position_id == position_id:
                    # Update position to closed status
                    from src.portfolio.base import PositionStatus
                    position.status = PositionStatus.CLOSED
                    position.exit_price = close_price
                    position.realized_pnl = (close_price - position.entry_price) * position.size
                    
                    self.logger.info("Position closed", 
                                   position_id=str(position_id),
                                   close_price=str(close_price),
                                   realized_pnl=str(position.realized_pnl))
                    return True
            
            self.logger.warning("Position not found for closing", position_id=str(position_id))
            return False
            
        except Exception as e:
            self.logger.error("Error closing position", 
                            position_id=str(position_id), 
                            error=str(e))
            return False
    
    async def update_all_positions_prices(self, price_data: Dict[str, Decimal]) -> int:
        """Update prices for all positions with provided price data."""
        updated_count = 0
        
        for position in self.portfolio.positions.values():
            # Extract base token from position symbol (e.g., "SOL/USDC" -> "SOL")
            base_token = position.symbol.split('/')[0]
            
            if base_token in price_data:
                old_price = position.current_price
                new_price = price_data[base_token]
                
                if old_price != new_price:
                    position.update_price(new_price)
                    updated_count += 1
                    
                    self.logger.debug("Position price updated in batch",
                                    symbol=position.symbol,
                                    old_price=str(old_price),
                                    new_price=str(new_price))
        
        if updated_count > 0:
            self.logger.info(f"Updated prices for {updated_count} positions")
        
        return updated_count
    
    def get_position_statistics(self) -> Dict[str, any]:
        """Get detailed position statistics."""
        positions = list(self.portfolio.positions.values())
        
        if not positions:
            return {
                "total_positions": 0,
                "avg_position_size": 0,
                "largest_position_size": 0,
                "smallest_position_size": 0,
                "total_unrealized_pnl": 0,
                "avg_unrealized_pnl": 0,
                "profitable_positions": 0,
                "losing_positions": 0
            }
        
        from src.portfolio.base import PositionStatus
        open_positions = [p for p in positions if p.status == PositionStatus.OPEN]
        
        if not open_positions:
            return {
                "total_positions": len(positions),
                "open_positions": 0,
                "avg_position_size": 0,
                "largest_position_size": 0,
                "smallest_position_size": 0,
                "total_unrealized_pnl": 0,
                "avg_unrealized_pnl": 0,
                "profitable_positions": 0,
                "losing_positions": 0
            }
        
        position_sizes = [float(p.market_value) for p in open_positions]
        unrealized_pnls = [float(p.unrealized_pnl) for p in open_positions]
        
        profitable_positions = len([p for p in open_positions if p.unrealized_pnl > 0])
        losing_positions = len([p for p in open_positions if p.unrealized_pnl < 0])
        
        return {
            "total_positions": len(positions),
            "open_positions": len(open_positions),
            "avg_position_size": sum(position_sizes) / len(position_sizes),
            "largest_position_size": max(position_sizes),
            "smallest_position_size": min(position_sizes),
            "total_unrealized_pnl": sum(unrealized_pnls),
            "avg_unrealized_pnl": sum(unrealized_pnls) / len(unrealized_pnls),
            "profitable_positions": profitable_positions,
            "losing_positions": losing_positions,
            "profit_ratio": profitable_positions / len(open_positions) if open_positions else 0
        }