"""
Real-time P&L Tracking System

This module contains the real-time profit and loss tracking system for live trading.
Extracted from live_mode.py for better maintainability.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
from uuid import uuid4
import structlog

from src.portfolio.base import Portfolio
from .config import LiveModeConfig, PnLAlert

logger = structlog.get_logger()


class RealTimePnLTracker:
    """Real-time P&L tracking system."""
    
    def __init__(self, portfolio: Portfolio, config: LiveModeConfig):
        self.portfolio = portfolio
        self.config = config
        self.update_frequency_seconds = config.pnl_update_frequency_seconds
        self.loss_alert_threshold = config.loss_alert_threshold
        self.is_tracking = False
        self.pnl_history: List[Dict[str, Any]] = []
        self.active_alerts: List[PnLAlert] = []
        self.logger = logger.bind(component="RealTimePnLTracker")
        self._tracking_task: Optional[asyncio.Task] = None
    
    async def start_tracking(self) -> None:
        """Start real-time P&L tracking."""
        if self.is_tracking:
            return
        
        self.is_tracking = True
        self._tracking_task = asyncio.create_task(self._tracking_loop())
        self.logger.info("Real-time P&L tracking started")
    
    async def stop_tracking(self) -> None:
        """Stop real-time P&L tracking."""
        self.is_tracking = False
        if self._tracking_task:
            self._tracking_task.cancel()
            try:
                await self._tracking_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Real-time P&L tracking stopped")
    
    async def update_pnl(self, current_value: Decimal, unrealized_pnl: Decimal) -> None:
        """Update P&L with current values."""
        timestamp = datetime.now()
        
        pnl_snapshot = {
            "timestamp": timestamp,
            "portfolio_value": current_value,
            "unrealized_pnl": unrealized_pnl,
        }
        
        self.pnl_history.append(pnl_snapshot)
        
        # Check for alerts
        await self._check_pnl_alerts(current_value, unrealized_pnl)
        
        # Keep only last 1000 entries
        if len(self.pnl_history) > 1000:
            self.pnl_history = self.pnl_history[-1000:]
    
    async def _tracking_loop(self) -> None:
        """Main P&L tracking loop."""
        while self.is_tracking:
            try:
                performance = self.portfolio.performance_metrics
                await self.update_pnl(performance.current_balance, performance.unrealized_pnl)
                await asyncio.sleep(self.update_frequency_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in P&L tracking loop", error=str(e))
                await asyncio.sleep(self.update_frequency_seconds)
    
    async def _check_pnl_alerts(self, current_value: Decimal, unrealized_pnl: Decimal) -> None:
        """Check for P&L alert conditions."""
        initial_value = Decimal("50000")  # Would get from config or portfolio
        loss_pct = (initial_value - current_value) / initial_value
        
        if loss_pct > self.loss_alert_threshold:
            alert = PnLAlert(
                alert_id=str(uuid4()),
                message=f"Loss threshold breached: {loss_pct:.2%} loss",
                severity="HIGH",
                timestamp=datetime.now(),
                pnl_amount=current_value - initial_value,
                threshold_breached="loss_threshold"
            )
            self.active_alerts.append(alert)
            
            self.logger.warning("P&L alert triggered", 
                              alert_id=alert.alert_id,
                              message=alert.message)
    
    def get_active_alerts(self) -> List[PnLAlert]:
        """Get currently active P&L alerts."""
        return self.active_alerts.copy()
    
    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self.active_alerts.clear()

    def get_pnl_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent P&L history."""
        return self.pnl_history[-limit:] if limit else self.pnl_history.copy()

    def get_current_pnl_snapshot(self) -> Dict[str, Any]:
        """Get current P&L snapshot."""
        if not self.pnl_history:
            return {}
        return self.pnl_history[-1].copy()

    def calculate_daily_pnl(self) -> Decimal:
        """Calculate daily P&L."""
        # Simplified implementation - would need proper daily calculation
        if len(self.pnl_history) < 2:
            return Decimal("0")
        
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_entries = [
            entry for entry in self.pnl_history 
            if entry["timestamp"] >= today_start
        ]
        
        if not daily_entries:
            return Decimal("0")
        
        return daily_entries[-1]["portfolio_value"] - daily_entries[0]["portfolio_value"]

    def get_pnl_statistics(self) -> Dict[str, Any]:
        """Get P&L tracking statistics."""
        if not self.pnl_history:
            return {
                "total_entries": 0,
                "tracking_active": self.is_tracking,
                "active_alerts": len(self.active_alerts)
            }
        
        values = [entry["portfolio_value"] for entry in self.pnl_history]
        
        return {
            "total_entries": len(self.pnl_history),
            "tracking_active": self.is_tracking,
            "active_alerts": len(self.active_alerts),
            "current_value": values[-1] if values else Decimal("0"),
            "min_value": min(values) if values else Decimal("0"),
            "max_value": max(values) if values else Decimal("0"),
            "value_range": max(values) - min(values) if values else Decimal("0"),
            "update_frequency": self.update_frequency_seconds,
            "last_update": self.pnl_history[-1]["timestamp"] if self.pnl_history else None
        }