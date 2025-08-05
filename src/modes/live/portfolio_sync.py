"""
Portfolio Synchronization System

This module contains the portfolio synchronization system for live trading.
Extracted from live_mode.py for better maintainability.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional
from uuid import UUID
import structlog

from src.portfolio.base import Portfolio, Position, PositionStatus
from src.dex.base import DEXBase

logger = structlog.get_logger()


@dataclass
class DiscrepancyReport:
    """Comprehensive discrepancy report."""
    timestamp: datetime
    type: str
    severity: str
    dex_name: str
    chain: str
    symbol: str
    position_id: Optional[UUID]
    expected_size: Optional[Decimal]
    actual_size: Optional[Decimal]
    expected_price: Optional[Decimal]
    actual_price: Optional[Decimal]
    difference: Optional[Decimal]
    auto_correctable: bool
    corrected: bool = False
    requires_manual_intervention: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncHealthMetrics:
    """Portfolio synchronization health metrics."""
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    total_discrepancies: int = 0
    auto_corrections: int = 0
    safety_alerts_triggered: int = 0
    average_sync_duration_ms: float = 0.0
    last_sync_time: Optional[datetime] = None
    error_rate: float = 0.0
    health_score: float = 1.0


class PortfolioSynchronizer:
    """
    Enhanced real-time portfolio synchronization with comprehensive reconciliation capabilities.
    
    Prevents state drift by periodically fetching on-chain balances from all connected wallets,
    comparing them against internal Portfolio state, and automatically correcting minor
    discrepancies while triggering safety alerts for significant discrepancies.
    
    Supports multi-chain, multi-DEX environments with robust error handling and monitoring.
    """
    
    def __init__(self, portfolio: Portfolio, dex_clients: Dict[str, DEXBase], 
                 sync_frequency_seconds: int = 30, config: Optional[Dict[str, Any]] = None,
                 monitor: Optional[Any] = None, alerting_system: Optional[Any] = None,
                 monitoring_hooks: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced portfolio synchronizer.
        
        Args:
            portfolio: Portfolio to synchronize
            dex_clients: Dictionary of DEX clients
            sync_frequency_seconds: Synchronization frequency
            config: Optional configuration dictionary
            monitor: Optional monitoring system
            alerting_system: Optional alerting system
            monitoring_hooks: Optional monitoring hooks
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate inputs
        if not dex_clients:
            raise ValueError("DEX clients cannot be empty")
        if sync_frequency_seconds <= 0:
            raise ValueError("Sync frequency must be positive")
        
        self.portfolio = portfolio
        self.dex_clients = dex_clients
        self.sync_frequency_seconds = sync_frequency_seconds
        self.monitor = monitor
        self.alerting_system = alerting_system
        self.monitoring_hooks = monitoring_hooks or {}
        
        # Configuration with defaults
        self.config = config or {}
        self.discrepancy_threshold = self.config.get("discrepancy_threshold", Decimal("0.01"))
        self.auto_correct_threshold = self.config.get("auto_correct_threshold", Decimal("0.005"))
        self.safety_alert_threshold = self.config.get("safety_alert_threshold", Decimal("0.05"))
        self.max_correction_attempts = self.config.get("max_correction_attempts", 3)
        self.enable_automatic_corrections = self.config.get("enable_automatic_corrections", True)
        self.enable_safety_alerts = self.config.get("enable_safety_alerts", True)
        
        # State tracking
        self.is_syncing = False
        self.last_sync_time: Optional[datetime] = None
        self.sync_errors = 0
        self.correction_attempts: Dict[str, int] = {}
        self.consecutive_alerts = 0
        self.health_metrics = SyncHealthMetrics()
        
        # Logger
        self.logger = logger.bind(component="PortfolioSynchronizer")
        self._sync_task: Optional[asyncio.Task] = None
        
        self.logger.info(
            "Enhanced PortfolioSynchronizer initialized",
            sync_frequency=sync_frequency_seconds,
            dex_count=len(dex_clients),
            auto_corrections_enabled=self.enable_automatic_corrections,
            safety_alerts_enabled=self.enable_safety_alerts
        )
    
    async def start_sync(self) -> None:
        """Start portfolio synchronization."""
        if self.is_syncing:
            return
        
        self.is_syncing = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        self.logger.info("Portfolio synchronization started")
    
    async def stop_sync(self) -> None:
        """Stop portfolio synchronization."""
        self.is_syncing = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Portfolio synchronization stopped")
    
    async def reconcile_positions(self) -> List[Dict[str, Any]]:
        """
        Comprehensive portfolio reconciliation with on-chain data.
        
        Fetches wallet balances from all DEXs, compares with internal state,
        detects discrepancies, applies automatic corrections, and triggers
        safety alerts as needed.
        
        Returns:
            List of discrepancy reports
        """
        start_time = datetime.now()
        discrepancies = []
        
        try:
            # Call pre-sync hook
            if "pre_sync" in self.monitoring_hooks:
                await self.monitoring_hooks["pre_sync"]()
            
            # Fetch wallet balances from all DEXs
            wallet_balances = await self._fetch_all_wallet_balances()
            
            # Get portfolio positions
            portfolio_positions = {pos.position_id: pos for pos in self.portfolio.positions.values()}
            
            # Compare positions and detect discrepancies
            discrepancies = await self._detect_discrepancies(portfolio_positions, wallet_balances)
            
            # Process discrepancies
            for discrepancy in discrepancies:
                await self._process_discrepancy(discrepancy)
            
            # Update health metrics
            self._update_health_metrics(start_time, True, len(discrepancies))
            
            # Record monitoring metrics
            if self.monitor:
                await self._record_monitoring_metrics(start_time, discrepancies)
            
            # Call post-sync hook
            if "post_sync" in self.monitoring_hooks:
                await self.monitoring_hooks["post_sync"](discrepancies)
            
            self.last_sync_time = datetime.now()
            self.sync_errors = 0
            
            self.logger.info(
                "Portfolio reconciliation completed",
                discrepancies_found=len(discrepancies),
                sync_duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            self.sync_errors += 1
            self._update_health_metrics(start_time, False, 0)
            
            self.logger.error("Portfolio reconciliation failed", error=str(e))
            
            # Record error in monitoring
            if self.monitor:
                self.monitor.record_metric("sync_error", 1)
            
            raise
        
        return [discrepancy.__dict__ for discrepancy in discrepancies]
    
    async def _fetch_all_wallet_balances(self) -> Dict[str, Dict[str, Any]]:
        """Fetch wallet balances from all DEX clients."""
        wallet_balances = {}
        
        for dex_name, dex_client in self.dex_clients.items():
            try:
                start_time = datetime.now()
                balances = await dex_client.get_wallet_balances()
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                wallet_balances[dex_name] = balances
                
                # Record DEX response time
                if self.monitor:
                    self.monitor.record_metric(f"dex_response_time_ms_{dex_name}", response_time)
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch balances from {dex_name}", error=str(e))
                
                # Add DEX error to results if configured to continue
                if self.config.get("continue_on_dex_failure", False):
                    wallet_balances[dex_name] = {"_error": str(e)}
                else:
                    raise
        
        return wallet_balances
    
    async def _detect_discrepancies(self, portfolio_positions: Dict[UUID, Position], 
                                  wallet_balances: Dict[str, Dict[str, Any]]) -> List[DiscrepancyReport]:
        """Detect discrepancies between portfolio and on-chain data."""
        discrepancies = []
        
        # Check each portfolio position against wallet balances
        for position_id, position in portfolio_positions.items():
            if position.status != PositionStatus.OPEN:
                continue
            
            dex_name = position.dex_name
            if dex_name not in wallet_balances:
                continue
            
            dex_balances = wallet_balances[dex_name]
            if "_error" in dex_balances:
                # DEX error - create error discrepancy
                discrepancy = DiscrepancyReport(
                    timestamp=datetime.now(),
                    type="dex_error",
                    severity="HIGH",
                    dex_name=dex_name,
                    chain=position.chain.value,
                    symbol=position.symbol,
                    position_id=position_id,
                    expected_size=position.size,
                    actual_size=None,
                    expected_price=position.current_price,
                    actual_price=None,
                    difference=None,
                    auto_correctable=False,
                    requires_manual_intervention=True,
                    metadata={"error": dex_balances["_error"]}
                )
                discrepancies.append(discrepancy)
                continue
            
            # Extract token symbol from position symbol (e.g., "SOL/USDC" -> "SOL")
            base_token = position.symbol.split('/')[0]
            
            if base_token in dex_balances:
                wallet_data = dex_balances[base_token]
                wallet_balance = wallet_data.get("balance", Decimal("0"))
                wallet_price = wallet_data.get("price_usd", position.current_price)
                
                # Check size discrepancy
                size_diff = abs(position.size - wallet_balance)
                size_diff_pct = size_diff / position.size if position.size > 0 else Decimal("1")
                
                if size_diff_pct > self.discrepancy_threshold:
                    severity = self._classify_discrepancy_severity(size_diff_pct)
                    auto_correctable = size_diff_pct <= self.auto_correct_threshold
                    
                    discrepancy = DiscrepancyReport(
                        timestamp=datetime.now(),
                        type="size_discrepancy",
                        severity=severity,
                        dex_name=dex_name,
                        chain=position.chain.value,
                        symbol=position.symbol,
                        position_id=position_id,
                        expected_size=position.size,
                        actual_size=wallet_balance,
                        difference=size_diff,
                        auto_correctable=auto_correctable,
                        requires_manual_intervention=severity in ["HIGH", "CRITICAL"]
                    )
                    discrepancies.append(discrepancy)
                
                # Check price discrepancy
                price_tolerance = self.config.get("price_tolerance_pct", Decimal("0.10"))  # 10% default
                price_diff_pct = abs(position.current_price - wallet_price) / position.current_price
                
                if price_diff_pct > price_tolerance:
                    severity = self._classify_discrepancy_severity(price_diff_pct)
                    
                    discrepancy = DiscrepancyReport(
                        timestamp=datetime.now(),
                        type="price_discrepancy",
                        severity=severity,
                        dex_name=dex_name,
                        chain=position.chain.value,
                        symbol=position.symbol,
                        position_id=position_id,
                        expected_price=position.current_price,
                        actual_price=wallet_price,
                        difference=wallet_price - position.current_price,
                        auto_correctable=False,  # Price discrepancies usually don't auto-correct
                        requires_manual_intervention=False
                    )
                    discrepancies.append(discrepancy)
            
            else:
                # Missing position on-chain
                discrepancy = DiscrepancyReport(
                    timestamp=datetime.now(),
                    type="missing_position",
                    severity="CRITICAL",
                    dex_name=dex_name,
                    chain=position.chain.value,
                    symbol=position.symbol,
                    position_id=position_id,
                    expected_size=position.size,
                    actual_size=Decimal("0"),
                    expected_price=position.current_price,
                    actual_price=None,
                    difference=position.size,
                    auto_correctable=False,
                    requires_manual_intervention=True
                )
                discrepancies.append(discrepancy)
        
        # Check for unexpected positions on-chain
        for dex_name, dex_balances in wallet_balances.items():
            if "_error" in dex_balances:
                continue
            
            for token, wallet_data in dex_balances.items():
                wallet_balance = wallet_data.get("balance", Decimal("0"))
                if wallet_balance == 0:
                    continue
                
                # Check if this token exists in portfolio
                token_symbol = f"{token}/USDC"  # Simplified assumption
                portfolio_has_token = any(
                    pos.symbol == token_symbol and pos.dex_name == dex_name 
                    for pos in portfolio_positions.values()
                )
                
                if not portfolio_has_token:
                    discrepancy = DiscrepancyReport(
                        timestamp=datetime.now(),
                        type="unexpected_position",
                        severity="MEDIUM",
                        dex_name=dex_name,
                        chain="unknown",  # Would need to derive from DEX
                        symbol=token,
                        position_id=None,
                        expected_size=Decimal("0"),
                        actual_size=wallet_balance,
                        expected_price=None,
                        actual_price=wallet_data.get("price_usd"),
                        difference=wallet_balance,
                        auto_correctable=False,
                        requires_manual_intervention=True
                    )
                    discrepancies.append(discrepancy)
        
        return discrepancies
    
    def _classify_discrepancy_severity(self, discrepancy_pct: Decimal) -> str:
        """Classify discrepancy severity based on percentage."""
        if discrepancy_pct <= self.auto_correct_threshold:
            return "LOW"
        elif discrepancy_pct <= self.safety_alert_threshold:
            return "MEDIUM"
        elif discrepancy_pct <= Decimal("0.20"):  # 20%
            return "HIGH"
        else:
            return "CRITICAL"
    
    async def _process_discrepancy(self, discrepancy: DiscrepancyReport) -> None:
        """Process a detected discrepancy."""
        # Call discrepancy hook
        if "discrepancy_detected" in self.monitoring_hooks:
            await self.monitoring_hooks["discrepancy_detected"](discrepancy)
        
        # Record discrepancy in monitoring
        if self.monitor:
            self.monitor.record_event(f"discrepancy_{discrepancy.type}", {
                "severity": discrepancy.severity,
                "symbol": discrepancy.symbol,
                "dex_name": discrepancy.dex_name
            })
        
        # Attempt automatic correction if applicable
        if (discrepancy.auto_correctable and 
            self.enable_automatic_corrections and 
            discrepancy.position_id):
            
            correction_key = f"{discrepancy.position_id}_{discrepancy.type}"
            attempts = self.correction_attempts.get(correction_key, 0)
            
            if attempts < self.max_correction_attempts:
                try:
                    success = await self.apply_automatic_correction(discrepancy)
                    if success:
                        discrepancy.corrected = True
                        self.health_metrics.auto_corrections += 1
                        self.logger.info(
                            "Automatic correction applied",
                            position_id=str(discrepancy.position_id),
                            type=discrepancy.type
                        )
                    else:
                        self.correction_attempts[correction_key] = attempts + 1
                
                except Exception as e:
                    self.logger.error("Automatic correction failed", error=str(e))
                    
                    # Rollback if enabled
                    if self.config.get("enable_correction_rollback", False):
                        await self.rollback_correction(discrepancy)
        
        # Trigger safety alerts if needed
        if (discrepancy.requires_manual_intervention and 
            self.enable_safety_alerts):
            await self.trigger_safety_alert(discrepancy)
    
    async def apply_automatic_correction(self, discrepancy: DiscrepancyReport) -> bool:
        """
        Apply automatic correction for minor discrepancies.
        
        This is a placeholder implementation. In a real system, this would:
        - Update position size based on actual wallet balance
        - Adjust portfolio cash balance accordingly
        - Create transaction records
        - Update position prices
        
        Returns:
            True if correction was successful, False otherwise
        """
        try:
            if discrepancy.type == "size_discrepancy" and discrepancy.position_id:
                # Find the position
                position = self.portfolio.positions.get(discrepancy.position_id)
                if position and discrepancy.actual_size is not None:
                    # Update position size to match on-chain balance
                    old_size = position.size
                    position.size = discrepancy.actual_size
                    position.updated_at = datetime.now()
                    
                    self.logger.info(
                        "Position size corrected",
                        position_id=str(discrepancy.position_id),
                        old_size=str(old_size),
                        new_size=str(discrepancy.actual_size)
                    )
                    return True
            
            elif discrepancy.type == "price_discrepancy" and discrepancy.position_id:
                # Update position price
                position = self.portfolio.positions.get(discrepancy.position_id)
                if position and discrepancy.actual_price is not None:
                    old_price = position.current_price
                    position.update_price(discrepancy.actual_price)
                    
                    self.logger.info(
                        "Position price corrected",
                        position_id=str(discrepancy.position_id),
                        old_price=str(old_price),
                        new_price=str(discrepancy.actual_price)
                    )
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error("Correction application failed", error=str(e))
            return False
    
    async def rollback_correction(self, discrepancy: DiscrepancyReport) -> bool:
        """
        Rollback a failed correction attempt.
        
        This is a placeholder implementation. In a real system, this would:
        - Restore original position state
        - Revert portfolio balance changes
        - Log rollback event
        
        Returns:
            True if rollback was successful, False otherwise
        """
        self.logger.warning(
            "Rolling back failed correction",
            position_id=str(discrepancy.position_id),
            type=discrepancy.type
        )
        return True
    
    async def trigger_safety_alert(self, discrepancy: DiscrepancyReport) -> None:
        """Trigger safety alert for significant discrepancies."""
        self.health_metrics.safety_alerts_triggered += 1
        alert_data = {
            "severity": discrepancy.severity,
            "type": f"portfolio_discrepancy_{discrepancy.type}",
            "symbol": discrepancy.symbol,
            "dex_name": discrepancy.dex_name,
            "chain": discrepancy.chain,
            "requires_manual_intervention": discrepancy.requires_manual_intervention,
            "timestamp": discrepancy.timestamp.isoformat(),
            "details": {
                "position_id": str(discrepancy.position_id) if discrepancy.position_id else None,
                "expected_size": str(discrepancy.expected_size) if discrepancy.expected_size else None,
                "actual_size": str(discrepancy.actual_size) if discrepancy.actual_size else None,
                "difference": str(discrepancy.difference) if discrepancy.difference else None
            }
        }
        
        # Send alert through alerting system
        if self.alerting_system:
            await self.alerting_system.send_alert(alert_data)
        
        # Check for escalation
        self.consecutive_alerts += 1
        escalation_threshold = self.config.get("escalation_threshold", 3)
        
        if self.consecutive_alerts >= escalation_threshold:
            await self.escalate_alert(alert_data)
        
        # Check for critical conditions
        if discrepancy.severity == "CRITICAL":
            critical_threshold = self.config.get("critical_alert_threshold", Decimal("0.50"))
            if discrepancy.difference and discrepancy.expected_size:
                diff_pct = discrepancy.difference / discrepancy.expected_size
                if diff_pct > critical_threshold:
                    await self.trigger_emergency_stop(alert_data)
        
        self.logger.critical(
            "Safety alert triggered",
            alert_type=alert_data["type"],
            severity=discrepancy.severity,
            symbol=discrepancy.symbol
        )
    
    async def escalate_alert(self, alert_data: Dict[str, Any]) -> None:
        """Escalate alert due to persistent issues."""
        alert_data["escalated"] = True
        alert_data["escalation_reason"] = f"Consecutive alerts threshold exceeded: {self.consecutive_alerts}"
        
        self.logger.critical("Alert escalated", alert_data=alert_data)
    
    async def trigger_emergency_stop(self, alert_data: Dict[str, Any]) -> None:
        """Trigger emergency stop for critical discrepancies."""
        self.logger.critical("EMERGENCY STOP TRIGGERED", alert_data=alert_data)
        
        # This would integrate with the emergency stop system
        # For now, just log the event
    
    def _update_health_metrics(self, start_time: datetime, success: bool, discrepancy_count: int) -> None:
        """Update synchronization health metrics."""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        self.health_metrics.total_syncs += 1
        if success:
            self.health_metrics.successful_syncs += 1
        else:
            self.health_metrics.failed_syncs += 1
        
        self.health_metrics.total_discrepancies += discrepancy_count
        self.health_metrics.last_sync_time = datetime.now()
        
        # Update averages
        total_duration = (self.health_metrics.average_sync_duration_ms * 
                         (self.health_metrics.total_syncs - 1) + duration_ms)
        self.health_metrics.average_sync_duration_ms = total_duration / self.health_metrics.total_syncs
        
        # Calculate error rate
        self.health_metrics.error_rate = (
            self.health_metrics.failed_syncs / self.health_metrics.total_syncs
        )
        
        # Calculate health score (simplified)
        success_rate = self.health_metrics.successful_syncs / self.health_metrics.total_syncs
        alert_penalty = min(self.health_metrics.safety_alerts_triggered * 0.1, 0.5)
        self.health_metrics.health_score = max(success_rate - alert_penalty, 0.0)
    
    async def _record_monitoring_metrics(self, start_time: datetime, discrepancies: List[DiscrepancyReport]) -> None:
        """Record metrics in monitoring system."""
        if not self.monitor:
            return
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Core metrics
        self.monitor.record_metric("sync_duration_ms", duration_ms)
        self.monitor.record_metric("sync_success_rate", 
                                 self.health_metrics.successful_syncs / self.health_metrics.total_syncs)
        self.monitor.record_metric("discrepancies_detected", len(discrepancies))
        
        # Health metrics
        self.monitor.record_metric("sync_health_score", self.health_metrics.health_score)
        self.monitor.record_metric("sync_error_rate", self.health_metrics.error_rate)
        
        # Discrepancy breakdown
        for severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            count = len([d for d in discrepancies if d.severity == severity])
            self.monitor.record_metric(f"discrepancies_{severity.lower()}", count)
    
    def get_error_rate(self) -> float:
        """Get current error rate."""
        return self.health_metrics.error_rate
    
    async def get_chain_summary(self) -> Dict[str, Any]:
        """Get summary of portfolio state by blockchain chain."""
        summary = {}
        
        for position in self.portfolio.positions.values():
            chain_name = position.chain.value.lower()
            if chain_name not in summary:
                summary[chain_name] = {
                    "positions": 0,
                    "total_value": Decimal("0"),
                    "dexs": set()
                }
            
            summary[chain_name]["positions"] += 1
            summary[chain_name]["total_value"] += position.market_value
            summary[chain_name]["dexs"].add(position.dex_name)
        
        # Convert sets to lists for JSON serialization
        for chain_data in summary.values():
            chain_data["dexs"] = list(chain_data["dexs"])
        
        return summary
    
    async def _sync_loop(self) -> None:
        """Main synchronization loop."""
        while self.is_syncing:
            try:
                await self.reconcile_positions()
                await asyncio.sleep(self.sync_frequency_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in sync loop", error=str(e))
                await asyncio.sleep(self.sync_frequency_seconds)

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current health metrics."""
        return {
            "total_syncs": self.health_metrics.total_syncs,
            "successful_syncs": self.health_metrics.successful_syncs,
            "failed_syncs": self.health_metrics.failed_syncs,
            "total_discrepancies": self.health_metrics.total_discrepancies,
            "auto_corrections": self.health_metrics.auto_corrections,
            "safety_alerts_triggered": self.health_metrics.safety_alerts_triggered,
            "average_sync_duration_ms": self.health_metrics.average_sync_duration_ms,
            "last_sync_time": self.health_metrics.last_sync_time.isoformat() if self.health_metrics.last_sync_time else None,
            "error_rate": self.health_metrics.error_rate,
            "health_score": self.health_metrics.health_score,
            "is_syncing": self.is_syncing
        }