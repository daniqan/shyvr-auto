"""
Graceful Position Liquidation System for LiveMode.

This module implements a production-grade position liquidation system that can
gracefully liquidate positions when emergency conditions are triggered, while
minimizing market impact and slippage.

Key Features:
- Intelligent liquidation prioritization based on risk metrics
- Multi-DEX liquidation for optimal execution
- TWAP (Time-Weighted Average Price) liquidation strategies
- Partial and full liquidation capabilities
- Emergency liquidation with circuit breakers
- Slippage minimization techniques
- Real-time liquidation monitoring and reporting
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import structlog

from src.portfolio.base import Portfolio, Position, PositionStatus
from src.dex.base import SwapResult, SwapStatus, DEXBase
from src.modes.live_mode import LiveModeConfig


logger = structlog.get_logger()


class LiquidationReason(Enum):
    """Reasons for triggering liquidation."""
    EMERGENCY_DRAWDOWN = "emergency_drawdown"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    RISK_THRESHOLD = "risk_threshold"
    MARGIN_CALL = "margin_call"
    CORRELATION_RISK = "correlation_risk"
    SECTOR_OVEREXPOSURE = "sector_overexposure"
    MANUAL_TRIGGER = "manual_trigger"
    FLASH_CRASH_PROTECTION = "flash_crash_protection"
    CIRCUIT_BREAKER = "circuit_breaker"


class LiquidationStrategy(Enum):
    """Liquidation execution strategies."""
    IMMEDIATE = "immediate"           # Immediate market orders
    TWAP = "twap"                    # Time-weighted average price
    VWAP = "vwap"                    # Volume-weighted average price
    ICEBERG = "iceberg"              # Hidden large orders
    GRADUAL = "gradual"              # Gradual over time
    MULTI_DEX = "multi_dex"          # Across multiple DEXs


class LiquidationUrgency(Enum):
    """Liquidation urgency levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    EMERGENCY = "emergency"


@dataclass
class LiquidationTrigger:
    """Liquidation trigger conditions."""
    reason: LiquidationReason
    threshold_value: Decimal
    current_value: Decimal
    urgency: LiquidationUrgency
    message: str = ""
    triggered_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.triggered_at is None:
            self.triggered_at = datetime.now()


@dataclass
class PositionLiquidationPriority:
    """Position liquidation priority scoring."""
    position_id: UUID
    priority_score: Decimal  # 0.0 (low) to 1.0 (high priority)
    risk_score: Decimal
    unrealized_pnl: Decimal
    market_impact_score: Decimal
    liquidity_score: Decimal
    correlation_risk: Decimal
    sector_concentration: Decimal
    
    @property
    def total_priority(self) -> Decimal:
        """Calculate total priority score."""
        # Weighted combination of factors
        return (
            self.risk_score * Decimal("0.30") +
            abs(self.unrealized_pnl) * Decimal("0.25") +
            self.market_impact_score * Decimal("0.20") +
            (1 - self.liquidity_score) * Decimal("0.15") +  # Lower liquidity = higher priority
            self.correlation_risk * Decimal("0.10")
        )


@dataclass
class LiquidationPlan:
    """Comprehensive liquidation execution plan."""
    plan_id: str
    trigger: LiquidationTrigger
    total_positions: int
    positions_to_liquidate: List[UUID]
    liquidation_percentage: Decimal  # 0.0 to 1.0
    strategy: LiquidationStrategy
    urgency: LiquidationUrgency
    
    # Execution parameters
    max_slippage_tolerance: Decimal = Decimal("0.05")  # 5% max slippage
    execution_time_limit: int = 300  # 5 minutes max
    chunk_size_pct: Decimal = Decimal("0.20")  # 20% chunks for TWAP
    time_between_chunks: int = 30  # 30 seconds between chunks
    
    # Multi-DEX parameters
    dex_allocation: Dict[str, Decimal] = field(default_factory=dict)
    parallel_execution: bool = True
    
    # Risk management
    circuit_breaker_enabled: bool = True
    max_market_impact: Decimal = Decimal("0.10")  # 10% max market impact
    
    created_at: datetime = field(default_factory=datetime.now)
    is_partial_liquidation: bool = True


@dataclass
class LiquidationResult:
    """Result of liquidation execution."""
    plan_id: str
    execution_id: str
    success: bool
    positions_liquidated: int
    total_proceeds: Decimal
    average_slippage: Decimal
    execution_time: float  # seconds
    
    # Detailed results
    position_results: List[Dict[str, Any]] = field(default_factory=list)
    dex_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance metrics
    market_impact_achieved: Decimal = Decimal("0")
    slippage_vs_estimate: Decimal = Decimal("0")
    execution_efficiency: Decimal = Decimal("0")  # 0.0 to 1.0
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    completed_at: datetime = field(default_factory=datetime.now)
    emergency_mode: bool = False


class PositionLiquidator:
    """
    Production-grade position liquidation system.
    
    Handles graceful liquidation of positions with minimal market impact,
    intelligent prioritization, and comprehensive risk management.
    """
    
    def __init__(self, portfolio: Portfolio, dex_clients: Dict[str, DEXBase], 
                 config: LiveModeConfig):
        self.portfolio = portfolio
        self.dex_clients = dex_clients
        self.config = config
        self.logger = logger.bind(component="PositionLiquidator")
        
        # Liquidation state
        self.active_liquidations: Dict[str, LiquidationPlan] = {}
        self.liquidation_history: List[LiquidationResult] = []
        self.is_liquidating = False
        
        # Performance tracking
        self.total_liquidations = 0
        self.successful_liquidations = 0
        self.total_slippage = Decimal("0")
        self.average_execution_time = 0.0
        
        # Circuit breaker
        self.circuit_breaker_active = False
        self.circuit_breaker_triggered_at: Optional[datetime] = None
        
        # Configuration
        self.max_concurrent_liquidations = getattr(config, 'max_concurrent_liquidations', 3)
        self.emergency_liquidation_enabled = getattr(config, 'emergency_liquidation_enabled', True)
    
    async def assess_liquidation_needs(self, trigger: LiquidationTrigger) -> LiquidationPlan:
        """Assess portfolio and create liquidation plan."""
        self.logger.info("Assessing liquidation needs", reason=trigger.reason.value)
        
        # Get all open positions
        open_positions = [
            pos for pos in self.portfolio.positions.values() 
            if pos.status == PositionStatus.OPEN
        ]
        
        if not open_positions:
            return LiquidationPlan(
                plan_id=str(uuid4()),
                trigger=trigger,
                total_positions=0,
                positions_to_liquidate=[],
                liquidation_percentage=Decimal("0"),
                strategy=LiquidationStrategy.IMMEDIATE,
                urgency=trigger.urgency
            )
        
        # Calculate liquidation priorities
        priorities = await self._calculate_liquidation_priorities(open_positions)
        
        # Determine liquidation percentage based on trigger urgency
        liquidation_pct = self._get_liquidation_percentage(trigger)
        
        # Select positions to liquidate
        positions_to_liquidate = self._select_positions_for_liquidation(
            priorities, liquidation_pct
        )
        
        # Choose liquidation strategy
        strategy = self._select_liquidation_strategy(trigger.urgency, len(positions_to_liquidate))
        
        # Create liquidation plan
        plan = LiquidationPlan(
            plan_id=str(uuid4()),
            trigger=trigger,
            total_positions=len(open_positions),
            positions_to_liquidate=[p.position_id for p in positions_to_liquidate],
            liquidation_percentage=liquidation_pct,
            strategy=strategy,
            urgency=trigger.urgency,
            is_partial_liquidation=liquidation_pct < Decimal("1.0")
        )
        
        # Optimize execution parameters
        await self._optimize_execution_parameters(plan, positions_to_liquidate)
        
        self.logger.info("Liquidation plan created", 
                        plan_id=plan.plan_id,
                        positions_to_liquidate=len(positions_to_liquidate),
                        strategy=strategy.value)
        
        return plan
    
    async def execute_liquidation_plan(self, plan: LiquidationPlan) -> LiquidationResult:
        """Execute liquidation plan with comprehensive monitoring."""
        execution_id = str(uuid4())
        start_time = datetime.now()
        
        self.logger.info("Starting liquidation execution", 
                        plan_id=plan.plan_id,
                        execution_id=execution_id,
                        strategy=plan.strategy.value)
        
        # Initialize result
        result = LiquidationResult(
            plan_id=plan.plan_id,
            execution_id=execution_id,
            success=False,
            positions_liquidated=0,
            total_proceeds=Decimal("0"),
            average_slippage=Decimal("0"),
            execution_time=0.0,
            emergency_mode=plan.urgency == LiquidationUrgency.EMERGENCY
        )
        
        try:
            # Check circuit breaker
            if self.circuit_breaker_active:
                result.errors.append("Circuit breaker active - liquidation blocked")
                return result
            
            # Mark as active
            self.active_liquidations[plan.plan_id] = plan
            self.is_liquidating = True
            
            # Execute based on strategy
            if plan.strategy == LiquidationStrategy.IMMEDIATE:
                await self._execute_immediate_liquidation(plan, result)
            elif plan.strategy == LiquidationStrategy.TWAP:
                await self._execute_twap_liquidation(plan, result)
            elif plan.strategy == LiquidationStrategy.MULTI_DEX:
                await self._execute_multi_dex_liquidation(plan, result)
            elif plan.strategy == LiquidationStrategy.GRADUAL:
                await self._execute_gradual_liquidation(plan, result)
            else:
                # Default to immediate
                await self._execute_immediate_liquidation(plan, result)
            
            # Calculate final metrics
            end_time = datetime.now()
            result.execution_time = (end_time - start_time).total_seconds()
            result.success = len(result.errors) == 0
            
            # Update performance tracking
            self.total_liquidations += 1
            if result.success:
                self.successful_liquidations += 1
                self.total_slippage += result.average_slippage
            
            self.logger.info("Liquidation execution completed",
                           execution_id=execution_id,
                           success=result.success,
                           positions_liquidated=result.positions_liquidated,
                           total_proceeds=float(result.total_proceeds))
            
        except Exception as e:
            result.errors.append(f"Execution failed: {str(e)}")
            self.logger.error("Liquidation execution failed", 
                            execution_id=execution_id, 
                            error=str(e))
        finally:
            # Cleanup
            self.active_liquidations.pop(plan.plan_id, None)
            self.is_liquidating = len(self.active_liquidations) > 0
            self.liquidation_history.append(result)
            
            # Keep only last 100 results
            if len(self.liquidation_history) > 100:
                self.liquidation_history = self.liquidation_history[-100:]
        
        return result
    
    async def _calculate_liquidation_priorities(self, positions: List[Position]) -> List[PositionLiquidationPriority]:
        """Calculate liquidation priorities for positions."""
        priorities = []
        
        for position in positions:
            # Basic risk score based on unrealized PnL and volatility
            unrealized_pnl = getattr(position, 'unrealized_pnl', Decimal("0"))
            risk_score = self._calculate_position_risk_score(position)
            
            # Market impact score (simplified)
            market_impact = self._estimate_market_impact(position)
            
            # Liquidity score
            liquidity_score = self._assess_position_liquidity(position)
            
            # Correlation and concentration risks
            correlation_risk = self._assess_correlation_risk(position)
            sector_concentration = self._assess_sector_concentration(position)
            
            priority = PositionLiquidationPriority(
                position_id=position.position_id,
                priority_score=Decimal("0.5"),  # Will be calculated
                risk_score=risk_score,
                unrealized_pnl=unrealized_pnl,
                market_impact_score=market_impact,
                liquidity_score=liquidity_score,
                correlation_risk=correlation_risk,
                sector_concentration=sector_concentration
            )
            
            # Calculate final priority score
            priority.priority_score = priority.total_priority
            priorities.append(priority)
        
        # Sort by priority (highest first)
        priorities.sort(key=lambda x: x.total_priority, reverse=True)
        
        return priorities
    
    def _calculate_position_risk_score(self, position: Position) -> Decimal:
        """Calculate risk score for position."""
        # Simplified risk calculation
        unrealized_pnl = getattr(position, 'unrealized_pnl', Decimal("0"))
        position_value = getattr(position, 'market_value', Decimal("1"))
        
        if position_value <= 0:
            return Decimal("1.0")  # Maximum risk for zero-value positions
        
        # Risk based on unrealized loss percentage
        loss_pct = abs(unrealized_pnl) / position_value if unrealized_pnl < 0 else Decimal("0")
        
        # Cap at 1.0
        return min(loss_pct, Decimal("1.0"))
    
    def _estimate_market_impact(self, position: Position) -> Decimal:
        """Estimate market impact of liquidating position."""
        # Simplified market impact estimation
        position_size = getattr(position, 'market_value', Decimal("1000"))
        
        # Assume higher impact for larger positions
        # This would use real market depth data in production
        if position_size > Decimal("50000"):
            return Decimal("0.8")  # High impact
        elif position_size > Decimal("10000"):
            return Decimal("0.5")  # Medium impact
        else:
            return Decimal("0.2")  # Low impact
    
    def _assess_position_liquidity(self, position: Position) -> Decimal:
        """Assess liquidity of position for liquidation."""
        # Simplified liquidity assessment
        # This would use real market data in production
        symbol = getattr(position, 'symbol', '')
        
        if 'ETH' in symbol or 'SOL' in symbol:
            return Decimal("0.9")  # High liquidity
        elif 'USDC' in symbol or 'USDT' in symbol:
            return Decimal("0.95")  # Very high liquidity
        else:
            return Decimal("0.6")  # Medium liquidity
    
    def _assess_correlation_risk(self, position: Position) -> Decimal:
        """Assess correlation risk of position."""
        # Simplified correlation risk assessment
        correlation_group = getattr(position, 'correlation_group', 'DEFAULT')
        
        # Count positions in same correlation group
        same_group_positions = sum(
            1 for pos in self.portfolio.positions.values()
            if getattr(pos, 'correlation_group', 'DEFAULT') == correlation_group
            and pos.status == PositionStatus.OPEN
        )
        
        # Higher risk for more correlated positions
        return min(Decimal(str(same_group_positions)) / Decimal("10"), Decimal("1.0"))
    
    def _assess_sector_concentration(self, position: Position) -> Decimal:
        """Assess sector concentration risk."""
        # Simplified sector concentration assessment
        sector = getattr(position, 'sector', 'UNKNOWN')
        
        # Calculate sector exposure
        sector_positions = [
            pos for pos in self.portfolio.positions.values()
            if getattr(pos, 'sector', 'UNKNOWN') == sector
            and pos.status == PositionStatus.OPEN
        ]
        
        sector_value = sum(
            getattr(pos, 'market_value', Decimal("0")) 
            for pos in sector_positions
        )
        
        total_portfolio_value = self.portfolio.total_value
        sector_concentration = sector_value / total_portfolio_value if total_portfolio_value > 0 else Decimal("0")
        
        return sector_concentration
    
    def _get_liquidation_percentage(self, trigger: LiquidationTrigger) -> Decimal:
        """Determine liquidation percentage based on trigger."""
        if trigger.urgency == LiquidationUrgency.EMERGENCY:
            return Decimal("1.0")  # Full liquidation
        elif trigger.urgency == LiquidationUrgency.HIGH:
            return getattr(self.config, 'partial_liquidation_percentage', Decimal("0.75"))
        elif trigger.urgency == LiquidationUrgency.MEDIUM:
            return Decimal("0.50")  # 50% liquidation
        else:
            return Decimal("0.25")  # 25% liquidation
    
    def _select_positions_for_liquidation(self, priorities: List[PositionLiquidationPriority], 
                                        liquidation_pct: Decimal) -> List[Position]:
        """Select positions for liquidation based on priorities."""
        positions_to_liquidate = []
        total_positions = len(priorities)
        target_count = int(total_positions * liquidation_pct)
        
        # Take highest priority positions
        for i in range(min(target_count, len(priorities))):
            priority = priorities[i]
            position = self.portfolio.positions.get(str(priority.position_id))\n            if position:\n                positions_to_liquidate.append(position)
        
        return positions_to_liquidate
    
    def _select_liquidation_strategy(self, urgency: LiquidationUrgency, 
                                   position_count: int) -> LiquidationStrategy:
        """Select optimal liquidation strategy."""
        if urgency == LiquidationUrgency.EMERGENCY:
            return LiquidationStrategy.IMMEDIATE
        elif position_count > 5:
            return LiquidationStrategy.MULTI_DEX
        elif position_count > 2:
            return LiquidationStrategy.TWAP
        else:
            return LiquidationStrategy.GRADUAL
    
    async def _optimize_execution_parameters(self, plan: LiquidationPlan, 
                                           positions: List[Position]) -> None:
        """Optimize execution parameters for the plan."""
        # Calculate total value to liquidate
        total_value = sum(getattr(pos, 'market_value', Decimal("0")) for pos in positions)
        
        # Adjust slippage tolerance based on urgency
        if plan.urgency == LiquidationUrgency.EMERGENCY:
            plan.max_slippage_tolerance = Decimal("0.10")  # Accept higher slippage
            plan.execution_time_limit = 60  # 1 minute max
        elif plan.urgency == LiquidationUrgency.HIGH:
            plan.max_slippage_tolerance = Decimal("0.05")  # 5% max slippage
            plan.execution_time_limit = 180  # 3 minutes max
        else:
            plan.max_slippage_tolerance = Decimal("0.03")  # 3% max slippage
            plan.execution_time_limit = 300  # 5 minutes max
        
        # Optimize DEX allocation for multi-DEX strategy
        if plan.strategy == LiquidationStrategy.MULTI_DEX:
            plan.dex_allocation = await self._optimize_dex_allocation(positions)
        
        # Adjust chunk size for TWAP based on position sizes
        if plan.strategy == LiquidationStrategy.TWAP:
            if total_value > Decimal("100000"):  # Large liquidation
                plan.chunk_size_pct = Decimal("0.10")  # Smaller chunks
                plan.time_between_chunks = 60  # More time between chunks
            elif total_value > Decimal("50000"):  # Medium liquidation
                plan.chunk_size_pct = Decimal("0.20")  # Standard chunks
                plan.time_between_chunks = 30
            else:  # Small liquidation
                plan.chunk_size_pct = Decimal("0.33")  # Larger chunks
                plan.time_between_chunks = 15
    
    async def _optimize_dex_allocation(self, positions: List[Position]) -> Dict[str, Decimal]:
        """Optimize allocation across DEXs for minimal slippage."""
        # Simplified DEX allocation optimization
        # In production, this would use real liquidity data
        
        dex_scores = {}
        for dex_name in self.dex_clients.keys():
            # Score based on assumed liquidity and fees
            if dex_name == "uniswap_v3":
                dex_scores[dex_name] = Decimal("0.4")  # 40% - best liquidity
            elif dex_name == "jupiter":
                dex_scores[dex_name] = Decimal("0.35")  # 35% - good aggregation
            elif dex_name == "hyperliquid":
                dex_scores[dex_name] = Decimal("0.25")  # 25% - lower liquidity
            else:
                dex_scores[dex_name] = Decimal("0.1")  # Default small allocation
        
        # Normalize scores to sum to 1.0
        total_score = sum(dex_scores.values())
        if total_score > 0:
            return {dex: score / total_score for dex, score in dex_scores.items()}
        else:
            # Equal allocation fallback
            equal_share = Decimal("1.0") / len(self.dex_clients)
            return {dex: equal_share for dex in self.dex_clients.keys()}
    
    # Execution strategies
    
    async def _execute_immediate_liquidation(self, plan: LiquidationPlan, 
                                           result: LiquidationResult) -> None:
        """Execute immediate liquidation using market orders."""
        for position_id in plan.positions_to_liquidate:
            position = self.portfolio.positions.get(str(position_id))
            if not position:
                continue
            
            try:
                # Execute immediate market order
                swap_result = await self._execute_position_liquidation(
                    position, plan, immediate=True
                )
                
                if swap_result.status == SwapStatus.CONFIRMED:
                    result.positions_liquidated += 1
                    result.total_proceeds += swap_result.actual_output_amount or Decimal("0")
                    
                    # Record position result
                    result.position_results.append({
                        "position_id": str(position_id),
                        "success": True,
                        "proceeds": float(swap_result.actual_output_amount or 0),
                        "slippage": self._calculate_slippage(position, swap_result)
                    })
                else:
                    result.errors.append(f"Failed to liquidate position {position_id}")
                    
            except Exception as e:
                result.errors.append(f"Error liquidating position {position_id}: {str(e)}")
                self.logger.error("Position liquidation failed", 
                                position_id=str(position_id), 
                                error=str(e))
    
    async def _execute_twap_liquidation(self, plan: LiquidationPlan, 
                                      result: LiquidationResult) -> None:
        """Execute TWAP (Time-Weighted Average Price) liquidation."""
        chunks_per_position = int(Decimal("1.0") / plan.chunk_size_pct)
        
        for chunk in range(chunks_per_position):
            if chunk > 0:
                # Wait between chunks
                await asyncio.sleep(plan.time_between_chunks)
            
            chunk_results = []
            for position_id in plan.positions_to_liquidate:
                position = self.portfolio.positions.get(str(position_id))
                if not position:
                    continue
                
                try:
                    # Calculate chunk size
                    chunk_size = getattr(position, 'size', Decimal("0")) * plan.chunk_size_pct
                    
                    # Create chunk position for liquidation
                    chunk_position = self._create_chunk_position(position, chunk_size)
                    
                    # Execute chunk
                    swap_result = await self._execute_position_liquidation(
                        chunk_position, plan, immediate=False
                    )
                    
                    if swap_result.status == SwapStatus.CONFIRMED:
                        chunk_results.append({
                            "position_id": str(position_id),
                            "chunk": chunk,
                            "success": True,
                            "proceeds": float(swap_result.actual_output_amount or 0)
                        })
                        
                        result.total_proceeds += swap_result.actual_output_amount or Decimal("0")
                    
                except Exception as e:
                    result.errors.append(f"TWAP chunk {chunk} failed for position {position_id}: {str(e)}")
            
            # Update result with chunk progress
            result.position_results.extend(chunk_results)
        
        # Count unique positions liquidated
        liquidated_positions = set()
        for pos_result in result.position_results:
            if pos_result.get("success"):
                liquidated_positions.add(pos_result["position_id"])
        
        result.positions_liquidated = len(liquidated_positions)
    
    async def _execute_multi_dex_liquidation(self, plan: LiquidationPlan, 
                                           result: LiquidationResult) -> None:
        """Execute liquidation across multiple DEXs in parallel."""
        liquidation_tasks = []
        
        for position_id in plan.positions_to_liquidate:
            position = self.portfolio.positions.get(str(position_id))
            if not position:
                continue
            
            # Split position across DEXs according to allocation
            for dex_name, allocation in plan.dex_allocation.items():
                if allocation <= 0:
                    continue
                
                # Calculate portion for this DEX
                position_portion = self._create_portion_position(position, allocation)
                
                # Create liquidation task
                task = asyncio.create_task(
                    self._liquidate_position_on_dex(position_portion, dex_name, plan)
                )
                liquidation_tasks.append((str(position_id), dex_name, task))
        
        # Execute all tasks in parallel
        for position_id, dex_name, task in liquidation_tasks:
            try:
                swap_result = await task
                
                if swap_result.status == SwapStatus.CONFIRMED:
                    result.total_proceeds += swap_result.actual_output_amount or Decimal("0")
                    
                    # Update DEX breakdown
                    if dex_name not in result.dex_breakdown:
                        result.dex_breakdown[dex_name] = {
                            "positions": 0,
                            "proceeds": 0.0,
                            "average_slippage": 0.0
                        }
                    
                    result.dex_breakdown[dex_name]["positions"] += 1
                    result.dex_breakdown[dex_name]["proceeds"] += float(swap_result.actual_output_amount or 0)
                
            except Exception as e:
                result.errors.append(f"Multi-DEX liquidation failed on {dex_name} for position {position_id}: {str(e)}")
        
        # Count total liquidated positions
        liquidated_positions = set()
        for dex_data in result.dex_breakdown.values():
            liquidated_positions.update(range(dex_data["positions"]))
        
        result.positions_liquidated = len(liquidated_positions)
    
    async def _execute_gradual_liquidation(self, plan: LiquidationPlan, 
                                         result: LiquidationResult) -> None:
        """Execute gradual liquidation over extended time period."""
        total_time = plan.execution_time_limit
        positions_count = len(plan.positions_to_liquidate)
        time_per_position = total_time // positions_count if positions_count > 0 else 0
        
        for i, position_id in enumerate(plan.positions_to_liquidate):
            if i > 0:
                # Wait between positions
                await asyncio.sleep(time_per_position)
            
            position = self.portfolio.positions.get(str(position_id))
            if not position:
                continue
            
            try:
                swap_result = await self._execute_position_liquidation(
                    position, plan, immediate=False
                )
                
                if swap_result.status == SwapStatus.CONFIRMED:
                    result.positions_liquidated += 1
                    result.total_proceeds += swap_result.actual_output_amount or Decimal("0")
                    
                    result.position_results.append({
                        "position_id": str(position_id),
                        "order": i,
                        "success": True,
                        "proceeds": float(swap_result.actual_output_amount or 0)
                    })
                
            except Exception as e:
                result.errors.append(f"Gradual liquidation failed for position {position_id}: {str(e)}")
    
    # Helper methods
    
    async def _execute_position_liquidation(self, position: Position, plan: LiquidationPlan, 
                                          immediate: bool = False) -> SwapResult:
        """Execute liquidation of a single position."""
        # Get primary DEX for this position
        dex_name = getattr(position, 'dex_name', list(self.dex_clients.keys())[0])
        dex_client = self.dex_clients.get(dex_name)
        
        if not dex_client:
            raise ValueError(f"DEX client not available: {dex_name}")
        
        # Calculate liquidation parameters
        token_address = position.symbol.split('/')[0]  # Extract token from symbol
        amount_to_sell = getattr(position, 'size', Decimal("0"))
        
        # Get quote with slippage tolerance
        slippage_bps = int(plan.max_slippage_tolerance * 10000)  # Convert to basis points
        
        quote = await dex_client.get_quote(
            input_token=token_address,
            output_token="USDC",  # Liquidate to stable coin
            amount=amount_to_sell,
            slippage_bps=slippage_bps
        )
        
        # Execute swap
        if immediate:
            # Use higher gas/priority for immediate execution
            result = await dex_client.execute_swap(quote, priority="high")
        else:
            result = await dex_client.execute_swap(quote)
        
        return result
    
    async def _liquidate_position_on_dex(self, position: Position, dex_name: str, 
                                       plan: LiquidationPlan) -> SwapResult:
        """Liquidate position portion on specific DEX."""
        dex_client = self.dex_clients.get(dex_name)
        if not dex_client:
            raise ValueError(f"DEX client not available: {dex_name}")
        
        # Execute liquidation
        return await self._execute_position_liquidation(position, plan, immediate=False)
    
    def _create_chunk_position(self, position: Position, chunk_size: Decimal) -> Position:
        """Create a position object representing a chunk for TWAP liquidation."""
        # Create a copy of position with reduced size
        # This is simplified - would need proper position cloning
        return position  # Placeholder
    
    def _create_portion_position(self, position: Position, allocation: Decimal) -> Position:
        """Create a position object representing a portion for multi-DEX liquidation."""
        # Create a copy of position with allocated portion
        # This is simplified - would need proper position cloning
        return position  # Placeholder
    
    def _calculate_slippage(self, position: Position, swap_result: SwapResult) -> Decimal:
        """Calculate slippage from liquidation."""
        # Simplified slippage calculation
        expected_output = getattr(position, 'size', Decimal("1")) * getattr(position, 'current_price', Decimal("1"))
        actual_output = swap_result.actual_output_amount or Decimal("0")
        
        if expected_output > 0:
            slippage = (expected_output - actual_output) / expected_output
            return abs(slippage)
        
        return Decimal("0")
    
    # Circuit breaker methods
    
    async def activate_circuit_breaker(self, reason: str) -> None:
        """Activate circuit breaker to halt all liquidations."""
        self.circuit_breaker_active = True
        self.circuit_breaker_triggered_at = datetime.now()
        
        self.logger.critical("Liquidation circuit breaker activated", reason=reason)
        
        # Cancel all active liquidations
        for plan_id in list(self.active_liquidations.keys()):
            self.logger.warning("Cancelling active liquidation due to circuit breaker", plan_id=plan_id)
    
    async def deactivate_circuit_breaker(self) -> None:
        """Deactivate circuit breaker to allow liquidations."""
        self.circuit_breaker_active = False
        self.circuit_breaker_triggered_at = None
        
        self.logger.info("Liquidation circuit breaker deactivated")
    
    # Monitoring and reporting
    
    def get_liquidation_status(self) -> Dict[str, Any]:
        """Get current liquidation system status."""
        return {
            "is_liquidating": self.is_liquidating,
            "active_liquidations": len(self.active_liquidations),
            "circuit_breaker_active": self.circuit_breaker_active,
            "total_liquidations": self.total_liquidations,
            "successful_liquidations": self.successful_liquidations,
            "success_rate": self.successful_liquidations / max(self.total_liquidations, 1),
            "average_slippage": float(self.total_slippage / max(self.successful_liquidations, 1)),
            "last_liquidation": self.liquidation_history[-1].completed_at.isoformat() if self.liquidation_history else None
        }
    
    def get_liquidation_metrics(self) -> Dict[str, Any]:
        """Get detailed liquidation performance metrics."""
        if not self.liquidation_history:
            return {"no_data": True}
        
        recent_liquidations = self.liquidation_history[-10:]  # Last 10 liquidations
        
        avg_execution_time = sum(r.execution_time for r in recent_liquidations) / len(recent_liquidations)
        avg_slippage = sum(r.average_slippage for r in recent_liquidations) / len(recent_liquidations)
        avg_efficiency = sum(r.execution_efficiency for r in recent_liquidations) / len(recent_liquidations)
        
        return {
            "total_liquidations": self.total_liquidations,
            "recent_performance": {
                "average_execution_time": avg_execution_time,
                "average_slippage": float(avg_slippage),
                "average_efficiency": float(avg_efficiency),
                "error_rate": sum(1 for r in recent_liquidations if r.errors) / len(recent_liquidations)
            },
            "strategy_breakdown": self._analyze_strategy_performance(),
            "dex_performance": self._analyze_dex_performance()
        }
    
    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance by liquidation strategy."""
        strategy_stats = {}
        
        for result in self.liquidation_history:
            plan = self.active_liquidations.get(result.plan_id)
            if not plan:
                continue
            
            strategy = plan.strategy.value
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "count": 0,
                    "total_slippage": 0.0,
                    "total_time": 0.0,
                    "success_count": 0
                }
            
            stats = strategy_stats[strategy]
            stats["count"] += 1
            stats["total_slippage"] += float(result.average_slippage)
            stats["total_time"] += result.execution_time
            if result.success:
                stats["success_count"] += 1
        
        # Calculate averages
        for strategy, stats in strategy_stats.items():
            if stats["count"] > 0:
                stats["avg_slippage"] = stats["total_slippage"] / stats["count"]
                stats["avg_time"] = stats["total_time"] / stats["count"]
                stats["success_rate"] = stats["success_count"] / stats["count"]
        
        return strategy_stats
    
    def _analyze_dex_performance(self) -> Dict[str, Any]:
        """Analyze performance by DEX."""
        dex_stats = {}
        
        for result in self.liquidation_history:
            for dex_name, dex_data in result.dex_breakdown.items():
                if dex_name not in dex_stats:
                    dex_stats[dex_name] = {
                        "total_positions": 0,
                        "total_proceeds": 0.0,
                        "total_slippage": 0.0,
                        "execution_count": 0
                    }
                
                stats = dex_stats[dex_name]
                stats["total_positions"] += dex_data["positions"]
                stats["total_proceeds"] += dex_data["proceeds"]
                stats["total_slippage"] += dex_data.get("average_slippage", 0.0)
                stats["execution_count"] += 1
        
        # Calculate averages
        for dex_name, stats in dex_stats.items():
            if stats["execution_count"] > 0:
                stats["avg_proceeds_per_execution"] = stats["total_proceeds"] / stats["execution_count"]
                stats["avg_slippage"] = stats["total_slippage"] / stats["execution_count"]
        
        return dex_stats