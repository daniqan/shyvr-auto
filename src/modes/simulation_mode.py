"""
Simulation mode implementation for paper trading.

This module implements a comprehensive simulation trading mode that provides
realistic paper trading capabilities with real-time market data integration,
virtual portfolio management, and risk controls.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import structlog

from src.modes.base import ModeBase, ModeConfig, ModeStatus
from src.portfolio.base import (
    Portfolio, Position, Transaction, PositionType, PositionStatus,
    TransactionType, PerformanceMetrics, RiskMetrics
)
from src.portfolio.virtual_portfolio import VirtualPortfolio
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import ExperienceReplayBuffer, ReplayBufferConfig
from src.modes.experience_collector import TradingExperienceCollector, ExperienceCollectorConfig
from src.modes.continuous_learning import ContinuousLearningEngine, ContinuousLearningConfig
from src.modes.simulation_safety import (
    SimulationSafetyManager, VirtualPortfolioProtector, SimulationSafetyConfig
)
from src.monitoring.base import MetricsRegistry
from src.monitoring.simulation_metrics import SimulationMetricsCollector, SimulationDashboard, SimulationAlertManager
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.rl_agent.base import AgentConfig, ModelType
from src.utils.base import Chain
from src.dex.base import SwapQuote, SwapResult, SwapStatus, DEXBase


logger = structlog.get_logger()


@dataclass
class SimulationConfig:
    """Configuration for simulation mode."""
    initial_balance: Decimal = Decimal("10000")
    base_currency: str = "USDC"
    enable_fees: bool = True
    slippage_bps: int = 10
    max_drawdown_pct: Decimal = Decimal("0.15")
    enable_risk_management: bool = True
    market_data_frequency_ms: int = 1000
    enable_real_time_pnl: bool = True
    enable_batch_execution: bool = True
    simulation_speed_multiplier: Decimal = Decimal("1.0")
    
    # Continuous learning configuration
    enable_continuous_learning: bool = False
    learning_trigger_threshold: int = 1000
    learning_episodes_per_session: int = 100
    enable_learning_persistence: bool = True
    learning_model_storage_path: str = "models/simulation/"
    min_learning_improvement_threshold: float = 0.02
    
    # Simulation safety configuration
    enable_simulation_safety: bool = True
    max_simulation_drawdown_pct: float = 20.0  # More lenient than live
    max_simulation_position_size_pct: float = 15.0  # More lenient than live
    enable_virtual_portfolio_protection: bool = True
    enable_simulation_circuit_breakers: bool = True
    simulation_risk_monitoring_interval: float = 2.0
    
    # Monitoring configuration
    enable_prometheus_metrics: bool = True
    enable_simulation_dashboards: bool = True
    enable_performance_alerts: bool = True
    metrics_collection_interval: float = 1.0
    enable_detailed_metrics: bool = True
    enable_real_time_monitoring: bool = True


@dataclass
class SimulationMetrics:
    """Simulation-specific performance metrics."""
    total_trades: int = 0
    virtual_balance: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    execution_latency_ms: List[float] = field(default_factory=list)
    average_slippage_bps: Decimal = Decimal("0")
    total_fees_paid: Decimal = Decimal("0")


class SimulationError(Exception):
    """Base simulation error."""
    pass


class MarketDataFeed:
    """Real-time market data feed for simulation."""
    
    def __init__(self, dex_clients: Dict[str, DEXBase], frequency_ms: int = 1000):
        self.dex_clients = dex_clients
        self.frequency_ms = frequency_ms
        self.is_active = False
        self._subscribers = []
        self._task = None
        self.logger = logger.bind(component="market_data_feed")
    
    async def start(self):
        """Start market data feed."""
        self.is_active = True
        self._task = asyncio.create_task(self._run_feed())
        self.logger.info("Market data feed started")
    
    async def stop(self):
        """Stop market data feed."""
        self.is_active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Market data feed stopped")
    
    def subscribe(self, callback):
        """Subscribe to market data updates."""
        self._subscribers.append(callback)
    
    async def _run_feed(self):
        """Run market data feed loop."""
        while self.is_active:
            try:
                # Get market data from DEX clients
                market_data = await self._fetch_market_data()
                
                # Notify subscribers
                for callback in self._subscribers:
                    try:
                        await callback(market_data)
                    except Exception as e:
                        self.logger.error("Error in market data callback", error=str(e))
                
                await asyncio.sleep(self.frequency_ms / 1000.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in market data feed", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch current market data from DEX clients."""
        market_data = {}
        
        for dex_name, dex_client in self.dex_clients.items():
            try:
                # This would fetch real market data
                # For now, return empty dict
                market_data[dex_name] = {}
            except Exception as e:
                self.logger.error(f"Error fetching data from {dex_name}", error=str(e))
        
        return market_data


@dataclass
class RiskValidationResult:
    """Result of risk validation check."""
    is_valid: bool
    reason: str = ""
    risk_level: str = "low"


class SimulationRiskManager:
    """Risk management for simulation trading."""
    
    def __init__(self, virtual_portfolio: VirtualPortfolio, config: Dict[str, Any]):
        self.virtual_portfolio = virtual_portfolio
        self.config = config
        self.logger = logger.bind(component="simulation_risk_manager")
    
    async def validate_position_size(self, token_address: str, amount_usd: Decimal) -> RiskValidationResult:
        """Validate position size against risk limits."""
        max_position_pct = Decimal(str(self.config.get("max_position_size_pct", 0.1)))
        portfolio_value = self.virtual_portfolio.equity
        max_position_value = portfolio_value * max_position_pct
        
        if amount_usd > max_position_value:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Position size exceeds maximum ({max_position_pct * 100}%)",
                risk_level="high"
            )
        
        return RiskValidationResult(is_valid=True)
    
    async def check_daily_loss_limit(self) -> RiskValidationResult:
        """Check daily loss limit."""
        max_daily_loss_pct = Decimal(str(self.config.get("max_daily_loss_pct", 0.05)))
        portfolio_value = self.virtual_portfolio.equity
        max_daily_loss = portfolio_value * max_daily_loss_pct
        
        if abs(self.virtual_portfolio.daily_pnl) > max_daily_loss:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Daily loss limit exceeded ({max_daily_loss_pct * 100}%)",
                risk_level="high"
            )
        
        return RiskValidationResult(is_valid=True)
    
    async def check_max_drawdown(self) -> RiskValidationResult:
        """Check maximum drawdown limit."""
        max_drawdown_pct = Decimal(str(self.config.get("max_drawdown_pct", 0.15)))
        current_drawdown = (self.virtual_portfolio.peak_value - self.virtual_portfolio.current_value) / self.virtual_portfolio.peak_value
        
        if current_drawdown > max_drawdown_pct:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Maximum drawdown exceeded ({max_drawdown_pct * 100}%)",
                risk_level="high"
            )
        
        return RiskValidationResult(is_valid=True)
    
    async def check_stop_losses(self) -> List[Position]:
        """Check for positions that should trigger stop losses."""
        stop_loss_pct = Decimal(str(self.config.get("stop_loss_pct", 0.08)))
        stop_loss_positions = []
        
        for position in self.virtual_portfolio.positions.values():
            if position.status == PositionStatus.OPEN:
                loss_pct = (position.entry_price - position.current_price) / position.entry_price
                if loss_pct > stop_loss_pct:
                    stop_loss_positions.append(position)
        
        return stop_loss_positions
    
    async def check_take_profits(self) -> List[Position]:
        """Check for positions that should trigger take profits."""
        take_profit_pct = Decimal(str(self.config.get("take_profit_pct", 0.4)))
        take_profit_positions = []
        
        for position in self.virtual_portfolio.positions.values():
            if position.status == PositionStatus.OPEN:
                profit_pct = (position.current_price - position.entry_price) / position.entry_price
                if profit_pct > take_profit_pct:
                    take_profit_positions.append(position)
        
        return take_profit_positions
    
    async def validate_new_position(self, token_address: str) -> RiskValidationResult:
        """Validate that a new position can be opened."""
        max_positions = self.config.get("max_open_positions", 10)
        current_positions = len([p for p in self.virtual_portfolio.positions.values() if p.status == PositionStatus.OPEN])
        
        if current_positions >= max_positions:
            return RiskValidationResult(
                is_valid=False,
                reason=f"Maximum number of positions exceeded ({max_positions})",
                risk_level="medium"
            )
        
        return RiskValidationResult(is_valid=True)
    
    async def assess_portfolio_risk(self) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment."""
        risk_factors = []
        
        # Check various risk factors
        daily_loss_check = await self.check_daily_loss_limit()
        if not daily_loss_check.is_valid:
            risk_factors.append("daily_loss_limit")
        
        drawdown_check = await self.check_max_drawdown()
        if not drawdown_check.is_valid:
            risk_factors.append("max_drawdown")
        
        stop_losses = await self.check_stop_losses()
        if stop_losses:
            risk_factors.append("stop_loss_triggers")
        
        # Determine overall risk level
        if len(risk_factors) >= 3:
            overall_risk_level = "high"
        elif len(risk_factors) >= 1:
            overall_risk_level = "medium"
        else:
            overall_risk_level = "low"
        
        return {
            "overall_risk_level": overall_risk_level,
            "risk_factors": risk_factors,
            "recommendations": self._generate_risk_recommendations(risk_factors)
        }
    
    def _generate_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if "daily_loss_limit" in risk_factors:
            recommendations.append("Consider stopping trading for the day")
        if "max_drawdown" in risk_factors:
            recommendations.append("Reduce position sizes and review strategy")
        if "stop_loss_triggers" in risk_factors:
            recommendations.append("Close positions that hit stop losses")
        
        return recommendations


class SimulationExecutor:
    """Executes paper trading operations with realistic simulation."""
    
    def __init__(self, virtual_portfolio: VirtualPortfolio, dex_clients: Dict[str, DEXBase], 
                 enable_fees: bool = True, slippage_bps: int = 10):
        self.virtual_portfolio = virtual_portfolio
        self.dex_clients = dex_clients
        self.enable_fees = enable_fees
        self.slippage_bps = slippage_bps
        self.is_active = False
        self.logger = logger.bind(component="simulation_executor")
    
    async def execute_buy_order(self, token_address: str, amount_usd: Decimal, max_slippage_bps: int = 10) -> SwapResult:
        """Execute simulated buy order."""
        try:
            start_time = datetime.now()
            
            # Get quote from DEX client
            dex_client = list(self.dex_clients.values())[0]  # Use first available DEX
            quote = await dex_client.get_quote(
                input_token="USDC",
                output_token=token_address,
                amount=amount_usd,
                slippage_bps=max_slippage_bps
            )
            
            # Apply simulation effects
            adjusted_result = await self._apply_simulation_effects(quote)
            
            # Create virtual position and update cash balance
            if adjusted_result.status == SwapStatus.CONFIRMED:
                # Deduct cost from virtual portfolio
                total_cost = amount_usd + Decimal(str(adjusted_result.transaction_data["fees"]))
                self.virtual_portfolio.spend_cash(total_cost, f"Buy {token_address}")
                
                # Create position
                await self._create_virtual_position(
                    token_address=token_address,
                    amount=adjusted_result.actual_output_amount,
                    entry_price=Decimal(str(adjusted_result.transaction_data["actual_price"]))
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(
                "Buy order executed",
                token=token_address,
                amount_usd=str(amount_usd),
                execution_time_ms=execution_time,
                status=adjusted_result.status.value
            )
            
            return adjusted_result
            
        except Exception as e:
            self.logger.error("Error executing buy order", error=str(e))
            return SwapResult(
                transaction_hash="SIMULATED_ERROR",
                status=SwapStatus.FAILED,
                input_token="USDC",
                output_token=token_address,
                input_amount=amount_usd,
                error_message=str(e)
            )
    
    async def execute_sell_order(self, token_address: str, amount: Decimal, max_slippage_bps: int = 10) -> SwapResult:
        """Execute simulated sell order."""
        try:
            start_time = datetime.now()
            
            # Get quote from DEX client
            dex_client = list(self.dex_clients.values())[0]
            quote = await dex_client.get_quote(
                input_token=token_address,
                output_token="USDC",
                amount=amount,
                slippage_bps=max_slippage_bps
            )
            
            # Apply simulation effects
            adjusted_result = await self._apply_simulation_effects(quote)
            
            # Update virtual portfolio
            if adjusted_result.status == SwapStatus.CONFIRMED:
                self.virtual_portfolio.add_cash(
                    adjusted_result.actual_output_amount - adjusted_result.fees,
                    "Token sale"
                )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(
                "Sell order executed",
                token=token_address,
                amount=str(amount),
                execution_time_ms=execution_time,
                status=adjusted_result.status.value
            )
            
            return adjusted_result
            
        except Exception as e:
            self.logger.error("Error executing sell order", error=str(e))
            return SwapResult(
                transaction_hash="SIMULATED_ERROR",
                status=SwapStatus.FAILED,
                input_token=token_address,
                output_token="USDC",
                input_amount=amount,
                error_message=str(e)
            )
    
    async def execute_batch_orders(self, orders: List[Dict[str, Any]]) -> List[SwapResult]:
        """Execute multiple orders in batch."""
        results = []
        
        for order in orders:
            if order["action"] == "buy":
                result = await self.execute_buy_order(
                    order["token"],
                    order["amount_usd"]
                )
            elif order["action"] == "sell":
                result = await self.execute_sell_order(
                    order["token"],
                    order["amount"]
                )
            else:
                result = SwapResult(
                    transaction_hash="INVALID_ACTION",
                    status=SwapStatus.FAILED,
                    input_token="",
                    output_token="",
                    input_amount=Decimal("0"),
                    error_message=f"Invalid action: {order['action']}"
                )
            
            results.append(result)
        
        return results
    
    async def update_market_prices(self, price_updates: Dict[str, Decimal]) -> None:
        """Update market prices for positions."""
        for token_address, new_price in price_updates.items():
            # Find positions with this token
            for position in self.virtual_portfolio.positions.values():
                if position.symbol.startswith(token_address.split('_')[0]):  # Simple matching
                    self.virtual_portfolio.update_position_price(position.position_id, new_price)
    
    async def _apply_simulation_effects(self, quote: SwapQuote) -> SwapResult:
        """Apply realistic simulation effects like slippage and fees."""
        # Calculate adjusted output with slippage
        slippage_factor = Decimal(1) - (Decimal(self.slippage_bps) / Decimal(10000))
        adjusted_output = quote.output_amount * slippage_factor
        
        # Calculate fees
        fees = Decimal("0")
        if self.enable_fees:
            fees = quote.input_amount * Decimal("0.003")  # 0.3% fee
        
        # Calculate actual price impact
        actual_price_impact_bps = quote.price_impact_bps + (self.slippage_bps // 2)
        
        actual_price = adjusted_output / quote.input_amount if quote.input_amount > 0 else Decimal("0")
        
        result = SwapResult(
            transaction_hash=f"SIMULATED_{uuid4().hex[:8]}",
            status=SwapStatus.CONFIRMED,
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            actual_output_amount=adjusted_output,
            timestamp=datetime.now(),
            dex_name=quote.dex_name,
            quote_used=quote,
            actual_price_impact_bps=actual_price_impact_bps,
            transaction_data={
                "is_simulated": True, 
                "fees": fees,
                "actual_price": actual_price
            }
        )
        
        return result
    
    async def _create_virtual_position(self, token_address: str, amount: Decimal, entry_price: Decimal) -> None:
        """Create virtual position in portfolio."""
        position = Position(
            position_id=uuid4(),
            symbol=f"{token_address}/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,  # Default to Solana
            dex_name="jupiter",   # Default to Jupiter
            size=amount,
            entry_price=entry_price,
            current_price=entry_price,
            status=PositionStatus.OPEN
        )
        
        self.virtual_portfolio.add_position(position)


class SimulationMode(ModeBase):
    """
    Comprehensive simulation trading mode implementation.
    
    Provides realistic paper trading with real-time market data,
    virtual portfolio management, and risk controls.
    """
    
    def __init__(self, mode_id: UUID, config: ModeConfig, portfolio: Portfolio):
        """Initialize simulation mode with enhanced simulation capabilities."""
        super().__init__(mode_id, config, portfolio)
        
        # Extract simulation parameters
        params = config.parameters
        self.virtual_balance = Decimal(str(params.get("initial_balance", 10000)))
        self.enable_fees = params.get("enable_fees", True)
        self.slippage_bps = params.get("slippage_bps", 10)
        self.max_drawdown_pct = Decimal(str(params.get("max_drawdown_pct", 0.15)))
        self.enable_risk_management = params.get("enable_risk_management", True)
        self.market_data_frequency_ms = params.get("market_data_frequency_ms", 1000)
        self.enable_real_time_pnl = params.get("enable_real_time_pnl", True)
        
        # Initialize simulation components
        self.virtual_portfolio = VirtualPortfolio(
            initial_balance=self.virtual_balance,
            enable_fees=self.enable_fees
        )
        
        self.dex_clients: Dict[str, DEXBase] = {}
        
        self.simulation_executor = SimulationExecutor(
            virtual_portfolio=self.virtual_portfolio,
            dex_clients=self.dex_clients,
            enable_fees=self.enable_fees,
            slippage_bps=self.slippage_bps
        )
        
        risk_config = {
            "max_position_size_pct": 0.1,
            "max_daily_loss_pct": 0.05,
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.4,
            "max_open_positions": 10
        }
        
        self.risk_manager = SimulationRiskManager(
            virtual_portfolio=self.virtual_portfolio,
            config=risk_config
        )
        
        self.market_data_feed = MarketDataFeed(
            dex_clients=self.dex_clients,
            frequency_ms=self.market_data_frequency_ms
        )
        
        # Simulation metrics
        self.simulation_metrics = SimulationMetrics(
            virtual_balance=self.virtual_balance
        )
        
        # Experience collection setup
        self.enable_experience_collection = params.get("enable_experience_collection", False)
        self.experience_collector = None
        self.replay_buffer = None
        
        if self.enable_experience_collection:
            # Create experience collector configuration
            experience_config = ExperienceCollectorConfig(
                buffer_size=params.get("experience_buffer_size", 10000),
                min_experience_gap_seconds=params.get("min_experience_gap_seconds", 1),
                enable_persistence=params.get("enable_experience_persistence", True),
                persistence_path=params.get("experience_persistence_path", "simulation_experiences.json"),
                max_experiences_per_session=params.get("max_experiences_per_session", 1000)
            )
            
            # Create replay buffer for experience collection
            replay_config = ReplayBufferConfig(
                max_size=experience_config.buffer_size,
                batch_size=32,
                min_size=100
            )
            self.replay_buffer = ExperienceReplayBuffer(replay_config)
            
            # Initialize experience collector
            self.experience_collector = TradingExperienceCollector(experience_config, self.replay_buffer)
            
            self.logger.info("Experience collection enabled for simulation mode",
                           buffer_size=experience_config.buffer_size,
                           persistence=experience_config.enable_persistence)
        
        # Continuous learning setup
        self.enable_continuous_learning = params.get("enable_continuous_learning", False)
        self.continuous_learning_engine = None
        self.continuous_learning_config = None
        self.dqn_agent = None
        
        if self.enable_continuous_learning:
            # Create continuous learning configuration
            self.continuous_learning_config = ContinuousLearningConfig(
                training_trigger_threshold=params.get("learning_trigger_threshold", 1000),
                training_episodes_per_session=params.get("learning_episodes_per_session", 100),
                min_improvement_threshold=params.get("min_learning_improvement_threshold", 0.02),
                enable_model_persistence=params.get("enable_learning_persistence", True),
                model_storage_path=params.get("learning_model_storage_path", "models/simulation/"),
                performance_rollback_threshold=-0.15,  # More lenient for simulation
                max_model_versions=params.get("max_learning_model_versions", 5),
                enable_incremental_learning=params.get("enable_incremental_learning", True)
            )
            
            # Initialize DQN agent for learning
            agent_config = AgentConfig(
                model_type=ModelType.DQN,
                learning_rate=0.001,
                batch_size=32,
                replay_buffer_size=10000,
                hidden_size=256,
                num_layers=3
            )
            self.dqn_agent = DQNTradingAgent(config=agent_config)
            
            # Initialize continuous learning engine if we have required components
            if self.experience_collector and self.replay_buffer:
                self.continuous_learning_engine = ContinuousLearningEngine(
                    config=self.continuous_learning_config,
                    replay_buffer=self.replay_buffer,
                    dqn_agent=self.dqn_agent,
                    experience_collector=self.experience_collector
                )
                
                self.logger.info("Continuous learning enabled for simulation mode",
                               trigger_threshold=self.continuous_learning_config.training_trigger_threshold,
                               episodes_per_session=self.continuous_learning_config.training_episodes_per_session,
                               model_storage=self.continuous_learning_config.model_storage_path)
            else:
                self.logger.warning("Continuous learning requested but experience collection not enabled")
                self.enable_continuous_learning = False
        
        # Simulation safety systems setup
        self.enable_simulation_safety = params.get("enable_simulation_safety", True)
        self.simulation_safety_manager = None
        self.virtual_portfolio_protector = None
        self.simulation_safety_config = None
        
        if self.enable_simulation_safety:
            # Create simulation safety configuration
            self.simulation_safety_config = SimulationSafetyConfig(
                max_drawdown_pct=params.get("max_simulation_drawdown_pct", 20.0),
                max_position_size_pct=params.get("max_simulation_position_size_pct", 15.0),
                max_daily_loss_pct=params.get("max_simulation_daily_loss_pct", 8.0),
                max_open_positions=params.get("max_simulation_open_positions", 15),
                enable_experimental_strategies=params.get("enable_experimental_strategies", True),
                experimental_position_limit_pct=params.get("experimental_position_limit_pct", 5.0),
                enable_simulation_circuit_breakers=params.get("enable_simulation_circuit_breakers", True),
                safety_check_interval=params.get("simulation_risk_monitoring_interval", 2.0)
            )
            
            # Initialize simulation safety manager
            self.simulation_safety_manager = SimulationSafetyManager(
                config=self.simulation_safety_config,
                virtual_portfolio=self.virtual_portfolio
            )
            
            # Initialize virtual portfolio protector if enabled
            if params.get("enable_virtual_portfolio_protection", True):
                self.virtual_portfolio_protector = VirtualPortfolioProtector(
                    config=self.simulation_safety_config
                )
            
            self.logger.info("Simulation safety systems enabled",
                           max_drawdown_pct=self.simulation_safety_config.max_drawdown_pct,
                           max_position_size_pct=self.simulation_safety_config.max_position_size_pct,
                           experimental_strategies=self.simulation_safety_config.enable_experimental_strategies)
        
        # Monitoring and metrics setup
        self.enable_prometheus_metrics = params.get("enable_prometheus_metrics", True)
        self.enable_simulation_dashboards = params.get("enable_simulation_dashboards", True)
        self.enable_performance_alerts = params.get("enable_performance_alerts", True)
        
        self.metrics_registry = None
        self.simulation_metrics_collector = None
        self.simulation_dashboard = None
        self.simulation_alert_manager = None
        
        if self.enable_prometheus_metrics:
            # Initialize metrics registry
            self.metrics_registry = MetricsRegistry()
            
            # Initialize simulation metrics collector
            self.simulation_metrics_collector = SimulationMetricsCollector(
                registry=self.metrics_registry,
                simulation_mode=self
            )
            
            # Initialize dashboard if enabled
            if self.enable_simulation_dashboards:
                dashboard_config = {
                    'dashboard_name': 'Simulation Trading Performance',
                    'refresh_interval': params.get("dashboard_refresh_interval", 5),
                    'panels': [
                        {
                            'type': 'portfolio_value_chart',
                            'title': 'Virtual Portfolio Value Over Time',
                            'metrics': ['simulation_portfolio_value'],
                            'time_range': '1h'
                        },
                        {
                            'type': 'pnl_distribution',
                            'title': 'P&L Distribution', 
                            'metrics': ['simulation_realized_pnl', 'simulation_unrealized_pnl'],
                            'chart_type': 'bar'
                        },
                        {
                            'type': 'trade_performance',
                            'title': 'Trade Performance Metrics',
                            'metrics': ['simulation_win_rate', 'simulation_sharpe_ratio'],
                            'display_type': 'gauge'
                        },
                        {
                            'type': 'risk_monitoring',
                            'title': 'Risk Metrics',
                            'metrics': ['simulation_drawdown', 'simulation_volatility'],
                            'alert_thresholds': {'drawdown': 0.15, 'volatility': 0.25}
                        }
                    ],
                    'alerts': [
                        {
                            'name': 'High Drawdown Alert',
                            'condition': 'simulation_drawdown > 0.15',
                            'severity': 'warning'
                        },
                        {
                            'name': 'Poor Win Rate Alert',
                            'condition': 'simulation_win_rate < 0.4',
                            'severity': 'warning'
                        }
                    ]
                }
                
                self.simulation_dashboard = SimulationDashboard(dashboard_config)
            
            # Initialize alert manager if enabled
            if self.enable_performance_alerts:
                alert_config = {
                    'alerts': [
                        {
                            'name': 'simulation_high_drawdown',
                            'condition': 'simulation_drawdown > 0.15',
                            'severity': 'warning',
                            'message': 'Simulation drawdown exceeds 15%',
                            'cooldown_minutes': 10
                        },
                        {
                            'name': 'simulation_poor_performance',
                            'condition': 'simulation_win_rate < 0.4 AND simulation_trades_total > 10',
                            'severity': 'warning',
                            'message': 'Simulation win rate below 40%',
                            'cooldown_minutes': 30
                        },
                        {
                            'name': 'simulation_execution_errors',
                            'condition': 'simulation_execution_errors > 5',
                            'severity': 'critical',
                            'message': 'Multiple simulation execution errors detected',
                            'cooldown_minutes': 5
                        }
                    ],
                    'notification_channels': ['console', 'metrics'],
                    'enable_alert_escalation': True
                }
                
                self.simulation_alert_manager = SimulationAlertManager(alert_config)
            
            self.logger.info("Simulation monitoring systems enabled",
                           prometheus_metrics=self.enable_prometheus_metrics,
                           dashboards=self.enable_simulation_dashboards,
                           alerts=self.enable_performance_alerts)
    
    async def initialize(self) -> None:
        """Initialize simulation mode resources."""
        self.logger.info("Initializing simulation mode")
        
        # Initialize DEX clients (would be injected in real implementation)
        await self._initialize_dex_clients()
        
        # Subscribe to market data updates
        self.market_data_feed.subscribe(self.process_market_update)
        
        self._set_status(ModeStatus.INACTIVE)
        self.logger.info("Simulation mode initialized")
    
    async def start(self) -> None:
        """Start simulation mode."""
        self.logger.info("Starting simulation mode")
        self.start_time = datetime.now()
        
        # Start market data feed
        await self.market_data_feed.start()
        
        # Activate simulation executor
        self.simulation_executor.is_active = True
        
        # Start experience collection if enabled
        if self.experience_collector:
            await self.experience_collector.start_collection()
            self.logger.info("Experience collection started")
        
        # Initialize continuous learning if enabled
        if self.continuous_learning_engine:
            # Load previous learning state if available
            state_file = f"{self.continuous_learning_config.model_storage_path}/learning_state.json"
            try:
                await self.continuous_learning_engine.load_state(state_file)
                self.logger.info("Continuous learning state loaded")
            except Exception as e:
                self.logger.info("No previous learning state found, starting fresh")
        
        self._set_status(ModeStatus.ACTIVE)
        self.logger.info("Simulation mode started")
    
    async def stop(self) -> None:
        """Stop simulation mode."""
        self.logger.info("Stopping simulation mode")
        self._set_status(ModeStatus.STOPPING)
        
        # Stop experience collection if enabled
        if self.experience_collector:
            await self.experience_collector.stop_collection()
            self.logger.info("Experience collection stopped")
        
        # Save continuous learning state if enabled
        if self.continuous_learning_engine:
            state_file = f"{self.continuous_learning_config.model_storage_path}/learning_state.json"
            try:
                await self.continuous_learning_engine.save_state(state_file)
                self.logger.info("Continuous learning state saved")
            except Exception as e:
                self.logger.error("Failed to save learning state", error=str(e))
        
        # Deactivate executor
        self.simulation_executor.is_active = False
        
        # Stop market data feed
        await self.market_data_feed.stop()
        
        self.logger.info("Simulation mode stopped")
    
    async def pause(self) -> None:
        """Pause simulation mode."""
        self.logger.info("Pausing simulation mode")
        self._set_status(ModeStatus.PAUSED)
        
        # Pause executor but keep market data active
        self.simulation_executor.is_active = False
    
    async def resume(self) -> None:
        """Resume simulation mode."""
        self.logger.info("Resuming simulation mode")
        self._set_status(ModeStatus.ACTIVE)
        
        # Resume executor
        self.simulation_executor.is_active = True
    
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process market tick and make simulated trading decisions."""
        if self.status != ModeStatus.ACTIVE:
            return None
        
        try:
            # Record market metrics
            self._record_metric("last_price", market_state.price_usd)
            self._record_metric("last_rsi", market_state.rsi)
            self._record_metric("last_volume", market_state.volume_24h)
            
            # Make trading decision based on market conditions
            action = self._make_trading_decision(market_state)
            
            # Capture pre-trade experience if not HOLD and experience collection enabled
            experience_id = None
            if action != TradeAction.HOLD and self.experience_collector and self.experience_collector.is_collecting:
                try:
                    experience_id = await self.experience_collector.capture_pre_trade_state(market_state, action)
                except Exception as e:
                    self.logger.warning("Failed to capture pre-trade experience", error=str(e))
            
            # Execute simulated trade if action is not HOLD
            if action != TradeAction.HOLD:
                await self._execute_simulated_trade(action, market_state, experience_id)
            
            # Check and trigger continuous learning if enabled
            if self.continuous_learning_engine:
                try:
                    learning_result = await self.continuous_learning_engine.check_and_trigger_training()
                    if learning_result and learning_result.get('training_triggered'):
                        self.logger.info(
                            "Continuous learning triggered",
                            session_id=learning_result.get('session_id'),
                            episodes=learning_result.get('episodes_completed'),
                            model_activated=learning_result.get('model_activated')
                        )
                        
                        # Record learning event in metrics
                        self._record_metric("learning_sessions_triggered", 1)
                        if learning_result.get('model_activated'):
                            self._record_metric("model_updates", 1)
                except Exception as e:
                    self.logger.warning("Continuous learning check failed", error=str(e))
            
            # Update simulation metrics
            self._update_simulation_metrics()
            
            # Update real-time monitoring metrics
            if self.simulation_metrics_collector:
                try:
                    await self.simulation_metrics_collector.update_real_time_metrics()
                except Exception as e:
                    self.logger.warning("Failed to update real-time metrics", error=str(e))
            
            return action
            
        except Exception as e:
            self.logger.error("Error processing market tick", error=str(e))
            return None
    
    async def cleanup(self) -> None:
        """Clean up simulation mode resources."""
        self.logger.info("Cleaning up simulation mode")
        
        # Stop market data feed
        if self.market_data_feed.is_active:
            await self.market_data_feed.stop()
        
        # Generate final performance report
        final_performance = self.virtual_portfolio.calculate_performance()
        self.logger.info(
            "Simulation completed",
            total_trades=self.simulation_metrics.total_trades,
            final_balance=str(self.virtual_portfolio.equity),
            total_pnl=str(final_performance.total_pnl),
            win_rate=str(final_performance.win_rate)
        )
    
    async def process_market_update(self, market_data: Dict[str, Any]) -> None:
        """Process real-time market data updates."""
        if not self.enable_real_time_pnl:
            return
        
        try:
            # Extract price updates from market data
            price_updates = {}
            for dex_name, dex_data in market_data.items():
                if isinstance(dex_data, dict):
                    for token, data in dex_data.items():
                        if isinstance(data, dict) and "price" in data:
                            price_updates[token] = Decimal(str(data["price"]))
            
            # Update position prices
            if price_updates:
                await self.simulation_executor.update_market_prices(price_updates)
            
        except Exception as e:
            self.logger.error("Error processing market update", error=str(e))
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        performance = self.virtual_portfolio.calculate_performance()
        risk_metrics = self.virtual_portfolio.calculate_risk_metrics()
        
        return {
            "total_trades": self.simulation_metrics.total_trades,
            "win_rate": float(performance.win_rate),
            "profit_factor": float(performance.profit_factor),
            "total_pnl": float(performance.total_pnl),
            "max_drawdown": float(performance.max_drawdown),
            "sharpe_ratio": float(performance.sharpe_ratio),
            "virtual_balance": float(self.virtual_portfolio.equity),
            "total_fees": float(performance.total_fees),
            "simulation_duration": str(datetime.now() - self.start_time) if self.start_time else "0:00:00"
        }
    
    def _make_trading_decision(self, market_state: MarketState) -> TradeAction:
        """Make trading decision based on market state."""
        # Simple RSI-based strategy for simulation
        if market_state.rsi is not None:
            if market_state.rsi < 25:  # Very oversold
                return TradeAction.STRONG_BUY
            elif market_state.rsi < 35:  # Oversold
                return TradeAction.BUY
            elif market_state.rsi > 75:  # Very overbought
                return TradeAction.STRONG_SELL
            elif market_state.rsi > 65:  # Overbought
                return TradeAction.SELL
        
        return TradeAction.HOLD
    
    async def _execute_simulated_trade(self, action: TradeAction, market_state: MarketState, 
                                       experience_id: Optional[str] = None) -> None:
        """Execute simulated trade based on action."""
        trading_result = None
        
        try:
            if action in [TradeAction.BUY, TradeAction.STRONG_BUY]:
                # Simulate buy order
                position_size = Decimal("1000")  # Fixed position size for simulation
                if action == TradeAction.STRONG_BUY:
                    position_size *= Decimal("1.5")
                
                # Check simulation safety limits first
                if self.simulation_safety_manager:
                    safety_validation = await self.simulation_safety_manager.validate_position_size(
                        market_state.token.address, position_size
                    )
                    
                    if not safety_validation.is_safe:
                        # Create failed result due to safety validation
                        trading_result = TradingResult(
                            action=action,
                            token=market_state.token,
                            executed_at=datetime.now(),
                            price=market_state.price_usd,
                            quantity=0.0,
                            value_usd=0.0,
                            success=False,
                            error_message=f"Safety validation failed: {safety_validation.message}"
                        )
                        
                        self.logger.warning("Trade blocked by simulation safety",
                                          reason=safety_validation.message,
                                          risk_level=safety_validation.risk_level.value)
                        return
                
                # Check legacy risk limits
                risk_check = await self.risk_manager.validate_position_size("SIMULATION_TOKEN", position_size)
                if risk_check.is_valid:
                    # Create simulated buy result
                    trading_result = TradingResult(
                        action=action,
                        token=market_state.token,
                        executed_at=datetime.now(),
                        price=market_state.price_usd,
                        quantity=float(position_size),
                        value_usd=float(position_size * Decimal(str(market_state.price_usd))),
                        success=True,
                        slippage=0.01,  # 1% simulated slippage
                        fees=float(position_size * Decimal("0.003")),  # 0.3% fees
                        portfolio_value_before=float(self.virtual_portfolio.equity),
                        portfolio_value_after=float(self.virtual_portfolio.equity),
                        cash_change=float(-position_size * Decimal(str(market_state.price_usd))),
                        position_change=float(position_size),
                        realized_pnl=0.0,
                        unrealized_pnl=0.0
                    )
                else:
                    # Risk check failed - create failed result
                    trading_result = TradingResult(
                        action=action,
                        token=market_state.token,
                        executed_at=datetime.now(),
                        price=market_state.price_usd,
                        quantity=0.0,
                        value_usd=0.0,
                        success=False,
                        error_message=risk_check.reason
                    )
            
            elif action in [TradeAction.SELL, TradeAction.STRONG_SELL]:
                # Simulate sell order for existing positions
                open_positions = [p for p in self.virtual_portfolio.positions.values() if p.status == PositionStatus.OPEN]
                if open_positions:
                    position = open_positions[0]  # Sell first position
                    sell_quantity = float(position.size)
                    
                    # Create simulated sell result
                    trading_result = TradingResult(
                        action=action,
                        token=market_state.token,
                        executed_at=datetime.now(),
                        price=market_state.price_usd,
                        quantity=sell_quantity,
                        value_usd=sell_quantity * market_state.price_usd,
                        success=True,
                        slippage=0.01,
                        fees=sell_quantity * market_state.price_usd * 0.003,
                        portfolio_value_before=float(self.virtual_portfolio.equity),
                        portfolio_value_after=float(self.virtual_portfolio.equity),
                        cash_change=sell_quantity * market_state.price_usd,
                        position_change=-sell_quantity,
                        realized_pnl=(market_state.price_usd - float(position.entry_price)) * sell_quantity,
                        unrealized_pnl=0.0
                    )
                else:
                    # No positions to sell
                    trading_result = TradingResult(
                        action=action,
                        token=market_state.token,
                        executed_at=datetime.now(),
                        price=market_state.price_usd,
                        quantity=0.0,
                        value_usd=0.0,
                        success=False,
                        error_message="No positions to sell"
                    )
            
            # Capture post-trade experience if experience collection enabled
            if experience_id and self.experience_collector and trading_result:
                try:
                    await self._capture_trade_experience(experience_id, trading_result, market_state)
                except Exception as e:
                    self.logger.warning("Failed to capture post-trade experience", error=str(e))
            
            # Update trade counter
            self.simulation_metrics.total_trades += 1
            self._record_metric("simulated_trades", self.simulation_metrics.total_trades)
            
        except Exception as e:
            self.logger.error("Error executing simulated trade", error=str(e))
    
    async def _capture_trade_experience(self, experience_id: str, trading_result: TradingResult, 
                                       market_state: MarketState) -> None:
        """Capture trading experience for RL training"""
        if self.experience_collector:
            await self.experience_collector.capture_post_trade_result(
                experience_id, trading_result, market_state
            )
    
    async def _flush_experiences_to_buffer(self) -> int:
        """Flush completed experiences to replay buffer"""
        if self.experience_collector:
            return await self.experience_collector.add_experiences_to_buffer()
        return 0
    
    async def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics enhanced with experience collection and learning data"""
        stats = {
            "simulation_metrics": {
                "total_trades": self.simulation_metrics.total_trades,
                "virtual_balance": float(self.simulation_metrics.virtual_balance),
                "unrealized_pnl": float(self.simulation_metrics.unrealized_pnl),
                "realized_pnl": float(self.simulation_metrics.realized_pnl)
            }
        }
        
        # Add experience collection statistics if enabled
        if self.experience_collector:
            try:
                experience_stats = await self.experience_collector.get_statistics()
                stats["experience_collection"] = experience_stats
            except Exception as e:
                self.logger.warning("Failed to get experience statistics", error=str(e))
        
        # Add continuous learning statistics if enabled
        if self.continuous_learning_engine:
            try:
                learning_stats = await self.continuous_learning_engine.get_performance_statistics()
                stats["continuous_learning"] = learning_stats
            except Exception as e:
                self.logger.warning("Failed to get learning statistics", error=str(e))
        
        return stats
    
    async def get_simulation_safety_metrics(self) -> Dict[str, Any]:
        """Get comprehensive simulation safety metrics."""
        if not self.simulation_safety_manager:
            return {"safety_disabled": True}
        
        return await self.simulation_safety_manager.get_simulation_safety_metrics()
    
    async def get_simulation_risk_metrics(self) -> Dict[str, Any]:
        """Get simulation-specific risk metrics."""
        if not self.simulation_safety_manager:
            return {"safety_disabled": True}
        
        return await self.simulation_safety_manager.get_simulation_risk_metrics()
    
    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get dashboard-ready metrics for simulation monitoring."""
        if not self.simulation_dashboard:
            return {"dashboard_disabled": True}
        
        try:
            # Prepare current metrics for dashboard
            current_metrics = {
                'current_portfolio_value': float(self.virtual_portfolio.equity),
                'total_pnl': float(self.virtual_portfolio.equity - self.virtual_portfolio.initial_balance),
                'unrealized_pnl': float(self.virtual_portfolio.total_unrealized_pnl),
                'win_rate': float(self.simulation_metrics.win_rate),
                'total_trades': self.simulation_metrics.total_trades,
                'drawdown': self._calculate_current_drawdown(),
                'sharpe_ratio': float(self.simulation_metrics.sharpe_ratio),
                'active_positions': len([p for p in self.virtual_portfolio.positions.values() if p.status.value == "OPEN"])
            }
            
            # Add performance charts data
            performance_charts = {
                'portfolio_value_history': [float(self.virtual_portfolio.equity)],  # Would track over time
                'pnl_history': [float(self.virtual_portfolio.total_unrealized_pnl)],
                'trade_distribution': {
                    'wins': max(0, int(self.simulation_metrics.total_trades * float(self.simulation_metrics.win_rate))),
                    'losses': max(0, self.simulation_metrics.total_trades - int(self.simulation_metrics.total_trades * float(self.simulation_metrics.win_rate)))
                }
            }
            
            # Add risk metrics
            risk_metrics = {
                'current_drawdown_pct': self._calculate_current_drawdown(),
                'max_position_risk_pct': self._get_max_position_risk(),
                'portfolio_volatility': 0.15,  # Placeholder
                'risk_score': self._calculate_risk_score()
            }
            
            # Add trading statistics
            trade_statistics = {
                'total_volume': float(self.simulation_metrics.total_trades * 1000),  # Placeholder calculation
                'avg_trade_size': 1000.0,  # Placeholder
                'avg_holding_period': "2.5 hours",  # Placeholder
                'success_rate': float(self.simulation_metrics.win_rate)
            }
            
            return {
                'current_portfolio_value': current_metrics['current_portfolio_value'],
                'total_pnl': current_metrics['total_pnl'],
                'trade_statistics': trade_statistics,
                'risk_metrics': risk_metrics,
                'performance_charts': performance_charts,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error generating dashboard metrics", error=str(e))
            return {"error": str(e)}
    
    def _update_simulation_metrics(self) -> None:
        """Update simulation performance metrics."""
        self.simulation_metrics.virtual_balance = self.virtual_portfolio.equity
        self.simulation_metrics.unrealized_pnl = self.virtual_portfolio.total_unrealized_pnl
        
        # Update metrics in mode base
        self._record_metric("virtual_balance", self.simulation_metrics.virtual_balance)
        self._record_metric("unrealized_pnl", self.simulation_metrics.unrealized_pnl)
        self._record_metric("simulated_action", "processed")
    
    async def _initialize_dex_clients(self) -> None:
        """Initialize DEX clients for market data."""
        # This would initialize real DEX clients in production
        # For now, create mock clients
        self.logger.info("DEX clients would be initialized here")
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current portfolio drawdown percentage."""
        try:
            if self.virtual_portfolio.peak_value <= 0:
                return 0.0
            
            current_value = self.virtual_portfolio.equity
            peak_value = self.virtual_portfolio.peak_value
            
            drawdown = ((peak_value - current_value) / peak_value) * 100
            return float(drawdown)
            
        except Exception:
            return 0.0
    
    def _get_max_position_risk(self) -> float:
        """Get maximum position risk percentage."""
        try:
            if not self.virtual_portfolio.positions:
                return 0.0
            
            portfolio_value = float(self.virtual_portfolio.equity)
            if portfolio_value <= 0:
                return 0.0
            
            max_position_value = 0.0
            for position in self.virtual_portfolio.positions.values():
                if position.status.value == "OPEN":
                    position_value = float(position.quantity * position.current_price)
                    max_position_value = max(max_position_value, position_value)
            
            return (max_position_value / portfolio_value) * 100
            
        except Exception:
            return 0.0
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall portfolio risk score."""
        try:
            risk_score = 0.0
            
            # Add drawdown risk
            drawdown = self._calculate_current_drawdown()
            risk_score += drawdown * 2  # 2 points per % drawdown
            
            # Add concentration risk
            max_position_risk = self._get_max_position_risk()
            if max_position_risk > 20:  # 20% concentration threshold
                risk_score += (max_position_risk - 20) * 1.5
            
            # Add active position count risk
            active_positions = len([p for p in self.virtual_portfolio.positions.values() if p.status.value == "OPEN"])
            max_positions = 15  # Simulation limit
            if active_positions > max_positions * 0.8:  # 80% of limit
                risk_score += (active_positions - max_positions * 0.8) * 5
            
            return min(100.0, max(0.0, risk_score))
            
        except Exception:
            return 50.0  # Default moderate risk
        
    def get_result(self):
        """Get simulation mode result with enhanced metrics."""
        result = super().get_result()
        
        # Add simulation-specific metadata
        result.metadata.update({
            "virtual_balance": float(self.virtual_portfolio.equity),
            "total_trades": self.simulation_metrics.total_trades,
            "simulation_duration": str(datetime.now() - self.start_time) if self.start_time else "0:00:00",
            "enable_fees": self.enable_fees,
            "slippage_bps": self.slippage_bps,
            "continuous_learning_enabled": self.enable_continuous_learning,
            "experience_collection_enabled": self.enable_experience_collection
        })
        
        # Add learning-specific metadata if enabled
        if self.continuous_learning_engine and self.continuous_learning_engine.active_model_version:
            result.metadata.update({
                "active_model_version": self.continuous_learning_engine.active_model_version.version_id,
                "total_learning_sessions": len(self.continuous_learning_engine.training_history),
                "learning_model_storage": self.continuous_learning_config.model_storage_path
            })
        
        return result