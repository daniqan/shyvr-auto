"""
Base classes and data structures for trading modes.

This module defines the core data structures for mode management,
including ModeBase abstract class, ModeType enum, and concrete mode
implementations for analysis, simulation, and live trading.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
import structlog
import itertools
import random
from decimal import Decimal

from src.portfolio.base import Portfolio
from src.rl_agent.base import TradeAction, MarketState
from src.discovery.base import DiscoveredToken


logger = structlog.get_logger()

# Import preservation components (lazy import to avoid circular dependencies)
try:
    from src.model_preservation.manager import PreservationManager
    from src.model_preservation.base import PreservationPriority, ModelMetadata
    PRESERVATION_AVAILABLE = True
except ImportError:
    PreservationManager = None
    PreservationPriority = None
    ModelMetadata = None
    PRESERVATION_AVAILABLE = False


class BacktestEngine:
    """Engine for running strategy backtests."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize backtest engine with configuration."""
        self.config = config
        self.start_date = config.get("start_date")
        self.end_date = config.get("end_date")
        self.initial_balance = config.get("initial_balance", Decimal("10000"))
        self.strategy_parameters = config.get("strategy_parameters", {})
        self.transaction_costs = config.get("transaction_costs", {})
        
    async def run_backtest(self, strategy_name: str, tokens: List[str]) -> Dict[str, Any]:
        """Run a backtest for a specific strategy."""
        # Simulate backtest execution with realistic results
        total_days = (self.end_date - self.start_date).days
        daily_returns = [random.uniform(-0.05, 0.08) for _ in range(total_days)]
        
        # Calculate performance metrics
        total_return = sum(daily_returns)
        annual_return = total_return * (365 / total_days) if total_days > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = []
        cumulative = 0
        for ret in daily_returns:
            cumulative += ret
            cumulative_returns.append(cumulative)
        
        peak = cumulative_returns[0]
        max_drawdown = 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / (1 + peak) if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        avg_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        volatility = (sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5 if daily_returns else 0
        sharpe_ratio = (avg_return * 365 ** 0.5) / (volatility * 365 ** 0.5) if volatility != 0 else 0
        
        # Generate some mock trades
        num_trades = random.randint(10, 50)
        winning_trades = random.randint(int(num_trades * 0.4), int(num_trades * 0.7))
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        profit_factor = random.uniform(1.1, 2.5)
        
        trade_history = []
        for i in range(num_trades):
            trade_history.append({
                "timestamp": self.start_date + timedelta(days=random.randint(0, total_days)),
                "symbol": random.choice(tokens),
                "side": random.choice(["buy", "sell"]),
                "size": random.uniform(0.05, 0.2),
                "price": random.uniform(1000, 50000),
                "pnl": random.uniform(-200, 500)
            })
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "total_trades": num_trades,
            "profit_factor": profit_factor,
            "trade_history": trade_history,
            "strategy_name": strategy_name,
            "tokens": tokens,
            "backtest_period": f"{self.start_date.date()} to {self.end_date.date()}"
        }
    
    async def get_performance_metrics(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and calculate additional performance metrics."""
        return {
            "return_metrics": {
                "total_return": backtest_result["total_return"],
                "annual_return": backtest_result["annual_return"],
                "monthly_return": backtest_result["annual_return"] / 12,
            },
            "risk_metrics": {
                "max_drawdown": backtest_result["max_drawdown"],
                "sharpe_ratio": backtest_result["sharpe_ratio"],
                "volatility": abs(backtest_result["total_return"]) * 1.5,  # Simplified
            },
            "trade_metrics": {
                "total_trades": backtest_result["total_trades"],
                "win_rate": backtest_result["win_rate"],
                "profit_factor": backtest_result["profit_factor"],
                "avg_trade_return": backtest_result["total_return"] / backtest_result["total_trades"] if backtest_result["total_trades"] > 0 else 0,
            }
        }


class ModeType(Enum):
    """Type of trading mode."""
    ANALYSIS = "analysis"           # Analysis-only mode (no trading)
    SIMULATION = "simulation"       # Paper trading simulation
    LIVE_TRADING = "live_trading"   # Live trading with real funds
    BACKTESTING = "backtesting"     # Historical backtesting
    PAPER_TRADING = "paper_trading" # Paper trading with real market data


class ModeStatus(Enum):
    """Status of a trading mode."""
    INACTIVE = "inactive"           # Mode is not running
    ACTIVE = "active"               # Mode is actively running
    PAUSED = "paused"               # Mode is paused
    ERROR = "error"                 # Mode encountered an error
    STOPPING = "stopping"           # Mode is in the process of stopping


@dataclass
class ModeConfig:
    """Configuration for trading modes."""
    mode_type: ModeType
    enabled: bool
    auto_start: bool = False
    max_runtime_minutes: Optional[int] = None
    stop_on_error: bool = True
    log_level: str = "INFO"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_runtime_minutes is not None and self.max_runtime_minutes <= 0:
            raise ValueError("max_runtime_minutes must be positive")
        
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}")


@dataclass
class ModeResult:
    """Result data structure for mode execution."""
    mode_id: UUID
    mode_type: ModeType
    status: ModeStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate execution duration."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class ModeBase(ABC):
    """
    Abstract base class for all trading modes.
    
    Defines the common interface and lifecycle management for different
    trading modes (analysis, simulation, live trading, etc.).
    """
    
    def __init__(self, mode_id: UUID, config: ModeConfig, portfolio: Portfolio):
        """Initialize mode with configuration and portfolio."""
        self.mode_id = mode_id
        self.config = config
        self.portfolio = portfolio
        self.status = ModeStatus.INACTIVE
        self.start_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Initialize preservation manager (will be set up in _initialize_preservation_manager)
        self.preservation_manager: Optional[Any] = None
        self._ml_models: Dict[str, Any] = {}
        self._required_model_types: List[str] = []
        
        # Sentiment tracking for fear/greed integration
        self.current_sentiment_regime: Optional[str] = None
        self.sentiment_confidence: float = 0.0
        self.last_sentiment_update: Optional[datetime] = None
        self.sentiment_history: List[Dict[str, Any]] = []
        
        # Configure logger
        self.logger = logger.bind(
            mode_id=str(mode_id),
            mode_type=config.mode_type.value
        )
    
    # Preservation methods
    async def _initialize_preservation_manager(self) -> None:
        """Initialize preservation manager if enabled in configuration."""
        if not PRESERVATION_AVAILABLE:
            self.logger.warning("Preservation system not available")
            return
        
        preservation_config = self.config.parameters.get('preservation', {})
        if not preservation_config.get('enabled', False):
            self.logger.debug("Preservation disabled in configuration")
            return
        
        try:
            # Create preservation manager with mode-specific configuration
            from src.model_preservation.manager import PreservationConfig
            
            config = PreservationConfig(
                gcs_bucket=preservation_config.get('gcs_bucket', 'shyvr-models-dev'),
                backup_interval_hours=preservation_config.get('backup_interval_hours', 6.0),
                max_versions_per_model=preservation_config.get('max_versions_per_model', 10),
                enable_compression=preservation_config.get('enable_compression', True),
                mode_isolation=preservation_config.get('mode_isolation', True),
                auto_backup=preservation_config.get('auto_backup_on_change', True),
                emergency_backup=preservation_config.get('emergency_backup', True)
            )
            
            self.preservation_manager = PreservationManager(config)
            self.logger.info("Preservation manager initialized", mode=self.config.mode_type.value)
            
        except Exception as e:
            self.logger.error("Failed to initialize preservation manager", error=str(e))
            self.preservation_manager = None
    
    async def _backup_models_on_mode_change(self) -> List[str]:
        """Backup all ML models when mode is changing."""
        if not self.preservation_manager or not self._ml_models:
            return []
        
        backup_ids = []
        start_time = datetime.now()
        
        try:
            # Get preservation priority based on mode type
            priority = self._get_preservation_priority()
            
            for model_name, model in self._ml_models.items():
                try:
                    # Create metadata for the model backup
                    from src.model_preservation.base import generate_model_id
                    model_type_str = f"mode_{self.config.mode_type.value}_{model_name}"
                    version_str = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    model_id = generate_model_id(model_type_str, version_str)
                    
                    metadata = ModelMetadata(
                        model_id=model_id,
                        model_type=model_type_str,
                        version=version_str,
                        created_at=datetime.now(),
                        preservation_priority=priority,
                        tags=["mode_change", "backup", self.config.mode_type.value],
                        mode=self.config.mode_type.value
                    )
                    
                    # Save the model with mode context
                    model_id = await self.preservation_manager.save_model(
                        model_type=f"mode_{self.config.mode_type.value}_{model_name}",
                        model_data=self._serialize_model(model),
                        metadata=metadata,
                        mode=self.config.mode_type.value,
                        priority=priority,
                        tags=["mode_change", "backup"]
                    )
                    
                    backup_ids.append(model_id)
                    self.logger.debug("Model backed up", model_name=model_name, model_id=model_id)
                    
                except Exception as e:
                    self.logger.error("Failed to backup model", model_name=model_name, error=str(e))
        
        except Exception as e:
            self.logger.error("Failed to backup models on mode change", error=str(e))
            return []
        
        # Record performance metrics
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._record_metric("preservation_backup_count", len(backup_ids))
        self._record_metric("preservation_backup_duration_ms", duration_ms)
        self._record_metric("preservation_last_backup", datetime.now().isoformat())
        
        return backup_ids
    
    async def _load_mode_specific_models(self, mode_filter: Optional[str] = None) -> Dict[str, Any]:
        """Load models specific to this mode from preservation storage."""
        if not self.preservation_manager:
            return {}
        
        # Apply mode isolation if enabled
        target_mode = mode_filter or self.config.mode_type.value
        if self.config.parameters.get('preservation', {}).get('mode_isolation', True):
            if mode_filter and mode_filter != self.config.mode_type.value:
                from src.model_preservation.base import PreservationError
                raise PreservationError("Access denied: Mode isolation prevents cross-mode access")
        
        loaded_models = {}
        
        try:
            for model_type in self._required_model_types:
                try:
                    model_key = f"mode_{target_mode}_{model_type}"
                    
                    # Try to load the model with fallback
                    try:
                        model_data, metadata = await self.preservation_manager.load_model(
                            model_type=model_key,
                            mode=target_mode,
                            tags=["resume"] if mode_filter else ["startup"]
                        )
                    except Exception as primary_error:
                        # Fallback attempt
                        self.logger.warning("Primary model load failed, trying fallback", 
                                          model_type=model_type, error=str(primary_error))
                        model_data, metadata = await self.preservation_manager.load_model(
                            model_type=model_key,
                            mode=target_mode,
                            fallback=True,
                            tags=["resume"] if mode_filter else ["startup"]
                        )
                    
                    # Deserialize the model
                    model = self._deserialize_model(model_data)
                    loaded_models[model_type] = model
                    
                    self.logger.debug("Model loaded", model_type=model_type, 
                                    model_id=metadata.get('model_id'))
                    
                except Exception as e:
                    self.logger.warning("Failed to load model", model_type=model_type, error=str(e))
                    # Continue loading other models
        
        except Exception as e:
            self.logger.error("Failed to load mode-specific models", error=str(e))
        
        return loaded_models
    
    def _get_preservation_priority(self) -> Any:
        """Get preservation priority based on mode type and configuration."""
        if not PRESERVATION_AVAILABLE:
            return None
        
        # Check if mode-specific priority is configured
        preservation_config = self.config.parameters.get('preservation', {})
        priority_str = preservation_config.get('priority', '').lower()
        
        # Map string priorities to enum values
        priority_mapping = {
            'critical': PreservationPriority.CRITICAL,
            'high': PreservationPriority.HIGH,
            'normal': PreservationPriority.NORMAL,
            'low': PreservationPriority.LOW
        }
        
        # Live trading modes should use critical priority by default
        if self.config.mode_type == ModeType.LIVE_TRADING:
            return priority_mapping.get(priority_str, PreservationPriority.CRITICAL)
        elif self.config.mode_type == ModeType.SIMULATION:
            return priority_mapping.get(priority_str, PreservationPriority.HIGH)
        else:
            return priority_mapping.get(priority_str, PreservationPriority.NORMAL)
    
    def _serialize_model(self, model: Any) -> bytes:
        """Serialize a model for storage."""
        import pickle
        try:
            return pickle.dumps(model)
        except Exception as e:
            self.logger.error("Failed to serialize model", error=str(e))
            # Return a placeholder for failed serialization
            return pickle.dumps({"error": "serialization_failed", "timestamp": datetime.now()})
    
    def _deserialize_model(self, model_data: bytes) -> Any:
        """Deserialize a model from storage."""
        import pickle
        try:
            return pickle.loads(model_data)
        except Exception as e:
            self.logger.error("Failed to deserialize model", error=str(e))
            return None
    
    # Additional preservation utility methods
    async def backup_on_mode_change(self, new_mode_type: ModeType) -> None:
        """Public method to backup models when changing modes."""
        self.logger.info("Backing up models for mode change", 
                        from_mode=self.config.mode_type.value, 
                        to_mode=new_mode_type.value)
        
        backup_ids = await self._backup_models_on_mode_change()
        self.logger.info("Mode change backup completed", backup_count=len(backup_ids))
    
    async def load_mode_models(self) -> Dict[str, Any]:
        """Public method to load mode-specific models."""
        return await self._load_mode_specific_models()
    
    async def _migrate_model_to_mode(self, model_type: str, version: str, 
                                   from_mode: str, to_mode: str) -> str:
        """Migrate a model from one mode to another."""
        if not self.preservation_manager:
            raise RuntimeError("Preservation manager not available")
        
        return await self.preservation_manager.migrate_model(
            model_type=model_type,
            version=version,
            from_mode=from_mode,
            to_mode=to_mode
        )
    
    async def _validate_migration_compatibility(self, model_type: str, 
                                              from_mode: str, to_mode: str) -> bool:
        """Validate if a model migration is compatible."""
        # Define incompatible migration patterns
        incompatible_migrations = [
            ('live_trading', 'analysis'),  # Don't migrate live models to analysis
        ]
        
        return (from_mode, to_mode) not in incompatible_migrations
    
    async def _migrate_models_bulk(self, models: List[Dict[str, str]], 
                                 from_mode: str, to_mode: str,
                                 rollback_on_failure: bool = False) -> List[Dict[str, Any]]:
        """Migrate multiple models between modes."""
        if not self.preservation_manager:
            raise RuntimeError("Preservation manager not available")
        
        migration_results = []
        successful_migrations = []
        
        try:
            for model_info in models:
                model_type = model_info['model_type']
                version = model_info['version']
                
                # Validate migration compatibility
                if not await self._validate_migration_compatibility(model_type, from_mode, to_mode):
                    raise Exception(f"Migration not compatible: {model_type} from {from_mode} to {to_mode}")
                
                migrated_id = await self.preservation_manager.migrate_model(
                    model_type=model_type,
                    version=version,
                    from_mode=from_mode,
                    to_mode=to_mode
                )
                
                result = {
                    'model_type': model_type,
                    'version': version,
                    'migrated_id': migrated_id,
                    'status': 'success'
                }
                
                migration_results.append(result)
                successful_migrations.append(result)
        
        except Exception as e:
            if rollback_on_failure and successful_migrations:
                # Implement rollback logic here
                self.logger.error("Migration failed, rollback required", error=str(e))
            raise
        
        return migration_results
    
    async def _handle_error(self, error_message: str) -> None:
        """Handle critical errors with emergency backup."""
        self.logger.error("Critical error occurred", error=error_message)
        self._set_status(ModeStatus.ERROR, error_message)
        
        # Trigger emergency backup if preservation manager is available
        if self.preservation_manager and self._ml_models:
            try:
                await self.preservation_manager.emergency_backup(
                    models=self._ml_models,
                    context=f"emergency_{self.config.mode_type.value}",
                    error_context=error_message
                )
                self.logger.info("Emergency backup completed")
            except Exception as backup_error:
                self.logger.error("Emergency backup failed", error=str(backup_error))
    
    def _get_effective_preservation_config(self) -> Dict[str, Any]:
        """Get effective preservation configuration with mode-specific overrides."""
        base_config = self.config.parameters.get('preservation', {})
        
        # Apply mode-specific overrides
        if self.config.mode_type == ModeType.LIVE_TRADING:
            base_config = {
                **base_config,
                'priority': 'critical',
                'mode_isolation': True
            }
        elif self.config.mode_type == ModeType.SIMULATION:
            base_config = {
                **base_config,
                'priority': base_config.get('priority', 'high'),
                'mode_isolation': True
            }
        
        return base_config
    
    async def _validate_isolation_boundary(self, requested_mode: str, operation: str) -> bool:
        """Validate isolation boundary access."""
        if not self.config.parameters.get('preservation', {}).get('mode_isolation', True):
            return True  # No isolation enforced
        
        # Allow same mode access
        if requested_mode == self.config.mode_type.value:
            return True
        
        # Strict isolation for live trading
        if (self.config.mode_type == ModeType.LIVE_TRADING and 
            self.config.parameters.get('preservation', {}).get('strict_isolation', False)):
            return False
        
        return True
    
    async def emergency_backup(self) -> None:
        """Perform emergency backup of current models."""
        if self.preservation_manager and self._ml_models:
            await self.preservation_manager.emergency_backup(
                models=self._ml_models,
                context=f"emergency_{self.config.mode_type.value}"
            )
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize mode-specific resources."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the trading mode."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the trading mode."""
        pass
    
    @abstractmethod
    async def pause(self) -> None:
        """Pause the trading mode."""
        pass
    
    @abstractmethod
    async def resume(self) -> None:
        """Resume the trading mode from paused state."""
        pass
    
    @abstractmethod
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process a market tick and return trading action if applicable."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up mode resources."""
        pass
    
    def get_result(self) -> ModeResult:
        """Get current mode execution result."""
        return ModeResult(
            mode_id=self.mode_id,
            mode_type=self.config.mode_type,
            status=self.status,
            start_time=self.start_time or datetime.now(),
            end_time=datetime.now() if self.status == ModeStatus.INACTIVE else None,
            error_message=self.error_message,
            metrics=self.metrics.copy(),
            metadata=self.metadata.copy()
        )
    
    def _set_status(self, status: ModeStatus, error_message: Optional[str] = None) -> None:
        """Internal method to update mode status."""
        self.status = status
        self.error_message = error_message
        
        self.logger.info(
            "Mode status changed",
            old_status=self.status.value if hasattr(self, '_previous_status') else None,
            new_status=status.value,
            error_message=error_message
        )
    
    def update_sentiment_state(self, sentiment_regime: str, confidence: float, 
                              fear_greed_value: float) -> None:
        """
        Update the current sentiment state for the mode
        
        Args:
            sentiment_regime: Current sentiment regime (extreme_fear, fear, etc.)
            confidence: Confidence in the sentiment classification (0.0-1.0)
            fear_greed_value: Raw fear/greed index value (0-100)
        """
        current_time = datetime.now()
        
        # Update current state
        previous_regime = self.current_sentiment_regime
        self.current_sentiment_regime = sentiment_regime
        self.sentiment_confidence = confidence
        self.last_sentiment_update = current_time
        
        # Add to history
        sentiment_record = {
            "timestamp": current_time.isoformat(),
            "regime": sentiment_regime,
            "confidence": confidence,
            "fear_greed_value": fear_greed_value,
            "previous_regime": previous_regime
        }
        
        self.sentiment_history.append(sentiment_record)
        
        # Keep only last 24 hours of sentiment history
        cutoff_time = current_time - timedelta(hours=24)
        self.sentiment_history = [
            record for record in self.sentiment_history
            if datetime.fromisoformat(record["timestamp"]) > cutoff_time
        ]
        
        # Log sentiment change if regime changed
        if previous_regime != sentiment_regime:
            self.logger.info(
                "Sentiment regime changed",
                previous_regime=previous_regime,
                new_regime=sentiment_regime,
                confidence=confidence,
                fear_greed_value=fear_greed_value
            )
        
        # Update metrics
        self.metrics[f'sentiment_regime'] = sentiment_regime
        self.metrics[f'sentiment_confidence'] = confidence
        self.metrics[f'fear_greed_value'] = fear_greed_value
    
    def get_sentiment_state(self) -> Dict[str, Any]:
        """Get current sentiment state information"""
        return {
            "current_regime": self.current_sentiment_regime,
            "confidence": self.sentiment_confidence,
            "last_update": self.last_sentiment_update.isoformat() if self.last_sentiment_update else None,
            "history_count": len(self.sentiment_history),
            "recent_regimes": [
                record["regime"] for record in self.sentiment_history[-10:]
            ] if self.sentiment_history else []
        }
    
    def _record_metric(self, key: str, value: Any) -> None:
        """Record a performance metric."""
        self.metrics[key] = value
        self.logger.debug("Metric recorded", metric=key, value=value)


class TradingMode(ModeBase):
    """
    Concrete implementation for live trading mode.
    
    Executes real trades using the RL agent and portfolio management.
    """
    
    def __init__(self, mode_id: UUID, config: ModeConfig, portfolio: Portfolio):
        """Initialize trading mode with required models."""
        super().__init__(mode_id, config, portfolio)
        
        # Define required model types for live trading including all transformer variants
        self._required_model_types = [
            'dqn', 'lstm', 'transformer', 'itransformer', 'patchtst', 
            'timesmixer', 'timesfm', 'risk_model', 'portfolio_optimizer'
        ]
    
    async def initialize(self) -> None:
        """Initialize trading mode resources."""
        self.logger.info("Initializing trading mode")
        
        # Initialize preservation manager first
        await self._initialize_preservation_manager()
        
        # Load mode-specific models
        if self.preservation_manager:
            self._ml_models = await self._load_mode_specific_models()
        
        # Initialize any trading-specific resources
        self._set_status(ModeStatus.INACTIVE)
    
    async def start(self) -> None:
        """Start live trading mode."""
        self.logger.info("Starting trading mode")
        
        # Load models if not already loaded
        if self.preservation_manager and not self._ml_models:
            self._ml_models = await self._load_mode_specific_models()
        
        self.start_time = datetime.now()
        self._set_status(ModeStatus.ACTIVE)
    
    async def stop(self) -> None:
        """Stop live trading mode."""
        self.logger.info("Stopping trading mode")
        
        # Backup models before stopping
        if self.preservation_manager and self._ml_models:
            await self._backup_models_on_mode_change()
        
        self._set_status(ModeStatus.STOPPING)
    
    async def pause(self) -> None:
        """Pause live trading mode."""
        self.logger.info("Pausing trading mode")
        self._set_status(ModeStatus.PAUSED)
    
    async def resume(self) -> None:
        """Resume live trading mode."""
        self.logger.info("Resuming trading mode")
        self._set_status(ModeStatus.ACTIVE)
    
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process market tick and execute trades if conditions are met."""
        if self.status != ModeStatus.ACTIVE:
            return None
        
        # In a real implementation, this would:
        # 1. Analyze market state using ML/RL models
        # 2. Make trading decisions based on portfolio state
        # 3. Execute trades through DEX clients
        # 4. Update portfolio positions
        
        # For now, return a simple action based on RSI
        if market_state.rsi is not None:
            if market_state.rsi < 30:  # Oversold
                return TradeAction.BUY
            elif market_state.rsi > 70:  # Overbought
                return TradeAction.SELL
        
        return TradeAction.HOLD
    
    async def cleanup(self) -> None:
        """Clean up trading mode resources."""
        self.logger.info("Cleaning up trading mode")
        # Close any open connections, save state, etc.


class AnalysisMode(ModeBase):
    """
    Concrete implementation for analysis-only mode.
    
    Performs market analysis without executing trades.
    """
    
    def __init__(self, mode_id: UUID, config: ModeConfig, portfolio: Portfolio):
        """Initialize analysis mode with required models."""
        super().__init__(mode_id, config, portfolio)
        
        # Define required model types for analysis including all transformer variants
        self._required_model_types = [
            'lstm', 'transformer', 'itransformer', 'patchtst', 
            'timesmixer', 'timesfm', 'technical_analyzer'
        ]
    
    async def initialize(self) -> None:
        """Initialize analysis mode resources."""
        self.logger.info("Initializing analysis mode")
        
        # Initialize preservation manager first
        await self._initialize_preservation_manager()
        
        # Load mode-specific models
        if self.preservation_manager:
            self._ml_models = await self._load_mode_specific_models()
        
        self._set_status(ModeStatus.INACTIVE)
    
    async def start(self) -> None:
        """Start analysis mode."""
        self.logger.info("Starting analysis mode")
        
        # Load models if not already loaded
        if self.preservation_manager and not self._ml_models:
            self._ml_models = await self._load_mode_specific_models()
        
        self.start_time = datetime.now()
        self._set_status(ModeStatus.ACTIVE)
    
    async def stop(self) -> None:
        """Stop analysis mode."""
        self.logger.info("Stopping analysis mode")
        
        # Backup models before stopping
        if self.preservation_manager and self._ml_models:
            await self._backup_models_on_mode_change()
        
        self._set_status(ModeStatus.STOPPING)
    
    async def pause(self) -> None:
        """Pause analysis mode."""
        self.logger.info("Pausing analysis mode")
        self._set_status(ModeStatus.PAUSED)
    
    async def resume(self) -> None:
        """Resume analysis mode."""
        self.logger.info("Resuming analysis mode")
        self._set_status(ModeStatus.ACTIVE)
    
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process market tick for analysis only (no trading actions)."""
        if self.status != ModeStatus.ACTIVE:
            return None
        
        # Analysis mode only analyzes, doesn't trade
        # Record analysis metrics
        self._record_metric("last_price", market_state.price_usd)
        self._record_metric("last_rsi", market_state.rsi)
        self._record_metric("last_volume", market_state.volume_24h)
        
        # Return HOLD or None since this is analysis only
        return TradeAction.HOLD
    
    async def cleanup(self) -> None:
        """Clean up analysis mode resources."""
        self.logger.info("Cleaning up analysis mode")
    
    async def initialize_backtest_engine(self, config: Dict[str, Any]) -> BacktestEngine:
        """Initialize and return a backtest engine with the given configuration."""
        self.logger.info("Initializing backtest engine", config_keys=list(config.keys()))
        engine = BacktestEngine(config)
        return engine
    
    async def run_strategy_backtest(
        self, 
        strategy_name: str, 
        config: Dict[str, Any], 
        tokens: List[str]
    ) -> Dict[str, Any]:
        """Run a complete strategy backtest and return performance metrics."""
        self.logger.info(
            "Running strategy backtest", 
            strategy=strategy_name, 
            tokens=tokens,
            period=f"{config.get('start_date')} to {config.get('end_date')}"
        )
        
        # Initialize backtest engine
        engine = await self.initialize_backtest_engine(config)
        
        # Run the backtest
        results = await engine.run_backtest(strategy_name, tokens)
        
        # Record metrics for this analysis
        self._record_metric(f"backtest_{strategy_name}_return", results["total_return"])
        self._record_metric(f"backtest_{strategy_name}_sharpe", results["sharpe_ratio"])
        self._record_metric(f"backtest_{strategy_name}_trades", results["total_trades"])
        
        return results
    
    async def compare_strategies(
        self, 
        strategies: List[str], 
        config: Dict[str, Any], 
        tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """Compare multiple strategies using backtesting and return ranked results."""
        self.logger.info(
            "Comparing strategies", 
            strategies=strategies, 
            tokens=tokens,
            num_strategies=len(strategies)
        )
        
        comparison_results = []
        
        for strategy in strategies:
            # Run backtest for each strategy
            backtest_result = await self.run_strategy_backtest(strategy, config, tokens)
            
            # Calculate ranking score (weighted combination of metrics)
            ranking_score = (
                backtest_result["total_return"] * 0.3 +
                backtest_result["sharpe_ratio"] * 0.3 +
                backtest_result["win_rate"] * 0.2 +
                (1 - backtest_result["max_drawdown"]) * 0.2
            )
            
            strategy_result = {
                "strategy_name": strategy,
                "performance_metrics": {
                    "total_return": backtest_result["total_return"],
                    "annual_return": backtest_result["annual_return"],
                    "max_drawdown": backtest_result["max_drawdown"],
                    "sharpe_ratio": backtest_result["sharpe_ratio"],
                    "win_rate": backtest_result["win_rate"],
                    "total_trades": backtest_result["total_trades"],
                    "profit_factor": backtest_result["profit_factor"]
                },
                "ranking_score": ranking_score,
                "backtest_details": backtest_result
            }
            
            comparison_results.append(strategy_result)
        
        # Sort by ranking score (descending)
        comparison_results.sort(key=lambda x: x["ranking_score"], reverse=True)
        
        # Record comparison metrics
        best_strategy = comparison_results[0]["strategy_name"] if comparison_results else None
        self._record_metric("strategy_comparison_best", best_strategy)
        self._record_metric("strategy_comparison_count", len(strategies))
        
        return comparison_results
    
    async def optimize_strategy_parameters(
        self, 
        strategy_name: str, 
        parameter_grid: Dict[str, List[Any]], 
        optimization_metric: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search over backtest results."""
        self.logger.info(
            "Optimizing strategy parameters", 
            strategy=strategy_name,
            parameters=list(parameter_grid.keys()),
            metric=optimization_metric
        )
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        optimization_history = []
        best_performance = float('-inf')
        best_parameters = None
        best_backtest_result = None
        
        # Default backtest config (can be enhanced)
        base_config = {
            "start_date": datetime.now() - timedelta(days=30),
            "end_date": datetime.now(),
            "initial_balance": Decimal("10000"),
            "transaction_costs": {
                "maker_fee": 0.001,
                "taker_fee": 0.001,
                "slippage_bps": 5
            }
        }
        
        for i, param_combo in enumerate(param_combinations):
            # Create parameter set
            current_params = dict(zip(param_names, param_combo))
            
            # Update backtest config with current parameters
            config = base_config.copy()
            config["strategy_parameters"] = current_params
            
            # Run backtest with current parameters
            backtest_result = await self.run_strategy_backtest(
                strategy_name, 
                config, 
                ["BTC/USDC", "ETH/USDC"]  # Default tokens
            )
            
            # Get optimization metric value
            metric_value = backtest_result.get(optimization_metric, 0)
            
            # Track optimization history
            optimization_entry = {
                "iteration": i + 1,
                "parameters": current_params,
                "metric_value": metric_value,
                "backtest_result": backtest_result
            }
            optimization_history.append(optimization_entry)
            
            # Update best if this is better
            if metric_value > best_performance:
                best_performance = metric_value
                best_parameters = current_params
                best_backtest_result = backtest_result
        
        # Calculate parameter sensitivity
        parameter_sensitivity = {}
        for param_name in param_names:
            # Group results by this parameter value
            param_groups = {}
            for entry in optimization_history:
                param_val = entry["parameters"][param_name]
                if param_val not in param_groups:
                    param_groups[param_val] = []
                param_groups[param_val].append(entry["metric_value"])
            
            # Calculate average performance for each parameter value
            param_averages = {
                val: sum(metrics) / len(metrics) 
                for val, metrics in param_groups.items()
            }
            
            # Calculate sensitivity as range of performance across parameter values
            if param_averages:
                sensitivity = max(param_averages.values()) - min(param_averages.values())
            else:
                sensitivity = 0
            
            parameter_sensitivity[param_name] = {
                "sensitivity_score": sensitivity,
                "performance_by_value": param_averages
            }
        
        # Record optimization metrics
        self._record_metric(f"optimization_{strategy_name}_best_score", best_performance)
        self._record_metric(f"optimization_{strategy_name}_combinations", len(param_combinations))
        
        return {
            "best_parameters": best_parameters,
            "best_performance": {
                "metric": optimization_metric,
                "value": best_performance,
                "backtest_result": best_backtest_result
            },
            "parameter_sensitivity": parameter_sensitivity,
            "optimization_history": optimization_history,
            "optimization_summary": {
                "strategy_name": strategy_name,
                "optimization_metric": optimization_metric,
                "total_combinations_tested": len(param_combinations),
                "improvement_over_baseline": best_performance  # Simplified
            }
        }


class SimulationMode(ModeBase):
    """
    Concrete implementation for simulation/paper trading mode.
    
    Simulates trading with virtual funds to test strategies.
    """
    
    def __init__(self, mode_id: UUID, config: ModeConfig, portfolio: Portfolio):
        """Initialize simulation mode with virtual portfolio."""
        super().__init__(mode_id, config, portfolio)
        
        # Initialize simulation-specific parameters
        self.virtual_balance = config.parameters.get("initial_balance", 10000)
        self.enable_fees = config.parameters.get("enable_fees", True)
        self.slippage_bps = config.parameters.get("slippage_bps", 10)
        
        # Define required model types for simulation including all transformer variants
        self._required_model_types = [
            'dqn', 'lstm', 'transformer', 'itransformer', 'patchtst', 
            'timesmixer', 'timesfm', 'simulator'
        ]
        
        # Initialize simulation metrics
        self._simulation_metrics = {}
    
    async def initialize(self) -> None:
        """Initialize simulation mode resources."""
        self.logger.info("Initializing simulation mode")
        
        # Initialize preservation manager first
        await self._initialize_preservation_manager()
        
        # Load mode-specific models
        if self.preservation_manager:
            self._ml_models = await self._load_mode_specific_models()
        
        self._set_status(ModeStatus.INACTIVE)
    
    async def start(self) -> None:
        """Start simulation mode."""
        self.logger.info("Starting simulation mode")
        
        # Load models if not already loaded
        if self.preservation_manager and not self._ml_models:
            self._ml_models = await self._load_mode_specific_models()
        
        self.start_time = datetime.now()
        self._set_status(ModeStatus.ACTIVE)
    
    async def stop(self) -> None:
        """Stop simulation mode."""
        self.logger.info("Stopping simulation mode")
        
        # Backup models before stopping
        if self.preservation_manager and self._ml_models:
            await self._backup_models_on_mode_change()
        
        self._set_status(ModeStatus.STOPPING)
    
    async def pause(self) -> None:
        """Pause simulation mode."""
        self.logger.info("Pausing simulation mode")
        
        # Save current state when pausing
        if self.preservation_manager:
            try:
                # Create state metadata
                state_data = {
                    'virtual_balance': self.virtual_balance,
                    'simulation_metrics': self._simulation_metrics,
                    'enable_fees': self.enable_fees,
                    'slippage_bps': self.slippage_bps
                }
                
                from src.model_preservation.base import generate_model_id
                state_type_str = f"mode_state_{self.config.mode_type.value}"
                state_version_str = f"pause_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                state_model_id = generate_model_id(state_type_str, state_version_str)
                
                metadata = ModelMetadata(
                    model_id=state_model_id,
                    model_type=state_type_str,
                    version=state_version_str,
                    created_at=datetime.now(),
                    preservation_priority=self._get_preservation_priority(),
                    tags=["pause", "state_save", self.config.mode_type.value],
                    mode=self.config.mode_type.value
                )
                
                await self.preservation_manager.save_model(
                    model_type=f"mode_state_{self.config.mode_type.value}",
                    model_data=self._serialize_model(state_data),
                    metadata=metadata,
                    mode=self.config.mode_type.value,
                    tags=["pause"]
                )
                
                self.logger.debug("Simulation state saved during pause")
            except Exception as e:
                self.logger.error("Failed to save state during pause", error=str(e))
        
        self._set_status(ModeStatus.PAUSED)
    
    async def resume(self) -> None:
        """Resume simulation mode."""
        self.logger.info("Resuming simulation mode")
        
        # Restore state when resuming
        if self.preservation_manager:
            try:
                model_data, metadata = await self.preservation_manager.load_model(
                    model_type=f"mode_state_{self.config.mode_type.value}",
                    mode=self.config.mode_type.value,
                    tags=["resume"]
                )
                
                state_data = self._deserialize_model(model_data)
                if state_data and isinstance(state_data, dict):
                    self.virtual_balance = state_data.get('virtual_balance', self.virtual_balance)
                    self._simulation_metrics = state_data.get('simulation_metrics', {})
                    self.enable_fees = state_data.get('enable_fees', self.enable_fees)
                    self.slippage_bps = state_data.get('slippage_bps', self.slippage_bps)
                    
                    self.logger.debug("Simulation state restored from pause")
                
            except Exception as e:
                self.logger.warning("Failed to restore state during resume", error=str(e))
                # Continue with current state
        
        self._set_status(ModeStatus.ACTIVE)
    
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process market tick and simulate trading decisions."""
        if self.status != ModeStatus.ACTIVE:
            return None
        
        # Simulation mode can make trading decisions
        # This would integrate with RL agent for decision making
        
        # Simple simulation logic based on technical indicators
        action = TradeAction.HOLD
        
        if market_state.rsi is not None:
            if market_state.rsi < 25:  # Very oversold
                action = TradeAction.STRONG_BUY
            elif market_state.rsi < 35:  # Oversold
                action = TradeAction.BUY
            elif market_state.rsi > 75:  # Very overbought
                action = TradeAction.STRONG_SELL
            elif market_state.rsi > 65:  # Overbought
                action = TradeAction.SELL
        
        # Record simulation metrics
        self._record_metric("simulated_action", action.value)
        self._record_metric("virtual_balance", self.virtual_balance)
        
        return action
    
    async def cleanup(self) -> None:
        """Clean up simulation mode resources."""
        self.logger.info("Cleaning up simulation mode")


# Mode Exception Classes
class ModeError(Exception):
    """Base mode error."""
    pass


class ModeNotFoundError(ModeError):
    """Error when mode is not found."""
    pass


class ModeConfigError(ModeError):
    """Error in mode configuration."""
    pass


class InvalidModeTransitionError(ModeError):
    """Error when invalid mode transition is attempted."""
    pass