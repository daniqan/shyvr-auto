"""
Mode manager for coordinating multiple trading modes.

This module provides the ModeManager class for managing the lifecycle
of multiple trading modes, mode switching, scheduling, and coordination.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
import structlog

from src.modes.base import (
    ModeType,
    ModeStatus,
    ModeBase,
    ModeConfig,
    ModeResult,
    TradingMode,
    AnalysisMode,
    SimulationMode,
)
from src.portfolio.base import Portfolio
from src.rl_agent.base import TradeAction, MarketState
from src.logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    TradingMode as LoggingTradingMode, performance_tracker
)


logger = structlog.get_logger()


@dataclass
class ModeManagerConfig:
    """Configuration for mode manager."""
    max_concurrent_modes: int = 1
    enable_mode_switching: bool = True
    auto_recovery: bool = True
    health_check_interval_seconds: int = 30
    max_error_retries: int = 3
    mode_timeout_minutes: int = 60
    default_modes: List[ModeConfig] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_concurrent_modes <= 0:
            raise ValueError("max_concurrent_modes must be positive")
        if self.health_check_interval_seconds <= 0:
            raise ValueError("health_check_interval_seconds must be positive")
        if self.max_error_retries < 0:
            raise ValueError("max_error_retries must be non-negative")
        if self.mode_timeout_minutes <= 0:
            raise ValueError("mode_timeout_minutes must be positive")


@dataclass
class ScheduledMode:
    """Represents a scheduled mode execution."""
    schedule_id: UUID
    mode_config: ModeConfig
    start_time: datetime
    max_runtime_minutes: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModeManager:
    """
    Central manager for coordinating multiple trading modes.
    
    Handles mode registration, lifecycle management, switching,
    and coordination between different trading modes.
    """
    
    def __init__(self, config: ModeManagerConfig, portfolio: Portfolio):
        """Initialize mode manager with configuration and portfolio."""
        self.config = config
        self.portfolio = portfolio
        self.active_modes: Dict[UUID, ModeBase] = {}
        self.mode_history: List[ModeResult] = []
        self._initialized = False
        self._mode_type_registry: Dict[ModeType, UUID] = {}
        
        # Configure logger
        self.logger = logger.bind(
            portfolio_id=str(portfolio.portfolio_id),
            max_concurrent=config.max_concurrent_modes
        )
    
    async def initialize(self) -> None:
        """Initialize the mode manager."""
        # Log mode manager initialization start
        await activity_logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.START,
            source="mode_manager",
            event_type="mode_manager_initialization",
            title="Mode manager initialization started",
            severity=ActivitySeverity.INFO,
            metadata={
                "portfolio_id": str(self.portfolio.portfolio_id),
                "max_concurrent_modes": self.config.max_concurrent_modes,
                "default_modes_count": len(self.config.default_modes)
            }
        )
        
        self.logger.info("Initializing mode manager")
        
        # Initialize any default modes
        for mode_config in self.config.default_modes:
            try:
                await self.register_mode(mode_config)
                if mode_config.auto_start:
                    mode_id = self._mode_type_registry.get(mode_config.mode_type)
                    if mode_id:
                        await self.start_mode(mode_id)
            except Exception as e:
                # Log mode initialization failure
                await activity_logger.log_error(
                    category=ActivityCategory.SYSTEM,
                    source="mode_manager",
                    event_type="default_mode_init_failed",
                    title=f"Failed to initialize default mode: {mode_config.mode_type.value}",
                    error_message=str(e),
                    exception=e,
                    severity=ActivitySeverity.ERROR,
                    metadata={
                        "mode_type": mode_config.mode_type.value,
                        "auto_start": mode_config.auto_start
                    }
                )
                
                self.logger.error(
                    "Failed to initialize default mode",
                    mode_type=mode_config.mode_type.value,
                    error=str(e)
                )
        
        self._initialized = True
        
        # Log successful initialization
        await activity_logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.SUCCESS,
            source="mode_manager",
            event_type="mode_manager_initialized",
            title="Mode manager initialized successfully",
            severity=ActivitySeverity.INFO,
            metadata={
                "initialized_modes": len(self.active_modes),
                "mode_registry": {mode_type.value: str(mode_id) for mode_type, mode_id in self._mode_type_registry.items()}
            }
        )
        
        self.logger.info("Mode manager initialized successfully")
    
    async def register_mode(self, mode_config: ModeConfig) -> UUID:
        """Register a new mode with the manager."""
        if not self._initialized:
            raise ModeManagerError("Mode manager not initialized")
        
        # Check for duplicate mode types
        if mode_config.mode_type in self._mode_type_registry:
            raise DuplicateModeError(f"Mode type {mode_config.mode_type.value} already registered")
        
        # Create mode based on type
        mode_id = uuid4()
        mode = self._create_mode(mode_id, mode_config)
        
        # Initialize the mode
        await mode.initialize()
        
        # Register the mode
        self.active_modes[mode_id] = mode
        self._mode_type_registry[mode_config.mode_type] = mode_id
        
        self.logger.info(
            "Mode registered successfully",
            mode_id=str(mode_id),
            mode_type=mode_config.mode_type.value
        )
        
        return mode_id
    
    async def unregister_mode(self, mode_id: UUID) -> bool:
        """Unregister a mode from the manager."""
        if mode_id not in self.active_modes:
            self.logger.warning("Attempt to unregister non-existent mode", mode_id=str(mode_id))
            return False
        
        mode = self.active_modes[mode_id]
        
        # Stop the mode if it's running
        if mode.status in [ModeStatus.ACTIVE, ModeStatus.PAUSED]:
            await self.stop_mode(mode_id)
        
        # Clean up the mode
        await mode.cleanup()
        
        # Remove from registries
        del self.active_modes[mode_id]
        # Remove from type registry
        mode_type_to_remove = None
        for mode_type, registered_id in self._mode_type_registry.items():
            if registered_id == mode_id:
                mode_type_to_remove = mode_type
                break
        if mode_type_to_remove:
            del self._mode_type_registry[mode_type_to_remove]
        
        # Add to history
        result = mode.get_result()
        self.mode_history.append(result)
        
        self.logger.info("Mode unregistered successfully", mode_id=str(mode_id))
        return True
    
    async def start_mode(self, mode_id: UUID) -> None:
        """Start a registered mode."""
        if mode_id not in self.active_modes:
            raise ModeManagerError(f"Mode {mode_id} not found")
        
        mode = self.active_modes[mode_id]
        
        # Check concurrent mode limits
        active_count = sum(
            1 for m in self.active_modes.values()
            if m.status == ModeStatus.ACTIVE
        )
        
        if active_count >= self.config.max_concurrent_modes:
            # Stop oldest active mode to make room
            oldest_mode = self._get_oldest_active_mode()
            if oldest_mode:
                await self.stop_mode(oldest_mode)
        
        await mode.start()
        
        self.logger.info(
            "Mode started successfully",
            mode_id=str(mode_id),
            mode_type=mode.config.mode_type.value
        )
    
    async def stop_mode(self, mode_id: UUID) -> None:
        """Stop a running mode."""
        if mode_id not in self.active_modes:
            raise ModeManagerError(f"Mode {mode_id} not found")
        
        mode = self.active_modes[mode_id]
        await mode.stop()
        
        self.logger.info(
            "Mode stopped",
            mode_id=str(mode_id),
            mode_type=mode.config.mode_type.value
        )
    
    async def pause_mode(self, mode_id: UUID) -> None:
        """Pause a running mode."""
        if mode_id not in self.active_modes:
            raise ModeManagerError(f"Mode {mode_id} not found")
        
        mode = self.active_modes[mode_id]
        if mode.status != ModeStatus.ACTIVE:
            raise ModeTransitionError(f"Cannot pause mode in status {mode.status.value}")
        
        await mode.pause()
        
        self.logger.info("Mode paused", mode_id=str(mode_id))
    
    async def resume_mode(self, mode_id: UUID) -> None:
        """Resume a paused mode."""
        if mode_id not in self.active_modes:
            raise ModeManagerError(f"Mode {mode_id} not found")
        
        mode = self.active_modes[mode_id]
        if mode.status != ModeStatus.PAUSED:
            raise ModeTransitionError(f"Cannot resume mode in status {mode.status.value}")
        
        await mode.resume()
        
        self.logger.info("Mode resumed", mode_id=str(mode_id))
    
    async def switch_mode(self, from_mode_id: UUID, to_mode_id: UUID) -> None:
        """Switch from one mode to another."""
        if not self.config.enable_mode_switching:
            raise ModeTransitionError("Mode switching is disabled")
        
        if from_mode_id not in self.active_modes or to_mode_id not in self.active_modes:
            raise ModeManagerError("One or both modes not found")
        
        from_mode = self.active_modes[from_mode_id]
        to_mode = self.active_modes[to_mode_id]
        
        # Log mode switch attempt
        await activity_logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.UPDATE,
            source="mode_manager",
            event_type="mode_switch_started",
            title=f"Mode switch started: {from_mode.config.mode_type.value} -> {to_mode.config.mode_type.value}",
            severity=ActivitySeverity.INFO,
            metadata={
                "from_mode_id": str(from_mode_id),
                "to_mode_id": str(to_mode_id),
                "from_mode_type": from_mode.config.mode_type.value,
                "to_mode_type": to_mode.config.mode_type.value
            }
        )
        
        try:
            # Stop the current mode
            await self.stop_mode(from_mode_id)
            
            # Start the new mode
            await self.start_mode(to_mode_id)
            
            # Log successful mode switch
            await activity_logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.SUCCESS,
                source="mode_manager",
                event_type="mode_switch_completed",
                title=f"Mode switch completed: {from_mode.config.mode_type.value} -> {to_mode.config.mode_type.value}",
                severity=ActivitySeverity.INFO,
                metadata={
                    "from_mode_id": str(from_mode_id),
                    "to_mode_id": str(to_mode_id),
                    "from_mode_type": from_mode.config.mode_type.value,
                    "to_mode_type": to_mode.config.mode_type.value
                }
            )
            
            self.logger.info(
                "Mode switch completed",
                from_mode=str(from_mode_id),
                to_mode=str(to_mode_id)
            )
            
        except Exception as e:
            # Log mode switch failure
            await activity_logger.log_error(
                category=ActivityCategory.SYSTEM,
                source="mode_manager",
                event_type="mode_switch_failed",
                title=f"Mode switch failed: {from_mode.config.mode_type.value} -> {to_mode.config.mode_type.value}",
                error_message=str(e),
                exception=e,
                severity=ActivitySeverity.ERROR,
                metadata={
                    "from_mode_id": str(from_mode_id),
                    "to_mode_id": str(to_mode_id),
                    "from_mode_type": from_mode.config.mode_type.value,
                    "to_mode_type": to_mode.config.mode_type.value
                }
            )
            raise
    
    def get_mode_status(self, mode_id: UUID) -> Optional[ModeStatus]:
        """Get the status of a specific mode."""
        if mode_id not in self.active_modes:
            return None
        return self.active_modes[mode_id].status
    
    def list_active_modes(self) -> Dict[UUID, ModeBase]:
        """List all currently registered modes."""
        return self.active_modes.copy()
    
    def get_mode_results(self) -> List[ModeResult]:
        """Get historical mode execution results."""
        return self.mode_history.copy()
    
    async def process_market_tick(self, market_state: MarketState) -> Dict[UUID, TradeAction]:
        """Process market tick through all active modes."""
        actions = {}
        
        for mode_id, mode in self.active_modes.items():
            if mode.status == ModeStatus.ACTIVE:
                try:
                    action = await mode.process_tick(market_state)
                    if action:
                        actions[mode_id] = action
                except Exception as e:
                    self.logger.error(
                        "Error processing tick in mode",
                        mode_id=str(mode_id),
                        error=str(e)
                    )
                    
                    if self.config.auto_recovery:
                        await self._handle_mode_error(mode_id, e)
        
        return actions
    
    async def cleanup(self) -> None:
        """Clean up all modes and resources."""
        self.logger.info("Starting mode manager cleanup")
        
        # Stop all active modes
        for mode_id in list(self.active_modes.keys()):
            try:
                await self.unregister_mode(mode_id)
            except Exception as e:
                self.logger.error(
                    "Error during mode cleanup",
                    mode_id=str(mode_id),
                    error=str(e)
                )
        
        self.logger.info("Mode manager cleanup completed")
    
    def _create_mode(self, mode_id: UUID, mode_config: ModeConfig) -> ModeBase:
        """Create a mode instance based on configuration."""
        if mode_config.mode_type == ModeType.LIVE_TRADING:
            return TradingMode(mode_id, mode_config, self.portfolio)
        elif mode_config.mode_type == ModeType.ANALYSIS:
            return AnalysisMode(mode_id, mode_config, self.portfolio)
        elif mode_config.mode_type in [ModeType.SIMULATION, ModeType.PAPER_TRADING, ModeType.BACKTESTING]:
            return SimulationMode(mode_id, mode_config, self.portfolio)
        else:
            raise ModeManagerError(f"Unsupported mode type: {mode_config.mode_type.value}")
    
    def _get_oldest_active_mode(self) -> Optional[UUID]:
        """Get the oldest active mode for replacement."""
        oldest_mode_id = None
        oldest_start_time = None
        
        for mode_id, mode in self.active_modes.items():
            if mode.status == ModeStatus.ACTIVE and mode.start_time:
                if oldest_start_time is None or mode.start_time < oldest_start_time:
                    oldest_start_time = mode.start_time
                    oldest_mode_id = mode_id
        
        return oldest_mode_id
    
    async def _handle_mode_error(self, mode_id: UUID, error: Exception) -> None:
        """Handle mode errors with recovery logic."""
        mode = self.active_modes.get(mode_id)
        if not mode:
            return
        
        self.logger.warning(
            "Mode error detected, attempting recovery",
            mode_id=str(mode_id),
            error=str(error)
        )
        
        # Simple recovery: restart the mode
        try:
            await mode.stop()
            await asyncio.sleep(1)  # Brief pause
            await mode.start()
            
            self.logger.info("Mode recovery successful", mode_id=str(mode_id))
        except Exception as recovery_error:
            self.logger.error(
                "Mode recovery failed",
                mode_id=str(mode_id),
                error=str(recovery_error)
            )
            # Mark mode as error state
            mode._set_status(ModeStatus.ERROR, str(recovery_error))


class ModeScheduler:
    """
    Scheduler for managing timed mode execution.
    
    Allows scheduling modes to start/stop at specific times
    or intervals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mode scheduler with configuration."""
        self.config = config
        self.scheduled_modes: List[ScheduledMode] = []
        
        self.logger = logger.bind(component="mode_scheduler")
    
    async def schedule_mode(
        self,
        mode_config: ModeConfig,
        start_time: datetime,
        max_runtime_minutes: Optional[int] = None
    ) -> UUID:
        """Schedule a mode to start at a specific time."""
        schedule_id = uuid4()
        
        scheduled_mode = ScheduledMode(
            schedule_id=schedule_id,
            mode_config=mode_config,
            start_time=start_time,
            max_runtime_minutes=max_runtime_minutes
        )
        
        self.scheduled_modes.append(scheduled_mode)
        
        self.logger.info(
            "Mode scheduled",
            schedule_id=str(schedule_id),
            mode_type=mode_config.mode_type.value,
            start_time=start_time.isoformat()
        )
        
        return schedule_id
    
    async def cancel_scheduled_mode(self, schedule_id: UUID) -> bool:
        """Cancel a scheduled mode."""
        for i, scheduled_mode in enumerate(self.scheduled_modes):
            if scheduled_mode.schedule_id == schedule_id:
                del self.scheduled_modes[i]
                
                self.logger.info("Scheduled mode canceled", schedule_id=str(schedule_id))
                return True
        
        return False
    
    def list_scheduled_modes(self) -> List[ScheduledMode]:
        """List all scheduled modes."""
        return self.scheduled_modes.copy()
    
    async def process_scheduled_modes(self, mode_manager: ModeManager) -> None:
        """Process scheduled modes that are due to start."""
        current_time = datetime.now()
        due_modes = []
        
        for scheduled_mode in self.scheduled_modes:
            if scheduled_mode.start_time <= current_time:
                due_modes.append(scheduled_mode)
        
        for scheduled_mode in due_modes:
            try:
                # Register and start the mode
                mode_id = await mode_manager.register_mode(scheduled_mode.mode_config)
                await mode_manager.start_mode(mode_id)
                
                self.logger.info(
                    "Scheduled mode started",
                    schedule_id=str(scheduled_mode.schedule_id),
                    mode_id=str(mode_id)
                )
                
                # Remove from scheduled list
                self.scheduled_modes.remove(scheduled_mode)
                
            except Exception as e:
                self.logger.error(
                    "Failed to start scheduled mode",
                    schedule_id=str(scheduled_mode.schedule_id),
                    error=str(e)
                )


# Mode Manager Exception Classes
class ModeManagerError(Exception):
    """Base mode manager error."""
    pass


class ModeTransitionError(ModeManagerError):
    """Error during mode transitions."""
    pass


class DuplicateModeError(ModeManagerError):
    """Error when attempting to register duplicate mode type."""
    pass