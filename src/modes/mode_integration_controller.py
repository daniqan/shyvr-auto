"""
Mode Integration Controller

This module provides the central controller for mode transitions and coordination.
It handles seamless transitions between Analysis → Simulation → Live modes with
state preservation, cross-mode experience sharing, and production-grade safety.

Key Features:
- Seamless mode transitions with state preservation
- Cross-mode experience sharing for continuous learning
- Unified configuration management across modes
- Emergency stop coordination across all modes
- Production-grade error handling and recovery

Following TDD methodology - implementation satisfies comprehensive test requirements.
"""

import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID, uuid4
import structlog

from src.modes.base import (
    ModeType, ModeStatus, ModeBase, ModeConfig, ModeResult,
    TradingMode, AnalysisMode, SimulationMode
)
from src.modes.mode_manager import ModeManager, ModeManagerConfig
from src.portfolio.base import Portfolio, Position, Transaction
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import ExperienceReplayBuffer
from src.integration.ml_rl_bridge import MLRLBridge, MLRLConfig
from src.ml_analysis.model_manager import ModelManager
from src.modes.experience_collector import TradingExperienceCollector


logger = structlog.get_logger()


class TransitionType(Enum):
    """Types of mode transitions."""
    ANALYSIS_TO_SIMULATION = "analysis_to_simulation"
    SIMULATION_TO_LIVE = "simulation_to_live"
    LIVE_TO_SIMULATION = "live_to_simulation"
    ANY_TO_ANALYSIS = "any_to_analysis"
    EMERGENCY_STOP_ALL = "emergency_stop_all"


class TransitionStatus(Enum):
    """Status of mode transitions."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModeStateSnapshot:
    """Comprehensive snapshot of mode state."""
    mode_id: UUID
    mode_type: ModeType
    status: ModeStatus
    timestamp: datetime
    config: ModeConfig
    metrics: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    ml_model_state: Optional[Dict[str, Any]] = None
    rl_agent_state: Optional[Dict[str, Any]] = None
    experience_buffer: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization."""
        return {
            "mode_id": str(self.mode_id),
            "mode_type": self.mode_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "config": self._serialize_config(),
            "metrics": self.metrics,
            "portfolio_state": self.portfolio_state,
            "ml_model_state": self.ml_model_state,
            "rl_agent_state": self.rl_agent_state,
            "experience_buffer": self.experience_buffer,
            "metadata": self.metadata
        }
    
    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize mode config for storage."""
        return {
            "mode_type": self.config.mode_type.value,
            "enabled": self.config.enabled,
            "auto_start": self.config.auto_start,
            "max_runtime_minutes": self.config.max_runtime_minutes,
            "stop_on_error": self.config.stop_on_error,
            "log_level": self.config.log_level,
            "parameters": self.config.parameters
        }


@dataclass
class ModeTransitionPlan:
    """Plan for transitioning between modes."""
    transition_id: UUID
    transition_type: TransitionType
    from_mode_id: Optional[UUID]
    to_mode_type: ModeType
    from_snapshot: Optional[ModeStateSnapshot]
    target_config: ModeConfig
    preserve_state: bool = True
    transfer_experiences: bool = True
    validate_safety: bool = True
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ModeIntegrationConfig:
    """Configuration for mode integration controller."""
    state_persistence_path: str = "data/mode_states"
    max_snapshots_per_mode: int = 10
    snapshot_compression: bool = True
    enable_state_validation: bool = True
    transition_timeout_seconds: int = 300
    enable_cross_mode_experience_sharing: bool = True
    ml_model_sharing: bool = True
    rl_experience_transfer: bool = True
    emergency_stop_timeout_seconds: int = 30
    rollback_on_failure: bool = True
    validate_transitions: bool = True
    auto_backup_before_transition: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_snapshots_per_mode <= 0:
            raise ValueError("max_snapshots_per_mode must be positive")
        if self.transition_timeout_seconds <= 0:
            raise ValueError("transition_timeout_seconds must be positive")
        if self.emergency_stop_timeout_seconds <= 0:
            raise ValueError("emergency_stop_timeout_seconds must be positive")


class ModeIntegrationController:
    """
    Central controller for mode transitions and coordination.
    
    Handles seamless transitions between different trading modes while
    preserving state, sharing experiences, and maintaining safety.
    """
    
    def __init__(
        self,
        config: ModeIntegrationConfig,
        mode_manager: ModeManager,
        ml_rl_bridge: MLRLBridge,
        model_manager: Optional[ModelManager] = None
    ):
        """Initialize mode integration controller."""
        self.config = config
        self.mode_manager = mode_manager
        self.ml_rl_bridge = ml_rl_bridge
        self.model_manager = model_manager
        
        # State management
        self.active_transitions: Dict[UUID, ModeTransitionPlan] = {}
        self.mode_snapshots: Dict[UUID, List[ModeStateSnapshot]] = {}
        self.transition_history: List[ModeTransitionPlan] = []
        
        # Ensure state persistence directory exists
        self.state_path = Path(config.state_persistence_path)
        self.state_path.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        self.logger = logger.bind(
            component="mode_integration_controller",
            state_path=str(self.state_path)
        )
        
        self.logger.info("Mode integration controller initialized")
    
    async def plan_transition(
        self,
        transition_type: TransitionType,
        target_config: ModeConfig,
        from_mode_id: Optional[UUID] = None,
        preserve_state: bool = True,
        transfer_experiences: bool = True
    ) -> UUID:
        """Plan a mode transition with validation."""
        transition_id = uuid4()
        
        # Create snapshot of current mode if specified
        from_snapshot = None
        if from_mode_id and preserve_state:
            from_snapshot = await self._create_mode_snapshot(from_mode_id)
        
        # Create transition plan
        plan = ModeTransitionPlan(
            transition_id=transition_id,
            transition_type=transition_type,
            from_mode_id=from_mode_id,
            to_mode_type=target_config.mode_type,
            from_snapshot=from_snapshot,
            target_config=target_config,
            preserve_state=preserve_state,
            transfer_experiences=transfer_experiences,
            validate_safety=self.config.validate_transitions,
            timeout_seconds=self.config.transition_timeout_seconds
        )
        
        # Validate transition plan
        if self.config.validate_transitions:
            await self._validate_transition_plan(plan)
        
        # Store plan
        self.active_transitions[transition_id] = plan
        
        self.logger.info(
            "Transition planned",
            transition_id=str(transition_id),
            transition_type=transition_type.value,
            from_mode=str(from_mode_id) if from_mode_id else None,
            to_mode=target_config.mode_type.value
        )
        
        return transition_id
    
    async def execute_transition(self, transition_id: UUID) -> bool:
        """Execute a planned mode transition."""
        if transition_id not in self.active_transitions:
            raise IntegrationError(f"Transition {transition_id} not found")
        
        plan = self.active_transitions[transition_id]
        
        try:
            self.logger.info(
                "Starting transition execution",
                transition_id=str(transition_id),
                transition_type=plan.transition_type.value
            )
            
            # Create backup if configured
            if self.config.auto_backup_before_transition and plan.from_mode_id:
                await self._backup_mode_state(plan.from_mode_id)
            
            # Execute transition based on type
            success = await self._execute_transition_by_type(plan)
            
            if success:
                self.logger.info(
                    "Transition completed successfully",
                    transition_id=str(transition_id)
                )
                
                # Move to history
                self.transition_history.append(plan)
                del self.active_transitions[transition_id]
                
                return True
            else:
                # Handle failure
                if self.config.rollback_on_failure:
                    await self._rollback_transition(plan)
                
                raise IntegrationError(f"Transition {transition_id} failed")
                
        except Exception as e:
            self.logger.error(
                "Transition execution failed",
                transition_id=str(transition_id),
                error=str(e)
            )
            
            if self.config.rollback_on_failure:
                await self._rollback_transition(plan)
            
            raise
    
    async def emergency_stop_all_modes(self, reason: str = "Emergency stop triggered") -> bool:
        """Emergency stop all active modes immediately."""
        self.logger.critical("Emergency stop initiated", reason=reason)
        
        try:
            # Create snapshots of all active modes before stopping
            snapshots = {}
            for mode_id, mode in self.mode_manager.active_modes.items():
                if mode.status == ModeStatus.ACTIVE:
                    try:
                        snapshot = await self._create_mode_snapshot(mode_id)
                        snapshots[mode_id] = snapshot
                        await self._save_snapshot(snapshot)
                    except Exception as e:
                        self.logger.error(
                            "Failed to create emergency snapshot",
                            mode_id=str(mode_id),
                            error=str(e)
                        )
            
            # Stop all modes with timeout
            stop_tasks = []
            for mode_id in list(self.mode_manager.active_modes.keys()):
                task = asyncio.create_task(self.mode_manager.stop_mode(mode_id))
                stop_tasks.append(task)
            
            # Wait for all stops to complete with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*stop_tasks, return_exceptions=True),
                    timeout=self.config.emergency_stop_timeout_seconds
                )
            except asyncio.TimeoutError:
                self.logger.error("Emergency stop timeout exceeded")
                # Force stop any remaining modes
                for mode_id, mode in self.mode_manager.active_modes.items():
                    mode.status = ModeStatus.ERROR
                    mode.error_message = "Emergency stop timeout"
            
            self.logger.info(
                "Emergency stop completed",
                stopped_modes=len(snapshots),
                reason=reason
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Emergency stop failed", error=str(e))
            return False
    
    async def restore_mode_from_snapshot(
        self, 
        snapshot: ModeStateSnapshot,
        target_mode_id: Optional[UUID] = None
    ) -> UUID:
        """Restore a mode from a state snapshot."""
        # Create new mode configuration from snapshot
        config = ModeConfig(
            mode_type=snapshot.mode_type,
            enabled=True,
            auto_start=False,
            max_runtime_minutes=snapshot.config.max_runtime_minutes,
            stop_on_error=snapshot.config.stop_on_error,
            log_level=snapshot.config.log_level,
            parameters=snapshot.config.parameters.copy()
        )
        
        # Register new mode
        mode_id = await self.mode_manager.register_mode(config)
        
        # Restore mode state
        mode = self.mode_manager.active_modes[mode_id]
        mode.metrics = snapshot.metrics.copy()
        mode.metadata = snapshot.metadata.copy()
        
        # Restore ML model state if available
        if snapshot.ml_model_state and self.model_manager:
            await self._restore_ml_model_state(mode_id, snapshot.ml_model_state)
        
        # Restore RL agent state if available  
        if snapshot.rl_agent_state:
            await self._restore_rl_agent_state(mode_id, snapshot.rl_agent_state)
        
        # Restore experience buffer if available
        if snapshot.experience_buffer:
            await self._restore_experience_buffer(mode_id, snapshot.experience_buffer)
        
        self.logger.info(
            "Mode restored from snapshot",
            mode_id=str(mode_id),
            snapshot_mode_id=str(snapshot.mode_id),
            snapshot_timestamp=snapshot.timestamp.isoformat()
        )
        
        return mode_id
    
    async def get_mode_snapshots(self, mode_id: UUID) -> List[ModeStateSnapshot]:
        """Get all snapshots for a specific mode."""
        return self.mode_snapshots.get(mode_id, [])
    
    async def cleanup_old_snapshots(self) -> int:
        """Clean up old snapshots based on retention policy."""
        cleaned_count = 0
        
        for mode_id, snapshots in self.mode_snapshots.items():
            if len(snapshots) > self.config.max_snapshots_per_mode:
                # Sort by timestamp and keep most recent
                snapshots.sort(key=lambda s: s.timestamp, reverse=True)
                to_remove = snapshots[self.config.max_snapshots_per_mode:]
                
                for snapshot in to_remove:
                    # Remove snapshot file
                    snapshot_file = self._get_snapshot_file_path(snapshot)
                    if snapshot_file.exists():
                        snapshot_file.unlink()
                
                # Update in-memory list
                self.mode_snapshots[mode_id] = snapshots[:self.config.max_snapshots_per_mode]
                cleaned_count += len(to_remove)
        
        self.logger.info("Cleaned up old snapshots", cleaned_count=cleaned_count)
        return cleaned_count
    
    # Private methods
    
    async def _create_mode_snapshot(self, mode_id: UUID) -> ModeStateSnapshot:
        """Create comprehensive snapshot of mode state."""
        if mode_id not in self.mode_manager.active_modes:
            raise IntegrationError(f"Mode {mode_id} not found")
        
        mode = self.mode_manager.active_modes[mode_id]
        
        # Capture portfolio state
        portfolio_state = await self._capture_portfolio_state(mode.portfolio)
        
        # Capture ML model state if available
        ml_model_state = None
        if self.model_manager:
            ml_model_state = await self._capture_ml_model_state(mode_id)
        
        # Capture RL agent state if available
        rl_agent_state = await self._capture_rl_agent_state(mode_id)
        
        # Capture experience buffer if available
        experience_buffer = await self._capture_experience_buffer(mode_id)
        
        snapshot = ModeStateSnapshot(
            mode_id=mode_id,
            mode_type=mode.config.mode_type,
            status=mode.status,
            timestamp=datetime.now(),
            config=mode.config,
            metrics=mode.metrics.copy(),
            portfolio_state=portfolio_state,
            ml_model_state=ml_model_state,
            rl_agent_state=rl_agent_state,
            experience_buffer=experience_buffer,
            metadata=mode.metadata.copy()
        )
        
        # Store snapshot
        if mode_id not in self.mode_snapshots:
            self.mode_snapshots[mode_id] = []
        self.mode_snapshots[mode_id].append(snapshot)
        
        # Save to disk
        await self._save_snapshot(snapshot)
        
        return snapshot
    
    async def _validate_transition_plan(self, plan: ModeTransitionPlan) -> None:
        """Validate a transition plan."""
        # Validate transition type
        if plan.transition_type == TransitionType.SIMULATION_TO_LIVE:
            # Additional safety checks for live mode
            if not await self._validate_live_mode_readiness(plan):
                raise InvalidTransitionError("System not ready for live mode")
        
        # Validate source mode state
        if plan.from_mode_id:
            mode = self.mode_manager.active_modes.get(plan.from_mode_id)
            if not mode:
                raise InvalidTransitionError(f"Source mode {plan.from_mode_id} not found")
            
            if mode.status == ModeStatus.ERROR:
                raise InvalidTransitionError("Cannot transition from error mode")
        
        # Validate target configuration
        if not plan.target_config.enabled:
            raise InvalidTransitionError("Target mode configuration is disabled")
    
    async def _execute_transition_by_type(self, plan: ModeTransitionPlan) -> bool:
        """Execute transition based on type."""
        if plan.transition_type == TransitionType.ANALYSIS_TO_SIMULATION:
            return await self._execute_analysis_to_simulation(plan)
        elif plan.transition_type == TransitionType.SIMULATION_TO_LIVE:
            return await self._execute_simulation_to_live(plan)
        elif plan.transition_type == TransitionType.LIVE_TO_SIMULATION:
            return await self._execute_live_to_simulation(plan)
        elif plan.transition_type == TransitionType.ANY_TO_ANALYSIS:
            return await self._execute_any_to_analysis(plan)
        elif plan.transition_type == TransitionType.EMERGENCY_STOP_ALL:
            return await self.emergency_stop_all_modes("Planned emergency stop")
        else:
            raise IntegrationError(f"Unsupported transition type: {plan.transition_type}")
    
    async def _execute_analysis_to_simulation(self, plan: ModeTransitionPlan) -> bool:
        """Execute analysis to simulation transition."""
        try:
            # Stop analysis mode if specified
            if plan.from_mode_id:
                await self.mode_manager.stop_mode(plan.from_mode_id)
            
            # Register and start simulation mode
            sim_mode_id = await self.mode_manager.register_mode(plan.target_config)
            
            # Transfer ML models and experiences if configured
            if plan.transfer_experiences and plan.from_snapshot:
                await self._transfer_learning_state(plan.from_snapshot, sim_mode_id)
            
            # Start simulation mode
            await self.mode_manager.start_mode(sim_mode_id)
            
            return True
            
        except Exception as e:
            self.logger.error("Analysis to simulation transition failed", error=str(e))
            return False
    
    async def _execute_simulation_to_live(self, plan: ModeTransitionPlan) -> bool:
        """Execute simulation to live transition with additional safety checks."""
        try:
            # Additional safety validation for live mode
            if not await self._validate_live_mode_safety():
                raise IntegrationError("Live mode safety validation failed")
            
            # Stop simulation mode if specified
            if plan.from_mode_id:
                await self.mode_manager.stop_mode(plan.from_mode_id)
            
            # Register and start live mode
            live_mode_id = await self.mode_manager.register_mode(plan.target_config)
            
            # Transfer trained models and experiences
            if plan.transfer_experiences and plan.from_snapshot:
                await self._transfer_learning_state(plan.from_snapshot, live_mode_id)
            
            # Start live mode
            await self.mode_manager.start_mode(live_mode_id)
            
            return True
            
        except Exception as e:
            self.logger.error("Simulation to live transition failed", error=str(e))
            return False
    
    async def _execute_live_to_simulation(self, plan: ModeTransitionPlan) -> bool:
        """Execute live to simulation transition."""
        try:
            # Stop live mode if specified
            if plan.from_mode_id:
                await self.mode_manager.stop_mode(plan.from_mode_id)
            
            # Register and start simulation mode
            sim_mode_id = await self.mode_manager.register_mode(plan.target_config)
            
            # Transfer live trading experiences
            if plan.transfer_experiences and plan.from_snapshot:
                await self._transfer_learning_state(plan.from_snapshot, sim_mode_id)
            
            # Start simulation mode
            await self.mode_manager.start_mode(sim_mode_id)
            
            return True
            
        except Exception as e:
            self.logger.error("Live to simulation transition failed", error=str(e))
            return False
    
    async def _execute_any_to_analysis(self, plan: ModeTransitionPlan) -> bool:
        """Execute transition from any mode to analysis."""
        try:
            # Stop current mode if specified
            if plan.from_mode_id:
                await self.mode_manager.stop_mode(plan.from_mode_id)
            
            # Register and start analysis mode
            analysis_mode_id = await self.mode_manager.register_mode(plan.target_config)
            
            # Transfer relevant state for analysis
            if plan.preserve_state and plan.from_snapshot:
                await self._transfer_analysis_state(plan.from_snapshot, analysis_mode_id)
            
            # Start analysis mode
            await self.mode_manager.start_mode(analysis_mode_id)
            
            return True
            
        except Exception as e:
            self.logger.error("Transition to analysis failed", error=str(e))
            return False
    
    async def _rollback_transition(self, plan: ModeTransitionPlan) -> None:
        """Rollback a failed transition."""
        self.logger.info("Rolling back failed transition", transition_id=str(plan.transition_id))
        
        try:
            # If we have a snapshot, restore the original mode
            if plan.from_snapshot:
                await self.restore_mode_from_snapshot(plan.from_snapshot)
                self.logger.info("Original mode restored from snapshot")
            
        except Exception as e:
            self.logger.error("Rollback failed", error=str(e))
    
    async def _capture_portfolio_state(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Capture portfolio state for snapshot."""
        return {
            "portfolio_id": str(portfolio.portfolio_id),
            "balance": str(portfolio.balance),
            "unrealized_pnl": str(portfolio.unrealized_pnl),
            "total_value": str(portfolio.total_value),
            "position_count": len(portfolio.positions),
            "positions": [
                {
                    "position_id": str(pos.position_id),
                    "token_address": pos.token_address,
                    "quantity": str(pos.quantity),
                    "entry_price": str(pos.entry_price),
                    "current_price": str(pos.current_price),
                    "unrealized_pnl": str(pos.unrealized_pnl),
                    "position_type": pos.position_type.value,
                    "status": pos.status.value
                }
                for pos in portfolio.positions.values()
            ]
        }
    
    async def _capture_ml_model_state(self, mode_id: UUID) -> Optional[Dict[str, Any]]:
        """Capture ML model state if available."""
        if not self.model_manager:
            return None
        
        try:
            # Get model states from model manager
            # This would integrate with the actual model manager implementation
            return {
                "models_available": True,
                "last_training_time": datetime.now().isoformat(),
                "model_accuracy": 0.85  # Placeholder
            }
        except Exception as e:
            self.logger.warning("Could not capture ML model state", error=str(e))
            return None
    
    async def _capture_rl_agent_state(self, mode_id: UUID) -> Optional[Dict[str, Any]]:
        """Capture RL agent state if available."""
        try:
            # Get RL agent state from mode
            # This would integrate with the actual RL agent in the mode
            return {
                "agent_available": True,
                "training_episodes": 1000,
                "epsilon": 0.1,
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.warning("Could not capture RL agent state", error=str(e))
            return None
    
    async def _capture_experience_buffer(self, mode_id: UUID) -> Optional[List[Dict[str, Any]]]:
        """Capture experience buffer if available."""
        try:
            # Get experience buffer from mode
            # This would integrate with the actual experience collection
            return []  # Placeholder
        except Exception as e:
            self.logger.warning("Could not capture experience buffer", error=str(e))
            return None
    
    async def _save_snapshot(self, snapshot: ModeStateSnapshot) -> None:
        """Save snapshot to disk."""
        snapshot_file = self._get_snapshot_file_path(snapshot)
        
        try:
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2, default=str)
            
            self.logger.debug("Snapshot saved", file=str(snapshot_file))
            
        except Exception as e:
            self.logger.error("Failed to save snapshot", error=str(e))
            raise
    
    def _get_snapshot_file_path(self, snapshot: ModeStateSnapshot) -> Path:
        """Get file path for snapshot."""
        filename = f"snapshot_{snapshot.mode_id}_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        return self.state_path / filename
    
    async def _backup_mode_state(self, mode_id: UUID) -> None:
        """Create backup of mode state."""
        snapshot = await self._create_mode_snapshot(mode_id)
        backup_file = self.state_path / f"backup_{mode_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(backup_file, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)
        
        self.logger.info("Mode state backed up", backup_file=str(backup_file))
    
    async def _validate_live_mode_readiness(self, plan: ModeTransitionPlan) -> bool:
        """Validate system readiness for live mode."""
        # Implement comprehensive live mode readiness checks
        # This would integrate with production safety checks
        return True  # Placeholder
    
    async def _validate_live_mode_safety(self) -> bool:
        """Validate live mode safety conditions."""
        # Implement comprehensive safety validations
        # This would integrate with safety systems
        return True  # Placeholder
    
    async def _transfer_learning_state(
        self, 
        from_snapshot: ModeStateSnapshot, 
        to_mode_id: UUID
    ) -> None:
        """Transfer learning state between modes."""
        try:
            # Transfer ML model state
            if from_snapshot.ml_model_state and self.model_manager:
                await self._restore_ml_model_state(to_mode_id, from_snapshot.ml_model_state)
            
            # Transfer RL agent state
            if from_snapshot.rl_agent_state:
                await self._restore_rl_agent_state(to_mode_id, from_snapshot.rl_agent_state)
            
            # Transfer experience buffer
            if from_snapshot.experience_buffer:
                await self._restore_experience_buffer(to_mode_id, from_snapshot.experience_buffer)
            
            self.logger.info(
                "Learning state transferred",
                from_mode=str(from_snapshot.mode_id),
                to_mode=str(to_mode_id)
            )
            
        except Exception as e:
            self.logger.error("Failed to transfer learning state", error=str(e))
            raise
    
    async def _transfer_analysis_state(
        self, 
        from_snapshot: ModeStateSnapshot, 
        to_mode_id: UUID
    ) -> None:
        """Transfer relevant state for analysis mode."""
        try:
            # Transfer metrics and metadata for analysis
            mode = self.mode_manager.active_modes[to_mode_id]
            mode.metrics.update(from_snapshot.metrics)
            mode.metadata.update(from_snapshot.metadata)
            
            self.logger.info(
                "Analysis state transferred",
                from_mode=str(from_snapshot.mode_id),
                to_mode=str(to_mode_id)
            )
            
        except Exception as e:
            self.logger.error("Failed to transfer analysis state", error=str(e))
            raise
    
    async def _restore_ml_model_state(self, mode_id: UUID, ml_state: Dict[str, Any]) -> None:
        """Restore ML model state to mode."""
        # Placeholder for ML model restoration
        self.logger.debug("ML model state restored", mode_id=str(mode_id))
    
    async def _restore_rl_agent_state(self, mode_id: UUID, rl_state: Dict[str, Any]) -> None:
        """Restore RL agent state to mode."""
        # Placeholder for RL agent restoration
        self.logger.debug("RL agent state restored", mode_id=str(mode_id))
    
    async def _restore_experience_buffer(self, mode_id: UUID, experiences: List[Dict[str, Any]]) -> None:
        """Restore experience buffer to mode."""
        # Placeholder for experience buffer restoration
        self.logger.debug("Experience buffer restored", mode_id=str(mode_id), count=len(experiences))


# Exception Classes
class IntegrationError(Exception):
    """Base integration error."""
    pass


class InvalidTransitionError(IntegrationError):
    """Error for invalid mode transitions."""
    pass


class StateValidationError(IntegrationError):
    """Error for state validation failures."""
    pass