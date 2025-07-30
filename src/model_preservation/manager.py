"""
Model Preservation Manager

Orchestrates model preservation across the system with:
- Graceful shutdown handling
- Automated backups
- Model versioning
- Rollback capabilities
- Mode-specific isolation
"""

import asyncio
import pickle
import signal
import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import structlog
import torch
import numpy as np

from .base import (
    ModelMetadata, ModelType, PreservationConfig, PreservationPriority,
    PreservationError, BackupError, RestoreError
)
from .gcs_handler import GCSModelPreservationHandler
from src.ml_analysis.model_manager import ModelManager
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.utils.config import get_config
from src.activity_logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity
)


logger = structlog.get_logger()


class ModelPreservationManager:
    """
    Central manager for model preservation across the system
    
    Handles:
    - Automatic versioning
    - Emergency backups
    - Graceful shutdown preservation
    - Mode-specific model isolation
    - Cross-mode model migration
    """
    
    def __init__(self, config: Optional[PreservationConfig] = None):
        self.logger = structlog.get_logger().bind(component="ModelPreservationManager")
        
        # Load config
        if config is None:
            app_config = get_config()
            config = self._build_preservation_config(app_config)
        
        self.config = config
        self.handler = GCSModelPreservationHandler(config)
        
        # Model registry
        self._ml_manager: Optional[ModelManager] = None
        self._rl_agents: Dict[str, DQNTradingAgent] = {}
        
        # State tracking
        self._current_mode = "analysis"
        self._is_shutting_down = False
        self._auto_backup_task: Optional[asyncio.Task] = None
        self._version_tracker: Dict[str, Dict[str, int]] = {}  # model_type -> version info
        
        # Register shutdown handlers
        self._register_shutdown_handlers()
    
    def _build_preservation_config(self, app_config) -> PreservationConfig:
        """Build preservation config from app config"""
        return PreservationConfig(
            gcs_bucket=app_config.apis.get("gcs", {}).get("bucket", "shyvr-models"),
            gcs_prefix="models",
            max_backups_per_model=10,
            backup_retention_days=30,
            emergency_backup_retention_days=90,
            versioning_enabled=True,
            compression_enabled=True,
            isolate_by_mode=True,
            emergency_backup_enabled=True,
            emergency_backup_interval_minutes=5
        )
    
    def _register_shutdown_handlers(self):
        """Register signal handlers for graceful shutdown"""
        def shutdown_handler(signum, frame):
            self.logger.info("Shutdown signal received", signal=signum)
            asyncio.create_task(self._emergency_shutdown())
        
        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)
    
    async def start(self):
        """Start preservation services"""
        try:
            await self.handler.start()
            
            # Start auto-backup task
            self._auto_backup_task = asyncio.create_task(self._auto_backup_loop())
            
            # Log startup
            await activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.START,
                source="model_preservation",
                event_type="preservation_started",
                title="Model preservation system started",
                severity=ActivitySeverity.INFO,
                metadata={
                    "config": {
                        "gcs_bucket": self.config.gcs_bucket,
                        "versioning_enabled": self.config.versioning_enabled,
                        "emergency_backup_enabled": self.config.emergency_backup_enabled,
                        "isolate_by_mode": self.config.isolate_by_mode
                    }
                }
            )
            
            self.logger.info("Model preservation manager started")
            
        except Exception as e:
            self.logger.error("Failed to start preservation manager", error=str(e))
            raise
    
    async def stop(self):
        """Stop preservation services gracefully"""
        try:
            self._is_shutting_down = True
            
            # Cancel auto-backup task
            if self._auto_backup_task:
                self._auto_backup_task.cancel()
                try:
                    await self._auto_backup_task
                except asyncio.CancelledError:
                    pass
            
            # Perform final backup
            await self._perform_shutdown_backup()
            
            # Stop handler
            await self.handler.stop()
            
            self.logger.info("Model preservation manager stopped")
            
        except Exception as e:
            self.logger.error("Error during preservation shutdown", error=str(e))
    
    def register_ml_manager(self, ml_manager: ModelManager):
        """Register ML model manager for preservation"""
        self._ml_manager = ml_manager
        self.logger.info("ML manager registered for preservation")
    
    def register_rl_agent(self, agent_id: str, agent: DQNTradingAgent):
        """Register RL agent for preservation"""
        self._rl_agents[agent_id] = agent
        self.logger.info("RL agent registered", agent_id=agent_id)
    
    async def set_mode(self, mode: str):
        """Set current operational mode"""
        if mode not in ["analysis", "simulation", "live"]:
            raise ValueError(f"Invalid mode: {mode}")
        
        if mode != self._current_mode:
            # Backup current mode models before switching
            await self._backup_models_for_mode_change(self._current_mode, mode)
            
        self._current_mode = mode
        self.logger.info("Mode changed", new_mode=mode)
    
    async def save_ml_model(
        self,
        model_type: ModelType,
        model_data: Any,
        performance_metrics: Optional[Dict[str, Any]] = None,
        reason: str = "manual_save",
        priority: PreservationPriority = PreservationPriority.NORMAL
    ) -> str:
        """Save an ML model with versioning"""
        try:
            # Serialize model
            if isinstance(model_data, torch.nn.Module):
                # PyTorch model
                buffer = io.BytesIO()
                torch.save({
                    'model_state_dict': model_data.state_dict(),
                    'model_config': getattr(model_data, 'config', {})
                }, buffer)
                serialized_data = buffer.getvalue()
            else:
                # Generic Python object
                serialized_data = pickle.dumps(model_data)
            
            # Get version
            version = self._get_next_version(model_type)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=f"{model_type.value}_{self._current_mode}",
                model_type=model_type,
                version=version,
                mode=self._current_mode,
                created_at=datetime.utcnow(),
                preserved_at=datetime.utcnow(),
                checksum="",  # Will be calculated by handler
                size_bytes=0,  # Will be calculated by handler
                performance_metrics=performance_metrics or {},
                preservation_reason=reason,
                priority=priority,
                tags=[self._current_mode, reason]
            )
            
            # Save using handler
            preservation_id = await self.handler.save_model(
                serialized_data,
                metadata,
                priority
            )
            
            # Log activity
            await activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.SAVE,
                source="model_preservation",
                event_type="ml_model_saved",
                title=f"ML model saved: {model_type.value}",
                severity=ActivitySeverity.INFO,
                metadata={
                    "preservation_id": preservation_id,
                    "model_type": model_type.value,
                    "version": version,
                    "mode": self._current_mode,
                    "reason": reason,
                    "size_mb": len(serialized_data) / 1024 / 1024
                }
            )
            
            return preservation_id
            
        except Exception as e:
            self.logger.error(
                "Failed to save ML model",
                model_type=model_type.value,
                error=str(e)
            )
            raise BackupError(f"Failed to save ML model: {str(e)}")
    
    async def save_rl_agent(
        self,
        agent_id: str,
        agent: Optional[DQNTradingAgent] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        reason: str = "manual_save",
        priority: PreservationPriority = PreservationPriority.NORMAL
    ) -> str:
        """Save an RL agent with versioning"""
        try:
            # Get agent if not provided
            if agent is None:
                agent = self._rl_agents.get(agent_id)
                if not agent:
                    raise ValueError(f"Unknown agent: {agent_id}")
            
            # Get agent state
            agent_state = {
                'model_state': agent.save_model_state(),
                'config': agent.config,
                'training_info': {
                    'episodes': getattr(agent, 'training_episodes', 0),
                    'epsilon': getattr(agent, 'epsilon', 0),
                    'replay_buffer_size': len(getattr(agent, 'replay_buffer', []))
                }
            }
            
            # Serialize
            serialized_data = pickle.dumps(agent_state)
            
            # Get version
            version = self._get_next_version(ModelType.DQN)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=f"rl_agent_{agent_id}",
                model_type=ModelType.DQN,
                version=version,
                mode=self._current_mode,
                created_at=datetime.utcnow(),
                preserved_at=datetime.utcnow(),
                checksum="",
                size_bytes=0,
                performance_metrics=performance_metrics or {},
                training_info=agent_state['training_info'],
                preservation_reason=reason,
                priority=priority,
                tags=[self._current_mode, reason, agent_id]
            )
            
            # Save using handler
            preservation_id = await self.handler.save_model(
                serialized_data,
                metadata,
                priority
            )
            
            # Log activity
            await activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.SAVE,
                source="model_preservation",
                event_type="rl_agent_saved",
                title=f"RL agent saved: {agent_id}",
                severity=ActivitySeverity.INFO,
                metadata={
                    "preservation_id": preservation_id,
                    "agent_id": agent_id,
                    "version": version,
                    "mode": self._current_mode,
                    "reason": reason,
                    "training_episodes": agent_state['training_info']['episodes']
                }
            )
            
            return preservation_id
            
        except Exception as e:
            self.logger.error(
                "Failed to save RL agent",
                agent_id=agent_id,
                error=str(e)
            )
            raise BackupError(f"Failed to save RL agent: {str(e)}")
    
    async def load_ml_model(
        self,
        model_type: ModelType,
        version: Optional[str] = None,
        mode: Optional[str] = None
    ) -> Tuple[Any, ModelMetadata]:
        """Load an ML model"""
        try:
            # Find latest model if version not specified
            if not version:
                models = await self.handler.list_models(
                    model_type=model_type,
                    mode=mode or self._current_mode,
                    limit=1
                )
                if not models:
                    raise RestoreError(f"No models found for {model_type.value}")
                
                # Use latest model
                latest_model = models[0]
                preservation_id = self._generate_preservation_id_from_metadata(latest_model)
            else:
                # Find specific version
                models = await self.handler.list_models(
                    model_type=model_type,
                    mode=mode or self._current_mode,
                    version=version,
                    limit=1
                )
                if not models:
                    raise RestoreError(
                        f"Model not found: {model_type.value} v{version}"
                    )
                
                preservation_id = self._generate_preservation_id_from_metadata(models[0])
            
            # Load model
            model_data, metadata = await self.handler.load_model(preservation_id)
            
            # Deserialize
            if model_type == ModelType.LSTM:
                # PyTorch model
                buffer = io.BytesIO(model_data)
                checkpoint = torch.load(buffer, map_location='cpu')
                model = checkpoint  # Return checkpoint for model manager to handle
            else:
                # Generic Python object
                model = pickle.loads(model_data)
            
            self.logger.info(
                "ML model loaded",
                model_type=model_type.value,
                version=metadata.version,
                mode=metadata.mode
            )
            
            return model, metadata
            
        except Exception as e:
            self.logger.error(
                "Failed to load ML model",
                model_type=model_type.value,
                error=str(e)
            )
            raise RestoreError(f"Failed to load ML model: {str(e)}")
    
    async def load_rl_agent(
        self,
        agent_id: str,
        version: Optional[str] = None,
        mode: Optional[str] = None
    ) -> Tuple[Dict[str, Any], ModelMetadata]:
        """Load an RL agent"""
        try:
            # Find agent models
            tags = [agent_id]
            models = await self.handler.list_models(
                model_type=ModelType.DQN,
                mode=mode or self._current_mode,
                version=version,
                tags=tags,
                limit=1
            )
            
            if not models:
                raise RestoreError(f"No models found for agent: {agent_id}")
            
            # Load model
            preservation_id = self._generate_preservation_id_from_metadata(models[0])
            model_data, metadata = await self.handler.load_model(preservation_id)
            
            # Deserialize
            agent_state = pickle.loads(model_data)
            
            self.logger.info(
                "RL agent loaded",
                agent_id=agent_id,
                version=metadata.version,
                mode=metadata.mode
            )
            
            return agent_state, metadata
            
        except Exception as e:
            self.logger.error(
                "Failed to load RL agent",
                agent_id=agent_id,
                error=str(e)
            )
            raise RestoreError(f"Failed to load RL agent: {str(e)}")
    
    async def rollback_model(
        self,
        model_type: ModelType,
        target_version: Optional[str] = None,
        target_date: Optional[datetime] = None
    ) -> Tuple[Any, ModelMetadata]:
        """Rollback to a previous model version"""
        try:
            # Perform rollback
            model_data, metadata = await self.handler.rollback_model(
                model_type=model_type,
                target_version=target_version,
                target_date=target_date
            )
            
            # Deserialize based on type
            if model_type in [ModelType.LSTM, ModelType.DQN]:
                model = pickle.loads(model_data)
            else:
                model = model_data
            
            # Log rollback
            await activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.UPDATE,
                source="model_preservation",
                event_type="model_rollback",
                title=f"Model rolled back: {model_type.value}",
                severity=ActivitySeverity.WARNING,
                metadata={
                    "model_type": model_type.value,
                    "rolled_back_to": metadata.version,
                    "target_version": target_version,
                    "target_date": target_date.isoformat() if target_date else None
                }
            )
            
            return model, metadata
            
        except Exception as e:
            self.logger.error(
                "Failed to rollback model",
                model_type=model_type.value,
                error=str(e)
            )
            raise
    
    async def list_available_models(
        self,
        model_type: Optional[ModelType] = None,
        mode: Optional[str] = None,
        limit: int = 50
    ) -> List[ModelMetadata]:
        """List available preserved models"""
        return await self.handler.list_models(
            model_type=model_type,
            mode=mode,
            limit=limit
        )
    
    async def get_preservation_health(self) -> Dict[str, Any]:
        """Get preservation system health status"""
        health = await self.handler.health_check()
        
        # Add manager-specific info
        health['manager'] = {
            'current_mode': self._current_mode,
            'registered_ml_models': self._ml_manager is not None,
            'registered_rl_agents': len(self._rl_agents),
            'auto_backup_active': self._auto_backup_task and not self._auto_backup_task.done(),
            'is_shutting_down': self._is_shutting_down
        }
        
        return health
    
    async def _auto_backup_loop(self):
        """Background task for automatic backups"""
        while not self._is_shutting_down:
            try:
                # Wait for interval
                await asyncio.sleep(self.config.emergency_backup_interval_minutes * 60)
                
                if not self._is_shutting_down:
                    await self._perform_auto_backup()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Auto-backup failed", error=str(e))
    
    async def _perform_auto_backup(self):
        """Perform automatic backup of active models"""
        try:
            backup_count = 0
            
            # Backup ML models
            if self._ml_manager:
                for model_type in [ModelType.LSTM, ModelType.ENSEMBLE]:
                    try:
                        # Get model performance
                        perf = self._ml_manager.get_model_performance()
                        
                        # Save if model is trained
                        if self._ml_manager._models.get(model_type):
                            await self.save_ml_model(
                                model_type=model_type,
                                model_data=self._ml_manager._models[model_type],
                                performance_metrics=perf.get('performance', {}).get(model_type, {}),
                                reason="auto_backup",
                                priority=PreservationPriority.NORMAL
                            )
                            backup_count += 1
                    except Exception as e:
                        self.logger.error(
                            "Failed to auto-backup ML model",
                            model_type=model_type.value,
                            error=str(e)
                        )
            
            # Backup RL agents
            for agent_id, agent in self._rl_agents.items():
                try:
                    await self.save_rl_agent(
                        agent_id=agent_id,
                        agent=agent,
                        reason="auto_backup",
                        priority=PreservationPriority.NORMAL
                    )
                    backup_count += 1
                except Exception as e:
                    self.logger.error(
                        "Failed to auto-backup RL agent",
                        agent_id=agent_id,
                        error=str(e)
                    )
            
            if backup_count > 0:
                self.logger.info(
                    "Auto-backup completed",
                    models_backed_up=backup_count
                )
                
        except Exception as e:
            self.logger.error("Auto-backup failed", error=str(e))
    
    async def _perform_shutdown_backup(self):
        """Perform emergency backup during shutdown"""
        try:
            self.logger.info("Performing shutdown backup")
            
            # Backup all models with CRITICAL priority
            if self._ml_manager:
                for model_type, model in self._ml_manager._models.items():
                    if model and model.is_model_trained():
                        await self.save_ml_model(
                            model_type=model_type,
                            model_data=model,
                            reason="shutdown_backup",
                            priority=PreservationPriority.CRITICAL
                        )
            
            # Backup all RL agents
            for agent_id, agent in self._rl_agents.items():
                await self.save_rl_agent(
                    agent_id=agent_id,
                    agent=agent,
                    reason="shutdown_backup",
                    priority=PreservationPriority.CRITICAL
                )
            
            self.logger.info("Shutdown backup completed")
            
        except Exception as e:
            self.logger.error("Shutdown backup failed", error=str(e))
    
    async def _backup_models_for_mode_change(self, old_mode: str, new_mode: str):
        """Backup models when changing modes"""
        try:
            self.logger.info(
                "Backing up models for mode change",
                old_mode=old_mode,
                new_mode=new_mode
            )
            
            # Backup all active models with HIGH priority
            if self._ml_manager:
                for model_type, model in self._ml_manager._models.items():
                    if model and model.is_model_trained():
                        await self.save_ml_model(
                            model_type=model_type,
                            model_data=model,
                            reason=f"mode_change_{old_mode}_to_{new_mode}",
                            priority=PreservationPriority.HIGH
                        )
            
            # Backup RL agents
            for agent_id, agent in self._rl_agents.items():
                await self.save_rl_agent(
                    agent_id=agent_id,
                    agent=agent,
                    reason=f"mode_change_{old_mode}_to_{new_mode}",
                    priority=PreservationPriority.HIGH
                )
            
        except Exception as e:
            self.logger.error(
                "Mode change backup failed",
                old_mode=old_mode,
                new_mode=new_mode,
                error=str(e)
            )
    
    async def _emergency_shutdown(self):
        """Emergency shutdown handler"""
        self._is_shutting_down = True
        
        try:
            # Perform critical backup
            await self._perform_shutdown_backup()
            
            # Log emergency shutdown
            await activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.STOP,
                source="model_preservation",
                event_type="emergency_shutdown",
                title="Emergency model preservation during shutdown",
                severity=ActivitySeverity.CRITICAL
            )
            
        except Exception as e:
            self.logger.error("Emergency shutdown failed", error=str(e))
    
    def _get_next_version(self, model_type: ModelType) -> str:
        """Get next version number for a model type"""
        if not self.config.versioning_enabled:
            return "latest"
        
        # Initialize version tracker if needed
        if model_type.value not in self._version_tracker:
            self._version_tracker[model_type.value] = {
                'major': 1,
                'minor': 0,
                'patch': 0
            }
        
        version_info = self._version_tracker[model_type.value]
        
        # Auto-increment patch version
        if self.config.auto_increment_version:
            version_info['patch'] += 1
        
        # Format version
        version = self.config.version_format.format(**version_info)
        
        return version
    
    def _generate_preservation_id_from_metadata(self, metadata: ModelMetadata) -> str:
        """Generate preservation ID from metadata"""
        components = [
            metadata.model_type.value,
            metadata.mode,
            metadata.version,
            metadata.preserved_at.isoformat()
        ]
        return "_".join(components).replace(":", "-")