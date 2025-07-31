"""
Model Preservation Integration

Integrates the preservation system with existing ML and RL components
"""

import asyncio
from typing import Optional, Dict, Any
import structlog

from src.ml_analysis.model_manager import ModelManager
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.modes.base import TradingMode
from .manager import PreservationManager
from .base import PreservationPriority


logger = structlog.get_logger()


class PreservationIntegration:
    """Integrates model preservation with the trading system"""
    
    def __init__(self, preservation_manager: PreservationManager):
        self.preservation_manager = preservation_manager
        self.logger = structlog.get_logger().bind(component="PreservationIntegration")
        
    def integrate_with_model_manager(self, model_manager: ModelManager):
        """Integrate preservation with ML model manager"""
        # Register the model manager
        self.preservation_manager.register_ml_manager(model_manager)
        
        # Patch save/load methods to use preservation
        original_save = model_manager._save_model
        original_load = model_manager.load_models
        
        async def preserved_save_model(model_type: str):
            """Enhanced save with preservation"""
            try:
                # Call original save
                await original_save(model_type)
                
                # Also preserve the model
                model = model_manager._models.get(model_type)
                if model:
                    # Use the standard save_model method
                    await self.preservation_manager.save_model(
                        model_type=model_type,
                        model_data=model,
                        mode="analysis"  # Default mode
                    )
                    
            except Exception as e:
                logger.error("Failed to preserve model", error=str(e))
                raise
        
        async def preserved_load_models(model_types=None):
            """Enhanced load with preservation fallback"""
            try:
                # Try original load first
                result = await original_load(model_types)
                
                # If any models failed to load, try preservation
                for model_type, success in result.items():
                    if not success:
                        try:
                            model_data, metadata = await self.preservation_manager.load_ml_model(
                                model_type=model_type
                            )
                            
                            # Restore the model
                            if model_type in model_manager._models:
                                model = model_manager._models[model_type]
                                if hasattr(model, 'load_from_checkpoint'):
                                    model.load_from_checkpoint(model_data)
                                    result[model_type] = True
                                    logger.info(
                                        "Model loaded from preservation",
                                        model_type=model_type.value,
                                        version=metadata.version
                                    )
                        except Exception as e:
                            logger.error(
                                "Failed to load from preservation",
                                model_type=model_type.value,
                                error=str(e)
                            )
                
                return result
                
            except Exception as e:
                logger.error("Failed to load models", error=str(e))
                raise
        
        # Replace methods
        model_manager._save_model = preserved_save_model
        model_manager.load_models = preserved_load_models
        
        self.logger.info("Model manager integrated with preservation")
    
    def integrate_with_rl_agent(self, agent_id: str, agent: DQNTradingAgent):
        """Integrate preservation with RL agent"""
        # Register the agent
        self.preservation_manager.register_rl_agent(agent_id, agent)
        
        # Patch save/load methods
        original_save = agent.save_model
        original_load = agent.load_model
        
        def preserved_save_model(filepath: str) -> bool:
            """Enhanced save with preservation"""
            try:
                # Call original save
                success = original_save(filepath)
                
                if success:
                    # Also preserve the model
                    asyncio.create_task(
                        self.preservation_manager.save_model(
                            model_type=agent_id,
                            model_data=agent,
                            mode="simulation"  # Default mode for RL
                        )
                    )
                
                return success
                
            except Exception as e:
                logger.error("Failed to preserve RL agent", error=str(e))
                return False
        
        def preserved_load_model(filepath: str) -> bool:
            """Enhanced load with preservation fallback"""
            try:
                # Try original load first
                success = original_load(filepath)
                
                if not success:
                    # Try loading from preservation
                    async def load_from_preservation():
                        try:
                            result = await self.preservation_manager.load_model(
                                model_type=agent_id,
                                mode="simulation"
                            )
                            
                            if result:
                                logger.info(
                                    "RL agent loaded from preservation",
                                    agent_id=agent_id
                                )
                                return True
                            return False
                            
                        except Exception as e:
                            logger.error(
                                "Failed to load RL agent from preservation",
                                agent_id=agent_id,
                                error=str(e)
                            )
                            return False
                    
                    # Run async load
                    loop = asyncio.get_event_loop()
                    success = loop.run_until_complete(load_from_preservation())
                
                return success
                
            except Exception as e:
                logger.error("Failed to load RL agent", error=str(e))
                return False
        
        # Replace methods
        agent.save_model = preserved_save_model
        agent.load_model = preserved_load_model
        
        self.logger.info("RL agent integrated with preservation", agent_id=agent_id)
    
    def integrate_with_trading_mode(self, trading_mode: TradingMode):
        """Integrate preservation with trading mode changes"""
        original_enter = trading_mode.enter_mode
        original_exit = trading_mode.exit_mode
        
        async def preserved_enter_mode():
            """Enhanced mode entry with model loading"""
            # Set preservation mode (method might not exist, wrap in try/catch)
            try:
                await self.preservation_manager.set_mode(trading_mode.mode_name)
            except AttributeError:
                # Method doesn't exist, skip
                pass
            
            # Load mode-specific models if available
            try:
                # Try to load ML models for this mode  
                ml_models = await self.preservation_manager.list_models(
                    mode=trading_mode.mode_name
                )
                
                if ml_models:
                    logger.info(
                        "Found preserved models for mode",
                        mode=trading_mode.mode_name,
                        count=len(ml_models)
                    )
                    
            except Exception as e:
                logger.error(
                    "Failed to check preserved models",
                    mode=trading_mode.mode_name,
                    error=str(e)
                )
            
            # Call original enter
            await original_enter()
        
        async def preserved_exit_mode():
            """Enhanced mode exit with model preservation"""
            # Preserve current models before exiting
            try:
                # Trigger emergency backup 
                await self.preservation_manager.emergency_backup()
            except Exception as e:
                logger.error(
                    "Failed to preserve models on mode exit",
                    mode=trading_mode.mode_name,
                    error=str(e)
                )
            
            # Call original exit
            await original_exit()
        
        # Replace methods
        trading_mode.enter_mode = preserved_enter_mode
        trading_mode.exit_mode = preserved_exit_mode
        
        self.logger.info(
            "Trading mode integrated with preservation",
            mode=trading_mode.mode_name
        )


async def setup_preservation_hooks(app):
    """Setup preservation hooks in the FastAPI app"""
    from . import get_preservation_manager
    
    preservation_manager = get_preservation_manager()
    integration = PreservationIntegration(preservation_manager)
    
    # Store integration in app state
    app.state.preservation_integration = integration
    
    # Add preservation health endpoint
    @app.get("/api/preservation/health")
    async def get_preservation_health():
        """Get preservation system health status"""
        return await preservation_manager.get_preservation_health()
    
    # Add model listing endpoint
    @app.get("/api/preservation/models")
    async def list_preserved_models(
        model_type: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 50
    ):
        """List available preserved models"""
        model_type_str = model_type if model_type else None
        models = await preservation_manager.list_models(
            model_type=model_type_str,
            mode=mode,
            limit=limit
        )
        
        return {
            "models": [
                {
                    "preservation_id": m.get("model_id", ""),
                    "model_type": m.get("model_type", ""),
                    "version": m.get("version", ""),
                    "mode": m.get("mode", ""),
                    "preserved_at": m.get("created_at", "").isoformat() if m.get("created_at") else "",
                    "size_mb": m.get("size_mb", 0.0),
                    "priority": "normal",
                    "reason": "preservation"
                }
                for m in models
            ],
            "total": len(models)
        }
    
    # Add rollback endpoint
    @app.post("/api/preservation/rollback/{model_type}")
    async def rollback_model(
        model_type: str,
        target_version: Optional[str] = None,
        target_date: Optional[str] = None
    ):
        """Rollback to a previous model version"""
        try:
            model_type_str = model_type
            target_datetime = None
            if target_date:
                from datetime import datetime
                target_datetime = datetime.fromisoformat(target_date)
            
            success = await preservation_manager.rollback_model(
                model_type=model_type_str,
                target_version=target_version
            )
            
            return {
                "success": success,
                "rolled_back_to": {
                    "version": target_version,
                    "target_date": target_date,
                    "model_type": model_type_str
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    logger.info("Preservation hooks setup complete")