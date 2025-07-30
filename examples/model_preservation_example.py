"""
Model Preservation System Usage Examples

This file demonstrates how to use the comprehensive model preservation
architecture in the Shyvr RLTE system.
"""

import asyncio
from datetime import datetime, timedelta
import torch
import torch.nn as nn

from src.model_preservation import (
    ModelPreservationManager,
    PreservationConfig,
    ModelType,
    PreservationPriority,
    get_preservation_manager,
    initialize_preservation_system,
    shutdown_preservation_system
)
from src.model_preservation.integration import PreservationIntegration
from src.ml_analysis.model_manager import ModelManager
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.rl_agent.base import AgentConfig, ModelType as RLModelType


async def example_basic_preservation():
    """Example: Basic model preservation operations"""
    print("\n=== Basic Model Preservation Example ===")
    
    # Initialize preservation system
    config = PreservationConfig(
        gcs_bucket="shyvr-models-dev",
        gcs_prefix="examples",
        versioning_enabled=True,
        compression_enabled=True,
        isolate_by_mode=True
    )
    
    manager = ModelPreservationManager(config)
    await manager.start()
    
    try:
        # Set current mode
        await manager.set_mode("simulation")
        
        # Create a dummy PyTorch model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        model = DummyModel()
        
        # Save the model
        preservation_id = await manager.save_ml_model(
            model_type=ModelType.LSTM,
            model_data=model,
            performance_metrics={
                "accuracy": 0.95,
                "loss": 0.05,
                "training_time": 3600
            },
            reason="example_save",
            priority=PreservationPriority.NORMAL
        )
        
        print(f"Model saved with ID: {preservation_id}")
        
        # List available models
        models = await manager.list_available_models(
            model_type=ModelType.LSTM,
            mode="simulation"
        )
        
        print(f"\nAvailable models: {len(models)}")
        for m in models:
            print(f"  - {m.version} ({m.preserved_at})")
        
        # Load the model back
        loaded_model, metadata = await manager.load_ml_model(
            model_type=ModelType.LSTM,
            mode="simulation"
        )
        
        print(f"\nModel loaded: {metadata.version}")
        print(f"Performance metrics: {metadata.performance_metrics}")
        
    finally:
        await manager.stop()


async def example_mode_changes():
    """Example: Handling mode changes with preservation"""
    print("\n=== Mode Change Preservation Example ===")
    
    manager = get_preservation_manager()
    await manager.start()
    
    try:
        # Start in analysis mode
        await manager.set_mode("analysis")
        print("Mode set to: analysis")
        
        # Save a model in analysis mode
        preservation_id = await manager.save_ml_model(
            model_type=ModelType.ENSEMBLE,
            model_data={"dummy": "model"},
            reason="analysis_model",
            priority=PreservationPriority.NORMAL
        )
        print(f"Analysis model saved: {preservation_id}")
        
        # Change to simulation mode (triggers backup)
        await manager.set_mode("simulation")
        print("\nMode changed to: simulation")
        
        # Save a model in simulation mode
        preservation_id = await manager.save_ml_model(
            model_type=ModelType.ENSEMBLE,
            model_data={"dummy": "simulation_model"},
            reason="simulation_model",
            priority=PreservationPriority.NORMAL
        )
        print(f"Simulation model saved: {preservation_id}")
        
        # List models for each mode
        for mode in ["analysis", "simulation"]:
            models = await manager.list_available_models(mode=mode)
            print(f"\nModels in {mode} mode: {len(models)}")
            
    finally:
        await manager.stop()


async def example_emergency_backup():
    """Example: Emergency backup and recovery"""
    print("\n=== Emergency Backup Example ===")
    
    # Configure with emergency backup enabled
    config = PreservationConfig(
        gcs_bucket="shyvr-models-dev",
        gcs_prefix="emergency",
        emergency_backup_enabled=True,
        emergency_backup_interval_minutes=0.1  # 6 seconds for demo
    )
    
    manager = ModelPreservationManager(config)
    await manager.start()
    
    try:
        # Save a critical model
        preservation_id = await manager.save_ml_model(
            model_type=ModelType.DQN,
            model_data={"critical": "model_data"},
            reason="emergency_test",
            priority=PreservationPriority.CRITICAL
        )
        print(f"Critical model saved: {preservation_id}")
        
        # Wait for auto-backup
        print("\nWaiting for emergency backup...")
        await asyncio.sleep(10)
        
        # Check preservation health
        health = await manager.get_preservation_health()
        print(f"\nPreservation system health: {health['status']}")
        print(f"Total models: {health.get('total_models', 0)}")
        
    finally:
        # Trigger shutdown backup
        print("\nTriggering shutdown backup...")
        await manager.stop()


async def example_version_rollback():
    """Example: Model versioning and rollback"""
    print("\n=== Version Rollback Example ===")
    
    manager = get_preservation_manager()
    await manager.start()
    
    try:
        # Save multiple versions of a model
        for i in range(3):
            preservation_id = await manager.save_ml_model(
                model_type=ModelType.LSTM,
                model_data={f"version": i+1},
                performance_metrics={"accuracy": 0.90 + i*0.02},
                reason=f"version_{i+1}"
            )
            print(f"Saved version {i+1}: {preservation_id}")
            await asyncio.sleep(1)  # Ensure different timestamps
        
        # List all versions
        models = await manager.list_available_models(
            model_type=ModelType.LSTM,
            limit=10
        )
        print(f"\nAll versions ({len(models)}):")
        for m in models:
            print(f"  - {m.version}: accuracy={m.performance_metrics.get('accuracy')}")
        
        # Rollback to version 2
        print("\nRolling back to v1.0.1...")
        rolled_back_model, metadata = await manager.rollback_model(
            model_type=ModelType.LSTM,
            target_version="v1.0.1"
        )
        print(f"Rolled back to: {metadata.version}")
        print(f"Model data: {rolled_back_model}")
        
        # Rollback by date
        target_date = datetime.utcnow() - timedelta(minutes=1)
        print(f"\nRolling back to models before {target_date}...")
        rolled_back_model, metadata = await manager.rollback_model(
            model_type=ModelType.LSTM,
            target_date=target_date
        )
        print(f"Rolled back to: {metadata.version}")
        
    finally:
        await manager.stop()


async def example_integration():
    """Example: Integration with existing components"""
    print("\n=== Integration Example ===")
    
    # Initialize preservation
    await initialize_preservation_system()
    manager = get_preservation_manager()
    integration = PreservationIntegration(manager)
    
    try:
        # Create and integrate ML model manager
        ml_config = {"lstm": {"hidden_size": 128}}
        model_manager = ModelManager(ml_config)
        integration.integrate_with_model_manager(model_manager)
        print("Model manager integrated")
        
        # Create and integrate RL agent
        agent_config = AgentConfig(
            model_type=RLModelType.DQN,
            learning_rate=0.001,
            batch_size=32,
            replay_buffer_size=1000
        )
        rl_agent = DQNTradingAgent(agent_config)
        integration.integrate_with_rl_agent("agent_001", rl_agent)
        print("RL agent integrated")
        
        # Now saves will automatically preserve models
        print("\nTesting integrated saves...")
        
        # The model manager's save will now also preserve
        await model_manager._save_model(ModelType.LSTM)
        
        # The RL agent's save will now also preserve
        rl_agent.save_model("dummy_path.pt")
        
        # Check preserved models
        await asyncio.sleep(2)  # Wait for async preservations
        
        models = await manager.list_available_models()
        print(f"\nTotal preserved models: {len(models)}")
        
    finally:
        await shutdown_preservation_system()


async def example_gcs_operations():
    """Example: Direct GCS operations and health monitoring"""
    print("\n=== GCS Operations Example ===")
    
    config = PreservationConfig(
        gcs_bucket="shyvr-models-dev",
        gcs_prefix="gcs-example",
        compression_enabled=True,
        chunk_size_mb=10,
        max_concurrent_operations=3
    )
    
    manager = ModelPreservationManager(config)
    await manager.start()
    
    try:
        # Save models with different priorities
        for priority in [PreservationPriority.LOW, 
                        PreservationPriority.NORMAL, 
                        PreservationPriority.HIGH,
                        PreservationPriority.CRITICAL]:
            
            preservation_id = await manager.save_ml_model(
                model_type=ModelType.ENSEMBLE,
                model_data={"priority": priority.value},
                reason=f"priority_test_{priority.value}",
                priority=priority
            )
            print(f"Saved with {priority.value} priority: {preservation_id}")
        
        # Check GCS health
        health = await manager.handler.health_check()
        print(f"\nGCS Health Status:")
        print(f"  - Status: {health['status']}")
        print(f"  - GCS Accessible: {health['gcs_accessible']}")
        print(f"  - DB Accessible: {health['database_accessible']}")
        print(f"  - Storage Used: {health['storage_used_mb']:.2f} MB")
        print(f"  - Total Models: {health['total_models']}")
        
        # Test soft delete
        models = await manager.list_available_models(limit=1)
        if models:
            preservation_id = manager._generate_preservation_id_from_metadata(models[0])
            success = await manager.handler.delete_model(preservation_id, soft_delete=True)
            print(f"\nSoft deleted model: {preservation_id} - Success: {success}")
        
    finally:
        await manager.stop()


# Main execution
async def main():
    """Run all examples"""
    examples = [
        example_basic_preservation,
        example_mode_changes,
        example_emergency_backup,
        example_version_rollback,
        example_integration,
        example_gcs_operations
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"\nExample failed: {e}")
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())