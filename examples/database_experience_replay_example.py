#!/usr/bin/env python3
"""
Example demonstrating database-backed experience replay buffer usage.

This example shows how to use the new DatabaseExperienceReplayBuffer
that integrates with the PostgreSQL database for persistent storage
while maintaining backward compatibility with existing APIs.
"""

import asyncio
import numpy as np
from datetime import datetime

from src.rl_agent.experience_replay import (
    Experience, ReplayBufferConfig, DatabaseExperienceReplayBuffer,
    create_database_replay_buffer
)


async def main():
    """Demonstrate database experience replay buffer usage"""
    
    print("=== Database Experience Replay Buffer Example ===")
    
    # Create configuration for database buffer
    config = ReplayBufferConfig(
        max_size=1000,
        batch_size=32,
        min_size=50,
        prioritized=True,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0
    )
    
    # Create database-backed buffer using factory function
    buffer = create_database_replay_buffer(config)
    
    print(f"Created database buffer with max_size={config.max_size}")
    
    # Initialize the database connection
    try:
        await buffer.initialize()
        print("Database buffer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize database buffer: {e}")
        print("Note: This example requires database setup and connection")
        return
    
    # Create sample experiences
    experiences = []
    for i in range(100):
        # Create random state and next_state vectors
        state = np.random.randn(10).astype(np.float32)
        next_state = state + np.random.randn(10).astype(np.float32) * 0.1
        
        experience = Experience(
            state=state,
            action=np.random.randint(0, 4),
            reward=np.random.uniform(-1, 1),
            next_state=next_state,
            done=np.random.random() < 0.1,  # 10% chance of episode end
            timestamp=datetime.now()
        )
        experiences.append(experience)
    
    print(f"Created {len(experiences)} sample experiences")
    
    # Add experiences using async batch method (recommended)
    await buffer.add_batch_async(experiences)
    print(f"Added {len(experiences)} experiences to database buffer")
    
    # Check buffer status
    size = await buffer.size_async()
    can_sample = await buffer.can_sample_async()
    print(f"Buffer size: {size}, can sample: {can_sample}")
    
    if can_sample:
        # Sample a batch of experiences
        batch = await buffer.sample_async()
        print(f"Sampled batch of {len(batch)} experiences")
        
        # Print sample experience
        if batch:
            sample_exp = batch[0]
            print("Sample experience:")
            print(f"  Action: {sample_exp['action']}")
            print(f"  Reward: {sample_exp['reward']:.4f}")
            print(f"  Done: {sample_exp['done']}")
            print(f"  State shape: {sample_exp['state'].shape}")
            
            # For prioritized replay, we get additional fields
            if 'weight' in sample_exp:
                print(f"  Importance weight: {sample_exp['weight']:.4f}")
                print(f"  Database index: {sample_exp['index']}")
    
    # Get performance metrics
    metrics = await buffer.get_performance_metrics_async()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Get comprehensive statistics
    stats = await buffer.get_statistics_async()
    print("\nBuffer Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Get recent experiences
    recent = await buffer.get_recent_async(limit=5)
    print(f"\nRetrieved {len(recent)} recent experiences")
    
    # Cleanup resources
    await buffer.cleanup()
    print("Database buffer cleaned up")
    
    print("\n=== Example completed successfully ===")


def synchronous_example():
    """Example showing backward compatibility with synchronous API"""
    print("\n=== Synchronous API Example (Backward Compatibility) ===")
    
    config = ReplayBufferConfig(max_size=100, batch_size=16, min_size=10)
    buffer = DatabaseExperienceReplayBuffer(config)
    
    print("Note: Synchronous API is provided for backward compatibility")
    print("For database operations, async API is recommended for better performance")
    
    # Create a simple experience
    state = np.random.randn(5).astype(np.float32)
    experience = Experience(
        state=state,
        action=1,
        reward=0.5,
        next_state=state + 0.1,
        done=False
    )
    
    # Add experience using synchronous API
    # Note: This will work in non-async contexts
    try:
        buffer.add(experience)
        print("Added experience using synchronous API")
        
        # Get statistics
        stats = buffer.get_statistics()
        print(f"Buffer utilization: {stats.get('utilization', 0):.4f}")
        
    except Exception as e:
        print(f"Synchronous API note: {e}")
        print("In async contexts, use the async API methods instead")


if __name__ == "__main__":
    # Run async example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Example failed: {e}")
        print("This is expected if database is not set up")
    
    # Run synchronous example
    synchronous_example()