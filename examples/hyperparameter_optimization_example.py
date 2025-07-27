#!/usr/bin/env python3
"""
Example: Hyperparameter Optimization for DQN Trading Agent

This example demonstrates how to use the HyperparameterSearch class
to systematically optimize DQN agent parameters for trading.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

from src.rl_agent.training_pipeline import HyperparameterSearch, TrainingConfig
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


def create_sample_data():
    """Create sample tokens and historical data for demonstration"""
    
    # Create sample tokens
    tokens = []
    for i in range(3):
        token = DiscoveredToken(
            address=f"0x{'0' * 38}{i:02d}",
            symbol=f"TOK{i}",
            name=f"Sample Token {i}",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="example",
            price_usd=1.0 + i * 0.5,
            volume_24h=1000000 + i * 500000
        )
        tokens.append(token)
    
    # Create historical price data
    historical_data = {}
    np.random.seed(42)  # For reproducible results
    
    for token in tokens:
        prices = []
        base_price = token.price_usd
        
        # Generate 100 price points with random walk
        for j in range(100):
            if j == 0:
                price = base_price
            else:
                # Random walk with slight upward drift
                change = np.random.normal(0.002, 0.03)
                price = prices[-1] * (1 + change)
                price = max(price, 0.01)  # Minimum price
            
            prices.append(price)
        
        # Create timestamps
        timestamps = [
            datetime.now() - timedelta(hours=100-i) 
            for i in range(100)
        ]
        
        historical_data[token.address] = {
            'prices': prices,
            'timestamps': timestamps
        }
    
    return tokens, historical_data


def basic_optimization_example():
    """Basic hyperparameter optimization example"""
    
    print("=== Basic Hyperparameter Optimization Example ===")
    
    # Create sample data
    tokens, historical_data = create_sample_data()
    
    # Define search space
    search_space = {
        'learning_rate': [0.001, 0.005, 0.01],
        'epsilon_decay': [0.995, 0.99, 0.985],
        'batch_size': [32, 64]
    }
    
    print(f"Search space: {search_space}")
    print(f"Total combinations: {len(search_space['learning_rate']) * len(search_space['epsilon_decay']) * len(search_space['batch_size'])}")
    
    # Create hyperparameter search
    search = HyperparameterSearch(search_space)
    
    # Run optimization
    print("\nStarting optimization...")
    best_params = search.optimize(
        tokens=tokens,
        historical_data=historical_data,
        num_trials=6,  # Limit for demonstration
        episodes_per_trial=20,  # Short episodes for demo
        runs_per_config=1  # Single run per config for speed
    )
    
    print(f"\nOptimization completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {search.best_score:.4f}")
    print(f"Configurations tested: {len(search.optimization_history)}")


def cross_validation_example():
    """Example using cross-validation for robust evaluation"""
    
    print("\n=== Cross-Validation Example ===")
    
    # Create sample data
    tokens, historical_data = create_sample_data()
    
    # Smaller search space for CV example
    search_space = {
        'learning_rate': [0.001, 0.01],
        'epsilon_decay': [0.995, 0.99]
    }
    
    print(f"Search space: {search_space}")
    
    # Create hyperparameter search
    search = HyperparameterSearch(search_space)
    
    # Run optimization with cross-validation
    print("\nStarting cross-validation optimization...")
    best_params = search.optimize(
        tokens=tokens,
        historical_data=historical_data,
        num_trials=4,
        episodes_per_trial=15,
        cross_validation_folds=3  # Use 3-fold CV
    )
    
    print(f"\nCross-validation optimization completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {search.best_score:.4f}")


def early_stopping_example():
    """Example with early stopping mechanisms"""
    
    print("\n=== Early Stopping Example ===")
    
    # Create sample data
    tokens, historical_data = create_sample_data()
    
    # Larger search space to demonstrate early stopping
    search_space = {
        'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.05],
        'epsilon_decay': [0.999, 0.995, 0.99, 0.985]
    }
    
    print(f"Search space: {search_space}")
    print(f"Total possible combinations: {len(search_space['learning_rate']) * len(search_space['epsilon_decay'])}")
    
    # Create hyperparameter search
    search = HyperparameterSearch(search_space)
    
    # Run optimization with early stopping
    print("\nStarting optimization with early stopping...")
    best_params = search.optimize(
        tokens=tokens,
        historical_data=historical_data,
        num_trials=20,  # Max combinations
        episodes_per_trial=10,
        early_stopping_patience=3,  # Stop after 3 configs without improvement
        early_stopping_threshold=0.7  # Stop if score > 0.7
    )
    
    print(f"\nEarly stopping optimization completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {search.best_score:.4f}")
    print(f"Configurations tested: {len(search.optimization_history)} out of {len(search_space['learning_rate']) * len(search_space['epsilon_decay'])} possible")


def multiple_runs_example():
    """Example with multiple runs per configuration for statistical robustness"""
    
    print("\n=== Multiple Runs Example ===")
    
    # Create sample data
    tokens, historical_data = create_sample_data()
    
    # Simple search space
    search_space = {
        'learning_rate': [0.001, 0.01],
        'epsilon_decay': [0.995, 0.99]
    }
    
    print(f"Search space: {search_space}")
    
    # Create hyperparameter search
    search = HyperparameterSearch(search_space)
    
    # Run optimization with multiple runs per configuration
    print("\nStarting optimization with multiple runs per configuration...")
    best_params = search.optimize(
        tokens=tokens,
        historical_data=historical_data,
        num_trials=4,
        episodes_per_trial=10,
        runs_per_config=3  # 3 runs per configuration for robustness
    )
    
    print(f"\nMultiple runs optimization completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {search.best_score:.4f}")
    print(f"Total training runs: {len(search.optimization_history) * 3}")


def analyze_optimization_results(search: HyperparameterSearch):
    """Analyze and display optimization results"""
    
    print("\n=== Optimization Analysis ===")
    
    if not search.optimization_history:
        print("No optimization history available.")
        return
    
    # Sort by score
    sorted_results = sorted(search.optimization_history, key=lambda x: x['score'], reverse=True)
    
    print(f"Top 3 configurations:")
    for i, result in enumerate(sorted_results[:3]):
        print(f"  {i+1}. Score: {result['score']:.4f}, Params: {result['params']}")
    
    print(f"\nWorst 3 configurations:")
    for i, result in enumerate(sorted_results[-3:]):
        rank = len(sorted_results) - 2 + i
        print(f"  {rank}. Score: {result['score']:.4f}, Params: {result['params']}")
    
    # Score distribution
    scores = [r['score'] for r in search.optimization_history]
    print(f"\nScore statistics:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")


def main():
    """Run all hyperparameter optimization examples"""
    
    print("Hyperparameter Optimization Examples for DQN Trading Agent")
    print("=" * 60)
    
    # Run examples
    basic_optimization_example()
    cross_validation_example()
    early_stopping_example()
    multiple_runs_example()
    
    # Create one more search for analysis
    tokens, historical_data = create_sample_data()
    search_space = {
        'learning_rate': [0.001, 0.005, 0.01],
        'epsilon_decay': [0.995, 0.99]
    }
    
    search = HyperparameterSearch(search_space)
    search.optimize(
        tokens=tokens,
        historical_data=historical_data,
        num_trials=6,
        episodes_per_trial=10
    )
    
    analyze_optimization_results(search)
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("\nKey takeaways:")
    print("1. Grid search systematically tests all parameter combinations")
    print("2. Cross-validation provides more robust evaluation")
    print("3. Early stopping saves computation time")
    print("4. Multiple runs improve statistical reliability")
    print("5. Composite scoring considers multiple performance metrics")


if __name__ == "__main__":
    main()