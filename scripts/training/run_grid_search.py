#!/usr/bin/env python3
"""
Grid search hyperparameter optimization example script.

Demonstrates how to use the new grid search functionality
for model hyperparameter optimization.
"""

import asyncio
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml_analysis.hyperparameter_optimizer import HyperparameterOptimizer
from scripts.training.hyperparameter_search import AutomatedHyperparameterSearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_grid_search_comparison():
    """
    Run both Bayesian and Grid search optimization for comparison.

    Shows how to use the new grid search functionality alongside
    the existing Bayesian optimization.
    """
    logger.info("Starting hyperparameter optimization comparison")

    # Initialize automated search with grid method
    grid_search = AutomatedHyperparameterSearch(
        timeframe="daily",
        token="WBTC",
        n_trials=50,  # Will test 50 parameter combinations
        save_dir="./hyperparameters/grid_search"
    )

    # Override optimizer to use grid search
    grid_search.optimizer = HyperparameterOptimizer(
        save_dir=Path("./hyperparameters/grid_search"),
        n_trials=50,
        method='grid',  # Use grid search
        grid_points=4,  # 4 values per parameter
        max_combinations=500  # Random sample if >500 combinations
    )

    logger.info("Running Grid Search optimization for LSTM")
    lstm_results_grid = await grid_search.search_lstm()

    # Now run Bayesian optimization for comparison
    bayesian_search = AutomatedHyperparameterSearch(
        timeframe="daily",
        token="WBTC",
        n_trials=20,  # Fewer trials needed for Bayesian
        save_dir="./hyperparameters/bayesian"
    )

    logger.info("Running Bayesian optimization for LSTM")
    lstm_results_bayesian = await bayesian_search.search_lstm()

    # Compare results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS COMPARISON")
    print("="*60)

    print("\nGrid Search Results:")
    print(f"  Best Score: {lstm_results_grid['best_score']:.4f}")
    print(f"  Evaluations: {lstm_results_grid['n_iterations']}")
    print(f"  Strategy: {lstm_results_grid.get('sampling_strategy', 'N/A')}")
    print(f"  Total Combinations: {lstm_results_grid.get('total_combinations', 'N/A')}")
    print(f"  Best Parameters:")
    for param, value in lstm_results_grid['best_params'].items():
        print(f"    {param}: {value}")

    print("\nBayesian Results:")
    print(f"  Best Score: {lstm_results_bayesian['best_score']:.4f}")
    print(f"  Evaluations: {lstm_results_bayesian['n_iterations']}")
    print(f"  Best Parameters:")
    for param, value in lstm_results_bayesian['best_params'].items():
        print(f"    {param}: {value}")

    # Save comparison
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'grid_search': {
            'best_score': lstm_results_grid['best_score'],
            'n_iterations': lstm_results_grid['n_iterations'],
            'best_params': lstm_results_grid['best_params']
        },
        'bayesian': {
            'best_score': lstm_results_bayesian['best_score'],
            'n_iterations': lstm_results_bayesian['n_iterations'],
            'best_params': lstm_results_bayesian['best_params']
        }
    }

    comparison_file = Path("./hyperparameters/method_comparison.json")
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info(f"Comparison saved to {comparison_file}")

    return comparison


async def run_quick_grid_search():
    """
    Run a quick grid search with limited combinations.

    Good for rapid prototyping and testing parameter ranges.
    """
    logger.info("Running quick grid search")

    optimizer = HyperparameterOptimizer(
        save_dir=Path("./hyperparameters/quick_grid"),
        n_trials=10,  # Only test 10 combinations
        method='grid',
        grid_points=3,  # 3 values per parameter
        max_combinations=10  # Force random sampling
    )

    # Create simple training function for demonstration
    async def dummy_train_func(params):
        """Dummy function that simulates training."""
        # In real usage, this would train the model and return validation score
        import random
        score = random.random() * 0.5 + 0.3  # Random score between 0.3-0.8
        logger.info(f"Testing params: {params}")
        return score

    # Run optimization
    result = await optimizer.optimize_lstm(dummy_train_func)

    print("\nQuick Grid Search Results:")
    print(f"  Best Score: {result['best_score']:.4f}")
    print(f"  Sampling: {result.get('sampling_strategy', 'N/A')}")
    print(f"  Best Parameters: {result['best_params']}")

    return result


async def run_exhaustive_grid_search():
    """
    Run exhaustive grid search for critical model.

    Use when you need to thoroughly explore parameter space.
    """
    logger.info("Running exhaustive grid search for transformer")

    search = AutomatedHyperparameterSearch(
        timeframe="daily",
        token="WBTC",
        n_trials=100,
        save_dir="./hyperparameters/exhaustive"
    )

    # Configure for exhaustive search with custom train function
    results = await search.optimizer.optimize_transformer_with_grid(
        model_type='transformer',
        train_func=search._create_transformer_train_func(),
        grid_points=5,  # 5 values per parameter for thorough search
        max_combinations=2000  # Allow up to 2000 combinations
    )

    print("\nExhaustive Search Results:")
    print(f"  Model: Transformer")
    print(f"  Best Score: {results['best_score']:.4f}")
    print(f"  Total Combinations: {results.get('total_combinations', 'N/A')}")
    print(f"  Evaluations: {results['n_iterations']}")

    return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Grid search hyperparameter optimization'
    )
    parser.add_argument(
        '--mode',
        choices=['comparison', 'quick', 'exhaustive', 'all'],
        default='quick',
        help='Optimization mode to run'
    )

    args = parser.parse_args()

    if args.mode == 'comparison':
        await run_grid_search_comparison()
    elif args.mode == 'quick':
        await run_quick_grid_search()
    elif args.mode == 'exhaustive':
        await run_exhaustive_grid_search()
    else:  # all
        logger.info("Running all optimization modes")
        await run_quick_grid_search()
        await run_grid_search_comparison()
        await run_exhaustive_grid_search()


if __name__ == "__main__":
    asyncio.run(main())