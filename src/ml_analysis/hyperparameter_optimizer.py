"""
Hyperparameter optimization for ML models using Bayesian optimization.

Provides automated hyperparameter search with efficient exploration
and exploitation of the parameter space.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from datetime import datetime
import itertools
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Defines the hyperparameter search space"""
    
    # LSTM parameters
    lstm_params: Dict[str, Tuple[Any, Any]] = field(default_factory=lambda: {
        'hidden_size': (64, 512),
        'num_layers': (1, 4),
        'dropout': (0.0, 0.5),
        'learning_rate': (1e-5, 1e-2),
        'batch_size': (16, 128),
    })
    
    # Transformer parameters
    transformer_params: Dict[str, Tuple[Any, Any]] = field(default_factory=lambda: {
        'd_model': (64, 256),
        'n_heads': (2, 8),
        'n_layers': (1, 4),
        'd_ff': (128, 1024),
        'dropout': (0.0, 0.3),
        'learning_rate': (1e-5, 1e-3),
        'batch_size': (8, 64),
    })
    
    # iTransformer parameters
    itransformer_params: Dict[str, Tuple[Any, Any]] = field(default_factory=lambda: {
        'n_variates': (5, 50),
        'd_model': (64, 256),
        'n_heads': (2, 8),
        'n_layers': (1, 4),
        'dropout': (0.0, 0.3),
        'learning_rate': (1e-5, 1e-3),
    })
    
    # PatchTST parameters
    patchtst_params: Dict[str, Tuple[Any, Any]] = field(default_factory=lambda: {
        'patch_length': (4, 32),
        'stride': (2, 16),
        'd_model': (32, 128),
        'n_heads': (2, 8),
        'n_layers': (1, 3),
        'dropout': (0.0, 0.3),
        'learning_rate': (1e-5, 1e-3),
    })
    
    # TimesMixer parameters
    timesmixer_params: Dict[str, Tuple[Any, Any]] = field(default_factory=lambda: {
        'd_model': (32, 128),
        'top_k': (2, 8),
        'num_kernels': (2, 6),
        'n_layers': (1, 3),
        'dropout': (0.0, 0.3),
        'learning_rate': (1e-5, 1e-3),
    })


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    
    Uses Gaussian Process to model the objective function and
    acquisition functions to guide the search.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[Any, Any]],
                 objective_func: Callable,
                 n_initial: int = 5,
                 acquisition: str = 'ei',
                 xi: float = 0.01,
                 kappa: float = 2.576):
        """
        Initialize Bayesian optimizer.
        
        Args:
            param_space: Dictionary of parameter names to (min, max) bounds
            objective_func: Function to optimize (returns score to maximize)
            n_initial: Number of random initial points
            acquisition: Acquisition function ('ei', 'ucb', 'poi')
            xi: Exploration parameter for EI/POI
            kappa: Exploration parameter for UCB
        """
        self.param_space = param_space
        self.objective_func = objective_func
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa
        
        # Parameter bounds and names
        self.param_names = list(param_space.keys())
        self.bounds = np.array(list(param_space.values()))
        self.n_params = len(self.param_names)
        
        # Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # Optimization history
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        
        logger.info(f"BayesianOptimizer initialized with {self.n_params} parameters")
    
    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        return (params - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def _denormalize_params(self, params: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0, 1] to original range."""
        return params * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
    
    def _params_to_dict(self, params: np.ndarray) -> Dict[str, Any]:
        """Convert parameter array to dictionary."""
        result = {}
        for i, name in enumerate(self.param_names):
            value = params[i]
            # Convert to int if parameter should be integer
            if name in ['num_layers', 'n_heads', 'n_layers', 'hidden_size', 
                       'd_model', 'd_ff', 'batch_size', 'n_variates', 
                       'patch_length', 'stride', 'top_k', 'num_kernels']:
                value = int(value)
            result[name] = value
        return result
    
    def _acquisition_ei(self, X: np.ndarray, gp: GaussianProcessRegressor, 
                        y_best: float) -> np.ndarray:
        """Expected Improvement acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.flatten()
        
        with np.errstate(divide='warn'):
            imp = mu - y_best - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _acquisition_ucb(self, X: np.ndarray, gp: GaussianProcessRegressor) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        return mu.flatten() + self.kappa * sigma
    
    def _acquisition_poi(self, X: np.ndarray, gp: GaussianProcessRegressor, 
                        y_best: float) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.flatten()
        
        with np.errstate(divide='warn'):
            Z = (mu - y_best - self.xi) / sigma
            poi = norm.cdf(Z)
            poi[sigma == 0.0] = 0.0
        
        return poi
    
    def _get_next_sample(self) -> np.ndarray:
        """Get next sample point using acquisition function."""
        # Random sampling for initial points
        if len(self.X_observed) < self.n_initial:
            return np.random.uniform(0, 1, self.n_params)
        
        # Fit GP to observed data
        X_norm = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X_norm, y)
        
        # Acquisition function
        if self.acquisition == 'ei':
            acq_func = lambda x: -self._acquisition_ei(
                x.reshape(1, -1), self.gp, self.best_score
            )
        elif self.acquisition == 'ucb':
            acq_func = lambda x: -self._acquisition_ucb(x.reshape(1, -1), self.gp)
        elif self.acquisition == 'poi':
            acq_func = lambda x: -self._acquisition_poi(
                x.reshape(1, -1), self.gp, self.best_score
            )
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")
        
        # Optimize acquisition function
        best_acq = np.inf
        best_x = None
        
        # Multiple random starts for optimization
        for _ in range(10):
            x0 = np.random.uniform(0, 1, self.n_params)
            
            res = minimize(
                acq_func,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * self.n_params
            )
            
            if res.fun < best_acq:
                best_acq = res.fun
                best_x = res.x
        
        return best_x
    
    async def optimize(self, n_iterations: int = 20) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Args:
            n_iterations: Number of optimization iterations
        
        Returns:
            Dictionary with best parameters and optimization history
        """
        logger.info(f"Starting Bayesian optimization for {n_iterations} iterations")
        
        for i in range(n_iterations):
            # Get next sample point
            x_next_norm = self._get_next_sample()
            x_next = self._denormalize_params(x_next_norm)
            
            # Convert to parameter dictionary
            params = self._params_to_dict(x_next)
            
            # Evaluate objective function
            try:
                if asyncio.iscoroutinefunction(self.objective_func):
                    score = await self.objective_func(params)
                else:
                    score = self.objective_func(params)
                
                logger.info(f"Iteration {i+1}/{n_iterations}: score={score:.4f}")
                
            except Exception as e:
                logger.error(f"Objective function failed: {e}")
                score = -np.inf
            
            # Update observations
            self.X_observed.append(x_next_norm)
            self.y_observed.append(score)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                logger.info(f"New best score: {score:.4f}")
                logger.info(f"Best params: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iterations': len(self.X_observed),
            'history': {
                'scores': self.y_observed,
                'params': [self._params_to_dict(self._denormalize_params(x))
                          for x in self.X_observed]
            }
        }


class GridSearchOptimizer:
    """
    Grid search optimization for hyperparameter tuning.

    Provides exhaustive or random sampling of parameter combinations
    with intelligent fallback to random sampling for large search spaces.
    """

    def __init__(self,
                 param_space: Dict[str, Tuple[Any, Any]],
                 objective_func: Callable,
                 grid_points: int = 3,
                 max_combinations: int = 1000,
                 random_sampling: bool = False):
        """
        Initialize Grid Search optimizer.

        Args:
            param_space: Dictionary of parameter names to (min, max) bounds
            objective_func: Function to optimize (returns score to maximize)
            grid_points: Number of grid points per parameter (3-5 recommended)
            max_combinations: Max combinations before switching to random sampling
            random_sampling: Force random sampling instead of exhaustive
        """
        self.param_space = param_space
        self.objective_func = objective_func
        self.grid_points = grid_points
        self.max_combinations = max_combinations
        self.random_sampling = random_sampling

        # Parameter names and bounds
        self.param_names = list(param_space.keys())
        self.bounds = {name: bounds for name, bounds in param_space.items()}
        self.n_params = len(self.param_names)

        # Generate parameter grids
        self.param_grids = self._generate_parameter_grids()
        self.total_combinations = self._calculate_total_combinations()

        # Determine sampling strategy
        self.use_random_sampling = (
            self.random_sampling or
            self.total_combinations > self.max_combinations
        )

        # Optimization history
        self.evaluated_params = []
        self.scores = []
        self.best_params = None
        self.best_score = -np.inf

        logger.info(f"GridSearchOptimizer initialized with {self.n_params} parameters")
        logger.info(f"Total combinations: {self.total_combinations}")
        logger.info(f"Using {'random' if self.use_random_sampling else 'exhaustive'} sampling")

    def _generate_parameter_grids(self) -> Dict[str, List[Any]]:
        """Generate discrete grid points for each parameter."""
        grids = {}

        for param_name, (min_val, max_val) in self.bounds.items():
            # Integer parameters
            if param_name in ['num_layers', 'n_heads', 'n_layers', 'hidden_size',
                             'd_model', 'd_ff', 'batch_size', 'n_variates',
                             'patch_length', 'stride', 'top_k', 'num_kernels']:
                if max_val - min_val + 1 <= self.grid_points:
                    # If range is small, use all values
                    grids[param_name] = list(range(int(min_val), int(max_val) + 1))
                else:
                    # Use evenly spaced integer values
                    grids[param_name] = [
                        int(val) for val in np.linspace(min_val, max_val, self.grid_points)
                    ]
            else:
                # Float parameters
                grids[param_name] = list(np.linspace(min_val, max_val, self.grid_points))

        return grids

    def _calculate_total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        total = 1
        for grid in self.param_grids.values():
            total *= len(grid)
        return total

    def _generate_parameter_combinations(self, n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for evaluation.

        Args:
            n_samples: Number of samples for random sampling (None for exhaustive)

        Returns:
            List of parameter dictionaries
        """
        if self.use_random_sampling and n_samples is not None:
            # Random sampling
            combinations = []
            for _ in range(n_samples):
                params = {}
                for param_name, grid in self.param_grids.items():
                    params[param_name] = random.choice(grid)
                combinations.append(params)
            return combinations
        else:
            # Exhaustive search
            param_names = list(self.param_grids.keys())
            param_values = [self.param_grids[name] for name in param_names]

            combinations = []
            for combo in itertools.product(*param_values):
                params = dict(zip(param_names, combo))
                combinations.append(params)

            return combinations

    async def optimize(self, n_iterations: int = 20) -> Dict[str, Any]:
        """
        Run grid search optimization.

        Args:
            n_iterations: Number of combinations to evaluate (for random sampling)

        Returns:
            Dictionary with best parameters and optimization history
        """
        if self.use_random_sampling:
            logger.info(f"Starting random grid search with {n_iterations} samples")
            combinations = self._generate_parameter_combinations(n_iterations)
        else:
            logger.info(f"Starting exhaustive grid search with {self.total_combinations} combinations")
            combinations = self._generate_parameter_combinations()
            # Limit to n_iterations if specified and less than total
            if n_iterations < len(combinations):
                combinations = random.sample(combinations, n_iterations)
                logger.info(f"Limited to {n_iterations} random combinations")

        total_evaluations = len(combinations)

        for i, params in enumerate(combinations):
            try:
                # Evaluate objective function
                if asyncio.iscoroutinefunction(self.objective_func):
                    score = await self.objective_func(params)
                else:
                    score = self.objective_func(params)

                logger.info(f"Evaluation {i+1}/{total_evaluations}: score={score:.4f}")

            except Exception as e:
                logger.error(f"Objective function failed: {e}")
                score = -np.inf

            # Update history
            self.evaluated_params.append(params)
            self.scores.append(score)

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"New best score: {score:.4f}")
                logger.info(f"Best params: {self.best_params}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iterations': len(self.evaluated_params),
            'total_combinations': self.total_combinations,
            'sampling_strategy': 'random' if self.use_random_sampling else 'exhaustive',
            'history': {
                'scores': self.scores,
                'params': self.evaluated_params
            }
        }


class HyperparameterOptimizer:
    """
    Main hyperparameter optimization interface for all models.
    Supports both Bayesian and Grid Search optimization methods.
    """

    def __init__(self,
                 save_dir: Optional[Path] = None,
                 n_trials: int = 20,
                 n_initial: int = 5,
                 method: str = 'bayesian',
                 grid_points: int = 3,
                 max_combinations: int = 1000):
        """
        Initialize hyperparameter optimizer.

        Args:
            save_dir: Directory to save optimization results
            n_trials: Number of optimization trials per model
            n_initial: Number of random initial points (Bayesian only)
            method: Optimization method ('bayesian' or 'grid')
            grid_points: Number of grid points per parameter (Grid only)
            max_combinations: Max combinations before random sampling (Grid only)
        """
        self.save_dir = Path(save_dir) if save_dir else Path("hyperparameters")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials
        self.n_initial = n_initial
        self.method = method.lower()
        self.grid_points = grid_points
        self.max_combinations = max_combinations

        if self.method not in ['bayesian', 'grid']:
            raise ValueError(f"Unknown optimization method: {method}. Use 'bayesian' or 'grid'")

        self.param_space = ParameterSpace()
        self.optimization_history = {}

        logger.info(f"HyperparameterOptimizer initialized with {n_trials} trials using {self.method} method")
    
    async def optimize_lstm(self, train_func: Callable) -> Dict[str, Any]:
        """
        Optimize LSTM hyperparameters.

        Args:
            train_func: Function that trains LSTM and returns validation score

        Returns:
            Best hyperparameters and optimization history
        """
        logger.info(f"Optimizing LSTM hyperparameters using {self.method} method")

        if self.method == 'bayesian':
            optimizer = BayesianOptimizer(
                param_space=self.param_space.lstm_params,
                objective_func=train_func,
                n_initial=self.n_initial
            )
        else:  # grid search
            optimizer = GridSearchOptimizer(
                param_space=self.param_space.lstm_params,
                objective_func=train_func,
                grid_points=self.grid_points,
                max_combinations=self.max_combinations
            )

        result = await optimizer.optimize(self.n_trials)

        # Save results
        self._save_results('lstm', result)
        self.optimization_history['lstm'] = result

        return result
    
    async def optimize_transformer(self, model_type: str,
                                  train_func: Callable) -> Dict[str, Any]:
        """
        Optimize transformer model hyperparameters.

        Args:
            model_type: Type of transformer ('transformer', 'itransformer', etc.)
            train_func: Function that trains model and returns validation score

        Returns:
            Best hyperparameters and optimization history
        """
        logger.info(f"Optimizing {model_type} hyperparameters using {self.method} method")

        # Get appropriate parameter space
        if model_type == 'transformer':
            param_space = self.param_space.transformer_params
        elif model_type == 'itransformer':
            param_space = self.param_space.itransformer_params
        elif model_type == 'patchtst':
            param_space = self.param_space.patchtst_params
        elif model_type == 'timesmixer':
            param_space = self.param_space.timesmixer_params
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if self.method == 'bayesian':
            optimizer = BayesianOptimizer(
                param_space=param_space,
                objective_func=train_func,
                n_initial=self.n_initial
            )
        else:  # grid search
            optimizer = GridSearchOptimizer(
                param_space=param_space,
                objective_func=train_func,
                grid_points=self.grid_points,
                max_combinations=self.max_combinations
            )

        result = await optimizer.optimize(self.n_trials)

        # Save results
        self._save_results(model_type, result)
        self.optimization_history[model_type] = result

        return result

    async def optimize_lstm_with_grid(self, train_func: Callable,
                                     grid_points: Optional[int] = None,
                                     max_combinations: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize LSTM hyperparameters using grid search specifically.

        Args:
            train_func: Function that trains LSTM and returns validation score
            grid_points: Override default grid points for this optimization
            max_combinations: Override default max combinations threshold

        Returns:
            Best hyperparameters and optimization history
        """
        logger.info("Optimizing LSTM hyperparameters using grid search")

        grid_points = grid_points or self.grid_points
        max_combinations = max_combinations or self.max_combinations

        optimizer = GridSearchOptimizer(
            param_space=self.param_space.lstm_params,
            objective_func=train_func,
            grid_points=grid_points,
            max_combinations=max_combinations
        )

        result = await optimizer.optimize(self.n_trials)

        # Save results with grid suffix
        self._save_results('lstm_grid', result)
        self.optimization_history['lstm_grid'] = result

        return result

    async def optimize_transformer_with_grid(self, model_type: str, train_func: Callable,
                                           grid_points: Optional[int] = None,
                                           max_combinations: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize transformer model hyperparameters using grid search specifically.

        Args:
            model_type: Type of transformer ('transformer', 'itransformer', etc.)
            train_func: Function that trains model and returns validation score
            grid_points: Override default grid points for this optimization
            max_combinations: Override default max combinations threshold

        Returns:
            Best hyperparameters and optimization history
        """
        logger.info(f"Optimizing {model_type} hyperparameters using grid search")

        grid_points = grid_points or self.grid_points
        max_combinations = max_combinations or self.max_combinations

        # Get appropriate parameter space
        if model_type == 'transformer':
            param_space = self.param_space.transformer_params
        elif model_type == 'itransformer':
            param_space = self.param_space.itransformer_params
        elif model_type == 'patchtst':
            param_space = self.param_space.patchtst_params
        elif model_type == 'timesmixer':
            param_space = self.param_space.timesmixer_params
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        optimizer = GridSearchOptimizer(
            param_space=param_space,
            objective_func=train_func,
            grid_points=grid_points,
            max_combinations=max_combinations
        )

        result = await optimizer.optimize(self.n_trials)

        # Save results with grid suffix
        model_name_grid = f"{model_type}_grid"
        self._save_results(model_name_grid, result)
        self.optimization_history[model_name_grid] = result

        return result

    async def optimize_all_with_grid(self, lstm_train_func: Optional[Callable] = None,
                                   transformer_train_func: Optional[Callable] = None,
                                   itransformer_train_func: Optional[Callable] = None,
                                   patchtst_train_func: Optional[Callable] = None,
                                   timesmixer_train_func: Optional[Callable] = None,
                                   grid_points: Optional[int] = None,
                                   max_combinations: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Optimize all models using grid search.

        Args:
            lstm_train_func: Training function for LSTM
            transformer_train_func: Training function for Transformer
            itransformer_train_func: Training function for iTransformer
            patchtst_train_func: Training function for PatchTST
            timesmixer_train_func: Training function for TimesMixer
            grid_points: Override default grid points
            max_combinations: Override default max combinations threshold

        Returns:
            Dictionary of optimization results for each model
        """
        results = {}

        if lstm_train_func:
            logger.info("Running grid search optimization for LSTM")
            results['lstm'] = await self.optimize_lstm_with_grid(
                lstm_train_func, grid_points, max_combinations
            )

        model_funcs = {
            'transformer': transformer_train_func,
            'itransformer': itransformer_train_func,
            'patchtst': patchtst_train_func,
            'timesmixer': timesmixer_train_func
        }

        for model_type, train_func in model_funcs.items():
            if train_func:
                logger.info(f"Running grid search optimization for {model_type}")
                results[model_type] = await self.optimize_transformer_with_grid(
                    model_type, train_func, grid_points, max_combinations
                )

        return results

    def _save_results(self, model_name: str, result: Dict[str, Any]) -> None:
        """Save optimization results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.save_dir / f"{model_name}_hyperparams_{timestamp}.json"

        # Add metadata about optimization method
        result_with_meta = result.copy()
        result_with_meta['optimization_method'] = getattr(result, 'sampling_strategy', self.method)
        result_with_meta['optimizer_settings'] = {
            'n_trials': self.n_trials,
            'method': self.method
        }

        # Add grid-specific metadata if available
        if 'sampling_strategy' in result:
            result_with_meta['optimizer_settings'].update({
                'grid_points': self.grid_points,
                'max_combinations': self.max_combinations,
                'total_combinations': result.get('total_combinations'),
                'sampling_strategy': result.get('sampling_strategy')
            })
        else:
            # Bayesian-specific metadata
            result_with_meta['optimizer_settings']['n_initial'] = self.n_initial

        with open(filename, 'w') as f:
            json.dump(result_with_meta, f, indent=2, default=str)

        # Also save best params separately
        best_params_file = self.save_dir / f"{model_name}_best_params.json"
        with open(best_params_file, 'w') as f:
            json.dump(result['best_params'], f, indent=2, default=str)

        logger.info(f"Saved optimization results to {filename}")
        logger.info(f"Method: {result_with_meta.get('optimization_method', 'unknown')}, "
                   f"Best score: {result['best_score']:.4f}")
    
    def load_best_params(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load best parameters for a model."""
        best_params_file = self.save_dir / f"{model_name}_best_params.json"
        
        if best_params_file.exists():
            with open(best_params_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of all optimization runs."""
        summary_data = []
        
        for model_name, history in self.optimization_history.items():
            summary_data.append({
                'model': model_name,
                'best_score': history['best_score'],
                'n_iterations': history['n_iterations'],
                'best_params': history['best_params']
            })
        
        return pd.DataFrame(summary_data)