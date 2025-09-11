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


class HyperparameterOptimizer:
    """
    Main hyperparameter optimization interface for all models.
    """
    
    def __init__(self, 
                 save_dir: Optional[Path] = None,
                 n_trials: int = 20,
                 n_initial: int = 5):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            save_dir: Directory to save optimization results
            n_trials: Number of optimization trials per model
            n_initial: Number of random initial points
        """
        self.save_dir = Path(save_dir) if save_dir else Path("hyperparameters")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials
        self.n_initial = n_initial
        
        self.param_space = ParameterSpace()
        self.optimization_history = {}
        
        logger.info(f"HyperparameterOptimizer initialized with {n_trials} trials")
    
    async def optimize_lstm(self, train_func: Callable) -> Dict[str, Any]:
        """
        Optimize LSTM hyperparameters.
        
        Args:
            train_func: Function that trains LSTM and returns validation score
        
        Returns:
            Best hyperparameters and optimization history
        """
        logger.info("Optimizing LSTM hyperparameters")
        
        optimizer = BayesianOptimizer(
            param_space=self.param_space.lstm_params,
            objective_func=train_func,
            n_initial=self.n_initial
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
        logger.info(f"Optimizing {model_type} hyperparameters")
        
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
        
        optimizer = BayesianOptimizer(
            param_space=param_space,
            objective_func=train_func,
            n_initial=self.n_initial
        )
        
        result = await optimizer.optimize(self.n_trials)
        
        # Save results
        self._save_results(model_type, result)
        self.optimization_history[model_type] = result
        
        return result
    
    def _save_results(self, model_name: str, result: Dict[str, Any]) -> None:
        """Save optimization results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.save_dir / f"{model_name}_hyperparams_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Also save best params separately
        best_params_file = self.save_dir / f"{model_name}_best_params.json"
        with open(best_params_file, 'w') as f:
            json.dump(result['best_params'], f, indent=2, default=str)
        
        logger.info(f"Saved optimization results to {filename}")
    
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