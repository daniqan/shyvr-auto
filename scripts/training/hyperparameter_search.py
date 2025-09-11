#!/usr/bin/env python3
"""
Automated hyperparameter search script for all models.

Uses Bayesian optimization to find optimal hyperparameters
for each model type.
"""

import asyncio
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml_analysis.hyperparameter_optimizer import HyperparameterOptimizer
from scripts.training.train_all_models import UnifiedTrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomatedHyperparameterSearch:
    """
    Automated hyperparameter search for all model types.
    
    Integrates with UnifiedTrainingPipeline to evaluate different
    hyperparameter configurations.
    """
    
    def __init__(self,
                 corpus_version: Optional[str] = None,
                 timeframe: str = "daily",
                 token: Optional[str] = "WBTC",
                 n_trials: int = 20,
                 save_dir: str = "./hyperparameters"):
        """
        Initialize hyperparameter search.
        
        Args:
            corpus_version: GCS corpus version to use
            timeframe: Data timeframe (daily, hourly)
            token: Token to use for optimization
            n_trials: Number of trials per model
            save_dir: Directory to save results
        """
        self.corpus_version = corpus_version
        self.timeframe = timeframe
        self.token = token
        self.n_trials = n_trials
        self.save_dir = Path(save_dir)
        
        # Initialize optimizer
        self.optimizer = HyperparameterOptimizer(
            save_dir=self.save_dir,
            n_trials=n_trials,
            n_initial=5
        )
        
        # Initialize training pipeline
        self.pipeline = UnifiedTrainingPipeline()
        
        # Load and cache data once
        self.train_data = None
        self.val_data = None
        
        logger.info(f"AutomatedHyperparameterSearch initialized with {n_trials} trials")
    
    async def _load_data(self):
        """Load and cache training/validation data."""
        if self.train_data is None:
            logger.info("Loading corpus data...")
            
            # Load corpus
            data = await self.pipeline.load_corpus_data(
                self.corpus_version, 
                self.timeframe, 
                self.token
            )
            
            # Split data
            train_data, val_data, _ = self.pipeline.prepare_train_val_test_split(data)
            
            # Apply feature engineering
            from src.ml_analysis.feature_engineer import FeatureEngineer
            feature_engineer = FeatureEngineer(
                enable_token_normalization=True,
                enable_advanced_features=True
            )
            
            # Process training data
            train_data = feature_engineer.calculate_advanced_features(train_data)
            train_data = feature_engineer.handle_missing_values(train_data)
            self.train_data = feature_engineer.normalize_features_by_token(train_data)
            
            # Process validation data
            val_data = feature_engineer.calculate_advanced_features(val_data)
            val_data = feature_engineer.handle_missing_values(val_data)
            self.val_data = feature_engineer.normalize_features_by_token(val_data)
            
            logger.info(f"Data loaded: train={len(self.train_data)}, val={len(self.val_data)}")
    
    async def _train_and_evaluate_lstm(self, params: Dict[str, Any]) -> float:
        """
        Train LSTM with given parameters and return validation score.
        
        Args:
            params: Hyperparameters to use
        
        Returns:
            Validation accuracy (higher is better)
        """
        try:
            # Ensure data is loaded
            await self._load_data()
            
            logger.info(f"Training LSTM with params: {params}")
            
            # Create LSTM with hyperparameters
            from src.ml_analysis.lstm_model import LSTMPricePredictor
            
            model = LSTMPricePredictor(
                sequence_length=50,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
            
            # Prepare data
            X, y, _ = model.prepare_training_from_corpus(self.train_data)
            X_val, y_val, _ = model.prepare_training_from_corpus(self.val_data)
            
            # Train with limited epochs for speed
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.model.to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X), 
                torch.FloatTensor(y)
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=params['batch_size'],
                shuffle=True
            )
            
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=params['batch_size'],
                shuffle=False
            )
            
            # Training
            optimizer = torch.optim.Adam(
                model.model.parameters(), 
                lr=params['learning_rate']
            )
            criterion = nn.MSELoss()
            
            # Quick training (10 epochs for hyperparameter search)
            for epoch in range(10):
                model.model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model.model(batch_X)
                    
                    # Handle output shape
                    if isinstance(outputs, dict):
                        predictions = torch.stack([
                            outputs['price_1h'].squeeze(),
                            outputs['price_4h'].squeeze(),
                            outputs['price_24h'].squeeze()
                        ], dim=-1)
                    else:
                        predictions = outputs
                    
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Validation
            model.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model.model(batch_X)
                    
                    if isinstance(outputs, dict):
                        predictions = torch.stack([
                            outputs['price_1h'].squeeze(),
                            outputs['price_4h'].squeeze(),
                            outputs['price_24h'].squeeze()
                        ], dim=-1)
                    else:
                        predictions = outputs
                    
                    loss = criterion(predictions, batch_y)
                    val_losses.append(loss.item())
            
            # Return negative loss (we want to maximize score)
            val_loss = np.mean(val_losses)
            score = 1.0 / (1.0 + val_loss)  # Convert to accuracy-like score
            
            logger.info(f"LSTM validation score: {score:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return 0.0
    
    async def _train_and_evaluate_transformer(self, 
                                             model_type: str,
                                             params: Dict[str, Any]) -> float:
        """
        Train transformer model with given parameters and return validation score.
        
        Args:
            model_type: Type of transformer model
            params: Hyperparameters to use
        
        Returns:
            Validation accuracy (higher is better)
        """
        try:
            # Ensure data is loaded
            await self._load_data()
            
            logger.info(f"Training {model_type} with params: {params}")
            
            # Import appropriate model
            if model_type == 'transformer':
                from src.ml_analysis.transformers.transformer_predictor import TransformerPredictor
                model_class = TransformerPredictor
            elif model_type == 'itransformer':
                from src.ml_analysis.transformers.itransformer import iTransformerPredictor
                model_class = iTransformerPredictor
            elif model_type == 'patchtst':
                from src.ml_analysis.transformers.patchtst import PatchTSTPredictor
                model_class = PatchTSTPredictor
            elif model_type == 'timesmixer':
                from src.ml_analysis.transformers.timesmixer import TimesMixerPredictor
                model_class = TimesMixerPredictor
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Create model with hyperparameters
            model = model_class(params)
            
            # Prepare data
            X, y, _ = model.prepare_training_from_corpus(self.train_data)
            X_val, y_val, _ = model.prepare_training_from_corpus(self.val_data)
            
            # Quick training setup
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.model.to(device)
            
            # Create data loaders
            batch_size = params.get('batch_size', 32)
            train_dataset = TensorDataset(
                torch.FloatTensor(X),
                torch.FloatTensor(y)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Training
            optimizer = torch.optim.Adam(
                model.model.parameters(),
                lr=params['learning_rate']
            )
            criterion = nn.MSELoss()
            
            # Quick training (5 epochs for transformers due to complexity)
            for epoch in range(5):
                model.model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model.model(batch_X)
                    
                    # Handle different output formats
                    if isinstance(outputs, dict):
                        if 'predictions' in outputs:
                            predictions = outputs['predictions']
                        elif 'price_1h' in outputs:
                            predictions = torch.stack([
                                outputs['price_1h'].squeeze(),
                                outputs['price_4h'].squeeze(),
                                outputs['price_24h'].squeeze()
                            ], dim=-1)
                        else:
                            predictions = list(outputs.values())[0]
                    else:
                        predictions = outputs
                    
                    # Ensure shapes match
                    if predictions.shape != batch_y.shape:
                        if predictions.dim() == 3 and batch_y.dim() == 2:
                            predictions = predictions.mean(dim=1)
                    
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Validation
            model.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model.model(batch_X)
                    
                    if isinstance(outputs, dict):
                        if 'predictions' in outputs:
                            predictions = outputs['predictions']
                        elif 'price_1h' in outputs:
                            predictions = torch.stack([
                                outputs['price_1h'].squeeze(),
                                outputs['price_4h'].squeeze(),
                                outputs['price_24h'].squeeze()
                            ], dim=-1)
                        else:
                            predictions = list(outputs.values())[0]
                    else:
                        predictions = outputs
                    
                    if predictions.shape != batch_y.shape:
                        if predictions.dim() == 3 and batch_y.dim() == 2:
                            predictions = predictions.mean(dim=1)
                    
                    loss = criterion(predictions, batch_y)
                    val_losses.append(loss.item())
            
            # Return score
            val_loss = np.mean(val_losses)
            score = 1.0 / (1.0 + val_loss)
            
            logger.info(f"{model_type} validation score: {score:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"{model_type} training failed: {e}")
            return 0.0
    
    async def search_all_models(self):
        """Run hyperparameter search for all models."""
        results = {}
        
        # LSTM optimization
        logger.info("="*50)
        logger.info("Optimizing LSTM hyperparameters")
        logger.info("="*50)
        
        lstm_result = await self.optimizer.optimize_lstm(
            self._train_and_evaluate_lstm
        )
        results['lstm'] = lstm_result
        
        # Transformer models
        for model_type in ['transformer', 'itransformer', 'patchtst', 'timesmixer']:
            logger.info("="*50)
            logger.info(f"Optimizing {model_type} hyperparameters")
            logger.info("="*50)
            
            result = await self.optimizer.optimize_transformer(
                model_type,
                lambda params: self._train_and_evaluate_transformer(model_type, params)
            )
            results[model_type] = result
        
        # Save combined results
        self._save_all_results(results)
        
        return results
    
    def _save_all_results(self, results: Dict[str, Any]):
        """Save all optimization results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save complete results
        all_results_file = self.save_dir / f"all_hyperparams_{timestamp}.json"
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.save_dir / "hyperparameter_summary.json"
        summary = {}
        
        for model_name, result in results.items():
            summary[model_name] = {
                'best_score': result['best_score'],
                'best_params': result['best_params']
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.save_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Best Score: {result['best_score']:.4f}")
            print(f"  Best Parameters:")
            for param, value in result['best_params'].items():
                print(f"    - {param}: {value}")


async def main():
    """Main entry point for hyperparameter search."""
    parser = argparse.ArgumentParser(description='Automated hyperparameter search')
    parser.add_argument('--corpus-version', type=str, default=None,
                       help='GCS corpus version to use')
    parser.add_argument('--timeframe', type=str, default='daily',
                       choices=['daily', 'hourly', 'hour'],
                       help='Data timeframe')
    parser.add_argument('--token', type=str, default='WBTC',
                       help='Token to use for optimization')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of trials per model')
    parser.add_argument('--save-dir', type=str, default='./hyperparameters',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run search
    search = AutomatedHyperparameterSearch(
        corpus_version=args.corpus_version,
        timeframe=args.timeframe,
        token=args.token,
        n_trials=args.n_trials,
        save_dir=args.save_dir
    )
    
    await search.search_all_models()


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main())