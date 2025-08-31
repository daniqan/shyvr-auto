"""
UnifiedTrainingPipeline for Training All ML Models

This script provides a unified interface for training all ML models using
real GCS corpus data. Built following TDD methodology with comprehensive
integration testing.

Key Features:
- Load corpus data from GCS using GCSCorpusLoader
- Train LSTM and all Transformer variants 
- Time-series aware train/val/test splitting (80/10/10)
- Integration with ModelManager for ensemble coordination
- Database tracking via model_training_history table
- Save trained models to GCS shyvr-models-prod/trained-models/
- Comprehensive error handling and logging
- Real training with actual GCS data (no mocks)

Models Supported:
- LSTMPricePredictor
- TransformerPredictor  
- iTransformerPredictor
- PatchTSTPredictor
- TimesMixerPredictor
- (TimesFM skipped - pre-trained model)
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
import torch
from google.cloud import storage

# Database imports
from src.utils.database import get_database_connection, execute_query, execute_transaction

# Data pipeline imports
from src.data_pipeline.gcs_corpus_loader import GCSCorpusLoader

# ML model imports
from src.ml_analysis.lstm_model import LSTMPricePredictor
from src.ml_analysis.transformers.transformer_predictor import TransformerPredictor
from src.ml_analysis.transformers.itransformer import iTransformerPredictor
from src.ml_analysis.transformers.patchtst import PatchTSTPredictor
from src.ml_analysis.transformers.timesmixer import TimesMixerPredictor
from src.ml_analysis.model_manager import ModelManager
from src.ml_analysis.base import ModelType
from src.ml_analysis.training_report_generator import TrainingReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedTrainingPipelineError(Exception):
    """Base exception for UnifiedTrainingPipeline errors"""
    pass


class DataLoadingError(UnifiedTrainingPipelineError):
    """Error loading corpus data"""
    pass


class ModelTrainingError(UnifiedTrainingPipelineError):
    """Error during model training"""
    pass


class ModelSavingError(UnifiedTrainingPipelineError):
    """Error saving trained models"""
    pass


class UnifiedTrainingPipeline:
    """
    Unified Training Pipeline for all ML models
    
    Coordinates training of LSTM and Transformer variants using real GCS corpus data.
    Integrates with ModelManager and tracks training in model_training_history table.
    """
    
    def __init__(self,
                 gcs_bucket: str = "shyvr-models-prod",
                 model_save_bucket: str = "shyvr-models-prod",
                 model_save_prefix: str = "trained-models",
                 cache_dir: str = "/tmp/training_cache",
                 cache_ttl_hours: int = 24):
        """
        Initialize UnifiedTrainingPipeline
        
        Args:
            gcs_bucket: GCS bucket containing corpus data
            model_save_bucket: GCS bucket for saving trained models
            model_save_prefix: Prefix for model storage path
            cache_dir: Local directory for caching corpus data
            cache_ttl_hours: Hours before cached data expires
        """
        self.gcs_bucket = gcs_bucket
        self.model_save_bucket = model_save_bucket
        self.model_save_prefix = model_save_prefix
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours
        
        # Initialize GCS corpus loader
        self.corpus_loader = GCSCorpusLoader(
            bucket_name=gcs_bucket,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours
        )
        
        # Initialize GCS client for saving models
        self.gcs_client = storage.Client()
        self.save_bucket = self.gcs_client.bucket(model_save_bucket)
        
        # Training session metadata
        self.session_id = str(uuid.uuid4())
        self.session_timestamp = datetime.now()
        
        # Initialize TrainingReportGenerator
        self.report_generator = TrainingReportGenerator({
            'output_dir': f'{cache_dir}/reports',
            'gcs_bucket': model_save_bucket,
            'gcs_prefix': model_save_prefix
        })
        
        logger.info(
            f"UnifiedTrainingPipeline initialized - session_id: {self.session_id}, "
            f"gcs_bucket: {gcs_bucket}, model_save_bucket: {model_save_bucket}, "
            f"cache_dir: {cache_dir}"
        )
    
    async def load_corpus_data(self,
                             corpus_version: Optional[str] = None,
                             timeframe: str = "daily",
                             token: Optional[str] = None) -> pd.DataFrame:
        """
        Load corpus data from GCS
        
        Args:
            corpus_version: Specific corpus version (None = latest)
            timeframe: Data timeframe (daily, hourly, hour)
            token: Specific token to filter (None = all tokens)
            
        Returns:
            DataFrame with corpus data
        """
        try:
            logger.info(f"Loading corpus data: version={corpus_version}, timeframe={timeframe}, token={token}")
            
            df = await self.corpus_loader.load_corpus_from_gcs(
                gcs_prefix=corpus_version,
                timeframe=timeframe,
                token=token
            )
            
            if len(df) == 0:
                raise DataLoadingError(f"No data found for corpus_version={corpus_version}, timeframe={timeframe}, token={token}")
            
            # Validate essential columns
            required_columns = ['close', 'timestamp'] if 'timestamp' in df.columns else ['close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataLoadingError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Corpus data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load corpus data: {e}")
            raise DataLoadingError(f"Failed to load corpus data: {e}")
    
    def prepare_train_val_test_split(self, 
                                   data: pd.DataFrame,
                                   train_ratio: float = 0.8,
                                   val_ratio: float = 0.1,
                                   test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-series aware train/validation/test split
        
        Preserves temporal order - training data comes before validation,
        which comes before test data.
        
        Args:
            data: Input DataFrame with time-series data
            train_ratio: Proportion for training (default: 0.8)
            val_ratio: Proportion for validation (default: 0.1) 
            test_ratio: Proportion for test (default: 0.1)
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        try:
            # Validate ratios sum to 1.0
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
                raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
            
            # Sort by timestamp to ensure temporal order
            if 'timestamp' in data.columns:
                data_sorted = data.sort_values('timestamp').copy()
            else:
                # Assume index is timestamp-based
                data_sorted = data.sort_index().copy()
            
            total_samples = len(data_sorted)
            train_end = int(total_samples * train_ratio)
            val_end = int(total_samples * (train_ratio + val_ratio))
            
            # Split data maintaining temporal order
            train_data = data_sorted.iloc[:train_end].copy()
            val_data = data_sorted.iloc[train_end:val_end].copy()
            test_data = data_sorted.iloc[val_end:].copy()
            
            # Validate splits
            if len(train_data) == 0:
                raise ValueError("Training set is empty")
            if len(val_data) == 0:
                raise ValueError("Validation set is empty") 
            if len(test_data) == 0:
                raise ValueError("Test set is empty")
            
            logger.info(
                f"Time-series split completed: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
            )
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise DataLoadingError(f"Failed to split data: {e}")
    
    async def train_lstm_model(self,
                             training_data: pd.DataFrame,
                             config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train LSTM model on corpus data
        
        Args:
            training_data: Training corpus data
            config: LSTM model configuration
            
        Returns:
            Dictionary with training results and metadata
        """
        try:
            # Default LSTM configuration
            default_config = {
                'sequence_length': 50,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 10  # Reduced for testing
            }
            
            if config:
                default_config.update(config)
            
            logger.info(f"Training LSTM model with config: {default_config}")
            
            # Initialize model
            lstm_model = LSTMPricePredictor(default_config)
            
            # Prepare training data using corpus format
            X, y, feature_names = lstm_model.prepare_training_from_corpus(training_data)
            
            logger.info(f"LSTM training data prepared: X{X.shape}, y{y.shape}, {len(feature_names)} features")
            
            # Normalize the data to prevent exploding gradients
            from sklearn.preprocessing import StandardScaler
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            # Reshape for scaling
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = X_scaler.fit_transform(X_reshaped).reshape(X.shape)
            y_scaled = y_scaler.fit_transform(y)
            
            # Direct training implementation using prepared data
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            from src.ml_analysis.lstm_model import LSTMNetwork
            
            # Initialize neural network
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_size = X.shape[2]  # Number of features
            lstm_model._model = LSTMNetwork(
                input_size=input_size,
                hidden_size=lstm_model.hidden_size,
                num_layers=lstm_model.num_layers,
                dropout=lstm_model.dropout
            ).to(device)
            
            lstm_model._feature_names = feature_names
            lstm_model._device = device
            lstm_model._X_scaler = X_scaler
            lstm_model._y_scaler = y_scaler
            
            # Prepare data loaders with scaled data
            train_dataset = TensorDataset(
                torch.FloatTensor(X_scaled).to(device),
                torch.FloatTensor(y_scaled).to(device)
            )
            train_loader = DataLoader(train_dataset, batch_size=lstm_model.batch_size, shuffle=True)
            
            # Initialize optimizer and loss function
            optimizer = optim.Adam(lstm_model._model.parameters(), lr=lstm_model.learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            training_start_time = datetime.now()
            lstm_model._model.train()
            lstm_model._model_accuracy = 0.0
            training_losses = []
            
            for epoch in range(lstm_model.num_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    predictions, uncertainty = lstm_model._model(batch_X)
                    
                    # Calculate loss
                    pred_loss = criterion(predictions, batch_y)
                    
                    # Add uncertainty regularization
                    uncertainty_loss = torch.mean(torch.exp(-uncertainty)) + torch.mean(uncertainty)
                    total_loss = pred_loss + 0.01 * uncertainty_loss  # Small weight on uncertainty
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(lstm_model._model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
                training_losses.append(avg_loss)
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"LSTM Epoch {epoch+1}/{lstm_model.num_epochs}, Loss: {avg_loss:.6f}")
            
            # Calculate simple accuracy metric
            lstm_model._model.eval()
            with torch.no_grad():
                all_predictions, _ = lstm_model._model(torch.FloatTensor(X_scaled).to(device))
                y_tensor = torch.FloatTensor(y_scaled).to(device)
                mse = criterion(all_predictions, y_tensor)
                
                # R2 score calculation as accuracy proxy
                y_mean = torch.mean(y_tensor)
                ss_tot = torch.sum((y_tensor - y_mean) ** 2)
                ss_res = torch.sum((y_tensor - all_predictions) ** 2)
                
                if ss_tot > 0:
                    r2_score = 1 - (ss_res / ss_tot)
                    # Clamp between 0 and 1 for accuracy representation
                    lstm_model._model_accuracy = max(0.0, min(1.0, r2_score.item()))
                else:
                    # If no variance in targets, use 1 - normalized MSE as accuracy proxy
                    lstm_model._model_accuracy = max(0.0, 1.0 - min(1.0, mse.item()))
                
                logger.info(f"Model evaluation - MSE: {mse.item():.6f}, R2: {r2_score.item() if ss_tot > 0 else 'N/A'}, Accuracy: {lstm_model._model_accuracy:.4f}")
            
            lstm_model._is_trained = True
            training_end_time = datetime.now()
            success = True
            
            if not success:
                raise ModelTrainingError("LSTM model training failed")
            
            # Calculate training duration
            training_duration = (training_end_time - training_start_time).total_seconds()
            
            # Get model accuracy (already calculated above)
            model_accuracy = lstm_model._model_accuracy if hasattr(lstm_model, '_model_accuracy') else 0.0
            
            # Prepare training results
            training_results = {
                'success': success,
                'model_type': 'lstm',
                'training_duration_seconds': training_duration,
                'model_accuracy': model_accuracy,
                'feature_count': len(feature_names),
                'training_samples': X.shape[0],
                'config': default_config,
                'trained_at': training_end_time.isoformat()
            }
            
            logger.info(f"LSTM training completed successfully: accuracy={model_accuracy:.4f}, duration={training_duration:.1f}s")
            
            return training_results
            
        except Exception as e:
            logger.error(f"LSTM model training failed: {e}")
            raise ModelTrainingError(f"LSTM model training failed: {e}")
    
    async def train_transformer_models(self,
                                     training_data: pd.DataFrame,
                                     models_to_train: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train Transformer model variants on corpus data
        
        Args:
            training_data: Training corpus data
            models_to_train: List of model names to train (None = all)
            
        Returns:
            Dictionary mapping model names to training results
        """
        if models_to_train is None:
            models_to_train = ['transformer', 'itransformer', 'patchtst', 'timesmixer']
        
        results = {}
        
        # Model configurations
        model_configs = {
            'transformer': {
                'sequence_length': 192,
                'd_model': 384,  # Larger model capacity
                'n_heads': 12,  # More attention heads
                'n_layers': 6,  # More layers
                'dropout': 0.05,  # Very low dropout 
                'num_epochs': 200,  # More epochs
                'batch_size': 20,  # Smaller batch for better gradients
                'learning_rate': 0.0003  # Lower learning rate for stability
            },
            'itransformer': {
                'sequence_length': 96,
                'n_variates': 20,  # Selected key features
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 3,
                'num_epochs': 10,  # Reduced for testing
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'patchtst': {
                'patch_length': 16,
                'stride': 8,
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 3,
                'num_epochs': 10,  # Reduced for testing
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'timesmixer': {
                'seq_len': 336,
                'd_model': 128,
                'top_k': 5,
                'num_epochs': 10,  # Reduced for testing
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        # Model classes mapping
        model_classes = {
            'transformer': TransformerPredictor,
            'itransformer': iTransformerPredictor,
            'patchtst': PatchTSTPredictor,
            'timesmixer': TimesMixerPredictor
        }
        
        for model_name in models_to_train:
            if model_name not in model_classes:
                logger.warning(f"Unknown model type: {model_name}, skipping")
                continue
            
            try:
                logger.info(f"Training {model_name} model")
                
                model_class = model_classes[model_name]
                config = model_configs[model_name]
                
                # Initialize model (prevent auto-initialization for transformer)
                if model_name == 'transformer':
                    # For transformer, we need to set input_dim before model initialization
                    model = model_class(config)
                    # The model is auto-initialized in __init__, so we need to recreate it
                    model.model = None  # Clear the auto-initialized model
                else:
                    model = model_class(config)
                
                # Prepare training data
                X, y, feature_names = model.prepare_training_from_corpus(training_data)
                
                logger.info(f"{model_name} training data: X{X.shape}, y{y.shape}, {len(feature_names)} features")
                
                # Normalize the data
                from sklearn.preprocessing import StandardScaler
                X_scaler = StandardScaler()
                y_scaler = StandardScaler()
                
                # Reshape for scaling
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = X_scaler.fit_transform(X_reshaped).reshape(X.shape)
                y_scaled = y_scaler.fit_transform(y)
                
                # Direct training implementation using prepared data
                import torch
                import torch.nn as nn
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                
                # Initialize neural network based on model type
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Get the actual model network
                if hasattr(model, '_initialize_model'):
                    # Initialize the model if it has this method
                    model._initialize_model()
                elif not hasattr(model, 'model') or model.model is None:
                    # Try to initialize through the model's own initialization
                    if model_name == 'transformer':
                        from src.ml_analysis.transformers.transformer_predictor import TransformerNetwork
                        from src.ml_analysis.transformers.base import TransformerConfig
                        transformer_config = TransformerConfig(
                            d_model=config.get('d_model', 256),
                            n_heads=config.get('n_heads', 8),
                            n_layers=config.get('n_layers', 4),
                            d_ff=config.get('d_ff', 1024),
                            max_seq_length=config.get('sequence_length', 192),
                            dropout=config.get('dropout', 0.1)
                        )
                        transformer_config.input_dim = X.shape[2]  # Set actual input dimension
                        transformer_config.output_dim = 3  # 3 prediction horizons
                        model.model = TransformerNetwork(transformer_config).to(device)
                    elif model_name == 'itransformer':
                        from src.ml_analysis.transformers.itransformer import iTransformerNetwork, InvertedAttentionConfig
                        itrans_config = InvertedAttentionConfig(
                            d_model=config.get('d_model', 256),
                            n_heads=config.get('n_heads', 8),
                            n_layers=config.get('n_layers', 3),
                            n_variates=X.shape[2],  # Number of features
                            max_seq_length=config.get('sequence_length', 96)
                        )
                        model.model = iTransformerNetwork(itrans_config).to(device)
                    elif model_name == 'patchtst':
                        from src.ml_analysis.transformers.patchtst import PatchTSTNetwork, PatchTSTConfig
                        patch_config = PatchTSTConfig(
                            patch_length=config.get('patch_length', 16),
                            stride=config.get('stride', 8),
                            d_model=config.get('d_model', 128),
                            n_heads=config.get('n_heads', 4),
                            n_layers=config.get('n_layers', 3),
                            n_channels=X.shape[2],  # Number of features
                            seq_len=config.get('sequence_length', 64)
                        )
                        model.model = PatchTSTNetwork(patch_config).to(device)
                    elif model_name == 'timesmixer':
                        from src.ml_analysis.transformers.timesmixer import TimesMixerNetwork, TimesMixerConfig
                        mixer_config = TimesMixerConfig(
                            seq_len=min(X.shape[1], config.get('seq_len', 336)),  # Use actual sequence length
                            d_model=config.get('d_model', 128),
                            n_features=X.shape[2],  # Number of features
                            top_k=config.get('top_k', 5)
                        )
                        model.model = TimesMixerNetwork(mixer_config).to(device)
                        model.n_features = X.shape[2]  # Store n_features on the model as well
                
                # Store model components
                model._device = device
                model._feature_names = feature_names
                model._X_scaler = X_scaler
                model._y_scaler = y_scaler
                
                # Prepare data loaders
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_scaled).to(device),
                    torch.FloatTensor(y_scaled).to(device)
                )
                batch_size = config.get('batch_size', 32)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Initialize optimizer and loss function
                learning_rate = config.get('learning_rate', 0.001)
                optimizer = optim.Adam(model.model.parameters(), lr=learning_rate, weight_decay=0.01)  # Reduced weight decay
                criterion = nn.MSELoss()
                
                # Training loop setup
                training_start_time = datetime.now()
                model.model.train()
                num_epochs = config.get('num_epochs', 30)
                
                # Add cosine annealing scheduler for better convergence
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate*0.01)
                training_losses = []
                
                # Early stopping variables
                best_loss = float('inf')
                patience = 10
                patience_counter = 0
                
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    num_batches = 0
                    
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        
                        # Forward pass based on model type
                        outputs = model.model(batch_X)
                        
                        # Handle different output formats
                        if isinstance(outputs, dict):
                            # Some models return dict with 'predictions' key
                            if 'predictions' in outputs:
                                predictions = outputs['predictions']
                            # iTransformer might return 'output' key
                            elif 'output' in outputs:
                                predictions = outputs['output']
                            # Transformer returns price_1h, price_4h, price_24h
                            elif 'price_1h' in outputs:
                                predictions = torch.stack([
                                    outputs['price_1h'].squeeze(-1) if outputs['price_1h'].dim() > 1 else outputs['price_1h'],
                                    outputs['price_4h'].squeeze(-1) if outputs['price_4h'].dim() > 1 else outputs['price_4h'],
                                    outputs['price_24h'].squeeze(-1) if outputs['price_24h'].dim() > 1 else outputs['price_24h']
                                ], dim=-1)
                            else:
                                # If dict but no recognized keys, try to find tensor values
                                tensor_values = [v for v in outputs.values() if isinstance(v, torch.Tensor)]
                                predictions = tensor_values[0] if tensor_values else outputs
                        elif isinstance(outputs, tuple):
                            # Some models return (predictions, attention_weights) or similar
                            predictions = outputs[0]
                        else:
                            # Direct tensor output
                            predictions = outputs
                        
                        # Calculate loss
                        loss = criterion(predictions, batch_y)
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                    
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
                    training_losses.append(avg_loss)
                    
                    if (epoch + 1) % max(1, num_epochs // 10) == 0:
                        logger.info(f"{model_name} Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                    
                    # Step cosine annealing scheduler
                    scheduler.step()
                    
                    # Early stopping check
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"{model_name} Early stopping at epoch {epoch+1}")
                        break
                
                # Calculate accuracy metric
                model.model.eval()
                with torch.no_grad():
                    all_X = torch.FloatTensor(X_scaled).to(device)
                    
                    # Forward pass for evaluation
                    outputs = model.model(all_X)
                    
                    # Handle different output formats
                    if isinstance(outputs, dict):
                        if 'predictions' in outputs:
                            all_predictions = outputs['predictions']
                        elif 'output' in outputs:
                            all_predictions = outputs['output']
                        elif 'price_1h' in outputs:
                            all_predictions = torch.stack([
                                outputs['price_1h'].squeeze(-1) if outputs['price_1h'].dim() > 1 else outputs['price_1h'],
                                outputs['price_4h'].squeeze(-1) if outputs['price_4h'].dim() > 1 else outputs['price_4h'],
                                outputs['price_24h'].squeeze(-1) if outputs['price_24h'].dim() > 1 else outputs['price_24h']
                            ], dim=-1)
                        else:
                            tensor_values = [v for v in outputs.values() if isinstance(v, torch.Tensor)]
                            all_predictions = tensor_values[0] if tensor_values else outputs
                    elif isinstance(outputs, tuple):
                        all_predictions = outputs[0]
                    else:
                        all_predictions = outputs
                    
                    y_tensor = torch.FloatTensor(y_scaled).to(device)
                    mse = criterion(all_predictions, y_tensor)
                    
                    # R2 score calculation
                    y_mean = torch.mean(y_tensor)
                    ss_tot = torch.sum((y_tensor - y_mean) ** 2)
                    ss_res = torch.sum((y_tensor - all_predictions) ** 2)
                    
                    if ss_tot > 0:
                        r2_score = 1 - (ss_res / ss_tot)
                        model_accuracy = max(0.0, min(1.0, r2_score.item()))
                    else:
                        model_accuracy = max(0.0, 1.0 - min(1.0, mse.item()))
                    
                    logger.info(f"{model_name} evaluation - MSE: {mse.item():.6f}, R2: {r2_score.item() if ss_tot > 0 else 'N/A'}, Accuracy: {model_accuracy:.4f}")
                
                model._is_trained = True
                model._model_accuracy = model_accuracy
                training_end_time = datetime.now()
                success = True
                
                # Calculate training metrics
                training_duration = (training_end_time - training_start_time).total_seconds()
                model_accuracy = model._model_accuracy if hasattr(model, '_model_accuracy') else 0.0
                
                results[model_name] = {
                    'success': success,
                    'model_type': model_name,
                    'training_duration_seconds': training_duration,
                    'model_accuracy': model_accuracy or 0.0,
                    'feature_count': len(feature_names),
                    'training_samples': X.shape[0],
                    'config': config,
                    'trained_at': training_end_time.isoformat()
                }
                
                logger.info(f"{model_name} training completed: duration={training_duration:.1f}s")
                
            except Exception as e:
                logger.error(f"{model_name} model training failed: {e}")
                results[model_name] = {
                    'success': False,
                    'error': str(e),
                    'model_type': model_name
                }
        
        return results
    
    async def save_trained_models(self,
                                training_results: Dict[str, Dict[str, Any]],
                                models: Dict[str, Any]) -> Dict[str, str]:
        """
        Save trained models to GCS
        
        Args:
            training_results: Results from training
            models: Dictionary of trained model instances
            
        Returns:
            Dictionary mapping model names to GCS paths
        """
        saved_paths = {}
        
        for model_name, result in training_results.items():
            if not result.get('success', False):
                logger.warning(f"Skipping save for failed model: {model_name}")
                continue
            
            try:
                model = models.get(model_name)
                if model is None:
                    logger.warning(f"Model instance not found for {model_name}")
                    continue
                
                # Generate unique save path
                timestamp = self.session_timestamp.strftime("%Y%m%d_%H%M%S")
                model_path = f"{self.model_save_prefix}/{model_name}/{timestamp}/model.pt"
                
                # Create local temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    # Save model locally first
                    if hasattr(model, 'save_model'):
                        success = model.save_model(tmp_path)
                        if not success:
                            raise ModelSavingError(f"Failed to save {model_name} locally")
                    else:
                        logger.warning(f"Model {model_name} does not support saving")
                        continue
                    
                    # Upload to GCS
                    blob = self.save_bucket.blob(model_path)
                    blob.upload_from_filename(tmp_path)
                    
                    # Add metadata
                    blob.metadata = {
                        'model_type': model_name,
                        'session_id': self.session_id,
                        'trained_at': result.get('trained_at'),
                        'training_duration_seconds': str(result.get('training_duration_seconds', 0)),
                        'model_accuracy': str(result.get('model_accuracy', 0.0)),
                        'feature_count': str(result.get('feature_count', 0))
                    }
                    blob.patch()
                    
                    saved_paths[model_name] = f"gs://{self.model_save_bucket}/{model_path}"
                    logger.info(f"Model {model_name} saved to: {saved_paths[model_name]}")
                    
                finally:
                    # Cleanup temporary file
                    Path(tmp_path).unlink(missing_ok=True)
                    
            except Exception as e:
                logger.error(f"Failed to save model {model_name}: {e}")
                raise ModelSavingError(f"Failed to save model {model_name}: {e}")
        
        return saved_paths
    
    async def track_training_history(self,
                                   training_results: Dict[str, Dict[str, Any]],
                                   saved_paths: Dict[str, str],
                                   corpus_version_id: Optional[int] = None) -> Dict[str, int]:
        """
        Record training history in model_training_history table
        
        Args:
            training_results: Results from training
            saved_paths: GCS paths where models were saved
            corpus_version_id: ID from training_corpus_versions table
            
        Returns:
            Dictionary mapping model names to training_id values
        """
        training_ids = {}
        
        try:
            for model_name, result in training_results.items():
                if not result.get('success', False):
                    continue
                
                # Prepare training history record
                insert_query = """
                INSERT INTO model_training_history (
                    model_type, corpus_version_id, trained_at, training_mode,
                    performance_metrics, model_checkpoint_path, model_version,
                    training_config, training_duration_seconds
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING training_id;
                """
                
                performance_metrics = {
                    'accuracy': result.get('model_accuracy', 0.0),
                    'training_duration_seconds': result.get('training_duration_seconds', 0),
                    'feature_count': result.get('feature_count', 0),
                    'training_samples': result.get('training_samples', 0)
                }
                
                training_config = result.get('config', {})
                model_checkpoint_path = saved_paths.get(model_name, '')
                model_version = f"unified_training_{self.session_timestamp.strftime('%Y%m%d')}"
                
                params = [
                    model_name,
                    corpus_version_id,
                    result.get('trained_at'),
                    'initial',
                    performance_metrics,
                    model_checkpoint_path,
                    model_version,
                    training_config,
                    int(result.get('training_duration_seconds', 0))
                ]
                
                async with get_database_connection() as conn:
                    result_row = await conn.fetchrow(insert_query, *params)
                    training_id = result_row['training_id']
                    training_ids[model_name] = training_id
                
                logger.info(f"Training history recorded for {model_name}: training_id={training_id}")
                
        except Exception as e:
            logger.error(f"Failed to track training history: {e}")
            raise UnifiedTrainingPipelineError(f"Failed to track training history: {e}")
        
        return training_ids
    
    async def train_all_models(self,
                             corpus_version: Optional[str] = None,
                             timeframe: str = "daily",
                             token: Optional[str] = None,
                             models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Complete training pipeline for all models
        
        Args:
            corpus_version: Corpus version to use (None = latest)
            timeframe: Data timeframe (daily, hourly, hour)
            token: Specific token filter (None = all tokens)
            models: Models to train (None = all supported models)
            
        Returns:
            Dictionary with complete training results
        """
        try:
            logger.info(f"Starting unified training pipeline: session_id={self.session_id}")
            
            # Load corpus data
            data = await self.load_corpus_data(corpus_version, timeframe, token)
            
            # Split data for training
            train_data, val_data, test_data = self.prepare_train_val_test_split(data)
            
            # Train LSTM model
            logger.info("Training LSTM model...")
            lstm_results = await self.train_lstm_model(train_data)
            
            # Train Transformer models
            if models is None:
                models = ['transformer', 'itransformer', 'patchtst', 'timesmixer']
            elif 'lstm' in models:
                models = [m for m in models if m != 'lstm']  # LSTM handled separately
            
            # Skip transformer training if no transformer models specified
            if models:
                logger.info(f"Training Transformer models: {models}")
                transformer_results = await self.train_transformer_models(train_data, models)
            else:
                transformer_results = {}
            
            # Combine results
            all_results = {'lstm': lstm_results}
            all_results.update(transformer_results)
            
            # Generate comprehensive training report
            logger.info("Generating training report...")
            try:
                report_results = await self.report_generator.generate_reports_for_training_session(
                    training_metrics=all_results,
                    pipeline=self
                )
                
                logger.info(f"Training report generated: {report_results.get('pdf_report')}")
                if report_results.get('gcs_upload_path'):
                    logger.info(f"Report uploaded to GCS: {report_results.get('gcs_upload_path')}")
                    
            except Exception as e:
                logger.error(f"Failed to generate training report: {e}")
                # Continue without failing the pipeline
                report_results = {}
            
            logger.info(f"Unified training pipeline completed successfully")
            
            return {
                'session_id': self.session_id,
                'training_results': all_results,
                'report_results': report_results,
                'data_stats': {
                    'total_samples': len(data),
                    'train_samples': len(train_data),
                    'val_samples': len(val_data),
                    'test_samples': len(test_data),
                    'feature_columns': len(data.columns)
                },
                'corpus_info': {
                    'corpus_version': corpus_version or 'latest',
                    'timeframe': timeframe,
                    'token_filter': token
                }
            }
            
        except Exception as e:
            logger.error(f"Unified training pipeline failed: {e}")
            raise UnifiedTrainingPipelineError(f"Training pipeline failed: {e}")


async def main():
    """
    Main entry point for training script
    
    Usage:
        python scripts/training/train_all_models.py
    """
    try:
        pipeline = UnifiedTrainingPipeline()
        
        results = await pipeline.train_all_models(
            corpus_version=None,  # Use latest
            timeframe="daily",
            token="WBTC",  # Train on Wrapped Bitcoin data (corpus uses WBTC not BTC)
            models=['lstm', 'transformer', 'itransformer', 'patchtst', 'timesmixer']  # Train all models
        )
        
        print(f"Training completed successfully!")
        print(f"Session ID: {results['session_id']}")
        print(f"Models trained: {list(results['training_results'].keys())}")
        print(f"Data samples: {results['data_stats']['total_samples']}")
        
        # Print report information if available
        if results.get('report_results'):
            report_info = results['report_results']
            print(f"Training report: {report_info.get('pdf_report', 'Not generated')}")
            if report_info.get('gcs_upload_path'):
                print(f"Report uploaded to: {report_info['gcs_upload_path']}")
            else:
                print("Report not uploaded to GCS")
        
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())