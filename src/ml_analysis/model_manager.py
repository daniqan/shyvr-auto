"""
Model Manager
Coordinates multiple ML models and provides ensemble predictions
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import structlog
import pickle

from src.discovery.base import DiscoveredToken
from .base import (
    MLAnalyzerBase, PredictionResult, ModelType, PredictionDirection,
    MLAnalysisError
)
from .lstm_model import LSTMPricePredictor
from .feature_engineer import FeatureEngineer
from .transformers import TransformerPredictor
from .transformers.itransformer import iTransformerPredictor
from .transformers.patchtst import PatchTSTPredictor
from .transformers.timesmixer import TimesMixerPredictor
from .transformers.timesfm_wrapper import TimesFMWrapper
from .market_data import FearGreedIndexClient, MarketSentimentData
from src.activity_logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    performance_tracker
)

# Import preservation components
try:
    from src.model_preservation.manager import PreservationManager, PreservationConfig
    from src.model_preservation.base import PreservationError, PreservationPriority
    PRESERVATION_AVAILABLE = True
except ImportError:
    PRESERVATION_AVAILABLE = False
    PreservationManager = None
    PreservationConfig = None
    PreservationError = Exception
    PreservationPriority = None


logger = structlog.get_logger()


class ModelManager:
    """Manages multiple ML models and provides ensemble predictions"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = structlog.get_logger().bind(component="ModelManager")
        
        # Model storage
        self._models: Dict[ModelType, MLAnalyzerBase] = {}
        self._model_weights: Dict[ModelType, float] = {}
        self._feature_engineer = FeatureEngineer()
        
        # Performance tracking
        self._model_performance: Dict[ModelType, Dict[str, float]] = {}
        self._ensemble_cache: Dict[str, Tuple[datetime, PredictionResult]] = {}
        self._cache_ttl_minutes = self.config.get('cache_ttl_minutes', 15)
        
        # Fear & Greed Index client for sentiment-based weight adjustment
        self._fear_greed_client = FearGreedIndexClient()
        self._last_sentiment_data: Optional[MarketSentimentData] = None
        self._sentiment_cache_ttl_minutes = 10  # Cache sentiment data for 10 minutes
        
        # Model file paths
        self.model_dir = Path(self.config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)
        
        # Current operational mode
        self._current_mode = 'analysis'
        
        # Initialize preservation manager if enabled
        self._preservation_manager = None
        preservation_config = self.config.get('preservation', {})
        if PRESERVATION_AVAILABLE and preservation_config.get('enabled', False):
            try:
                pres_config = PreservationConfig(
                    gcs_bucket=preservation_config.get('gcs_bucket', 'shyvr-models-prod'),
                    backup_interval_hours=preservation_config.get('backup_interval_hours', 6),
                    max_versions_per_model=preservation_config.get('max_versions_per_model', 10),
                    enable_compression=preservation_config.get('enable_compression', True),
                    mode_isolation=preservation_config.get('mode_isolation', True)
                )
                self._preservation_manager = PreservationManager(pres_config)
                self.logger.info("Model preservation enabled", bucket=pres_config.gcs_bucket)
            except Exception as e:
                self.logger.warning("Failed to initialize preservation manager", error=str(e))
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models based on environment"""
        import os
        
        try:
            # Get environment from environment variable or config
            environment = os.environ.get('ENVIRONMENT', 
                                       self.config.get('environment', 'production'))
            
            self.logger.info("Initializing ML models", 
                           model_dir=str(self.model_dir),
                           environment=environment)
            
            # Always initialize LSTM model (used in all environments)
            lstm_config = self.config.get('lstm', {})
            self._models[ModelType.LSTM] = LSTMPricePredictor(lstm_config)
            
            if environment == 'development':
                # Development mode: LSTM only
                self._model_weights[ModelType.LSTM] = 1.0
                
                self.logger.info("Development mode: initialized LSTM only",
                               models=[ModelType.LSTM.value])
                
            else:
                # Production/staging mode: Full ensemble
                self._model_weights[ModelType.LSTM] = 0.2  # Reduced for 6-model ensemble
                
                # Initialize basic Transformer model
                transformer_config = self.config.get('transformer', {})
                self._models[ModelType.TRANSFORMER] = TransformerPredictor(transformer_config)
                self._model_weights[ModelType.TRANSFORMER] = 0.15  # Base Transformer weight
                
                # Initialize iTransformer model (inverted attention for multivariate)
                itransformer_config = self.config.get('itransformer', transformer_config)
                self._models[ModelType.ITRANSFORMER] = iTransformerPredictor(itransformer_config)
                self._model_weights[ModelType.ITRANSFORMER] = 0.2  # Higher weight for multivariate focus
                
                # Initialize PatchTST model (patch-based for long sequences)
                patchtst_config = self.config.get('patchtst', transformer_config)
                self._models[ModelType.PATCHTST] = PatchTSTPredictor(patchtst_config)
                self._model_weights[ModelType.PATCHTST] = 0.15  # Good for long-horizon predictions
                
                # Initialize TimesMixer model (decomposition-based mixing)
                timesmixer_config = self.config.get('timesmixer', transformer_config)
                self._models[ModelType.TIMESMIXER] = TimesMixerPredictor(timesmixer_config)
                self._model_weights[ModelType.TIMESMIXER] = 0.1  # Experimental model, lower initial weight
                
                # Initialize TimesFM model (Google's foundation model for zero-shot predictions)
                timesfm_config = self.config.get('timesfm', {
                    'model_name': 'google/timesfm-1.0-200m',
                    'prediction_length': 24,
                    'context_length': 512,
                    'use_zero_shot': True,
                    'gcp_optimized': True
                })
                self._models[ModelType.TIMESFM] = TimesFMWrapper(timesfm_config)
                self._model_weights[ModelType.TIMESFM] = 0.2  # Higher weight due to foundation model capabilities
                
                # Normalize weights to sum to 1.0
                total_weight = sum(self._model_weights.values())
                if total_weight != 1.0:
                    for model_type in self._model_weights:
                        self._model_weights[model_type] /= total_weight
                
                self.logger.info("Production/staging mode: initialized full ensemble",
                               models=[mt.value for mt in self._models.keys()])
            
            # Initialize performance tracking for all models
            for model_type in self._models.keys():
                self._model_performance[model_type] = {
                    'accuracy': 0.0,
                    'predictions_made': 0,
                    'last_updated': datetime.now().timestamp(),
                    'memory_usage_mb': 0.0,  # Track Transformer memory usage
                    'avg_inference_time_ms': 0.0  # Track inference performance
                }
            
            self.logger.info("Models initialized successfully", 
                           environment=environment,
                           models=[mt.value for mt in self._models.keys()],
                           model_weights=self._model_weights,
                           model_dir=str(self.model_dir))
            
        except Exception as e:
            self.logger.error("Model initialization failed", error=str(e))
            raise MLAnalysisError(f"Failed to initialize models: {str(e)}")
    
    async def analyze_token(self, token: DiscoveredToken,
                          historical_data: Optional[pd.DataFrame] = None,
                          use_ensemble: bool = True) -> PredictionResult:
        """
        Analyze a token using ML models
        
        Args:
            token: Token to analyze
            historical_data: Historical price data
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            PredictionResult with ML predictions
        """
        cache_key = f"{token.chain_address}_{use_ensemble}"
        
        # Check cache first
        if cache_key in self._ensemble_cache:
            cached_time, cached_result = self._ensemble_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self._cache_ttl_minutes):
                # Log cache hit
                await activity_logger.log_activity(
                    category=ActivityCategory.ML_RL,
                    action=ActivityAction.READ,
                    source="model_manager",
                    event_type="prediction_cache_hit",
                    title=f"Returning cached prediction for {token.address}",
                    severity=ActivitySeverity.DEBUG,
                    token_address=token.address,
                    metadata={
                        "cache_key": cache_key,
                        "cached_time": cached_time.isoformat(),
                        "use_ensemble": use_ensemble,
                        "confidence": cached_result.confidence
                    }
                )
                self.logger.debug("Returning cached prediction", token=token.address)
                return cached_result
        
        # Log analysis start
        await activity_logger.log_activity(
            category=ActivityCategory.ML_RL,
            action=ActivityAction.EXECUTE,
            source="model_manager",
            event_type="token_analysis_started",
            title=f"Starting ML analysis for token {token.address}",
            severity=ActivitySeverity.INFO,
            token_address=token.address,
            metadata={
                "token_symbol": token.symbol,
                "token_name": token.name,
                "use_ensemble": use_ensemble,
                "has_historical_data": historical_data is not None,
                "available_models": [model_type.value for model_type in self._models.keys()]
            }
        )
        
        async with performance_tracker(
            source="model_manager",
            operation="analyze_token",
            category=ActivityCategory.ML_RL,
            metadata={"token": token.address, "use_ensemble": use_ensemble}
        ) as tracker:
            try:
                start_time = datetime.now()
                
                if use_ensemble and len(self._models) > 1:
                    result = await self._ensemble_prediction(token, historical_data)
                else:
                    # Use best performing single model
                    best_model = self._get_best_model()
                    result = await best_model.analyze_token(token, historical_data)
                
                # Calculate inference time
                inference_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Cache the result
                self._ensemble_cache[cache_key] = (datetime.now(), result)
                
                # Update model performance tracking with inference time
                await self._update_performance_tracking(result, inference_time_ms)
                
                # Log successful analysis
                await activity_logger.log_activity(
                    category=ActivityCategory.ML_RL,
                    action=ActivityAction.SUCCESS,
                    source="model_manager",
                    event_type="token_analysis_completed",
                    title=f"ML analysis completed for token {token.address}",
                    severity=ActivitySeverity.INFO,
                    token_address=token.address,
                    metadata={
                        "model_type": result.model_type.value if hasattr(result, 'model_type') and hasattr(result.model_type, 'value') else str(getattr(result, 'model_type', 'unknown')),
                        "confidence": result.confidence,
                        "direction": result.direction.value if hasattr(result, 'direction') and hasattr(result.direction, 'value') else str(getattr(result, 'direction', 'unknown')),
                        "use_ensemble": use_ensemble,
                        "cached": True
                    }
                )
                
                self.logger.info("Token analysis completed",
                               token=token.address,
                               model_type=result.model_type.value if hasattr(result, 'model_type') and hasattr(result.model_type, 'value') else str(getattr(result, 'model_type', 'unknown')),
                               confidence=result.confidence,
                               direction=result.direction.value if hasattr(result, 'direction') and hasattr(result.direction, 'value') else str(getattr(result, 'direction', 'unknown')))
                
                return result
                
            except Exception as e:
                # Log analysis failure
                await activity_logger.log_error(
                    category=ActivityCategory.ML_RL,
                    source="model_manager",
                    event_type="token_analysis_failed",
                    title=f"ML analysis failed for token {token.address}",
                    error_message=str(e),
                    exception=e,
                    severity=ActivitySeverity.ERROR,
                    token_address=token.address,
                    metadata={
                        "token_symbol": token.symbol,
                        "token_name": token.name,
                        "use_ensemble": use_ensemble,
                        "available_models": [model_type.value for model_type in self._models.keys()]
                    }
                )
                
                self.logger.error("Token analysis failed", 
                                token=token.address, 
                                error=str(e))
                raise MLAnalysisError(f"Analysis failed for {token.address}: {str(e)}")
    
    async def batch_analyze(self, tokens: List[DiscoveredToken],
                          historical_data: Optional[Dict[str, pd.DataFrame]] = None,
                          use_ensemble: bool = True) -> List[PredictionResult]:
        """Analyze multiple tokens in batch"""
        try:
            # Run analyses concurrently
            tasks = []
            for token in tokens:
                token_data = historical_data.get(token.address) if historical_data else None
                task = self.analyze_token(token, token_data, use_ensemble)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log errors
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error("Batch analysis item failed", 
                                    token=tokens[i].address,
                                    error=str(result))
                else:
                    valid_results.append(result)
            
            self.logger.info("Batch analysis completed",
                           total_tokens=len(tokens),
                           successful=len(valid_results),
                           failed=len(tokens) - len(valid_results))
            
            return valid_results
            
        except Exception as e:
            self.logger.error("Batch analysis failed", error=str(e))
            raise MLAnalysisError(f"Batch analysis failed: {str(e)}")
    
    async def train_models(self, training_data: pd.DataFrame,
                         model_types: Optional[List[ModelType]] = None) -> Dict[ModelType, bool]:
        """
        Train specified models on historical data
        
        Args:
            training_data: Historical training data
            model_types: Which models to train (None = all)
            
        Returns:
            Dictionary of training results
        """
        if model_types is None:
            model_types = list(self._models.keys())
        
        training_results = {}
        
        for model_type in model_types:
            if model_type not in self._models:
                self.logger.warning("Model type not available", model_type=model_type)
                continue
            
            try:
                self.logger.info("Starting model training", model_type=model_type.value)
                success = await self._models[model_type].train_model(training_data)
                training_results[model_type] = success
                
                # Save model after successful training
                if success:
                    await self._save_model(model_type)
                    self.logger.info("Model training completed", model_type=model_type.value)
                else:
                    self.logger.error("Model training failed", model_type=model_type.value)
                    
            except Exception as e:
                self.logger.error("Model training exception", 
                                model_type=model_type.value,
                                error=str(e))
                training_results[model_type] = False
        
        # Update model weights based on training success
        await self._update_model_weights()
        
        return training_results
    
    async def _ensemble_prediction(self, token: DiscoveredToken,
                                 historical_data: Optional[pd.DataFrame]) -> PredictionResult:
        """Generate ensemble prediction from multiple models"""
        try:
            # Get predictions from all available models
            model_predictions = {}
            
            for model_type, model in self._models.items():
                if not model.is_model_trained():
                    self.logger.warning("Skipping untrained model", model_type=model_type.value)
                    continue
                
                try:
                    prediction = await model.analyze_token(token, historical_data)
                    model_predictions[model_type] = prediction
                except Exception as e:
                    self.logger.error("Model prediction failed", 
                                    model_type=model_type.value,
                                    error=str(e))
            
            if not model_predictions:
                raise MLAnalysisError("No models available for ensemble prediction")
            
            # Combine predictions using weighted average
            ensemble_result = self._combine_predictions(model_predictions, token)
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error("Ensemble prediction failed", error=str(e))
            raise MLAnalysisError(f"Ensemble prediction failed: {str(e)}")
    
    def _combine_predictions(self, predictions: Dict[ModelType, PredictionResult],
                           token: DiscoveredToken) -> PredictionResult:
        """Combine multiple predictions into ensemble result with robust weighting"""
        try:
            # Calculate individual timeframe weights for proper handling of missing predictions
            timeframe_weights = {'1h': {}, '4h': {}, '24h': {}}
            timeframe_totals = {'1h': 0.0, '4h': 0.0, '24h': 0.0}
            
            # Calculate available weights for each timeframe
            for model_type, prediction in predictions.items():
                base_weight = self._model_weights[model_type] * prediction.confidence
                
                if prediction.price_prediction_1h is not None:
                    timeframe_weights['1h'][model_type] = base_weight
                    timeframe_totals['1h'] += base_weight
                    
                if prediction.price_prediction_4h is not None:
                    timeframe_weights['4h'][model_type] = base_weight
                    timeframe_totals['4h'] += base_weight
                    
                if prediction.price_prediction_24h is not None:
                    timeframe_weights['24h'][model_type] = base_weight
                    timeframe_totals['24h'] += base_weight
            
            # Normalize weights for each timeframe separately
            for timeframe in timeframe_weights:
                if timeframe_totals[timeframe] > 0:
                    for model_type in timeframe_weights[timeframe]:
                        timeframe_weights[timeframe][model_type] /= timeframe_totals[timeframe]
            
            # Calculate weighted predictions with proper normalization
            ensemble_predictions = {'1h': 0.0, '4h': 0.0, '24h': 0.0}
            prediction_variances = {'1h': 0.0, '4h': 0.0, '24h': 0.0}
            
            # Calculate ensemble predictions for each timeframe
            for timeframe in ['1h', '4h', '24h']:
                if timeframe_totals[timeframe] > 0:
                    predictions_for_timeframe = []
                    weights_for_timeframe = []
                    
                    for model_type, prediction in predictions.items():
                        if model_type in timeframe_weights[timeframe]:
                            price_key = f'price_prediction_{timeframe}'
                            price_pred = getattr(prediction, price_key, None)
                            if price_pred is not None:
                                predictions_for_timeframe.append(price_pred)
                                weights_for_timeframe.append(timeframe_weights[timeframe][model_type])
                    
                    if predictions_for_timeframe:
                        # Weighted average
                        ensemble_predictions[timeframe] = sum(
                            pred * weight for pred, weight in zip(predictions_for_timeframe, weights_for_timeframe)
                        )
                        
                        # Calculate prediction variance for uncertainty quantification
                        if len(predictions_for_timeframe) > 1:
                            weighted_mean = ensemble_predictions[timeframe]
                            prediction_variances[timeframe] = sum(
                                weight * (pred - weighted_mean) ** 2 
                                for pred, weight in zip(predictions_for_timeframe, weights_for_timeframe)
                            )
            
            # Calculate overall ensemble confidence and probability
            total_weight = sum(self._model_weights[mt] * pred.confidence 
                             for mt, pred in predictions.items())
            
            if total_weight == 0:
                total_weight = 1.0  # Prevent division by zero
            
            ensemble_confidence = 0.0
            ensemble_prob_up = 0.0
            
            for model_type, prediction in predictions.items():
                model_weight = self._model_weights[model_type] * prediction.confidence / total_weight
                ensemble_confidence += prediction.confidence * model_weight
                ensemble_prob_up += prediction.probability_up * model_weight
            
            # Determine ensemble direction based on 24h prediction
            current_price = token.price_usd or 1.0
            direction = PredictionDirection.HOLD
            
            if ensemble_predictions['24h'] > 0:
                price_change_24h = (ensemble_predictions['24h'] - current_price) / current_price
                
                if price_change_24h > 0.1:
                    direction = PredictionDirection.STRONG_BUY
                elif price_change_24h > 0.02:
                    direction = PredictionDirection.BUY
                elif price_change_24h < -0.1:
                    direction = PredictionDirection.STRONG_SELL
                elif price_change_24h < -0.02:
                    direction = PredictionDirection.SELL
                else:
                    direction = PredictionDirection.HOLD
            
            # Use the best prediction's technical indicators
            best_prediction = max(predictions.values(), key=lambda p: p.confidence)
            
            # Calculate uncertainty as the maximum variance across timeframes
            max_variance = max(prediction_variances.values())
            prediction_uncertainty = min(max_variance / (current_price ** 2), 1.0) if max_variance > 0 else 0.0
            
            # Adjust confidence based on prediction agreement (lower variance = higher confidence)
            variance_penalty = prediction_uncertainty * 0.2  # Reduce confidence by up to 20% for high variance
            adjusted_confidence = max(ensemble_confidence - variance_penalty, 0.1)
            
            # Create ensemble result
            ensemble_result = PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.ENSEMBLE,
                price_prediction_1h=ensemble_predictions['1h'] if ensemble_predictions['1h'] > 0 else None,
                price_prediction_4h=ensemble_predictions['4h'] if ensemble_predictions['4h'] > 0 else None,
                price_prediction_24h=ensemble_predictions['24h'] if ensemble_predictions['24h'] > 0 else None,
                direction=direction,
                confidence=min(adjusted_confidence, 1.0),
                probability_up=min(ensemble_prob_up, 1.0),
                prediction_uncertainty=prediction_uncertainty,
                technical_indicators=best_prediction.technical_indicators,
                market_features=best_prediction.market_features,
                model_accuracy=sum(p.model_accuracy or 0 for p in predictions.values()) / len(predictions),
                features_used=[f"ensemble_{len(predictions)}_models"] + [f"{mt.value}_weight_{self._model_weights[mt]:.3f}" for mt in predictions.keys()],
                model_version="ensemble_2.0",
            )
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error("Prediction combination failed", error=str(e))
            raise MLAnalysisError(f"Failed to combine predictions: {str(e)}")
    
    def _get_best_model(self) -> MLAnalyzerBase:
        """Get the best performing trained model"""
        trained_models = {mt: model for mt, model in self._models.items() 
                         if model.is_model_trained()}
        
        if not trained_models:
            # Return first available model even if untrained
            return next(iter(self._models.values()))
        
        # Find model with highest weighted performance
        best_model_type = max(trained_models.keys(), 
                            key=lambda mt: self._model_weights[mt] * 
                                         self._model_performance[mt]['accuracy'])
        
        return trained_models[best_model_type]
    
    async def _update_performance_tracking(self, result: PredictionResult, inference_time_ms: float = 0.0):
        """Update model performance metrics with Transformer-specific monitoring"""
        try:
            model_type = result.model_type
            if model_type not in self._model_performance:
                return
            
            # Update prediction count
            self._model_performance[model_type]['predictions_made'] += 1
            
            # Update accuracy if available
            if result.model_accuracy is not None:
                current_accuracy = self._model_performance[model_type]['accuracy']
                prediction_count = self._model_performance[model_type]['predictions_made']
                
                # Running average
                new_accuracy = ((current_accuracy * (prediction_count - 1)) + result.model_accuracy) / prediction_count
                self._model_performance[model_type]['accuracy'] = new_accuracy
            
            # Update inference time tracking
            if inference_time_ms > 0:
                current_time = self._model_performance[model_type].get('avg_inference_time_ms', 0.0)
                prediction_count = self._model_performance[model_type]['predictions_made']
                
                # Running average for inference time
                new_avg_time = ((current_time * (prediction_count - 1)) + inference_time_ms) / prediction_count
                self._model_performance[model_type]['avg_inference_time_ms'] = new_avg_time
            
            # Update memory usage for Transformer models
            if model_type in [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST, ModelType.TIMESMIXER]:
                memory_usage = await self._get_model_memory_usage(model_type)
                self._model_performance[model_type]['memory_usage_mb'] = memory_usage
            
            self._model_performance[model_type]['last_updated'] = datetime.now().timestamp()
            
        except Exception as e:
            self.logger.error("Performance tracking update failed", error=str(e))
    
    async def _get_model_memory_usage(self, model_type: ModelType) -> float:
        """Get memory usage for a specific model in MB"""
        try:
            import psutil
            import gc
            import torch
            
            model = self._models.get(model_type)
            if model is None:
                return 0.0
            
            # For PyTorch models, get GPU memory if available
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                try:
                    # Calculate model parameter memory
                    param_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
                    
                    # Check GPU memory if using CUDA
                    if torch.cuda.is_available() and next(model.model.parameters()).is_cuda:
                        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
                        return max(param_size / (1024 * 1024), gpu_memory)
                    else:
                        return param_size / (1024 * 1024)  # Convert to MB
                        
                except Exception:
                    pass
            
            # Fallback: estimate based on process memory
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Rough estimate: divide by number of models for per-model usage
            return memory_mb / len(self._models)
            
        except Exception as e:
            self.logger.debug("Failed to get memory usage", model_type=model_type.value, error=str(e))
            return 0.0
    
    async def _update_model_weights(self, market_regime: str = "normal", fear_greed_regime: Optional[str] = None):
        """Update model weights based on performance with Fear & Greed Index integration"""
        try:
            start_time = datetime.now()
            current_time = start_time.timestamp()
            
            # Get current fear/greed sentiment if not provided
            if fear_greed_regime is None:
                fear_greed_regime = await self.get_fear_greed_regime()
            
            # Store current base weights for comparison
            original_weights = self._model_weights.copy()
            
            for model_type, performance in self._model_performance.items():
                accuracy = performance['accuracy']
                prediction_count = performance['predictions_made']
                last_updated = performance['last_updated']
                memory_usage = performance.get('memory_usage_mb', 0.0)
                inference_time = performance.get('avg_inference_time_ms', 0.0)
                
                # Calculate time-based decay factor (performance degrades over time)
                time_since_update = current_time - last_updated
                hours_since_update = time_since_update / 3600  # Convert to hours
                
                # Decay factor: 1.0 for recent (< 1 hour), decays to 0.5 over 24 hours
                decay_factor = max(0.5, 1.0 - (hours_since_update / 48))  # 48 hours to reach 0.5
                
                # Experience factor: models with more predictions get higher weight
                experience_factor = min(prediction_count / 100, 1.0)  # Max experience at 100 predictions
                
                # Recency bonus: models that have been updated recently get a small bonus
                recency_bonus = 1.0 if hours_since_update < 1 else max(0.9, 1.0 - (hours_since_update / 24))
                
                # Transformer-specific efficiency factors
                efficiency_factor = 1.0
                if model_type in [ModelType.TRANSFORMER, ModelType.ITRANSFORMER, ModelType.PATCHTST, ModelType.TIMESMIXER, ModelType.TIMESFM]:
                    # Memory efficiency penalty for high memory usage (>2GB)
                    memory_penalty = max(0.8, 1.0 - (memory_usage / 2048))  # Penalty starts at 2GB
                    
                    # Inference speed bonus for fast models (<100ms)
                    speed_bonus = 1.1 if inference_time < 100 else max(0.9, 1.0 - (inference_time / 1000))
                    
                    efficiency_factor = memory_penalty * speed_bonus
                
                # Market regime-aware weighting
                regime_factor = self._get_regime_factor(model_type, market_regime)
                
                # Fear & Greed sentiment-based weight adjustment
                sentiment_factor = self._get_sentiment_factor(model_type, fear_greed_regime)
                
                # Combined weight calculation with new factors
                base_weight = accuracy * (0.3 + 0.7 * experience_factor)
                adjusted_weight = base_weight * decay_factor * recency_bonus * efficiency_factor * regime_factor * sentiment_factor
                
                # Ensure minimum weight to prevent models from being completely ignored
                self._model_weights[model_type] = max(adjusted_weight, 0.02)  # Lower minimum for more models
            
            # Normalize weights to sum to 1.0
            total_weight = sum(self._model_weights.values())
            if total_weight > 0:
                for model_type in self._model_weights:
                    self._model_weights[model_type] /= total_weight
            
            # Calculate adjustment latency
            adjustment_latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            self.logger.info("Enhanced model weights updated with Fear & Greed integration", 
                           weights=self._model_weights,
                           original_weights=original_weights,
                           market_regime=market_regime,
                           fear_greed_regime=fear_greed_regime,
                           adjustment_latency_ms=adjustment_latency_ms,
                           performance_metrics={
                               mt.value: {
                                   'accuracy': perf['accuracy'],
                                   'predictions': perf['predictions_made'],
                                   'hours_since_update': (current_time - perf['last_updated']) / 3600,
                                   'memory_mb': perf.get('memory_usage_mb', 0),
                                   'inference_ms': perf.get('avg_inference_time_ms', 0)
                               } for mt, perf in self._model_performance.items()
                           })
            
            # Ensure latency requirement is met (<100ms)
            if adjustment_latency_ms > 100:
                self.logger.warning("Weight adjustment latency exceeded requirement", 
                                  latency_ms=adjustment_latency_ms,
                                  requirement_ms=100)
            
        except Exception as e:
            self.logger.error("Model weight update failed", error=str(e))
    
    def _get_sentiment_factor(self, model_type: ModelType, fear_greed_regime: str) -> float:
        """
        Get sentiment-based weight adjustment factor based on Fear & Greed Index
        
        Weight Adjustment Rules:
        - Extreme Fear (0-15): LSTM +40%, TimesMixer +30%, Transformers -20%
        - Fear (16-30): LSTM +20%, iTransformer +10%, others balanced
        - Cautious (31-40): LSTM +10%, others slight adjustments
        - Neutral (41-60): Default weights
        - Optimistic (61-70): Transformers +10%, LSTM slight decrease
        - Greed (71-85): Transformers +20%, LSTM -10%
        - Extreme Greed (86-100): Transformers +40%, LSTM -30%
        """
        try:
            sentiment_adjustments = {
                "extreme_fear": {
                    ModelType.LSTM: 1.4,  # +40% - Conservative model preferred
                    ModelType.TRANSFORMER: 0.8,  # -20% - Aggressive models reduced
                    ModelType.ITRANSFORMER: 0.8,  # -20%
                    ModelType.PATCHTST: 0.8,  # -20%
                    ModelType.TIMESMIXER: 1.3,  # +30% - Good at volatility decomposition
                    ModelType.TIMESFM: 0.8  # -20%
                },
                "fear": {
                    ModelType.LSTM: 1.2,  # +20% - Conservative bias
                    ModelType.TRANSFORMER: 1.0,  # Neutral
                    ModelType.ITRANSFORMER: 1.1,  # +10% - Good correlation analysis in fear
                    ModelType.PATCHTST: 1.0,  # Neutral
                    ModelType.TIMESMIXER: 1.0,  # Neutral
                    ModelType.TIMESFM: 1.0  # Neutral
                },
                "cautious": {
                    ModelType.LSTM: 1.1,  # +10% - Slight conservative bias
                    ModelType.TRANSFORMER: 0.95,  # -5%
                    ModelType.ITRANSFORMER: 1.0,  # Neutral
                    ModelType.PATCHTST: 1.0,  # Neutral
                    ModelType.TIMESMIXER: 1.0,  # Neutral
                    ModelType.TIMESFM: 0.95  # -5%
                },
                "neutral": {
                    # Default balanced weights
                    ModelType.LSTM: 1.0,
                    ModelType.TRANSFORMER: 1.0,
                    ModelType.ITRANSFORMER: 1.0,
                    ModelType.PATCHTST: 1.0,
                    ModelType.TIMESMIXER: 1.0,
                    ModelType.TIMESFM: 1.0
                },
                "optimistic": {
                    ModelType.LSTM: 0.95,  # -5% - Slight decrease
                    ModelType.TRANSFORMER: 1.1,  # +10% - Better trend detection
                    ModelType.ITRANSFORMER: 1.1,  # +10%
                    ModelType.PATCHTST: 1.1,  # +10%
                    ModelType.TIMESMIXER: 1.0,  # Neutral
                    ModelType.TIMESFM: 1.1  # +10%
                },
                "greed": {
                    ModelType.LSTM: 0.9,  # -10% - Less conservative
                    ModelType.TRANSFORMER: 1.2,  # +20% - Aggressive models preferred
                    ModelType.ITRANSFORMER: 1.2,  # +20%
                    ModelType.PATCHTST: 1.2,  # +20%
                    ModelType.TIMESMIXER: 1.0,  # Neutral
                    ModelType.TIMESFM: 1.2  # +20%
                },
                "extreme_greed": {
                    ModelType.LSTM: 0.7,  # -30% - Minimize conservative approach
                    ModelType.TRANSFORMER: 1.4,  # +40% - Maximum aggressive weighting
                    ModelType.ITRANSFORMER: 1.4,  # +40%
                    ModelType.PATCHTST: 1.4,  # +40%
                    ModelType.TIMESMIXER: 1.1,  # +10% - Slight increase
                    ModelType.TIMESFM: 1.4  # +40%
                }
            }
            
            return sentiment_adjustments.get(fear_greed_regime, sentiment_adjustments["neutral"]).get(model_type, 1.0)
            
        except Exception as e:
            self.logger.warning("Failed to calculate sentiment factor", error=str(e))
            return 1.0
    
    def _get_regime_factor(self, model_type: ModelType, market_regime: str) -> float:
        """Get regime-specific weighting factor for different model types"""
        try:
            # Define model strengths in different market regimes
            regime_strengths = {
                "bull": {
                    ModelType.LSTM: 0.9,  # LSTM less effective in trending markets
                    ModelType.TRANSFORMER: 1.1,  # Better at capturing trends
                    ModelType.ITRANSFORMER: 1.2,  # Excellent for multivariate trend detection
                    ModelType.PATCHTST: 1.1,  # Good for long-term trends
                    ModelType.TIMESMIXER: 1.0,  # Balanced performance
                    ModelType.TIMESFM: 1.2  # Foundation model excellent for trend detection
                },
                "bear": {
                    ModelType.LSTM: 1.0,  # LSTM handles bear markets reasonably
                    ModelType.TRANSFORMER: 1.1,  # Good at pattern recognition
                    ModelType.ITRANSFORMER: 1.3,  # Best for correlated selloffs
                    ModelType.PATCHTST: 1.0,  # Stable performance
                    ModelType.TIMESMIXER: 1.2,  # Good at decomposing market stress
                    ModelType.TIMESFM: 1.1  # Good generalization in bear markets
                },
                "sideways": {
                    ModelType.LSTM: 1.1,  # LSTM good at range-bound markets
                    ModelType.TRANSFORMER: 1.0,  # Neutral performance
                    ModelType.ITRANSFORMER: 1.0,  # Less advantage in low correlation
                    ModelType.PATCHTST: 0.9,  # Less effective in choppy markets
                    ModelType.TIMESMIXER: 1.2,  # Excellent at noise filtering
                    ModelType.TIMESFM: 1.0  # Balanced performance in sideways markets
                },
                "volatile": {
                    ModelType.LSTM: 0.8,  # LSTM struggles with high volatility
                    ModelType.TRANSFORMER: 1.1,  # Better attention to volatility patterns
                    ModelType.ITRANSFORMER: 1.3,  # Best for volatility clustering
                    ModelType.PATCHTST: 1.0,  # Stable under volatility
                    ModelType.TIMESMIXER: 1.4,  # Excellent volatility decomposition
                    ModelType.TIMESFM: 1.2  # Foundation model handles volatility well
                },
                "normal": {
                    ModelType.LSTM: 1.0,
                    ModelType.TRANSFORMER: 1.0,
                    ModelType.ITRANSFORMER: 1.0,
                    ModelType.PATCHTST: 1.0,
                    ModelType.TIMESMIXER: 1.0,
                    ModelType.TIMESFM: 1.0
                }
            }
            
            return regime_strengths.get(market_regime, regime_strengths["normal"]).get(model_type, 1.0)
            
        except Exception as e:
            self.logger.warning("Failed to calculate regime factor", error=str(e))
            return 1.0
    
    async def _save_model(self, model_type: ModelType):
        """Save a trained model to disk with preservation support"""
        try:
            model = self._models[model_type]
            if not hasattr(model, 'save_model'):
                self.logger.warning("Model does not support saving", model_type=model_type.value)
                return
            
            filepath = self.model_dir / f"{model_type.value}_model.pt"
            success = model.save_model(str(filepath))
            
            if success:
                self.logger.info("Model saved", model_type=model_type.value, filepath=str(filepath))
                
                # Trigger preservation if enabled
                if self._preservation_manager:
                    try:
                        # Serialize model state
                        model_data = self._serialize_model(model_type)
                        
                        # Prepare metadata
                        metadata = {
                            'performance': self._model_performance.get(model_type, {}),
                            'weights': self._model_weights.get(model_type, 1.0),
                            'saved_from': str(filepath),
                            'feature_engineer_version': getattr(self._feature_engineer, 'version', '1.0')
                        }
                        
                        # Save to preservation
                        await self._preservation_manager.save_model(
                            model_data=model_data,
                            model_type=model_type.value,
                            mode=self._current_mode,
                            tags=['model_manager', 'training_checkpoint'],
                            metadata=metadata,
                            priority=PreservationPriority.HIGH if self._current_mode == 'live_trading' else PreservationPriority.NORMAL
                        )
                        
                        self.logger.info("Model preserved", model_type=model_type.value)
                        
                    except Exception as e:
                        # Log error but don't fail the save operation
                        self.logger.error("Model preservation failed", 
                                        model_type=model_type.value, 
                                        error=str(e))
            else:
                self.logger.error("Model save failed", model_type=model_type.value)
                
        except Exception as e:
            self.logger.error("Model save exception", model_type=model_type.value, error=str(e))
    
    def _serialize_model(self, model_type: ModelType) -> bytes:
        """Serialize model to bytes for preservation"""
        model = self._models[model_type]
        
        # Get model state
        if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
            # PyTorch model
            state = model.model.state_dict()
        elif hasattr(model, 'get_state'):
            # Custom state method
            state = model.get_state()
        else:
            # Fallback to entire model object
            state = {
                'model_type': model_type.value,
                'model_class': model.__class__.__name__
            }
        
        # Serialize with pickle
        return pickle.dumps(state)
    
    async def load_models(self, model_types: Optional[List[ModelType]] = None) -> Dict[ModelType, bool]:
        """Load saved models from disk with preservation fallback"""
        if model_types is None:
            model_types = list(self._models.keys())
        
        load_results = {}
        
        for model_type in model_types:
            try:
                model = self._models[model_type]
                if not hasattr(model, 'load_model'):
                    self.logger.warning("Model does not support loading", model_type=model_type.value)
                    load_results[model_type] = False
                    continue
                
                filepath = self.model_dir / f"{model_type.value}_model.pt"
                success = False
                
                # Try local load first
                if filepath.exists():
                    success = model.load_model(str(filepath))
                    if success:
                        self.logger.info("Model loaded from disk", model_type=model_type.value)
                
                # If local load failed, try preservation fallback
                if not success and self._preservation_manager:
                    try:
                        self.logger.info("Attempting preservation fallback", model_type=model_type.value)
                        
                        # Load from preservation
                        model_data, metadata = await self._preservation_manager.load_model(
                            model_type=model_type.value,
                            version=None,  # Get latest version
                            mode=self._current_mode,
                            fallback=True
                        )
                        
                        # Restore model state
                        success = self._restore_model(model_type, model_data)
                        
                        if success:
                            self.logger.info("Model restored from preservation", 
                                           model_type=model_type.value,
                                           version=metadata.get('version'))
                            
                            # Update performance metrics if available
                            if 'performance' in metadata.get('metadata', {}):
                                self._model_performance[model_type] = metadata['metadata']['performance']
                        
                    except Exception as e:
                        self.logger.error("Preservation fallback failed", 
                                        model_type=model_type.value, 
                                        error=str(e))
                
                load_results[model_type] = success
                
                if not success:
                    self.logger.error("Model load failed", model_type=model_type.value)
                    
            except Exception as e:
                self.logger.error("Model load exception", 
                                model_type=model_type.value, 
                                error=str(e))
                load_results[model_type] = False
        
        return load_results
    
    def _restore_model(self, model_type: ModelType, model_data: bytes) -> bool:
        """Restore model from preserved data"""
        try:
            model = self._models[model_type]
            
            # Deserialize state
            state = pickle.loads(model_data)
            
            # Restore based on model type
            if hasattr(model, 'model') and hasattr(model.model, 'load_state_dict'):
                # PyTorch model
                model.model.load_state_dict(state)
                if hasattr(model, 'is_trained'):
                    model.is_trained = True
            elif hasattr(model, 'load_state'):
                # Custom load method
                model.load_state(state)
            else:
                self.logger.warning("Model does not support state restoration", 
                                  model_type=model_type.value)
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Model restoration failed", 
                            model_type=model_type.value, 
                            error=str(e))
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all models and return status"""
        try:
            status = {
                'overall_healthy': True,
                'models': {},
                'ensemble_available': False,
                'cache_size': len(self._ensemble_cache),
                'model_weights': self._model_weights.copy(),
                'performance': self._model_performance.copy()
            }
            
            healthy_models = 0
            
            for model_type, model in self._models.items():
                model_healthy = await model.health_check()
                status['models'][model_type.value] = {
                    'healthy': model_healthy,
                    'trained': model.is_model_trained(),
                    'weight': self._model_weights[model_type]
                }
                
                if model_healthy:
                    healthy_models += 1
            
            status['overall_healthy'] = healthy_models > 0
            status['ensemble_available'] = healthy_models > 1
            
            return status
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {'overall_healthy': False, 'error': str(e)}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get detailed model performance metrics"""
        return {
            'weights': self._model_weights.copy(),
            'performance': self._model_performance.copy(),
            'cache_stats': {
                'size': len(self._ensemble_cache),
                'ttl_minutes': self._cache_ttl_minutes
            }
        }
    
    def clear_cache(self):
        """Clear prediction cache"""
        self._ensemble_cache.clear()
        self.logger.info("Prediction cache cleared")
    
    def set_mode(self, mode: str):
        """Set the current operational mode"""
        self._current_mode = mode
        self.logger.info("Operational mode changed", mode=mode)
    
    async def get_fear_greed_regime(self) -> str:
        """
        Get current Fear & Greed Index and classify into sentiment regimes
        
        Returns:
            str: One of 'extreme_fear', 'fear', 'cautious', 'neutral', 'optimistic', 'greed', 'extreme_greed'
        """
        try:
            # Check if we have recent cached sentiment data
            current_time = datetime.now()
            if (self._last_sentiment_data and 
                (current_time - self._last_sentiment_data.timestamp).total_seconds() < self._sentiment_cache_ttl_minutes * 60):
                sentiment_data = self._last_sentiment_data
            else:
                # Fetch fresh sentiment data
                sentiment_data = await self._fear_greed_client.get_market_data()
                self._last_sentiment_data = sentiment_data
            
            fear_greed_value = sentiment_data.fear_greed_index
            
            # Classify into 7 refined sentiment regimes
            if fear_greed_value <= 15:
                return "extreme_fear"
            elif fear_greed_value <= 30:
                return "fear"
            elif fear_greed_value <= 40:
                return "cautious"
            elif fear_greed_value <= 60:
                return "neutral"
            elif fear_greed_value <= 70:
                return "optimistic"
            elif fear_greed_value <= 85:
                return "greed"
            else:
                return "extreme_greed"
                
        except Exception as e:
            self.logger.error("Failed to get fear/greed regime", error=str(e))
            # Return neutral as fallback
            return "neutral"