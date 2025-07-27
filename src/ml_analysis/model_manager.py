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

from src.discovery.base import DiscoveredToken
from .base import (
    MLAnalyzerBase, PredictionResult, ModelType, PredictionDirection,
    MLAnalysisError
)
from .lstm_model import LSTMPricePredictor
from .feature_engineer import FeatureEngineer
from src.logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    performance_tracker
)


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
        
        # Model file paths
        self.model_dir = Path(self.config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available ML models"""
        try:
            # Log model initialization start
            asyncio.create_task(activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.START,
                source="model_manager",
                event_type="model_initialization",
                title="Initializing ML models",
                severity=ActivitySeverity.INFO,
                metadata={"model_dir": str(self.model_dir)}
            ))
            
            # Initialize LSTM model
            lstm_config = self.config.get('lstm', {})
            self._models[ModelType.LSTM] = LSTMPricePredictor(lstm_config)
            self._model_weights[ModelType.LSTM] = 1.0
            
            # Initialize performance tracking
            for model_type in self._models.keys():
                self._model_performance[model_type] = {
                    'accuracy': 0.0,
                    'predictions_made': 0,
                    'last_updated': datetime.now().timestamp()
                }
            
            # Log successful model initialization
            asyncio.create_task(activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.SUCCESS,
                source="model_manager",
                event_type="models_initialized",
                title="ML models initialized successfully",
                severity=ActivitySeverity.INFO,
                metadata={
                    "initialized_models": [model_type.value for model_type in self._models.keys()],
                    "model_count": len(self._models),
                    "model_dir": str(self.model_dir)
                }
            ))
            
            self.logger.info("Models initialized", 
                           models=list(self._models.keys()),
                           model_dir=str(self.model_dir))
            
        except Exception as e:
            # Log model initialization failure
            asyncio.create_task(activity_logger.log_error(
                category=ActivityCategory.ML_RL,
                source="model_manager",
                event_type="model_initialization_failed",
                title="Failed to initialize ML models",
                error_message=str(e),
                exception=e,
                severity=ActivitySeverity.ERROR,
                metadata={"model_dir": str(self.model_dir)}
            ))
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
                if use_ensemble and len(self._models) > 1:
                    result = await self._ensemble_prediction(token, historical_data)
                else:
                    # Use best performing single model
                    best_model = self._get_best_model()
                    result = await best_model.analyze_token(token, historical_data)
                
                # Cache the result
                self._ensemble_cache[cache_key] = (datetime.now(), result)
                
                # Update model performance tracking
                await self._update_performance_tracking(result)
                
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
        """Combine multiple predictions into ensemble result"""
        try:
            # Weighted averages
            total_weight = sum(self._model_weights[mt] * pred.confidence 
                             for mt, pred in predictions.items())
            
            if total_weight == 0:
                total_weight = 1.0  # Prevent division by zero
            
            # Calculate weighted predictions
            ensemble_predictions = {
                '1h': 0.0, '4h': 0.0, '24h': 0.0
            }
            ensemble_confidence = 0.0
            ensemble_prob_up = 0.0
            
            for model_type, prediction in predictions.items():
                model_weight = self._model_weights[model_type] * prediction.confidence / total_weight
                
                if prediction.price_prediction_1h:
                    ensemble_predictions['1h'] += prediction.price_prediction_1h * model_weight
                if prediction.price_prediction_4h:
                    ensemble_predictions['4h'] += prediction.price_prediction_4h * model_weight
                if prediction.price_prediction_24h:
                    ensemble_predictions['24h'] += prediction.price_prediction_24h * model_weight
                
                ensemble_confidence += prediction.confidence * model_weight
                ensemble_prob_up += prediction.probability_up * model_weight
            
            # Determine ensemble direction
            current_price = token.price_usd or 1.0
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
            
            # Create ensemble result
            ensemble_result = PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.ENSEMBLE,
                price_prediction_1h=ensemble_predictions['1h'] if ensemble_predictions['1h'] > 0 else None,
                price_prediction_4h=ensemble_predictions['4h'] if ensemble_predictions['4h'] > 0 else None,
                price_prediction_24h=ensemble_predictions['24h'] if ensemble_predictions['24h'] > 0 else None,
                direction=direction,
                confidence=min(ensemble_confidence, 1.0),
                probability_up=min(ensemble_prob_up, 1.0),
                technical_indicators=best_prediction.technical_indicators,
                market_features=best_prediction.market_features,
                model_accuracy=sum(p.model_accuracy or 0 for p in predictions.values()) / len(predictions),
                features_used=[f"ensemble_{len(predictions)}_models"],
                model_version="ensemble_1.0",
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
    
    async def _update_performance_tracking(self, result: PredictionResult):
        """Update model performance metrics"""
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
            
            self._model_performance[model_type]['last_updated'] = datetime.now().timestamp()
            
        except Exception as e:
            self.logger.error("Performance tracking update failed", error=str(e))
    
    async def _update_model_weights(self):
        """Update model weights based on performance"""
        try:
            for model_type, performance in self._model_performance.items():
                accuracy = performance['accuracy']
                prediction_count = performance['predictions_made']
                
                # Weight based on accuracy and experience
                experience_factor = min(prediction_count / 100, 1.0)  # Max experience at 100 predictions
                
                # Base weight on accuracy, boosted by experience
                new_weight = accuracy * (0.5 + 0.5 * experience_factor)
                self._model_weights[model_type] = max(new_weight, 0.1)  # Minimum weight
            
            # Normalize weights
            total_weight = sum(self._model_weights.values())
            if total_weight > 0:
                for model_type in self._model_weights:
                    self._model_weights[model_type] /= total_weight
            
            self.logger.info("Model weights updated", weights=self._model_weights)
            
        except Exception as e:
            self.logger.error("Model weight update failed", error=str(e))
    
    async def _save_model(self, model_type: ModelType):
        """Save a trained model to disk"""
        try:
            model = self._models[model_type]
            if not hasattr(model, 'save_model'):
                self.logger.warning("Model does not support saving", model_type=model_type.value)
                return
            
            filepath = self.model_dir / f"{model_type.value}_model.pt"
            success = model.save_model(str(filepath))
            
            if success:
                self.logger.info("Model saved", model_type=model_type.value, filepath=str(filepath))
            else:
                self.logger.error("Model save failed", model_type=model_type.value)
                
        except Exception as e:
            self.logger.error("Model save exception", model_type=model_type.value, error=str(e))
    
    async def load_models(self, model_types: Optional[List[ModelType]] = None) -> Dict[ModelType, bool]:
        """Load saved models from disk"""
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
                if not filepath.exists():
                    self.logger.info("Model file not found", model_type=model_type.value)
                    load_results[model_type] = False
                    continue
                
                success = model.load_model(str(filepath))
                load_results[model_type] = success
                
                if success:
                    self.logger.info("Model loaded", model_type=model_type.value)
                else:
                    self.logger.error("Model load failed", model_type=model_type.value)
                    
            except Exception as e:
                self.logger.error("Model load exception", 
                                model_type=model_type.value, 
                                error=str(e))
                load_results[model_type] = False
        
        return load_results
    
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