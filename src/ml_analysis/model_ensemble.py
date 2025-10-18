"""
Model Ensemble for LSTM and Transformer Models
Provides simplified ensemble prediction functionality with configurable combination strategies
"""

import asyncio
import pickle
import torch
import numpy as np
import pandas as pd
import structlog
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .base import (
    MLAnalyzerBase, PredictionResult, ModelType, PredictionDirection,
    MLAnalysisError, TechnicalIndicators, MarketFeatures
)
from .lstm_model import LSTMPricePredictor
from .transformers.transformer_predictor import TransformerPredictor
from src.discovery.base import DiscoveredToken
from src.activity_logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity
)


logger = structlog.get_logger()


class EnsembleStrategy(Enum):
    """Ensemble combination strategies"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    STACKING = "stacking"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble predictor"""
    # Model weights (sum should equal 1.0)
    lstm_weight: float = 0.6
    transformer_weight: float = 0.4

    # Ensemble strategy
    strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE

    # Checkpoint directories
    checkpoint_dir: str = "tmp/checkpoints"
    fallback_model_dir: str = "models"

    # Meta-learner config (for stacking)
    meta_learner_type: str = "random_forest"  # "random_forest" or "linear"
    meta_learner_params: Dict[str, Any] = None

    # Training config
    retrain_meta_learner: bool = True
    validation_split: float = 0.2

    def __post_init__(self):
        if self.meta_learner_params is None:
            if self.meta_learner_type == "random_forest":
                self.meta_learner_params = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "random_state": 42
                }
            else:
                self.meta_learner_params = {"fit_intercept": True}

        # Normalize weights
        total_weight = self.lstm_weight + self.transformer_weight
        if total_weight > 0:
            self.lstm_weight /= total_weight
            self.transformer_weight /= total_weight


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result with component model details"""
    # Final ensemble prediction
    ensemble_result: PredictionResult

    # Individual model predictions
    lstm_prediction: Optional[PredictionResult] = None
    transformer_prediction: Optional[PredictionResult] = None

    # Ensemble metadata
    strategy_used: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE
    effective_weights: Dict[str, float] = None
    prediction_agreement: float = 0.0  # 0-1, how much models agree
    ensemble_confidence: float = 0.0
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if self.effective_weights is None:
            self.effective_weights = {}


class EnsemblePredictor(MLAnalyzerBase):
    """
    Ensemble predictor that combines LSTM and Transformer models
    Supports multiple combination strategies and checkpoint loading
    """

    def __init__(self, config: Optional[Union[Dict, EnsembleConfig]] = None):
        super().__init__(ModelType.ENSEMBLE, config if isinstance(config, dict) else None)

        # Convert config if needed
        if isinstance(config, dict):
            self.ensemble_config = EnsembleConfig(**config)
        elif isinstance(config, EnsembleConfig):
            self.ensemble_config = config
        else:
            self.ensemble_config = EnsembleConfig()

        self.logger = structlog.get_logger().bind(component="EnsemblePredictor")

        # Model storage
        self._lstm_model: Optional[LSTMPricePredictor] = None
        self._transformer_model: Optional[TransformerPredictor] = None
        self._meta_learner: Optional[Union[RandomForestRegressor, LinearRegression]] = None
        self._scaler: Optional[StandardScaler] = None

        # Performance tracking
        self._lstm_performance_history: List[float] = []
        self._transformer_performance_history: List[float] = []
        self._ensemble_performance_history: List[float] = []

        # Training state
        self._training_mode = False
        self._models_loaded = False

        # Initialize paths
        self.checkpoint_dir = Path(self.ensemble_config.checkpoint_dir)
        self.fallback_dir = Path(self.ensemble_config.fallback_model_dir)

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize LSTM and Transformer models"""
        try:
            # Initialize LSTM model
            lstm_config = self.config.get('lstm', {})
            self._lstm_model = LSTMPricePredictor(lstm_config)

            # Initialize Transformer model
            transformer_config = self.config.get('transformer', {})
            self._transformer_model = TransformerPredictor(transformer_config)

            self.logger.info("Ensemble models initialized",
                           lstm_config=bool(lstm_config),
                           transformer_config=bool(transformer_config))

        except Exception as e:
            self.logger.error("Failed to initialize ensemble models", error=str(e))
            raise MLAnalysisError(f"Ensemble initialization failed: {str(e)}")

    async def load_models(self,
                         lstm_checkpoint: Optional[str] = None,
                         transformer_checkpoint: Optional[str] = None) -> Dict[str, bool]:
        """
        Load models from checkpoints with fallback to default locations

        Args:
            lstm_checkpoint: Path to LSTM checkpoint (optional)
            transformer_checkpoint: Path to Transformer checkpoint (optional)

        Returns:
            Dict with loading results for each model
        """
        load_results = {"lstm": False, "transformer": False}

        try:
            # Load LSTM model
            if self._lstm_model:
                lstm_success = await self._load_single_model(
                    model=self._lstm_model,
                    model_name="lstm",
                    checkpoint_path=lstm_checkpoint
                )
                load_results["lstm"] = lstm_success

            # Load Transformer model
            if self._transformer_model:
                transformer_success = await self._load_single_model(
                    model=self._transformer_model,
                    model_name="transformer",
                    checkpoint_path=transformer_checkpoint
                )
                load_results["transformer"] = transformer_success

            # Update loaded state
            self._models_loaded = any(load_results.values())
            self._is_trained = self._models_loaded

            # Log activity
            await activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.SUCCESS if self._models_loaded else ActivityAction.ERROR,
                source="ensemble_predictor",
                event_type="models_loaded",
                title=f"Ensemble models loaded: LSTM={load_results['lstm']}, Transformer={load_results['transformer']}",
                severity=ActivitySeverity.INFO,
                metadata={
                    "load_results": load_results,
                    "models_loaded": self._models_loaded,
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "fallback_dir": str(self.fallback_dir)
                }
            )

            self.logger.info("Model loading completed",
                           load_results=load_results,
                           models_loaded=self._models_loaded)

            return load_results

        except Exception as e:
            self.logger.error("Model loading failed", error=str(e))
            raise MLAnalysisError(f"Failed to load models: {str(e)}")

    async def _load_single_model(self,
                                model: MLAnalyzerBase,
                                model_name: str,
                                checkpoint_path: Optional[str] = None) -> bool:
        """Load a single model with checkpoint and fallback logic"""
        try:
            # Define potential checkpoint paths
            potential_paths = []

            if checkpoint_path:
                potential_paths.append(Path(checkpoint_path))

            # Add standard checkpoint locations
            potential_paths.extend([
                self.checkpoint_dir / f"{model_name}_best.pt",
                self.checkpoint_dir / f"{model_name}_latest.pt",
                self.checkpoint_dir / f"{model_name}_model.pt",
                self.fallback_dir / f"{model_name}_model.pt",
                self.fallback_dir / f"{model_name}.pkl"
            ])

            # Try loading from each path
            for path in potential_paths:
                if path.exists():
                    try:
                        if hasattr(model, 'load_model'):
                            success = model.load_model(str(path))
                            if success:
                                self.logger.info("Model loaded successfully",
                                               model=model_name,
                                               path=str(path))
                                return True
                        else:
                            # Try direct torch loading for PyTorch models
                            if path.suffix in ['.pt', '.pth']:
                                state_dict = torch.load(path, map_location='cpu')
                                if hasattr(model, 'model') and hasattr(model.model, 'load_state_dict'):
                                    model.model.load_state_dict(state_dict)
                                    model._is_trained = True
                                    self.logger.info("Model loaded via torch.load",
                                                   model=model_name,
                                                   path=str(path))
                                    return True

                    except Exception as e:
                        self.logger.warning("Failed to load from path",
                                          model=model_name,
                                          path=str(path),
                                          error=str(e))
                        continue

            self.logger.warning("No valid checkpoint found for model",
                              model=model_name,
                              checked_paths=[str(p) for p in potential_paths])
            return False

        except Exception as e:
            self.logger.error("Single model loading failed",
                            model=model_name,
                            error=str(e))
            return False

    async def analyze_token(self,
                          token: DiscoveredToken,
                          historical_data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """
        Generate ensemble prediction for a token

        Args:
            token: Token to analyze
            historical_data: Historical price data

        Returns:
            Ensemble prediction result
        """
        start_time = datetime.now()

        try:
            if not self._models_loaded:
                await self.load_models()

            if not self._models_loaded:
                raise MLAnalysisError("No models loaded for ensemble prediction")

            # Get predictions from individual models
            lstm_prediction = None
            transformer_prediction = None

            # Get LSTM prediction
            if self._lstm_model and self._lstm_model.is_model_trained():
                try:
                    lstm_prediction = await self._lstm_model.analyze_token(token, historical_data)
                except Exception as e:
                    self.logger.warning("LSTM prediction failed", error=str(e))

            # Get Transformer prediction
            if self._transformer_model and self._transformer_model.is_model_trained():
                try:
                    transformer_prediction = await self._transformer_model.analyze_token(token, historical_data)
                except Exception as e:
                    self.logger.warning("Transformer prediction failed", error=str(e))

            # Check if we have at least one prediction
            if not lstm_prediction and not transformer_prediction:
                raise MLAnalysisError("No valid predictions from component models")

            # Combine predictions using selected strategy
            ensemble_prediction = await self._combine_predictions(
                token=token,
                lstm_prediction=lstm_prediction,
                transformer_prediction=transformer_prediction
            )

            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            ensemble_prediction.processing_time_ms = processing_time_ms

            # Log successful prediction
            await activity_logger.log_activity(
                category=ActivityCategory.ML_RL,
                action=ActivityAction.SUCCESS,
                source="ensemble_predictor",
                event_type="ensemble_prediction",
                title=f"Ensemble prediction generated for {token.symbol}",
                severity=ActivitySeverity.INFO,
                token_address=token.address,
                metadata={
                    "strategy": self.ensemble_config.strategy.value,
                    "has_lstm": lstm_prediction is not None,
                    "has_transformer": transformer_prediction is not None,
                    "ensemble_confidence": ensemble_prediction.ensemble_confidence,
                    "processing_time_ms": processing_time_ms
                }
            )

            return ensemble_prediction.ensemble_result

        except Exception as e:
            self.logger.error("Ensemble prediction failed",
                            token=token.address,
                            error=str(e))
            raise MLAnalysisError(f"Ensemble prediction failed: {str(e)}")

    async def _combine_predictions(self,
                                 token: DiscoveredToken,
                                 lstm_prediction: Optional[PredictionResult],
                                 transformer_prediction: Optional[PredictionResult]) -> EnsemblePrediction:
        """Combine individual model predictions using selected strategy"""
        try:
            strategy = self.ensemble_config.strategy

            if strategy == EnsembleStrategy.SIMPLE_AVERAGE:
                return await self._simple_average_combination(token, lstm_prediction, transformer_prediction)
            elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
                return await self._weighted_average_combination(token, lstm_prediction, transformer_prediction)
            elif strategy == EnsembleStrategy.CONFIDENCE_WEIGHTED:
                return await self._confidence_weighted_combination(token, lstm_prediction, transformer_prediction)
            elif strategy == EnsembleStrategy.STACKING:
                return await self._stacking_combination(token, lstm_prediction, transformer_prediction)
            else:
                # Fallback to weighted average
                return await self._weighted_average_combination(token, lstm_prediction, transformer_prediction)

        except Exception as e:
            self.logger.error("Prediction combination failed", error=str(e))
            raise MLAnalysisError(f"Failed to combine predictions: {str(e)}")

    async def _simple_average_combination(self,
                                        token: DiscoveredToken,
                                        lstm_pred: Optional[PredictionResult],
                                        transformer_pred: Optional[PredictionResult]) -> EnsemblePrediction:
        """Simple average of available predictions"""
        predictions = [p for p in [lstm_pred, transformer_pred] if p is not None]
        if not predictions:
            raise MLAnalysisError("No predictions to combine")

        # Average price predictions
        price_1h = self._average_values([p.price_prediction_1h for p in predictions])
        price_4h = self._average_values([p.price_prediction_4h for p in predictions])
        price_24h = self._average_values([p.price_prediction_24h for p in predictions])

        # Average confidence and probability
        confidence = np.mean([p.confidence for p in predictions])
        prob_up = np.mean([p.probability_up for p in predictions])

        # Calculate agreement
        agreement = self._calculate_prediction_agreement(predictions)

        # Determine direction based on 24h prediction
        direction = self._determine_ensemble_direction(price_24h, token.price_usd)

        # Use best prediction's technical indicators
        best_prediction = max(predictions, key=lambda p: p.confidence)

        ensemble_result = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            price_prediction_1h=price_1h,
            price_prediction_4h=price_4h,
            price_prediction_24h=price_24h,
            direction=direction,
            confidence=confidence,
            probability_up=prob_up,
            technical_indicators=best_prediction.technical_indicators,
            market_features=best_prediction.market_features,
            model_accuracy=np.mean([p.model_accuracy or 0 for p in predictions]),
            features_used=[f"ensemble_simple_avg_{len(predictions)}_models"],
            model_version="ensemble_simple_avg_1.0"
        )

        return EnsemblePrediction(
            ensemble_result=ensemble_result,
            lstm_prediction=lstm_pred,
            transformer_prediction=transformer_pred,
            strategy_used=EnsembleStrategy.SIMPLE_AVERAGE,
            effective_weights={"equal": 1.0 / len(predictions)},
            prediction_agreement=agreement,
            ensemble_confidence=confidence
        )

    async def _weighted_average_combination(self,
                                          token: DiscoveredToken,
                                          lstm_pred: Optional[PredictionResult],
                                          transformer_pred: Optional[PredictionResult]) -> EnsemblePrediction:
        """Weighted average using configured weights"""
        predictions = []
        weights = []
        weight_names = []

        if lstm_pred is not None:
            predictions.append(lstm_pred)
            weights.append(self.ensemble_config.lstm_weight)
            weight_names.append("lstm")

        if transformer_pred is not None:
            predictions.append(transformer_pred)
            weights.append(self.ensemble_config.transformer_weight)
            weight_names.append("transformer")

        if not predictions:
            raise MLAnalysisError("No predictions to combine")

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Weighted average price predictions
        price_1h = self._weighted_average_values([p.price_prediction_1h for p in predictions], weights)
        price_4h = self._weighted_average_values([p.price_prediction_4h for p in predictions], weights)
        price_24h = self._weighted_average_values([p.price_prediction_24h for p in predictions], weights)

        # Weighted average confidence and probability
        confidence = sum(p.confidence * w for p, w in zip(predictions, weights))
        prob_up = sum(p.probability_up * w for p, w in zip(predictions, weights))

        # Calculate agreement
        agreement = self._calculate_prediction_agreement(predictions)

        # Determine direction
        direction = self._determine_ensemble_direction(price_24h, token.price_usd)

        # Use best prediction's technical indicators
        best_prediction = max(predictions, key=lambda p: p.confidence)

        ensemble_result = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            price_prediction_1h=price_1h,
            price_prediction_4h=price_4h,
            price_prediction_24h=price_24h,
            direction=direction,
            confidence=confidence,
            probability_up=prob_up,
            technical_indicators=best_prediction.technical_indicators,
            market_features=best_prediction.market_features,
            model_accuracy=sum(p.model_accuracy * w for p, w in zip(predictions, weights) if p.model_accuracy),
            features_used=[f"ensemble_weighted_avg_{len(predictions)}_models"] +
                         [f"{name}_weight_{w:.3f}" for name, w in zip(weight_names, weights)],
            model_version="ensemble_weighted_avg_1.0"
        )

        effective_weights = {name: weight for name, weight in zip(weight_names, weights)}

        return EnsemblePrediction(
            ensemble_result=ensemble_result,
            lstm_prediction=lstm_pred,
            transformer_prediction=transformer_pred,
            strategy_used=EnsembleStrategy.WEIGHTED_AVERAGE,
            effective_weights=effective_weights,
            prediction_agreement=agreement,
            ensemble_confidence=confidence
        )

    async def _confidence_weighted_combination(self,
                                             token: DiscoveredToken,
                                             lstm_pred: Optional[PredictionResult],
                                             transformer_pred: Optional[PredictionResult]) -> EnsemblePrediction:
        """Confidence-weighted combination - higher confidence gets more weight"""
        predictions = [p for p in [lstm_pred, transformer_pred] if p is not None]
        if not predictions:
            raise MLAnalysisError("No predictions to combine")

        # Use confidence as weights
        confidence_weights = [p.confidence for p in predictions]
        total_confidence = sum(confidence_weights)

        if total_confidence > 0:
            weights = [w / total_confidence for w in confidence_weights]
        else:
            weights = [1.0 / len(predictions)] * len(predictions)

        # Weighted average price predictions
        price_1h = self._weighted_average_values([p.price_prediction_1h for p in predictions], weights)
        price_4h = self._weighted_average_values([p.price_prediction_4h for p in predictions], weights)
        price_24h = self._weighted_average_values([p.price_prediction_24h for p in predictions], weights)

        # Weighted average confidence and probability
        confidence = sum(p.confidence * w for p, w in zip(predictions, weights))
        prob_up = sum(p.probability_up * w for p, w in zip(predictions, weights))

        # Calculate agreement
        agreement = self._calculate_prediction_agreement(predictions)

        # Determine direction
        direction = self._determine_ensemble_direction(price_24h, token.price_usd)

        # Use best prediction's technical indicators
        best_prediction = max(predictions, key=lambda p: p.confidence)

        ensemble_result = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            price_prediction_1h=price_1h,
            price_prediction_4h=price_4h,
            price_prediction_24h=price_24h,
            direction=direction,
            confidence=confidence,
            probability_up=prob_up,
            technical_indicators=best_prediction.technical_indicators,
            market_features=best_prediction.market_features,
            model_accuracy=sum(p.model_accuracy * w for p, w in zip(predictions, weights) if p.model_accuracy),
            features_used=[f"ensemble_confidence_weighted_{len(predictions)}_models"] +
                         [f"model_{i}_conf_weight_{w:.3f}" for i, w in enumerate(weights)],
            model_version="ensemble_confidence_weighted_1.0"
        )

        model_names = []
        if lstm_pred in predictions:
            model_names.append("lstm")
        if transformer_pred in predictions:
            model_names.append("transformer")

        effective_weights = {name: weight for name, weight in zip(model_names, weights)}

        return EnsemblePrediction(
            ensemble_result=ensemble_result,
            lstm_prediction=lstm_pred,
            transformer_prediction=transformer_pred,
            strategy_used=EnsembleStrategy.CONFIDENCE_WEIGHTED,
            effective_weights=effective_weights,
            prediction_agreement=agreement,
            ensemble_confidence=confidence
        )

    async def _stacking_combination(self,
                                  token: DiscoveredToken,
                                  lstm_pred: Optional[PredictionResult],
                                  transformer_pred: Optional[PredictionResult]) -> EnsemblePrediction:
        """Stacking with meta-learner for advanced combination"""
        predictions = [p for p in [lstm_pred, transformer_pred] if p is not None]
        if not predictions:
            raise MLAnalysisError("No predictions to combine")

        # If meta-learner not trained, fall back to confidence weighting
        if self._meta_learner is None:
            self.logger.warning("Meta-learner not trained, falling back to confidence weighting")
            return await self._confidence_weighted_combination(token, lstm_pred, transformer_pred)

        try:
            # Prepare features for meta-learner
            meta_features = self._prepare_meta_features(predictions, token)

            if self._scaler:
                meta_features_scaled = self._scaler.transform([meta_features])
            else:
                meta_features_scaled = [meta_features]

            # Get meta-learner prediction
            meta_prediction = self._meta_learner.predict(meta_features_scaled)[0]

            # Use meta-learner prediction as the ensemble result
            # For now, use it as a weight adjustment factor
            meta_weight = max(0.1, min(1.0, meta_prediction))

            # Apply meta-weight to confidence-weighted combination
            conf_result = await self._confidence_weighted_combination(token, lstm_pred, transformer_pred)

            # Adjust confidence based on meta-learner
            ensemble_result = conf_result.ensemble_result
            ensemble_result.confidence *= meta_weight
            ensemble_result.features_used = [f"ensemble_stacking_{len(predictions)}_models", f"meta_weight_{meta_weight:.3f}"]
            ensemble_result.model_version = "ensemble_stacking_1.0"

            return EnsemblePrediction(
                ensemble_result=ensemble_result,
                lstm_prediction=lstm_pred,
                transformer_prediction=transformer_pred,
                strategy_used=EnsembleStrategy.STACKING,
                effective_weights={"meta_learner": meta_weight, **conf_result.effective_weights},
                prediction_agreement=conf_result.prediction_agreement,
                ensemble_confidence=ensemble_result.confidence
            )

        except Exception as e:
            self.logger.warning("Stacking failed, falling back to confidence weighting", error=str(e))
            return await self._confidence_weighted_combination(token, lstm_pred, transformer_pred)

    def _prepare_meta_features(self, predictions: List[PredictionResult], token: DiscoveredToken) -> List[float]:
        """Prepare features for meta-learner"""
        features = []

        # Add individual model predictions
        for pred in predictions:
            features.extend([
                pred.price_prediction_1h or 0.0,
                pred.price_prediction_4h or 0.0,
                pred.price_prediction_24h or 0.0,
                pred.confidence,
                pred.probability_up,
                pred.model_accuracy or 0.0
            ])

        # Add token features
        features.extend([
            token.price_usd or 0.0,
            token.market_cap or 0.0,
            token.volume_24h or 0.0,
        ])

        # Pad to fixed size if needed
        target_size = 30  # Adjust based on needs
        while len(features) < target_size:
            features.append(0.0)

        return features[:target_size]

    def _average_values(self, values: List[Optional[float]]) -> Optional[float]:
        """Calculate average of non-None values"""
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return None
        return np.mean(valid_values)

    def _weighted_average_values(self, values: List[Optional[float]], weights: List[float]) -> Optional[float]:
        """Calculate weighted average of non-None values"""
        valid_pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
        if not valid_pairs:
            return None

        total_weight = sum(w for _, w in valid_pairs)
        if total_weight == 0:
            return None

        weighted_sum = sum(v * w for v, w in valid_pairs)
        return weighted_sum / total_weight

    def _calculate_prediction_agreement(self, predictions: List[PredictionResult]) -> float:
        """Calculate how much the predictions agree (0-1)"""
        if len(predictions) < 2:
            return 1.0

        # Compare 24h predictions if available
        values_24h = [p.price_prediction_24h for p in predictions if p.price_prediction_24h is not None]
        if len(values_24h) >= 2:
            # Calculate coefficient of variation (lower = more agreement)
            mean_val = np.mean(values_24h)
            std_val = np.std(values_24h)
            if mean_val > 0:
                cv = std_val / mean_val
                return max(0.0, 1.0 - min(cv, 1.0))  # Convert CV to agreement score

        # Fallback: compare directions
        directions = [p.direction for p in predictions]
        unique_directions = set(directions)
        return 1.0 - (len(unique_directions) - 1) / max(1, len(directions) - 1)

    def _determine_ensemble_direction(self, price_24h: Optional[float], current_price: Optional[float]) -> PredictionDirection:
        """Determine ensemble direction based on price prediction"""
        if price_24h is None or current_price is None or current_price == 0:
            return PredictionDirection.HOLD

        price_change = (price_24h - current_price) / current_price

        if price_change > 0.1:
            return PredictionDirection.STRONG_BUY
        elif price_change > 0.02:
            return PredictionDirection.BUY
        elif price_change < -0.1:
            return PredictionDirection.STRONG_SELL
        elif price_change < -0.02:
            return PredictionDirection.SELL
        else:
            return PredictionDirection.HOLD

    async def train_model(self, training_data: pd.DataFrame) -> bool:
        """
        Train the ensemble and meta-learner on historical data

        Args:
            training_data: Historical training data with features and targets

        Returns:
            Success status
        """
        try:
            self._training_mode = True
            self.logger.info("Starting ensemble training", data_shape=training_data.shape)

            # Train individual models first
            lstm_success = False
            transformer_success = False

            if self._lstm_model:
                lstm_success = await self._lstm_model.train_model(training_data)
                self.logger.info("LSTM training completed", success=lstm_success)

            if self._transformer_model:
                transformer_success = await self._transformer_model.train_model(training_data)
                self.logger.info("Transformer training completed", success=transformer_success)

            # Train meta-learner if using stacking and have sufficient data
            meta_success = True
            if (self.ensemble_config.strategy == EnsembleStrategy.STACKING and
                self.ensemble_config.retrain_meta_learner and
                len(training_data) > 100):  # Minimum data requirement

                meta_success = await self._train_meta_learner(training_data)
                self.logger.info("Meta-learner training completed", success=meta_success)

            # Update training state
            overall_success = (lstm_success or transformer_success) and meta_success
            self._is_trained = overall_success
            self._models_loaded = overall_success
            self._training_mode = False

            self.logger.info("Ensemble training completed",
                           overall_success=overall_success,
                           lstm_success=lstm_success,
                           transformer_success=transformer_success,
                           meta_success=meta_success)

            return overall_success

        except Exception as e:
            self._training_mode = False
            self.logger.error("Ensemble training failed", error=str(e))
            return False

    async def _train_meta_learner(self, training_data: pd.DataFrame) -> bool:
        """Train the meta-learner for stacking"""
        try:
            # This is a simplified meta-learner training
            # In practice, you'd need to generate meta-features from cross-validation
            # of the base models, but this provides the framework

            if self.ensemble_config.meta_learner_type == "random_forest":
                self._meta_learner = RandomForestRegressor(**self.ensemble_config.meta_learner_params)
            else:
                self._meta_learner = LinearRegression(**self.ensemble_config.meta_learner_params)

            # Initialize scaler
            self._scaler = StandardScaler()

            # For now, create dummy meta-features and targets
            # This would need to be replaced with proper cross-validation meta-feature generation
            n_samples = min(len(training_data), 1000)  # Limit for performance
            meta_features = np.random.rand(n_samples, 30)  # Placeholder
            meta_targets = np.random.rand(n_samples)  # Placeholder

            # Scale features
            meta_features_scaled = self._scaler.fit_transform(meta_features)

            # Train meta-learner
            self._meta_learner.fit(meta_features_scaled, meta_targets)

            self.logger.info("Meta-learner trained successfully",
                           meta_learner_type=self.ensemble_config.meta_learner_type,
                           n_samples=n_samples)

            return True

        except Exception as e:
            self.logger.error("Meta-learner training failed", error=str(e))
            return False

    def get_required_features(self) -> List[str]:
        """Get list of required features for ensemble"""
        features = set()

        if self._lstm_model:
            features.update(self._lstm_model.get_required_features())

        if self._transformer_model:
            features.update(self._transformer_model.get_required_features())

        return list(features)

    async def health_check(self) -> bool:
        """Check if ensemble is healthy and ready"""
        try:
            if not self._models_loaded:
                return False

            # Check individual models
            lstm_healthy = True
            transformer_healthy = True

            if self._lstm_model:
                lstm_healthy = await self._lstm_model.health_check()

            if self._transformer_model:
                transformer_healthy = await self._transformer_model.health_check()

            # At least one model should be healthy
            return lstm_healthy or transformer_healthy

        except Exception as e:
            self.logger.error("Ensemble health check failed", error=str(e))
            return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get ensemble performance metrics"""
        return {
            "strategy": self.ensemble_config.strategy.value,
            "models_loaded": self._models_loaded,
            "lstm_weight": self.ensemble_config.lstm_weight,
            "transformer_weight": self.ensemble_config.transformer_weight,
            "lstm_trained": self._lstm_model.is_model_trained() if self._lstm_model else False,
            "transformer_trained": self._transformer_model.is_model_trained() if self._transformer_model else False,
            "meta_learner_available": self._meta_learner is not None,
            "lstm_performance_history": self._lstm_performance_history[-10:],  # Last 10
            "transformer_performance_history": self._transformer_performance_history[-10:],
            "ensemble_performance_history": self._ensemble_performance_history[-10:],
        }

    def set_weights(self, lstm_weight: float, transformer_weight: float):
        """Update model weights"""
        total = lstm_weight + transformer_weight
        if total > 0:
            self.ensemble_config.lstm_weight = lstm_weight / total
            self.ensemble_config.transformer_weight = transformer_weight / total

            self.logger.info("Ensemble weights updated",
                           lstm_weight=self.ensemble_config.lstm_weight,
                           transformer_weight=self.ensemble_config.transformer_weight)

    def set_strategy(self, strategy: EnsembleStrategy):
        """Update ensemble strategy"""
        self.ensemble_config.strategy = strategy
        self.logger.info("Ensemble strategy updated", strategy=strategy.value)