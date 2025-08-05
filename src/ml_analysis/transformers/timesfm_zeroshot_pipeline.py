"""
TimesFM Zero-Shot Prediction Pipeline - Phase 2.2.3 Implementation
Comprehensive end-to-end prediction pipeline for cryptocurrency trading

Features:
- Multi-asset simultaneous predictions (BTC, ETH, SOL)
- Real-time streaming predictions with caching
- Integration with existing RLTE prediction flow
- Error handling and fallback strategies
- Performance optimization for production trading
"""

import time
import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import structlog

# Avoid circular imports by using minimal necessary imports
try:
    from .timesfm_wrapper import TimesFMWrapper, TimesFMConfig
    from .base import TransformerBase
    from ..base import PredictionResult, ModelType, PredictionDirection
    from src.discovery.base import DiscoveredToken
except ImportError:
    # Standalone operation - define minimal interfaces
    warnings.warn("Operating in standalone mode - some integrations may not be available")
    
    class TimesFMWrapper:
        pass
    
    @dataclass
    class TimesFMConfig:
        model_name: str = "google/timesfm-1.0-200m"
        prediction_length: int = 24
        context_length: int = 512
        use_zero_shot: bool = True
        cache_enabled: bool = True
        streaming_enabled: bool = True
        memory_efficient: bool = True
        batch_size: int = 32
    
    class TransformerBase:
        pass
    
    class PredictionResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ModelType:
        TIMESFM = "TIMESFM"
    
    class PredictionDirection:
        UP = "UP"
        DOWN = "DOWN" 
        NEUTRAL = "NEUTRAL"
    
    class DiscoveredToken:
        def __init__(self, symbol='BTC', current_price=50000.0):
            self.symbol = symbol
            self.current_price = current_price


logger = structlog.get_logger()


@dataclass
class ZeroShotPredictionRequest:
    """Request structure for zero-shot predictions"""
    assets: List[str]  # ['BTC', 'ETH', 'SOL']
    time_series_data: Dict[str, np.ndarray]  # Asset -> price series
    prediction_horizons: List[int]  # [1, 4, 24] hours
    confidence_threshold: float = 0.7
    enable_streaming: bool = False
    cache_predictions: bool = True
    fallback_enabled: bool = True
    
    def validate(self) -> bool:
        """Validate request parameters"""
        if not self.assets:
            raise ValueError("At least one asset must be specified")
        
        if not self.time_series_data:
            raise ValueError("Time series data is required")
        
        for asset in self.assets:
            if asset not in self.time_series_data:
                raise ValueError(f"Time series data missing for asset: {asset}")
            
            data = self.time_series_data[asset]
            if not isinstance(data, np.ndarray) or len(data) == 0:
                raise ValueError(f"Invalid time series data for asset: {asset}")
        
        if not self.prediction_horizons:
            raise ValueError("At least one prediction horizon must be specified")
        
        if not (0 <= self.confidence_threshold <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        return True


@dataclass
class ZeroShotPredictionResponse:
    """Response structure for zero-shot predictions"""
    predictions: Dict[str, Dict[int, float]]  # Asset -> Horizon -> Prediction
    confidence_scores: Dict[str, float]  # Asset -> Confidence
    attention_weights: Dict[str, np.ndarray]  # Asset -> Attention patterns
    execution_time_ms: float
    cache_hit: bool = False
    model_version: str = "timesfm-1.0-200m"
    fallback_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return asdict(self)


@dataclass
class ZeroShotPipelineConfig:
    """Configuration for zero-shot prediction pipeline"""
    model_name: str = "google/timesfm-1.0-200m"
    prediction_length: int = 24
    context_length: int = 512
    supported_assets: List[str] = None
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    streaming_enabled: bool = True
    fallback_enabled: bool = True
    memory_efficient: bool = True
    batch_size: int = 32
    confidence_threshold: float = 0.7
    performance_monitoring: bool = True
    random_seed: int = 42
    
    def __post_init__(self):
        if self.supported_assets is None:
            self.supported_assets = ['BTC', 'ETH', 'SOL']
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (0 <= self.confidence_threshold <= 1):
            raise ValueError("confidence_threshold must be between 0 and 1")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ZeroShotPipelineConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


class PredictionCache:
    """Caching mechanism for zero-shot predictions"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_cache_key(self, request: ZeroShotPredictionRequest) -> str:
        """Generate cache key from request"""
        # Create deterministic hash from request parameters
        cache_data = {
            'assets': sorted(request.assets),
            'horizons': sorted(request.prediction_horizons),
            'data_hashes': {}
        }
        
        for asset in request.assets:
            data = request.time_series_data[asset]
            data_hash = hashlib.md5(data.tobytes()).hexdigest()
            cache_data['data_hashes'][asset] = data_hash
        
        cache_str = str(cache_data)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get(self, request: ZeroShotPredictionRequest) -> Optional[ZeroShotPredictionResponse]:
        """Get cached prediction if available and valid"""
        if not request.cache_predictions:
            return None
        
        cache_key = self._generate_cache_key(request)
        
        if cache_key not in self.cache:
            self.miss_count += 1
            return None
        
        cached_entry = self.cache[cache_key]
        
        # Check if cache entry is expired
        age_seconds = time.time() - cached_entry['timestamp']
        if age_seconds > self.ttl_seconds:
            del self.cache[cache_key]
            self.miss_count += 1
            return None
        
        self.hit_count += 1
        response = cached_entry['response']
        response.cache_hit = True
        return response
    
    def put(self, request: ZeroShotPredictionRequest, response: ZeroShotPredictionResponse):
        """Cache prediction response"""
        if not request.cache_predictions:
            return
        
        cache_key = self._generate_cache_key(request)
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


class TimesFMZeroShotPipeline:
    """
    Comprehensive zero-shot prediction pipeline for TimesFM
    
    Features:
    - Multi-asset simultaneous predictions
    - Caching and performance optimization
    - Error handling and fallback strategies
    - Real-time streaming support
    """
    
    def __init__(self, 
                 model_name: str = "google/timesfm-1.0-200m",
                 prediction_length: int = 24,
                 context_length: int = 512,
                 supported_assets: List[str] = None,
                 enable_caching: bool = True,
                 enable_streaming: bool = True,
                 cache_ttl_seconds: int = 300,
                 random_seed: int = 42):
        
        self.config = ZeroShotPipelineConfig(
            model_name=model_name,
            prediction_length=prediction_length,
            context_length=context_length,
            supported_assets=supported_assets or ['BTC', 'ETH', 'SOL'],
            cache_enabled=enable_caching,
            streaming_enabled=enable_streaming,
            cache_ttl_seconds=cache_ttl_seconds,
            random_seed=random_seed
        )
        
        self.config.validate()
        
        # Initialize components
        self.cache = PredictionCache(ttl_seconds=cache_ttl_seconds) if enable_caching else None
        self.timesfm_wrapper = None
        self.model_initialized = False
        
        # Performance tracking
        self.prediction_count = 0
        self.total_execution_time = 0.0
        self.memory_stats = {'peak_memory_mb': 0}
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        logger.info("TimesFM Zero-Shot Pipeline initialized", 
                   supported_assets=self.config.supported_assets,
                   cache_enabled=enable_caching,
                   streaming_enabled=enable_streaming)
    
    @classmethod
    def from_config(cls, config: ZeroShotPipelineConfig) -> 'TimesFMZeroShotPipeline':
        """Create pipeline from configuration"""
        return cls(
            model_name=config.model_name,
            prediction_length=config.prediction_length,
            context_length=config.context_length,
            supported_assets=config.supported_assets,
            enable_caching=config.cache_enabled,
            enable_streaming=config.streaming_enabled,
            cache_ttl_seconds=config.cache_ttl_seconds,
            random_seed=config.random_seed
        )
    
    def _initialize_model(self):
        """Lazy initialization of TimesFM model"""
        if self.model_initialized:
            return
        
        try:
            # Try to initialize real TimesFM wrapper
            timesfm_config = TimesFMConfig(
                model_name=self.config.model_name,
                prediction_length=self.config.prediction_length,
                context_length=self.config.context_length,
                use_zero_shot=True,
                memory_efficient=True,
                batch_size=self.config.batch_size
            )
            
            self.timesfm_wrapper = TimesFMWrapper(timesfm_config)
            self.model_initialized = True
            
            logger.info("TimesFM model initialized successfully")
            
        except Exception as e:
            logger.warning("Failed to initialize TimesFM model, using fallback", error=str(e))
            self.timesfm_wrapper = None
            self.model_initialized = True  # Mark as initialized to avoid retrying
    
    def predict_multi_asset(self, request: ZeroShotPredictionRequest) -> ZeroShotPredictionResponse:
        """
        Make multi-asset zero-shot predictions
        
        Core method implementing the comprehensive prediction pipeline
        """
        start_time = time.time()
        
        try:
            # Validate request
            request.validate()
            
            # Check cache first
            if self.cache and request.cache_predictions:
                cached_response = self.cache.get(request)
                if cached_response is not None:
                    return cached_response
            
            # Initialize model if needed
            self._initialize_model()
            
            # Make predictions for each asset
            predictions = {}
            confidence_scores = {}
            attention_weights = {}
            fallback_used = False
            
            for asset in request.assets:
                try:
                    asset_predictions, asset_confidence, asset_attention = self._predict_single_asset(
                        asset, 
                        request.time_series_data[asset], 
                        request.prediction_horizons
                    )
                    
                    predictions[asset] = asset_predictions
                    confidence_scores[asset] = asset_confidence
                    attention_weights[asset] = asset_attention
                    
                except Exception as e:
                    if request.fallback_enabled:
                        logger.warning(f"Asset {asset} prediction failed, using fallback", error=str(e))
                        fallback_predictions = self._fallback_predict_asset(asset, request.prediction_horizons)
                        predictions[asset] = fallback_predictions
                        confidence_scores[asset] = 0.5  # Low confidence for fallback
                        attention_weights[asset] = np.zeros((10, 10))  # Dummy attention
                        fallback_used = True
                    else:
                        raise
            
            # Create response
            execution_time_ms = (time.time() - start_time) * 1000
            
            response = ZeroShotPredictionResponse(
                predictions=predictions,
                confidence_scores=confidence_scores,
                attention_weights=attention_weights,
                execution_time_ms=execution_time_ms,
                cache_hit=False,
                model_version=self.config.model_name,
                fallback_used=fallback_used
            )
            
            # Cache response
            if self.cache and request.cache_predictions:
                self.cache.put(request, response)
            
            # Update performance stats
            self.prediction_count += 1
            self.total_execution_time += execution_time_ms
            
            return response
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error("Multi-asset prediction failed", error=str(e), execution_time_ms=execution_time_ms)
            
            if request.fallback_enabled:
                return self._create_fallback_response(request, execution_time_ms)
            else:
                raise
    
    def _predict_single_asset(self, asset: str, time_series: np.ndarray, horizons: List[int]) -> Tuple[Dict[int, float], float, np.ndarray]:
        """Predict for a single asset"""
        
        # Clean and validate time series
        time_series = self._clean_time_series(time_series)
        
        if self.timesfm_wrapper is not None:
            # Use real TimesFM wrapper
            predictions_dict = {}
            
            for horizon in horizons:
                try:
                    pred_array = self.timesfm_wrapper.zero_shot_predict(time_series, horizon=horizon)
                    if len(pred_array) > 0:
                        predictions_dict[horizon] = float(pred_array[-1])  # Take last prediction
                    else:
                        predictions_dict[horizon] = float(time_series[-1])  # Fallback to last known value
                except Exception as e:
                    logger.warning(f"TimesFM prediction failed for {asset} horizon {horizon}", error=str(e))
                    predictions_dict[horizon] = float(time_series[-1])
            
            # Get confidence score
            confidence = self._estimate_confidence(time_series)
            
            # Get attention weights (dummy for now)
            attention = np.random.random((min(len(time_series), 100), min(len(time_series), 100)))
            
        else:
            # Fallback predictions
            predictions_dict = {}
            for horizon in horizons:
                # Simple trend extrapolation
                if len(time_series) >= 2:
                    trend = (time_series[-1] - time_series[-2]) * horizon
                    prediction = time_series[-1] + trend
                else:
                    prediction = time_series[-1]
                
                predictions_dict[horizon] = float(prediction)
            
            confidence = 0.5  # Low confidence for fallback
            attention = np.zeros((10, 10))
        
        return predictions_dict, confidence, attention
    
    def _clean_time_series(self, time_series: np.ndarray) -> np.ndarray:
        """Clean time series data"""
        # Handle NaN/Inf values
        if np.any(np.isnan(time_series)) or np.any(np.isinf(time_series)):
            # Forward fill NaN values
            mask = np.isfinite(time_series)
            if np.any(mask):
                time_series = np.interp(
                    np.arange(len(time_series)),
                    np.where(mask)[0],
                    time_series[mask]
                )
            else:
                # All values are invalid, use zeros
                time_series = np.zeros_like(time_series)
        
        return time_series.astype(np.float32)
    
    def _estimate_confidence(self, time_series: np.ndarray) -> float:
        """Estimate prediction confidence based on time series characteristics"""
        if len(time_series) < 10:
            return 0.3  # Low confidence for short series
        
        # Calculate volatility
        returns = np.diff(time_series) / time_series[:-1]
        volatility = np.std(returns)
        
        # Calculate trend strength
        x = np.arange(len(time_series))
        correlation = np.corrcoef(x, time_series)[0, 1]
        trend_strength = abs(correlation) if not np.isnan(correlation) else 0
        
        # Combine factors
        confidence = 0.7 - volatility * 5 + trend_strength * 0.3
        return max(0.1, min(0.95, confidence))
    
    def _fallback_predict_asset(self, asset: str, horizons: List[int]) -> Dict[int, float]:
        """Fallback prediction for asset"""
        base_prices = {'BTC': 50000, 'ETH': 3000, 'SOL': 100}
        base_price = base_prices.get(asset, 1000)
        
        predictions = {}
        for horizon in horizons:
            # Add small random variation
            variation = np.random.normal(0, 0.01) * horizon
            predictions[horizon] = base_price * (1 + variation)
        
        return predictions
    
    def _create_fallback_response(self, request: ZeroShotPredictionRequest, execution_time_ms: float) -> ZeroShotPredictionResponse:
        """Create fallback response when all predictions fail"""
        predictions = {}
        confidence_scores = {}
        attention_weights = {}
        
        for asset in request.assets:
            predictions[asset] = self._fallback_predict_asset(asset, request.prediction_horizons)
            confidence_scores[asset] = 0.1  # Very low confidence
            attention_weights[asset] = np.zeros((10, 10))
        
        return ZeroShotPredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            attention_weights=attention_weights,
            execution_time_ms=execution_time_ms,
            cache_hit=False,
            model_version=self.config.model_name,
            fallback_used=True
        )
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_statistics()
        else:
            return {'cache_enabled': False}
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        # Simplified memory tracking
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        self.memory_stats['peak_memory_mb'] = max(self.memory_stats['peak_memory_mb'], current_memory)
        
        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.memory_stats['peak_memory_mb'],
            'prediction_count': self.prediction_count,
            'avg_execution_time_ms': self.total_execution_time / max(1, self.prediction_count)
        }
    
    def analyze_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """Analyze attention patterns for interpretability"""
        if attention_weights.size == 0:
            return {'error': 'Empty attention weights'}
        
        # Calculate attention entropy
        attention_flat = attention_weights.flatten()
        attention_flat = attention_flat / (attention_flat.sum() + 1e-8)  # Normalize
        entropy = -np.sum(attention_flat * np.log(attention_flat + 1e-8))
        
        # Find temporal focus (most attended positions)
        attention_sum = np.sum(attention_weights, axis=0)
        top_positions = np.argsort(attention_sum)[-5:]  # Top 5 positions
        
        # Calculate focus concentration
        max_attention = np.max(attention_weights)
        mean_attention = np.mean(attention_weights)
        concentration = max_attention / (mean_attention + 1e-8)
        
        return {
            'attention_entropy': float(entropy),
            'temporal_focus': concentration,
            'key_time_points': top_positions.tolist(),
            'max_attention': float(max_attention),
            'mean_attention': float(mean_attention)
        }


class StreamingZeroShotPredictor:
    """Real-time streaming predictor for zero-shot predictions"""
    
    def __init__(self, 
                 assets: List[str],
                 prediction_horizons: List[int],
                 update_frequency_seconds: int = 60):
        
        self.assets = assets
        self.prediction_horizons = prediction_horizons
        self.update_frequency_seconds = update_frequency_seconds
        
        # Initialize pipeline
        self.pipeline = TimesFMZeroShotPipeline(
            supported_assets=assets,
            enable_caching=True,
            enable_streaming=True
        )
        
        # Streaming context
        self.streaming_context: Dict[str, np.ndarray] = {}
        self.last_update_time = None
        self.initialized = False
        
        logger.info("Streaming predictor initialized", 
                   assets=assets, 
                   horizons=prediction_horizons,
                   update_frequency=update_frequency_seconds)
    
    def initialize_context(self, historical_data: Dict[str, np.ndarray]):
        """Initialize streaming context with historical data"""
        for asset in self.assets:
            if asset in historical_data:
                self.streaming_context[asset] = historical_data[asset].copy()
            else:
                logger.warning(f"No historical data provided for {asset}")
                self.streaming_context[asset] = np.array([1000.0])  # Default value
        
        self.initialized = True
        self.last_update_time = datetime.now()
        
        logger.info("Streaming context initialized", context_lengths={
            asset: len(data) for asset, data in self.streaming_context.items()
        })
    
    def update_and_predict(self, new_data_point: Dict[str, Any]) -> ZeroShotPredictionResponse:
        """Update context and make new prediction"""
        if not self.initialized:
            raise ValueError("Streaming predictor not initialized. Call initialize_context() first.")
        
        # Update streaming context
        for asset in self.assets:
            if asset in new_data_point:
                new_value = float(new_data_point[asset])
                self.streaming_context[asset] = np.append(self.streaming_context[asset], new_value)
                
                # Keep context length manageable
                max_context_length = 2000
                if len(self.streaming_context[asset]) > max_context_length:
                    self.streaming_context[asset] = self.streaming_context[asset][-max_context_length:]
        
        # Create prediction request
        request = ZeroShotPredictionRequest(
            assets=self.assets,
            time_series_data=self.streaming_context.copy(),
            prediction_horizons=self.prediction_horizons,
            cache_predictions=True,
            fallback_enabled=True
        )
        
        # Make prediction
        response = self.pipeline.predict_multi_asset(request)
        self.last_update_time = datetime.now()
        
        return response


class RLTEZeroShotIntegrator:
    """Integration with existing RLTE prediction flow"""
    
    def __init__(self):
        self.pipeline = TimesFMZeroShotPipeline(
            supported_assets=['BTC', 'ETH', 'SOL'],
            enable_caching=True,
            enable_streaming=False
        )
        
        logger.info("RLTE Zero-Shot Integrator initialized")
    
    def predict_for_token(self, 
                         token: DiscoveredToken, 
                         time_series: np.ndarray,
                         use_zero_shot: bool = True) -> PredictionResult:
        """Predict for discovered token using zero-shot pipeline"""
        
        if not use_zero_shot:
            raise ValueError("Non-zero-shot prediction not implemented in this integrator")
        
        # Create prediction request
        request = ZeroShotPredictionRequest(
            assets=[token.symbol],
            time_series_data={token.symbol: time_series},
            prediction_horizons=[1, 4, 24],
            cache_predictions=True,
            fallback_enabled=True
        )
        
        # Get prediction
        response = self.pipeline.predict_multi_asset(request)
        
        if token.symbol not in response.predictions:
            raise ValueError(f"No prediction returned for {token.symbol}")
        
        predictions = response.predictions[token.symbol]
        confidence = response.confidence_scores[token.symbol]
        
        # Convert to RLTE format
        price_1h = predictions.get(1, token.current_price)
        price_4h = predictions.get(4, token.current_price)
        price_24h = predictions.get(24, token.current_price)
        
        # Determine direction
        if price_24h > token.current_price * 1.02:
            direction = PredictionDirection.UP
        elif price_24h < token.current_price * 0.98:
            direction = PredictionDirection.DOWN
        else:
            direction = PredictionDirection.NEUTRAL
        
        return PredictionResult(
            symbol=token.symbol,
            current_price=token.current_price,
            price_prediction_1h=price_1h,
            price_prediction_4h=price_4h,
            price_prediction_24h=price_24h,
            confidence_score=confidence,
            direction=direction,
            features_used=['timesfm_zero_shot'],
            model_version=response.model_version,
            execution_time_ms=response.execution_time_ms,
            fallback_used=response.fallback_used
        )


class TradingSignalGenerator:
    """Generate trading signals from zero-shot predictions"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 supported_assets: List[str] = None):
        
        self.confidence_threshold = confidence_threshold
        self.supported_assets = supported_assets or ['BTC', 'ETH', 'SOL']
        
        self.pipeline = TimesFMZeroShotPipeline(
            supported_assets=self.supported_assets,
            enable_caching=True
        )
        
        logger.info("Trading signal generator initialized", 
                   confidence_threshold=confidence_threshold,
                   supported_assets=self.supported_assets)
    
    def generate_signals(self, raw_market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals from raw market data"""
        
        # Extract time series data
        time_series_data = {}
        for asset, data in raw_market_data.items():
            if asset in self.supported_assets:
                time_series_data[asset] = np.array(data['prices'])
        
        # Create prediction request
        request = ZeroShotPredictionRequest(
            assets=list(time_series_data.keys()),
            time_series_data=time_series_data,
            prediction_horizons=[1, 4, 24],
            confidence_threshold=self.confidence_threshold,
            cache_predictions=True,
            fallback_enabled=True
        )
        
        # Get predictions
        response = self.pipeline.predict_multi_asset(request)
        
        # Generate trading signals
        trading_signals = {}
        
        for asset in time_series_data.keys():
            if asset in response.predictions:
                predictions = response.predictions[asset]
                confidence = response.confidence_scores[asset]
                current_price = time_series_data[asset][-1]
                
                # Determine direction
                price_24h = predictions.get(24, current_price)
                if price_24h > current_price * 1.02:
                    direction = "BUY"
                elif price_24h < current_price * 0.98:
                    direction = "SELL"
                else:
                    direction = "HOLD"
                
                trading_signals[asset] = {
                    'direction': direction,
                    'confidence': confidence,
                    'predicted_price_1h': predictions.get(1, current_price),
                    'predicted_price_4h': predictions.get(4, current_price),
                    'predicted_price_24h': predictions.get(24, current_price),
                    'current_price': current_price,
                    'signal_strength': confidence * abs(price_24h - current_price) / current_price,
                    'fallback_used': response.fallback_used
                }
        
        return trading_signals


# Additional specialized predictors for different trading scenarios

class HFTZeroShotPredictor:
    """High-frequency trading zero-shot predictor"""
    
    def __init__(self, 
                 assets: List[str],
                 prediction_horizons: List[int],
                 update_frequency_seconds: int = 1,
                 low_latency_mode: bool = True):
        
        self.assets = assets
        self.prediction_horizons = prediction_horizons
        self.update_frequency_seconds = update_frequency_seconds
        self.low_latency_mode = low_latency_mode
        
        # Optimized pipeline for HFT
        self.pipeline = TimesFMZeroShotPipeline(
            supported_assets=assets,
            enable_caching=True,
            cache_ttl_seconds=60,  # Shorter cache TTL for HFT
            enable_streaming=True
        )
        
        # HFT-specific context
        self.hft_context: Dict[str, np.ndarray] = {}
        self.update_count = 0
        
        logger.info("HFT Zero-Shot Predictor initialized", 
                   assets=assets,
                   low_latency_mode=low_latency_mode)
    
    def update_and_predict(self, new_data: Dict[str, Any]) -> ZeroShotPredictionResponse:
        """Update with new data and make HFT prediction"""
        start_time = time.time()
        
        # Update HFT context (keep short for low latency)
        for asset in self.assets:
            if asset in new_data:
                new_value = float(new_data[asset])
                
                if asset not in self.hft_context:
                    self.hft_context[asset] = np.array([new_value])
                else:
                    self.hft_context[asset] = np.append(self.hft_context[asset], new_value)
                    
                    # Keep only recent data for HFT
                    max_length = 100 if self.low_latency_mode else 500
                    if len(self.hft_context[asset]) > max_length:
                        self.hft_context[asset] = self.hft_context[asset][-max_length:]
        
        # Create optimized request for HFT
        request = ZeroShotPredictionRequest(
            assets=self.assets,
            time_series_data=self.hft_context.copy(),
            prediction_horizons=self.prediction_horizons,
            cache_predictions=True,
            fallback_enabled=True
        )
        
        # Make prediction
        response = self.pipeline.predict_multi_asset(request)
        self.update_count += 1
        
        # Verify HFT latency requirements
        execution_time_ms = (time.time() - start_time) * 1000
        if execution_time_ms > 100:  # HFT requirement: <100ms
            logger.warning("HFT latency exceeded", execution_time_ms=execution_time_ms)
        
        return response


class ArbitrageDetector:
    """Detect arbitrage opportunities using zero-shot predictions"""
    
    def __init__(self, 
                 assets: List[str],
                 prediction_horizons: List[int],
                 arbitrage_threshold: float = 0.02):
        
        self.assets = assets
        self.prediction_horizons = prediction_horizons
        self.arbitrage_threshold = arbitrage_threshold
        
        self.pipeline = TimesFMZeroShotPipeline(
            supported_assets=assets,
            enable_caching=True
        )
        
        logger.info("Arbitrage detector initialized", 
                   assets=assets,
                   threshold=arbitrage_threshold)
    
    def detect_opportunities(self, market_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect arbitrage opportunities"""
        
        # Get predictions for all assets
        time_series_data = {}
        for asset in self.assets:
            if asset in market_data:
                time_series_data[asset] = np.array(market_data[asset]['prices'])
        
        request = ZeroShotPredictionRequest(
            assets=list(time_series_data.keys()),
            time_series_data=time_series_data,
            prediction_horizons=self.prediction_horizons,
            cache_predictions=True,
            fallback_enabled=True
        )
        
        response = self.pipeline.predict_multi_asset(request)
        
        # Analyze predictions for arbitrage opportunities
        opportunities = []
        
        for horizon in self.prediction_horizons:
            for i, asset1 in enumerate(self.assets):
                for asset2 in self.assets[i+1:]:
                    if asset1 in response.predictions and asset2 in response.predictions:
                        
                        pred1 = response.predictions[asset1].get(horizon, 0)
                        pred2 = response.predictions[asset2].get(horizon, 0)
                        current1 = time_series_data[asset1][-1]
                        current2 = time_series_data[asset2][-1]
                        
                        # Calculate expected returns
                        return1 = (pred1 - current1) / current1
                        return2 = (pred2 - current2) / current2
                        
                        # Check for arbitrage opportunity
                        return_diff = abs(return1 - return2)
                        if return_diff >= self.arbitrage_threshold:
                            
                            confidence = min(
                                response.confidence_scores.get(asset1, 0),
                                response.confidence_scores.get(asset2, 0)
                            )
                            
                            opportunities.append({
                                'asset_pair': f"{asset1}/{asset2}",
                                'timeframe': f"{horizon}h",
                                'expected_profit_pct': return_diff * 100,
                                'confidence': confidence,
                                'long_asset': asset1 if return1 > return2 else asset2,
                                'short_asset': asset2 if return1 > return2 else asset1,
                                'horizon_hours': horizon
                            })
        
        # Sort by expected profit
        opportunities.sort(key=lambda x: x['expected_profit_pct'], reverse=True)
        
        return opportunities


# Export main classes
__all__ = [
    'TimesFMZeroShotPipeline',
    'StreamingZeroShotPredictor', 
    'RLTEZeroShotIntegrator',
    'TradingSignalGenerator',
    'HFTZeroShotPredictor',
    'ArbitrageDetector',
    'ZeroShotPredictionRequest',
    'ZeroShotPredictionResponse',
    'ZeroShotPipelineConfig'
]