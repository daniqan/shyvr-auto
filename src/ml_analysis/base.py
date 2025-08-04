"""
Base classes and data structures for ML analysis
Defines interfaces for price prediction and feature engineering
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import structlog

from src.utils.base import Chain
from src.discovery.base import DiscoveredToken


logger = structlog.get_logger()


class ModelType(Enum):
    """Types of ML models available"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ITRANSFORMER = "itransformer"  # Inverted Transformer for multivariate time-series
    PATCHTST = "patchtst"          # Patch-based Transformer for time-series
    TIMESMIXER = "timesmixer"      # Times-Mixer with decomposable mixing
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"


class PredictionDirection(Enum):
    """Price prediction directions"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    # Price-based indicators
    sma_20: Optional[float] = None        # Simple Moving Average (20 periods)
    sma_50: Optional[float] = None        # Simple Moving Average (50 periods)
    ema_12: Optional[float] = None        # Exponential Moving Average (12 periods)
    ema_26: Optional[float] = None        # Exponential Moving Average (26 periods)
    
    # Momentum indicators
    rsi: Optional[float] = None           # Relative Strength Index
    macd: Optional[float] = None          # MACD line
    macd_signal: Optional[float] = None   # MACD signal line
    macd_histogram: Optional[float] = None # MACD histogram
    
    # Volatility indicators
    bollinger_upper: Optional[float] = None    # Bollinger Band upper
    bollinger_lower: Optional[float] = None    # Bollinger Band lower
    bollinger_width: Optional[float] = None    # Bollinger Band width
    atr: Optional[float] = None                # Average True Range
    
    # Volume indicators
    volume_sma: Optional[float] = None         # Volume Simple Moving Average
    volume_ratio: Optional[float] = None       # Current volume / Average volume
    obv: Optional[float] = None                # On-Balance Volume
    
    # Custom indicators
    price_momentum: Optional[float] = None     # Price momentum (custom)
    volatility_score: Optional[float] = None  # Volatility score (custom)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert indicators to numpy feature vector for ML models"""
        features = [
            self.sma_20 or 0.0,
            self.sma_50 or 0.0,
            self.ema_12 or 0.0,
            self.ema_26 or 0.0,
            self.rsi or 50.0,
            self.macd or 0.0,
            self.macd_signal or 0.0,
            self.macd_histogram or 0.0,
            self.bollinger_upper or 0.0,
            self.bollinger_lower or 0.0,
            self.bollinger_width or 0.0,
            self.atr or 0.0,
            self.volume_sma or 0.0,
            self.volume_ratio or 1.0,
            self.obv or 0.0,
            self.price_momentum or 0.0,
            self.volatility_score or 0.0,
        ]
        return np.array(features, dtype=np.float32)


@dataclass
class MarketFeatures:
    """Market context features for ML prediction"""
    # Market sentiment
    fear_greed_index: Optional[float] = None      # Market fear/greed index (0-100)
    fear_greed_classification: Optional[str] = None  # "Fear", "Greed", etc.
    market_trend: Optional[str] = None            # Bull/bear/sideways
    volatility_regime: Optional[str] = None       # High/medium/low volatility
    
    # Cross-asset correlations
    btc_correlation: Optional[float] = None       # Correlation with Bitcoin
    eth_correlation: Optional[float] = None       # Correlation with Ethereum
    btc_dominance: Optional[float] = None         # Bitcoin market dominance %
    eth_dominance: Optional[float] = None         # Ethereum market dominance %
    stablecoin_dominance: Optional[float] = None  # Stablecoin dominance %
    market_beta: Optional[float] = None           # Beta relative to market
    
    # DeFi ecosystem metrics
    total_value_locked: Optional[float] = None    # Total DeFi TVL in USD
    tvl_change_24h: Optional[float] = None        # 24h TVL change %
    tvl_change_7d: Optional[float] = None         # 7d TVL change %
    defi_dominance: Optional[float] = None        # DeFi TVL / Total market cap
    active_protocols: Optional[int] = None        # Number of active DeFi protocols
    
    # On-chain activity metrics
    transaction_count_24h: Optional[int] = None   # 24h transaction count
    active_addresses_24h: Optional[int] = None    # 24h active addresses
    transaction_volume_24h: Optional[float] = None  # 24h transaction volume USD
    network_fees_24h: Optional[float] = None      # 24h network fees USD
    whale_activity_score: Optional[float] = None  # Large holder activity score
    
    # Social sentiment
    social_score: Optional[float] = None          # Aggregated social sentiment (0-1)
    mention_volume: Optional[int] = None          # Social media mentions
    sentiment_trend: Optional[float] = None       # Sentiment change trend
    influencer_sentiment: Optional[float] = None  # Weighted influencer sentiment
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert market features to numpy feature vector"""
        features = [
            # Market sentiment features
            self.fear_greed_index or 50.0,
            1.0 if self.market_trend == "bull" else 0.0,
            1.0 if self.market_trend == "bear" else 0.0,
            1.0 if self.volatility_regime == "high" else 0.0,
            1.0 if self.volatility_regime == "low" else 0.0,
            
            # Correlation and dominance features
            self.btc_correlation or 0.0,
            self.eth_correlation or 0.0,
            self.btc_dominance or 40.0,
            self.eth_dominance or 15.0,
            self.stablecoin_dominance or 10.0,
            self.market_beta or 1.0,
            
            # DeFi ecosystem features
            min(self.total_value_locked or 0, 200e9) / 200e9,  # Normalized to $200B max
            self.tvl_change_24h or 0.0,
            self.tvl_change_7d or 0.0,
            self.defi_dominance or 0.05,
            min(self.active_protocols or 0, 500) / 500.0,  # Normalized to 500 max
            
            # On-chain activity features
            min(self.transaction_count_24h or 0, 2e6) / 2e6,  # Normalized to 2M max
            min(self.active_addresses_24h or 0, 1e6) / 1e6,   # Normalized to 1M max
            min(self.transaction_volume_24h or 0, 50e9) / 50e9,  # Normalized to $50B max
            min(self.network_fees_24h or 0, 10e6) / 10e6,     # Normalized to $10M max
            self.whale_activity_score or 0.5,
            
            # Social sentiment features
            self.social_score or 0.5,
            min(self.mention_volume or 0, 10000) / 10000.0,   # Normalized to 10k max
            self.sentiment_trend or 0.0,
            self.influencer_sentiment or 0.5,
        ]
        return np.array(features, dtype=np.float32)


@dataclass
class PredictionResult:
    """Result of ML price prediction analysis"""
    # Basic info
    token: DiscoveredToken
    analyzed_at: datetime
    model_type: ModelType
    
    # Price predictions
    price_prediction_1h: Optional[float] = None     # 1-hour price prediction
    price_prediction_4h: Optional[float] = None     # 4-hour price prediction
    price_prediction_24h: Optional[float] = None    # 24-hour price prediction
    
    # Direction and confidence
    direction: PredictionDirection = PredictionDirection.HOLD
    confidence: float = 0.0                         # 0-1 confidence score
    probability_up: float = 0.5                     # Probability of price increase
    
    # Technical analysis
    technical_indicators: Optional[TechnicalIndicators] = None
    market_features: Optional[MarketFeatures] = None
    
    # Model performance metrics
    model_accuracy: Optional[float] = None          # Historical accuracy
    prediction_uncertainty: Optional[float] = None  # Model uncertainty
    
    # Risk assessment
    volatility_forecast: Optional[float] = None     # Predicted volatility
    downside_risk: Optional[float] = None           # Downside risk estimate
    upside_potential: Optional[float] = None        # Upside potential estimate
    
    # Trading signals
    entry_signal: Optional[str] = None              # Entry signal (buy/sell/wait)
    stop_loss_level: Optional[float] = None         # Suggested stop loss
    take_profit_level: Optional[float] = None       # Suggested take profit
    position_size_multiplier: float = 1.0          # Position sizing modifier
    
    # Metadata
    features_used: List[str] = None                 # Features used in prediction
    model_version: Optional[str] = None             # Model version used
    processing_time_ms: Optional[float] = None      # Analysis time
    
    def __post_init__(self):
        if self.features_used is None:
            self.features_used = []
    
    def get_expected_return(self, timeframe: str = "24h") -> Optional[float]:
        """Calculate expected return for given timeframe"""
        current_price = self.token.price_usd
        if not current_price:
            return None
            
        if timeframe == "1h" and self.price_prediction_1h:
            return (self.price_prediction_1h - current_price) / current_price
        elif timeframe == "4h" and self.price_prediction_4h:
            return (self.price_prediction_4h - current_price) / current_price
        elif timeframe == "24h" and self.price_prediction_24h:
            return (self.price_prediction_24h - current_price) / current_price
        
        return None
    
    def get_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio"""
        if self.upside_potential is None or self.downside_risk is None:
            return None
        
        if self.downside_risk == 0:
            return float('inf') if self.upside_potential > 0 else 0
        
        return abs(self.upside_potential / self.downside_risk)
    
    def is_buy_signal(self, min_confidence: float = 0.6) -> bool:
        """Check if this is a buy signal with sufficient confidence"""
        return (
            self.direction in [PredictionDirection.BUY, PredictionDirection.STRONG_BUY] and
            self.confidence >= min_confidence and
            self.probability_up > 0.6
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "token_address": self.token.address,
            "chain": self.token.chain.value,
            "analyzed_at": self.analyzed_at.isoformat(),
            "model_type": self.model_type.value,
            "price_prediction_1h": self.price_prediction_1h,
            "price_prediction_4h": self.price_prediction_4h,
            "price_prediction_24h": self.price_prediction_24h,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "probability_up": self.probability_up,
            "model_accuracy": self.model_accuracy,
            "volatility_forecast": self.volatility_forecast,
            "entry_signal": self.entry_signal,
            "stop_loss_level": self.stop_loss_level,
            "take_profit_level": self.take_profit_level,
            "position_size_multiplier": self.position_size_multiplier,
            "expected_return_24h": self.get_expected_return("24h"),
            "risk_reward_ratio": self.get_risk_reward_ratio(),
            "features_used": self.features_used,
            "model_version": self.model_version,
            "processing_time_ms": self.processing_time_ms,
        }


class MLAnalyzerBase(ABC):
    """Abstract base class for ML analyzers"""
    
    def __init__(self, model_type: ModelType, config: Optional[Dict] = None):
        self.model_type = model_type
        self.config = config or {}
        self.logger = structlog.get_logger().bind(analyzer=self.__class__.__name__)
        self._model = None
        self._is_trained = False
    
    @abstractmethod
    async def analyze_token(self, token: DiscoveredToken, 
                          historical_data: Optional[pd.DataFrame] = None) -> PredictionResult:
        """Analyze a token and generate ML predictions"""
        pass
    
    @abstractmethod
    async def train_model(self, training_data: pd.DataFrame) -> bool:
        """Train the ML model on historical data"""
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Get list of required features for this model"""
        pass
    
    async def batch_analyze(self, tokens: List[DiscoveredToken], 
                          historical_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[PredictionResult]:
        """Analyze multiple tokens in batch"""
        results = []
        for token in tokens:
            try:
                token_data = historical_data.get(token.address) if historical_data else None
                result = await self.analyze_token(token, token_data)
                results.append(result)
            except Exception as e:
                self.logger.error("Token analysis failed", 
                                token=token.address, 
                                error=str(e))
                # Create failed result
                results.append(PredictionResult(
                    token=token,
                    analyzed_at=datetime.now(),
                    model_type=self.model_type,
                    confidence=0.0,
                    features_used=["error"],
                ))
        
        return results
    
    def is_model_trained(self) -> bool:
        """Check if model is trained and ready"""
        return self._is_trained
    
    async def health_check(self) -> bool:
        """Check if the analyzer is healthy and ready"""
        try:
            return self.is_model_trained() and self._model is not None
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False


class MLAnalysisError(Exception):
    """Base exception for ML analysis errors"""
    pass


class ModelNotTrainedError(MLAnalysisError):
    """Raised when trying to use an untrained model"""
    pass


class FeatureEngineeringError(MLAnalysisError):
    """Raised when feature engineering fails"""
    pass


class PredictionError(MLAnalysisError):
    """Raised when prediction generation fails"""
    pass