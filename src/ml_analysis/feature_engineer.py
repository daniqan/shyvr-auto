"""
Feature Engineering Module
Technical indicators and market features for ML models
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import structlog

from src.discovery.base import DiscoveredToken
from .base import TechnicalIndicators, MarketFeatures, FeatureEngineeringError


logger = structlog.get_logger()


class FeatureEngineer:
    """Feature engineering for ML models"""
    
    def __init__(self, cache_ttl_minutes: int = 30):
        self.cache_ttl_minutes = cache_ttl_minutes
        self._indicator_cache: Dict[str, Tuple[datetime, TechnicalIndicators]] = {}
        self._market_cache: Optional[Tuple[datetime, MarketFeatures]] = None
        self.logger = structlog.get_logger().bind(component="FeatureEngineer")
    
    async def calculate_technical_indicators(self, 
                                           token: DiscoveredToken,
                                           price_data: pd.DataFrame) -> TechnicalIndicators:
        """
        Calculate technical indicators from price data
        
        Args:
            token: Token to analyze
            price_data: DataFrame with columns: timestamp, open, high, low, close, volume
        
        Returns:
            TechnicalIndicators object with calculated values
        """
        cache_key = f"{token.chain_address}"
        
        # Check cache first
        if cache_key in self._indicator_cache:
            cached_time, cached_indicators = self._indicator_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                return cached_indicators
        
        try:
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low', 'volume']
            if not all(col in price_data.columns for col in required_cols):
                raise FeatureEngineeringError(f"Missing required columns. Need: {required_cols}")
            
            if len(price_data) < 50:  # Need sufficient data for indicators
                self.logger.warning("Insufficient data for technical indicators", 
                                  token=token.address, 
                                  rows=len(price_data))
                return self._create_default_indicators()
            
            # Calculate indicators
            indicators = TechnicalIndicators()
            
            # Moving averages
            indicators.sma_20 = self._calculate_sma(price_data['close'], 20)
            indicators.sma_50 = self._calculate_sma(price_data['close'], 50)
            indicators.ema_12 = self._calculate_ema(price_data['close'], 12)
            indicators.ema_26 = self._calculate_ema(price_data['close'], 26)
            
            # Momentum indicators
            indicators.rsi = self._calculate_rsi(price_data['close'])
            macd_values = self._calculate_macd(price_data['close'])
            indicators.macd = macd_values['macd']
            indicators.macd_signal = macd_values['signal']
            indicators.macd_histogram = macd_values['histogram']
            
            # Volatility indicators
            bollinger = self._calculate_bollinger_bands(price_data['close'])
            indicators.bollinger_upper = bollinger['upper']
            indicators.bollinger_lower = bollinger['lower']
            indicators.bollinger_width = bollinger['width']
            indicators.atr = self._calculate_atr(price_data)
            
            # Volume indicators
            indicators.volume_sma = self._calculate_sma(price_data['volume'], 20)
            indicators.volume_ratio = self._calculate_volume_ratio(price_data['volume'])
            indicators.obv = self._calculate_obv(price_data['close'], price_data['volume'])
            
            # Custom indicators
            indicators.price_momentum = self._calculate_price_momentum(price_data['close'])
            indicators.volatility_score = self._calculate_volatility_score(price_data['close'])
            
            # Cache the result
            self._indicator_cache[cache_key] = (datetime.now(), indicators)
            
            self.logger.info("Technical indicators calculated", 
                           token=token.address,
                           indicators_count=len([x for x in indicators.__dict__.values() if x is not None]))
            
            return indicators
            
        except Exception as e:
            self.logger.error("Technical indicator calculation failed", 
                            token=token.address, 
                            error=str(e))
            raise FeatureEngineeringError(f"Failed to calculate indicators: {str(e)}")
    
    async def calculate_market_features(self) -> MarketFeatures:
        """
        Calculate market-wide features
        
        Returns:
            MarketFeatures object with market context
        """
        # Check cache first
        if self._market_cache:
            cached_time, cached_features = self._market_cache
            if datetime.now() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                return cached_features
        
        try:
            features = MarketFeatures()
            
            # For now, use placeholder values - in production these would come from APIs
            features.fear_greed_index = 50.0  # Neutral
            features.market_trend = "sideways"
            features.volatility_regime = "medium"
            features.btc_correlation = 0.5
            features.eth_correlation = 0.4
            features.market_beta = 1.0
            features.social_score = 0.5
            features.mention_volume = 100
            features.sentiment_trend = 0.0
            
            # Cache the result
            self._market_cache = (datetime.now(), features)
            
            self.logger.info("Market features calculated")
            return features
            
        except Exception as e:
            self.logger.error("Market feature calculation failed", error=str(e))
            raise FeatureEngineeringError(f"Failed to calculate market features: {str(e)}")
    
    def _create_default_indicators(self) -> TechnicalIndicators:
        """Create default indicators when calculation fails"""
        return TechnicalIndicators(
            rsi=50.0,  # Neutral RSI
            volume_ratio=1.0,  # Normal volume
            price_momentum=0.0,  # No momentum
            volatility_score=0.5,  # Medium volatility
        )
    
    def _calculate_sma(self, series: pd.Series, window: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(series) < window:
            return None
        return float(series.rolling(window=window).mean().iloc[-1])
    
    def _calculate_ema(self, series: pd.Series, window: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(series) < window:
            return None
        return float(series.ewm(span=window).mean().iloc[-1])
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(series) < window + 1:
            return 50.0  # Return default instead of None
        
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Optional[float]]:
        """Calculate MACD indicator"""
        if len(series) < slow:
            return {'macd': None, 'signal': None, 'histogram': None}
        
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            'signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
            'histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0,
        }
    
    def _calculate_bollinger_bands(self, series: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, Optional[float]]:
        """Calculate Bollinger Bands"""
        if len(series) < window:
            return {'upper': None, 'lower': None, 'width': None}
        
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        width = upper - lower
        
        return {
            'upper': float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None,
            'lower': float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None,
            'width': float(width.iloc[-1]) if not pd.isna(width.iloc[-1]) else None,
        }
    
    def _calculate_atr(self, price_data: pd.DataFrame, window: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        if len(price_data) < window:
            return None
        
        high = price_data['high']
        low = price_data['low']
        close = price_data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
    
    def _calculate_volume_ratio(self, volume_series: pd.Series, window: int = 20) -> Optional[float]:
        """Calculate current volume to average volume ratio"""
        if len(volume_series) < window:
            return 1.0
        
        # Calculate average from all but the last (current) volume
        historical_volumes = volume_series.iloc[:-1]
        if len(historical_volumes) < window - 1:
            return 1.0
            
        avg_volume = historical_volumes.tail(window - 1).mean()
        current_volume = volume_series.iloc[-1]
        
        if pd.isna(avg_volume) or avg_volume == 0:
            return 1.0
        
        return float(current_volume / avg_volume)
    
    def _calculate_obv(self, close_series: pd.Series, volume_series: pd.Series) -> Optional[float]:
        """Calculate On-Balance Volume"""
        if len(close_series) < 2:
            return None
        
        obv_values = []
        obv = 0
        
        for i in range(1, len(close_series)):
            if close_series.iloc[i] > close_series.iloc[i-1]:
                obv += volume_series.iloc[i]
            elif close_series.iloc[i] < close_series.iloc[i-1]:
                obv -= volume_series.iloc[i]
            obv_values.append(obv)
        
        return float(obv_values[-1]) if obv_values else 0.0
    
    def _calculate_price_momentum(self, series: pd.Series, window: int = 10) -> Optional[float]:
        """Calculate price momentum (custom indicator)"""
        if len(series) < window:
            return 0.0
        
        recent_prices = series.tail(window)
        momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        return float(momentum) * 100  # Convert to percentage
    
    def _calculate_volatility_score(self, series: pd.Series, window: int = 20) -> Optional[float]:
        """Calculate volatility score (custom indicator)"""
        if len(series) < window:
            return 0.5
        
        returns = series.pct_change().dropna()
        if len(returns) < window:
            return 0.5
        
        volatility = returns.tail(window).std()
        
        # Normalize volatility to 0-1 scale (assuming max volatility of 50% daily)
        normalized_vol = min(volatility / 0.5, 1.0)
        
        return float(normalized_vol)
    
    async def create_feature_matrix(self, 
                                  tokens: List[DiscoveredToken],
                                  price_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix for ML model training
        
        Args:
            tokens: List of tokens to analyze
            price_data: Dictionary mapping token addresses to price DataFrames
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        try:
            feature_vectors = []
            feature_names = []
            
            # Get market features once
            market_features = await self.calculate_market_features()
            
            for token in tokens:
                if token.address not in price_data:
                    self.logger.warning("No price data for token", token=token.address)
                    continue
                
                # Calculate technical indicators
                tech_indicators = await self.calculate_technical_indicators(token, price_data[token.address])
                
                # Combine all features
                tech_vector = tech_indicators.to_feature_vector()
                market_vector = market_features.to_feature_vector()
                
                # Add token-specific features
                token_features = self._extract_token_features(token)
                
                combined_vector = np.concatenate([tech_vector, market_vector, token_features])
                feature_vectors.append(combined_vector)
            
            if not feature_vectors:
                raise FeatureEngineeringError("No valid feature vectors created")
            
            # Create feature names
            if not feature_names:
                feature_names = self._get_feature_names()
            
            feature_matrix = np.array(feature_vectors)
            
            self.logger.info("Feature matrix created", 
                           shape=feature_matrix.shape,
                           tokens=len(tokens))
            
            return feature_matrix, feature_names
            
        except Exception as e:
            self.logger.error("Feature matrix creation failed", error=str(e))
            raise FeatureEngineeringError(f"Failed to create feature matrix: {str(e)}")
    
    def _extract_token_features(self, token: DiscoveredToken) -> np.ndarray:
        """Extract token-specific features"""
        features = [
            # Price features
            np.log(token.price_usd) if token.price_usd and token.price_usd > 0 else 0.0,
            np.log(token.market_cap) if token.market_cap and token.market_cap > 0 else 0.0,
            np.log(token.volume_24h) if token.volume_24h and token.volume_24h > 0 else 0.0,
            token.price_change_24h or 0.0,
            
            # Chain encoding (one-hot for major chains)
            1.0 if token.chain.value == "ethereum" else 0.0,
            1.0 if token.chain.value == "solana" else 0.0,
            1.0 if token.chain.value == "base" else 0.0,
            
            # Token age (days since discovery)
            (datetime.now() - token.discovered_at).days if token.discovered_at else 0.0,
            
            # Trading score (based on available data)
            self._calculate_token_score(token),
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_token_score(self, token: DiscoveredToken) -> float:
        """Calculate a composite token quality score"""
        score = 0.0
        
        # Volume score
        if token.volume_24h and token.market_cap:
            volume_ratio = token.volume_24h / token.market_cap
            score += min(volume_ratio * 10, 1.0)  # Cap at 1.0
        
        # Price stability (inverse of absolute price change)
        if token.price_change_24h is not None:
            stability = 1.0 - min(abs(token.price_change_24h) / 100, 1.0)
            score += stability * 0.5
        
        # Social engagement
        if token.tags:
            if "verified" in token.tags:
                score += 0.3
            if any(tag in ["trending", "rising"] for tag in token.tags):
                score += 0.2
        
        return min(score, 1.0)
    
    def _get_feature_names(self) -> List[str]:
        """Get standardized feature names"""
        tech_names = [
            "sma_20", "sma_50", "ema_12", "ema_26", "rsi", "macd", "macd_signal", 
            "macd_histogram", "bollinger_upper", "bollinger_lower", "bollinger_width", 
            "atr", "volume_sma", "volume_ratio", "obv", "price_momentum", "volatility_score"
        ]
        
        market_names = [
            "fear_greed_index", "is_bull_market", "is_high_volatility", "btc_correlation",
            "eth_correlation", "market_beta", "social_score", "mention_volume_normalized",
            "sentiment_trend"
        ]
        
        token_names = [
            "log_price", "log_market_cap", "log_volume", "price_change_24h",
            "is_ethereum", "is_solana", "is_base", "token_age_days", "token_quality_score"
        ]
        
        return tech_names + market_names + token_names
    
    def clear_cache(self):
        """Clear all cached indicators and features"""
        self._indicator_cache.clear()
        self._market_cache = None
        self.logger.info("Feature cache cleared")