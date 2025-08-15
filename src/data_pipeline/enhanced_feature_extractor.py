"""
Enhanced Feature Extractor for Training-Ready Data
Calculates ALL features needed by ML models during corpus collection
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime


class EnhancedFeatureExtractor:
    """
    Extracts all features needed for ML training during corpus collection.
    This eliminates the need for a middle layer during training.
    """
    
    @staticmethod
    def extract_all_features(ohlcv_data: pd.DataFrame, 
                            timestamp: datetime = None) -> Dict[str, float]:
        """
        Extract ALL features that any model might need.
        This is called during corpus collection to store training-ready data.
        
        Args:
            ohlcv_data: Window of OHLCV data for feature calculation
            timestamp: Current timestamp (for time features)
            
        Returns:
            Dictionary with all calculated features
        """
        features = {}
        
        if ohlcv_data.empty:
            return EnhancedFeatureExtractor._get_complete_default_features()
        
        # Get the last row for current values
        current = ohlcv_data.iloc[-1]
        close_series = ohlcv_data['close']
        high_series = ohlcv_data['high']
        low_series = ohlcv_data['low']
        volume_series = ohlcv_data['volume']
        
        # ========== Time Features ==========
        if timestamp:
            features['hour'] = timestamp.hour
            features['day_of_week'] = timestamp.dayofweek
            features['month'] = timestamp.month
            features['quarter'] = (timestamp.month - 1) // 3 + 1
            features['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
            features['trading_session'] = EnhancedFeatureExtractor._get_trading_session(timestamp.hour)
            # Cyclical encoding for time
            features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
        else:
            # Default time features
            for feat in ['hour', 'day_of_week', 'month', 'quarter', 'is_weekend', 
                        'trading_session', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']:
                features[feat] = 0
        
        # ========== Price-Based Features ==========
        features['price_change'] = close_series.pct_change().iloc[-1] if len(close_series) > 1 else 0
        features['high_low_ratio'] = current['high'] / current['low'] if current['low'] > 0 else 1
        features['close_to_high'] = current['close'] / current['high'] if current['high'] > 0 else 1
        features['close_to_low'] = current['close'] / current['low'] if current['low'] > 0 else 1
        features['volume_price_ratio'] = current['volume'] / current['close'] if current['close'] > 0 else 0
        
        # Price momentum at different scales
        for period in [5, 10, 20]:
            if len(close_series) > period:
                features[f'momentum_{period}'] = (current['close'] / close_series.iloc[-period-1] - 1) * 100
            else:
                features[f'momentum_{period}'] = 0
        
        # ========== Volatility Features ==========
        if len(close_series) > 2:
            returns = close_series.pct_change().dropna()
            features['volatility'] = returns.std() if len(returns) > 0 else 0
            features['volatility_ratio'] = features['volatility'] / returns.rolling(20).std().mean() if len(returns) > 20 else 1
            features['realized_volatility'] = np.sqrt(252) * returns.std() if len(returns) > 0 else 0
        else:
            features['volatility'] = 0
            features['volatility_ratio'] = 1
            features['realized_volatility'] = 0
        
        # Volatility at different time scales
        for period in [5, 10, 20]:
            if len(close_series) > period:
                features[f'volatility_{period}'] = close_series.pct_change().tail(period).std()
            else:
                features[f'volatility_{period}'] = 0
        
        # ========== Technical Indicators (Complete Set) ==========
        
        # RSI with multiple periods
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = EnhancedFeatureExtractor._calculate_rsi(close_series, period)
        
        # Moving Averages (SMA)
        for period in [5, 10, 20, 50, 100, 200]:
            if len(close_series) >= period:
                features[f'sma_{period}'] = close_series.rolling(period).mean().iloc[-1]
                features[f'sma_{period}_ratio'] = current['close'] / features[f'sma_{period}']
            else:
                features[f'sma_{period}'] = current['close']
                features[f'sma_{period}_ratio'] = 1
        
        # Exponential Moving Averages (EMA)
        for period in [8, 12, 21, 26, 50, 200]:
            if len(close_series) >= period:
                features[f'ema_{period}'] = close_series.ewm(span=period).mean().iloc[-1]
                features[f'ema_{period}_ratio'] = current['close'] / features[f'ema_{period}']
            else:
                features[f'ema_{period}'] = current['close']
                features[f'ema_{period}_ratio'] = 1
        
        # MACD
        if len(close_series) >= 26:
            ema_12 = close_series.ewm(span=12).mean()
            ema_26 = close_series.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            features['macd'] = macd_line.iloc[-1]
            features['macd_signal'] = signal_line.iloc[-1]
            features['macd_histogram'] = features['macd'] - features['macd_signal']
        else:
            features['macd'] = 0
            features['macd_signal'] = 0
            features['macd_histogram'] = 0
        
        # Bollinger Bands
        for period in [10, 20]:
            if len(close_series) >= period:
                sma = close_series.rolling(period).mean().iloc[-1]
                std = close_series.rolling(period).std().iloc[-1]
                features[f'bb_upper_{period}'] = sma + (2 * std)
                features[f'bb_lower_{period}'] = sma - (2 * std)
                features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
                features[f'bb_position_{period}'] = (current['close'] - features[f'bb_lower_{period}']) / features[f'bb_width_{period}'] if features[f'bb_width_{period}'] > 0 else 0.5
            else:
                features[f'bb_upper_{period}'] = current['close']
                features[f'bb_lower_{period}'] = current['close']
                features[f'bb_width_{period}'] = 0
                features[f'bb_position_{period}'] = 0.5
        
        # ATR (Average True Range)
        features['atr'] = EnhancedFeatureExtractor._calculate_atr(ohlcv_data)
        
        # Stochastic Oscillator
        if len(ohlcv_data) >= 14:
            lowest_low = low_series.tail(14).min()
            highest_high = high_series.tail(14).max()
            if highest_high > lowest_low:
                features['stoch_k'] = ((current['close'] - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                features['stoch_k'] = 50
            # Stochastic D is 3-period SMA of K
            if len(ohlcv_data) >= 17:
                k_values = []
                for i in range(3):
                    idx = -(i+1)
                    ll = low_series.iloc[idx-13:idx+1].min()
                    hh = high_series.iloc[idx-13:idx+1].max()
                    if hh > ll:
                        k_values.append(((close_series.iloc[idx] - ll) / (hh - ll)) * 100)
                    else:
                        k_values.append(50)
                features['stoch_d'] = np.mean(k_values)
            else:
                features['stoch_d'] = features['stoch_k']
        else:
            features['stoch_k'] = 50
            features['stoch_d'] = 50
        
        # ========== Volume Indicators ==========
        
        # Volume moving averages
        for period in [10, 20, 50]:
            if len(volume_series) >= period:
                features[f'volume_sma_{period}'] = volume_series.rolling(period).mean().iloc[-1]
                features[f'volume_ratio_{period}'] = current['volume'] / features[f'volume_sma_{period}'] if features[f'volume_sma_{period}'] > 0 else 1
            else:
                features[f'volume_sma_{period}'] = current['volume']
                features[f'volume_ratio_{period}'] = 1
        
        # OBV (On-Balance Volume)
        features['obv'] = EnhancedFeatureExtractor._calculate_obv(ohlcv_data)
        features['obv_sma'] = features['obv']  # Simplified for now
        
        # Volume-Weighted Average Price (VWAP)
        if len(ohlcv_data) > 0:
            typical_price = (high_series + low_series + close_series) / 3
            features['vwap'] = (typical_price * volume_series).sum() / volume_series.sum() if volume_series.sum() > 0 else current['close']
            features['vwap_ratio'] = current['close'] / features['vwap'] if features['vwap'] > 0 else 1
        else:
            features['vwap'] = current['close']
            features['vwap_ratio'] = 1
        
        # ========== Pattern Recognition Features ==========
        
        # Support and Resistance levels
        if len(ohlcv_data) >= 20:
            features['resistance_level'] = high_series.tail(20).max()
            features['support_level'] = low_series.tail(20).min()
            features['distance_from_resistance'] = (features['resistance_level'] - current['close']) / current['close']
            features['distance_from_support'] = (current['close'] - features['support_level']) / current['close']
        else:
            features['resistance_level'] = current['high']
            features['support_level'] = current['low']
            features['distance_from_resistance'] = 0
            features['distance_from_support'] = 0
        
        # ========== Market Regime Features ==========
        features['market_regime'] = EnhancedFeatureExtractor._classify_market_regime(ohlcv_data)
        features['trend_strength'] = EnhancedFeatureExtractor._calculate_trend_strength(close_series)
        
        # ========== Normalized Scores ==========
        features['volatility_score'] = min(features['volatility'] / 0.5, 1.0) if features['volatility'] > 0 else 0
        features['volume_score'] = min(features.get('volume_ratio_20', 1), 2.0) / 2.0
        features['momentum_score'] = np.tanh(features.get('momentum_20', 0) / 100)
        
        return features
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        if loss.iloc[-1] == 0:
            return 100.0
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi) if not pd.isna(rsi) else 50.0
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < 2:
            return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=min(period, len(true_range))).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    @staticmethod
    def _calculate_obv(df: pd.DataFrame) -> float:
        """Calculate On-Balance Volume"""
        if len(df) < 2:
            return 0.0
        
        obv = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv += df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv -= df['volume'].iloc[i]
        
        return float(obv)
    
    @staticmethod
    def _get_trading_session(hour: int) -> int:
        """Get trading session from hour"""
        if 0 <= hour < 6:
            return 0  # Asian
        elif 6 <= hour < 12:
            return 1  # European
        elif 12 <= hour < 18:
            return 2  # US
        else:
            return 3  # After hours
    
    @staticmethod
    def _classify_market_regime(df: pd.DataFrame) -> int:
        """Classify market regime based on price action"""
        if len(df) < 20:
            return 1  # Neutral
        
        close = df['close']
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        
        current_price = close.iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return 2  # Bullish
        elif current_price < sma_20 < sma_50:
            return 0  # Bearish
        else:
            return 1  # Neutral
    
    @staticmethod
    def _calculate_trend_strength(prices: pd.Series) -> float:
        """Calculate trend strength using linear regression slope"""
        if len(prices) < 10:
            return 0.0
        
        recent = prices.tail(20) if len(prices) >= 20 else prices
        x = np.arange(len(recent))
        y = recent.values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to [-1, 1]
        normalized = np.tanh(slope / recent.mean() * 100)
        
        return float(normalized)
    
    @staticmethod
    def _get_complete_default_features() -> Dict[str, float]:
        """Get complete dictionary of default feature values"""
        defaults = {}
        
        # Time features
        for feat in ['hour', 'day_of_week', 'month', 'quarter', 'is_weekend', 
                    'trading_session', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']:
            defaults[feat] = 0
        
        # Price features
        for feat in ['price_change', 'high_low_ratio', 'close_to_high', 'close_to_low', 
                    'volume_price_ratio']:
            defaults[feat] = 0 if 'change' in feat else 1
        
        # Momentum
        for period in [5, 10, 20]:
            defaults[f'momentum_{period}'] = 0
        
        # Volatility
        for feat in ['volatility', 'realized_volatility']:
            defaults[feat] = 0
        defaults['volatility_ratio'] = 1
        for period in [5, 10, 20]:
            defaults[f'volatility_{period}'] = 0
        
        # RSI
        for period in [7, 14, 21]:
            defaults[f'rsi_{period}'] = 50
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            defaults[f'sma_{period}'] = 0
            defaults[f'sma_{period}_ratio'] = 1
        
        for period in [8, 12, 21, 26, 50, 200]:
            defaults[f'ema_{period}'] = 0
            defaults[f'ema_{period}_ratio'] = 1
        
        # MACD
        defaults['macd'] = 0
        defaults['macd_signal'] = 0
        defaults['macd_histogram'] = 0
        
        # Bollinger Bands
        for period in [10, 20]:
            defaults[f'bb_upper_{period}'] = 0
            defaults[f'bb_lower_{period}'] = 0
            defaults[f'bb_width_{period}'] = 0
            defaults[f'bb_position_{period}'] = 0.5
        
        # Other indicators
        defaults['atr'] = 0
        defaults['stoch_k'] = 50
        defaults['stoch_d'] = 50
        
        # Volume
        for period in [10, 20, 50]:
            defaults[f'volume_sma_{period}'] = 0
            defaults[f'volume_ratio_{period}'] = 1
        
        defaults['obv'] = 0
        defaults['obv_sma'] = 0
        defaults['vwap'] = 0
        defaults['vwap_ratio'] = 1
        
        # Pattern recognition
        defaults['resistance_level'] = 0
        defaults['support_level'] = 0
        defaults['distance_from_resistance'] = 0
        defaults['distance_from_support'] = 0
        
        # Market regime
        defaults['market_regime'] = 1
        defaults['trend_strength'] = 0
        
        # Scores
        defaults['volatility_score'] = 0
        defaults['volume_score'] = 0.5
        defaults['momentum_score'] = 0
        
        return defaults