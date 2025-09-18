"""
Feature Engineering Module
Technical indicators and market features for ML models
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import structlog

from src.discovery.base import DiscoveredToken
from src.utils.base import Chain
from .base import TechnicalIndicators, MarketFeatures, FeatureEngineeringError
from .market_data_aggregator import MarketDataAggregator


logger = structlog.get_logger()


class FeatureEngineer:
    """Feature engineering for ML models"""
    
    def __init__(self, 
                 cache_ttl_minutes: int = 30,
                 coingecko_api_key: Optional[str] = None,
                 enable_live_data: bool = True,
                 enable_token_normalization: bool = True,
                 enable_advanced_features: bool = True):
        self.cache_ttl_minutes = cache_ttl_minutes
        self.enable_live_data = enable_live_data
        self.enable_token_normalization = enable_token_normalization
        self.enable_advanced_features = enable_advanced_features
        self._indicator_cache: Dict[str, Tuple[datetime, TechnicalIndicators]] = {}
        self._market_cache: Optional[Tuple[datetime, MarketFeatures]] = None
        self.logger = structlog.get_logger().bind(component="FeatureEngineer")
        
        # Token-specific normalization scalers
        self._token_scalers: Dict[str, Dict[str, Any]] = {}
        self._token_price_ranges: Dict[str, Tuple[float, float]] = {
            'BTC': (10000, 100000),
            'ETH': (500, 10000),
            'WBTC': (10000, 100000),
            'SOL': (1, 500),
            'PEPE': (0.000001, 0.001),
            'SHIB': (0.000001, 0.001),
            'DOGE': (0.01, 1.0),
        }
        
        # Initialize market data aggregator for live data
        if self.enable_live_data:
            self.market_aggregator = MarketDataAggregator(
                coingecko_api_key=coingecko_api_key,
                cache_ttl=cache_ttl_minutes * 60  # Convert to seconds
            )
        else:
            self.market_aggregator = None
    
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
    
    async def calculate_market_features(self, primary_chain: Optional[str] = None) -> MarketFeatures:
        """
        Calculate market-wide features using live data APIs
        
        Args:
            primary_chain: Primary chain for on-chain metrics (defaults to the most relevant chain)
        
        Returns:
            MarketFeatures object with market context
        """
        # Check cache first
        if self._market_cache:
            cached_time, cached_features = self._market_cache
            if datetime.now() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                return cached_features
        
        try:
            if self.enable_live_data and self.market_aggregator:
                # Use live market data from APIs
                from src.utils.base import Chain
                
                # Determine primary chain for on-chain metrics
                chain = Chain.ETHEREUM  # Default
                if primary_chain:
                    try:
                        chain = Chain(primary_chain.lower())
                    except ValueError:
                        self.logger.warning("Invalid chain specified, using Ethereum", chain=primary_chain)
                
                features = await self.market_aggregator.get_market_features(primary_chain=chain)
                
                self.logger.info("Live market features retrieved", 
                               fear_greed=features.fear_greed_index,
                               btc_dominance=features.btc_dominance,
                               defi_tvl=features.total_value_locked)
            else:
                # Live data is disabled, raise error
                raise FeatureEngineeringError("Market data aggregator not available - enable_live_data is False")
            
            # Cache the result
            self._market_cache = (datetime.now(), features)
            
            return features
            
        except Exception as e:
            self.logger.error("Market feature calculation failed", error=str(e))
            # Always raise the error - no fallback to placeholder data
            raise FeatureEngineeringError(f"Failed to calculate market features: {str(e)}")
    
    def normalize_features_by_token(self, df: pd.DataFrame, token_symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Apply token-specific normalization to features.
        
        Args:
            df: DataFrame with features to normalize
            token_symbol: Token symbol for specific normalization
        
        Returns:
            Normalized DataFrame
        """
        if not self.enable_token_normalization:
            return df
        
        # Infer token from price if not provided
        if token_symbol is None and 'close' in df.columns:
            avg_price = df['close'].mean()
            if avg_price > 10000:
                token_symbol = 'WBTC'
            elif avg_price > 1000:
                token_symbol = 'ETH'
            elif avg_price < 0.001:
                token_symbol = 'PEPE'
            else:
                token_symbol = 'UNKNOWN'
            
            self.logger.info(f"Inferred token '{token_symbol}' from avg price {avg_price:.8f}")
        
        # Apply normalization based on token
        if token_symbol and token_symbol in self._token_price_ranges:
            min_price, max_price = self._token_price_ranges[token_symbol]
            
            # Normalize price features
            price_cols = ['close', 'open', 'high', 'low', 'sma_20', 'sma_50', 
                         'ema_12', 'ema_26', 'bollinger_upper', 'bollinger_lower']
            
            for col in price_cols:
                if col in df.columns:
                    df[col] = (df[col] - min_price) / (max_price - min_price)
            
            # Log-normalize volume
            if 'volume' in df.columns:
                df['volume'] = np.log1p(df['volume'])
        
        return df
    
    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced features including microstructure and regime detection.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with additional advanced features
        """
        if not self.enable_advanced_features:
            return df
        
        result = df.copy()
        
        # Microstructure features
        if all(col in df.columns for col in ['close', 'volume']):
            # Kyle's Lambda (price impact)
            returns = df['close'].pct_change()
            signed_volume = df['volume'] * np.sign(returns)
            result['kyle_lambda'] = (returns.rolling(20).std() / 
                                    (signed_volume.rolling(20).std() + 1e-8)).fillna(0)
            
            # Amihud illiquidity
            returns_abs = returns.abs()
            dollar_volume = df['close'] * df['volume']
            result['amihud_illiquidity'] = (returns_abs / (dollar_volume + 1e-8)).rolling(20).mean().fillna(0)
        
        # Volatility regime features
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # GARCH-like volatility
            result['volatility_regime'] = returns.rolling(20).std().fillna(0)
            
            # Volatility of volatility
            result['vol_of_vol'] = result['volatility_regime'].rolling(20).std().fillna(0)
            
            # Trend strength
            result['trend_strength'] = df['close'].rolling(20).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (x.mean() + 1e-8)
            ).fillna(0)
        
        # Support/Resistance levels
        if 'close' in df.columns:
            result['resistance_20'] = df['close'].rolling(20).max()
            result['support_20'] = df['close'].rolling(20).min()
            result['price_position'] = ((df['close'] - result['support_20']) / 
                                       (result['resistance_20'] - result['support_20'] + 1e-8)).fillna(0.5)
        
        # Order flow imbalance
        if all(col in df.columns for col in ['high', 'low', 'close']):
            mid_price = (df['high'] + df['low']) / 2
            result['order_flow_imbalance'] = ((df['close'] - mid_price) / 
                                             (df['high'] - df['low'] + 1e-8)).fillna(0)
        
        return result
    
    def augment_training_data(self, df: pd.DataFrame, augmentation_factor: float = 0.3) -> pd.DataFrame:
        """
        Apply data augmentation techniques for training.
        
        Args:
            df: DataFrame with training data
            augmentation_factor: Fraction of data to augment (0.3 = 30% more data)
        
        Returns:
            Augmented DataFrame
        """
        if augmentation_factor <= 0:
            return df
        
        n_augment = int(len(df) * augmentation_factor)
        augmented_rows = []
        
        for _ in range(n_augment):
            # Random row selection
            idx = np.random.randint(0, len(df))
            row = df.iloc[idx].copy()
            
            # Add small noise to numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['timestamp', 'hour', 'day_of_week', 'month']:
                    noise = np.random.normal(0, 0.01 * row[col] if row[col] != 0 else 0.01)
                    row[col] += noise
            
            # Ensure price relationships are maintained
            if all(col in row.index for col in ['high', 'low', 'open', 'close']):
                row['high'] = max(row['high'], row['open'], row['close'])
                row['low'] = min(row['low'], row['open'], row['close'])
            
            augmented_rows.append(row)
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            result = pd.concat([df, augmented_df], ignore_index=True)
            self.logger.info(f"Augmented data from {len(df)} to {len(result)} samples")
            return result
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Improved missing value handling with multiple strategies.
        
        Args:
            df: DataFrame with potential missing values
            method: 'interpolate', 'forward_fill', 'mean', or 'drop'
        
        Returns:
            DataFrame with handled missing values
        """
        result = df.copy()
        
        if method == 'interpolate':
            # Check if we need to set timestamp as index for time interpolation
            needs_reset = False
            
            # Handle timestamp column if present
            if 'timestamp' in result.columns and not isinstance(result.index, pd.DatetimeIndex):
                # Convert timestamp to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
                    result['timestamp'] = pd.to_datetime(result['timestamp'])
                result = result.set_index('timestamp')
                needs_reset = True
            
            # Perform interpolation based on index type
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            if isinstance(result.index, pd.DatetimeIndex):
                # Use time-weighted interpolation with DatetimeIndex
                result[numeric_cols] = result[numeric_cols].interpolate(method='time', limit_direction='both')
            else:
                # Fall back to linear interpolation if no DatetimeIndex
                result[numeric_cols] = result[numeric_cols].interpolate(method='linear', limit_direction='both')
            
            # Reset index if we set it temporarily
            if needs_reset:
                result = result.reset_index()
        
        elif method == 'forward_fill':
            # Forward fill then backward fill
            result = result.fillna(method='ffill').fillna(method='bfill')
        
        elif method == 'mean':
            # Fill with rolling mean
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                rolling_mean = result[col].rolling(window=10, min_periods=1).mean()
                result[col] = result[col].fillna(rolling_mean)
        
        elif method == 'drop':
            # Drop rows with any missing values
            result = result.dropna()
        
        # Final safety fill with zeros for any remaining NaN
        result = result.fillna(0)
        
        return result
    
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
    
    def calculate_features_for_corpus(self, ohlcv_data: pd.DataFrame, 
                                     min_periods: bool = True,
                                     timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calculate ALL features needed for ML training - complete training-ready dataset
        Handles limited data gracefully with min_periods support
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns
            min_periods: Whether to use minimum periods for calculations
            timestamp: Current timestamp for time-based features
        
        Returns:
            Dictionary of ALL feature names to values (training-ready)
        """
        features = {}
        
        if ohlcv_data.empty:
            return self._get_complete_training_features()
        
        close = ohlcv_data['close']
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        volume = ohlcv_data['volume']
        current = ohlcv_data.iloc[-1]
        
        # ========== TIME FEATURES (for ML models) ==========
        if timestamp:
            features['hour'] = timestamp.hour
            features['day_of_week'] = timestamp.dayofweek
            features['month'] = timestamp.month
            features['quarter'] = (timestamp.month - 1) // 3 + 1
            features['is_weekend'] = 1 if timestamp.dayofweek >= 5 else 0
            features['trading_session'] = self._get_trading_session(timestamp.hour)
            # Cyclical encoding for neural networks
            features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
        else:
            # Default time features
            for feat in ['hour', 'day_of_week', 'month', 'quarter', 'is_weekend', 
                        'trading_session', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']:
                features[feat] = 0
        
        # ========== PRICE-BASED FEATURES ==========
        features['price_change'] = close.pct_change().iloc[-1] if len(close) > 1 else 0
        features['high_low_ratio'] = current['high'] / current['low'] if current['low'] > 0 else 1
        features['close_to_high'] = current['close'] / current['high'] if current['high'] > 0 else 1
        features['close_to_low'] = current['close'] / current['low'] if current['low'] > 0 else 1
        features['volume_price_ratio'] = current['volume'] / current['close'] if current['close'] > 0 else 0
        
        # Price momentum at different scales
        for period in [5, 10, 20]:
            if len(close) > period:
                features[f'momentum_{period}'] = (current['close'] / close.iloc[-period-1] - 1) * 100
            else:
                features[f'momentum_{period}'] = 0
        
        # RSI - multiple periods for different trading strategies
        for rsi_period in [7, 14, 21]:
            period = min(rsi_period, len(close) - 1) if min_periods and len(close) > 1 else rsi_period
            if len(close) > period:
                features[f'rsi_{rsi_period}'] = self._calculate_rsi(close, period) or 50.0
            else:
                features[f'rsi_{rsi_period}'] = 50.0
        
        # MACD
        if len(close) >= 26:
            macd_vals = self._calculate_macd(close)
            features['macd'] = macd_vals['macd'] or 0.0
            features['macd_signal'] = macd_vals['signal'] or 0.0
            features['macd_histogram'] = macd_vals['histogram'] or 0.0
        else:
            features['macd'] = 0.0
            features['macd_signal'] = 0.0
            features['macd_histogram'] = 0.0
        
        # Bollinger Bands - adaptive period with position
        bb_period = min(20, len(close)) if min_periods else 20
        if len(close) >= max(2, bb_period):
            bb_vals = self._calculate_bollinger_bands(close, bb_period)
            features['bb_upper'] = bb_vals['upper'] or 0.0
            features['bb_middle'] = float(close.rolling(window=bb_period).mean().iloc[-1]) if len(close) >= bb_period else 0.0
            features['bb_lower'] = bb_vals['lower'] or 0.0
            features['bb_width'] = bb_vals['width'] or 0.0
            # Add Bollinger position for ML models
            if features['bb_width'] > 0:
                features['bb_position'] = (current['close'] - features['bb_lower']) / features['bb_width']
            else:
                features['bb_position'] = 0.5
        else:
            features['bb_upper'] = 0.0
            features['bb_middle'] = 0.0
            features['bb_lower'] = 0.0
            features['bb_width'] = 0.0
            features['bb_position'] = 0.5
        
        # ATR - adaptive period
        atr_period = min(14, len(ohlcv_data) - 1) if min_periods and len(ohlcv_data) > 1 else 14
        if len(ohlcv_data) > atr_period:
            features['atr'] = self._calculate_atr(ohlcv_data, atr_period) or 0.0
        else:
            features['atr'] = 0.0
        
        # Moving averages - calculate what's possible
        for period, name_prefix in [(20, 'sma_20'), (50, 'sma_50'), (200, 'sma_200')]:
            if len(close) >= period:
                features[name_prefix] = self._calculate_sma(close, period) or 0.0
            else:
                features[name_prefix] = 0.0
        
        for period, name_prefix in [(12, 'ema_12'), (26, 'ema_26'), (50, 'ema_50'), (200, 'ema_200')]:
            if len(close) >= period:
                features[name_prefix] = self._calculate_ema(close, period) or 0.0
            else:
                features[name_prefix] = 0.0
        
        # Volume indicators
        vol_period = min(20, len(volume)) if min_periods else 20
        if len(volume) >= vol_period:
            features['volume_sma_20'] = self._calculate_sma(volume, vol_period) or 0.0
            features['volume_ratio'] = self._calculate_volume_ratio(volume, vol_period) or 1.0
        else:
            features['volume_sma_20'] = 0.0
            features['volume_ratio'] = 1.0
        
        # OBV
        if len(close) >= 2:
            features['obv'] = self._calculate_obv(close, volume) or 0.0
        else:
            features['obv'] = 0.0
        
        # ADX (placeholder - not implemented in base class yet)
        features['adx'] = 0.0
        
        # CCI (placeholder)
        features['cci'] = 0.0
        
        # Stochastic (placeholder)
        features['stoch_k'] = 0.0
        features['stoch_d'] = 0.0
        
        # Williams %R (placeholder)
        features['williams_r'] = 0.0
        
        # Volume EMA
        if len(volume) >= 12:
            features['volume_ema'] = self._calculate_ema(volume, 12) or 0.0
        else:
            features['volume_ema'] = 0.0
        
        # Returns and price changes
        if len(close) > 1:
            features['returns_1h'] = float((close.iloc[-1] / close.iloc[-2] - 1))
            features['price_change_1h'] = float(close.iloc[-1] - close.iloc[-2])
        else:
            features['returns_1h'] = 0.0
            features['price_change_1h'] = 0.0
        
        if len(close) > 24:
            features['returns_24h'] = float((close.iloc[-1] / close.iloc[-24] - 1))
            features['price_change_24h'] = float(close.iloc[-1] - close.iloc[-24])
            features['volatility_24h'] = float(close.pct_change().tail(24).std())
        else:
            features['returns_24h'] = 0.0
            features['price_change_24h'] = 0.0
            features['volatility_24h'] = 0.0
        
        if len(close) > 168:  # 7 days
            features['returns_7d'] = float((close.iloc[-1] / close.iloc[-168] - 1))
            features['price_change_7d'] = float(close.iloc[-1] - close.iloc[-168])
        else:
            features['returns_7d'] = 0.0
            features['price_change_7d'] = 0.0
        
        # 4h price change (placeholder for now)
        features['price_change_4h'] = 0.0
        
        # 1h volatility
        if len(close) > 2:
            features['volatility_1h'] = float(close.pct_change().tail(2).std())
        else:
            features['volatility_1h'] = 0.0
        
        # ========== VOLATILITY FEATURES ==========
        if len(close) > 2:
            returns = close.pct_change().dropna()
            features['volatility'] = returns.std() if len(returns) > 0 else 0
            features['volatility_ratio'] = features['volatility'] / returns.rolling(20).std().mean() if len(returns) > 20 else 1
            features['realized_volatility'] = np.sqrt(252) * returns.std() if len(returns) > 0 else 0
        else:
            features['volatility'] = 0
            features['volatility_ratio'] = 1
            features['realized_volatility'] = 0
        
        # ========== NORMALIZED SCORES FOR ML ==========
        features['volatility_score'] = min(features.get('volatility', 0) / 0.5, 1.0) if features.get('volatility', 0) > 0 else 0
        features['volume_score'] = min(features.get('volume_ratio', 1), 2.0) / 2.0
        features['momentum_score'] = np.tanh(features.get('momentum_20', 0) / 100)
        features['price_momentum'] = features.get('momentum_10', 0)  # Alias for compatibility
        
        # ========== MARKET REGIME (simplified) ==========
        features['market_regime'] = self._classify_market_regime(close)
        features['trend_strength'] = self._calculate_trend_strength(close)
        
        return features
    
    def _get_trading_session(self, hour: int) -> int:
        """Classify hour into trading session"""
        if 0 <= hour < 6:
            return 0  # Asian session
        elif 6 <= hour < 12:
            return 1  # European session
        elif 12 <= hour < 18:
            return 2  # US session
        else:
            return 3  # After hours
    
    def _classify_market_regime(self, prices: pd.Series) -> int:
        """Classify market regime based on price action"""
        if len(prices) < 20:
            return 1  # Neutral
        
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else sma_20
        current_price = prices.iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return 2  # Bullish
        elif current_price < sma_20 < sma_50:
            return 0  # Bearish
        else:
            return 1  # Neutral
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression slope"""
        if len(prices) < 10:
            return 0.0
        
        recent = prices.tail(20) if len(prices) >= 20 else prices
        x = np.arange(len(recent))
        y = recent.values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to [-1, 1]
        normalized = np.tanh(slope / recent.mean() * 100) if recent.mean() > 0 else 0
        
        return float(normalized)
    
    def _get_complete_training_features(self) -> Dict[str, float]:
        """Get complete dictionary of ALL training-ready feature defaults"""
        return self._get_default_feature_dict()  # Will be expanded
    
    def _get_default_feature_dict(self) -> Dict[str, float]:
        """Get dictionary with ALL training-ready features set to default values"""
        defaults = {
            # Time features
            'hour': 0, 'day_of_week': 0, 'month': 0, 'quarter': 0, 'is_weekend': 0,
            'trading_session': 0, 'hour_sin': 0, 'hour_cos': 0, 'day_sin': 0, 'day_cos': 0,
            
            # Price features
            'price_change': 0, 'high_low_ratio': 1, 'close_to_high': 1, 'close_to_low': 1,
            'volume_price_ratio': 0,
            
            # Momentum
            'momentum_5': 0, 'momentum_10': 0, 'momentum_20': 0,
            
            # RSI variants
            'rsi_7': 50.0, 'rsi_14': 50.0, 'rsi_21': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'bb_upper': 0.0,
            'bb_middle': 0.0,
            'bb_lower': 0.0,
            'bb_width': 0.0,
            'bb_position': 0.5,
            'atr': 0.0,
            'sma_20': 0.0,
            'sma_50': 0.0,
            'sma_200': 0.0,
            'ema_12': 0.0,
            'ema_26': 0.0,
            'ema_50': 0.0,
            'ema_200': 0.0,
            'volume_sma_20': 0.0,
            'volume_ratio': 1.0,
            'volume_ema': 0.0,
            'obv': 0.0,
            'adx': 0.0,
            'cci': 0.0,
            'stoch_k': 0.0,
            'stoch_d': 0.0,
            'williams_r': 0.0,
            'returns_1h': 0.0,
            'returns_24h': 0.0,
            'returns_7d': 0.0,
            'volatility_24h': 0.0,
            'volatility_1h': 0.0,
            'price_change_1h': 0.0,
            'price_change_4h': 0.0,
            'price_change_24h': 0.0,
            'price_change_7d': 0.0,
            
            # Additional volatility features
            'volatility': 0.0,
            'volatility_ratio': 1.0,
            'realized_volatility': 0.0,
            
            # ML scores
            'volatility_score': 0.0,
            'volume_score': 0.5,
            'momentum_score': 0.0,
            'price_momentum': 0.0,
            
            # Market regime
            'market_regime': 1,
            'trend_strength': 0.0
        }
        
        return defaults
    
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
            # Market sentiment features
            "fear_greed_index", "is_bull_market", "is_bear_market", "is_high_volatility", "is_low_volatility",
            
            # Correlation and dominance features  
            "btc_correlation", "eth_correlation", "btc_dominance", "eth_dominance", "stablecoin_dominance", "market_beta",
            
            # DeFi ecosystem features
            "total_value_locked_normalized", "tvl_change_24h", "tvl_change_7d", "defi_dominance", "active_protocols_normalized",
            
            # On-chain activity features
            "transaction_count_normalized", "active_addresses_normalized", "transaction_volume_normalized", 
            "network_fees_normalized", "whale_activity_score",
            
            # Social sentiment features
            "social_score", "mention_volume_normalized", "sentiment_trend", "influencer_sentiment"
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
        
        # Clear market data aggregator cache if available
        if self.market_aggregator:
            self.market_aggregator.clear_cache()
        
        self.logger.info("Feature cache cleared")
    
    def create_transformer_sequences(self, token: DiscoveredToken, price_data: pd.DataFrame, 
                                   sequence_length: int = 60, prediction_horizons: List[int] = [1, 4, 24]) -> dict:
        """
        Create sequence-based features for Transformer models
        
        Args:
            token: Token to analyze
            price_data: Historical price data
            sequence_length: Length of input sequences
            prediction_horizons: Prediction horizons in hours
            
        Returns:
            Dictionary containing input sequences and targets
        """
        try:
            if len(price_data) < sequence_length + max(prediction_horizons):
                self.logger.warning("Insufficient data for sequence creation", 
                                  data_length=len(price_data),
                                  required_length=sequence_length + max(prediction_horizons))
                return {'input_sequences': np.array([]), 'target_sequences': np.array([])}
            
            sequences = []
            targets = []
            
            # Create sliding windows
            for i in range(sequence_length, len(price_data) - max(prediction_horizons)):
                # Input sequence
                seq_data = price_data.iloc[i-sequence_length:i].copy()
                
                # Basic features for each timestep
                seq_features = []
                for _, row in seq_data.iterrows():
                    features = [
                        row.get('close', 0) / 100.0,  # Normalized price
                        row.get('volume', 0) / 1000000.0,  # Normalized volume
                        row.get('high', 0) / 100.0,
                        row.get('low', 0) / 100.0,
                        row.get('open', 0) / 100.0,
                    ]
                    seq_features.append(features)
                
                sequences.append(seq_features)
                
                # Target values (price changes for different horizons)
                current_price = price_data.iloc[i]['close']
                horizon_targets = []
                for h in prediction_horizons:
                    if i + h < len(price_data):
                        future_price = price_data.iloc[i + h]['close']
                        price_change = (future_price - current_price) / current_price
                        horizon_targets.append(price_change)
                    else:
                        horizon_targets.append(0.0)
                
                targets.append(horizon_targets)
            
            input_sequences = np.array(sequences, dtype=np.float32)
            target_sequences = np.array(targets, dtype=np.float32)
            
            self.logger.info("Transformer sequences created", 
                           input_shape=input_sequences.shape,
                           target_shape=target_sequences.shape)
            
            return {
                'input_sequences': input_sequences,
                'target_sequences': target_sequences,
                'sequence_length': sequence_length,
                'prediction_horizons': prediction_horizons
            }
            
        except Exception as e:
            self.logger.error("Failed to create transformer sequences", error=str(e))
            return {'input_sequences': np.array([]), 'target_sequences': np.array([])}
    
    def normalize_for_attention(self, indicators: TechnicalIndicators, 
                              market_features: MarketFeatures) -> np.ndarray:
        """
        Normalize features for attention mechanisms
        
        Args:
            indicators: Technical indicators
            market_features: Market features
            
        Returns:
            Normalized feature array suitable for attention
        """
        try:
            # Get feature vectors
            tech_vector = indicators.to_feature_vector()
            market_vector = market_features.to_feature_vector()
            
            # Combine features
            combined = np.concatenate([tech_vector, market_vector])
            
            # Apply attention-friendly normalization
            # 1. Replace NaN/Inf with zeros
            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 2. Clip extreme values
            combined = np.clip(combined, -10.0, 10.0)
            
            # 3. Apply tanh scaling to keep values in [-1, 1] range
            combined = np.tanh(combined)
            
            return combined
            
        except Exception as e:
            self.logger.error("Failed to normalize features for attention", error=str(e))
            # Return default normalized vector
            return np.zeros(50, dtype=np.float32)  # Reasonable default size
    
    def validate_sequence_length(self, sequence_length: int, model_type: str) -> bool:
        """
        Validate sequence length for different Transformer model types
        
        Args:
            sequence_length: Proposed sequence length
            model_type: Type of transformer model
            
        Returns:
            True if valid, False otherwise
        """
        model_requirements = {
            'transformer': {'min_seq_len': 10, 'max_seq_len': 1000},
            'itransformer': {'min_seq_len': 20, 'max_seq_len': 500},
            'patchtst': {'min_seq_len': 32, 'max_seq_len': 1000}
        }
        
        if model_type not in model_requirements:
            return True  # Unknown model type, assume valid
        
        req = model_requirements[model_type]
        is_valid = req['min_seq_len'] <= sequence_length <= req['max_seq_len']
        
        # Special validation for PatchTST - sequence length should be divisible by patch length
        if model_type == 'patchtst' and is_valid:
            patch_len = 16  # Default patch length
            is_valid = sequence_length >= patch_len and sequence_length % patch_len == 0
        
        return is_valid

    def calculate_multi_scale_temporal_features(self, price_data: pd.DataFrame) -> dict:
        """
        Calculate multi-scale temporal features for Transformer models
        
        Args:
            price_data: DataFrame with timestamp, OHLCV data
            
        Returns:
            Dictionary containing features at different time scales
        """
        try:
            if len(price_data) < 48:  # Need at least 48 hours for meaningful patterns
                self.logger.warning("Insufficient data for multi-scale temporal features",
                                  rows=len(price_data))
                return self._get_default_temporal_features()
            
            # Ensure timestamp column exists and is datetime
            if 'timestamp' in price_data.columns:
                price_data = price_data.copy()
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                price_data.set_index('timestamp', inplace=True)
            
            features = {}
            
            # Extract features at different scales
            features['minute_features'] = self.extract_minute_level_features(price_data)
            features['hour_features'] = self.extract_hour_level_features(price_data)
            features['day_features'] = self.extract_day_level_features(price_data)
            
            # Add temporal momentum patterns
            features['momentum_patterns'] = self.calculate_temporal_momentum_patterns(price_data)
            
            # Add volatility regime indicators
            features['volatility_regimes'] = self.calculate_volatility_regime_indicators(price_data)
            
            self.logger.info("Multi-scale temporal features calculated",
                           features_count=sum(len(v) if isinstance(v, dict) else 1 for v in features.values()))
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to calculate multi-scale temporal features", error=str(e))
            return self._get_default_temporal_features()
    
    def extract_minute_level_features(self, price_data: pd.DataFrame) -> dict:
        """Extract minute-level temporal features"""
        try:
            # Resample to minute-level if not already
            if hasattr(price_data.index, 'freq') and price_data.index.freq is None:
                minute_data = price_data.resample('1min').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            else:
                minute_data = price_data
            
            if len(minute_data) < 60:  # Need at least an hour of minute data
                return {'minute_volatility': 0.0, 'minute_volume_profile': 0.5}
            
            # Calculate minute-level features
            minute_returns = minute_data['close'].pct_change().dropna()
            
            features = {
                'minute_volatility': float(minute_returns.std() * np.sqrt(1440)),  # Annualized
                'minute_skewness': float(minute_returns.skew()) if len(minute_returns) > 10 else 0.0,
                'minute_kurtosis': float(minute_returns.kurtosis()) if len(minute_returns) > 10 else 0.0,
                'minute_volume_profile': float(minute_data['volume'].tail(60).mean() / minute_data['volume'].mean()) if minute_data['volume'].mean() > 0 else 1.0,
                'minute_price_acceleration': float(minute_returns.diff().tail(10).mean()) if len(minute_returns) > 10 else 0.0
            }
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to extract minute-level features", error=str(e))
            return {'minute_volatility': 0.0, 'minute_volume_profile': 0.5}
    
    def extract_hour_level_features(self, price_data: pd.DataFrame) -> dict:
        """Extract hour-level temporal features"""
        try:
            # Resample to hourly if not already
            hourly_data = price_data.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(hourly_data) < 24:  # Need at least a day of hourly data
                return self._get_default_hour_features()
            
            # Calculate hourly features
            hourly_returns = hourly_data['close'].pct_change().dropna()
            hourly_volumes = hourly_data['volume']
            
            # Hour-of-day patterns
            hourly_data['hour'] = hourly_data.index.hour
            hour_volume_pattern = hourly_data.groupby('hour')['volume'].mean()
            current_hour = hourly_data.index[-1].hour
            hour_volume_ratio = float(hour_volume_pattern.loc[current_hour] / hour_volume_pattern.mean()) if hour_volume_pattern.mean() > 0 else 1.0
            
            # Intraday momentum
            intraday_momentum = float(hourly_returns.tail(6).mean())  # Last 6 hours
            
            # Volume-weighted features
            vwap_1h = float((hourly_data['close'] * hourly_data['volume']).sum() / hourly_data['volume'].sum()) if hourly_data['volume'].sum() > 0 else float(hourly_data['close'].iloc[-1])
            current_price = float(hourly_data['close'].iloc[-1])
            vwap_deviation = (current_price - vwap_1h) / vwap_1h if vwap_1h > 0 else 0.0
            
            features = {
                'hourly_volatility': float(hourly_returns.std() * np.sqrt(24 * 365)),  # Annualized
                'hourly_trend_strength': float(abs(hourly_returns.tail(24).mean())) if len(hourly_returns) >= 24 else 0.0,
                'hour_volume_ratio': hour_volume_ratio,
                'intraday_momentum': intraday_momentum,
                'vwap_1h_deviation': float(vwap_deviation),
                'hourly_range_ratio': float((hourly_data['high'].iloc[-1] - hourly_data['low'].iloc[-1]) / hourly_data['close'].iloc[-1]) if hourly_data['close'].iloc[-1] > 0 else 0.0,
                'volume_trend_1h': float(hourly_volumes.tail(6).mean() / hourly_volumes.tail(24).mean()) if len(hourly_volumes) >= 24 and hourly_volumes.tail(24).mean() > 0 else 1.0,
                'price_momentum_6h': float(hourly_returns.tail(6).sum()),
                'price_momentum_12h': float(hourly_returns.tail(12).sum()) if len(hourly_returns) >= 12 else 0.0,
                'price_momentum_24h': float(hourly_returns.tail(24).sum()) if len(hourly_returns) >= 24 else 0.0
            }
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to extract hour-level features", error=str(e))
            return self._get_default_hour_features()
    
    def extract_day_level_features(self, price_data: pd.DataFrame) -> dict:
        """Extract day-level temporal features"""
        try:
            # Resample to daily if not already
            daily_data = price_data.resample('1D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last', 
                'volume': 'sum'
            }).dropna()
            
            if len(daily_data) < 7:  # Need at least a week of daily data
                return self._get_default_day_features()
            
            # Calculate daily features
            daily_returns = daily_data['close'].pct_change().dropna()
            
            # Day-of-week patterns
            daily_data['dayofweek'] = daily_data.index.dayofweek
            dow_volume_pattern = daily_data.groupby('dayofweek')['volume'].mean()
            current_dow = daily_data.index[-1].dayofweek
            dow_volume_ratio = float(dow_volume_pattern.loc[current_dow] / dow_volume_pattern.mean()) if dow_volume_pattern.mean() > 0 else 1.0
            
            # Weekly patterns
            weekly_trend = float(daily_returns.tail(7).mean()) if len(daily_returns) >= 7 else 0.0
            
            features = {
                'daily_volatility': float(daily_returns.std() * np.sqrt(365)),  # Annualized
                'weekly_trend': weekly_trend,
                'dow_volume_ratio': dow_volume_ratio,
                'daily_range_avg_7d': float(((daily_data['high'] - daily_data['low']) / daily_data['close']).tail(7).mean()) if len(daily_data) >= 7 else 0.0,
                'daily_volume_trend_7d': float(daily_data['volume'].tail(7).mean() / daily_data['volume'].tail(30).mean()) if len(daily_data) >= 30 and daily_data['volume'].tail(30).mean() > 0 else 1.0,
                'consecutive_days_direction': self._calculate_consecutive_days(daily_returns),
                'max_daily_return_7d': float(daily_returns.tail(7).max()) if len(daily_returns) >= 7 else 0.0,
                'min_daily_return_7d': float(daily_returns.tail(7).min()) if len(daily_returns) >= 7 else 0.0
            }
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to extract day-level features", error=str(e))
            return self._get_default_day_features()
    
    def calculate_temporal_momentum_patterns(self, price_data: pd.DataFrame) -> dict:
        """Calculate temporal momentum patterns across different scales"""
        try:
            if len(price_data) < 48:
                return {'momentum_consistency': 0.0, 'momentum_acceleration': 0.0}
            
            # Get returns at different frequencies
            returns_1h = price_data['close'].resample('1h').last().pct_change().dropna()
            returns_4h = price_data['close'].resample('4h').last().pct_change().dropna()
            returns_1d = price_data['close'].resample('1D').last().pct_change().dropna()
            
            # Calculate momentum consistency across scales
            momentum_1h = returns_1h.tail(24).mean() if len(returns_1h) >= 24 else 0.0
            momentum_4h = returns_4h.tail(6).mean() if len(returns_4h) >= 6 else 0.0  
            momentum_1d = returns_1d.tail(7).mean() if len(returns_1d) >= 7 else 0.0
            
            # Momentum consistency (same direction across scales)
            momentum_signs = [np.sign(momentum_1h), np.sign(momentum_4h), np.sign(momentum_1d)]
            momentum_consistency = float(abs(sum(momentum_signs)) / 3.0)
            
            # Momentum acceleration (increasing momentum over time)
            short_momentum = float(returns_1h.tail(6).mean()) if len(returns_1h) >= 6 else 0.0
            medium_momentum = float(returns_1h.tail(24).mean()) if len(returns_1h) >= 24 else 0.0
            momentum_acceleration = float(short_momentum - medium_momentum)
            
            return {
                'momentum_consistency': momentum_consistency,
                'momentum_acceleration': momentum_acceleration,
                'momentum_1h': float(momentum_1h),
                'momentum_4h': float(momentum_4h), 
                'momentum_1d': float(momentum_1d)
            }
            
        except Exception as e:
            self.logger.error("Failed to calculate temporal momentum patterns", error=str(e))
            return {'momentum_consistency': 0.0, 'momentum_acceleration': 0.0}
    
    def calculate_volatility_regime_indicators(self, price_data: pd.DataFrame) -> dict:
        """Calculate volatility regime indicators"""
        try:
            if len(price_data) < 48:
                return {'volatility_regime': 'medium', 'regime_persistence': 0.5}
            
            # Calculate returns
            returns = price_data['close'].pct_change().dropna()
            
            if len(returns) < 24:
                return {'volatility_regime': 'medium', 'regime_persistence': 0.5}
            
            # Rolling volatility at different windows
            vol_6h = returns.rolling(6).std() * np.sqrt(24 * 365)  # Annualized
            vol_24h = returns.rolling(24).std() * np.sqrt(24 * 365)
            vol_168h = returns.rolling(168).std() * np.sqrt(24 * 365) if len(returns) >= 168 else vol_24h
            
            # Current volatility regime
            current_vol = float(vol_24h.iloc[-1]) if not pd.isna(vol_24h.iloc[-1]) else 0.5
            historical_vol = float(vol_168h.mean()) if len(vol_168h) > 0 and not pd.isna(vol_168h.mean()) else current_vol
            
            # Classify regime
            if current_vol > historical_vol * 1.5:
                regime = 'high_vol'
                regime_score = min(current_vol / historical_vol, 3.0) / 3.0
            elif current_vol < historical_vol * 0.7:
                regime = 'low_vol'
                regime_score = max(0.0, (historical_vol * 0.7 - current_vol) / (historical_vol * 0.7))
            else:
                regime = 'medium'
                regime_score = 0.5
            
            # Regime persistence (how long has this regime lasted)
            vol_threshold_high = historical_vol * 1.5
            vol_threshold_low = historical_vol * 0.7
            
            recent_vols = vol_24h.tail(48) if len(vol_24h) >= 48 else vol_24h.tail(len(vol_24h))
            
            if regime == 'high_vol':
                persistence_periods = (recent_vols > vol_threshold_high).sum()
            elif regime == 'low_vol':
                persistence_periods = (recent_vols < vol_threshold_low).sum()
            else:
                persistence_periods = ((recent_vols >= vol_threshold_low) & (recent_vols <= vol_threshold_high)).sum()
            
            regime_persistence = float(persistence_periods / len(recent_vols))
            
            return {
                'volatility_regime': regime,
                'regime_score': float(regime_score),
                'regime_persistence': regime_persistence,
                'current_vol_ratio': float(current_vol / historical_vol) if historical_vol > 0 else 1.0,
                'vol_trend_6h_24h': float(vol_6h.iloc[-1] / vol_24h.iloc[-1]) if not pd.isna(vol_6h.iloc[-1]) and not pd.isna(vol_24h.iloc[-1]) and vol_24h.iloc[-1] > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.error("Failed to calculate volatility regime indicators", error=str(e))
            return {'volatility_regime': 'medium', 'regime_persistence': 0.5}
    
    def _get_default_temporal_features(self) -> dict:
        """Get default temporal features when calculation fails"""
        return {
            'minute_features': {'minute_volatility': 0.0, 'minute_volume_profile': 0.5},
            'hour_features': self._get_default_hour_features(),
            'day_features': self._get_default_day_features(),
            'momentum_patterns': {'momentum_consistency': 0.0, 'momentum_acceleration': 0.0},
            'volatility_regimes': {'volatility_regime': 'medium', 'regime_persistence': 0.5}
        }
    
    def _get_default_hour_features(self) -> dict:
        """Get default hour-level features"""
        return {
            'hourly_volatility': 0.5,
            'hourly_trend_strength': 0.0,
            'hour_volume_ratio': 1.0,
            'intraday_momentum': 0.0,
            'vwap_1h_deviation': 0.0,
            'hourly_range_ratio': 0.02,
            'volume_trend_1h': 1.0,
            'price_momentum_6h': 0.0,
            'price_momentum_12h': 0.0,
            'price_momentum_24h': 0.0
        }
    
    def _get_default_day_features(self) -> dict:
        """Get default day-level features"""
        return {
            'daily_volatility': 0.5,
            'weekly_trend': 0.0,
            'dow_volume_ratio': 1.0,
            'daily_range_avg_7d': 0.03,
            'daily_volume_trend_7d': 1.0,
            'consecutive_days_direction': 0,
            'max_daily_return_7d': 0.0,
            'min_daily_return_7d': 0.0
        }
    
    def _calculate_consecutive_days(self, daily_returns: pd.Series) -> int:
        """Calculate consecutive days in the same direction"""
        if len(daily_returns) < 2:
            return 0
        
        recent_returns = daily_returns.tail(7)  # Look at last 7 days
        consecutive = 0
        current_direction = None
        
        for ret in reversed(recent_returns.tolist()):
            direction = 1 if ret > 0 else -1 if ret < 0 else 0
            
            if current_direction is None:
                current_direction = direction
                consecutive = 1 if direction != 0 else 0
            elif direction == current_direction and direction != 0:
                consecutive += 1
            else:
                break
        
        return consecutive

    def calculate_cross_asset_correlations(self, price_data: Dict[str, pd.DataFrame], tokens: List[DiscoveredToken]) -> dict:
        """
        Calculate cross-asset correlation features for multivariate Transformer inputs
        
        Args:
            price_data: Dictionary mapping token addresses to price DataFrames
            tokens: List of tokens to analyze
            
        Returns:
            Dictionary containing correlation features
        """
        try:
            if len(price_data) < 2:
                self.logger.warning("Need at least 2 assets for correlation analysis", assets=len(price_data))
                return self._get_default_correlation_features()
            
            # Align timestamps and extract returns
            aligned_returns = self._align_multivariate_returns(price_data, tokens)
            
            if aligned_returns.empty or len(aligned_returns.columns) < 2:
                return self._get_default_correlation_features()
            
            # Calculate correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(aligned_returns)
            
            # Calculate rolling correlations
            rolling_corr = self.calculate_rolling_correlations(aligned_returns, window=24)  # 24-hour window
            
            # Calculate correlation regime features
            regime_features = self.calculate_correlation_regime_features(aligned_returns)
            
            return {
                'correlation_matrix': correlation_matrix,
                'rolling_correlations': rolling_corr,
                'regime_features': regime_features,
                'num_assets': len(aligned_returns.columns)
            }
            
        except Exception as e:
            self.logger.error("Failed to calculate cross-asset correlations", error=str(e))
            return self._get_default_correlation_features()
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> dict:
        """Calculate correlation matrix and derived statistics"""
        try:
            if len(returns_data) < 10 or len(returns_data.columns) < 2:
                return {'mean_correlation': 0.0, 'max_correlation': 0.0, 'correlation_stability': 0.5}
            
            # Calculate correlation matrix
            corr_matrix = returns_data.corr()
            
            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper_triangle = corr_matrix.where(mask)
            
            # Calculate statistics
            correlations = upper_triangle.stack().dropna()
            
            if len(correlations) == 0:
                return {'mean_correlation': 0.0, 'max_correlation': 0.0, 'correlation_stability': 0.5}
            
            features = {
                'mean_correlation': float(correlations.mean()),
                'max_correlation': float(correlations.max()),
                'min_correlation': float(correlations.min()),
                'correlation_std': float(correlations.std()),
                'positive_correlations_ratio': float((correlations > 0).sum() / len(correlations)),
                'high_correlation_pairs': int((correlations.abs() > 0.7).sum()),
                'correlation_stability': float(1.0 - correlations.std()) if correlations.std() < 1.0 else 0.0
            }
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to calculate correlation matrix", error=str(e))
            return {'mean_correlation': 0.0, 'max_correlation': 0.0, 'correlation_stability': 0.5}
    
    def calculate_rolling_correlations(self, returns_data: pd.DataFrame, window: int = 24) -> dict:
        """Calculate rolling correlation features"""
        try:
            if len(returns_data) < window * 2 or len(returns_data.columns) < 2:
                return {'rolling_corr_mean': 0.0, 'rolling_corr_trend': 0.0}
            
            # Calculate rolling correlations for each pair
            rolling_corrs = {}
            columns = list(returns_data.columns)
            
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    col1, col2 = columns[i], columns[j]
                    rolling_corr = returns_data[col1].rolling(window).corr(returns_data[col2])
                    rolling_corrs[f"{col1}_{col2}"] = rolling_corr.dropna()
            
            if not rolling_corrs:
                return {'rolling_corr_mean': 0.0, 'rolling_corr_trend': 0.0}
            
            # Aggregate rolling correlations
            all_rolling_corrs = pd.concat(rolling_corrs.values(), axis=1)
            mean_rolling_corr = all_rolling_corrs.mean(axis=1)
            
            features = {
                'rolling_corr_mean': float(mean_rolling_corr.tail(1).iloc[0]) if len(mean_rolling_corr) > 0 else 0.0,
                'rolling_corr_trend': float(mean_rolling_corr.tail(6).mean() - mean_rolling_corr.tail(24).mean()) if len(mean_rolling_corr) >= 24 else 0.0,
                'rolling_corr_volatility': float(mean_rolling_corr.tail(24).std()) if len(mean_rolling_corr) >= 24 else 0.0,
                'correlation_regime_changes': int((mean_rolling_corr.diff().abs() > 0.1).sum()) if len(mean_rolling_corr) > 1 else 0
            }
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to calculate rolling correlations", error=str(e))
            return {'rolling_corr_mean': 0.0, 'rolling_corr_trend': 0.0}
    
    def calculate_correlation_regime_features(self, returns_data: pd.DataFrame) -> dict:
        """Calculate correlation regime features"""
        try:
            if len(returns_data) < 48 or len(returns_data.columns) < 2:
                return {'correlation_regime': 'medium', 'regime_strength': 0.5}
            
            # Calculate recent correlation
            recent_corr = returns_data.tail(24).corr()
            mask = np.triu(np.ones_like(recent_corr, dtype=bool), k=1)
            recent_correlations = recent_corr.where(mask).stack().dropna()
            
            # Calculate historical correlation
            historical_corr = returns_data.corr()
            historical_correlations = historical_corr.where(mask).stack().dropna()
            
            if len(recent_correlations) == 0 or len(historical_correlations) == 0:
                return {'correlation_regime': 'medium', 'regime_strength': 0.5}
            
            # Determine correlation regime
            recent_mean = recent_correlations.mean()
            historical_mean = historical_correlations.mean()
            
            if recent_mean > 0.7:
                regime = 'high_correlation'
                regime_strength = min(recent_mean, 1.0)
            elif recent_mean < 0.3:
                regime = 'low_correlation'  
                regime_strength = max(0.0, 1.0 - recent_mean)
            else:
                regime = 'medium_correlation'
                regime_strength = 0.5
            
            # Calculate regime persistence
            rolling_corr = returns_data.rolling(24).corr().groupby(level=1).mean()
            if len(rolling_corr) > 0:
                recent_regime_periods = 0
                for _, corr_row in rolling_corr.tail(48).iterrows():
                    corr_values = corr_row.dropna()
                    if len(corr_values) > 0:
                        period_mean = corr_values.mean()
                        if regime == 'high_correlation' and period_mean > 0.7:
                            recent_regime_periods += 1
                        elif regime == 'low_correlation' and period_mean < 0.3:
                            recent_regime_periods += 1
                        elif regime == 'medium_correlation' and 0.3 <= period_mean <= 0.7:
                            recent_regime_periods += 1
                
                regime_persistence = recent_regime_periods / min(48, len(rolling_corr))
            else:
                regime_persistence = 0.5
            
            return {
                'correlation_regime': regime,
                'regime_strength': float(regime_strength),
                'regime_persistence': float(regime_persistence),
                'correlation_change': float(recent_mean - historical_mean),
                'correlation_dispersion': float(recent_correlations.std()) if len(recent_correlations) > 1 else 0.0
            }
            
        except Exception as e:
            self.logger.error("Failed to calculate correlation regime features", error=str(e))
            return {'correlation_regime': 'medium', 'regime_strength': 0.5}
    
    def _align_multivariate_returns(self, price_data: Dict[str, pd.DataFrame], tokens: List[DiscoveredToken]) -> pd.DataFrame:
        """Align multivariate price data and calculate returns"""
        try:
            returns_dict = {}
            
            for token in tokens:
                if token.address not in price_data:
                    continue
                
                df = price_data[token.address].copy()
                
                # Ensure datetime index
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif not isinstance(df.index, pd.DatetimeIndex):
                    continue
                
                # Calculate returns
                if 'close' in df.columns:
                    returns = df['close'].pct_change().dropna()
                    returns_dict[token.symbol or token.address] = returns
            
            if not returns_dict:
                return pd.DataFrame()
            
            # Align timestamps
            aligned_returns = pd.DataFrame(returns_dict)
            aligned_returns = aligned_returns.dropna()
            
            return aligned_returns
            
        except Exception as e:
            self.logger.error("Failed to align multivariate returns", error=str(e))
            return pd.DataFrame()
    
    def _get_default_correlation_features(self) -> dict:
        """Get default correlation features when calculation fails"""
        return {
            'correlation_matrix': {'mean_correlation': 0.0, 'max_correlation': 0.0, 'correlation_stability': 0.5},
            'rolling_correlations': {'rolling_corr_mean': 0.0, 'rolling_corr_trend': 0.0},
            'regime_features': {'correlation_regime': 'medium', 'regime_strength': 0.5},
            'num_assets': 1
        }

    def detect_market_regime(self, price_data: pd.DataFrame) -> dict:
        """
        Detect current market regime based on price action and volatility
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing regime information
        """
        try:
            if len(price_data) < 48:
                return {'current_regime': 'sideways', 'regime_probability': 0.5}
            
            # Calculate returns
            returns = price_data['close'].pct_change().dropna()
            
            if len(returns) < 24:
                return {'current_regime': 'sideways', 'regime_probability': 0.5}
            
            # Calculate momentum and volatility metrics
            short_momentum = returns.tail(24).mean()  # 24h momentum
            medium_momentum = returns.tail(168).mean() if len(returns) >= 168 else short_momentum  # 1 week
            
            volatility = returns.tail(24).std()
            historical_vol = returns.std()
            
            # Trend strength
            trend_strength = abs(short_momentum) / (volatility + 1e-8)
            
            # Regime classification
            regime_scores = {
                'bull': 0.0,
                'bear': 0.0,
                'sideways': 0.0,
                'high_vol': 0.0,
                'low_vol': 0.0
            }
            
            # Bull/Bear classification
            if short_momentum > 0.005 and trend_strength > 0.5:  # Strong upward momentum
                regime_scores['bull'] = min(trend_strength, 2.0) / 2.0
            elif short_momentum < -0.005 and trend_strength > 0.5:  # Strong downward momentum
                regime_scores['bear'] = min(trend_strength, 2.0) / 2.0
            else:
                regime_scores['sideways'] = 1.0 - trend_strength
            
            # Volatility classification
            vol_ratio = volatility / (historical_vol + 1e-8)
            if vol_ratio > 1.5:
                regime_scores['high_vol'] = min(vol_ratio - 1.0, 1.0)
            elif vol_ratio < 0.7:
                regime_scores['low_vol'] = min(1.5 - vol_ratio, 1.0)
            
            # Select dominant regime
            current_regime = max(regime_scores, key=regime_scores.get)
            regime_probability = regime_scores[current_regime]
            
            # Calculate regime transition probabilities
            transition_probs = self.calculate_regime_transition_probabilities(returns)
            
            # Additional regime features
            regime_info = {
                'current_regime': current_regime,
                'regime_probability': float(regime_probability),
                'momentum_score': float(short_momentum),
                'trend_strength': float(trend_strength),
                'volatility_ratio': float(vol_ratio),
                'regime_stability': float(1.0 - transition_probs.get('transition_entropy', 0.5)),
                'momentum_acceleration': float(short_momentum - medium_momentum)
            }
            
            return regime_info
            
        except Exception as e:
            self.logger.error("Failed to detect market regime", error=str(e))
            return {'current_regime': 'sideways', 'regime_probability': 0.5}
    
    def calculate_regime_transition_probabilities(self, returns: pd.Series) -> dict:
        """Calculate regime transition probabilities"""
        try:
            if len(returns) < 48:
                return {'transition_entropy': 0.5}
            
            # Create regime history based on rolling momentum
            rolling_momentum = returns.rolling(24).mean()
            rolling_vol = returns.rolling(24).std()
            
            regime_history = []
            for i in range(len(rolling_momentum)):
                if pd.isna(rolling_momentum.iloc[i]) or pd.isna(rolling_vol.iloc[i]):
                    continue
                
                momentum = rolling_momentum.iloc[i]
                vol = rolling_vol.iloc[i]
                
                # Simple regime classification
                if momentum > 0.003:
                    regime = 'bull'
                elif momentum < -0.003:
                    regime = 'bear'
                else:
                    regime = 'sideways'
                
                regime_history.append(regime)
            
            if len(regime_history) < 10:
                return {'transition_entropy': 0.5}
            
            # Calculate transition matrix
            transitions = {}
            for i in range(len(regime_history) - 1):
                current = regime_history[i]
                next_regime = regime_history[i + 1]
                
                if current not in transitions:
                    transitions[current] = {}
                if next_regime not in transitions[current]:
                    transitions[current][next_regime] = 0
                
                transitions[current][next_regime] += 1
            
            # Calculate transition entropy (measure of regime stability)
            total_transitions = sum(sum(next_states.values()) for next_states in transitions.values())
            if total_transitions == 0:
                return {'transition_entropy': 0.5}
            
            entropy = 0.0
            for current_regime, next_states in transitions.items():
                regime_total = sum(next_states.values())
                for next_regime, count in next_states.items():
                    prob = count / regime_total
                    if prob > 0:
                        entropy -= prob * np.log2(prob)
            
            # Normalize entropy
            max_entropy = np.log2(3)  # Maximum entropy for 3 regimes
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return {
                'transition_entropy': float(normalized_entropy),
                'regime_persistence': float(1.0 - normalized_entropy),
                'total_transitions': int(total_transitions)
            }
            
        except Exception as e:
            self.logger.error("Failed to calculate regime transition probabilities", error=str(e))
            return {'transition_entropy': 0.5}
    
    def calculate_volatility_clustering_features(self, price_data: pd.DataFrame) -> dict:
        """Calculate volatility clustering and GARCH-like features"""
        try:
            if len(price_data) < 48:
                return {'volatility_clustering': 0.5, 'garch_effect': 0.0}
            
            returns = price_data['close'].pct_change().dropna()
            
            if len(returns) < 24:
                return {'volatility_clustering': 0.5, 'garch_effect': 0.0}
            
            # Calculate squared returns (proxy for volatility)
            squared_returns = returns ** 2
            
            # Calculate volatility clustering using autocorrelation of squared returns
            lag_1_corr = squared_returns.autocorr(lag=1) if len(squared_returns) > 1 else 0.0
            lag_2_corr = squared_returns.autocorr(lag=2) if len(squared_returns) > 2 else 0.0
            lag_3_corr = squared_returns.autocorr(lag=3) if len(squared_returns) > 3 else 0.0
            
            # Average autocorrelation as clustering measure
            volatility_clustering = (lag_1_corr + lag_2_corr + lag_3_corr) / 3.0
            
            # GARCH effect: high volatility followed by high volatility
            rolling_vol = returns.rolling(6).std()
            vol_persistence = rolling_vol.autocorr(lag=1) if len(rolling_vol.dropna()) > 1 else 0.0
            
            # Volatility regime switching
            high_vol_threshold = returns.std() * 1.5
            low_vol_threshold = returns.std() * 0.7
            
            vol_regimes = []
            for vol in rolling_vol.dropna():
                if vol > high_vol_threshold:
                    vol_regimes.append('high')
                elif vol < low_vol_threshold:
                    vol_regimes.append('low')
                else:
                    vol_regimes.append('medium')
            
            # Calculate regime persistence  
            regime_changes = sum(1 for i in range(1, len(vol_regimes)) if vol_regimes[i] != vol_regimes[i-1])
            regime_persistence = 1.0 - (regime_changes / max(len(vol_regimes) - 1, 1))
            
            return {
                'volatility_clustering': float(volatility_clustering) if not pd.isna(volatility_clustering) else 0.5,
                'garch_effect': float(vol_persistence) if not pd.isna(vol_persistence) else 0.0,
                'volatility_regime_persistence': float(regime_persistence),
                'current_vol_regime': vol_regimes[-1] if vol_regimes else 'medium',
                'vol_lag1_autocorr': float(lag_1_corr) if not pd.isna(lag_1_corr) else 0.0
            }
            
        except Exception as e:
            self.logger.error("Failed to calculate volatility clustering features", error=str(e))
            return {'volatility_clustering': 0.5, 'garch_effect': 0.0}
    
    def calculate_market_stress_indicators(self, market_data: dict) -> dict:
        """Calculate market stress and risk indicators"""
        try:
            # This would integrate with market-wide data when available
            # For now, provide reasonable defaults
            
            stress_indicators = {
                'market_stress_level': 0.3,  # Low-medium stress
                'risk_off_sentiment': 0.0,   # Neutral
                'flight_to_quality': 0.0,    # No flight to quality
                'correlation_stress': 0.2,   # Low correlation stress
                'liquidity_stress': 0.1,     # Low liquidity stress
                'volatility_spike_risk': 0.25  # Low spike risk
            }
            
            self.logger.info("Market stress indicators calculated with default values")
            return stress_indicators
            
        except Exception as e:
            self.logger.error("Failed to calculate market stress indicators", error=str(e))
            return {
                'market_stress_level': 0.5,
                'risk_off_sentiment': 0.0,
                'flight_to_quality': 0.0,
                'correlation_stress': 0.5,
                'liquidity_stress': 0.5,
                'volatility_spike_risk': 0.5
            }

    def apply_transformer_normalization(self, features: np.ndarray) -> np.ndarray:
        """
        Apply attention-friendly normalization schemes
        
        Args:
            features: Feature array to normalize
            
        Returns:
            Normalized features suitable for attention mechanisms
        """
        try:
            # Replace NaN/Inf with zeros first
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply robust scaling to handle outliers
            features = self._apply_robust_scaling(features)
            
            # Apply layer normalization
            features = self._apply_layer_normalization(features)
            
            # Final clipping for attention stability
            features = np.clip(features, -5.0, 5.0)
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to apply transformer normalization", error=str(e))
            return np.zeros_like(features)
    
    def calculate_attention_scaling_factors(self, feature_matrix: np.ndarray) -> dict:
        """Calculate scaling factors optimized for attention mechanisms"""
        try:
            scaling_factors = {}
            
            # Calculate per-feature statistics
            for i in range(feature_matrix.shape[1]):
                feature_col = feature_matrix[:, i]
                
                # Robust statistics
                median = np.median(feature_col)
                mad = np.median(np.abs(feature_col - median))  # Median Absolute Deviation
                q75, q25 = np.percentile(feature_col, [75, 25])
                iqr = q75 - q25
                
                scaling_factors[f'feature_{i}'] = {
                    'median': float(median),
                    'mad': float(mad),
                    'iqr': float(iqr),
                    'robust_scale': float(iqr if iqr > 0 else mad if mad > 0 else 1.0)
                }
            
            return scaling_factors
            
        except Exception as e:
            self.logger.error("Failed to calculate attention scaling factors", error=str(e))
            return {}
    
    def _apply_robust_scaling(self, features: np.ndarray) -> np.ndarray:
        """Apply robust scaling using median and IQR"""
        if len(features.shape) == 1:
            median = np.median(features)
            q75, q25 = np.percentile(features, [75, 25])
            iqr = q75 - q25
            return (features - median) / (iqr + 1e-8)
        else:
            # Apply per-feature scaling
            scaled = features.copy()
            for i in range(features.shape[1]):
                col = features[:, i]
                median = np.median(col)
                q75, q25 = np.percentile(col, [75, 25])
                iqr = q75 - q25
                scaled[:, i] = (col - median) / (iqr + 1e-8)
            return scaled
    
    def _apply_layer_normalization(self, features: np.ndarray) -> np.ndarray:
        """Apply layer normalization across feature dimension"""
        if len(features.shape) == 1:
            mean = np.mean(features)
            std = np.std(features)
            return (features - mean) / (std + 1e-8)
        else:
            # Apply across feature dimension
            mean = np.mean(features, axis=1, keepdims=True)
            std = np.std(features, axis=1, keepdims=True)
            return (features - mean) / (std + 1e-8)

    # Crypto-Specific Features
    def calculate_funding_rate_features(self, symbol: str) -> dict:
        """Calculate funding rate features for crypto derivatives"""
        try:
            # Placeholder implementation - would integrate with real funding rate APIs
            # For now, return simulated funding rate features
            
            funding_features = {
                'current_funding_rate': np.random.normal(0.0001, 0.0005),  # Typical funding rates
                'funding_rate_trend_1h': np.random.normal(0.0, 0.0001),
                'funding_rate_trend_8h': np.random.normal(0.0, 0.0002),
                'funding_rate_volatility': abs(np.random.normal(0.0002, 0.0001)),
                'funding_payments_expected': np.random.normal(0.0, 0.001),
                'funding_rate_percentile': np.random.uniform(0.1, 0.9)
            }
            
            self.logger.info("Funding rate features calculated", symbol=symbol)
            return funding_features
            
        except Exception as e:
            self.logger.error("Failed to calculate funding rate features", symbol=symbol, error=str(e))
            return self._get_default_funding_features()
    
    def calculate_basis_spread_features(self, spot_prices: pd.Series, futures_prices: pd.Series) -> dict:
        """Calculate basis spread features between spot and futures"""
        try:
            if len(spot_prices) != len(futures_prices) or len(spot_prices) < 24:
                return self._get_default_basis_features()
            
            # Calculate basis spread
            basis_spread = futures_prices - spot_prices
            basis_spread_pct = (futures_prices - spot_prices) / spot_prices * 100
            
            # Calculate features
            features = {
                'current_basis_spread': float(basis_spread.iloc[-1]),
                'basis_spread_pct': float(basis_spread_pct.iloc[-1]),
                'basis_spread_mean_24h': float(basis_spread.tail(24).mean()),
                'basis_spread_std_24h': float(basis_spread.tail(24).std()),
                'basis_spread_trend': float(basis_spread.tail(6).mean() - basis_spread.tail(24).mean()),
                'contango_indicator': 1.0 if basis_spread.iloc[-1] > 0 else 0.0,
                'backwardation_indicator': 1.0 if basis_spread.iloc[-1] < 0 else 0.0,
                'basis_spread_percentile': float((basis_spread.iloc[-1] > basis_spread).sum() / len(basis_spread))
            }
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to calculate basis spread features", error=str(e))
            return self._get_default_basis_features()
    
    def calculate_arbitrage_indicators(self, exchange_prices: Dict[str, pd.Series]) -> dict:
        """Calculate cross-exchange arbitrage indicators"""
        try:
            if len(exchange_prices) < 2:
                return self._get_default_arbitrage_features()
            
            # Convert to DataFrame for easier handling
            price_df = pd.DataFrame(exchange_prices)
            price_df = price_df.dropna()
            
            if len(price_df) < 10:
                return self._get_default_arbitrage_features()
            
            # Calculate arbitrage opportunities
            max_prices = price_df.max(axis=1)
            min_prices = price_df.min(axis=1)
            arbitrage_spread = (max_prices - min_prices) / min_prices * 100
            
            # Calculate features
            features = {
                'current_arbitrage_spread': float(arbitrage_spread.iloc[-1]),
                'max_arbitrage_spread_24h': float(arbitrage_spread.tail(24).max()) if len(arbitrage_spread) >= 24 else float(arbitrage_spread.max()),
                'mean_arbitrage_spread': float(arbitrage_spread.mean()),
                'arbitrage_opportunity_frequency': float((arbitrage_spread > 0.1).sum() / len(arbitrage_spread)),  # >0.1% spread
                'price_dispersion': float(price_df.iloc[-1].std() / price_df.iloc[-1].mean()),
                'exchange_count': len(exchange_prices),
                'arbitrage_trend': float(arbitrage_spread.tail(6).mean() - arbitrage_spread.tail(24).mean()) if len(arbitrage_spread) >= 24 else 0.0
            }
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to calculate arbitrage indicators", error=str(e))
            return self._get_default_arbitrage_features()
    
    def calculate_trading_session_features(self, price_data: pd.DataFrame) -> dict:
        """Calculate trading session-based features"""
        try:
            if 'timestamp' in price_data.columns:
                price_data = price_data.copy()
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                price_data.set_index('timestamp', inplace=True)
            
            if not isinstance(price_data.index, pd.DatetimeIndex):
                return self._get_default_session_features()
            
            # Add time-based features
            price_data['hour'] = price_data.index.hour
            price_data['day_of_week'] = price_data.index.dayofweek
            
            # Define trading sessions (UTC times)
            asian_session = (price_data['hour'] >= 0) & (price_data['hour'] < 8)
            european_session = (price_data['hour'] >= 8) & (price_data['hour'] < 16)
            us_session = (price_data['hour'] >= 16) & (price_data['hour'] < 24)
            
            # Calculate session-based features
            current_hour = price_data.index[-1].hour
            
            features = {
                'is_asian_session': 1.0 if 0 <= current_hour < 8 else 0.0,
                'is_european_session': 1.0 if 8 <= current_hour < 16 else 0.0,
                'is_us_session': 1.0 if 16 <= current_hour < 24 else 0.0,
                'current_hour': float(current_hour),
                'is_weekend': 1.0 if price_data.index[-1].dayofweek >= 5 else 0.0,
                'session_volume_ratio': self._calculate_session_volume_ratio(price_data, current_hour),
                'session_volatility_ratio': self._calculate_session_volatility_ratio(price_data, current_hour),
                'hours_until_major_session': self._calculate_hours_to_major_session(current_hour)
            }
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to calculate trading session features", error=str(e))
            return self._get_default_session_features()
    
    def calculate_block_time_features(self, chain: Chain) -> dict:
        """Calculate blockchain-specific timing features"""
        try:
            # Chain-specific block time characteristics
            block_times = {
                Chain.ETHEREUM: 12.0,  # ~12 seconds
                Chain.SOLANA: 0.4,     # ~400ms
                Chain.BASE: 2.0,       # ~2 seconds
                Chain.POLYGON: 2.0,    # ~2 seconds
                Chain.BSC: 3.0,        # ~3 seconds
            }
            
            base_block_time = block_times.get(chain, 15.0)  # Default to 15s
            
            # Simulate block time variability and network congestion effects
            current_block_time = base_block_time * np.random.uniform(0.8, 1.5)
            network_congestion = np.random.uniform(0.0, 1.0)
            
            features = {
                'chain_block_time': float(base_block_time),
                'current_block_time_estimate': float(current_block_time),
                'block_time_variability': float(abs(current_block_time - base_block_time) / base_block_time),
                'network_congestion_estimate': float(network_congestion),
                'blocks_per_hour': float(3600 / current_block_time),
                'transaction_finality_time': float(current_block_time * 6),  # Assume 6 confirmations
                'chain_efficiency_score': float(1.0 / (current_block_time + 1))
            }
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to calculate block time features", chain=chain, error=str(e))
            return self._get_default_block_features()
    
    def _calculate_session_volume_ratio(self, price_data: pd.DataFrame, current_hour: int) -> float:
        """Calculate volume ratio for current session vs average"""
        try:
            if 'volume' not in price_data.columns or len(price_data) < 24:
                return 1.0
            
            # Get current session
            if 0 <= current_hour < 8:
                session_mask = (price_data['hour'] >= 0) & (price_data['hour'] < 8)
            elif 8 <= current_hour < 16:
                session_mask = (price_data['hour'] >= 8) & (price_data['hour'] < 16)
            else:
                session_mask = (price_data['hour'] >= 16) & (price_data['hour'] < 24)
            
            session_volume = price_data[session_mask]['volume'].mean()
            overall_volume = price_data['volume'].mean()
            
            return float(session_volume / overall_volume) if overall_volume > 0 else 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_session_volatility_ratio(self, price_data: pd.DataFrame, current_hour: int) -> float:
        """Calculate volatility ratio for current session vs average"""
        try:
            if 'close' not in price_data.columns or len(price_data) < 24:
                return 1.0
            
            returns = price_data['close'].pct_change().dropna()
            
            # Get current session
            if 0 <= current_hour < 8:
                session_mask = (price_data['hour'] >= 0) & (price_data['hour'] < 8)
            elif 8 <= current_hour < 16:
                session_mask = (price_data['hour'] >= 8) & (price_data['hour'] < 16)
            else:
                session_mask = (price_data['hour'] >= 16) & (price_data['hour'] < 24)
            
            session_returns = returns[session_mask[1:]]  # Adjust for pct_change offset
            session_vol = session_returns.std() if len(session_returns) > 1 else 0.0
            overall_vol = returns.std()
            
            return float(session_vol / overall_vol) if overall_vol > 0 else 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_hours_to_major_session(self, current_hour: int) -> float:
        """Calculate hours until next major trading session opens"""
        # Major sessions: US market open (14:30 UTC), Asian market open (23:00 UTC)
        major_sessions = [14.5, 23.0]  # 14:30 and 23:00 UTC
        
        hours_to_sessions = []
        for session_hour in major_sessions:
            if current_hour <= session_hour:
                hours_to_sessions.append(session_hour - current_hour)
            else:
                hours_to_sessions.append(24 - current_hour + session_hour)
        
        return float(min(hours_to_sessions))
    
    def _get_default_funding_features(self) -> dict:
        """Default funding rate features"""
        return {
            'current_funding_rate': 0.0001,
            'funding_rate_trend_1h': 0.0,
            'funding_rate_trend_8h': 0.0,
            'funding_rate_volatility': 0.0002,
            'funding_payments_expected': 0.0,
            'funding_rate_percentile': 0.5
        }
    
    def _get_default_basis_features(self) -> dict:
        """Default basis spread features"""
        return {
            'current_basis_spread': 0.0,
            'basis_spread_pct': 0.0,
            'basis_spread_mean_24h': 0.0,
            'basis_spread_std_24h': 0.0,
            'basis_spread_trend': 0.0,
            'contango_indicator': 0.0,
            'backwardation_indicator': 0.0,
            'basis_spread_percentile': 0.5
        }
    
    def _get_default_arbitrage_features(self) -> dict:
        """Default arbitrage features"""
        return {
            'current_arbitrage_spread': 0.0,
            'max_arbitrage_spread_24h': 0.0,
            'mean_arbitrage_spread': 0.0,
            'arbitrage_opportunity_frequency': 0.0,
            'price_dispersion': 0.0,
            'exchange_count': 1,
            'arbitrage_trend': 0.0
        }
    
    def _get_default_session_features(self) -> dict:
        """Default trading session features"""
        return {
            'is_asian_session': 0.0,
            'is_european_session': 0.0,
            'is_us_session': 0.0,
            'current_hour': 12.0,
            'is_weekend': 0.0,
            'session_volume_ratio': 1.0,
            'session_volatility_ratio': 1.0,
            'hours_until_major_session': 6.0
        }
    
    def extract_stablecoin_features(self, ohlcv_data: pd.DataFrame, 
                                   token_symbol: str = None,
                                   target_price: float = 1.0) -> dict:
        """
        Extract stablecoin-specific features for market stress detection
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            token_symbol: Token symbol to check if it's a stablecoin
            target_price: Expected stable price (default 1.0 for USD stables)
            
        Returns:
            Dictionary of stablecoin-specific features
        """
        # Check if this is a stablecoin
        stablecoins = ['USDC', 'USDT', 'DAI', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'FRAX']
        if token_symbol and token_symbol.upper() not in stablecoins:
            return {}  # Not a stablecoin, return empty dict
        
        try:
            if ohlcv_data is None or ohlcv_data.empty or len(ohlcv_data) < 2:
                return self._get_default_stablecoin_features()
            
            features = {}
            close_prices = ohlcv_data['close']
            volumes = ohlcv_data['volume']
            
            # Depeg Magnitude Features
            features['stable_current_depeg'] = float(abs(close_prices.iloc[-1] - target_price))
            features['stable_depeg_ratio'] = float(close_prices.iloc[-1] / target_price)
            
            # Max depeg in last 24 hours
            lookback = min(24, len(close_prices))
            recent_prices = close_prices.tail(lookback)
            features['stable_max_depeg_24h'] = float(max(abs(recent_prices.max() - target_price), 
                                                        abs(recent_prices.min() - target_price)))
            features['stable_depeg_direction'] = 1.0 if close_prices.iloc[-1] > target_price else -1.0
            
            # Depeg Persistence Features
            depeg_threshold = 0.002  # 0.2%
            features['stable_hours_depegged'] = float(sum(abs(recent_prices - target_price) > depeg_threshold))
            features['stable_consecutive_depeg'] = float(self._count_consecutive_depeg(close_prices, target_price, depeg_threshold))
            features['stable_depeg_volatility'] = float(recent_prices.std()) if len(recent_prices) > 1 else 0.0
            
            # Volume Spike Features (Flight to Safety)
            baseline_lookback = min(168, len(volumes))  # 7 days
            avg_volume_7d = volumes.tail(baseline_lookback).mean() if baseline_lookback > 0 else volumes.mean()
            
            if avg_volume_7d > 0:
                features['stable_volume_spike_ratio'] = float(volumes.iloc[-1] / avg_volume_7d)
                features['stable_volume_spike_24h'] = float(volumes.tail(lookback).max() / avg_volume_7d)
            else:
                features['stable_volume_spike_ratio'] = 1.0
                features['stable_volume_spike_24h'] = 1.0
            
            features['stable_panic_volume_score'] = float(min(features['stable_volume_spike_ratio'] / 10, 1.0))
            
            # Recovery/Stress Features
            features['stable_mean_reversion'] = float(self._calculate_stable_mean_reversion(close_prices, target_price))
            features['stable_stress_persistence'] = float(self._calculate_stable_stress_persistence(close_prices, target_price, depeg_threshold))
            
            # Market Regime Features
            features['stable_is_panic_mode'] = 1.0 if (features['stable_current_depeg'] > 0.01 and 
                                                       features['stable_volume_spike_ratio'] > 3) else 0.0
            features['stable_is_premium'] = 1.0 if close_prices.iloc[-1] > (target_price + 0.005) else 0.0
            features['stable_is_discount'] = 1.0 if close_prices.iloc[-1] < (target_price - 0.005) else 0.0
            
            # Arbitrage Opportunities
            features['stable_arbitrage_opportunity'] = float(abs(features['stable_current_depeg']) * features['stable_volume_spike_ratio'])
            features['stable_profit_potential_bps'] = float(abs(features['stable_current_depeg']) * 10000)  # In basis points
            
            # Liquidity Stress Features
            if 'high' in ohlcv_data.columns and 'low' in ohlcv_data.columns:
                features['stable_range_ratio'] = float((ohlcv_data['high'].iloc[-1] - ohlcv_data['low'].iloc[-1]) / 
                                                      close_prices.iloc[-1]) if close_prices.iloc[-1] > 0 else 0.0
                ranges = (ohlcv_data['high'].tail(lookback) - ohlcv_data['low'].tail(lookback)) / close_prices.tail(lookback)
                features['stable_avg_range_24h'] = float(ranges.mean())
            else:
                features['stable_range_ratio'] = 0.001
                features['stable_avg_range_24h'] = 0.001
            
            return features
            
        except Exception as e:
            self.logger.error("Failed to extract stablecoin features", error=str(e))
            return self._get_default_stablecoin_features()
    
    def _count_consecutive_depeg(self, prices: pd.Series, target: float, threshold: float) -> int:
        """Count consecutive hours of depeg from target price"""
        if prices is None or prices.empty:
            return 0
        
        depegged = abs(prices - target) > threshold
        count = 0
        
        # Count from the end backwards
        for i in range(len(depegged) - 1, -1, -1):
            if depegged.iloc[i]:
                count += 1
            else:
                break
        
        return count
    
    def _calculate_stable_mean_reversion(self, prices: pd.Series, target: float) -> float:
        """Calculate mean reversion strength towards target price (0-1)"""
        if prices is None or len(prices) < 6:
            return 0.5
        
        # Check if price is reverting to target
        lookback = min(6, len(prices))
        distance_now = abs(prices.iloc[-1] - target)
        distance_before = abs(prices.iloc[-lookback] - target)
        
        if distance_before > 0:
            reversion = 1 - (distance_now / distance_before)
            return max(0.0, min(1.0, reversion))
        
        return 0.5
    
    def _calculate_stable_stress_persistence(self, prices: pd.Series, target: float, threshold: float) -> float:
        """Calculate how persistent the stress is (0-1)"""
        if prices is None or len(prices) < 2:
            return 0.0
        
        # Measure how long price stays away from target
        lookback = min(24, len(prices))
        recent_prices = prices.tail(lookback)
        deviations = abs(recent_prices - target)
        persistence = (deviations > threshold).mean()
        
        return float(persistence)
    
    def _get_default_stablecoin_features(self) -> dict:
        """Default stablecoin features when data is insufficient"""
        return {
            'stable_current_depeg': 0.0,
            'stable_depeg_ratio': 1.0,
            'stable_max_depeg_24h': 0.0,
            'stable_depeg_direction': 0.0,
            'stable_hours_depegged': 0.0,
            'stable_consecutive_depeg': 0.0,
            'stable_depeg_volatility': 0.0,
            'stable_volume_spike_ratio': 1.0,
            'stable_volume_spike_24h': 1.0,
            'stable_panic_volume_score': 0.0,
            'stable_mean_reversion': 0.5,
            'stable_stress_persistence': 0.0,
            'stable_is_panic_mode': 0.0,
            'stable_is_premium': 0.0,
            'stable_is_discount': 0.0,
            'stable_arbitrage_opportunity': 0.0,
            'stable_profit_potential_bps': 0.0,
            'stable_range_ratio': 0.001,
            'stable_avg_range_24h': 0.001
        }
    
    def _get_default_block_features(self) -> dict:
        """Default block time features"""
        return {
            'chain_block_time': 15.0,
            'current_block_time_estimate': 15.0,
            'block_time_variability': 0.1,
            'network_congestion_estimate': 0.3,
            'blocks_per_hour': 240.0,
            'transaction_finality_time': 90.0,
            'chain_efficiency_score': 0.06
        }

    async def close(self):
        """Close market data aggregator and cleanup resources"""
        if self.market_aggregator:
            await self.market_aggregator.close()
            self.logger.info("Market data aggregator closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of feature engineering system including market data sources"""
        health_status = {
            "cache_size": len(self._indicator_cache),
            "market_cache_valid": self._market_cache is not None,
            "live_data_enabled": self.enable_live_data,
            "market_data_sources": {}
        }
        
        if self.market_aggregator:
            try:
                market_health = await self.market_aggregator.health_check()
                health_status["market_data_sources"] = market_health
                health_status["market_data_healthy"] = all(market_health.values())
            except Exception as e:
                health_status["market_data_error"] = str(e)
                health_status["market_data_healthy"] = False
        else:
            health_status["market_data_healthy"] = True  # Placeholder mode
        
        return health_status