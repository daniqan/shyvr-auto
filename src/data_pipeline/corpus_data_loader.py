"""
Corpus Data Loader
Loads and prepares corpus data for ML model training
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import structlog

from src.utils.database import get_database_connection


logger = structlog.get_logger(__name__)


class CorpusDataLoader:
    """
    Loads corpus data from database or Parquet files and prepares it for ML training
    """
    
    def __init__(self, data_source: str = 'initial'):
        """
        Initialize corpus data loader
        
        Args:
            data_source: Source identifier for corpus data (default: 'initial')
        """
        self.data_source = data_source
        self.logger = logger.bind(component="CorpusDataLoader")
    
    async def load_from_database(self, 
                                token_id: str, 
                                granularity: str = 'daily',
                                limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load corpus data from database and join OHLCV with features
        
        Args:
            token_id: Token identifier (e.g., 'weth', 'usdc')
            granularity: Time granularity ('daily', 'hourly', etc.)
            limit: Optional limit on number of records
            
        Returns:
            DataFrame with OHLCV and features combined
        """
        async with get_database_connection() as conn:
            # Build query to join OHLCV and features
            query = """
                SELECT 
                    o.timestamp,
                    o.open, o.high, o.low, o.close, o.volume,
                    o.market_cap, o.circulating_supply,
                    f.rsi_14, f.macd, f.macd_signal, f.macd_histogram,
                    f.bb_upper, f.bb_middle, f.bb_lower,
                    f.sma_20, f.sma_50, f.sma_200,
                    f.ema_12, f.ema_26, f.ema_50, f.ema_200,
                    f.atr, f.adx,
                    f.volume_sma_20, f.volume_ratio,
                    f.obv,
                    f.returns_1h, f.returns_24h, f.returns_7d,
                    f.volatility_24h,
                    f.price_change_1h, f.price_change_24h, f.price_change_7d
                FROM crypto_ohlcv o
                LEFT JOIN crypto_features f ON o.id = f.ohlcv_id
                WHERE o.token_id = $1
                  AND o.granularity = $2
                  AND o.data_source = $3
                ORDER BY o.timestamp
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            records = await conn.fetch(query, token_id.lower(), granularity, self.data_source)
            
            if not records:
                self.logger.warning("No corpus data found", 
                                  token_id=token_id, 
                                  granularity=granularity)
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Add calculated features
            df = self._add_calculated_features(df)
            
            self.logger.info("Loaded corpus data from database",
                           token_id=token_id,
                           granularity=granularity,
                           records=len(df))
            
            return df
    
    def load_from_parquet(self, 
                         filepath: Path,
                         add_features: bool = True) -> pd.DataFrame:
        """
        Load corpus data from Parquet file
        
        Args:
            filepath: Path to Parquet file
            add_features: Whether to add calculated features
            
        Returns:
            DataFrame with corpus data
        """
        try:
            # Read Parquet file
            df = pq.read_table(filepath).to_pandas()
            
            # Add calculated features if requested
            if add_features:
                df = self._add_calculated_features(df)
            
            self.logger.info("Loaded corpus data from Parquet",
                           filepath=str(filepath),
                           records=len(df))
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to load Parquet file", 
                            filepath=str(filepath),
                            error=str(e))
            return pd.DataFrame()
    
    def _add_calculated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated features that models expect but aren't stored
        
        Args:
            df: DataFrame with basic corpus data
            
        Returns:
            DataFrame with additional calculated features
        """
        if df.empty:
            return df
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Trading session (simplified)
        df['trading_session'] = df['hour'].apply(self._get_trading_session)
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_price_ratio'] = df['volume'] / df['close']
        
        # Bollinger Band position
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bollinger_width'] = df['bb_upper'] - df['bb_lower']
            df['bollinger_position'] = (df['close'] - df['bb_lower']) / df['bollinger_width']
            df['bollinger_position'] = df['bollinger_position'].clip(0, 1)
        
        # Additional moving averages if missing
        if 'sma_5' not in df.columns:
            df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        if 'sma_10' not in df.columns:
            df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        
        # Returns if missing
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50, min_periods=1).mean()
        
        # Price momentum (custom indicator)
        df['price_momentum'] = df['close'].pct_change(periods=10) * 100
        
        # Volatility score (normalized volatility)
        df['volatility_score'] = df['volatility'] / df['volatility'].max() if df['volatility'].max() > 0 else 0
        
        # Market regime (simplified - would need market data for real implementation)
        df['market_regime'] = self._classify_market_regime(df)
        
        # Fill NaN values
        df = df.fillna(method='forward').fillna(method='backward').fillna(0)
        
        return df
    
    def _get_trading_session(self, hour: int) -> int:
        """
        Classify hour into trading session
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Trading session identifier (0-3)
        """
        if 0 <= hour < 6:
            return 0  # Asian session
        elif 6 <= hour < 12:
            return 1  # European session
        elif 12 <= hour < 18:
            return 2  # US session
        else:
            return 3  # After hours
    
    def _classify_market_regime(self, df: pd.DataFrame) -> np.ndarray:
        """
        Classify market regime based on price action
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Array of market regime classifications
        """
        # Simple regime classification based on moving averages
        regimes = []
        
        for i in range(len(df)):
            if i < 20:
                regimes.append(1)  # Neutral (not enough data)
            else:
                # Compare current price to moving averages
                price = df['close'].iloc[i]
                sma_20 = df['sma_20'].iloc[i] if 'sma_20' in df.columns else price
                sma_50 = df['sma_50'].iloc[i] if 'sma_50' in df.columns else price
                
                if price > sma_20 > sma_50:
                    regimes.append(2)  # Bullish
                elif price < sma_20 < sma_50:
                    regimes.append(0)  # Bearish
                else:
                    regimes.append(1)  # Neutral/Choppy
        
        return np.array(regimes)
    
    async def prepare_for_lstm(self, 
                              df: pd.DataFrame,
                              sequence_length: int = 60,
                              prediction_horizons: List[int] = [1, 4, 24]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare corpus data for LSTM training
        
        Args:
            df: Corpus DataFrame
            sequence_length: Length of input sequences
            prediction_horizons: Hours ahead to predict
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Select features LSTM expects
        lstm_features = [
            'open', 'high', 'low', 'close', 'volume',
            'price_change', 'high_low_ratio', 'volume_price_ratio',
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'atr',
            'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'bollinger_position',
            'volume_sma_20', 'volume_ratio', 'obv',
            'returns', 'volatility', 'volatility_ratio',
            'hour', 'day_of_week'
        ]
        
        # Filter to available features
        available_features = [f for f in lstm_features if f in df.columns]
        feature_data = df[available_features].values
        
        # Create sequences
        X = []
        y = []
        
        for i in range(sequence_length, len(df) - max(prediction_horizons)):
            # Input sequence
            X.append(feature_data[i-sequence_length:i])
            
            # Target prices
            targets = []
            for h in prediction_horizons:
                if i + h < len(df):
                    targets.append(df['close'].iloc[i + h])
                else:
                    targets.append(df['close'].iloc[-1])
            
            y.append(targets)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), available_features
    
    async def prepare_for_transformer(self,
                                     df: pd.DataFrame,
                                     sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare corpus data for Transformer training
        
        Args:
            df: Corpus DataFrame
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        # Create sliding windows
        for i in range(sequence_length, len(df) - 24):
            # Input sequence
            seq_data = df.iloc[i-sequence_length:i]
            
            # Normalize and extract features
            seq_features = []
            for _, row in seq_data.iterrows():
                features = [
                    row.get('close', 0) / 100.0,  # Normalized price
                    row.get('volume', 0) / 1000000.0,  # Normalized volume
                    row.get('high', 0) / 100.0,
                    row.get('low', 0) / 100.0,
                    row.get('open', 0) / 100.0,
                    row.get('rsi_14', 50) / 100.0,  # Normalized RSI
                    row.get('macd', 0) / 100.0,
                    row.get('atr', 0) / 100.0,
                    row.get('volume_ratio', 1.0),
                    row.get('hour', 12) / 24.0,  # Normalized hour
                    row.get('day_of_week', 3) / 7.0,  # Normalized day
                    row.get('volatility_score', 0.5),
                    row.get('price_momentum', 0) / 100.0,
                    row.get('bollinger_position', 0.5),
                    row.get('trading_session', 1) / 4.0,
                ]
                seq_features.append(features)
            
            sequences.append(seq_features)
            
            # Target values (relative price changes)
            current_price = df.iloc[i]['close']
            target_1h = (df.iloc[i+1]['close'] - current_price) / current_price if i+1 < len(df) else 0
            target_4h = (df.iloc[i+4]['close'] - current_price) / current_price if i+4 < len(df) else 0
            target_24h = (df.iloc[i+24]['close'] - current_price) / current_price if i+24 < len(df) else 0
            
            targets.append([target_1h, target_4h, target_24h])
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    async def load_multi_token_corpus(self, 
                                     tokens: List[str],
                                     granularity: str = 'daily') -> Dict[str, pd.DataFrame]:
        """
        Load corpus data for multiple tokens
        
        Args:
            tokens: List of token IDs
            granularity: Time granularity
            
        Returns:
            Dictionary mapping token_id to DataFrame
        """
        corpus_data = {}
        
        for token_id in tokens:
            df = await self.load_from_database(token_id, granularity)
            if not df.empty:
                corpus_data[token_id] = df
        
        self.logger.info("Loaded multi-token corpus",
                       tokens=len(corpus_data),
                       granularity=granularity)
        
        return corpus_data