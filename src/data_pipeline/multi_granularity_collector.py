"""
Multi-Granularity Corpus Collector
Collects training data at multiple timeframes for comprehensive market analysis

This module extends the InitialCorpusCollector to support multiple granularities
(daily, 4-hour, hourly, 15-minute) for each token to capture different market dynamics.
"""

import asyncio
import yaml
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import structlog
from dataclasses import dataclass
from enum import Enum
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector
from src.ml_analysis.market_data import CoinGeckoClient
from src.ml_analysis.feature_engineer import FeatureEngineer
from src.utils.database import get_database_connection, execute_query

logger = structlog.get_logger(__name__)


class Timeframe(Enum):
    """Supported timeframe granularities"""
    DAILY = "daily"
    FOUR_HOUR = "four_hour"
    HOURLY = "hourly"
    FIFTEEN_MINUTE = "fifteen_minute"


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe"""
    name: str
    interval: str  # 'day', 'hour', 'minute'
    aggregate: int  # aggregation factor
    days_back: int
    expected_candles: int
    features_focus: List[str]
    enabled: bool = True


@dataclass
class TokenConfig:
    """Configuration for a specific token"""
    symbol: str
    contract_address: Optional[str]
    network: Optional[str]
    coingecko_id: str
    category: str
    extract_stablecoin_features: bool = False


class MultiGranularityCollector(InitialCorpusCollector):
    """
    Extended corpus collector that collects data at multiple granularities
    
    This collector fetches OHLCV data at different timeframes to capture:
    - Long-term trends (daily)
    - Medium-term patterns (4-hour)
    - Short-term volatility (hourly)
    - Micro-structure (15-minute, optional)
    """
    
    def __init__(self, 
                 config_path: str = "config/corpus_collection.yaml",
                 **kwargs):
        """
        Initialize multi-granularity collector
        
        Args:
            config_path: Path to corpus collection configuration
            **kwargs: Additional arguments for parent InitialCorpusCollector
        """
        # Initialize parent with default args if not provided
        if 'collection_days' not in kwargs:
            kwargs['collection_days'] = 365  # Default to 1 year
        super().__init__(**kwargs)
        
        self.config = self._load_config(config_path)
        self.tokens = self._parse_tokens(self.config['tokens'])
        self.timeframes = self._parse_timeframes(self.config['timeframes'])
        self.collection_strategy = self.config.get('collection_strategy', {})
        self.feature_extraction = self.config.get('feature_extraction', {})
        
        # Initialize feature engineer with stablecoin support
        self.feature_engineer = FeatureEngineer(enable_live_data=False)
        
        logger.info(
            "Multi-granularity collector initialized",
            tokens=len(self.tokens),
            timeframes=len([tf for tf in self.timeframes.values() if tf.enabled]),
            version=self.config.get('corpus_version', 'v2.0')
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            # Use default configuration if file doesn't exist
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is missing"""
        return {
            'corpus_version': 'v2.0',
            'tokens': [
                {
                    'symbol': 'WETH',
                    'contract_address': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
                    'network': 'eth',
                    'coingecko_id': 'weth',
                    'category': 'wrapped_crypto'
                },
                {
                    'symbol': 'USDC',
                    'contract_address': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
                    'network': 'eth',
                    'coingecko_id': 'usd-coin',
                    'category': 'stablecoin',
                    'extract_stablecoin_features': True
                }
            ],
            'timeframes': {
                'daily': {
                    'interval': 'day',
                    'aggregate': 1,
                    'days_back': 365,
                    'expected_candles': 365,
                    'features_focus': ['trend_following'],
                    'enabled': True
                },
                'hourly': {
                    'interval': 'hour',
                    'aggregate': 1,
                    'days_back': 90,
                    'expected_candles': 2190,
                    'features_focus': ['intraday_patterns'],
                    'enabled': True
                }
            },
            'collection_strategy': {
                'parallel_collection': True,
                'max_concurrent_requests': 3,
                'rate_limit_per_minute': 30
            }
        }
    
    def _parse_tokens(self, tokens_config: List[Dict]) -> Dict[str, TokenConfig]:
        """Parse token configurations"""
        tokens = {}
        for token_dict in tokens_config:
            token = TokenConfig(
                symbol=token_dict['symbol'],
                contract_address=token_dict.get('contract_address'),
                network=token_dict.get('network'),
                coingecko_id=token_dict['coingecko_id'],
                category=token_dict.get('category', 'crypto'),
                extract_stablecoin_features=token_dict.get('extract_stablecoin_features', False)
            )
            tokens[token.symbol] = token
        return tokens
    
    def _parse_timeframes(self, timeframes_config: Dict) -> Dict[Timeframe, TimeframeConfig]:
        """Parse timeframe configurations"""
        timeframes = {}
        
        for tf_name, tf_dict in timeframes_config.items():
            timeframe_enum = Timeframe(tf_name)
            timeframe = TimeframeConfig(
                name=tf_name,
                interval=tf_dict['interval'],
                aggregate=tf_dict.get('aggregate', 1),
                days_back=tf_dict['days_back'],
                expected_candles=tf_dict['expected_candles'],
                features_focus=tf_dict.get('features_focus', []),
                enabled=tf_dict.get('enabled', True)
            )
            timeframes[timeframe_enum] = timeframe
        
        return timeframes
    
    async def collect_multi_granularity_corpus(self) -> Dict[str, pd.DataFrame]:
        """
        Collect corpus data at multiple granularities
        
        Returns:
            Dictionary mapping 'token_timeframe' to DataFrames with OHLCV + features
        """
        corpus_data = {}
        
        # Determine collection strategy
        if self.collection_strategy.get('parallel_collection', False):
            corpus_data = await self._collect_parallel()
        else:
            corpus_data = await self._collect_sequential()
        
        # Add cross-timeframe features if enabled
        if self.feature_extraction.get('enable_multi_scale_features', False):
            corpus_data = await self._add_multi_scale_features(corpus_data)
        
        logger.info(
            "Multi-granularity corpus collection completed",
            total_datasets=len(corpus_data),
            total_candles=sum(len(df) for df in corpus_data.values())
        )
        
        return corpus_data
    
    async def _collect_parallel(self) -> Dict[str, pd.DataFrame]:
        """Collect data for all tokens and timeframes in parallel"""
        tasks = []
        task_keys = []
        
        max_concurrent = self.collection_strategy.get('max_concurrent_requests', 3)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        for token_symbol, token_config in self.tokens.items():
            for timeframe_enum, timeframe_config in self.timeframes.items():
                if not timeframe_config.enabled:
                    continue
                
                key = f"{token_symbol}_{timeframe_enum.value}"
                task = self._collect_with_semaphore(
                    semaphore, 
                    token_config, 
                    timeframe_config
                )
                tasks.append(task)
                task_keys.append(key)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        corpus_data = {}
        for key, result in zip(task_keys, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to collect {key}", error=str(result))
                continue
            corpus_data[key] = result
        
        return corpus_data
    
    async def _collect_with_semaphore(self, 
                                     semaphore: asyncio.Semaphore,
                                     token: TokenConfig,
                                     timeframe: TimeframeConfig) -> pd.DataFrame:
        """Collect data with semaphore for rate limiting"""
        async with semaphore:
            return await self._collect_token_timeframe(token, timeframe)
    
    async def _collect_sequential(self) -> Dict[str, pd.DataFrame]:
        """Collect data for all tokens and timeframes sequentially"""
        corpus_data = {}
        
        for token_symbol, token_config in self.tokens.items():
            for timeframe_enum, timeframe_config in self.timeframes.items():
                if not timeframe_config.enabled:
                    continue
                
                key = f"{token_symbol}_{timeframe_enum.value}"
                try:
                    data = await self._collect_token_timeframe(token_config, timeframe_config)
                    corpus_data[key] = data
                    
                    logger.info(f"Collected {key}", candles=len(data))
                    
                except Exception as e:
                    logger.error(f"Failed to collect {key}", error=str(e))
                    if not self.collection_strategy.get('skip_on_error', False):
                        raise
        
        return corpus_data
    
    async def _collect_token_timeframe(self,
                                      token: TokenConfig,
                                      timeframe: TimeframeConfig) -> pd.DataFrame:
        """
        Collect OHLCV data for a specific token and timeframe
        
        Args:
            token: Token configuration
            timeframe: Timeframe configuration
            
        Returns:
            DataFrame with OHLCV data and extracted features
        """
        logger.info(
            f"Collecting {token.symbol} at {timeframe.name}",
            days_back=timeframe.days_back,
            expected_candles=timeframe.expected_candles
        )
        
        # Determine collection method based on token configuration
        if token.contract_address and self.coingecko_client:
            # Use contract OHLCV endpoint for better data quality
            ohlcv_data = await self._collect_contract_ohlcv(
                token, timeframe
            )
        else:
            # Use standard coin ID endpoint
            ohlcv_data = await self._collect_coin_ohlcv(
                token, timeframe
            )
        
        # Extract features
        features_df = await self._extract_features(
            ohlcv_data, token, timeframe
        )
        
        # Combine OHLCV with features
        result_df = pd.concat([ohlcv_data, features_df], axis=1)
        
        # Add metadata columns
        result_df['token_symbol'] = token.symbol
        result_df['timeframe'] = timeframe.name
        result_df['category'] = token.category
        
        return result_df
    
    async def _collect_contract_ohlcv(self,
                                     token: TokenConfig,
                                     timeframe: TimeframeConfig) -> pd.DataFrame:
        """Collect OHLCV using contract address endpoint"""
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=timeframe.days_back)
        
        # Map our timeframe interval to CoinGecko API timeframe
        # CoinGecko supports: 'day', 'hour', 'minute', 'second'
        if timeframe.interval == 'day':
            api_timeframe = 'day'
        elif timeframe.interval == 'hour':
            api_timeframe = 'hour'
        else:
            # For 4-hour or 15-minute, we fetch hourly and aggregate
            api_timeframe = 'hour'
        
        # Use the contract OHLCV method with pagination
        df = await self.coingecko_client._get_contract_ohlcv_with_pagination(
            contract_address=token.contract_address,
            network=token.network,
            days=timeframe.days_back,
            from_date=start_date,
            to_date=end_date,
            timeframe=api_timeframe  # Pass the explicit timeframe
        )
        
        # Aggregate if needed for non-native timeframes
        if timeframe.aggregate > 1 and not df.empty:
            # Aggregate for timeframes like 4-hour
            df = self._aggregate_ohlcv(df, timeframe.aggregate)
        
        return df
    
    async def _collect_coin_ohlcv(self,
                                 token: TokenConfig,
                                 timeframe: TimeframeConfig) -> pd.DataFrame:
        """Collect OHLCV using coin ID endpoint"""
        
        # For coin IDs, we need to map timeframe configs to API parameters
        if timeframe.interval == 'day':
            # Use daily endpoint
            df = await self.coingecko_client.get_ohlcv_data(
                coin_id=token.coingecko_id,
                days=timeframe.days_back
            )
        else:
            # For other intervals, try to use appropriate granularity
            # This is a simplified approach - you might need to adjust
            df = await self.coingecko_client.get_ohlcv_data(
                coin_id=token.coingecko_id,
                days=timeframe.days_back,
                interval='hourly' if timeframe.interval == 'hour' else 'daily'
            )
            
            # Aggregate if needed
            if timeframe.aggregate > 1:
                df = self._aggregate_ohlcv(df, timeframe.aggregate)
        
        return df
    
    def _aggregate_ohlcv(self, df: pd.DataFrame, aggregate: int) -> pd.DataFrame:
        """Aggregate OHLCV data to coarser timeframe"""
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Resample based on aggregation factor
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Create aggregation period string (e.g., '4H' for 4-hour)
        period = f"{aggregate}H"
        
        aggregated = df.resample(period).agg(agg_rules).dropna()
        aggregated.reset_index(inplace=True)
        
        return aggregated
    
    def _aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate hourly OHLCV data to daily candles
        
        Args:
            df: DataFrame with hourly OHLCV data
            
        Returns:
            DataFrame with daily OHLCV data
        """
        if df.empty:
            return df
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame doesn't have proper timestamp index for aggregation")
            return df
        
        # Aggregate hourly data to daily using proper OHLCV rules
        agg_rules = {
            'open': 'first',   # First price of the day
            'high': 'max',     # Highest price of the day
            'low': 'min',      # Lowest price of the day
            'close': 'last',   # Last price of the day
            'volume': 'sum'    # Total volume for the day
        }
        
        # Resample to daily frequency ('D' for calendar day)
        daily_df = df.resample('D').agg(agg_rules).dropna()
        
        # Reset index to have timestamp as a column
        daily_df.reset_index(inplace=True)
        
        logger.info(
            f"Aggregated {len(df)} hourly candles to {len(daily_df)} daily candles",
            hourly_start=df.index[0] if len(df) > 0 else None,
            hourly_end=df.index[-1] if len(df) > 0 else None,
            daily_start=daily_df['timestamp'].iloc[0] if len(daily_df) > 0 else None,
            daily_end=daily_df['timestamp'].iloc[-1] if len(daily_df) > 0 else None
        )
        
        return daily_df
    
    async def _extract_features(self,
                               ohlcv_data: pd.DataFrame,
                               token: TokenConfig,
                               timeframe: TimeframeConfig) -> pd.DataFrame:
        """
        Extract features from OHLCV data
        
        Includes:
        - Standard technical indicators
        - Timeframe-specific features
        - Stablecoin features (if applicable)
        """
        features_list = []
        
        for i in range(len(ohlcv_data)):
            # Get window of data for feature calculation
            window_end = min(i + 1, len(ohlcv_data))
            window_start = max(0, window_end - 100)  # Use up to 100 candles for features
            window_data = ohlcv_data.iloc[window_start:window_end]
            
            features = {}
            
            # Extract technical indicators
            # For testing with limited data, lower the threshold
            min_candles = 5 if len(ohlcv_data) < 30 else 20
            if len(window_data) >= min_candles:
                tech_features = await self._extract_technical_features(window_data)
                features.update(tech_features)
            else:
                # Add default values for technical features when not enough data
                features.update({
                    'rsi': 50.0,  # Neutral RSI
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0,
                    'bb_upper': 0.0,
                    'bb_lower': 0.0,
                    'bb_width': 0.0,
                    'atr': 0.0,
                    'volume_sma': 0.0,
                    'volume_ratio': 1.0
                })
            
            # Extract timeframe-specific features
            tf_features = self._extract_timeframe_features(
                window_data, timeframe
            )
            features.update(tf_features)
            
            # Extract stablecoin features if applicable
            if token.extract_stablecoin_features:
                stable_features = self.feature_engineer.extract_stablecoin_features(
                    window_data,
                    token_symbol=token.symbol
                )
                features.update(stable_features)
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    async def _extract_technical_features(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """Extract standard technical indicators using FeatureEngineer"""
        try:
            # Use FeatureEngineer's centralized calculation
            features = self.feature_engineer.calculate_features_for_corpus(
                ohlcv_data, 
                min_periods=True  # Enable adaptive periods for limited data
            )
            
            # Map some feature names to match expected format
            if 'rsi_14' in features:
                features['rsi'] = features['rsi_14']
            
            # Add volume_sma as alias for volume_sma_20
            if 'volume_sma_20' in features:
                features['volume_sma'] = features['volume_sma_20']
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default features on error
            return self.feature_engineer._get_default_feature_dict()
    
    def _extract_timeframe_features(self,
                                   ohlcv_data: pd.DataFrame,
                                   timeframe: TimeframeConfig) -> Dict[str, float]:
        """Extract timeframe-specific features"""
        features = {}
        
        if timeframe.name == 'daily':
            # Long-term trend features
            if len(ohlcv_data) >= 30:
                features['trend_30d'] = (ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[-30] - 1)
            else:
                features['trend_30d'] = 0.0
            
            if len(ohlcv_data) >= 7:
                features['weekly_return'] = (ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[-7] - 1)
            else:
                features['weekly_return'] = 0.0
                
        elif timeframe.name == 'four_hour':
            # Swing trading features
            if len(ohlcv_data) >= 6:  # 24 hours
                features['intraday_range'] = (ohlcv_data['high'].tail(6).max() - ohlcv_data['low'].tail(6).min()) / ohlcv_data['close'].iloc[-1]
                features['session_momentum'] = (ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[-6] - 1)
            else:
                features['intraday_range'] = 0.0
                features['session_momentum'] = 0.0
                
        elif timeframe.name == 'hourly':
            # Short-term volatility features
            if len(ohlcv_data) >= 24:
                features['hourly_volatility'] = ohlcv_data['close'].pct_change().tail(24).std()
                features['mean_reversion_score'] = (ohlcv_data['close'].iloc[-1] - ohlcv_data['close'].tail(24).mean()) / ohlcv_data['close'].tail(24).std() if ohlcv_data['close'].tail(24).std() > 0 else 0
            else:
                features['hourly_volatility'] = 0.0
                features['mean_reversion_score'] = 0.0
                
        elif timeframe.name == 'fifteen_minute':
            # Microstructure features
            if len(ohlcv_data) >= 4:  # 1 hour
                features['micro_volatility'] = ohlcv_data['close'].pct_change().tail(4).std()
                features['bid_ask_proxy'] = (ohlcv_data['high'].iloc[-1] - ohlcv_data['low'].iloc[-1]) / ohlcv_data['close'].iloc[-1]
            else:
                features['micro_volatility'] = 0.0
                features['bid_ask_proxy'] = 0.0
        
        return features
    
    
    async def _add_multi_scale_features(self, 
                                       corpus_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add cross-timeframe features for multi-scale learning"""
        
        # Group data by token
        tokens_data = {}
        for key, df in corpus_data.items():
            token = key.split('_')[0]
            if token not in tokens_data:
                tokens_data[token] = {}
            timeframe = '_'.join(key.split('_')[1:])
            tokens_data[token][timeframe] = df
        
        # Add multi-scale features for each token
        enhanced_corpus = {}
        for token, timeframes_dict in tokens_data.items():
            for timeframe, df in timeframes_dict.items():
                # Add features from other timeframes
                multi_scale_features = []
                
                for i in range(len(df)):
                    features = {}
                    current_time = df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i]
                    
                    # Get corresponding data points from other timeframes
                    for other_tf, other_df in timeframes_dict.items():
                        if other_tf == timeframe:
                            continue
                        
                        # Find nearest timestamp in other timeframe
                        if 'timestamp' in other_df.columns:
                            time_diff = abs(other_df['timestamp'] - current_time)
                            nearest_idx = time_diff.idxmin()
                        else:
                            time_diff = abs(other_df.index - current_time)
                            nearest_idx = time_diff.argmin()
                        
                        # Add cross-timeframe features
                        features[f'{other_tf}_close'] = other_df.iloc[nearest_idx]['close']
                        features[f'{other_tf}_volume'] = other_df.iloc[nearest_idx]['volume']
                        if f'{other_tf}_rsi' in other_df.columns:
                            features[f'{other_tf}_rsi'] = other_df.iloc[nearest_idx][f'{other_tf}_rsi']
                    
                    multi_scale_features.append(features)
                
                # Add multi-scale features to dataframe
                multi_scale_df = pd.DataFrame(multi_scale_features)
                enhanced_df = pd.concat([df, multi_scale_df], axis=1)
                enhanced_corpus[f"{token}_{timeframe}"] = enhanced_df
        
        return enhanced_corpus
    
    async def _store_ohlcv_with_granularity(self, conn, df: pd.DataFrame, 
                                           token_symbol: str, granularity: str):
        """Store OHLCV records with granularity"""
        if df.empty:
            return
        
        # Prepare data for insertion
        records = []
        for _, row in df.iterrows():
            # Get timestamp and ensure it's timezone-aware
            ts = row.get('timestamp', row.name if isinstance(row.name, pd.Timestamp) else None)
            if ts is not None and pd.notna(ts):
                # Convert to pandas Timestamp if needed
                if not isinstance(ts, pd.Timestamp):
                    ts = pd.to_datetime(ts)
                # Ensure timezone-aware (UTC)
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')
                # Convert to Python datetime for asyncpg
                ts = ts.to_pydatetime()
            else:
                # Skip records without valid timestamps
                continue
            
            record = {
                'token_id': token_symbol.lower(),
                'symbol': token_symbol,
                'timestamp': ts,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'granularity': granularity,  # Add granularity
                'data_source': 'initial',
                'collection_timestamp': datetime.now(timezone.utc)
            }
            records.append(record)
        
        # Insert records
        if records:
            await conn.executemany(
                """
                INSERT INTO crypto_ohlcv (
                    token_id, symbol, timestamp, open, high, low, close, volume,
                    granularity, data_source, collection_timestamp
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
                ON CONFLICT (id, data_source, timestamp) DO NOTHING
                """,
                [(r['token_id'], r['symbol'], r['timestamp'], r['open'], r['high'],
                  r['low'], r['close'], r['volume'], r['granularity'], 
                  r['data_source'], r['collection_timestamp']) for r in records]
            )
            logger.info(f"Stored {len(records)} OHLCV records for {token_symbol} at {granularity}")
    
    async def _store_features_with_granularity(self, conn, df: pd.DataFrame,
                                              token_symbol: str, granularity: str):
        """Store feature records with granularity"""
        if df.empty:
            return
        
        # Get OHLCV IDs first - we need them to link features
        ohlcv_ids = await conn.fetch("""
            SELECT id, timestamp 
            FROM crypto_ohlcv 
            WHERE symbol = $1 
              AND granularity = $2 
              AND data_source = 'initial'
            ORDER BY timestamp
        """, token_symbol, granularity)
        
        if not ohlcv_ids:
            logger.warning(f"No OHLCV records found for {token_symbol} at {granularity}")
            return
        
        # Create timestamp to ID mapping
        ts_to_id = {}
        for record in ohlcv_ids:
            # Normalize timestamp to match DataFrame timestamps
            ts = pd.Timestamp(record['timestamp']).tz_convert('UTC')
            ts_to_id[ts] = record['id']
        
        # Build feature records with proper mapping
        feature_records = []
        for _, row in df.iterrows():
            # Get timestamp
            ts = row.get('timestamp', row.name if isinstance(row.name, pd.Timestamp) else None)
            if ts is not None and pd.notna(ts):
                # Convert to pandas Timestamp if needed
                if not isinstance(ts, pd.Timestamp):
                    ts = pd.to_datetime(ts)
                # Ensure timezone-aware (UTC)
                if ts.tz is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')
            else:
                continue
            
            # Find matching OHLCV ID
            ohlcv_id = ts_to_id.get(ts)
            if not ohlcv_id:
                logger.debug(f"No OHLCV ID found for timestamp {ts}")
                continue
            
            # Convert timestamp to Python datetime for asyncpg
            ts_datetime = ts.to_pydatetime()
            
            # Helper function to get safe float value
            def safe_float(value, default=0.0):
                if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                    return default
                return float(value)
            
            # Build feature record with all available columns
            feature_records.append((
                ohlcv_id,  # $1: ohlcv_id
                token_symbol.lower(),  # $2: token_id
                ts_datetime,  # $3: timestamp
                safe_float(row.get('rsi', row.get('rsi_14', 50.0)), 50.0),  # $4: rsi_14
                safe_float(row.get('macd', 0.0)),  # $5: macd
                safe_float(row.get('macd_signal', 0.0)),  # $6: macd_signal
                safe_float(row.get('macd_histogram', 0.0)),  # $7: macd_histogram
                safe_float(row.get('bb_upper', 0.0)),  # $8: bb_upper
                safe_float(row.get('bb_middle', 0.0)),  # $9: bb_middle
                safe_float(row.get('bb_lower', 0.0)),  # $10: bb_lower
                safe_float(row.get('volume_sma', row.get('volume_sma_20', 0.0))),  # $11: volume_sma_20
                safe_float(row.get('ema_12', 0.0)),  # $12: ema_12
                safe_float(row.get('ema_26', 0.0)),  # $13: ema_26
                safe_float(row.get('ema_50', 0.0)),  # $14: ema_50
                safe_float(row.get('ema_200', 0.0)),  # $15: ema_200
                safe_float(row.get('sma_20', 0.0)),  # $16: sma_20
                safe_float(row.get('sma_50', 0.0)),  # $17: sma_50
                safe_float(row.get('sma_200', 0.0)),  # $18: sma_200
                safe_float(row.get('atr', 0.0)),  # $19: atr
                safe_float(row.get('adx', 0.0)),  # $20: adx
                safe_float(row.get('returns_1h', 0.0)),  # $21: returns_1h
                safe_float(row.get('returns_24h', 0.0)),  # $22: returns_24h
                safe_float(row.get('returns_7d', 0.0)),  # $23: returns_7d
                safe_float(row.get('volatility_24h', 0.0)),  # $24: volatility_24h
                safe_float(row.get('price_change_1h', 0.0)),  # $25: price_change_1h
                safe_float(row.get('price_change_24h', 0.0)),  # $26: price_change_24h
                safe_float(row.get('price_change_7d', 0.0)),  # $27: price_change_7d
                '1.0',  # $28: feature_version
                datetime.now(timezone.utc),  # $29: calculated_at
                'initial',  # $30: data_source
                granularity  # $31: granularity
            ))
        
        # Insert features into database
        if feature_records:
            query = """
                INSERT INTO crypto_features (
                    ohlcv_id, token_id, timestamp,
                    rsi_14, macd, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower,
                    volume_sma_20,
                    ema_12, ema_26, ema_50, ema_200,
                    sma_20, sma_50, sma_200,
                    atr, adx,
                    returns_1h, returns_24h, returns_7d,
                    volatility_24h,
                    price_change_1h, price_change_24h, price_change_7d,
                    feature_version, calculated_at, data_source, granularity
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, 
                         $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31)
            """
            
            await conn.executemany(query, feature_records)
            logger.info(f"Stored {len(feature_records)} feature records for {token_symbol} at {granularity}")
    
    async def store_corpus_to_database(self, corpus_data: Dict[str, pd.DataFrame]):
        """
        Store multi-granularity corpus to database
        
        Args:
            corpus_data: Dictionary of DataFrames with OHLCV and features
        """
        total_records = 0
        
        async with get_database_connection() as conn:
            for key, df in corpus_data.items():
                token_symbol = key.split('_')[0]
                granularity = '_'.join(key.split('_')[1:])
                
                logger.info(f"Storing {key} to database", records=len(df))
                
                # Store OHLCV records with granularity
                await self._store_ohlcv_with_granularity(conn, df, token_symbol, granularity)
                
                # Store feature records with granularity
                await self._store_features_with_granularity(conn, df, token_symbol, granularity)
                
                total_records += len(df)
        
        # Create corpus version record
        # Note: Parent class method expects different parameters
        # For now, we'll skip creating corpus version here since it requires more context
        # TODO: Override _create_corpus_version for multi-granularity support
        
        logger.info(
            "Corpus stored to database",
            total_records=total_records,
            datasets=len(corpus_data)
        )
    
    async def export_to_parquet(
        self, 
        corpus_data: Dict[str, pd.DataFrame], 
        local_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Export corpus data to Parquet files
        
        Args:
            corpus_data: Dictionary of DataFrames with OHLCV and features
            local_dir: Local directory to save files (default: data/corpus/v2.0)
            
        Returns:
            Dictionary mapping dataset keys to Parquet file paths
        """
        if local_dir is None:
            local_dir = Path("data/corpus/v2.0")
        
        local_dir.mkdir(parents=True, exist_ok=True)
        parquet_paths = {}
        
        for key, df in corpus_data.items():
            # Parse key to get token and timeframe
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                token, timeframe = parts
            else:
                token = key
                timeframe = "unknown"
            
            # Create subdirectory for timeframe
            timeframe_dir = local_dir / timeframe
            timeframe_dir.mkdir(exist_ok=True)
            
            # Generate filename
            filename = f"{token}_{timeframe}.parquet"
            filepath = timeframe_dir / filename
            
            # Write to Parquet with compression
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table, 
                filepath,
                compression='snappy',  # Good balance of speed and compression
                use_dictionary=True,  # Enable dictionary encoding for strings
                coerce_timestamps='ms',  # Millisecond precision for timestamps
                allow_truncated_timestamps=True
            )
            
            parquet_paths[key] = filepath
            logger.info(
                f"Exported {key} to Parquet",
                path=str(filepath),
                rows=len(df),
                size_mb=filepath.stat().st_size / 1024 / 1024
            )
        
        return parquet_paths
    
    async def export_to_gcs(
        self,
        corpus_data: Dict[str, pd.DataFrame],
        bucket_name: str = "shyvr-models-prod",
        gcs_prefix: str = "training-data/initial-corpus/v2.0"
    ) -> Dict[str, str]:
        """
        Export corpus data to Google Cloud Storage as Parquet files
        
        Args:
            corpus_data: Dictionary of DataFrames with OHLCV and features
            bucket_name: GCS bucket name
            gcs_prefix: Prefix for GCS paths
            
        Returns:
            Dictionary mapping dataset keys to GCS paths
        """
        # First export to local Parquet files
        local_paths = await self.export_to_parquet(corpus_data)
        
        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        gcs_paths = {}
        total_size = 0
        
        for key, local_path in local_paths.items():
            # Parse key to get token and timeframe
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                token, timeframe = parts
            else:
                token = key
                timeframe = "unknown"
            
            # Create GCS path
            gcs_path = f"{gcs_prefix}/{timeframe}/{token}_{timeframe}.parquet"
            blob = bucket.blob(gcs_path)
            
            # Upload file
            blob.upload_from_filename(str(local_path))
            
            gcs_paths[key] = f"gs://{bucket_name}/{gcs_path}"
            file_size = local_path.stat().st_size
            total_size += file_size
            
            logger.info(
                f"Uploaded {key} to GCS",
                gcs_path=gcs_paths[key],
                size_mb=file_size / 1024 / 1024
            )
        
        # Create metadata file
        metadata = {
            "corpus_version": self.corpus_version,
            "collection_name": self.collection_name,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "tokens": list(self.tokens.keys()),
            "timeframes": [tf.value for tf, config in self.timeframes.items() if config.enabled],
            "total_datasets": len(corpus_data),
            "total_records": sum(len(df) for df in corpus_data.values()),
            "total_size_mb": total_size / 1024 / 1024,
            "gcs_paths": gcs_paths
        }
        
        # Save metadata
        metadata_path = Path("data/corpus/v2.0/metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Upload metadata to GCS
        metadata_blob = bucket.blob(f"{gcs_prefix}/metadata.json")
        metadata_blob.upload_from_filename(str(metadata_path))
        
        logger.info(
            "Corpus export to GCS completed",
            total_datasets=len(gcs_paths),
            total_size_mb=total_size / 1024 / 1024,
            metadata_path=f"gs://{bucket_name}/{gcs_prefix}/metadata.json"
        )
        
        return gcs_paths