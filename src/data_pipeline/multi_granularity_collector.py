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
        
        # Use the contract OHLCV method with pagination
        df = await self.coingecko_client._get_contract_ohlcv_with_pagination(
            contract_address=token.contract_address,
            network=token.network,
            days=timeframe.days_back,
            from_date=start_date,
            to_date=end_date,
            timeframe=timeframe.interval,
            aggregate=timeframe.aggregate
        )
        
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
            if len(window_data) >= 20:  # Minimum for most indicators
                tech_features = await self._extract_technical_features(window_data)
                features.update(tech_features)
            
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
        """Extract standard technical indicators"""
        features = {}
        
        try:
            # RSI
            features['rsi'] = self._calculate_rsi(ohlcv_data['close'])
            
            # MACD
            macd_result = self._calculate_macd(ohlcv_data['close'])
            features.update(macd_result)
            
            # Bollinger Bands
            bb_result = self._calculate_bollinger_bands(ohlcv_data['close'])
            features.update(bb_result)
            
            # ATR
            features['atr'] = self._calculate_atr(ohlcv_data)
            
            # Volume features
            features['volume_sma'] = ohlcv_data['volume'].rolling(20).mean().iloc[-1]
            features['volume_ratio'] = ohlcv_data['volume'].iloc[-1] / features['volume_sma'] if features['volume_sma'] > 0 else 1.0
            
        except Exception as e:
            logger.warning(f"Failed to extract technical features: {e}")
        
        return features
    
    def _extract_timeframe_features(self,
                                   ohlcv_data: pd.DataFrame,
                                   timeframe: TimeframeConfig) -> Dict[str, float]:
        """Extract timeframe-specific features"""
        features = {}
        
        if timeframe.name == 'daily':
            # Long-term trend features
            if len(ohlcv_data) >= 30:
                features['trend_30d'] = (ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[-30] - 1)
            if len(ohlcv_data) >= 7:
                features['weekly_return'] = (ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[-7] - 1)
                
        elif timeframe.name == 'four_hour':
            # Swing trading features
            if len(ohlcv_data) >= 6:  # 24 hours
                features['intraday_range'] = (ohlcv_data['high'].tail(6).max() - ohlcv_data['low'].tail(6).min()) / ohlcv_data['close'].iloc[-1]
                features['session_momentum'] = (ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[-6] - 1)
                
        elif timeframe.name == 'hourly':
            # Short-term volatility features
            if len(ohlcv_data) >= 24:
                features['hourly_volatility'] = ohlcv_data['close'].pct_change().tail(24).std()
                features['mean_reversion_score'] = (ohlcv_data['close'].iloc[-1] - ohlcv_data['close'].tail(24).mean()) / ohlcv_data['close'].tail(24).std() if ohlcv_data['close'].tail(24).std() > 0 else 0
                
        elif timeframe.name == 'fifteen_minute':
            # Microstructure features
            if len(ohlcv_data) >= 4:  # 1 hour
                features['micro_volatility'] = ohlcv_data['close'].pct_change().tail(4).std()
                features['bid_ask_proxy'] = (ohlcv_data['high'].iloc[-1] - ohlcv_data['low'].iloc[-1]) / ohlcv_data['close'].iloc[-1]
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate MACD indicators"""
        if len(prices) < 26:
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
        
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': float(macd.iloc[-1]),
            'macd_signal': float(signal.iloc[-1]),
            'macd_histogram': float(histogram.iloc[-1])
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {'bb_upper': 0.0, 'bb_lower': 0.0, 'bb_width': 0.0}
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        width = upper - lower
        
        return {
            'bb_upper': float(upper.iloc[-1]),
            'bb_lower': float(lower.iloc[-1]),
            'bb_width': float(width.iloc[-1])
        }
    
    def _calculate_atr(self, ohlcv_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(ohlcv_data) < period + 1:
            return 0.0
        
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        close = ohlcv_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
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
    
    async def store_corpus_to_database(self, corpus_data: Dict[str, pd.DataFrame]):
        """
        Store multi-granularity corpus to database
        
        Args:
            corpus_data: Dictionary of DataFrames with OHLCV and features
        """
        total_records = 0
        
        for key, df in corpus_data.items():
            token_symbol = key.split('_')[0]
            timeframe = '_'.join(key.split('_')[1:])
            
            logger.info(f"Storing {key} to database", records=len(df))
            
            # Store OHLCV records
            await self._store_ohlcv_records(df, token_symbol, timeframe)
            
            # Store feature records
            await self._store_feature_records(df, token_symbol, timeframe)
            
            total_records += len(df)
        
        # Create corpus version record
        await self._create_corpus_version(
            total_records=total_records,
            tokens=list(self.tokens.keys()),
            timeframes=[tf.value for tf in self.timeframes.keys() if self.timeframes[tf].enabled]
        )
        
        logger.info(
            "Corpus stored to database",
            total_records=total_records,
            datasets=len(corpus_data)
        )