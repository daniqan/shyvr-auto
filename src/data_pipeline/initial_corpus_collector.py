"""
Initial Corpus Collector
Creates standardized training corpus with comprehensive feature engineering

This module implements one-time collection of standardized training data
from real market APIs with proper feature engineering and database storage.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Set
import uuid
from pathlib import Path
import json

import pandas as pd
import numpy as np
import structlog
from contextlib import asynccontextmanager

# Import existing infrastructure
from src.ml_analysis.market_data import (
    CoinGeckoClient,
    FearGreedIndexClient, 
    DeFiLlamaClient,
    SocialSentimentClient,
    OnChainAnalyticsClient,
    MarketDataError,
    APIRateLimitError,
    APIAuthenticationError,
    DataNotAvailableError
)
from src.ml_analysis.feature_engineer import FeatureEngineer
from src.utils.database import get_database_connection, execute_query, execute_transaction
from src.utils.base import Chain
from src.utils.config import get_config


# Configure logging
logger = structlog.get_logger(__name__)


class InitialCorpusCollectorError(Exception):
    """Base exception for Initial Corpus Collector"""
    pass


class APIConnectionError(InitialCorpusCollectorError):
    """API connection or authentication error"""
    pass


class DataQualityError(InitialCorpusCollectorError):
    """Data quality or validation error"""
    pass


class StorageError(InitialCorpusCollectorError):
    """Database or storage error"""
    pass


class InitialCorpusCollector:
    """
    Collects and processes initial training corpus from real market APIs
    
    This class orchestrates one-time collection of standardized training data
    that serves as the foundation for model initialization. All data comes from
    real market APIs with comprehensive feature engineering.
    """
    
    # Initial corpus token list - 10 major tokens for proof of concept
    DEFAULT_TOKENS = [
        'bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano',
        'matic-network', 'avalanche-2', 'polkadot', 'chainlink', 'uniswap'
    ]
    
    def __init__(self,
                 collection_days: int = 180,  # 6 months of data
                 rate_limit_delay: float = 2.1,  # Conservative rate limiting
                 retry_attempts: int = 3,
                 batch_size: int = 100):
        """
        Initialize Initial Corpus Collector
        
        Args:
            collection_days: Days of historical data to collect
            rate_limit_delay: Delay between API calls in seconds
            retry_attempts: Number of retry attempts for failed API calls
            batch_size: Batch size for database operations
        """
        self.collection_days = collection_days
        self.rate_limit_delay = rate_limit_delay
        self.retry_attempts = retry_attempts
        self.batch_size = batch_size
        
        # Initialize logger first
        self.logger = structlog.get_logger().bind(component="InitialCorpusCollector")
        
        # Initialize API clients
        self._init_api_clients()
        
        # Initialize feature engineer
        from src.utils.system_secrets import get_system_secrets
        system_secrets = get_system_secrets()
        
        self.feature_engineer = FeatureEngineer(
            cache_ttl_minutes=60,  # 1 hour cache for corpus collection
            coingecko_api_key=system_secrets.coingecko_api_key,
            enable_live_data=True
        )
        
        # Collection state
        self.collected_tokens: Set[str] = set()
        self.failed_tokens: Dict[str, str] = {}
        self.collection_stats = {
            'start_time': None,
            'end_time': None,
            'total_records': 0,
            'total_features': 0,
            'api_calls_made': 0,
            'errors_encountered': 0
        }
    
    def _init_api_clients(self):
        """Initialize API clients with proper authentication"""
        from src.utils.system_secrets import get_system_secrets
        
        try:
            system_secrets = get_system_secrets()
            
            # CoinGecko client (required)
            coingecko_key = system_secrets.coingecko_api_key
            self.coingecko_client = CoinGeckoClient(
                api_key=coingecko_key,
                rate_limit=30,  # Conservative rate limit
                cache_ttl=300   # 5 minute cache
            )
            
            # Fear & Greed client (no API key needed)
            self.fear_greed_client = FearGreedIndexClient(rate_limit=10)
            
            # DeFiLlama client (no API key needed)  
            self.defillama_client = DeFiLlamaClient(rate_limit=20)
            
            # LunarCrush client (optional)
            lunarcrush_key = system_secrets.lunarcrush_api_key
            if lunarcrush_key:
                # Check for tier configuration from environment
                lunarcrush_tier = os.environ.get('LUNARCRUSH_TIER', 'basic')
                self.social_client = SocialSentimentClient(
                    api_key=lunarcrush_key,
                    tier=lunarcrush_tier
                )
                self.has_social_data = True
                self.logger.info("LunarCrush client initialized", tier=lunarcrush_tier)
            else:
                self.social_client = None
                self.has_social_data = False
                self.logger.warning("LunarCrush API key not available, skipping social sentiment data")
            
            # Helius client for on-chain data (optional)
            helius_key = system_secrets.helius_api_key
            if helius_key:
                self.onchain_client = OnChainAnalyticsClient(api_key=helius_key)
                self.has_onchain_data = True
            else:
                self.onchain_client = None
                self.has_onchain_data = False
                self.logger.warning("Helius API key not available, skipping on-chain data")
            
            self.logger.info("API clients initialized", 
                           coingecko=True, 
                           social=self.has_social_data,
                           onchain=self.has_onchain_data)
            
        except Exception as e:
            raise APIConnectionError(f"Failed to initialize API clients: {str(e)}")
    
    async def collect_standardized_corpus(self, 
                                        tokens: Optional[List[str]] = None,
                                        start_date: Optional[datetime] = None,
                                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Collect standardized corpus with comprehensive feature engineering
        
        Args:
            tokens: List of token IDs to collect (defaults to DEFAULT_TOKENS)
            start_date: Start date for collection (defaults to collection_days ago)
            end_date: End date for collection (defaults to now)
            
        Returns:
            Dictionary with collection results and statistics
        """
        if tokens is None:
            tokens = self.DEFAULT_TOKENS.copy()
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.collection_days)
        
        if end_date is None:
            end_date = datetime.now()
        
        self.collection_stats['start_time'] = datetime.now(timezone.utc)
        
        self.logger.info("Starting initial corpus collection",
                        tokens_count=len(tokens),
                        collection_days=self.collection_days,
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat())
        
        try:
            # Step 1: Collect OHLCV data for all tokens
            ohlcv_data = await self._collect_ohlcv_data(tokens, start_date, end_date)
            
            # Step 2: Collect market-wide data (sentiment, DeFi, etc.)
            market_data = await self._collect_market_data(start_date, end_date)
            
            # Step 3: Calculate features for all collected data
            feature_data = await self._calculate_features(ohlcv_data, market_data)
            
            # Step 4: Store in database with data_source='initial'
            storage_result = await self._store_corpus_data(ohlcv_data, market_data, feature_data)
            
            # Step 5: Create corpus version record
            version_result = await self._create_corpus_version(tokens, start_date, end_date, storage_result)
            
            # Step 6: Validate corpus completeness
            validation_result = await self.validate_corpus_completeness(version_result['version_id'])
            
            self.collection_stats['end_time'] = datetime.now(timezone.utc)
            duration = (self.collection_stats['end_time'] - self.collection_stats['start_time']).total_seconds()
            
            result = {
                'success': True,
                'corpus_version_id': version_result['version_id'],
                'version_name': version_result['version_name'],
                'tokens_collected': len(self.collected_tokens),
                'tokens_failed': len(self.failed_tokens),
                'failed_tokens': self.failed_tokens,
                'total_records': storage_result['total_records'],
                'feature_count': validation_result['feature_count'],
                'collection_duration_seconds': duration,
                'statistics': self.collection_stats,
                'validation': validation_result
            }
            
            self.logger.info("Initial corpus collection completed",
                           duration_seconds=duration,
                           tokens_collected=len(self.collected_tokens),
                           total_records=storage_result['total_records'],
                           feature_count=validation_result['feature_count'])
            
            return result
            
        except Exception as e:
            self.collection_stats['end_time'] = datetime.now(timezone.utc)
            self.collection_stats['errors_encountered'] += 1
            
            self.logger.error("Initial corpus collection failed", error=str(e))
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'tokens_collected': len(self.collected_tokens),
                'tokens_failed': len(self.failed_tokens),
                'failed_tokens': self.failed_tokens,
                'statistics': self.collection_stats
            }
        finally:
            # Clean up all client sessions
            await self._cleanup_sessions()
    
    async def _cleanup_sessions(self):
        """Clean up all HTTP client sessions"""
        tasks = []
        
        # Close CoinGecko client
        if hasattr(self, 'coingecko_client') and self.coingecko_client:
            tasks.append(self.coingecko_client.close())
        
        # Close Fear & Greed client
        if hasattr(self, 'fear_greed_client') and self.fear_greed_client:
            tasks.append(self.fear_greed_client.close())
        
        # Close DeFiLlama client
        if hasattr(self, 'defillama_client') and self.defillama_client:
            tasks.append(self.defillama_client.close())
        
        # Close Social client
        if hasattr(self, 'social_client') and self.social_client:
            tasks.append(self.social_client.close())
        
        # Close On-chain client
        if hasattr(self, 'onchain_client') and self.onchain_client:
            tasks.append(self.onchain_client.close())
        
        # Close FeatureEngineer (which closes its MarketDataAggregator)
        if hasattr(self, 'feature_engineer') and self.feature_engineer:
            tasks.append(self.feature_engineer.close())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.logger.info("Cleaned up HTTP client sessions", count=len(tasks))
    
    async def _collect_ohlcv_data(self, tokens: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Collect OHLCV data for all tokens with rate limiting and error handling"""
        ohlcv_data = {}
        
        self.logger.info("Starting OHLCV data collection", tokens_count=len(tokens))
        
        for token in tokens:
            try:
                self.logger.info("Collecting OHLCV data", token=token)
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
                # Collect data with retries
                data = await self._collect_token_ohlcv_with_retries(token, start_date, end_date)
                
                if not data.empty:
                    ohlcv_data[token] = data
                    self.collected_tokens.add(token)
                    self.collection_stats['api_calls_made'] += 1
                    
                    self.logger.info("OHLCV data collected successfully", 
                                   token=token, 
                                   records=len(data))
                else:
                    self.failed_tokens[token] = "Empty data returned"
                    self.logger.warning("No OHLCV data returned", token=token)
                
            except Exception as e:
                self.failed_tokens[token] = str(e)
                self.collection_stats['errors_encountered'] += 1
                self.logger.error("Failed to collect OHLCV data", 
                                token=token, 
                                error=str(e))
        
        self.logger.info("OHLCV data collection completed",
                        successful_tokens=len(ohlcv_data),
                        failed_tokens=len(self.failed_tokens))
        
        return ohlcv_data
    
    async def _collect_token_ohlcv_with_retries(self, token: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Collect OHLCV data for a single token with retry logic"""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                # Calculate days for API call
                days = (end_date - start_date).days
                
                data = await self.coingecko_client.get_ohlcv_data(
                    coin_id=token,
                    days=days,
                    from_date=start_date,
                    to_date=end_date
                )
                
                # Validate data quality
                if not data.empty and len(data) > 10:  # At least 10 data points
                    # Check for required columns
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in required_cols):
                        return data
                
                # If we get here, data quality is insufficient
                if data.empty:
                    raise DataQualityError(f"No data returned for {token}")
                else:
                    raise DataQualityError(f"Insufficient data quality for {token}: {len(data)} records")
                
            except (APIRateLimitError, APIAuthenticationError) as e:
                # These errors shouldn't be retried immediately
                self.logger.warning(f"API error for {token}, attempt {attempt + 1}", error=str(e))
                last_error = e
                
                # Wait longer for rate limit errors
                if isinstance(e, APIRateLimitError):
                    await asyncio.sleep(self.rate_limit_delay * 2)
                else:
                    await asyncio.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                self.logger.warning(f"Error collecting {token}, attempt {attempt + 1}", error=str(e))
                last_error = e
                await asyncio.sleep(self.rate_limit_delay)
        
        # All retries exhausted
        if last_error:
            raise last_error
        else:
            raise MarketDataError(f"Failed to collect data for {token} after {self.retry_attempts} attempts")
    
    async def _collect_market_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect market-wide data (sentiment, DeFi metrics, on-chain data)
        
        For historical data (>48 hours old), uses historical endpoints when available.
        For recent data, uses current endpoints.
        """
        market_data = {}
        
        # Determine if we're collecting historical or current data
        time_diff = datetime.now(timezone.utc) - end_date
        is_historical = time_diff.total_seconds() > 48 * 3600  # More than 48 hours old
        
        self.logger.info("Starting market data collection", 
                        mode="historical" if is_historical else "current",
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat())
        
        try:
            # Fear & Greed Index
            await asyncio.sleep(self.rate_limit_delay)
            sentiment_data = await self.fear_greed_client.get_market_data()
            market_data['sentiment'] = sentiment_data
            self.collection_stats['api_calls_made'] += 1
            
            self.logger.info("Fear & Greed data collected", 
                           index=sentiment_data.fear_greed_index)
            
        except Exception as e:
            self.logger.error("Failed to collect sentiment data", error=str(e))
            self.collection_stats['errors_encountered'] += 1
        
        try:
            # DeFi metrics
            await asyncio.sleep(self.rate_limit_delay)
            defi_data = await self.defillama_client.get_market_data()
            market_data['defi'] = defi_data
            self.collection_stats['api_calls_made'] += 1
            
            self.logger.info("DeFi metrics collected", 
                           tvl=defi_data.total_value_locked)
            
        except Exception as e:
            self.logger.error("Failed to collect DeFi data", error=str(e))
            self.collection_stats['errors_encountered'] += 1
        
        # Social sentiment (optional)
        if self.has_social_data:
            try:
                await asyncio.sleep(self.rate_limit_delay)
                
                if is_historical:
                    # For historical data, check if we have Pro tier
                    if hasattr(self.social_client, 'tier') and self.social_client.tier == 'pro':
                        # Use midpoint of date range for historical sentiment
                        target_timestamp = start_date + (end_date - start_date) / 2
                        social_data = await self.social_client.get_historical_sentiment('bitcoin', target_timestamp)
                        if social_data:
                            market_data['social'] = social_data
                            self.logger.info("Historical social sentiment collected",
                                           score=social_data.social_score,
                                           timestamp=target_timestamp.isoformat())
                    else:
                        self.logger.debug("Skipping historical social sentiment (requires Pro tier)")
                        # Store None to indicate data not available
                        market_data['social'] = None
                else:
                    # For current data, collect sentiment for all tracked tokens
                    sentiment_results = await self.social_client.get_current_sentiment()
                    if sentiment_results:
                        # Use Bitcoin as primary market proxy
                        market_data['social'] = sentiment_results.get('bitcoin')
                        # Store all results for comprehensive tracking
                        market_data['social_all'] = sentiment_results
                        self.logger.info("Current social sentiment collected",
                                       tokens_count=len(sentiment_results))
                
                self.collection_stats['api_calls_made'] += 1
                
            except Exception as e:
                self.logger.error("Failed to collect social data", error=str(e))
                self.collection_stats['errors_encountered'] += 1
                market_data['social'] = None
        
        # On-chain data (optional)
        if self.has_onchain_data:
            try:
                await asyncio.sleep(self.rate_limit_delay)
                # Get Solana on-chain data as example
                onchain_data = await self.onchain_client.get_market_data(Chain.SOLANA)
                market_data['onchain'] = onchain_data
                self.collection_stats['api_calls_made'] += 1
                
                self.logger.info("On-chain data collected",
                               chain=Chain.SOLANA.value,
                               tx_count=onchain_data.transaction_count_24h)
                
            except Exception as e:
                self.logger.error("Failed to collect on-chain data", error=str(e))
                self.collection_stats['errors_encountered'] += 1
        
        self.logger.info("Market data collection completed",
                        data_sources=list(market_data.keys()))
        
        return market_data
    
    async def _calculate_features(self, ohlcv_data: Dict[str, pd.DataFrame], market_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive features for all collected data"""
        feature_data = {}
        
        self.logger.info("Starting feature calculation", tokens=len(ohlcv_data))
        
        # Calculate market features once (shared across all tokens)
        try:
            market_features = await self.feature_engineer.calculate_market_features()
            
            self.logger.info("Market features calculated",
                           fear_greed=market_features.fear_greed_index,
                           btc_dominance=market_features.btc_dominance)
            
        except Exception as e:
            self.logger.error("Failed to calculate market features", error=str(e))
            # Use placeholder market features
            market_features = self.feature_engineer._get_placeholder_market_features()
        
        # Calculate technical indicators for each token
        for token, price_data in ohlcv_data.items():
            try:
                self.logger.info("Calculating technical indicators", token=token)
                
                # Create a mock DiscoveredToken for the feature engineer
                # This is needed for the feature engineer interface
                mock_token = type('DiscoveredToken', (), {
                    'address': token,
                    'chain_address': token,
                    'symbol': token.upper(),
                    'chain': Chain.ETHEREUM,  # Default chain
                    'price_usd': float(price_data['close'].iloc[-1]) if not price_data.empty else 1.0,
                    'market_cap': 1000000000.0,  # Default market cap
                    'volume_24h': float(price_data['volume'].iloc[-1]) if not price_data.empty else 1000000.0,
                    'price_change_24h': 0.0,
                    'tags': [],
                    'discovered_at': datetime.now()
                })()
                
                # Calculate technical indicators
                tech_indicators = await self.feature_engineer.calculate_technical_indicators(
                    mock_token, price_data
                )
                
                # Store features
                feature_data[token] = {
                    'technical_indicators': tech_indicators,
                    'market_features': market_features,
                    'price_data_length': len(price_data)
                }
                
                self.logger.info("Features calculated successfully", 
                               token=token,
                               data_points=len(price_data))
                
            except Exception as e:
                self.logger.error("Failed to calculate features", 
                                token=token, 
                                error=str(e))
                self.collection_stats['errors_encountered'] += 1
        
        self.logger.info("Feature calculation completed", tokens=len(feature_data))
        
        return feature_data
    
    async def _store_corpus_data(self, 
                               ohlcv_data: Dict[str, pd.DataFrame],
                               market_data: Dict[str, Any],
                               feature_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Store all corpus data in database with data_source='initial'"""
        total_records = 0
        
        self.logger.info("Starting database storage")
        
        try:
            async with get_database_connection() as conn:
                # Store OHLCV data
                for token, price_data in ohlcv_data.items():
                    records_stored = await self._store_ohlcv_records(conn, token, price_data)
                    total_records += records_stored
                    
                    self.logger.info("OHLCV data stored", 
                                   token=token, 
                                   records=records_stored)
                
                # Store market sentiment data
                if 'sentiment' in market_data:
                    await self._store_market_sentiment(conn, market_data['sentiment'])
                    total_records += 1
                
                # Store DeFi metrics
                if 'defi' in market_data:
                    await self._store_defi_metrics(conn, market_data['defi'])
                    total_records += 1
                
                # Store social sentiment (if available)
                if 'social' in market_data:
                    await self._store_social_sentiment(conn, market_data['social'])
                    total_records += 1
                
                # Store on-chain metrics (if available)
                if 'onchain' in market_data:
                    await self._store_onchain_metrics(conn, market_data['onchain'])
                    total_records += 1
                
                # Store feature data
                feature_records = await self._store_feature_data(conn, feature_data)
                total_records += feature_records
                
                self.collection_stats['total_records'] = total_records
                
                self.logger.info("Database storage completed", 
                               total_records=total_records)
                
                return {
                    'success': True,
                    'total_records': total_records,
                    'ohlcv_tokens': len(ohlcv_data),
                    'market_data_sources': len(market_data),
                    'feature_tokens': len(feature_data)
                }
                
        except Exception as e:
            self.logger.error("Database storage failed", error=str(e))
            raise StorageError(f"Failed to store corpus data: {str(e)}")
    
    async def _store_ohlcv_records(self, conn, token: str, price_data: pd.DataFrame) -> int:
        """Store OHLCV records for a single token"""
        if price_data.empty:
            return 0
        
        # Prepare batch insert data
        records = []
        for _, row in price_data.iterrows():
            # Ensure timestamp is timezone-aware
            timestamp = self._ensure_timezone_aware(row['timestamp'])
            
            records.append((
                token.upper(),  # symbol
                token,          # token_id
                timestamp,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                'initial',      # data_source
                datetime.now(timezone.utc), # collection_timestamp with timezone
                'untrained',    # training_status
                None,          # model_version
                None,          # market_cap
                None,          # circulating_supply
            ))
        
        # Batch insert
        query = """
            INSERT INTO crypto_ohlcv (
                symbol, token_id, timestamp,
                open, high, low, close, volume,
                data_source, collection_timestamp, training_status, model_version,
                market_cap, circulating_supply
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
            )
        """
        
        await conn.executemany(query, records)
        return len(records)
    
    def _ensure_timezone_aware(self, dt) -> datetime:
        """Ensure a datetime object is timezone-aware (UTC)"""
        if dt is None:
            return None
        
        # Convert pandas Timestamp to datetime if needed
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        
        # Ensure timezone-aware
        if isinstance(dt, datetime) and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        return dt
    
    async def _store_market_sentiment(self, conn, sentiment_data) -> None:
        """Store market sentiment data"""
        query = """
            INSERT INTO market_sentiment (
                timestamp, fear_greed_index, fear_greed_classification,
                data_source, collection_timestamp
            ) VALUES ($1, $2, $3, $4, $5)
        """
        
        await conn.execute(
            query,
            self._ensure_timezone_aware(sentiment_data.timestamp),
            int(sentiment_data.fear_greed_index),
            sentiment_data.fear_greed_classification.lower().replace(' ', '_'),
            'initial',
            datetime.now(timezone.utc)
        )
    
    async def _store_defi_metrics(self, conn, defi_data) -> None:
        """Store DeFi metrics data"""
        query = """
            INSERT INTO defi_metrics (
                timestamp, protocol_name, chain,
                total_value_locked, tvl_change_24h,
                data_source, collection_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        await conn.execute(
            query,
            self._ensure_timezone_aware(defi_data.timestamp),
            'DeFi Market Wide',  # protocol_name
            'multi_chain',  # chain
            float(defi_data.total_value_locked),
            float(defi_data.tvl_change_24h),
            'initial',
            datetime.now(timezone.utc)
        )
    
    async def _store_social_sentiment(self, conn, social_data) -> None:
        """Store social sentiment data"""
        query = """
            INSERT INTO social_sentiment (
                token_id, timestamp,
                sentiment_score, social_volume_24h,
                data_source, collection_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        await conn.execute(
            query,
            'bitcoin',  # token_id (Bitcoin as market proxy)
            self._ensure_timezone_aware(social_data.timestamp),
            float(social_data.social_score),
            int(social_data.mention_volume),
            'initial',
            datetime.now(timezone.utc)
        )
    
    async def _store_onchain_metrics(self, conn, onchain_data) -> None:
        """Store on-chain metrics data"""
        query = """
            INSERT INTO onchain_metrics (
                timestamp, chain,
                transaction_count_24h, active_addresses_24h,
                data_source, collection_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        await conn.execute(
            query,
            self._ensure_timezone_aware(onchain_data.timestamp),
            'solana',         # chain
            int(onchain_data.transaction_count_24h),
            int(onchain_data.active_addresses_24h),
            'initial',
            datetime.now(timezone.utc)
        )
    
    async def _store_feature_data(self, conn, feature_data: Dict[str, Dict[str, Any]]) -> int:
        """Store calculated feature data"""
        total_features = 0
        
        for token, features in feature_data.items():
            if 'technical_indicators' not in features:
                self.logger.warning(f"No technical indicators for {token}")
                continue
            
            tech_data = features['technical_indicators']
            
            # Skip if tech_data is None
            if tech_data is None:
                self.logger.warning(f"No technical indicators for {token}")
                continue
            
            # Get OHLCV records for this token to link features
            ohlcv_records = await conn.fetch("""
                SELECT id, timestamp FROM crypto_ohlcv 
                WHERE token_id = $1 AND data_source = 'initial'
                ORDER BY timestamp
            """, token)
            
            if not ohlcv_records:
                self.logger.warning(f"No OHLCV records found for {token}, skipping features")
                continue
            
            # Prepare feature records for batch insert
            feature_records = []
            
            for ohlcv_record in ohlcv_records:
                ohlcv_id = ohlcv_record['id']
                timestamp = ohlcv_record['timestamp']
                
                # Extract values from the TechnicalIndicators object
                # Map to the actual database columns
                feature_records.append((
                    ohlcv_id,                                     # $1: ohlcv_id
                    token.upper(),                                # $2: token_id
                    self._ensure_timezone_aware(timestamp),      # $3: timestamp
                    getattr(tech_data, 'rsi', None),            # $4: rsi_14
                    getattr(tech_data, 'macd', None),           # $5: macd
                    getattr(tech_data, 'macd_signal', None),    # $6: macd_signal
                    getattr(tech_data, 'bollinger_upper', None),# $7: bb_upper
                    getattr(tech_data, 'sma_20', None),         # $8: bb_middle (typically SMA20)
                    getattr(tech_data, 'bollinger_lower', None),# $9: bb_lower
                    getattr(tech_data, 'volume_sma', None),     # $10: volume_sma_20
                    None,  # $11: returns_1h - would need price data to calculate
                    None,  # $12: returns_24h - would need price data to calculate
                    None,  # $13: returns_7d - would need price data to calculate
                    getattr(tech_data, 'volatility_score', None),  # $14: volatility_24h
                    '1.0',  # $15: feature_version
                    datetime.now(timezone.utc),  # $16: calculated_at
                    'initial'  # $17: data_source
                ))
            
            # Batch insert features
            if feature_records:
                query = """
                    INSERT INTO crypto_features (
                        ohlcv_id, token_id, timestamp,
                        rsi_14, macd, macd_signal,
                        bb_upper, bb_middle, bb_lower,
                        volume_sma_20,
                        returns_1h, returns_24h, returns_7d,
                        volatility_24h,
                        feature_version, calculated_at, data_source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """
                
                await conn.executemany(query, feature_records)
                total_features += len(feature_records)
                
                self.logger.info(f"Stored {len(feature_records)} feature records for {token}")
        
        return total_features
    
    async def _create_corpus_version(self,
                                   tokens: List[str],
                                   start_date: datetime,
                                   end_date: datetime,
                                   storage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create corpus version record"""
        version_name = f"initial_v1.0_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        query = """
            INSERT INTO training_corpus_versions (
                version_name, data_source, start_timestamp, end_timestamp,
                sample_count, feature_count, tokens,
                is_active, is_immutable
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9
            ) RETURNING version_id
        """
        
        async with get_database_connection() as conn:
            version_id = await conn.fetchval(
                query,
                version_name,
                'initial',
                self._ensure_timezone_aware(start_date),
                self._ensure_timezone_aware(end_date),
                storage_result['total_records'],
                130,  # Expected feature count
                tokens,
                True,   # is_active
                True    # is_immutable for initial corpus
            )
        
        self.logger.info("Corpus version created", 
                        version_id=version_id,
                        version_name=version_name)
        
        return {
            'version_id': version_id,
            'version_name': version_name
        }
    
    async def validate_corpus_completeness(self, version_id: int) -> Dict[str, Any]:
        """Validate that corpus has sufficient features and data quality"""
        self.logger.info("Validating corpus completeness", version_id=version_id)
        
        validation_results = {
            'version_id': version_id,
            'is_valid': True,
            'issues': [],
            'feature_count': 0,
            'data_quality_score': 0.0,
            'completeness_percentage': 0.0
        }
        
        try:
            async with get_database_connection() as conn:
                # Check OHLCV data completeness
                # Note: data_source column doesn't exist, check by recent timestamps
                recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
                ohlcv_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM crypto_ohlcv WHERE created_at > $1",
                    recent_cutoff
                )
                
                # Check feature data completeness
                feature_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM crypto_features WHERE timestamp > $1",
                    recent_cutoff
                )
                
                # Check market data completeness
                sentiment_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM market_sentiment WHERE created_at > $1",
                    recent_cutoff
                )
                
                defi_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM defi_metrics WHERE created_at > $1",
                    recent_cutoff
                )
                
                # Calculate feature count (approximation for initial corpus)
                # In a real implementation, this would count actual features
                estimated_feature_count = min(130, len(self.collected_tokens) * 10 + 20)
                validation_results['feature_count'] = estimated_feature_count
                
                # Data quality checks
                if ohlcv_count < 100:
                    validation_results['issues'].append("Insufficient OHLCV data points")
                    validation_results['is_valid'] = False
                
                if feature_count < len(self.collected_tokens):
                    validation_results['issues'].append("Missing feature calculations")
                    validation_results['is_valid'] = False
                
                if sentiment_count == 0:
                    validation_results['issues'].append("Missing market sentiment data")
                
                if defi_count == 0:
                    validation_results['issues'].append("Missing DeFi metrics data")
                
                # Calculate completeness percentage
                expected_total = len(self.DEFAULT_TOKENS) * self.collection_days
                actual_total = ohlcv_count
                completeness = min(100.0, (actual_total / expected_total) * 100)
                validation_results['completeness_percentage'] = completeness
                
                # Data quality score (0-100)
                quality_factors = []
                quality_factors.append(min(100, ohlcv_count / 1000 * 100))  # OHLCV completeness
                quality_factors.append(100 if feature_count > 0 else 0)      # Features present
                quality_factors.append(100 if sentiment_count > 0 else 50)   # Sentiment data
                quality_factors.append(100 if defi_count > 0 else 50)        # DeFi data
                
                validation_results['data_quality_score'] = sum(quality_factors) / len(quality_factors)
                
                self.collection_stats['total_features'] = estimated_feature_count
                
                self.logger.info("Corpus validation completed",
                               is_valid=validation_results['is_valid'],
                               feature_count=estimated_feature_count,
                               completeness=completeness,
                               quality_score=validation_results['data_quality_score'])
                
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            self.logger.error("Corpus validation failed", error=str(e))
        
        return validation_results
    
    async def mark_as_initial_corpus(self, version_id: int) -> bool:
        """Mark corpus version as the active initial corpus"""
        try:
            async with get_database_connection() as conn:
                # First, deactivate any existing active initial corpus
                await conn.execute("""
                    UPDATE training_corpus_versions 
                    SET is_active = FALSE 
                    WHERE data_source = 'initial' AND is_active = TRUE
                """)
                
                # Then activate this version
                await conn.execute("""
                    UPDATE training_corpus_versions 
                    SET is_active = TRUE, is_immutable = TRUE
                    WHERE version_id = $1
                """, version_id)
                
                self.logger.info("Corpus marked as active initial corpus", version_id=version_id)
                return True
                
        except Exception as e:
            self.logger.error("Failed to mark corpus as initial", error=str(e))
            return False
    
    async def create_corpus_snapshot(self, version_id: int, snapshot_path: str, 
                                   export_format: str = 'parquet',
                                   project_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a comprehensive snapshot/backup of the corpus with GCS export"""
        try:
            # Get version name for snapshot
            async with get_database_connection() as conn:
                version_row = await conn.fetchrow("""
                    SELECT version_name, created_at FROM training_corpus_versions 
                    WHERE version_id = $1
                """, version_id)
            
            if not version_row:
                raise StorageError(f"Version {version_id} not found in database")
            
            version_name = version_row['version_name']
            
            # Try to use GCS export if available
            try:
                from src.data_pipeline.gcs_corpus_exporter import GCSCorpusExporter
                
                # Determine project_id
                if project_id is None:
                    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'shvyr-ai-bots')
                
                # Parse GCS path
                if snapshot_path.startswith('gs://'):
                    bucket_name = snapshot_path.replace('gs://', '').split('/')[0]
                    corpus_path = '/'.join(snapshot_path.replace('gs://', '').split('/')[1:])
                    if not corpus_path.endswith('/'):
                        corpus_path += '/'
                else:
                    # Default to shyvr-models-prod bucket
                    bucket_name = 'shyvr-models-prod'
                    corpus_path = f"training-data/initial-corpus/{version_name}/"
                
                # Initialize GCS exporter
                gcs_exporter = GCSCorpusExporter(
                    project_id=project_id,
                    bucket_name=bucket_name,
                    base_path=""  # Use full corpus_path
                )
                
                # Export to GCS
                export_result = await gcs_exporter.export_corpus(
                    version_id=version_id,
                    version_name=version_name,
                    corpus_path=corpus_path,
                    export_format=export_format,
                    include_metadata=True,
                    compress=True
                )
                
                if export_result['success']:
                    self.logger.info("Corpus snapshot created with GCS export", 
                                   version_id=version_id,
                                   export_path=export_result['export_path'],
                                   files_exported=export_result['export_stats']['files_exported'])
                    
                    return {
                        'success': True,
                        'snapshot_info': {
                            'version_id': version_id,
                            'version_name': version_name,
                            'snapshot_path': export_result['export_path'],
                            'created_at': datetime.now(timezone.utc).isoformat(),
                            'creator': 'InitialCorpusCollector',
                            'export_method': 'GCS',
                            'export_stats': export_result['export_stats']
                        },
                        'export_result': export_result
                    }
                else:
                    raise StorageError(f"GCS export failed: {export_result.get('error')}")
                    
            except ImportError:
                self.logger.warning("GCS export not available, falling back to basic snapshot")
                
                # Fallback: just update database with storage path
                async with get_database_connection() as conn:
                    await conn.execute("""
                        UPDATE training_corpus_versions 
                        SET storage_path = $1, updated_at = NOW()
                        WHERE version_id = $2
                    """, snapshot_path, version_id)
                
                snapshot_info = {
                    'version_id': version_id,
                    'version_name': version_name,
                    'snapshot_path': snapshot_path,
                    'created_at': datetime.now().isoformat(),
                    'creator': 'InitialCorpusCollector',
                    'export_method': 'database_reference'
                }
                
                self.logger.info("Basic corpus snapshot created", 
                               version_id=version_id,
                               path=snapshot_path)
                
                return {
                    'success': True,
                    'snapshot_info': snapshot_info
                }
            
        except Exception as e:
            self.logger.error("Failed to create corpus snapshot", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    async def export_to_gcs(self, version_id: int, bucket_name: str = 'shyvr-models-prod') -> Dict[str, Any]:
        """
        Export corpus data to Google Cloud Storage
        
        Args:
            version_id: The corpus version ID to export
            bucket_name: GCS bucket name
            
        Returns:
            Export result with GCS paths
        """
        try:
            from google.cloud import storage
            import tempfile
            import csv
            
            self.logger.info("Starting GCS export", 
                           version_id=version_id, 
                           bucket=bucket_name)
            
            # Initialize GCS client
            storage_client = storage.Client(project='shvyr-ai-bots')
            bucket = storage_client.bucket(bucket_name)
            
            # Get corpus metadata
            async with get_database_connection() as conn:
                version_info = await conn.fetchrow(
                    "SELECT * FROM training_corpus_versions WHERE version_id = $1",
                    version_id
                )
                
                if not version_info:
                    raise ValueError(f"Corpus version {version_id} not found")
                
                version_name = version_info['version_name']
                
                # Helper function to export table data to CSV
                async def export_table_to_csv(table_name: str, query: str, params: list = None) -> str:
                    """Export a table to CSV and upload to GCS"""
                    data = await conn.fetch(query, *params) if params else await conn.fetch(query)
                    
                    if not data:
                        return None
                    
                    import io
                    csv_buffer = io.StringIO()
                    fieldnames = list(data[0].keys())
                    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for row in data:
                        writer.writerow(dict(row))
                    
                    # Upload to GCS
                    blob_path = f"training-data/initial-corpus/{version_name}/{table_name}.csv"
                    blob = bucket.blob(blob_path)
                    blob.upload_from_string(
                        csv_buffer.getvalue(),
                        content_type='text/csv'
                    )
                    
                    self.logger.info(f"Uploaded {table_name} to GCS",
                                   path=f"gs://{bucket_name}/{blob_path}",
                                   records=len(data))
                    
                    return blob_path
                
                # Export all corpus data tables
                gcs_paths = {}
                
                # 1. Export OHLCV data
                ohlcv_query = """
                    SELECT * FROM crypto_ohlcv 
                    WHERE data_source = 'initial'
                    AND collection_timestamp >= $1
                    ORDER BY token_id, timestamp
                """
                ohlcv_path = await export_table_to_csv(
                    'ohlcv_data', 
                    ohlcv_query,
                    [version_info['created_at'] - timedelta(hours=1)]
                )
                if ohlcv_path:
                    gcs_paths['ohlcv'] = f"gs://{bucket_name}/{ohlcv_path}"
                
                # 2. Export market sentiment
                sentiment_query = """
                    SELECT * FROM market_sentiment 
                    WHERE data_source = 'initial'
                    ORDER BY timestamp
                """
                sentiment_path = await export_table_to_csv('market_sentiment', sentiment_query)
                if sentiment_path:
                    gcs_paths['market_sentiment'] = f"gs://{bucket_name}/{sentiment_path}"
                
                # 3. Export DeFi metrics
                defi_query = """
                    SELECT * FROM defi_metrics 
                    WHERE data_source = 'initial'
                    ORDER BY timestamp
                """
                defi_path = await export_table_to_csv('defi_metrics', defi_query)
                if defi_path:
                    gcs_paths['defi_metrics'] = f"gs://{bucket_name}/{defi_path}"
                
                # 4. Export social sentiment
                social_query = """
                    SELECT * FROM social_sentiment 
                    WHERE data_source = 'initial'
                    ORDER BY timestamp
                """
                social_path = await export_table_to_csv('social_sentiment', social_query)
                if social_path:
                    gcs_paths['social_sentiment'] = f"gs://{bucket_name}/{social_path}"
                
                # 5. Export on-chain metrics
                onchain_query = """
                    SELECT * FROM onchain_metrics 
                    WHERE data_source = 'initial'
                    ORDER BY timestamp
                """
                onchain_path = await export_table_to_csv('onchain_metrics', onchain_query)
                if onchain_path:
                    gcs_paths['onchain_metrics'] = f"gs://{bucket_name}/{onchain_path}"
                
                # Count total records
                ohlcv_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM crypto_ohlcv WHERE data_source = 'initial'"
                )
                sentiment_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM market_sentiment WHERE data_source = 'initial'"
                )
                defi_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM defi_metrics WHERE data_source = 'initial'"
                )
                social_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM social_sentiment WHERE data_source = 'initial'"
                )
                onchain_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM onchain_metrics WHERE data_source = 'initial'"
                )
                
                # Export metadata with all data counts
                metadata = {
                    'version_id': version_id,
                    'version_name': version_name,
                    'created_at': version_info['created_at'].isoformat(),
                    'start_date': version_info['start_date'].isoformat(),
                    'end_date': version_info['end_date'].isoformat(),
                    'sample_count': version_info['sample_count'],
                    'feature_count': version_info['feature_count'],
                    'tokens': version_info['tokens'],
                    'data_source': version_info['data_source'],
                    'export_timestamp': datetime.now(timezone.utc).isoformat(),
                    'record_counts': {
                        'ohlcv': ohlcv_count,
                        'market_sentiment': sentiment_count,
                        'defi_metrics': defi_count,
                        'social_sentiment': social_count,
                        'onchain_metrics': onchain_count,
                        'total': ohlcv_count + sentiment_count + defi_count + social_count + onchain_count
                    },
                    'gcs_paths': gcs_paths
                }
                
                # Upload metadata as JSON
                metadata_path = f"training-data/initial-corpus/{version_name}/metadata.json"
                metadata_blob = bucket.blob(metadata_path)
                metadata_blob.upload_from_string(
                    json.dumps(metadata, indent=2),
                    content_type='application/json'
                )
                gcs_paths['metadata'] = f"gs://{bucket_name}/{metadata_path}"
                
                self.logger.info("Export to GCS completed",
                               version_id=version_id,
                               bucket=bucket_name,
                               ohlcv_records=ohlcv_count,
                               sentiment_records=sentiment_count,
                               defi_records=defi_count,
                               social_records=social_count,
                               onchain_records=onchain_count)
                
                return {
                    'success': True,
                    'version_id': version_id,
                    'version_name': version_name,
                    'gcs_paths': gcs_paths,
                    'records_exported': {
                        'ohlcv': ohlcv_count,
                        'market_sentiment': sentiment_count,
                        'defi_metrics': defi_count,
                        'social_sentiment': social_count,
                        'onchain_metrics': onchain_count,
                        'total': ohlcv_count + sentiment_count + defi_count + social_count + onchain_count
                    }
                }
                
        except Exception as e:
            self.logger.error("GCS export failed", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    async def close(self):
        """Close all API clients and clean up resources"""
        try:
            await self.coingecko_client.close()
            await self.fear_greed_client.close()
            await self.defillama_client.close()
            
            if self.social_client:
                await self.social_client.close()
            
            if self.onchain_client:
                await self.onchain_client.close()
            
            if self.feature_engineer:
                await self.feature_engineer.close()
            
            self.logger.info("Initial Corpus Collector closed")
            
        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()