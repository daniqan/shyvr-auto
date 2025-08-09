"""
Parallel Corpus Collector with Better Rate Limiting
Optimized for production data collection with parallel API calls
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Any
import pandas as pd
import structlog
from asyncio import Semaphore
import aiohttp

from src.data_pipeline.enhanced_corpus_collector import EnhancedCorpusCollector
from src.utils.config import DatabaseConfig


class ParallelCorpusCollector(EnhancedCorpusCollector):
    """Enhanced corpus collector with parallel processing and rate limiting"""
    
    def __init__(
        self,
        database_config: Optional[DatabaseConfig] = None,
        rate_limit_delay: float = 0.5,
        max_concurrent_tokens: int = 3,
        max_concurrent_api_calls: int = 5,
        chunk_size: int = 30,
        retry_attempts: int = 3,
        timeout_seconds: int = 30
    ):
        """
        Initialize parallel corpus collector
        
        Args:
            database_config: Database configuration
            rate_limit_delay: Delay between API calls in seconds
            max_concurrent_tokens: Max tokens to process in parallel
            max_concurrent_api_calls: Max concurrent API calls per token
            chunk_size: Days per chunk for granular data collection
            retry_attempts: Number of retry attempts for failed requests
            timeout_seconds: Timeout for individual API calls
        """
        super().__init__(
            database_config=database_config,
            rate_limit_delay=rate_limit_delay,
            chunk_size=chunk_size,
            retry_attempts=retry_attempts
        )
        
        self.max_concurrent_tokens = max_concurrent_tokens
        self.max_concurrent_api_calls = max_concurrent_api_calls
        self.timeout_seconds = timeout_seconds
        
        # Semaphores for rate limiting
        self.token_semaphore = Semaphore(max_concurrent_tokens)
        self.api_semaphore = Semaphore(max_concurrent_api_calls)
        
        # Rate limiter for different API endpoints
        self.rate_limiters = {
            'coingecko': RateLimiter(calls_per_minute=50),  # CoinGecko free tier
            'defillama': RateLimiter(calls_per_minute=100),  # DeFiLlama is generous
            'feargreed': RateLimiter(calls_per_minute=30),   # Conservative
            'social': RateLimiter(calls_per_minute=60)       # LunarCrush
        }
        
        self.logger = structlog.get_logger().bind(component="ParallelCorpusCollector")
    
    async def collect_corpus(
        self,
        tokens: List[str],
        start_date: datetime,
        end_date: datetime,
        clean_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Collect corpus data with parallel processing
        
        Args:
            tokens: List of token symbols to collect
            start_date: Start date for historical data
            end_date: End date for historical data
            clean_existing: Whether to clean existing data first
            
        Returns:
            Collection statistics
        """
        self.logger.info(
            "Starting parallel corpus collection",
            tokens_count=len(tokens),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        
        # Start timer
        self.collection_stats['start_time'] = datetime.now(timezone.utc)
        
        # Clean existing data if requested
        if clean_existing:
            await self.clean_existing_data()
        
        # Collect market-wide data first (not token-specific)
        market_data = await self._collect_market_data_parallel(start_date, end_date)
        
        # Process tokens in parallel batches
        token_results = await self._process_tokens_parallel(tokens, start_date, end_date)
        
        # Store all collected data
        await self._store_all_data(token_results, market_data)
        
        # End timer
        self.collection_stats['end_time'] = datetime.now(timezone.utc)
        
        # Calculate final statistics
        duration = (self.collection_stats['end_time'] - self.collection_stats['start_time']).total_seconds()
        self.collection_stats['duration_seconds'] = duration
        self.collection_stats['successful_tokens'] = len(self.collected_tokens)
        self.collection_stats['failed_tokens'] = len(self.failed_tokens)
        
        self.logger.info(
            "Parallel corpus collection completed",
            **self.collection_stats
        )
        
        return self.collection_stats
    
    async def _process_tokens_parallel(
        self,
        tokens: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Process multiple tokens in parallel"""
        results = {}
        
        # Create tasks for each token
        tasks = []
        for token in tokens:
            task = self._process_single_token_with_semaphore(
                token, start_date, end_date
            )
            tasks.append(task)
        
        # Process all tokens in parallel with progress tracking
        completed = 0
        for task in asyncio.as_completed(tasks):
            try:
                token, data = await task
                results[token] = data
                completed += 1
                self.logger.info(
                    f"Token processed ({completed}/{len(tokens)})",
                    token=token,
                    records=len(data.get('ohlcv', pd.DataFrame()))
                )
            except Exception as e:
                self.logger.error("Failed to process token", error=str(e))
                self.collection_stats['errors_encountered'] += 1
        
        return results
    
    async def _process_single_token_with_semaphore(
        self,
        token: str,
        start_date: datetime,
        end_date: datetime
    ) -> tuple[str, Dict[str, Any]]:
        """Process a single token with semaphore rate limiting"""
        async with self.token_semaphore:
            return await self._process_single_token(token, start_date, end_date)
    
    async def _process_single_token(
        self,
        token: str,
        start_date: datetime,
        end_date: datetime
    ) -> tuple[str, Dict[str, Any]]:
        """Process a single token's data collection"""
        token_data = {}
        
        try:
            # Collect OHLCV data with chunking for granularity
            async with self.api_semaphore:
                await self.rate_limiters['coingecko'].acquire()
                ohlcv_data = await self._collect_token_ohlcv_with_retries(
                    token, start_date, end_date
                )
                token_data['ohlcv'] = ohlcv_data
                self.collection_stats['api_calls_made'] += 1
            
            # Calculate features if we have OHLCV data
            if not ohlcv_data.empty:
                features = await self._calculate_features(ohlcv_data)
                token_data['features'] = features
                
                self.collected_tokens.add(token)
                self.collection_stats['total_records'] += len(ohlcv_data)
            else:
                self.failed_tokens[token] = "No OHLCV data available"
            
            return token, token_data
            
        except Exception as e:
            self.logger.error(f"Failed to process token {token}", error=str(e))
            self.failed_tokens[token] = str(e)
            return token, {}
    
    async def _collect_market_data_parallel(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Collect market-wide data in parallel"""
        market_data = {}
        
        # Create parallel tasks for different market data sources
        tasks = {
            'sentiment': self._collect_sentiment_data(),
            'defi': self._collect_defi_data(),
            'social': self._collect_social_data() if self.has_social_data else None
        }
        
        # Remove None tasks
        tasks = {k: v for k, v in tasks.items() if v is not None}
        
        # Execute all tasks in parallel
        results = await asyncio.gather(
            *[self._execute_market_task(name, task) for name, task in tasks.items()],
            return_exceptions=True
        )
        
        # Process results
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to collect {name} data", error=str(result))
                self.collection_stats['errors_encountered'] += 1
            else:
                market_data[name] = result
                self.collection_stats['api_calls_made'] += 1
        
        return market_data
    
    async def _execute_market_task(self, name: str, task):
        """Execute a market data collection task with rate limiting"""
        async with self.api_semaphore:
            await self.rate_limiters.get(name, self.rate_limiters['feargreed']).acquire()
            return await task
    
    async def _collect_sentiment_data(self):
        """Collect Fear & Greed Index data"""
        return await self.fear_greed_client.get_market_data()
    
    async def _collect_defi_data(self):
        """Collect DeFi metrics"""
        return await self.defillama_client.get_market_data()
    
    async def _collect_social_data(self):
        """Collect social sentiment data"""
        if self.social_client:
            return await self.social_client.get_market_data('bitcoin')
        return None
    
    async def _calculate_features(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators and features"""
        try:
            # Use feature engineer to calculate all indicators
            result = await self.feature_engineer.engineer_features(
                ohlcv_data,
                include_technical=True,
                include_market=False  # Market features are collected separately
            )
            
            if result and result.technical_indicators:
                self.collection_stats['total_features'] += 1
                return {
                    'technical_indicators': result.technical_indicators,
                    'calculated_at': datetime.now(timezone.utc)
                }
            
        except Exception as e:
            self.logger.error("Failed to calculate features", error=str(e))
        
        return {}


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        async with self.lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                wait_time = self.min_interval - time_since_last_call
                await asyncio.sleep(wait_time)
            
            self.last_call_time = asyncio.get_event_loop().time()