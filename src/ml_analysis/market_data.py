"""
Market Data API Clients for Cryptocurrency Trading
Provides real-time market sentiment, DeFi metrics, and on-chain data
"""

import asyncio
import aiohttp
import ssl
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
import structlog
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
import time
import os

from src.utils.base import Chain


logger = structlog.get_logger()


@dataclass
class MarketSentimentData:
    """Market sentiment data from various sources"""
    fear_greed_index: float                    # 0-100 scale
    fear_greed_classification: str             # "Fear", "Greed", etc.
    market_trend: str                          # "bull", "bear", "sideways"
    volatility_regime: str                     # "low", "medium", "high"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeFiMetrics:
    """DeFi ecosystem metrics"""
    total_value_locked: float                  # Total TVL in USD
    tvl_change_24h: float                      # 24h change in TVL
    tvl_change_7d: float                       # 7d change in TVL
    defi_dominance: float                      # DeFi TVL / Total market cap
    protocols_count: int                       # Number of active protocols
    chains_tvl: Dict[str, float]              # TVL per chain
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CorrelationMetrics:
    """Cross-asset correlation data"""
    btc_correlation: float                     # Correlation with Bitcoin
    eth_correlation: float                     # Correlation with Ethereum
    btc_dominance: float                       # Bitcoin market dominance %
    eth_dominance: float                       # Ethereum market dominance %
    stablecoin_dominance: float               # Stablecoin market dominance %
    market_beta: float                         # Beta relative to total crypto market
    correlation_matrix: Dict[str, float]       # Correlation with major assets
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OnChainMetrics:
    """On-chain network activity metrics"""
    network_activity: Dict[str, Any]           # Network-specific activity data
    transaction_count_24h: int                 # 24h transaction count
    active_addresses_24h: int                  # 24h active addresses
    transaction_volume_24h: float              # 24h transaction volume in USD
    network_fees_24h: float                    # 24h network fees in USD
    hash_rate: Optional[float]                 # Network hash rate (for PoW)
    staking_ratio: Optional[float]             # Staking ratio (for PoS)
    whale_activity: Dict[str, Any]             # Large transaction activity
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SocialSentimentData:
    """Social sentiment and mention data"""
    social_score: float                        # Aggregated sentiment score 0-1
    mention_volume: int                        # Total mentions across platforms
    sentiment_trend: float                     # Sentiment change trend
    platform_mentions: Dict[str, int]         # Mentions per platform
    sentiment_breakdown: Dict[str, float]      # Sentiment by platform
    trending_keywords: List[str]               # Trending keywords/hashtags
    influencer_sentiment: Optional[float]      # Weighted influencer sentiment
    timestamp: datetime = field(default_factory=datetime.now)


class MarketDataError(Exception):
    """Base exception for market data errors"""
    pass


class APIRateLimitError(MarketDataError):
    """Raised when API rate limit is exceeded"""
    pass


class APIAuthenticationError(MarketDataError):
    """Raised when API authentication fails"""
    pass


class DataNotAvailableError(MarketDataError):
    """Raised when requested data is not available"""
    pass


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int, time_window: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self._requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit permission"""
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside time window
            self._requests = [req_time for req_time in self._requests 
                           if now - req_time < self.time_window]
            
            # Check if we can make a request
            if len(self._requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self._requests)
                wait_time = self.time_window - (now - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()  # Recursive call after waiting
            
            # Record this request
            self._requests.append(now)


class MarketDataClientBase(ABC):
    """Base class for market data API clients"""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 60, cache_ttl: int = 300):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.cache_ttl = cache_ttl
        self.logger = structlog.get_logger().bind(client=self.__class__.__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._rate_limiter = RateLimiter(max_requests=rate_limit, time_window=60.0)
    
    def _create_unverified_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with verification disabled for development/testing"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with SSL verification disabled"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            # Create SSL-disabled connector for development/testing
            ssl_context = self._create_unverified_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        age = datetime.now() - self._cache_timestamps[cache_key]
        return age.total_seconds() < self.cache_ttl
    
    def _cache_data(self, cache_key: str, data: Any) -> None:
        """Cache data with timestamp"""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached data if valid"""
        if self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        return None
    
    async def _make_request(self, url: str, params: Optional[Dict] = None, 
                           headers: Optional[Dict] = None) -> Dict:
        """Make HTTP request with error handling and rate limiting"""
        # Apply rate limiting
        await self._rate_limiter.acquire()
        
        session = await self._get_session()
        
        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise APIRateLimitError(f"Rate limit exceeded for {self.__class__.__name__}")
                elif response.status == 401:
                    raise APIAuthenticationError(f"Authentication failed for {self.__class__.__name__}")
                elif response.status == 404:
                    raise DataNotAvailableError(f"Data not found: {url}")
                elif response.status != 200:
                    text = await response.text()
                    raise MarketDataError(f"API error {response.status}: {text}")
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise MarketDataError(f"Request failed: {str(e)}")
    
    @abstractmethod
    async def get_market_data(self) -> Any:
        """Get market data from the API"""
        pass
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class FearGreedIndexClient(MarketDataClientBase):
    """Client for Crypto Fear & Greed Index API"""
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    async def get_market_data(self, days: int = 1) -> MarketSentimentData:
        """Get Fear & Greed Index data"""
        cache_key = f"fear_greed_{days}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            params = {"limit": days, "format": "json"}
            data = await self._make_request(self.BASE_URL, params=params)
            
            if not data.get("data"):
                raise DataNotAvailableError("No Fear & Greed data available")
            
            latest = data["data"][0]
            
            # Classify fear/greed level
            value = int(latest["value"])
            if value <= 25:
                classification = "Extreme Fear"
            elif value <= 45:
                classification = "Fear"
            elif value <= 55:
                classification = "Neutral"
            elif value <= 75:
                classification = "Greed"
            else:
                classification = "Extreme Greed"
            
            # Determine market trend based on fear/greed
            if value <= 30:
                trend = "bear"
            elif value >= 70:
                trend = "bull"
            else:
                trend = "sideways"
            
            # Determine volatility regime
            if value <= 20 or value >= 80:
                volatility = "high"
            elif value <= 40 or value >= 60:
                volatility = "medium"
            else:
                volatility = "low"
            
            result = MarketSentimentData(
                fear_greed_index=float(value),
                fear_greed_classification=classification,
                market_trend=trend,
                volatility_regime=volatility,
                timestamp=datetime.fromtimestamp(int(latest["timestamp"]))
            )
            
            self._cache_data(cache_key, result)
            self.logger.info("Retrieved Fear & Greed Index", value=value, classification=classification)
            return result
            
        except Exception as e:
            self.logger.error("Failed to get Fear & Greed Index", error=str(e))
            raise MarketDataError(f"Failed to get Fear & Greed Index: {str(e)}")


class DeFiLlamaClient(MarketDataClientBase):
    """Client for DeFiLlama API for DeFi metrics"""
    
    BASE_URL = "https://api.llama.fi"
    
    async def get_market_data(self) -> DeFiMetrics:
        """Get DeFi TVL and protocol data"""
        cache_key = "defi_metrics"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            # Get historical TVL data (v2 endpoint)
            tvl_history = await self._make_request(f"{self.BASE_URL}/v2/historicalChainTvl")
            
            # Get chain TVL
            chains_data = await self._make_request(f"{self.BASE_URL}/v2/chains")
            
            # Get protocols count
            protocols_data = await self._make_request(f"{self.BASE_URL}/protocols")
            
            # Extract current TVL from latest data point
            if tvl_history and len(tvl_history) > 0:
                current_tvl = float(tvl_history[-1].get("tvl", 0))
                
                # Find TVL 24h ago (assuming daily data points)
                tvl_24h_ago = float(tvl_history[-2].get("tvl", current_tvl)) if len(tvl_history) > 1 else current_tvl
                tvl_7d_ago = float(tvl_history[-8].get("tvl", current_tvl)) if len(tvl_history) > 7 else current_tvl
            else:
                # Fallback values if API fails
                current_tvl = 100_000_000_000  # $100B default
                tvl_24h_ago = current_tvl
                tvl_7d_ago = current_tvl
            
            tvl_change_24h = ((current_tvl - tvl_24h_ago) / tvl_24h_ago * 100) if tvl_24h_ago > 0 else 0
            tvl_change_7d = ((current_tvl - tvl_7d_ago) / tvl_7d_ago * 100) if tvl_7d_ago > 0 else 0
            
            # Extract chain TVL data
            chains_tvl = {}
            for chain in chains_data:
                if chain.get("name") and chain.get("tvl"):
                    chains_tvl[chain["name"]] = float(chain["tvl"])
            
            # Count active protocols
            protocols_count = len([p for p in protocols_data if (p.get("tvl") or 0) > 1000000])  # > $1M TVL
            
            # Estimate DeFi dominance (rough calculation)
            # This would need total crypto market cap data for accuracy
            defi_dominance = min(current_tvl / 1e12, 1.0)  # Cap at 100%
            
            result = DeFiMetrics(
                total_value_locked=current_tvl,
                tvl_change_24h=tvl_change_24h,
                tvl_change_7d=tvl_change_7d,
                defi_dominance=defi_dominance,
                protocols_count=protocols_count,
                chains_tvl=chains_tvl
            )
            
            self._cache_data(cache_key, result)
            self.logger.info("Retrieved DeFi metrics", tvl=current_tvl, protocols=protocols_count)
            return result
            
        except Exception as e:
            self.logger.error("Failed to get DeFi metrics", error=str(e))
            raise MarketDataError(f"Failed to get DeFi metrics: {str(e)}")


class CoinGeckoClient(MarketDataClientBase):
    """Client for CoinGecko API for market data and correlations
    
    Supports both free and Pro API endpoints:
    - Free API: Basic OHLC (no volume), requires separate volume fetch
    - Pro API: Full OHLCV data, direct contract address support
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"  # Default to free API
    PRO_BASE_URL = "https://pro-api.coingecko.com/api/v3"  # Pro API URL
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.headers = {}
        if api_key:
            self.headers["X-CG-Pro-API-Key"] = api_key
            # Use Pro API URL when API key is provided
            self.base_url = self.PRO_BASE_URL
        else:
            self.base_url = self.BASE_URL
    
    async def get_market_data(self) -> CorrelationMetrics:
        """Get Bitcoin dominance and correlation metrics"""
        cache_key = "correlation_metrics"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            # Get global market data
            global_data = await self._make_request(
                f"{self.base_url}/global", 
                headers=self.headers
            )
            
            # Get top cryptocurrencies for correlation analysis
            coins_data = await self._make_request(
                f"{self.base_url}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 10,
                    "page": 1,
                    "sparkline": "false",
                    "price_change_percentage": "24h,7d"
                },
                headers=self.headers
            )
            
            global_info = global_data.get("data", {})
            market_cap_percentages = global_info.get("market_cap_percentage", {})
            
            # Extract dominance data
            btc_dominance = market_cap_percentages.get("btc", 0.0)
            eth_dominance = market_cap_percentages.get("eth", 0.0)
            
            # Calculate stablecoin dominance (approximate)
            stablecoins = ["usdt", "usdc", "busd", "dai"]
            stablecoin_dominance = sum(market_cap_percentages.get(coin, 0.0) for coin in stablecoins)
            
            # Calculate correlations (simplified - would need price history for real correlation)
            correlation_matrix = {}
            btc_change = 0.0
            eth_change = 0.0
            
            for coin in coins_data:
                symbol = coin.get("symbol", "").upper()
                change_24h = coin.get("price_change_percentage_24h", 0.0)
                
                if symbol == "BTC":
                    btc_change = change_24h
                elif symbol == "ETH":
                    eth_change = change_24h
                
                correlation_matrix[symbol] = change_24h
            
            # Simplified correlation calculation (in real implementation, use price history)
            btc_correlation = 1.0  # Self-correlation
            eth_correlation = 0.8 if abs(btc_change - eth_change) < 5 else 0.6
            market_beta = 1.0  # Would calculate vs market index
            
            result = CorrelationMetrics(
                btc_correlation=btc_correlation,
                eth_correlation=eth_correlation,
                btc_dominance=btc_dominance,
                eth_dominance=eth_dominance,
                stablecoin_dominance=stablecoin_dominance,
                market_beta=market_beta,
                correlation_matrix=correlation_matrix
            )
            
            self._cache_data(cache_key, result)
            self.logger.info("Retrieved correlation metrics", 
                           btc_dominance=btc_dominance, 
                           eth_dominance=eth_dominance)
            return result
            
        except Exception as e:
            self.logger.error("Failed to get correlation metrics", error=str(e))
            raise MarketDataError(f"Failed to get correlation metrics: {str(e)}")
    
    async def get_ohlcv_data(self, coin_id: str, days: int = 7, 
                            from_date: Optional[datetime] = None, 
                            to_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get OHLCV (Open, High, Low, Close, Volume) data for a coin
        
        Note: CoinGecko's /ohlc endpoint returns only OHLC data (5 values),
        so we fetch volume separately from /market_chart and merge the data.
        """
        cache_key = f"ohlcv_{coin_id}_{days}_{from_date}_{to_date}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Fetch OHLC and volume data in parallel
            ohlc_task = self._get_ohlc_data(coin_id, days, from_date, to_date)
            volume_task = self._get_volume_data(coin_id, days, from_date, to_date)
            
            ohlc_df, volume_df = await asyncio.gather(ohlc_task, volume_task)
            
            if ohlc_df.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Merge OHLC with volume data
            df = self._merge_ohlc_with_volume(ohlc_df, volume_df)
            
            # Cache the result
            self._cache_data(cache_key, df)
            
            self.logger.info("Retrieved OHLCV data", 
                           coin_id=coin_id, 
                           days=days, 
                           records=len(df))
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to get OHLCV data", 
                            coin_id=coin_id, 
                            error=str(e))
            raise MarketDataError(f"Failed to get OHLCV data for {coin_id}: {str(e)}")
    
    async def _get_ohlc_data(self, coin_id: str, days: int = 7,
                            from_date: Optional[datetime] = None,
                            to_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get OHLC data from CoinGecko /ohlc endpoint"""
        try:
            # Prepare API parameters
            params = {"vs_currency": "usd", "days": days}
            
            # Add date range if specified
            if from_date and to_date:
                params["from"] = int(from_date.timestamp())
                params["to"] = int(to_date.timestamp())
            
            # Make API request
            url = f"{self.base_url}/coins/{coin_id}/ohlc"
            data = await self._make_request(url, params=params, headers=self.headers)
            
            if not data:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
            
            # Convert to DataFrame
            ohlc_data = []
            for entry in data:
                if len(entry) < 5:  # timestamp, o, h, l, c (no volume)
                    self.logger.warning(f"Unexpected OHLC data format: got {len(entry)} values")
                    continue
                
                ohlc_data.append({
                    'timestamp': pd.to_datetime(entry[0], unit='ms'),
                    'open': float(entry[1]),
                    'high': float(entry[2]),
                    'low': float(entry[3]),
                    'close': float(entry[4])
                })
            
            df = pd.DataFrame(ohlc_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to get OHLC data", coin_id=coin_id, error=str(e))
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
    
    async def _get_volume_data(self, coin_id: str, days: int = 7,
                              from_date: Optional[datetime] = None,
                              to_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get volume data from CoinGecko /market_chart endpoint"""
        try:
            # Prepare API parameters for market_chart
            params = {"vs_currency": "usd", "days": days}
            
            # Add date range if specified
            if from_date and to_date:
                params["from"] = int(from_date.timestamp())
                params["to"] = int(to_date.timestamp())
            
            # Make API request to market_chart endpoint
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            data = await self._make_request(url, params=params, headers=self.headers)
            
            if not data or 'total_volumes' not in data:
                return pd.DataFrame(columns=['timestamp', 'volume'])
            
            # Convert volume data to DataFrame
            volume_data = []
            for entry in data['total_volumes']:
                if len(entry) >= 2:
                    volume_data.append({
                        'timestamp': pd.to_datetime(entry[0], unit='ms'),
                        'volume': float(entry[1])
                    })
            
            df = pd.DataFrame(volume_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to get volume data", coin_id=coin_id, error=str(e))
            return pd.DataFrame(columns=['timestamp', 'volume'])
    
    def _merge_ohlc_with_volume(self, ohlc_df: pd.DataFrame, volume_df: pd.DataFrame) -> pd.DataFrame:
        """Merge OHLC data with volume data, handling granularity differences"""
        if ohlc_df.empty:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # If no volume data, use 0 as default
        if volume_df.empty:
            ohlc_df['volume'] = 0.0
            return ohlc_df
        
        # Calculate the granularity of OHLC data (time between consecutive entries)
        if len(ohlc_df) > 1:
            ohlc_granularity = (ohlc_df['timestamp'].iloc[1] - ohlc_df['timestamp'].iloc[0]).total_seconds()
        else:
            ohlc_granularity = 3600  # Default to 1 hour
        
        # For each OHLC entry, find the matching volume
        volumes = []
        for idx, row in ohlc_df.iterrows():
            ohlc_time = row['timestamp']
            
            # Find volume entries within the OHLC period
            # Look for volume data between current OHLC timestamp and next one
            if idx < len(ohlc_df) - 1:
                next_time = ohlc_df.iloc[idx + 1]['timestamp']
            else:
                # For the last entry, use the granularity to estimate next period
                next_time = ohlc_time + pd.Timedelta(seconds=ohlc_granularity)
            
            # Get volume entries in this time window
            mask = (volume_df['timestamp'] >= ohlc_time) & (volume_df['timestamp'] < next_time)
            period_volumes = volume_df[mask]['volume']
            
            if not period_volumes.empty:
                # Use the average volume for this period
                avg_volume = period_volumes.mean()
            else:
                # If no exact match, find nearest volume entry
                time_diffs = abs(volume_df['timestamp'] - ohlc_time)
                if not time_diffs.empty:
                    nearest_idx = time_diffs.idxmin()
                    # Only use if within reasonable time range (2x the granularity)
                    if time_diffs[nearest_idx].total_seconds() <= ohlc_granularity * 2:
                        avg_volume = volume_df.loc[nearest_idx, 'volume']
                    else:
                        avg_volume = 0.0
                else:
                    avg_volume = 0.0
            
            volumes.append(avg_volume)
        
        ohlc_df['volume'] = volumes
        
        # Log any significant mismatches
        zero_volume_count = sum(1 for v in volumes if v == 0.0)
        if zero_volume_count > 0:
            self.logger.warning(f"Could not match volume for {zero_volume_count}/{len(ohlc_df)} OHLC entries")
        
        return ohlc_df
    
    async def search_coin_id(self, query: str) -> str:
        """Search for coin ID by name or symbol"""
        cache_key = f"search_{query.lower()}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Get coin list
            url = f"{self.BASE_URL}/coins/list"
            coins_list = await self._make_request(url, headers=self.headers)
            
            query_lower = query.lower()
            
            # First, try exact match by ID
            for coin in coins_list:
                if coin.get("id", "").lower() == query_lower:
                    self._cache_data(cache_key, coin["id"])
                    return coin["id"]
            
            # Then try exact match by symbol
            for coin in coins_list:
                if coin.get("symbol", "").lower() == query_lower:
                    self._cache_data(cache_key, coin["id"])
                    return coin["id"]
            
            # Finally, try partial match by name
            for coin in coins_list:
                if query_lower in coin.get("name", "").lower():
                    self._cache_data(cache_key, coin["id"])
                    return coin["id"]
            
            raise DataNotAvailableError(f"Coin not found: {query}")
            
        except Exception as e:
            self.logger.error("Failed to search coin ID", query=query, error=str(e))
            raise MarketDataError(f"Failed to search coin ID for {query}: {str(e)}")
    
    async def get_historical_data_for_token(self, token_address: str, 
                                          platform_id: str = "ethereum",
                                          days: int = 7,
                                          from_date: Optional[datetime] = None,
                                          to_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical OHLCV data for a token by its contract address
        
        This method uses the CoinGecko Pro API endpoint that provides OHLCV data
        directly by contract address. If Pro API is not available, falls back to
        the free API with contract resolution.
        
        Args:
            token_address: Contract address of the token
            platform_id: Blockchain platform (ethereum, binance-smart-chain, polygon-pos, solana, etc.)
            days: Number of days of historical data
            from_date: Start date for data range
            to_date: End date for data range
            
        Returns:
            DataFrame with OHLCV data
        """
        
        # Try Pro API first if we have an API key
        if self.api_key:
            try:
                # Attempt to use Pro API endpoint for direct contract address lookup
                return await self._get_ohlcv_by_contract_pro(token_address, platform_id, days, from_date, to_date)
            except Exception as e:
                self.logger.info("Pro API not available or failed, using free API fallback", 
                               error=str(e)[:100])
        
        # Fall back to free API with contract resolution
        self.logger.info("Using free API for contract address lookup", 
                       token_address=token_address[:10] + "...")
        
        try:
            # Try to resolve contract to coin ID
            coin_id = await self._resolve_contract_to_coin_id(token_address, platform_id)
            return await self.get_ohlcv_data(coin_id, days, from_date, to_date)
        except DataNotAvailableError:
            # If resolution fails, return empty DataFrame
            self.logger.warning("Could not resolve contract address", 
                              token_address=token_address[:10] + "...", 
                              platform=platform_id)
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    async def _get_ohlcv_by_contract_pro(self, contract_address: str, 
                                        platform_id: str,
                                        days: int = 7,
                                        from_date: Optional[datetime] = None,
                                        to_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get OHLCV data using CoinGecko Pro API contract endpoint
        
        Pro API endpoint: /coins/{platform_id}/contract/{contract_address}/ohlc
        This endpoint returns proper 6-value OHLCV data including volume.
        """
        cache_key = f"ohlcv_contract_{platform_id}_{contract_address[:10]}_{days}_{from_date}_{to_date}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Prepare API parameters
            params = {"vs_currency": "usd", "days": days}
            
            # Add date range if specified
            if from_date and to_date:
                params["from"] = int(from_date.timestamp())
                params["to"] = int(to_date.timestamp())
            
            # Pro API endpoint for contract address OHLC - always use PRO_BASE_URL
            url = f"{self.PRO_BASE_URL}/coins/{platform_id}/contract/{contract_address.lower()}/ohlc"
            
            data = await self._make_request(url, params=params, headers=self.headers)
            
            if not data:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to DataFrame - Pro API returns 6 values including volume
            ohlcv_data = []
            for entry in data:
                if len(entry) == 6:
                    # Pro API returns: [timestamp, open, high, low, close, volume]
                    ohlcv_data.append({
                        'timestamp': pd.to_datetime(entry[0], unit='ms'),
                        'open': float(entry[1]),
                        'high': float(entry[2]),
                        'low': float(entry[3]),
                        'close': float(entry[4]),
                        'volume': float(entry[5])
                    })
                elif len(entry) == 5:
                    # Fallback if Pro API doesn't include volume for some reason
                    self.logger.warning("Pro API returned OHLC without volume", 
                                      contract=contract_address[:10] + "...")
                    ohlcv_data.append({
                        'timestamp': pd.to_datetime(entry[0], unit='ms'),
                        'open': float(entry[1]),
                        'high': float(entry[2]),
                        'low': float(entry[3]),
                        'close': float(entry[4]),
                        'volume': 0.0  # No volume data
                    })
            
            df = pd.DataFrame(ohlcv_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Cache the result
            self._cache_data(cache_key, df)
            
            self.logger.info("Retrieved OHLCV data via Pro API", 
                           platform=platform_id,
                           contract=contract_address[:10] + "...",
                           days=days, 
                           records=len(df))
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to get OHLCV data via Pro API", 
                            platform=platform_id,
                            contract=contract_address[:10] + "...",
                            error=str(e))
            
            # Fall back to free API method
            self.logger.info("Falling back to free API method")
            coin_id = await self._resolve_contract_to_coin_id(contract_address, platform_id)
            return await self.get_ohlcv_data(coin_id, days, from_date, to_date)
    
    async def _resolve_contract_to_coin_id(self, contract_address: str, platform_id: str = "ethereum") -> str:
        """Resolve contract address to CoinGecko coin ID using free API
        
        Uses the /coins/{platform_id}/contract/{contract_address} endpoint
        to get coin information and extract the coin ID.
        """
        cache_key = f"contract_resolve_{platform_id}_{contract_address[:10]}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        try:
            # First check common mappings for efficiency
            contract_mappings = {
                # Ethereum mainnet tokens
                "ethereum": {
                    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "weth",  # WETH
                    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "usd-coin",  # USDC
                    "0xdac17f958d2ee523a2206206994597c13d831ec7": "tether",  # USDT
                    "0x6b175474e89094c44da98b954eedeac495271d0f": "dai",  # DAI
                    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "wrapped-bitcoin",  # WBTC
                    "0x514910771af9ca656af840dff83e8264ecf986ca": "chainlink",  # LINK
                    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": "aave",  # AAVE
                    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "uniswap",  # UNI
                },
                # Binance Smart Chain tokens
                "binance-smart-chain": {
                    "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "wbnb",  # WBNB
                    "0xe9e7cea3dedca5984780bafc599bd69add087d56": "busd",  # BUSD
                    "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d": "usd-coin",  # USDC on BSC
                    "0x55d398326f99059ff775485246999027b3197955": "tether",  # USDT on BSC
                    "0x2170ed0880ac9a755fd29b2688956bd959f933f8": "ethereum",  # ETH on BSC
                },
                # Polygon tokens
                "polygon-pos": {
                    "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": "wmatic",  # WMATIC
                    "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": "usd-coin",  # USDC on Polygon
                    "0xc2132d05d31c914a87c6611c10748aeb04b58e8f": "tether",  # USDT on Polygon
                    "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063": "dai",  # DAI on Polygon
                },
                # Solana tokens (using mint addresses)
                "solana": {
                    "So11111111111111111111111111111111111111112": "wrapped-solana",  # WSOL
                    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "usd-coin",  # USDC on Solana
                    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "tether",  # USDT on Solana
                    "7kbnvuGBxxj8AG9qp8Scn56muWGaRaFqxg1FsRp3PaFT": "ux-protocol-token",  # UXD
                },
            }
            
            address_lower = contract_address.lower()
            platform_mappings = contract_mappings.get(platform_id, {})
            
            if address_lower in platform_mappings:
                coin_id = platform_mappings[address_lower]
                self._cache_data(cache_key, coin_id)
                return coin_id
            
            # If not in mapping, try CoinGecko's contract endpoint
            url = f"{self.BASE_URL}/coins/{platform_id}/contract/{address_lower}"
            
            try:
                coin_data = await self._make_request(url, headers=self.headers)
                
                if coin_data and 'id' in coin_data:
                    coin_id = coin_data['id']
                    self._cache_data(cache_key, coin_id)
                    self.logger.info("Resolved contract to coin ID", 
                                   platform=platform_id,
                                   contract=contract_address[:10] + "...",
                                   coin_id=coin_id)
                    return coin_id
            except Exception as e:
                self.logger.debug("Could not resolve via API", error=str(e))
            
            # If all else fails, raise error
            raise DataNotAvailableError(
                f"Cannot resolve contract {contract_address} on {platform_id} to coin ID. "
                f"Consider using CoinGecko Pro API for direct contract OHLCV data."
            )
            
        except DataNotAvailableError:
            raise
        except Exception as e:
            self.logger.error("Failed to resolve contract address", 
                            platform=platform_id,
                            contract=contract_address[:10] + "...",
                            error=str(e))
            raise MarketDataError(f"Failed to resolve contract address: {str(e)}")


class OnChainAnalyticsClient(MarketDataClientBase):
    """Client for on-chain analytics using Helius API for Solana and fallback APIs for other chains"""
    
    # Helius API endpoints
    HELIUS_RPC_URL = "https://mainnet.helius-rpc.com"
    HELIUS_API_BASE_URL = "https://api.helius.xyz"
    HELIUS_ENHANCED_TRANSACTIONS_ENDPOINT = "/v0/transactions"
    
    # Chain-specific endpoints
    ETHEREUM_RPC_URL = "https://eth-mainnet.g.alchemy.com/v2"
    
    # Whale activity thresholds (in native tokens)
    WHALE_THRESHOLDS = {
        Chain.SOLANA: 10_000,  # 10,000 SOL
        Chain.ETHEREUM: 100,   # 100 ETH  
        Chain.BASE: 100,       # 100 ETH (Base uses ETH)
        Chain.POLYGON: 10_000, # 10,000 MATIC
        Chain.BSC: 1_000,      # 1,000 BNB
        Chain.ARBITRUM: 100,   # 100 ETH (Arbitrum uses ETH)
        Chain.AVALANCHE: 1_000 # 1,000 AVAX
    }
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        # Set up chain-specific configurations
        self.supported_chains = {
            Chain.SOLANA: self._get_solana_data,
            Chain.ETHEREUM: self._get_ethereum_data,
            Chain.BASE: self._get_ethereum_data,  # Base uses similar structure to Ethereum
            Chain.POLYGON: self._get_ethereum_data,  # Polygon is EVM-compatible
            Chain.BSC: self._get_ethereum_data,  # BSC is EVM-compatible
            Chain.ARBITRUM: self._get_ethereum_data,  # Arbitrum is EVM-compatible
            Chain.AVALANCHE: self._get_ethereum_data  # Avalanche C-Chain is EVM-compatible
        }
    
    async def get_market_data(self, chain: Chain = Chain.SOLANA) -> OnChainMetrics:
        """Get on-chain metrics for specified chain"""
        cache_key = f"onchain_{chain.value}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            # Route to appropriate chain handler
            if chain in self.supported_chains:
                result = await self.supported_chains[chain](chain)
            else:
                # Fallback for unsupported chains
                result = await self._get_fallback_data(chain)
            
            self._cache_data(cache_key, result)
            self.logger.info("Retrieved on-chain metrics", 
                           chain=chain.value,
                           tx_count=result.transaction_count_24h,
                           active_addresses=result.active_addresses_24h)
            return result
            
        except Exception as e:
            self.logger.error("Failed to get on-chain metrics", chain=chain.value, error=str(e))
            raise MarketDataError(f"Failed to get on-chain metrics for {chain.value}: {str(e)}")
    
    async def _get_solana_data(self, chain: Chain) -> OnChainMetrics:
        """Get Solana on-chain data using Helius API"""
        try:
            # Get transaction count using RPC method
            tx_count = await self._get_transaction_count_24h(chain)
            
            # Get active addresses count
            active_addresses = await self._get_active_addresses_24h(chain)
            
            # Get transaction volume
            volume = await self._get_transaction_volume_24h(chain)
            
            # Get network fees
            fees = await self._get_network_fees_24h(chain)
            
            # Get whale activity using enhanced transactions
            whale_activity = await self._get_whale_activity(chain)
            
            # Get network-specific metrics
            network_activity = await self._get_network_activity(chain)
            
            return OnChainMetrics(
                network_activity=network_activity,
                transaction_count_24h=tx_count,
                active_addresses_24h=active_addresses,
                transaction_volume_24h=volume,
                network_fees_24h=fees,
                hash_rate=None,  # Solana is PoS
                staking_ratio=network_activity.get("staking_ratio", 0.7),  # Approximate
                whale_activity=whale_activity
            )
            
        except Exception as e:
            self.logger.error("Failed to get Solana data", error=str(e))
            raise MarketDataError(f"Failed to get Solana data: {str(e)}")
    
    async def _get_ethereum_data(self, chain: Chain) -> OnChainMetrics:
        """Get Ethereum on-chain data using fallback APIs"""
        try:
            # Use public Ethereum APIs or RPC endpoints
            # This is a simplified implementation - would use Alchemy/Infura/etc.
            
            # Mock implementation with reasonable fallback data
            tx_count = 1_200_000  # Ethereum processes ~1.2M transactions per day
            active_addresses = 600_000  # Active addresses per day
            volume = 15_000_000_000.0  # ~$15B daily volume
            fees = 25_000_000.0  # ~$25M daily fees
            
            whale_activity = {
                "large_transactions_24h": 150,
                "whale_net_flow": 5000.0,
                "top_addresses_activity": {}
            }
            
            network_activity = {
                "chain": chain.value,
                "gas_price": 25.0,  # Current gas price in gwei
                "block_time": 12.0,
                "staking_ratio": 0.65
            }
            
            return OnChainMetrics(
                network_activity=network_activity,
                transaction_count_24h=tx_count,
                active_addresses_24h=active_addresses,
                transaction_volume_24h=volume,
                network_fees_24h=fees,
                hash_rate=None,  # Ethereum is PoS
                staking_ratio=0.65,
                whale_activity=whale_activity
            )
            
        except Exception as e:
            self.logger.error("Failed to get Ethereum data", error=str(e))
            raise MarketDataError(f"Failed to get Ethereum data: {str(e)}")
    
    
    async def _get_fallback_data(self, chain: Chain) -> OnChainMetrics:
        """Fallback data for unsupported chains"""
        return OnChainMetrics(
            network_activity={"chain": chain.value, "status": "limited_data"},
            transaction_count_24h=0,
            active_addresses_24h=0,
            transaction_volume_24h=0.0,
            network_fees_24h=0.0,
            hash_rate=None,
            staking_ratio=None,
            whale_activity={"large_transactions_24h": 0, "whale_net_flow": 0.0}
        )
    
    async def _get_transaction_count_24h(self, chain: Chain) -> int:
        """Get 24h transaction count using Helius RPC"""
        if not self.api_key:
            raise APIAuthenticationError("Helius API key required")
        
        url = f"{self.HELIUS_RPC_URL}?api-key={self.api_key}"
        
        # Use getTransactionCount method with recent slots
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransactionCount"
        }
        
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise MarketDataError(f"RPC error: {response.status}")
                
                data = await response.json()
                if "error" in data:
                    raise MarketDataError(f"RPC error: {data['error']}")
                
                # For simplicity, returning the total count
                # In production, would calculate 24h difference
                result = data.get("result", 0)
                if isinstance(result, dict):
                    return result.get("value", 150_000)
                elif isinstance(result, (int, float)):
                    return int(result)
                else:
                    return 150_000
                
        except Exception as e:
            self.logger.error("Failed to get transaction count", error=str(e))
            # Return reasonable fallback
            return 150_000  # Solana daily tx count
    
    async def _get_active_addresses_24h(self, chain: Chain) -> int:
        """Get 24h active addresses count"""
        # This would require enhanced analytics - using approximation
        # In production, would use Helius enhanced transactions to count unique addresses
        
        try:
            # Enhanced transactions endpoint to get unique addresses
            # Use sample address for transaction queries (would aggregate in production)
            sample_address = "So11111111111111111111111111111111111111112"  # Wrapped SOL
            url = f"{self.HELIUS_API_BASE_URL}/v0/addresses/{sample_address}/transactions"
            params = {
                "api-key": self.api_key,
                "limit": 100  # Get recent transactions
            }
            
            data = await self._make_request(url, params=params)
            
            # Count unique addresses from recent transactions
            unique_addresses = set()
            for tx in data[:1000]:  # Process up to 1000 transactions
                if "feePayer" in tx:
                    unique_addresses.add(tx["feePayer"])
                
                # Add addresses from transfers
                for transfer in tx.get("nativeTransfers", []):
                    if "fromUserAccount" in transfer:
                        unique_addresses.add(transfer["fromUserAccount"])
                    if "toUserAccount" in transfer:
                        unique_addresses.add(transfer["toUserAccount"])
            
            # Scale up estimate for 24h (rough approximation)
            estimated_24h = len(unique_addresses) * 24  # Scale by hours
            return min(estimated_24h, 500_000)  # Cap at reasonable maximum
            
        except Exception as e:
            self.logger.error("Failed to get active addresses", error=str(e))
            return 75_000  # Reasonable fallback
    
    async def _get_transaction_volume_24h(self, chain: Chain) -> float:
        """Get 24h transaction volume"""
        try:
            # Use enhanced transactions to calculate volume
            # Use sample address for transaction queries (would aggregate in production)
            sample_address = "So11111111111111111111111111111111111111112"  # Wrapped SOL
            url = f"{self.HELIUS_API_BASE_URL}/v0/addresses/{sample_address}/transactions"
            params = {
                "api-key": self.api_key,
                "limit": 100
            }
            
            data = await self._make_request(url, params=params)
            
            total_volume = 0.0
            for tx in data:
                # Sum native transfers (in SOL)
                for transfer in tx.get("nativeTransfers", []):
                    amount = transfer.get("amount", 0)
                    # Convert lamports to SOL (1 SOL = 1e9 lamports)
                    sol_amount = amount / 1e9
                    total_volume += sol_amount
                
                # Could also include token transfers valued in USD
                # This is simplified implementation
            
            # Scale up for 24h estimate
            scaled_volume = total_volume * 144  # Scale by 10-minute periods in a day
            return float(scaled_volume)
            
        except Exception as e:
            self.logger.error("Failed to get transaction volume", error=str(e))
            return 250_000_000.0  # Fallback volume
    
    async def _get_network_fees_24h(self, chain: Chain) -> float:
        """Get 24h network fees"""
        try:
            # Use enhanced transactions to sum fees
            # Use sample address for transaction queries (would aggregate in production)
            sample_address = "So11111111111111111111111111111111111111112"  # Wrapped SOL
            url = f"{self.HELIUS_API_BASE_URL}/v0/addresses/{sample_address}/transactions"
            params = {
                "api-key": self.api_key,
                "limit": 100
            }
            
            data = await self._make_request(url, params=params)
            
            total_fees = 0.0
            for tx in data:
                fee = tx.get("fee", 0)
                # Convert lamports to SOL
                sol_fee = fee / 1e9
                total_fees += sol_fee
            
            # Scale up for 24h estimate
            scaled_fees = total_fees * 144  # Scale by 10-minute periods
            return float(scaled_fees)
            
        except Exception as e:
            self.logger.error("Failed to get network fees", error=str(e))
            return 50_000.0  # Fallback fees
    
    async def _get_whale_activity(self, chain: Chain) -> Dict[str, Any]:
        """Get whale activity data using enhanced transactions"""
        try:
            # Use sample address for transaction queries (would aggregate in production)
            sample_address = "So11111111111111111111111111111111111111112"  # Wrapped SOL
            url = f"{self.HELIUS_API_BASE_URL}/v0/addresses/{sample_address}/transactions"
            params = {
                "api-key": self.api_key,
                "limit": 100
            }
            
            data = await self._make_request(url, params=params)
            
            large_transactions = 0
            total_whale_volume = 0.0
            whale_addresses = set()
            
            threshold = self.WHALE_THRESHOLDS.get(chain, 10_000)
            
            for tx in data:
                tx_volume = 0.0
                
                # Check native transfers
                for transfer in tx.get("nativeTransfers", []):
                    amount = transfer.get("amount", 0)
                    sol_amount = amount / 1e9
                    tx_volume += sol_amount
                    
                    if sol_amount >= threshold:
                        large_transactions += 1
                        total_whale_volume += sol_amount
                        whale_addresses.add(transfer.get("fromUserAccount", ""))
                
                # Check token transfers for large amounts
                for transfer in tx.get("tokenTransfers", []):
                    token_amount = transfer.get("tokenAmount", 0)
                    # Simplified: treat large token transfers as whale activity
                    if token_amount >= 1_000_000:  # 1M tokens threshold
                        large_transactions += 1
                        whale_addresses.add(transfer.get("fromUserAccount", ""))
            
            return {
                "large_transactions_24h": large_transactions * 24,  # Scale to 24h
                "whale_net_flow": total_whale_volume * 24,
                "unique_whale_addresses": len(whale_addresses),
                "whale_threshold": threshold
            }
            
        except Exception as e:
            self.logger.error("Failed to get whale activity", error=str(e))
            return {
                "large_transactions_24h": 100,
                "whale_net_flow": 1000.0,
                "unique_whale_addresses": 25,
                "whale_threshold": self.WHALE_THRESHOLDS.get(chain, 10_000)
            }
    
    async def _get_network_activity(self, chain: Chain) -> Dict[str, Any]:
        """Get general network activity metrics"""
        try:
            if chain == Chain.SOLANA:
                # Get cluster info for Solana
                url = f"{self.HELIUS_RPC_URL}?api-key={self.api_key}"
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getClusterNodes"
                }
                
                session = await self._get_session()
                async with session.post(url, json=payload) as response:
                    data = await response.json()
                    
                    validators_count = len(data.get("result", []))
                    
                    return {
                        "chain": chain.value,
                        "validators": validators_count,
                        "block_time": 0.4,  # Solana block time ~400ms
                        "staking_ratio": 0.7,  # Approximate Solana staking ratio
                        "tps_capacity": 65_000
                    }
            
            return {
                "chain": chain.value,
                "status": "active"
            }
            
        except Exception as e:
            self.logger.error("Failed to get network activity", error=str(e))
            return {
                "chain": chain.value,
                "status": "active"
            }
    
    async def _parse_enhanced_transactions(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Parse enhanced transactions data for metrics"""
        transaction_count = len(transactions)
        total_volume = 0.0
        unique_addresses = set()
        
        for tx in transactions:
            # Count unique addresses
            if "feePayer" in tx:
                unique_addresses.add(tx["feePayer"])
            
            # Sum volume from native transfers
            for transfer in tx.get("nativeTransfers", []):
                amount = transfer.get("amount", 0)
                total_volume += amount / 1e9  # Convert to SOL
                
                if "fromUserAccount" in transfer:
                    unique_addresses.add(transfer["fromUserAccount"])
                if "toUserAccount" in transfer:
                    unique_addresses.add(transfer["toUserAccount"])
        
        return {
            "transaction_count": transaction_count,
            "total_volume": total_volume,
            "unique_addresses": len(unique_addresses)
        }


class SocialSentimentClient(MarketDataClientBase):
    """Client for social sentiment analysis using LunarCrush API
    
    Supports two modes:
    1. Historical mode (requires Pro tier): For initial corpus collection
    2. Current mode (Basic tier): For continuous real-time collection
    """
    
    BASE_URL = "https://lunarcrush.com/api4"
    
    # Default tokens to track continuously in production
    DEFAULT_TRACKING_TOKENS = [
        "bitcoin", "ethereum", "solana", "cardano", "polygon",
        "avalanche", "chainlink", "uniswap", "aave", "curve"
    ]
    
    def __init__(self, api_key: Optional[str] = None, tier: str = "basic", **kwargs):
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("LUNARCRUSH_API_KEY")
        
        super().__init__(api_key=api_key, **kwargs)
        
        if not self.api_key:
            raise APIAuthenticationError("LunarCrush API key is required. Set LUNARCRUSH_API_KEY environment variable or pass api_key parameter.")
        
        self.tier = tier  # 'basic' or 'pro'
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def get_historical_sentiment(self, asset: str, timestamp: datetime) -> Optional[SocialSentimentData]:
        """Get historical sentiment data for a specific timestamp (requires Pro tier)
        
        Args:
            asset: Cryptocurrency to get sentiment for (e.g., 'bitcoin')
            timestamp: Historical timestamp to get data for
            
        Returns:
            SocialSentimentData or None if not available
        """
        if self.tier != "pro":
            self.logger.warning(
                "Historical sentiment requires Pro tier",
                asset=asset,
                timestamp=timestamp.isoformat()
            )
            return None
        
        try:
            # Use time-series endpoint for historical data
            url = f"{self.BASE_URL}/public/topic/{asset}/time-series/v2"
            
            # Calculate the time range around the timestamp
            end_time = timestamp
            start_time = timestamp - timedelta(hours=24)
            
            params = {
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "interval": "1h"
            }
            
            response = await self._make_request(url, params=params, headers=self.headers)
            data_points = response.get("data", [])
            
            if not data_points:
                return None
            
            # Find the data point closest to our timestamp
            closest_point = min(
                data_points,
                key=lambda x: abs(datetime.fromtimestamp(x.get("time", 0)) - timestamp)
            )
            
            # Extract sentiment data from historical point
            return SocialSentimentData(
                social_score=(closest_point.get("sentiment", 3) - 1) / 4,  # Convert 1-5 to 0-1
                mention_volume=closest_point.get("posts", 0),
                sentiment_trend=self._calculate_trend_from_series(data_points),
                platform_mentions=closest_point.get("platforms", {}),
                sentiment_breakdown={},  # Not available in time series
                trending_keywords=[],  # Not available in time series
                influencer_sentiment=None,  # Not available in time series
                timestamp=timestamp
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to get historical sentiment",
                asset=asset,
                timestamp=timestamp.isoformat(),
                error=str(e)
            )
            return None
    
    def _calculate_trend_from_series(self, data_points: List[Dict]) -> float:
        """Calculate trend from time series data points"""
        if len(data_points) < 2:
            return 0.0
        
        # Simple linear trend of sentiment over time
        sentiments = [p.get("sentiment", 3) for p in data_points]
        if len(sentiments) < 2:
            return 0.0
        
        # Calculate change from first half to second half average
        mid = len(sentiments) // 2
        first_half_avg = sum(sentiments[:mid]) / mid if mid > 0 else 3
        second_half_avg = sum(sentiments[mid:]) / len(sentiments[mid:]) if sentiments[mid:] else 3
        
        # Normalize to -1 to 1 range
        trend = (second_half_avg - first_half_avg) / 4
        return max(-1.0, min(1.0, trend))
    
    async def get_current_sentiment(self, assets: Optional[List[str]] = None) -> Dict[str, SocialSentimentData]:
        """Get current sentiment for multiple assets and store in database
        
        Args:
            assets: List of assets to get sentiment for. If None, uses DEFAULT_TRACKING_TOKENS
            
        Returns:
            Dictionary mapping asset to SocialSentimentData
        """
        if assets is None:
            assets = self.DEFAULT_TRACKING_TOKENS
        else:
            # Combine with default tokens to ensure we always track core assets
            assets = list(set(assets + self.DEFAULT_TRACKING_TOKENS))
        
        results = {}
        
        for asset in assets:
            try:
                sentiment_data = await self.get_market_data(asset)
                if sentiment_data:
                    results[asset] = sentiment_data
                    # Store in database for continuous tracking
                    await self._store_sentiment_in_db(asset, sentiment_data)
            except Exception as e:
                self.logger.error(
                    "Failed to get current sentiment",
                    asset=asset,
                    error=str(e)
                )
                continue
        
        self.logger.info(
            "Collected current sentiment",
            assets_count=len(results),
            total_requested=len(assets)
        )
        
        return results
    
    async def _store_sentiment_in_db(self, asset: str, sentiment_data: SocialSentimentData):
        """Store sentiment data in database for continuous tracking"""
        from src.utils.database import get_database_connection
        
        try:
            async with get_database_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO social_sentiment (
                        token_id, timestamp, social_score, mention_volume,
                        sentiment_trend, influencer_sentiment, platform_data,
                        keywords, data_source, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (token_id, timestamp, data_source) DO UPDATE SET
                        social_score = EXCLUDED.social_score,
                        mention_volume = EXCLUDED.mention_volume,
                        sentiment_trend = EXCLUDED.sentiment_trend,
                        influencer_sentiment = EXCLUDED.influencer_sentiment,
                        platform_data = EXCLUDED.platform_data,
                        keywords = EXCLUDED.keywords,
                        updated_at = NOW()
                    """,
                    asset.upper(),
                    sentiment_data.timestamp,
                    sentiment_data.social_score,
                    sentiment_data.mention_volume,
                    sentiment_data.sentiment_trend,
                    sentiment_data.influencer_sentiment,
                    json.dumps({
                        "platforms": sentiment_data.platform_mentions,
                        "sentiment_breakdown": sentiment_data.sentiment_breakdown
                    }),
                    sentiment_data.trending_keywords[:10],  # Store top 10 keywords
                    "lunarcrush_current",  # Mark as current/real-time data
                    datetime.now(timezone.utc)
                )
                
                self.logger.debug(
                    "Stored sentiment in database",
                    asset=asset,
                    score=sentiment_data.social_score
                )
                
        except Exception as e:
            self.logger.error(
                "Failed to store sentiment in database",
                asset=asset,
                error=str(e)
            )
    
    async def get_market_data(self, asset: str = "bitcoin") -> SocialSentimentData:
        """Get social sentiment data for specified asset using LunarCrush API"""
        cache_key = f"social_{asset}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            # Get topic data from LunarCrush
            topic_data = await self._get_topic_data(asset)
            
            # Get sentiment trend from time series
            sentiment_trend = await self._calculate_sentiment_trend(asset)
            
            # Get influencer sentiment from top posts
            influencer_sentiment = await self._calculate_influencer_sentiment(asset)
            
            # Extract data from LunarCrush response
            data = topic_data.get("data", {})
            if not data:
                raise DataNotAvailableError(f"No social sentiment data available for {asset}")
            
            # Extract platform data
            platforms = data.get("platforms", {})
            platform_mentions = {}
            sentiment_breakdown = {}
            
            for platform, platform_data in platforms.items():
                platform_mentions[platform] = platform_data.get("posts", 0)
                platform_sentiment = platform_data.get("sentiment", 3.0)  # 1-5 scale
                # Convert 1-5 scale to 0-1 scale
                sentiment_breakdown[platform] = (platform_sentiment - 1) / 4
            
            # Extract keywords
            trending_keywords = data.get("keywords", [])
            
            # Create result
            result = SocialSentimentData(
                social_score=data.get("sentiment_absolute", 0.5),  # Already 0-1 scale
                mention_volume=data.get("posts_24h", 0),
                sentiment_trend=sentiment_trend,
                platform_mentions=platform_mentions,
                sentiment_breakdown=sentiment_breakdown,
                trending_keywords=trending_keywords,
                influencer_sentiment=influencer_sentiment
            )
            
            self._cache_data(cache_key, result)
            self.logger.info(
                "Retrieved social sentiment from LunarCrush", 
                asset=asset, 
                score=result.social_score,
                mentions=result.mention_volume
            )
            return result
            
        except (APIRateLimitError, APIAuthenticationError, DataNotAvailableError):
            # Re-raise API-specific errors
            raise
        except Exception as e:
            self.logger.error("Failed to get social sentiment", asset=asset, error=str(e))
            raise MarketDataError(f"Failed to get social sentiment for {asset}: {str(e)}")
    
    async def _get_topic_data(self, asset: str) -> Dict[str, Any]:
        """Get topic data from LunarCrush API"""
        try:
            url = f"{self.BASE_URL}/public/topic/{asset}/v1"
            return await self._make_request(url, headers=self.headers)
        except Exception as e:
            self.logger.error("Failed to get topic data", asset=asset, error=str(e))
            raise MarketDataError(f"Failed to get topic data for {asset}: {str(e)}")
    
    async def _calculate_sentiment_trend(self, asset: str) -> float:
        """Calculate sentiment trend from time series data"""
        try:
            # Time series endpoint requires paid subscription
            # Return neutral trend for free tier
            return 0.0  # Neutral trend
            
            # Original code for paid subscription:
            # url = f"{self.BASE_URL}/public/topic/{asset}/time-series/v2"
            # params = {"interval": "1h", "data_points": 24}  # Last 24 hours
            # response = await self._make_request(url, params=params, headers=self.headers)
            # data_points = response.get("data", [])
            # if len(data_points) < 2:
            #     return 0.0  # No trend available
            
        except Exception as e:
            self.logger.warning("Failed to calculate sentiment trend", asset=asset, error=str(e))
            return 0.0  # Return neutral trend on error
    
    async def _calculate_influencer_sentiment(self, asset: str) -> Optional[float]:
        """Calculate weighted influencer sentiment from top posts"""
        try:
            url = f"{self.BASE_URL}/public/topic/{asset}/posts/v1"
            params = {"limit": 50}  # Get top 50 posts
            
            response = await self._make_request(url, params=params, headers=self.headers)
            posts = response.get("data", [])
            
            if not posts:
                return None
            
            # Calculate weighted sentiment based on influence and interactions
            total_weight = 0
            weighted_sentiment = 0
            
            for post in posts:
                creator = post.get("creator", {})
                influence_score = creator.get("influence_score", 0)
                interactions = post.get("interactions", 0)
                sentiment = post.get("sentiment", 3)  # 1-5 scale
                
                # Weight by both influence and interactions
                weight = (influence_score * 0.7) + (min(interactions, 10000) / 100 * 0.3)
                
                if weight > 0:
                    # Convert sentiment from 1-5 to 0-1 scale
                    normalized_sentiment = (sentiment - 1) / 4
                    weighted_sentiment += normalized_sentiment * weight
                    total_weight += weight
            
            if total_weight == 0:
                return None
            
            return weighted_sentiment / total_weight
            
        except Exception as e:
            self.logger.warning("Failed to calculate influencer sentiment", asset=asset, error=str(e))
            return None