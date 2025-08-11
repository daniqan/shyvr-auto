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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.utils.base import Chain
from src.utils.system_secrets import get_system_secrets


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
        """Get or create aiohttp session with proper SSL verification"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            # Use certifi for SSL certificates (fixes macOS SSL issues)
            try:
                import certifi
                import ssl
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                # Fallback to unverified SSL if certifi not available
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


class GrokMarketClient:
    """
    Specialized Grok client for extracting historical market data from X/Twitter.
    Uses Grok's x_keyword_search capability to find real historical data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Grok Market Client"""
        if OpenAI is None:
            raise ImportError("openai package required for Grok client. Install with: pip install openai")
        
        # Get API key from SystemSecrets if not provided
        if not api_key:
            try:
                system_secrets = get_system_secrets()
                api_key = system_secrets.get_secret('XAI_API_KEY')
            except Exception as e:
                logger.warning(f"Could not get XAI_API_KEY from SystemSecrets: {e}")
                api_key = None
        
        if not api_key:
            raise APIAuthenticationError("XAI_API_KEY not found. Please set it in Google Secret Manager.")
        
        self.api_key = api_key
        self.base_url = 'https://api.x.ai/v1'
        self.model = "grok-4-0709"  # Model with search capabilities for real-time data
        self.logger = structlog.get_logger().bind(client=self.__class__.__name__)
        
        # Initialize OpenAI client with xAI configuration
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=180
        )
        
        self.logger.info("GrokMarketClient initialized successfully")
    
    async def get_historical_fear_greed(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Extract historical Fear & Greed Index values from @BitcoinFear tweets"""
        self.logger.info(f"Requesting {days_back} days of historical Fear & Greed data")
        
        prompt = f"""Extract the Bitcoin Fear & Greed Index values from @BitcoinFear for the last {days_back} days.
    
Use internal x_keyword_search for "Bitcoin Fear and Greed Index is [int]" to locate values. There is only one unique value per day.

Return ONLY a Python dictionary with the following format, no other text:
{{
    "data": [
        {{"date": "YYYY-MM-DD", "value": XX, "classification": "fear/neutral/greed/extreme_fear/extreme_greed"}},
        ...
    ]
}}

Important:
- Include values for the last {days_back} days
- Use the actual numerical values (0-100) from @BitcoinFear posts
- Use the correct classification based on the value:
  - 0-24: extreme_fear
  - 25-44: fear
  - 45-55: neutral
  - 56-75: greed
  - 76-100: extreme_greed
- Order by date ascending (oldest first)
- Return ONLY the Python dictionary, no explanations or other text"""
        
        try:
            response = await self._call_grok_api(prompt)
            data = self._parse_json_response(response)
            
            if data and 'data' in data:
                historical_data = data['data']
                self.logger.info(f"Successfully retrieved {len(historical_data)} days of Fear & Greed data")
                return historical_data
            else:
                self.logger.warning("No data found in Grok response")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get historical Fear & Greed data: {e}")
            return []  # Return empty list instead of raising to allow fallback
    
    async def _call_grok_api(self, prompt: str) -> str:
        """Make an API call to Grok"""
        try:
            # Use asyncio.to_thread for the synchronous OpenAI client
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Return only the requested data format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Grok API call failed: {e}")
            raise MarketDataError(f"Grok API error: {str(e)}")
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON dictionary from Grok's response"""
        if not response:
            return None
        
        response = response.strip()
        
        # Try to find dictionary in response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            dict_str = response[start_idx:end_idx]
            
            try:
                return json.loads(dict_str)
            except json.JSONDecodeError:
                # Try Python eval as fallback (safely)
                try:
                    if dict_str.strip().startswith('{') and dict_str.strip().endswith('}'):
                        import ast
                        return ast.literal_eval(dict_str)
                except:
                    self.logger.warning("Could not parse response as JSON or Python dict")
                    return None
        
        return None


class FearGreedIndexClient(MarketDataClientBase):
    """Enhanced client for Crypto Fear & Greed Index with historical data support"""
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 60, cache_ttl: int = 300):
        """Initialize Fear & Greed client with historical data support"""
        super().__init__(api_key, rate_limit, cache_ttl)
        
        # Alternative.me API supports extensive historical data (2000+ days / 5+ years)
        self.historical_enabled = True
        self.grok_client = None  # Optional, not required since alternative.me works well
        self.max_historical_days = 2000  # API supports at least 2000 days (5+ years)
        
        self.logger.info("Fear & Greed client initialized with 2000-day (5+ years) historical data support")
    
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
    
    async def get_historical_data(self, 
                                 start_date: datetime, 
                                 end_date: datetime) -> List[MarketSentimentData]:
        """
        Get historical Fear & Greed data.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            List of MarketSentimentData objects for each day
        """
        historical_data = []
        
        # Calculate days needed
        days_back = (end_date - start_date).days + 1
        
        # Use alternative.me API which supports days of historical data
        if self.historical_enabled:
            try:
                self.logger.info(f"Fetching {days_back} days of historical Fear & Greed data from alternative.me")
                # Alternative.me supports extensive historical data
                params = {"limit": days_back, "format": "json"}
                data = await self._make_request(self.BASE_URL, params=params)
                
                if data and data.get("data"):
                    for entry in data["data"]:
                        entry_date = datetime.fromtimestamp(int(entry["timestamp"]))
                        
                        # Skip if outside requested range
                        if entry_date.date() < start_date.date() or entry_date.date() > end_date.date():
                            continue
                        
                        value = int(entry["value"])
                        
                        # Classify
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
                        
                        # Determine market trend
                        if value <= 30:
                            trend = "bear"
                        elif value >= 70:
                            trend = "bull"
                        else:
                            trend = "sideways"
                        
                        # Determine volatility
                        if value <= 20 or value >= 80:
                            volatility = "high"
                        elif value <= 40 or value >= 60:
                            volatility = "medium"
                        else:
                            volatility = "low"
                        
                        sentiment_data = MarketSentimentData(
                            fear_greed_index=float(value),
                            fear_greed_classification=classification,
                            market_trend=trend,
                            volatility_regime=volatility,
                            timestamp=entry_date
                        )
                        
                        historical_data.append(sentiment_data)
                    
                    if historical_data:
                        self.logger.info(f"Retrieved {len(historical_data)} days from alternative.me")
                        return sorted(historical_data, key=lambda x: x.timestamp)
                        
            except Exception as e:
                self.logger.warning(f"Failed to get historical data from alternative.me: {e}")
        
        # If no historical data available, return current snapshot
        if not historical_data:
            self.logger.warning("No historical data available, returning current snapshot")
            try:
                current = await self.get_market_data()
                return [current]
            except:
                return []
        
        return historical_data


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
            # Get real-time network statistics using Helius RPC
            url = f"{self.HELIUS_RPC_URL}?api-key={self.api_key}"
            
            # Get recent performance samples for TPS calculation
            performance_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getRecentPerformanceSamples",
                "params": [1]  # Get 1 sample (most recent)
            }
            
            session = await self._get_session()
            async with session.post(url, json=performance_payload) as response:
                perf_data = await response.json()
                perf_result = perf_data.get("result", [{}])[0]
                
                # Calculate actual TPS from performance samples
                num_transactions = perf_result.get("numTransactions", 0)
                sample_period = perf_result.get("samplePeriodSecs", 60)
                current_tps = num_transactions / sample_period if sample_period > 0 else 0
            
            # Get epoch info for staking statistics
            epoch_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getEpochInfo"
            }
            
            async with session.post(url, json=epoch_payload) as response:
                epoch_data = await response.json()
                epoch_info = epoch_data.get("result", {})
                current_slot = epoch_info.get("absoluteSlot", 0)
            
            # Get supply info for staking ratio calculation
            supply_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSupply"
            }
            
            async with session.post(url, json=supply_payload) as response:
                supply_data = await response.json()
                supply_info = supply_data.get("result", {}).get("value", {})
                total_supply = supply_info.get("total", 0) / 1e9  # Convert lamports to SOL
                circulating_supply = supply_info.get("circulating", 0) / 1e9
                non_circulating = supply_info.get("nonCirculating", 0) / 1e9
                
                # Estimate staking ratio from non-circulating supply
                staking_ratio = non_circulating / total_supply if total_supply > 0 else 0.7
            
            # Get validator count
            validators_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getVoteAccounts"
            }
            
            async with session.post(url, json=validators_payload) as response:
                validators_data = await response.json()
                validators_result = validators_data.get("result", {})
                current_validators = len(validators_result.get("current", []))
                delinquent_validators = len(validators_result.get("delinquent", []))
                total_validators = current_validators + delinquent_validators
            
            # Get transaction count using real metrics
            tx_count = await self._get_transaction_count_24h(chain)
            
            # Get active addresses count with improved estimation
            active_addresses = await self._get_active_addresses_24h(chain)
            
            # Get transaction volume
            volume = await self._get_transaction_volume_24h(chain)
            
            # Get network fees
            fees = await self._get_network_fees_24h(chain)
            
            # Get whale activity using enhanced transactions
            whale_activity = await self._get_whale_activity(chain)
            
            # Build comprehensive network activity metrics
            network_activity = {
                "chain": chain.value,
                "validators": total_validators,
                "active_validators": current_validators,
                "delinquent_validators": delinquent_validators,
                "current_slot": current_slot,
                "epoch": epoch_info.get("epoch", 0),
                "slot_index": epoch_info.get("slotIndex", 0),
                "slots_in_epoch": epoch_info.get("slotsInEpoch", 432000),
                "block_time": 0.4,  # Solana's target block time is 400ms
                "current_tps": current_tps,
                "tps_capacity": 65000,  # Theoretical maximum
                "staking_ratio": staking_ratio,
                "total_supply_sol": total_supply,
                "circulating_supply_sol": circulating_supply
            }
            
            return OnChainMetrics(
                network_activity=network_activity,
                transaction_count_24h=tx_count,
                active_addresses_24h=active_addresses,
                transaction_volume_24h=volume,
                network_fees_24h=fees,
                hash_rate=None,  # Solana is PoS
                staking_ratio=staking_ratio,
                whale_activity=whale_activity
            )
            
        except Exception as e:
            self.logger.error("Failed to get Solana data", error=str(e))
            # Fallback to reasonable estimates
            return OnChainMetrics(
                network_activity={
                    "chain": chain.value,
                    "validators": 2000,
                    "block_time": 0.4,
                    "staking_ratio": 0.7,
                    "tps_capacity": 65000,
                    "current_tps": 3000
                },
                transaction_count_24h=400_000_000,  # ~400M transactions per day
                active_addresses_24h=100_000,
                transaction_volume_24h=1_000_000_000.0,  # $1B daily volume
                network_fees_24h=500_000.0,  # ~$500K in fees
                hash_rate=None,
                staking_ratio=0.7,
                whale_activity={
                    "large_transactions_24h": 1000,
                    "whale_net_flow": 100_000.0,
                    "whale_threshold": 10_000
                }
            )
    
    async def _get_ethereum_data(self, chain: Chain) -> OnChainMetrics:
        """Get Ethereum on-chain data using Alchemy API (preferred) or Etherscan API"""
        try:
            # Get API keys from system secrets
            from src.utils.system_secrets import get_system_secrets
            system_secrets = get_system_secrets()
            alchemy_api_key = system_secrets.alchemy_api_key
            etherscan_api_key = system_secrets.etherscan_api_key
            
            # Prefer Alchemy API if available
            if alchemy_api_key:
                return await self._get_ethereum_data_alchemy(chain, alchemy_api_key)
            elif etherscan_api_key:
                self.logger.info("Alchemy API key not found, falling back to Etherscan")
                return await self._get_ethereum_data_etherscan(chain, etherscan_api_key)
            else:
                self.logger.warning("Neither Alchemy nor Etherscan API keys found, using fallback data")
                return await self._get_ethereum_fallback_data(chain)
            
        except Exception as e:
            self.logger.error(f"Failed to get {chain.value} data", error=str(e))
            return await self._get_ethereum_fallback_data(chain)
    
    async def _get_ethereum_data_alchemy(self, chain: Chain, api_key: str) -> OnChainMetrics:
        """Get Ethereum on-chain data using Alchemy API"""
        try:
            # Map chain to Alchemy network endpoints
            chain_networks = {
                Chain.ETHEREUM: "eth-mainnet",
                Chain.POLYGON: "polygon-mainnet",
                Chain.ARBITRUM: "arb-mainnet",
                Chain.BASE: "base-mainnet",
                Chain.AVALANCHE: "avax-mainnet"
            }
            
            network = chain_networks.get(chain, "eth-mainnet")
            base_url = f"https://{network}.g.alchemy.com/v2/{api_key}"
            
            # Get latest block number
            latest_block_payload = {
                "jsonrpc": "2.0",
                "method": "eth_blockNumber",
                "params": [],
                "id": 1
            }
            
            session = await self._get_session()
            async with session.post(base_url, json=latest_block_payload) as response:
                block_data = await response.json()
                latest_block = int(block_data.get("result", "0x0"), 16)
            
            # Calculate block 24 hours ago
            blocks_per_day = 7200 if chain == Chain.ETHEREUM else 43200  # ETH: 12s, Others: ~2s blocks
            block_24h_ago = latest_block - blocks_per_day
            
            # Get gas price
            gas_price_payload = {
                "jsonrpc": "2.0",
                "method": "eth_gasPrice",
                "params": [],
                "id": 1
            }
            
            async with session.post(base_url, json=gas_price_payload) as response:
                gas_data = await response.json()
                gas_price_wei = int(gas_data.get("result", "0x0"), 16)
                gas_price_gwei = gas_price_wei / 1e9
            
            # Get block details for transaction count estimation
            block_payload = {
                "jsonrpc": "2.0",
                "method": "eth_getBlockByNumber",
                "params": [hex(latest_block), False],  # False = don't include full tx objects
                "id": 1
            }
            
            async with session.post(base_url, json=block_payload) as response:
                block_info = await response.json()
                block_result = block_info.get("result", {})
                tx_per_block = len(block_result.get("transactions", []))
                gas_used = int(block_result.get("gasUsed", "0x0"), 16)
                gas_limit = int(block_result.get("gasLimit", "0x0"), 16)
                base_fee = int(block_result.get("baseFeePerGas", "0x0"), 16) / 1e9 if "baseFeePerGas" in block_result else gas_price_gwei
            
            # Estimate daily transaction count
            estimated_tx_count = tx_per_block * blocks_per_day
            
            # Use Alchemy's enhanced APIs for better metrics
            # Get asset transfers for volume estimation (last 1000 blocks sample)
            transfers_payload = {
                "jsonrpc": "2.0",
                "method": "alchemy_getAssetTransfers",
                "params": [{
                    "fromBlock": hex(latest_block - 100),
                    "toBlock": hex(latest_block),
                    "category": ["external", "internal"],
                    "maxCount": "0x64"  # 100 transfers
                }],
                "id": 1
            }
            
            async with session.post(base_url, json=transfers_payload) as response:
                transfers_data = await response.json()
                transfers = transfers_data.get("result", {}).get("transfers", [])
                
                # Calculate volume and active addresses from transfers
                unique_addresses = set()
                total_value_eth = 0.0
                large_transfers = 0
                whale_threshold = 100  # ETH
                
                for transfer in transfers:
                    if transfer.get("from"):
                        unique_addresses.add(transfer["from"])
                    if transfer.get("to"):
                        unique_addresses.add(transfer["to"])
                    
                    value = float(transfer.get("value", 0))
                    total_value_eth += value
                    
                    if value >= whale_threshold:
                        large_transfers += 1
                
                # Scale up estimates
                scaling_factor = blocks_per_day / 100  # We sampled 100 blocks
                estimated_active_addresses = int(len(unique_addresses) * scaling_factor)
                estimated_volume_eth = total_value_eth * scaling_factor
                estimated_large_tx = int(large_transfers * scaling_factor)
            
            # Get ETH price for USD conversions
            # Using CoinGecko as price oracle (could use Chainlink in production)
            eth_price = 3000.0  # Default fallback
            try:
                coingecko_url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
                async with session.get(coingecko_url) as response:
                    price_data = await response.json()
                    eth_price = price_data.get("ethereum", {}).get("usd", 3000.0)
            except:
                pass
            
            # Calculate network fees
            avg_gas_per_tx = 21000  # Basic transfer
            estimated_fees_eth = (estimated_tx_count * avg_gas_per_tx * base_fee) / 1e9
            estimated_fees_usd = estimated_fees_eth * eth_price
            
            # Calculate staking ratio (post-merge Ethereum)
            staking_ratio = 0.27 if chain == Chain.ETHEREUM else None
            
            whale_activity = {
                "large_transactions_24h": estimated_large_tx,
                "whale_net_flow": estimated_large_tx * whale_threshold,
                "whale_threshold": whale_threshold,
                "unique_whale_addresses": estimated_large_tx // 10  # Rough estimate
            }
            
            network_activity = {
                "chain": chain.value,
                "gas_price_gwei": gas_price_gwei,
                "base_fee_gwei": base_fee,
                "gas_used": gas_used,
                "gas_limit": gas_limit,
                "utilization": (gas_used / gas_limit * 100) if gas_limit > 0 else 0,
                "block_number": latest_block,
                "block_time": 12.0 if chain == Chain.ETHEREUM else 2.0,
                "eth_price": eth_price,
                "staking_ratio": staking_ratio,
                "network": network
            }
            
            self.logger.info(f"Retrieved Alchemy data for {chain.value}",
                           tx_count=estimated_tx_count,
                           active_addresses=estimated_active_addresses)
            
            return OnChainMetrics(
                network_activity=network_activity,
                transaction_count_24h=estimated_tx_count,
                active_addresses_24h=estimated_active_addresses,
                transaction_volume_24h=estimated_volume_eth * eth_price,
                network_fees_24h=estimated_fees_usd,
                hash_rate=None,
                staking_ratio=staking_ratio,
                whale_activity=whale_activity
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get {chain.value} data from Alchemy", error=str(e))
            # Fall back to Etherscan if available
            from src.utils.system_secrets import get_system_secrets
            system_secrets = get_system_secrets()
            if system_secrets.etherscan_api_key:
                return await self._get_ethereum_data_etherscan(chain, system_secrets.etherscan_api_key)
            else:
                return await self._get_ethereum_fallback_data(chain)
    
    async def _get_ethereum_data_etherscan(self, chain: Chain, api_key: str) -> OnChainMetrics:
        """Get Ethereum on-chain data using Etherscan API"""
        try:
            # Map chain to Etherscan endpoints
            chain_endpoints = {
                Chain.ETHEREUM: "https://api.etherscan.io/api",
                Chain.POLYGON: "https://api.polygonscan.com/api",
                Chain.BSC: "https://api.bscscan.com/api",
                Chain.ARBITRUM: "https://api.arbiscan.io/api",
                Chain.BASE: "https://api.basescan.org/api",
                Chain.AVALANCHE: "https://api.snowtrace.io/api"
            }
            
            base_url = chain_endpoints.get(chain, "https://api.etherscan.io/api")
            
            # Get gas oracle for current gas prices
            gas_oracle_url = f"{base_url}?module=gastracker&action=gasoracle&apikey={api_key}"
            gas_data = await self._make_request(gas_oracle_url)
            gas_result = gas_data.get("result", {})
            
            # Get current ETH price
            eth_price_url = f"{base_url}?module=stats&action=ethprice&apikey={api_key}"
            price_data = await self._make_request(eth_price_url)
            eth_price = float(price_data.get("result", {}).get("ethusd", 3000))
            
            # Get latest block number for calculating 24h range
            latest_block_url = f"{base_url}?module=proxy&action=eth_blockNumber&apikey={api_key}"
            block_data = await self._make_request(latest_block_url)
            latest_block = int(block_data.get("result", "0x0"), 16)
            
            # Calculate block 24 hours ago (assuming ~12 second block time)
            blocks_per_day = 7200  # 24 * 60 * 60 / 12
            block_24h_ago = latest_block - blocks_per_day
            
            # Get transaction count (simplified - would need more complex logic for accurate count)
            # Using proxy module to get block transaction counts
            tx_count_url = f"{base_url}?module=proxy&action=eth_getBlockTransactionCountByNumber&tag={hex(latest_block)}&apikey={api_key}"
            tx_count_data = await self._make_request(tx_count_url)
            tx_per_block = int(tx_count_data.get("result", "0x100"), 16)
            estimated_tx_count = tx_per_block * blocks_per_day
            
            # Get network utilization and calculate fees
            gas_price_gwei = float(gas_result.get("ProposeGasPrice", 25))
            gas_used_per_tx = 21000  # Basic transfer gas
            estimated_fees_eth = (estimated_tx_count * gas_used_per_tx * gas_price_gwei) / 1e9
            estimated_fees_usd = estimated_fees_eth * eth_price
            
            # Estimate active addresses (would need transaction iteration for accuracy)
            # Using reasonable estimates based on network activity
            active_addresses = int(estimated_tx_count * 0.5)  # Rough estimate
            
            # Estimate transaction volume
            avg_tx_value_eth = 0.5  # Average transaction value estimate
            volume_eth = estimated_tx_count * avg_tx_value_eth
            volume_usd = volume_eth * eth_price
            
            # Get staking statistics (for Ethereum post-merge)
            # These would come from beacon chain APIs in production
            staking_ratio = 0.27  # ~27% of ETH staked as of 2024
            
            # Estimate whale activity
            whale_threshold_eth = 100
            whale_threshold_usd = whale_threshold_eth * eth_price
            large_tx_estimate = int(estimated_tx_count * 0.001)  # 0.1% of transactions
            
            whale_activity = {
                "large_transactions_24h": large_tx_estimate,
                "whale_net_flow": large_tx_estimate * whale_threshold_eth,
                "whale_threshold": whale_threshold_eth,
                "whale_threshold_usd": whale_threshold_usd
            }
            
            network_activity = {
                "chain": chain.value,
                "gas_price_gwei": gas_price_gwei,
                "safe_gas_price": float(gas_result.get("SafeGasPrice", 20)),
                "fast_gas_price": float(gas_result.get("FastGasPrice", 30)),
                "block_number": latest_block,
                "block_time": 12.0,
                "eth_price": eth_price,
                "staking_ratio": staking_ratio
            }
            
            return OnChainMetrics(
                network_activity=network_activity,
                transaction_count_24h=estimated_tx_count,
                active_addresses_24h=active_addresses,
                transaction_volume_24h=volume_usd,
                network_fees_24h=estimated_fees_usd,
                hash_rate=None,  # Ethereum is PoS
                staking_ratio=staking_ratio,
                whale_activity=whale_activity
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get {chain.value} data from Etherscan", error=str(e))
            # Fall back to reasonable estimates
            return await self._get_ethereum_fallback_data(chain)
    
    async def _get_ethereum_fallback_data(self, chain: Chain) -> OnChainMetrics:
        """Fallback data for Ethereum when API fails"""
        # Reasonable fallback values based on typical network activity
        chain_defaults = {
            Chain.ETHEREUM: {
                "tx_count": 1_200_000,
                "active_addresses": 600_000,
                "volume": 15_000_000_000.0,
                "fees": 25_000_000.0,
                "gas_price": 25.0,
                "staking_ratio": 0.27
            },
            Chain.POLYGON: {
                "tx_count": 3_000_000,
                "active_addresses": 800_000,
                "volume": 1_000_000_000.0,
                "fees": 100_000.0,
                "gas_price": 30.0,
                "staking_ratio": 0.35
            },
            Chain.BSC: {
                "tx_count": 4_000_000,
                "active_addresses": 1_000_000,
                "volume": 2_000_000_000.0,
                "fees": 500_000.0,
                "gas_price": 3.0,
                "staking_ratio": 0.45
            },
            Chain.ARBITRUM: {
                "tx_count": 800_000,
                "active_addresses": 400_000,
                "volume": 500_000_000.0,
                "fees": 1_000_000.0,
                "gas_price": 0.1,
                "staking_ratio": None
            },
            Chain.BASE: {
                "tx_count": 500_000,
                "active_addresses": 250_000,
                "volume": 200_000_000.0,
                "fees": 50_000.0,
                "gas_price": 0.01,
                "staking_ratio": None
            },
            Chain.AVALANCHE: {
                "tx_count": 600_000,
                "active_addresses": 300_000,
                "volume": 300_000_000.0,
                "fees": 150_000.0,
                "gas_price": 25.0,
                "staking_ratio": 0.60
            }
        }
        
        defaults = chain_defaults.get(chain, chain_defaults[Chain.ETHEREUM])
        
        whale_activity = {
            "large_transactions_24h": int(defaults["tx_count"] * 0.001),
            "whale_net_flow": 5000.0,
            "whale_threshold": 100.0
        }
        
        network_activity = {
            "chain": chain.value,
            "gas_price_gwei": defaults["gas_price"],
            "block_time": 12.0 if chain == Chain.ETHEREUM else 2.0,
            "staking_ratio": defaults["staking_ratio"]
        }
        
        return OnChainMetrics(
            network_activity=network_activity,
            transaction_count_24h=defaults["tx_count"],
            active_addresses_24h=defaults["active_addresses"],
            transaction_volume_24h=defaults["volume"],
            network_fees_24h=defaults["fees"],
            hash_rate=None,
            staking_ratio=defaults["staking_ratio"],
            whale_activity=whale_activity
        )
    
    
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
        """Get general network activity metrics - now handled in chain-specific methods"""
        # This method is deprecated as network activity is now gathered
        # in _get_solana_data and _get_ethereum_data methods with real-time data
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


class GraphProtocolClient(MarketDataClientBase):
    """Client for The Graph Protocol - decentralized blockchain data indexing
    
    Provides access to historical and real-time DeFi metrics via GraphQL subgraphs
    """
    
    # Use direct subgraph studio URLs (these work without API keys for limited queries)
    SUBGRAPH_ENDPOINTS = {
        # Uniswap V3 public endpoint on Polygon
        "uniswap_v3": "https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon",
        # DeFiLlama endpoints for aggregated data
        "defi_aggregated": "https://api.llama.fi/protocol/uniswap",
    }
    
    # Subgraph IDs for future use with proper API keys
    SUBGRAPH_IDS = {
        "uniswap_v3": "ELUcwgpm14LKPLrBRuVvPvNKHQ9HvwmtKgKSH6123cr7",
        "uniswap_v2": "EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLRFu",
        "aave_v3": "GQFbb95cE6d8mV989mL5figjaGH5qC3qJUqYrfEPqhXP",
        "aave_v2": "8wR23o3HiXQHqMXpX1oqY4xwgJXsMRTqfJGcsY1V8Dxy",
        "compound_v3": "Ehks5TUAiLwJKQmYNfHQVxYBwb9wBWFcZKCfWUBfP4JE",
        "compound_v2": "6tGbL7RjaT8Wuu9v2vKQ8yfZQ4KUzUQdXnqPb7MiPPNP",
    }
    
    def __init__(self, api_key: Optional[str] = None, api_token: Optional[str] = None, **kwargs):
        """Initialize Graph Protocol client
        
        Args:
            api_key: Graph API key for gateway access
            api_token: Graph API token for authenticated queries
        """
        super().__init__(api_key=api_key, **kwargs)
        
        # Get API credentials from environment or secrets
        if not api_key and not api_token:
            from src.utils.system_secrets import get_system_secrets
            system_secrets = get_system_secrets()
            self.api_key = system_secrets.graph_api_key
            self.api_token = system_secrets.graph_api_token
        else:
            self.api_key = api_key
            self.api_token = api_token
        
        # Use Arbitrum gateway (decentralized network) without API key for now
        # The Graph allows some free queries
        # TODO: Add proper API key authentication once format is confirmed
        self.base_url = "https://api.studio.thegraph.com/query/1931"
        
        # For testing, we'll use public endpoints that don't require auth
        self.logger.info("Using The Graph public endpoints (rate limited)")
        
        # Set authorization header if token is available
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"
    
    async def query_subgraph(
        self, 
        subgraph_name: str, 
        query: str, 
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute GraphQL query against a subgraph
        
        Args:
            subgraph_name: Name of the subgraph from SUBGRAPH_IDS
            query: GraphQL query string
            variables: Optional query variables
            
        Returns:
            Query result data
            
        Raises:
            ValueError: If subgraph_name not found
            MarketDataError: If query fails
        """
        # Use public endpoints for now (no auth required)
        if subgraph_name not in self.SUBGRAPH_ENDPOINTS:
            raise ValueError(f"Unknown subgraph: {subgraph_name}. Available: {list(self.SUBGRAPH_ENDPOINTS.keys())}")
        
        # Use the direct endpoint URL
        url = self.SUBGRAPH_ENDPOINTS[subgraph_name]
        
        # Prepare GraphQL request
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        try:
            # Execute query
            session = await self._get_session()
            async with session.post(url, json=payload, headers=self.headers) as response:
                result = await response.json()
                
                # Check for errors
                if "errors" in result:
                    error_msg = "; ".join([e.get("message", str(e)) for e in result["errors"]])
                    raise MarketDataError(f"GraphQL query error: {error_msg}")
                
                # Return data
                return result.get("data", {})
                
        except Exception as e:
            self.logger.error(f"Failed to query subgraph {subgraph_name}", error=str(e))
            raise MarketDataError(f"Failed to query {subgraph_name}: {str(e)}")
    
    async def get_historical_defi_metrics(
        self,
        protocols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "daily"
    ) -> List[DeFiMetrics]:
        """Get historical DeFi metrics from multiple protocols
        
        Args:
            protocols: List of protocol names (from SUBGRAPH_IDS)
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (daily, hourly)
            
        Returns:
            List of DeFiMetrics objects with aggregated data
        """
        all_metrics = []
        protocol_data = {}
        
        # Calculate timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Query each protocol
        for protocol in protocols:
            try:
                if protocol == "uniswap_v3":
                    data = await self._get_uniswap_v3_metrics(start_timestamp, end_timestamp)
                elif protocol == "aave_v3":
                    data = await self._get_aave_v3_metrics(start_timestamp, end_timestamp)
                elif protocol == "compound_v3":
                    data = await self._get_compound_v3_metrics(start_timestamp, end_timestamp)
                else:
                    self.logger.warning(f"Protocol {protocol} not yet implemented")
                    continue
                
                protocol_data[protocol] = data
                
            except Exception as e:
                self.logger.error(f"Failed to get data for {protocol}", error=str(e))
                continue
        
        # Aggregate data by date
        date_metrics = {}
        
        for protocol, data_points in protocol_data.items():
            for point in data_points:
                date = point["date"]
                if date not in date_metrics:
                    date_metrics[date] = {
                        "total_tvl": 0,
                        "total_volume": 0,
                        "protocols": [],
                        "timestamp": datetime.fromtimestamp(date, tz=timezone.utc)
                    }
                
                date_metrics[date]["total_tvl"] += point.get("tvl", 0)
                date_metrics[date]["total_volume"] += point.get("volume", 0)
                date_metrics[date]["protocols"].append(protocol)
        
        # Convert to DeFiMetrics objects
        for date, metrics in sorted(date_metrics.items()):
            defi_metric = DeFiMetrics(
                total_value_locked=metrics["total_tvl"],
                tvl_change_24h=0,  # Calculate from previous day if needed
                tvl_change_7d=0,
                protocol_count=len(set(metrics["protocols"])),
                defi_dominance=min(metrics["total_tvl"] / 100_000_000_000, 1.0),  # Rough estimate
                chains_tvl={},  # Could aggregate by chain if needed
                lending_tvl=0,  # Could separate by protocol type
                dex_tvl=metrics["total_tvl"] if "uniswap" in metrics["protocols"] else 0,
                derivatives_tvl=0,
                timestamp=metrics["timestamp"]
            )
            all_metrics.append(defi_metric)
        
        return all_metrics
    
    async def _get_uniswap_v3_metrics(self, start_timestamp: int, end_timestamp: int) -> List[Dict]:
        """Get Uniswap V3 historical metrics"""
        query = """
        query($start: Int!, $end: Int!) {
            uniswapDayDatas(
                first: 1000,
                orderBy: date,
                orderDirection: asc,
                where: { 
                    date_gte: $start,
                    date_lte: $end
                }
            ) {
                date
                tvlUSD
                volumeUSD
                txCount
                feesUSD
            }
        }
        """
        
        variables = {
            "start": start_timestamp,
            "end": end_timestamp
        }
        
        result = await self.query_subgraph("uniswap_v3", query, variables)
        
        metrics = []
        for day_data in result.get("uniswapDayDatas", []):
            metrics.append({
                "date": int(day_data["date"]),
                "tvl": float(day_data["tvlUSD"]),
                "volume": float(day_data["volumeUSD"]),
                "tx_count": int(day_data["txCount"]),
                "fees": float(day_data.get("feesUSD", 0))
            })
        
        return metrics
    
    async def _get_aave_v3_metrics(self, start_timestamp: int, end_timestamp: int) -> List[Dict]:
        """Get Aave V3 historical metrics"""
        # Note: Aave subgraph structure may be different
        # This is a simplified query - adjust based on actual subgraph schema
        query = """
        query($start: Int!, $end: Int!) {
            markets(first: 100) {
                id
                totalValueLockedUSD
                totalBorrowsUSD
                totalDepositBalanceUSD
            }
        }
        """
        
        variables = {
            "start": start_timestamp,
            "end": end_timestamp
        }
        
        result = await self.query_subgraph("aave_v3", query, variables)
        
        # Aggregate market TVLs
        total_tvl = 0
        for market in result.get("markets", []):
            total_tvl += float(market.get("totalValueLockedUSD", 0))
        
        # Return single point for now (could enhance with daily snapshots)
        return [{
            "date": start_timestamp,
            "tvl": total_tvl,
            "volume": 0,  # Not directly available
            "tx_count": 0,
            "fees": 0
        }]
    
    async def _get_compound_v3_metrics(self, start_timestamp: int, end_timestamp: int) -> List[Dict]:
        """Get Compound V3 historical metrics"""
        # Simplified query - adjust based on actual subgraph schema
        query = """
        query {
            markets(first: 10) {
                id
                name
                totalSupply
                totalBorrow
            }
        }
        """
        
        result = await self.query_subgraph("compound_v3", query)
        
        # Calculate total TVL (supply - borrow)
        total_tvl = 0
        for market in result.get("markets", []):
            supply = float(market.get("totalSupply", 0))
            borrow = float(market.get("totalBorrow", 0))
            total_tvl += (supply - borrow)
        
        return [{
            "date": start_timestamp,
            "tvl": total_tvl,
            "volume": 0,
            "tx_count": 0,
            "fees": 0
        }]
    
    async def get_market_data(self) -> Optional[DeFiMetrics]:
        """Get current DeFi metrics (implements base class method)"""
        # Get latest data from major protocols
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=1)
        
        metrics = await self.get_historical_defi_metrics(
            protocols=["uniswap_v3", "aave_v3"],
            start_date=start_date,
            end_date=end_date,
            interval="daily"
        )
        
        return metrics[-1] if metrics else None


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
    
    def __init__(self, api_key: Optional[str] = None, tier: str = "enterprise", **kwargs):
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("LUNARCRUSH_API_KEY")
        
        super().__init__(api_key=api_key, **kwargs)
        
        if not self.api_key:
            raise APIAuthenticationError("LunarCrush API key is required. Set LUNARCRUSH_API_KEY environment variable or pass api_key parameter.")
        
        self.tier = tier  # 'basic', 'pro', or 'enterprise'
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def get_historical_sentiment(self, asset: str, timestamp: datetime) -> Optional[SocialSentimentData]:
        """Get historical sentiment data for a specific timestamp (requires Pro/Enterprise tier)
        
        Args:
            asset: Cryptocurrency to get sentiment for (e.g., 'bitcoin')
            timestamp: Historical timestamp to get data for
            
        Returns:
            SocialSentimentData or None if not available
        """
        if self.tier not in ["pro", "enterprise"]:
            self.logger.warning(
                "Historical sentiment requires Pro or Enterprise tier",
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
                key=lambda x: abs(datetime.fromtimestamp(x.get("time", 0), tz=timezone.utc) - timestamp)
            )
            
            # Extract sentiment data from historical point
            return SocialSentimentData(
                social_score=closest_point.get("sentiment", 50) / 100.0,  # Convert 0-100 to 0-1
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
        
        # Simple linear trend of sentiment over time (0-100 scale)
        sentiments = [p.get("sentiment", 50) for p in data_points]
        if len(sentiments) < 2:
            return 0.0
        
        # Calculate change from first half to second half average
        mid = len(sentiments) // 2
        first_half_avg = sum(sentiments[:mid]) / mid if mid > 0 else 50
        second_half_avg = sum(sentiments[mid:]) / len(sentiments[mid:]) if sentiments[mid:] else 50
        
        # Normalize to -1 to 1 range (50 point swing = full trend)
        trend = (second_half_avg - first_half_avg) / 50
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
        
        # LunarCrush rate limits: 5 requests per second for basic tier
        rate_limit_delay = 0.25  # 4 requests per second to be safe
        
        for i, asset in enumerate(assets):
            try:
                # Add delay between requests to respect rate limit
                if i > 0:
                    await asyncio.sleep(rate_limit_delay)
                
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
                        token_id, timestamp, sentiment_score, 
                        twitter_mentions, reddit_posts, 
                        social_volume_24h, social_engagement_24h,
                        bullish_percentage, bearish_percentage,
                        data_source, collection_timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    asset.upper(),  # token_id
                    sentiment_data.timestamp,  # timestamp
                    sentiment_data.social_score,  # sentiment_score
                    sentiment_data.platform_mentions.get('twitter', 0) if sentiment_data.platform_mentions else 0,  # twitter_mentions
                    sentiment_data.platform_mentions.get('reddit', 0) if sentiment_data.platform_mentions else 0,  # reddit_posts
                    sentiment_data.mention_volume,  # social_volume_24h
                    len(sentiment_data.platform_mentions) if sentiment_data.platform_mentions else 0,  # social_engagement_24h
                    sentiment_data.social_score * 100,  # bullish_percentage (convert 0-1 to percentage)
                    (1 - sentiment_data.social_score) * 100,  # bearish_percentage
                    "live",  # data_source (must match enum)
                    datetime.now(timezone.utc)  # collection_timestamp
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
    
    async def get_historical_data(self, 
                                 asset: str,
                                 start_date: datetime, 
                                 end_date: datetime,
                                 interval: str = "1d") -> List[SocialSentimentData]:
        """Get historical sentiment data for a date range (requires Enterprise tier)
        
        Args:
            asset: Cryptocurrency to get sentiment for (e.g., 'bitcoin')
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval ('1h', '1d', '1w')
            
        Returns:
            List of SocialSentimentData objects for each time period
        """
        if self.tier not in ["pro", "enterprise"]:
            self.logger.warning(
                "Historical time series requires Pro or Enterprise tier",
                asset=asset,
                tier=self.tier
            )
            return []
        
        historical_data = []
        
        try:
            # Use time-series endpoint for historical data
            url = f"{self.BASE_URL}/public/topic/{asset}/time-series/v2"
            
            params = {
                "start": int(start_date.timestamp()),
                "end": int(end_date.timestamp()),
                "interval": interval
            }
            
            self.logger.info(f"Fetching LunarCrush historical data for {asset}", 
                           start=start_date.isoformat(), 
                           end=end_date.isoformat(),
                           interval=interval)
            
            response = await self._make_request(url, params=params, headers=self.headers)
            data_points = response.get("data", [])
            
            if not data_points:
                self.logger.warning(f"No historical data available for {asset}")
                return []
            
            self.logger.debug(f"Sample data point: {data_points[0] if data_points else 'None'}")
            
            # Convert each data point to SocialSentimentData
            for point in data_points:
                timestamp = datetime.fromtimestamp(point.get("time", 0), tz=timezone.utc)
                
                sentiment_data = SocialSentimentData(
                    social_score=point.get("sentiment", 50) / 100.0,  # Convert 0-100 to 0-1
                    mention_volume=point.get("posts", 0),
                    sentiment_trend=0.0,  # Will be calculated from overall series
                    platform_mentions={
                        "twitter": point.get("twitter_posts", 0),
                        "reddit": point.get("reddit_posts", 0),
                        "youtube": point.get("youtube_posts", 0),
                        "telegram": point.get("telegram_posts", 0)
                    },
                    sentiment_breakdown={},  # Not available in time series
                    trending_keywords=[],  # Not available in time series
                    influencer_sentiment=None,  # Not available in time series
                    timestamp=timestamp
                )
                
                historical_data.append(sentiment_data)
            
            # Calculate trend for each data point based on surrounding context
            for i, data in enumerate(historical_data):
                # Calculate trend based on surrounding points
                window_start = max(0, i - 3)
                window_end = min(len(historical_data), i + 4)
                window_data = historical_data[window_start:window_end]
                
                if len(window_data) >= 2:
                    # Simple linear trend calculation
                    first_half = window_data[:len(window_data)//2]
                    second_half = window_data[len(window_data)//2:]
                    
                    first_avg = sum(d.social_score for d in first_half) / len(first_half)
                    second_avg = sum(d.social_score for d in second_half) / len(second_half)
                    
                    data.sentiment_trend = max(-1.0, min(1.0, (second_avg - first_avg) * 2))
            
            # If we requested daily data but got hourly, aggregate to daily
            if interval == '1d' and len(historical_data) > (end_date - start_date).days * 2:
                self.logger.info(f"Aggregating {len(historical_data)} hourly points to daily")
                daily_data = {}
                
                for data_point in historical_data:
                    date_key = data_point.timestamp.date()
                    
                    if date_key not in daily_data:
                        daily_data[date_key] = []
                    daily_data[date_key].append(data_point)
                
                # Average each day's data
                aggregated_data = []
                for date_key, day_points in sorted(daily_data.items()):
                    if not day_points:
                        continue
                    
                    # Average the scores and sum the volumes for the day
                    avg_score = sum(p.social_score for p in day_points) / len(day_points)
                    total_volume = sum(p.mention_volume for p in day_points)
                    avg_trend = sum(p.sentiment_trend for p in day_points) / len(day_points)
                    
                    # Use the last data point of the day for timestamp
                    daily_sentiment = SocialSentimentData(
                        social_score=avg_score,
                        mention_volume=total_volume,
                        sentiment_trend=avg_trend,
                        platform_mentions=day_points[-1].platform_mentions,
                        sentiment_breakdown=day_points[-1].sentiment_breakdown,
                        trending_keywords=day_points[-1].trending_keywords,
                        influencer_sentiment=day_points[-1].influencer_sentiment,
                        timestamp=day_points[-1].timestamp
                    )
                    aggregated_data.append(daily_sentiment)
                
                self.logger.info(f"Aggregated to {len(aggregated_data)} daily data points for {asset}")
                return aggregated_data
            
            self.logger.info(f"Retrieved {len(historical_data)} historical data points for {asset}")
            return sorted(historical_data, key=lambda x: x.timestamp)
            
        except Exception as e:
            self.logger.error(
                "Failed to get historical time series",
                asset=asset,
                error=str(e)
            )
            return []
    
    async def _calculate_sentiment_trend(self, asset: str) -> float:
        """Calculate sentiment trend from time series data"""
        try:
            # For Enterprise tier, use time series data
            if self.tier in ["pro", "enterprise"]:
                url = f"{self.BASE_URL}/public/topic/{asset}/time-series/v2"
                params = {"interval": "1h", "data_points": 24}  # Last 24 hours
                response = await self._make_request(url, params=params, headers=self.headers)
                data_points = response.get("data", [])
                
                if len(data_points) >= 2:
                    return self._calculate_trend_from_series(data_points)
            
            # Return neutral trend for free tier
            return 0.0
            
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