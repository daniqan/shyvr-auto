"""
Market Data API Clients for Cryptocurrency Trading
Provides real-time market sentiment, DeFi metrics, and on-chain data
"""

import asyncio
import aiohttp
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import structlog
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
import time

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
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
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
            # Get total TVL
            tvl_data = await self._make_request(f"{self.BASE_URL}/tvl")
            
            # Get chain TVL
            chains_data = await self._make_request(f"{self.BASE_URL}/chains")
            
            # Get protocols count
            protocols_data = await self._make_request(f"{self.BASE_URL}/protocols")
            
            # Extract current and historical TVL
            current_tvl = float(tvl_data.get("totalTvl", 0))
            
            # Calculate TVL changes
            tvl_24h_ago = float(tvl_data.get("totalTvl24hAgo", current_tvl))
            tvl_7d_ago = float(tvl_data.get("totalTvl7dAgo", current_tvl))
            
            tvl_change_24h = ((current_tvl - tvl_24h_ago) / tvl_24h_ago * 100) if tvl_24h_ago > 0 else 0
            tvl_change_7d = ((current_tvl - tvl_7d_ago) / tvl_7d_ago * 100) if tvl_7d_ago > 0 else 0
            
            # Extract chain TVL data
            chains_tvl = {}
            for chain in chains_data:
                if chain.get("name") and chain.get("tvl"):
                    chains_tvl[chain["name"]] = float(chain["tvl"])
            
            # Count active protocols
            protocols_count = len([p for p in protocols_data if p.get("tvl", 0) > 1000000])  # > $1M TVL
            
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
    """Client for CoinGecko API for market data and correlations"""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.headers = {}
        if api_key:
            self.headers["X-CG-Pro-API-Key"] = api_key
    
    async def get_market_data(self) -> CorrelationMetrics:
        """Get Bitcoin dominance and correlation metrics"""
        cache_key = "correlation_metrics"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            # Get global market data
            global_data = await self._make_request(
                f"{self.BASE_URL}/global", 
                headers=self.headers
            )
            
            # Get top cryptocurrencies for correlation analysis
            coins_data = await self._make_request(
                f"{self.BASE_URL}/coins/markets",
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
        """Get OHLCV (Open, High, Low, Close, Volume) data for a coin"""
        cache_key = f"ohlcv_{coin_id}_{days}_{from_date}_{to_date}"
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
            
            # Make API request
            url = f"{self.BASE_URL}/coins/{coin_id}/ohlc"
            data = await self._make_request(url, params=params, headers=self.headers)
            
            if not data:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to DataFrame
            ohlcv_data = []
            for entry in data:
                if len(entry) != 6:  # timestamp, o, h, l, c, v
                    raise MarketDataError(f"Invalid OHLCV data format: expected 6 values, got {len(entry)}")
                
                ohlcv_data.append({
                    'timestamp': pd.to_datetime(entry[0], unit='ms'),
                    'open': float(entry[1]),
                    'high': float(entry[2]),
                    'low': float(entry[3]),
                    'close': float(entry[4]),
                    'volume': float(entry[5])
                })
            
            df = pd.DataFrame(ohlcv_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
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
    
    async def get_historical_data_for_token(self, token_address: str, days: int = 7,
                                          from_date: Optional[datetime] = None,
                                          to_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical OHLCV data for a token by its contract address"""
        try:
            # First, try to find the coin by contract address
            # Note: CoinGecko API doesn't directly support contract address lookup for OHLCV
            # This would need to be enhanced with contract platform mapping
            
            # For now, we'll assume the token_address maps to a known coin ID
            # In production, you would need to:
            # 1. Use CoinGecko's coins/{id}/contract/{contract_address} endpoint
            # 2. Or maintain a mapping of contract addresses to coin IDs
            
            # Placeholder implementation - would need proper contract address resolution
            coin_id = await self._resolve_contract_to_coin_id(token_address)
            
            return await self.get_ohlcv_data(coin_id, days, from_date, to_date)
            
        except Exception as e:
            self.logger.error("Failed to get historical data for token", 
                            token_address=token_address, 
                            error=str(e))
            raise MarketDataError(f"Failed to get historical data for token {token_address}: {str(e)}")
    
    async def _resolve_contract_to_coin_id(self, contract_address: str) -> str:
        """Resolve contract address to CoinGecko coin ID"""
        # This is a simplified implementation
        # In production, you would use CoinGecko's contract address endpoints
        # or maintain a mapping of known contract addresses
        
        # For common tokens, we can provide direct mappings
        contract_mappings = {
            "0xa0b86a33e6b58ee28de3a76f8a9e54e4da5e7f2f": "bitcoin",  # Example
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "ethereum", # WETH
            # Add more mappings as needed
        }
        
        address_lower = contract_address.lower()
        if address_lower in contract_mappings:
            return contract_mappings[address_lower]
        
        # If not in mapping, try to search by the address
        # This would need enhancement with proper CoinGecko contract lookup
        raise DataNotAvailableError(f"Cannot resolve contract address to coin ID: {contract_address}")


class OnChainAnalyticsClient(MarketDataClientBase):
    """Client for on-chain analytics (using multiple data sources)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Would integrate with services like Glassnode, Messari, etc.
        # For now, providing a framework that can be extended
    
    async def get_market_data(self, chain: Chain = Chain.ETHEREUM) -> OnChainMetrics:
        """Get on-chain metrics for specified chain"""
        cache_key = f"onchain_{chain.value}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            # This is a placeholder implementation
            # In production, would integrate with services like:
            # - Glassnode API
            # - Messari API  
            # - Etherscan API
            # - Solscan API
            # - Chain-specific RPC endpoints
            
            # Simulate some realistic on-chain data
            result = OnChainMetrics(
                network_activity={
                    "chain": chain.value,
                    "gas_price": 20.0,  # Placeholder
                    "block_time": 12.0,  # Placeholder
                },
                transaction_count_24h=1000000,  # Placeholder
                active_addresses_24h=500000,    # Placeholder
                transaction_volume_24h=5e9,     # Placeholder
                network_fees_24h=1e6,           # Placeholder
                hash_rate=200000.0 if chain == Chain.BITCOIN else None,
                staking_ratio=0.65 if chain == Chain.ETHEREUM else None,
                whale_activity={
                    "large_transactions_24h": 100,
                    "whale_net_flow": 1000.0
                }
            )
            
            self._cache_data(cache_key, result)
            self.logger.info("Retrieved on-chain metrics", chain=chain.value)
            return result
            
        except Exception as e:
            self.logger.error("Failed to get on-chain metrics", chain=chain.value, error=str(e))
            raise MarketDataError(f"Failed to get on-chain metrics: {str(e)}")


class SocialSentimentClient(MarketDataClientBase):
    """Client for social sentiment analysis"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Would integrate with services like:
        # - LunarCrush API
        # - Santiment API
        # - Twitter API
        # - Reddit API
    
    async def get_market_data(self, asset: str = "bitcoin") -> SocialSentimentData:
        """Get social sentiment data for specified asset"""
        cache_key = f"social_{asset}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            # This is a placeholder implementation
            # In production, would integrate with sentiment analysis APIs
            
            # Simulate social sentiment data
            result = SocialSentimentData(
                social_score=0.65,  # Placeholder positive sentiment
                mention_volume=5000,  # Placeholder mentions
                sentiment_trend=0.05,  # Placeholder trending up
                platform_mentions={
                    "twitter": 3000,
                    "reddit": 1500,
                    "telegram": 500
                },
                sentiment_breakdown={
                    "twitter": 0.7,
                    "reddit": 0.6,
                    "telegram": 0.65
                },
                trending_keywords=["bullish", "moon", "hodl"],
                influencer_sentiment=0.75
            )
            
            self._cache_data(cache_key, result)
            self.logger.info("Retrieved social sentiment", asset=asset, score=result.social_score)
            return result
            
        except Exception as e:
            self.logger.error("Failed to get social sentiment", asset=asset, error=str(e))
            raise MarketDataError(f"Failed to get social sentiment: {str(e)}")