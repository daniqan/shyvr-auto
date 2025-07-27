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
        """Make HTTP request with error handling"""
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