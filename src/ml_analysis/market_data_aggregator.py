"""
Market Data Aggregator
Aggregates data from multiple market data sources and provides unified interface
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog

from .market_data import (
    FearGreedIndexClient,
    DeFiLlamaClient, 
    CoinGeckoClient,
    OnChainAnalyticsClient,
    SocialSentimentClient,
    MarketDataError
)
from .base import MarketFeatures
from .config import get_market_data_config, MarketDataConfig
from src.utils.base import Chain


logger = structlog.get_logger()


class MarketDataAggregator:
    """Aggregates market data from multiple sources for ML feature engineering"""
    
    def __init__(self, 
                 config: Optional[MarketDataConfig] = None,
                 coingecko_api_key: Optional[str] = None,  # Legacy parameter for backward compatibility
                 cache_ttl: Optional[int] = None,  # Legacy parameter for backward compatibility
                 max_retries: Optional[int] = None,
                 retry_delay: Optional[float] = None):
        self.logger = structlog.get_logger().bind(component="MarketDataAggregator")
        
        # Load configuration
        if config is None:
            config = get_market_data_config()
        self.config = config
        
        # Override config with legacy parameters if provided
        if cache_ttl is not None:
            self.config.default_cache_ttl = cache_ttl
        if max_retries is not None:
            for api_config in [self.config.fear_greed, self.config.defi_llama, 
                              self.config.coingecko, self.config.coingecko_pro]:
                api_config.max_retries = max_retries
        if retry_delay is not None:
            for api_config in [self.config.fear_greed, self.config.defi_llama, 
                              self.config.coingecko, self.config.coingecko_pro]:
                api_config.retry_delay = retry_delay
        
        # Override CoinGecko API key if provided
        if coingecko_api_key:
            if self.config.coingecko_pro.enabled:
                self.config.coingecko_pro.api_key = coingecko_api_key
                self.config.coingecko_pro.headers["X-CG-Pro-API-Key"] = coingecko_api_key
            else:
                self.config.coingecko.api_key = coingecko_api_key
        
        # Initialize API clients based on configuration
        self._init_clients()
        
        # Cache for aggregated market features
        self._aggregated_cache: Optional[MarketFeatures] = None
        self._cache_timestamp: Optional[datetime] = None
    
    def _init_clients(self):
        """Initialize API clients based on configuration"""
        self.clients = {}
        
        # Fear & Greed Index
        if self.config.fear_greed.enabled:
            self.fear_greed_client = FearGreedIndexClient(
                cache_ttl=self.config.fear_greed.cache_ttl,
                rate_limit=self.config.fear_greed.rate_limit
            )
            self.clients['fear_greed'] = self.fear_greed_client
        else:
            self.fear_greed_client = None
        
        # DeFi Llama
        if self.config.defi_llama.enabled:
            self.defi_client = DeFiLlamaClient(
                cache_ttl=self.config.defi_llama.cache_ttl,
                rate_limit=self.config.defi_llama.rate_limit
            )
            self.clients['defi_llama'] = self.defi_client
        else:
            self.defi_client = None
        
        # CoinGecko (Pro or Free)
        if self.config.coingecko_pro.enabled and self.config.coingecko_pro.api_key:
            self.coingecko_client = CoinGeckoClient(
                api_key=self.config.coingecko_pro.api_key,
                cache_ttl=self.config.coingecko_pro.cache_ttl,
                rate_limit=self.config.coingecko_pro.rate_limit
            )
            self.clients['coingecko'] = self.coingecko_client
        elif self.config.coingecko.enabled:
            self.coingecko_client = CoinGeckoClient(
                api_key=self.config.coingecko.api_key,
                cache_ttl=self.config.coingecko.cache_ttl,
                rate_limit=self.config.coingecko.rate_limit
            )
            self.clients['coingecko'] = self.coingecko_client
        else:
            self.coingecko_client = None
        
        # Get API keys from SystemSecrets
        from src.utils.system_secrets import get_system_secrets
        system_secrets = get_system_secrets()
        
        # On-chain analytics
        helius_key = system_secrets.helius_api_key
        if helius_key:
            self.onchain_client = OnChainAnalyticsClient(
                api_key=helius_key,
                cache_ttl=self.config.default_cache_ttl
            )
            self.clients['onchain'] = self.onchain_client
        else:
            self.onchain_client = None
            self.logger.warning("Helius API key not available, on-chain analytics disabled")
        
        # Social sentiment
        lunarcrush_key = system_secrets.lunarcrush_api_key
        
        if lunarcrush_key:
            self.social_client = SocialSentimentClient(
                api_key=lunarcrush_key,
                cache_ttl=self.config.default_cache_ttl
            )
            self.clients['social'] = self.social_client
        else:
            self.social_client = None
            self.logger.warning("LunarCrush API key not available, social sentiment disabled")
        
        self.logger.info("Market data clients initialized", 
                        enabled_clients=list(self.clients.keys()))
    
    def _is_cache_valid(self) -> bool:
        """Check if aggregated cache is still valid"""
        if not self._cache_timestamp:
            return False
        age = datetime.now() - self._cache_timestamp
        return age.total_seconds() < self.config.default_cache_ttl
    
    async def _retry_request(self, func, client_name: str, *args, **kwargs) -> Optional[Any]:
        """Execute function with retry logic based on client configuration"""
        # Get retry config for the specific client
        if client_name == 'fear_greed':
            retry_config = self.config.fear_greed
        elif client_name == 'defi_llama':
            retry_config = self.config.defi_llama
        elif client_name == 'coingecko':
            retry_config = self.config.coingecko_pro if self.config.coingecko_pro.enabled else self.config.coingecko
        else:
            # Default config for other clients
            retry_config = self.config.fear_greed
        
        for attempt in range(retry_config.max_retries):
            try:
                return await func(*args, **kwargs)
            except MarketDataError as e:
                if attempt == retry_config.max_retries - 1:
                    self.logger.error("Max retries exceeded", 
                                    client=client_name,
                                    function=func.__name__, 
                                    error=str(e))
                    return None
                
                await asyncio.sleep(retry_config.retry_delay * (2 ** attempt))  # Exponential backoff
                self.logger.warning("Retrying request", 
                                  client=client_name,
                                  function=func.__name__, 
                                  attempt=attempt + 1, 
                                  error=str(e))
        return None
    
    async def get_market_features(self, primary_chain: Chain = Chain.ETHEREUM) -> MarketFeatures:
        """
        Aggregate market features from all available sources
        
        Args:
            primary_chain: Primary chain for on-chain metrics
        
        Returns:
            MarketFeatures object with aggregated data
        """
        # Check cache first
        if self._is_cache_valid() and self._aggregated_cache:
            return self._aggregated_cache
        
        self.logger.info("Aggregating market data from all sources")
        
        # Fetch data from all sources concurrently
        tasks = []
        
        # Only create tasks for enabled clients
        if self.fear_greed_client:
            tasks.append(('fear_greed', self._retry_request(self.fear_greed_client.get_market_data, 'fear_greed')))
        else:
            tasks.append(('fear_greed', asyncio.coroutine(lambda: None)()))
        
        if self.defi_client:
            tasks.append(('defi', self._retry_request(self.defi_client.get_market_data, 'defi_llama')))
        else:
            tasks.append(('defi', asyncio.coroutine(lambda: None)()))
        
        if self.coingecko_client:
            tasks.append(('correlation', self._retry_request(self.coingecko_client.get_market_data, 'coingecko')))
        else:
            tasks.append(('correlation', asyncio.coroutine(lambda: None)()))
        
        tasks.append(('onchain', self._retry_request(self.onchain_client.get_market_data, 'onchain', primary_chain)))
        tasks.append(('social', self._retry_request(self.social_client.get_market_data, 'social', "bitcoin")))
        
        # Execute tasks
        task_coroutines = [task[1] for task in tasks]
        
        try:
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            sentiment_data, defi_data, correlation_data, onchain_data, social_data = results
            
            # Handle exceptions from individual sources
            if isinstance(sentiment_data, Exception):
                self.logger.warning("Fear & Greed data unavailable", error=str(sentiment_data))
                sentiment_data = None
            
            if isinstance(defi_data, Exception):
                self.logger.warning("DeFi data unavailable", error=str(defi_data))
                defi_data = None
            
            if isinstance(correlation_data, Exception):
                self.logger.warning("Correlation data unavailable", error=str(correlation_data))
                correlation_data = None
            
            if isinstance(onchain_data, Exception):
                self.logger.warning("On-chain data unavailable", error=str(onchain_data))
                onchain_data = None
            
            if isinstance(social_data, Exception):
                self.logger.warning("Social data unavailable", error=str(social_data))
                social_data = None
            
            # Aggregate all data into MarketFeatures
            market_features = MarketFeatures(
                # Market sentiment features
                fear_greed_index=sentiment_data.fear_greed_index if sentiment_data else None,
                fear_greed_classification=sentiment_data.fear_greed_classification if sentiment_data else None,
                market_trend=sentiment_data.market_trend if sentiment_data else None,
                volatility_regime=sentiment_data.volatility_regime if sentiment_data else None,
                
                # Cross-asset correlations
                btc_correlation=correlation_data.btc_correlation if correlation_data else None,
                eth_correlation=correlation_data.eth_correlation if correlation_data else None,
                btc_dominance=correlation_data.btc_dominance if correlation_data else None,
                eth_dominance=correlation_data.eth_dominance if correlation_data else None,
                stablecoin_dominance=correlation_data.stablecoin_dominance if correlation_data else None,
                market_beta=correlation_data.market_beta if correlation_data else None,
                
                # DeFi ecosystem metrics
                total_value_locked=defi_data.total_value_locked if defi_data else None,
                tvl_change_24h=defi_data.tvl_change_24h if defi_data else None,
                tvl_change_7d=defi_data.tvl_change_7d if defi_data else None,
                defi_dominance=defi_data.defi_dominance if defi_data else None,
                active_protocols=defi_data.protocols_count if defi_data else None,
                
                # On-chain activity metrics
                transaction_count_24h=onchain_data.transaction_count_24h if onchain_data else None,
                active_addresses_24h=onchain_data.active_addresses_24h if onchain_data else None,
                transaction_volume_24h=onchain_data.transaction_volume_24h if onchain_data else None,
                network_fees_24h=onchain_data.network_fees_24h if onchain_data else None,
                whale_activity_score=self._calculate_whale_activity_score(onchain_data) if onchain_data else None,
                
                # Social sentiment
                social_score=social_data.social_score if social_data else None,
                mention_volume=social_data.mention_volume if social_data else None,
                sentiment_trend=social_data.sentiment_trend if social_data else None,
                influencer_sentiment=social_data.influencer_sentiment if social_data else None,
            )
            
            # Cache the result
            self._aggregated_cache = market_features
            self._cache_timestamp = datetime.now()
            
            # Log successful aggregation
            available_sources = sum([
                1 for data in [sentiment_data, defi_data, correlation_data, onchain_data, social_data]
                if data is not None
            ])
            
            self.logger.info("Market data aggregated successfully", 
                           available_sources=available_sources,
                           total_sources=5,
                           cache_ttl=self.cache_ttl)
            
            return market_features
            
        except Exception as e:
            self.logger.error("Failed to aggregate market data", error=str(e))
            # Return market features with default values
            return self._get_default_market_features()
    
    def _calculate_whale_activity_score(self, onchain_data) -> float:
        """Calculate whale activity score from on-chain data"""
        if not onchain_data or not onchain_data.whale_activity:
            return 0.5
        
        whale_data = onchain_data.whale_activity
        large_tx_count = whale_data.get("large_transactions_24h", 0)
        net_flow = whale_data.get("whale_net_flow", 0)
        
        # Normalize to 0-1 scale
        # More large transactions = higher score
        tx_score = min(large_tx_count / 200, 1.0)  # Normalized to 200 max
        
        # Positive net flow (accumulation) increases score
        flow_score = min(abs(net_flow) / 10000, 0.5)  # Up to 0.5 contribution
        if net_flow > 0:  # Accumulation
            flow_score = 0.5 + flow_score
        else:  # Distribution
            flow_score = 0.5 - flow_score
        
        return min((tx_score + flow_score) / 2, 1.0)
    
    def _get_default_market_features(self) -> MarketFeatures:
        """Return default market features when data is unavailable"""
        return MarketFeatures(
            fear_greed_index=50.0,
            fear_greed_classification="Neutral",
            market_trend="sideways",
            volatility_regime="medium",
            btc_correlation=0.5,
            eth_correlation=0.4,
            btc_dominance=40.0,
            eth_dominance=15.0,
            stablecoin_dominance=10.0,
            market_beta=1.0,
            total_value_locked=100e9,  # $100B default
            tvl_change_24h=0.0,
            tvl_change_7d=0.0,
            defi_dominance=0.05,
            active_protocols=300,
            transaction_count_24h=1000000,
            active_addresses_24h=500000,
            transaction_volume_24h=5e9,
            network_fees_24h=1e6,
            whale_activity_score=0.5,
            social_score=0.5,
            mention_volume=1000,
            sentiment_trend=0.0,
            influencer_sentiment=0.5,
        )
    
    async def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get a summary of current market sentiment"""
        try:
            features = await self.get_market_features()
            
            return {
                "fear_greed_index": features.fear_greed_index,
                "fear_greed_classification": features.fear_greed_classification,
                "market_trend": features.market_trend,
                "volatility_regime": features.volatility_regime,
                "btc_dominance": features.btc_dominance,
                "defi_tvl": features.total_value_locked,
                "social_sentiment": features.social_score,
                "whale_activity": features.whale_activity_score,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error("Failed to get sentiment summary", error=str(e))
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all market data sources"""
        health_status = {}
        
        clients = [
            ("fear_greed", self.fear_greed_client),
            ("defi_llama", self.defi_client),
            ("coingecko", self.coingecko_client),
            ("onchain", self.onchain_client),
            ("social", self.social_client)
        ]
        
        for name, client in clients:
            try:
                # Try a simple request to check if the client is working
                if hasattr(client, 'get_market_data'):
                    await asyncio.wait_for(client.get_market_data(), timeout=10)
                    health_status[name] = True
                else:
                    health_status[name] = False
            except Exception as e:
                self.logger.warning(f"{name} health check failed", error=str(e))
                health_status[name] = False
        
        return health_status
    
    def clear_cache(self):
        """Clear all cached data"""
        self._aggregated_cache = None
        self._cache_timestamp = None
        
        # Clear individual client caches
        for client in [self.fear_greed_client, self.defi_client, self.coingecko_client, 
                      self.onchain_client, self.social_client]:
            if hasattr(client, '_cache'):
                client._cache.clear()
                client._cache_timestamps.clear()
        
        self.logger.info("Market data cache cleared")
    
    async def close(self):
        """Close all HTTP sessions"""
        await asyncio.gather(
            self.fear_greed_client.close(),
            self.defi_client.close(),
            self.coingecko_client.close(),
            self.onchain_client.close(),
            self.social_client.close(),
            return_exceptions=True
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()