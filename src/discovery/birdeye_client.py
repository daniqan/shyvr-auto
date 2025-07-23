"""
Birdeye API Client for Multi-Chain Token Discovery and Trending Analysis
Provides trending token data trusted by 3M+ users across multiple chains
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog

from src.utils.base import Chain
from .base import (
    TokenDiscoveryBase, 
    DiscoveredToken, 
    TokenStatus,
    DiscoveryError,
    APIRateLimitError,
    TokenNotFoundError,
    ChainNotSupportedError
)


logger = structlog.get_logger()


class BirdeyeTokenDiscovery(TokenDiscoveryBase):
    """Birdeye API client for multi-chain token discovery and trending analysis"""
    
    BASE_URL = "https://public-api.birdeye.so"
    
    # Chain mapping for Birdeye API
    CHAIN_MAPPING = {
        Chain.SOLANA: "solana",
        Chain.ETHEREUM: "ethereum", 
        Chain.BASE: "base",
        Chain.POLYGON: "polygon",
        Chain.BSC: "bsc",
        Chain.ARBITRUM: "arbitrum",
    }
    
    def __init__(self, api_key: str, rate_limit: int = 60):
        super().__init__(api_key=api_key, rate_limit=rate_limit)
        self.session: Optional[aiohttp.ClientSession] = None
        self._token_cache: Dict[str, DiscoveredToken] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=10)  # Cache for 10 minutes
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with API key"""
        if self.session is None:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self.session
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make HTTP request to Birdeye API"""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    raise APIRateLimitError("Birdeye API rate limit exceeded")
                elif response.status == 401:
                    raise DiscoveryError("Birdeye API authentication failed - check API key")
                elif response.status != 200:
                    text = await response.text()
                    raise DiscoveryError(f"Birdeye API error {response.status}: {text}")
                
                data = await response.json()
                return data
                
        except aiohttp.ClientError as e:
            raise DiscoveryError(f"Birdeye API request failed: {str(e)}")
    
    def _get_chain_param(self, chain: Chain) -> str:
        """Convert Chain enum to Birdeye chain parameter"""
        if chain not in self.CHAIN_MAPPING:
            raise ChainNotSupportedError(f"Chain {chain} not supported by Birdeye")
        return self.CHAIN_MAPPING[chain]
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[cache_key]
    
    def _cache_token(self, token: DiscoveredToken):
        """Cache token data"""
        cache_key = f"{token.chain.value}:{token.address}"
        self._token_cache[cache_key] = token
        self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
    
    def _parse_birdeye_token(self, token_data: Dict, chain: Chain, discovery_source: str = "birdeye") -> DiscoveredToken:
        """Parse Birdeye API token data into DiscoveredToken"""
        
        # Calculate trending score based on various metrics
        trending_score = 0.0
        price_change_24h = token_data.get("priceChange24h", 0)
        volume_24h = token_data.get("volume24h", 0)
        
        # Higher score for positive price movement
        if price_change_24h > 10:  # >10% gain
            trending_score += 0.4
        elif price_change_24h > 5:  # >5% gain
            trending_score += 0.2
        
        # Higher score for high volume
        if volume_24h > 1000000:  # >$1M volume
            trending_score += 0.3
        elif volume_24h > 100000:  # >$100k volume
            trending_score += 0.2
        
        # Boost score if explicitly in trending endpoint
        if "trending" in discovery_source:
            trending_score += 0.3
        
        # Determine status
        status = TokenStatus.DISCOVERED
        if trending_score > 0.6:
            status = TokenStatus.TRENDING
        elif token_data.get("verified", False):
            status = TokenStatus.VALIDATED
        
        # Extract social mentions if available
        social_mentions = None
        if "socialScore" in token_data:
            social_mentions = token_data.get("socialScore", 0)
        
        return DiscoveredToken(
            address=token_data.get("address", ""),
            chain=chain,
            symbol=token_data.get("symbol", "UNKNOWN"),
            name=token_data.get("name", "Unknown Token"),
            discovered_at=datetime.now(),
            discovery_source=discovery_source,
            status=status,
            price_usd=token_data.get("price"),
            market_cap=token_data.get("marketCap"),
            volume_24h=token_data.get("volume24h"),
            price_change_24h=token_data.get("priceChange24h"),
            decimals=token_data.get("decimals"),
            total_supply=token_data.get("supply"),
            trending_score=min(trending_score, 1.0),
            social_mentions=social_mentions,
            tags=self._extract_tags(token_data),
            metadata={
                "liquidity": token_data.get("liquidity"),
                "holder_count": token_data.get("holderCount"),
                "verified": token_data.get("verified", False),
                "logo_uri": token_data.get("logoURI"),
                "creation_time": token_data.get("creationTime"),
            }
        )
    
    def _extract_tags(self, token_data: Dict) -> List[str]:
        """Extract relevant tags from token data"""
        tags = []
        
        if token_data.get("verified", False):
            tags.append("verified")
        
        # Add volume-based tags
        volume_24h = token_data.get("volume24h", 0)
        if volume_24h > 1000000:
            tags.append("high-volume")
        elif volume_24h > 100000:
            tags.append("medium-volume")
        
        # Add price movement tags
        price_change = token_data.get("priceChange24h", 0)
        if price_change > 20:
            tags.append("pumping")
        elif price_change > 5:  # Lowered from 10 to 5
            tags.append("rising")
        elif price_change < -20:
            tags.append("dumping")
        elif price_change < -5:  # Lowered from -10 to -5
            tags.append("falling")
        
        return tags
    
    async def get_trending_tokens(self, limit: int = 50) -> List[DiscoveredToken]:
        """Get currently trending tokens across all supported chains"""
        all_trending = []
        
        # Get trending tokens from each supported chain
        for chain in self.get_supported_chains():
            try:
                chain_trending = await self.get_trending_tokens_by_chain(chain, limit // len(self.get_supported_chains()))
                all_trending.extend(chain_trending)
            except Exception as e:
                self.logger.warning("Failed to get trending tokens for chain", chain=chain.value, error=str(e))
        
        # Sort by trending score and limit results
        all_trending.sort(key=lambda x: x.trending_score or 0, reverse=True)
        return all_trending[:limit]
    
    async def get_trending_tokens_by_chain(self, chain: Chain, limit: int = 50) -> List[DiscoveredToken]:
        """Get trending tokens for a specific chain"""
        try:
            chain_param = self._get_chain_param(chain)
            params = {
                "sort_by": "volume24hUSD",
                "sort_type": "desc",
                "offset": 0,
                "limit": limit
            }
            
            # Different endpoints for different chains
            if chain == Chain.SOLANA:
                endpoint = "/defi/trending"
            else:
                endpoint = f"/defi/trending/{chain_param}"
            
            data = await self._make_request(endpoint, params=params)
            
            # Parse response - handle different API response structures
            data_section = data.get("data", {})
            if isinstance(data_section, dict):
                tokens_data = data_section.get("tokens", [])
                if not tokens_data:
                    tokens_data = data_section
            else:
                # data is a list directly
                tokens_data = data_section if isinstance(data_section, list) else []
            
            tokens = []
            for token_data in tokens_data:
                try:
                    token = self._parse_birdeye_token(token_data, chain, "birdeye_trending")
                    self._cache_token(token)
                    tokens.append(token)
                except Exception as e:
                    self.logger.warning("Failed to parse trending token", error=str(e))
            
            self.logger.info("Retrieved trending tokens", chain=chain.value, count=len(tokens))
            return tokens
            
        except Exception as e:
            self.logger.error("Failed to get trending tokens", chain=chain.value, error=str(e))
            return []
    
    async def discover_new_tokens(self, limit: int = 100) -> List[DiscoveredToken]:
        """
        Discover newly created tokens across all chains
        Uses newest token listings sorted by creation time
        """
        all_new = []
        
        # Discover new tokens from each supported chain
        for chain in self.get_supported_chains():
            try:
                chain_new = await self.discover_new_tokens_by_chain(chain, limit // len(self.get_supported_chains()))
                all_new.extend(chain_new)
            except Exception as e:
                self.logger.warning("Failed to discover new tokens for chain", chain=chain.value, error=str(e))
        
        # Sort by discovery time (newest first)
        all_new.sort(key=lambda x: x.discovered_at, reverse=True)
        return all_new[:limit]
    
    async def discover_new_tokens_by_chain(self, chain: Chain, limit: int = 50) -> List[DiscoveredToken]:
        """Discover new tokens for a specific chain"""
        try:
            chain_param = self._get_chain_param(chain)
            params = {
                "sort_by": "creationTime",
                "sort_type": "desc",
                "offset": 0,
                "limit": limit
            }
            
            endpoint = f"/defi/tokenlist/{chain_param}"
            data = await self._make_request(endpoint, params=params)
            
            # Parse response - handle different API response structures
            data_section = data.get("data", {})
            if isinstance(data_section, dict):
                tokens_data = data_section.get("tokens", [])
                if not tokens_data:
                    tokens_data = data_section
            else:
                # data is a list directly
                tokens_data = data_section if isinstance(data_section, list) else []
            
            tokens = []
            for token_data in tokens_data:
                try:
                    token = self._parse_birdeye_token(token_data, chain, "birdeye_new")
                    self._cache_token(token)
                    tokens.append(token)
                except Exception as e:
                    self.logger.warning("Failed to parse new token", error=str(e))
            
            self.logger.info("Discovered new tokens", chain=chain.value, count=len(tokens))
            return tokens
            
        except Exception as e:
            self.logger.error("Failed to discover new tokens", chain=chain.value, error=str(e))
            return []
    
    async def get_token_details(self, address: str, chain: Chain) -> Optional[DiscoveredToken]:
        """Get detailed information for a specific token"""
        cache_key = f"{chain.value}:{address}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self._token_cache[cache_key]
        
        try:
            chain_param = self._get_chain_param(chain)
            endpoint = f"/defi/token_overview"
            params = {
                "address": address,
                "chain": chain_param
            }
            
            data = await self._make_request(endpoint, params=params)
            token_data = data.get("data", {})
            
            if not token_data:
                raise TokenNotFoundError(f"Token {address} not found on {chain.value}")
            
            token = self._parse_birdeye_token(token_data, chain, "birdeye_details")
            self._cache_token(token)
            return token
            
        except TokenNotFoundError:
            raise
        except Exception as e:
            self.logger.error("Failed to get token details", address=address, chain=chain.value, error=str(e))
            return None
    
    def get_supported_chains(self) -> List[Chain]:
        """Get list of supported blockchain networks"""
        return list(self.CHAIN_MAPPING.keys())
    
    async def get_token_security_info(self, address: str, chain: Chain) -> Optional[Dict[str, Any]]:
        """Get security information for a token (if available)"""
        try:
            chain_param = self._get_chain_param(chain)
            endpoint = f"/defi/token_security"
            params = {
                "address": address,
                "chain": chain_param
            }
            
            data = await self._make_request(endpoint, params=params)
            return data.get("data", {})
            
        except Exception as e:
            self.logger.error("Failed to get token security info", address=address, error=str(e))
            return None
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()