"""
Jupiter API Client for Solana Token Discovery
Provides fast token discovery and metadata via Jupiter's Token API V2
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import structlog

from src.utils.base import Chain
from .base import (
    TokenDiscoveryBase, 
    DiscoveredToken, 
    TokenStatus,
    DiscoveryError,
    APIRateLimitError,
    TokenNotFoundError
)


logger = structlog.get_logger()


class JupiterTokenDiscovery(TokenDiscoveryBase):
    """Jupiter API client for Solana token discovery and metadata"""
    
    BASE_URL = "https://tokens.jup.ag"
    
    def __init__(self, rate_limit: int = 100):
        super().__init__(rate_limit=rate_limit)
        self.session: Optional[aiohttp.ClientSession] = None
        self._token_cache: Dict[str, DiscoveredToken] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=15)  # Cache for 15 minutes
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make HTTP request to Jupiter API"""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    raise APIRateLimitError("Jupiter API rate limit exceeded")
                elif response.status != 200:
                    raise DiscoveryError(f"Jupiter API error: {response.status}")
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise DiscoveryError(f"Jupiter API request failed: {str(e)}")
    
    def _is_cache_valid(self, token_address: str) -> bool:
        """Check if cached token data is still valid"""
        if token_address not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[token_address]
    
    def _cache_token(self, token: DiscoveredToken):
        """Cache token data"""
        self._token_cache[token.address] = token
        self._cache_expiry[token.address] = datetime.now() + self._cache_duration
    
    async def get_all_tokens(self) -> List[Dict]:
        """Get all tokens available on Jupiter"""
        try:
            data = await self._make_request("/tokens")
            return data if isinstance(data, list) else []
        except Exception as e:
            self.logger.error("Failed to get all tokens", error=str(e))
            return []
    
    async def get_verified_tokens(self) -> List[Dict]:
        """Get only verified tokens"""
        try:
            data = await self._make_request("/tokens", params={"tags": "verified"})
            return data if isinstance(data, list) else []
        except Exception as e:
            self.logger.error("Failed to get verified tokens", error=str(e))
            return []
    
    async def get_trending_tokens_raw(self) -> List[Dict]:
        """Get tokens tagged as trending by Birdeye"""
        try:
            data = await self._make_request("/tokens", params={"tags": "birdeye-trending"})
            return data if isinstance(data, list) else []
        except Exception as e:
            self.logger.error("Failed to get trending tokens", error=str(e))
            return []
    
    def _parse_jupiter_token(self, token_data: Dict, discovery_source: str = "jupiter") -> DiscoveredToken:
        """Parse Jupiter API token data into DiscoveredToken"""
        # Extract tags
        tags = token_data.get("tags", [])
        
        # Determine status based on tags
        status = TokenStatus.DISCOVERED
        if "birdeye-trending" in tags:
            status = TokenStatus.TRENDING
        elif "verified" in tags:
            status = TokenStatus.VALIDATED
        
        # Calculate trending score based on tags and daily volume
        trending_score = 0.0
        if "birdeye-trending" in tags:
            trending_score += 0.5
        if "community" in tags:
            trending_score += 0.2
        if token_data.get("daily_volume", 0) > 100000:  # $100k+ daily volume
            trending_score += 0.3
        
        return DiscoveredToken(
            address=token_data["address"],
            chain=Chain.SOLANA,
            symbol=token_data.get("symbol", "UNKNOWN"),
            name=token_data.get("name", "Unknown Token"),
            discovered_at=datetime.now(),
            discovery_source=discovery_source,
            status=status,
            decimals=token_data.get("decimals"),
            tags=tags,
            trending_score=min(trending_score, 1.0),  # Cap at 1.0
            metadata={
                "daily_volume": token_data.get("daily_volume"),
                "freeze_authority": token_data.get("freeze_authority"),
                "mint_authority": token_data.get("mint_authority"),
                "logo_uri": token_data.get("logoURI"),
            }
        )
    
    async def discover_new_tokens(self, limit: int = 100) -> List[DiscoveredToken]:
        """
        Discover newly created tokens on Solana
        Jupiter doesn't have a direct 'new tokens' endpoint, so we get community tokens
        """
        try:
            # Get community tokens as they're likely to be newer
            data = await self._make_request("/tokens", params={"tags": "community"})
            if not isinstance(data, list):
                return []
            
            # Parse and limit results
            tokens = []
            for token_data in data[:limit]:
                try:
                    token = self._parse_jupiter_token(token_data, "jupiter_community")
                    self._cache_token(token)
                    tokens.append(token)
                except Exception as e:
                    self.logger.warning("Failed to parse token", token=token_data.get("address"), error=str(e))
            
            self.logger.info("Discovered new tokens", count=len(tokens), source="jupiter")
            return tokens
            
        except Exception as e:
            self.logger.error("Failed to discover new tokens", error=str(e))
            return []
    
    async def get_trending_tokens(self, limit: int = 50) -> List[DiscoveredToken]:
        """Get currently trending tokens from Jupiter/Birdeye integration"""
        try:
            # Get trending tokens
            trending_data = await self.get_trending_tokens_raw()
            
            # If no trending tokens, fall back to verified tokens with high volume
            if not trending_data:
                self.logger.info("No trending tokens found, falling back to verified tokens")
                verified_data = await self.get_verified_tokens()
                # Sort by daily volume if available
                trending_data = sorted(
                    verified_data,
                    key=lambda x: x.get("daily_volume", 0),
                    reverse=True
                )[:limit]
            
            # Parse tokens
            tokens = []
            for token_data in trending_data[:limit]:
                try:
                    token = self._parse_jupiter_token(token_data, "jupiter_trending")
                    token.status = TokenStatus.TRENDING
                    self._cache_token(token)
                    tokens.append(token)
                except Exception as e:
                    self.logger.warning("Failed to parse trending token", token=token_data.get("address"), error=str(e))
            
            self.logger.info("Retrieved trending tokens", count=len(tokens), source="jupiter")
            return tokens
            
        except Exception as e:
            self.logger.error("Failed to get trending tokens", error=str(e))
            return []
    
    async def get_token_details(self, address: str, chain: Chain) -> Optional[DiscoveredToken]:
        """Get detailed information for a specific token"""
        if chain != Chain.SOLANA:
            return None
        
        # Check cache first
        if self._is_cache_valid(address):
            return self._token_cache[address]
        
        try:
            # Jupiter doesn't have a single token endpoint, so we search in all tokens
            all_tokens = await self.get_all_tokens()
            
            for token_data in all_tokens:
                if token_data.get("address") == address:
                    token = self._parse_jupiter_token(token_data, "jupiter_details")
                    self._cache_token(token)
                    return token
            
            raise TokenNotFoundError(f"Token {address} not found on Jupiter")
            
        except TokenNotFoundError:
            raise
        except Exception as e:
            self.logger.error("Failed to get token details", address=address, error=str(e))
            return None
    
    def get_supported_chains(self) -> List[Chain]:
        """Jupiter only supports Solana"""
        return [Chain.SOLANA]
    
    async def get_tradeable_tokens(self) -> List[DiscoveredToken]:
        """Get all tokens that are tradeable on Jupiter"""
        try:
            data = await self._make_request("/tokens")
            if not isinstance(data, list):
                return []
            
            tokens = []
            for token_data in data:
                try:
                    token = self._parse_jupiter_token(token_data, "jupiter_tradeable")
                    self._cache_token(token)
                    tokens.append(token)
                except Exception as e:
                    self.logger.warning("Failed to parse tradeable token", error=str(e))
            
            self.logger.info("Retrieved tradeable tokens", count=len(tokens))
            return tokens
            
        except Exception as e:
            self.logger.error("Failed to get tradeable tokens", error=str(e))
            return []
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()