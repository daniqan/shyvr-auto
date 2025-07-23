"""
Base classes for token discovery system
Defines interfaces and data structures for multi-chain token discovery
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import structlog

from src.utils.base import Chain


logger = structlog.get_logger()


class TokenStatus(Enum):
    """Token discovery status"""
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    REJECTED = "rejected"
    TRENDING = "trending"
    SUSPICIOUS = "suspicious"


@dataclass
class DiscoveredToken:
    """Represents a newly discovered token"""
    # Basic identification
    address: str
    chain: Chain
    symbol: str
    name: str
    
    # Discovery metadata
    discovered_at: datetime
    discovery_source: str  # jupiter, birdeye, social, manual
    status: TokenStatus = TokenStatus.DISCOVERED
    
    # Market data (when available)
    price_usd: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    
    # Token metadata
    decimals: Optional[int] = None
    total_supply: Optional[float] = None
    circulating_supply: Optional[float] = None
    
    # Discovery context
    trending_score: Optional[float] = None
    social_mentions: Optional[int] = None
    tags: List[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def chain_address(self) -> str:
        """Chain-prefixed address for unique identification"""
        return f"{self.chain.value}:{self.address}"
    
    def is_trending(self) -> bool:
        """Check if token is currently trending"""
        return self.status == TokenStatus.TRENDING or (
            self.trending_score is not None and self.trending_score > 0.7
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "address": self.address,
            "chain": self.chain.value,
            "symbol": self.symbol,
            "name": self.name,
            "discovered_at": self.discovered_at.isoformat(),
            "discovery_source": self.discovery_source,
            "status": self.status.value,
            "price_usd": self.price_usd,
            "market_cap": self.market_cap,
            "volume_24h": self.volume_24h,
            "price_change_24h": self.price_change_24h,
            "decimals": self.decimals,
            "total_supply": self.total_supply,
            "circulating_supply": self.circulating_supply,
            "trending_score": self.trending_score,
            "social_mentions": self.social_mentions,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class TokenDiscoveryBase(ABC):
    """Abstract base class for token discovery clients"""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 60):
        self.api_key = api_key  
        self.rate_limit = rate_limit
        self.logger = structlog.get_logger().bind(client=self.__class__.__name__)
    
    @abstractmethod
    async def discover_new_tokens(self, limit: int = 100) -> List[DiscoveredToken]:
        """Discover newly created tokens"""
        pass
    
    @abstractmethod
    async def get_trending_tokens(self, limit: int = 50) -> List[DiscoveredToken]:
        """Get currently trending tokens"""
        pass
    
    @abstractmethod
    async def get_token_details(self, address: str, chain: Chain) -> Optional[DiscoveredToken]:
        """Get detailed information for a specific token"""
        pass
    
    @abstractmethod
    def get_supported_chains(self) -> List[Chain]:
        """Get list of supported blockchain networks"""
        pass
    
    async def health_check(self) -> bool:
        """Check if the discovery client is healthy"""
        try:
            # Try to get a small number of trending tokens as health check
            tokens = await self.get_trending_tokens(limit=1)
            return len(tokens) >= 0  # Even 0 tokens is OK for health check
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False


@dataclass
class DiscoveryMetrics:
    """Metrics for token discovery performance"""
    tokens_discovered: int = 0
    tokens_validated: int = 0
    tokens_rejected: int = 0
    trending_tokens_found: int = 0
    api_calls_made: int = 0
    errors_encountered: int = 0
    discovery_latency_ms: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of token discovery"""
        total = self.tokens_discovered + self.tokens_rejected
        if total == 0:
            return 0.0
        return self.tokens_discovered / total
    
    @property
    def validation_rate(self) -> float:
        """Calculate validation rate of discovered tokens"""
        if self.tokens_discovered == 0:
            return 0.0
        return self.tokens_validated / self.tokens_discovered


class DiscoveryError(Exception):
    """Base exception for token discovery errors"""
    pass


class APIRateLimitError(DiscoveryError):
    """Raised when API rate limits are exceeded"""
    pass


class TokenNotFoundError(DiscoveryError):
    """Raised when a specific token cannot be found"""
    pass


class ChainNotSupportedError(DiscoveryError):
    """Raised when a blockchain is not supported by the discovery client"""
    pass