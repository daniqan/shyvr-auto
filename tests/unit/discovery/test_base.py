"""
Unit tests for discovery base classes and data structures
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.utils.base import Chain
from src.discovery.base import (
    DiscoveredToken,
    TokenStatus,
    TokenDiscoveryBase,
    DiscoveryMetrics,
    DiscoveryError,
    APIRateLimitError,
    TokenNotFoundError,
    ChainNotSupportedError,
)


class TestDiscoveredToken:
    """Test DiscoveredToken data class"""

    def test_discovered_token_creation(self):
        """Test basic token creation"""
        token = DiscoveredToken(
            address="So11111111111111111111111111111111111111112",
            chain=Chain.SOLANA,
            symbol="SOL",
            name="Solana",
            discovered_at=datetime.now(),
            discovery_source="test",
        )

        assert token.address == "So11111111111111111111111111111111111111112"
        assert token.chain == Chain.SOLANA
        assert token.symbol == "SOL"
        assert token.name == "Solana"
        assert token.discovery_source == "test"
        assert token.status == TokenStatus.DISCOVERED
        assert token.tags == []
        assert token.metadata == {}

    def test_chain_address_property(self):
        """Test chain-prefixed address generation"""
        token = DiscoveredToken(
            address="0x123",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
        )

        assert token.chain_address == "ethereum:0x123"

    def test_is_trending(self):
        """Test trending detection logic"""
        # Test with trending status
        trending_token = DiscoveredToken(
            address="0x123",
            chain=Chain.ETHEREUM,
            symbol="TREND",
            name="Trending Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            status=TokenStatus.TRENDING,
        )
        assert trending_token.is_trending()

        # Test with high trending score
        high_score_token = DiscoveredToken(
            address="0x456",
            chain=Chain.ETHEREUM,
            symbol="HIGH",
            name="High Score Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            trending_score=0.8,
        )
        assert high_score_token.is_trending()

        # Test with low trending score
        low_score_token = DiscoveredToken(
            address="0x789",
            chain=Chain.ETHEREUM,
            symbol="LOW",
            name="Low Score Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            trending_score=0.5,
        )
        assert not low_score_token.is_trending()

    def test_to_dict(self):
        """Test dictionary serialization"""
        token = DiscoveredToken(
            address="0x123",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime(2025, 1, 1, 12, 0, 0),
            discovery_source="test",
            price_usd=1.50,
            market_cap=1000000.0,
            tags=["verified", "trending"],
            metadata={"test": "data"},
        )

        token_dict = token.to_dict()

        assert token_dict["address"] == "0x123"
        assert token_dict["chain"] == "ethereum"
        assert token_dict["symbol"] == "TEST"
        assert token_dict["name"] == "Test Token"
        assert token_dict["discovered_at"] == "2025-01-01T12:00:00"
        assert token_dict["discovery_source"] == "test"
        assert token_dict["price_usd"] == 1.50
        assert token_dict["market_cap"] == 1000000.0
        assert token_dict["tags"] == ["verified", "trending"]
        assert token_dict["metadata"] == {"test": "data"}


class TestDiscoveryMetrics:
    """Test DiscoveryMetrics data class"""

    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        metrics = DiscoveryMetrics(tokens_discovered=80, tokens_rejected=20)
        assert metrics.success_rate == 0.8

        # Test edge case with no tokens
        empty_metrics = DiscoveryMetrics()
        assert empty_metrics.success_rate == 0.0

    def test_validation_rate_calculation(self):
        """Test validation rate calculation"""
        metrics = DiscoveryMetrics(tokens_discovered=100, tokens_validated=75)
        assert metrics.validation_rate == 0.75

        # Test edge case with no discovered tokens
        empty_metrics = DiscoveryMetrics()
        assert empty_metrics.validation_rate == 0.0


class MockTokenDiscovery(TokenDiscoveryBase):
    """Mock implementation for testing base class"""

    def __init__(self, should_fail=False):
        super().__init__()
        self.should_fail = should_fail

    async def discover_new_tokens(self, limit=100):
        if self.should_fail:
            raise DiscoveryError("Mock discovery failed")
        return [
            DiscoveredToken(
                address="0x123",
                chain=Chain.ETHEREUM,
                symbol="MOCK",
                name="Mock Token",
                discovered_at=datetime.now(),
                discovery_source="mock",
            )
        ]

    async def get_trending_tokens(self, limit=50):
        if self.should_fail:
            raise DiscoveryError("Mock trending failed")
        return []

    async def get_token_details(self, address, chain):
        if address == "not_found":
            return None
        return DiscoveredToken(
            address=address,
            chain=chain,
            symbol="DETAIL",
            name="Detail Token",
            discovered_at=datetime.now(),
            discovery_source="mock",
        )

    def get_supported_chains(self):
        return [Chain.ETHEREUM, Chain.SOLANA]


class TestTokenDiscoveryBase:
    """Test TokenDiscoveryBase abstract class"""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check"""
        discovery = MockTokenDiscovery()
        is_healthy = await discovery.health_check()
        assert is_healthy

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check"""
        discovery = MockTokenDiscovery(should_fail=True)
        is_healthy = await discovery.health_check()
        assert not is_healthy


class TestDiscoveryExceptions:
    """Test discovery-related exceptions"""

    def test_discovery_error(self):
        """Test base DiscoveryError"""
        error = DiscoveryError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_api_rate_limit_error(self):
        """Test APIRateLimitError inheritance"""
        error = APIRateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, DiscoveryError)
        assert isinstance(error, Exception)

    def test_token_not_found_error(self):
        """Test TokenNotFoundError inheritance"""
        error = TokenNotFoundError("Token not found")
        assert str(error) == "Token not found"
        assert isinstance(error, DiscoveryError)

    def test_chain_not_supported_error(self):
        """Test ChainNotSupportedError inheritance"""
        error = ChainNotSupportedError("Chain not supported")
        assert str(error) == "Chain not supported"
        assert isinstance(error, DiscoveryError)