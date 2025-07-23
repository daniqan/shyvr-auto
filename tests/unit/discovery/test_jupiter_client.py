"""
Unit tests for Jupiter API client
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from src.utils.base import Chain
from src.discovery.jupiter_client import JupiterTokenDiscovery
from src.discovery.base import (
    DiscoveredToken,
    TokenStatus,
    DiscoveryError,
    APIRateLimitError,
    TokenNotFoundError,
)


class TestJupiterTokenDiscovery:
    """Test Jupiter API client"""

    @pytest.fixture
    def jupiter_client(self):
        """Create Jupiter client for testing"""
        return JupiterTokenDiscovery(rate_limit=100)

    @pytest.fixture
    def mock_token_data(self):
        """Mock token data from Jupiter API"""
        return {
            "address": "So11111111111111111111111111111111111111112",
            "symbol": "SOL",
            "name": "Solana",
            "decimals": 9,
            "tags": ["verified"],
            "daily_volume": 1000000.0,
            "logoURI": "https://example.com/sol.png",
        }

    @pytest.fixture
    def mock_trending_token_data(self):
        """Mock trending token data"""
        return {
            "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "symbol": "USDC",
            "name": "USD Coin",
            "decimals": 6,
            "tags": ["verified", "birdeye-trending"],
            "daily_volume": 5000000.0,
        }

    def test_initialization(self, jupiter_client):
        """Test client initialization"""
        assert jupiter_client.rate_limit == 100
        assert jupiter_client.session is None
        assert jupiter_client._token_cache == {}
        assert jupiter_client._cache_expiry == {}

    def test_supported_chains(self, jupiter_client):
        """Test supported chains"""
        chains = jupiter_client.get_supported_chains()
        assert chains == [Chain.SOLANA]

    def test_parse_jupiter_token(self, jupiter_client, mock_token_data):
        """Test parsing Jupiter token data"""
        token = jupiter_client._parse_jupiter_token(mock_token_data, "test_source")

        assert token.address == "So11111111111111111111111111111111111111112"
        assert token.chain == Chain.SOLANA
        assert token.symbol == "SOL"
        assert token.name == "Solana"
        assert token.decimals == 9
        assert token.discovery_source == "test_source"
        assert token.status == TokenStatus.VALIDATED  # Has "verified" tag
        assert "verified" in token.tags
        assert token.metadata["daily_volume"] == 1000000.0
        assert token.metadata["logo_uri"] == "https://example.com/sol.png"

    def test_parse_trending_token(self, jupiter_client, mock_trending_token_data):
        """Test parsing trending token data"""
        token = jupiter_client._parse_jupiter_token(mock_trending_token_data, "trending")

        assert token.status == TokenStatus.TRENDING  # Has "birdeye-trending" tag
        assert "birdeye-trending" in token.tags
        assert token.trending_score >= 0.5  # Should have high trending score

    def test_cache_functionality(self, jupiter_client, mock_token_data):
        """Test token caching"""
        token = jupiter_client._parse_jupiter_token(mock_token_data, "test")
        jupiter_client._cache_token(token)

        # Check cache
        assert token.address in jupiter_client._token_cache
        assert token.address in jupiter_client._cache_expiry
        assert jupiter_client._is_cache_valid(token.address)

    @pytest.mark.asyncio
    async def test_make_request_success(self, jupiter_client):
        """Test successful API request"""
        mock_response_data = [{"address": "test", "symbol": "TEST"}]

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await jupiter_client._make_request("/test")
            assert result == mock_response_data

    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self, jupiter_client):
        """Test API rate limit handling"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(APIRateLimitError, match="Jupiter API rate limit exceeded"):
                await jupiter_client._make_request("/test")

    @pytest.mark.asyncio
    async def test_make_request_error(self, jupiter_client):
        """Test API error handling"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(DiscoveryError, match="Jupiter API error: 500"):
                await jupiter_client._make_request("/test")

    @pytest.mark.asyncio
    async def test_discover_new_tokens(self, jupiter_client, mock_token_data):
        """Test discovering new tokens"""
        mock_response = [mock_token_data]

        with patch.object(jupiter_client, "_make_request", return_value=mock_response):
            tokens = await jupiter_client.discover_new_tokens(limit=10)

            assert len(tokens) == 1
            assert tokens[0].symbol == "SOL"
            assert tokens[0].discovery_source == "jupiter_community"

    @pytest.mark.asyncio
    async def test_get_trending_tokens(self, jupiter_client, mock_trending_token_data):
        """Test getting trending tokens"""
        mock_response = [mock_trending_token_data]

        with patch.object(jupiter_client, "get_trending_tokens_raw", return_value=mock_response):
            tokens = await jupiter_client.get_trending_tokens(limit=10)

            assert len(tokens) == 1
            assert tokens[0].symbol == "USDC"
            assert tokens[0].status == TokenStatus.TRENDING
            assert tokens[0].discovery_source == "jupiter_trending"

    @pytest.mark.asyncio
    async def test_get_trending_tokens_fallback(self, jupiter_client, mock_token_data):
        """Test trending tokens fallback to verified tokens"""
        mock_verified_response = [mock_token_data]

        with patch.object(jupiter_client, "get_trending_tokens_raw", return_value=[]):
            with patch.object(jupiter_client, "get_verified_tokens", return_value=mock_verified_response):
                tokens = await jupiter_client.get_trending_tokens(limit=10)

                assert len(tokens) == 1
                assert tokens[0].symbol == "SOL"

    @pytest.mark.asyncio
    async def test_get_token_details_found(self, jupiter_client, mock_token_data):
        """Test getting token details for existing token"""
        mock_response = [mock_token_data]

        with patch.object(jupiter_client, "get_all_tokens", return_value=mock_response):
            token = await jupiter_client.get_token_details(
                "So11111111111111111111111111111111111111112", Chain.SOLANA
            )

            assert token is not None
            assert token.symbol == "SOL"
            assert token.discovery_source == "jupiter_details"

    @pytest.mark.asyncio
    async def test_get_token_details_not_found(self, jupiter_client):
        """Test getting token details for non-existent token"""
        with patch.object(jupiter_client, "get_all_tokens", return_value=[]):
            with pytest.raises(TokenNotFoundError):
                await jupiter_client.get_token_details("nonexistent", Chain.SOLANA)

    @pytest.mark.asyncio
    async def test_get_token_details_wrong_chain(self, jupiter_client):
        """Test getting token details for unsupported chain"""
        token = await jupiter_client.get_token_details("0x123", Chain.ETHEREUM)
        assert token is None

    @pytest.mark.asyncio
    async def test_get_token_details_cached(self, jupiter_client, mock_token_data):
        """Test cached token details"""
        # First, cache a token
        token = jupiter_client._parse_jupiter_token(mock_token_data, "test")
        jupiter_client._cache_token(token)

        # Now request it - should return cached version without API call
        with patch.object(jupiter_client, "get_all_tokens") as mock_get_all:
            cached_token = await jupiter_client.get_token_details(
                "So11111111111111111111111111111111111111112", Chain.SOLANA
            )

            # Should not have called the API
            mock_get_all.assert_not_called()
            assert cached_token.symbol == "SOL"

    @pytest.mark.asyncio
    async def test_get_tradeable_tokens(self, jupiter_client, mock_token_data):
        """Test getting all tradeable tokens"""
        mock_response = [mock_token_data]

        with patch.object(jupiter_client, "_make_request", return_value=mock_response):
            tokens = await jupiter_client.get_tradeable_tokens()

            assert len(tokens) == 1
            assert tokens[0].symbol == "SOL"
            assert tokens[0].discovery_source == "jupiter_tradeable"

    @pytest.mark.asyncio
    async def test_error_handling(self, jupiter_client):
        """Test error handling in API methods"""
        with patch.object(jupiter_client, "_make_request", side_effect=Exception("API Error")):
            # Should not raise, should return empty list
            tokens = await jupiter_client.discover_new_tokens()
            assert tokens == []

            trending = await jupiter_client.get_trending_tokens()
            assert trending == []

            tradeable = await jupiter_client.get_tradeable_tokens()
            assert tradeable == []

    @pytest.mark.asyncio
    async def test_session_management(self, jupiter_client):
        """Test HTTP session management"""
        # Session should be None initially
        assert jupiter_client.session is None

        # Getting session should create it
        session = await jupiter_client._get_session()
        assert isinstance(session, aiohttp.ClientSession)
        assert jupiter_client.session is session

        # Getting session again should return same instance
        same_session = await jupiter_client._get_session()
        assert same_session is session

        # Closing should set to None
        await jupiter_client.close()
        assert jupiter_client.session is None

    @pytest.mark.asyncio
    async def test_context_manager(self, jupiter_client):
        """Test using client as context manager"""
        async with jupiter_client as client:
            assert client is jupiter_client
            # Session should be available during context
            session = await client._get_session()
            assert session is not None

        # Session should be closed after context
        assert jupiter_client.session is None