"""
Unit tests for Birdeye API client
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from src.utils.base import Chain
from src.discovery.birdeye_client import BirdeyeTokenDiscovery
from src.discovery.base import (
    DiscoveredToken,
    TokenStatus,
    DiscoveryError,
    APIRateLimitError,
    TokenNotFoundError,
    ChainNotSupportedError,
)


class TestBirdeyeTokenDiscovery:
    """Test Birdeye API client"""

    @pytest.fixture
    def birdeye_client(self):
        """Create Birdeye client for testing"""
        return BirdeyeTokenDiscovery(api_key="test_api_key", rate_limit=60)

    @pytest.fixture
    def mock_token_data(self):
        """Mock token data from Birdeye API"""
        return {
            "address": "0x123456789abcdef",
            "symbol": "TEST",
            "name": "Test Token",
            "decimals": 18,
            "price": 1.50,
            "marketCap": 1500000.0,
            "volume24h": 50000.0,
            "priceChange24h": 5.5,
            "supply": 1000000.0,
            "liquidity": 100000.0,
            "holderCount": 250,
            "verified": True,
            "logoURI": "https://example.com/test.png",
            "creationTime": 1640995200,  # Unix timestamp
        }

    @pytest.fixture
    def mock_trending_data(self):
        """Mock trending token with high metrics"""
        return {
            "address": "0xtrending123",
            "symbol": "TREND",
            "name": "Trending Token", 
            "decimals": 18,
            "price": 10.0,
            "marketCap": 10000000.0,
            "volume24h": 2000000.0,  # High volume
            "priceChange24h": 25.0,   # High price increase
            "verified": False,
            "socialScore": 85,
        }

    def test_initialization(self, birdeye_client):
        """Test client initialization"""
        assert birdeye_client.api_key == "test_api_key"
        assert birdeye_client.rate_limit == 60
        assert birdeye_client.session is None
        assert birdeye_client._token_cache == {}

    def test_supported_chains(self, birdeye_client):
        """Test supported chains"""
        chains = birdeye_client.get_supported_chains()
        expected_chains = [Chain.SOLANA, Chain.ETHEREUM, Chain.BASE, Chain.POLYGON, Chain.BSC, Chain.ARBITRUM]
        assert all(chain in chains for chain in expected_chains)

    def test_get_chain_param(self, birdeye_client):
        """Test chain parameter mapping"""
        assert birdeye_client._get_chain_param(Chain.SOLANA) == "solana"
        assert birdeye_client._get_chain_param(Chain.ETHEREUM) == "ethereum"
        assert birdeye_client._get_chain_param(Chain.BASE) == "base"
        assert birdeye_client._get_chain_param(Chain.POLYGON) == "polygon"
        assert birdeye_client._get_chain_param(Chain.BSC) == "bsc"
        assert birdeye_client._get_chain_param(Chain.ARBITRUM) == "arbitrum"

        # Test unsupported chain
        with pytest.raises(ChainNotSupportedError):
            birdeye_client._get_chain_param(Chain.AVALANCHE)

    def test_parse_birdeye_token(self, birdeye_client, mock_token_data):
        """Test parsing Birdeye token data"""
        token = birdeye_client._parse_birdeye_token(mock_token_data, Chain.ETHEREUM, "test_source")

        assert token.address == "0x123456789abcdef"
        assert token.chain == Chain.ETHEREUM
        assert token.symbol == "TEST"
        assert token.name == "Test Token"
        assert token.decimals == 18
        assert token.price_usd == 1.50
        assert token.market_cap == 1500000.0
        assert token.volume_24h == 50000.0
        assert token.price_change_24h == 5.5
        assert token.total_supply == 1000000.0
        assert token.discovery_source == "test_source"
        assert token.status == TokenStatus.VALIDATED  # verified: true
        assert token.metadata["verified"] is True
        assert token.metadata["liquidity"] == 100000.0
        assert token.metadata["holder_count"] == 250

    def test_parse_trending_token(self, birdeye_client, mock_trending_data):
        """Test parsing trending token with high trending score"""
        token = birdeye_client._parse_birdeye_token(mock_trending_data, Chain.ETHEREUM, "birdeye_trending")

        # Should have high trending score due to high volume and price change
        assert token.trending_score > 0.6
        assert token.status == TokenStatus.TRENDING
        assert token.social_mentions == 85

    def test_extract_tags(self, birdeye_client, mock_token_data):
        """Test tag extraction from token data"""
        tags = birdeye_client._extract_tags(mock_token_data)

        assert "verified" in tags
        assert "rising" in tags  # 5.5% price change
        # Volume is 50k, should be in medium-volume range

    def test_extract_tags_pumping(self, birdeye_client):
        """Test tag extraction for pumping token"""
        pumping_data = {
            "verified": False,
            "volume24h": 2000000.0,  # High volume
            "priceChange24h": 25.0,  # Pumping (>20%)
        }
        tags = birdeye_client._extract_tags(pumping_data)

        assert "high-volume" in tags
        assert "pumping" in tags

    def test_extract_tags_dumping(self, birdeye_client):
        """Test tag extraction for dumping token"""
        dumping_data = {
            "verified": True,
            "volume24h": 10000.0,   # Low volume
            "priceChange24h": -25.0,  # Dumping (<-20%)
        }
        tags = birdeye_client._extract_tags(dumping_data)

        assert "verified" in tags
        assert "dumping" in tags

    @pytest.mark.asyncio
    async def test_make_request_success(self, birdeye_client):
        """Test successful API request with authentication"""
        mock_response_data = {"data": {"tokens": []}}

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await birdeye_client._make_request("/test")
            assert result == mock_response_data

    @pytest.mark.asyncio
    async def test_make_request_auth_error(self, birdeye_client):
        """Test authentication error handling"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(DiscoveryError, match="authentication failed"):
                await birdeye_client._make_request("/test")

    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self, birdeye_client):
        """Test rate limit error handling"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(APIRateLimitError, match="Birdeye API rate limit exceeded"):
                await birdeye_client._make_request("/test")

    @pytest.mark.asyncio
    async def test_get_trending_tokens_by_chain(self, birdeye_client, mock_trending_data):
        """Test getting trending tokens for specific chain"""
        mock_response = {"data": {"tokens": [mock_trending_data]}}

        with patch.object(birdeye_client, "_make_request", return_value=mock_response):
            tokens = await birdeye_client.get_trending_tokens_by_chain(Chain.ETHEREUM, limit=10)

            assert len(tokens) == 1
            assert tokens[0].symbol == "TREND"
            assert tokens[0].discovery_source == "birdeye_trending"

    @pytest.mark.asyncio
    async def test_get_trending_tokens_all_chains(self, birdeye_client, mock_trending_data):
        """Test getting trending tokens across all chains"""
        with patch.object(birdeye_client, "get_trending_tokens_by_chain") as mock_get_by_chain:
            mock_get_by_chain.return_value = [
                birdeye_client._parse_birdeye_token(mock_trending_data, Chain.ETHEREUM, "birdeye_trending")
            ]

            tokens = await birdeye_client.get_trending_tokens(limit=50)

            # Should call get_trending_tokens_by_chain for each supported chain
            assert mock_get_by_chain.call_count == len(birdeye_client.get_supported_chains())
            assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_discover_new_tokens_by_chain(self, birdeye_client, mock_token_data):
        """Test discovering new tokens for specific chain"""
        mock_response = {"data": {"tokens": [mock_token_data]}}

        with patch.object(birdeye_client, "_make_request", return_value=mock_response):
            tokens = await birdeye_client.discover_new_tokens_by_chain(Chain.ETHEREUM, limit=10)

            assert len(tokens) == 1
            assert tokens[0].symbol == "TEST"
            assert tokens[0].discovery_source == "birdeye_new"

    @pytest.mark.asyncio
    async def test_discover_new_tokens_all_chains(self, birdeye_client, mock_token_data):
        """Test discovering new tokens across all chains"""
        with patch.object(birdeye_client, "discover_new_tokens_by_chain") as mock_discover_by_chain:
            mock_discover_by_chain.return_value = [
                birdeye_client._parse_birdeye_token(mock_token_data, Chain.ETHEREUM, "birdeye_new")
            ]

            tokens = await birdeye_client.discover_new_tokens(limit=100)

            # Should call discover_new_tokens_by_chain for each supported chain
            assert mock_discover_by_chain.call_count == len(birdeye_client.get_supported_chains())
            assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_get_token_details_found(self, birdeye_client, mock_token_data):
        """Test getting token details for existing token"""
        mock_response = {"data": mock_token_data}

        with patch.object(birdeye_client, "_make_request", return_value=mock_response):
            token = await birdeye_client.get_token_details("0x123456789abcdef", Chain.ETHEREUM)

            assert token is not None
            assert token.symbol == "TEST"
            assert token.discovery_source == "birdeye_details"

    @pytest.mark.asyncio
    async def test_get_token_details_not_found(self, birdeye_client):
        """Test getting token details for non-existent token"""
        mock_response = {"data": {}}

        with patch.object(birdeye_client, "_make_request", return_value=mock_response):
            with pytest.raises(TokenNotFoundError):
                await birdeye_client.get_token_details("0xnonexistent", Chain.ETHEREUM)

    @pytest.mark.asyncio
    async def test_get_token_details_cached(self, birdeye_client, mock_token_data):
        """Test cached token details"""
        # Cache a token
        token = birdeye_client._parse_birdeye_token(mock_token_data, Chain.ETHEREUM, "test")
        birdeye_client._cache_token(token)

        # Request should return cached version
        with patch.object(birdeye_client, "_make_request") as mock_request:
            cached_token = await birdeye_client.get_token_details("0x123456789abcdef", Chain.ETHEREUM)

            # Should not make API request
            mock_request.assert_not_called()
            assert cached_token.symbol == "TEST"

    @pytest.mark.asyncio
    async def test_get_token_security_info(self, birdeye_client):
        """Test getting token security information"""
        mock_response = {"data": {"is_honeypot": False, "rugpull_risk": "low"}}

        with patch.object(birdeye_client, "_make_request", return_value=mock_response):
            security_info = await birdeye_client.get_token_security_info("0x123", Chain.ETHEREUM)

            assert security_info == {"is_honeypot": False, "rugpull_risk": "low"}

    @pytest.mark.asyncio
    async def test_error_handling_chain_specific(self, birdeye_client):
        """Test error handling in chain-specific methods"""
        with patch.object(birdeye_client, "_make_request", side_effect=Exception("API Error")):
            # Should not raise, should return empty list
            tokens = await birdeye_client.get_trending_tokens_by_chain(Chain.ETHEREUM)
            assert tokens == []

            new_tokens = await birdeye_client.discover_new_tokens_by_chain(Chain.ETHEREUM)
            assert new_tokens == []

    @pytest.mark.asyncio
    async def test_session_with_api_key(self, birdeye_client):
        """Test HTTP session includes API key in headers"""
        session = await birdeye_client._get_session()
        
        assert "X-API-KEY" in session._default_headers
        assert session._default_headers["X-API-KEY"] == "test_api_key"
        assert session._default_headers["Content-Type"] == "application/json"

        await birdeye_client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, birdeye_client):
        """Test using client as context manager"""
        async with birdeye_client as client:
            assert client is birdeye_client
            # Should be able to make requests
            session = await client._get_session()
            assert session is not None

        # Session should be closed after context
        assert birdeye_client.session is None

    def test_cache_functionality(self, birdeye_client, mock_token_data):
        """Test token caching with chain-specific keys"""
        token = birdeye_client._parse_birdeye_token(mock_token_data, Chain.ETHEREUM, "test")
        birdeye_client._cache_token(token)

        cache_key = f"{Chain.ETHEREUM.value}:{token.address}"
        assert cache_key in birdeye_client._token_cache
        assert cache_key in birdeye_client._cache_expiry
        assert birdeye_client._is_cache_valid(cache_key)

    @pytest.mark.asyncio
    async def test_response_structure_variations(self, birdeye_client, mock_token_data):
        """Test handling different API response structures"""
        # Test with nested data structure
        nested_response = {"data": {"tokens": [mock_token_data]}}
        
        with patch.object(birdeye_client, "_make_request", return_value=nested_response):
            tokens = await birdeye_client.get_trending_tokens_by_chain(Chain.ETHEREUM)
            assert len(tokens) == 1

        # Test with direct data structure  
        direct_response = {"data": [mock_token_data]}
        
        with patch.object(birdeye_client, "_make_request", return_value=direct_response):
            tokens = await birdeye_client.get_trending_tokens_by_chain(Chain.ETHEREUM)
            assert len(tokens) == 1