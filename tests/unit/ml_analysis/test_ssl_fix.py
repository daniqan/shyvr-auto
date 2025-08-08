"""
Test SSL certificate verification fix for market data API clients

This test module follows TDD methodology to first demonstrate the SSL issue
and then verify the fix works correctly.
"""

import pytest
import ssl
import aiohttp
import os
from unittest.mock import patch, MagicMock, AsyncMock

# Set up minimal config to avoid validation errors
os.environ['SECRET_KEY'] = 'test_secret_key_1234567890'

from src.ml_analysis.market_data import (
    MarketDataClientBase,
    CoinGeckoClient,
    FearGreedIndexClient,
    DeFiLlamaClient,
    SocialSentimentClient,
    OnChainAnalyticsClient
)


class ConcreteMarketDataClient(MarketDataClientBase):
    """Concrete implementation for testing the base class"""
    async def get_market_data(self):
        return {}


class TestSSLConfiguration:
    """Test SSL certificate verification is disabled for all market data API clients"""

    @pytest.mark.asyncio
    async def test_base_client_creates_session_with_ssl_disabled(self):
        """Test that MarketDataClientBase creates aiohttp session with SSL verification disabled"""
        
        # Create client
        client = ConcreteMarketDataClient()
        
        # Get the session
        session = await client._get_session()
        
        # Check that SSL is disabled in the connector
        assert hasattr(session.connector, '_ssl')
        # The SSL context should be False (disabled) or an unverified context
        ssl_context = session.connector._ssl
        assert ssl_context is False or (
            isinstance(ssl_context, ssl.SSLContext) and
            ssl_context.check_hostname is False and
            ssl_context.verify_mode == ssl.CERT_NONE
        )
        
        await client.close()

    @pytest.mark.asyncio
    async def test_coingecko_client_ssl_disabled(self):
        """Test that CoinGecko client has SSL verification disabled"""
        
        client = CoinGeckoClient()
        session = await client._get_session()
        
        # Check SSL configuration
        ssl_context = session.connector._ssl
        assert ssl_context is False or (
            isinstance(ssl_context, ssl.SSLContext) and
            ssl_context.check_hostname is False and
            ssl_context.verify_mode == ssl.CERT_NONE
        )
        
        await client.close()

    @pytest.mark.asyncio
    async def test_fear_greed_client_ssl_disabled(self):
        """Test that FearGreed client has SSL verification disabled"""
        
        client = FearGreedIndexClient()
        session = await client._get_session()
        
        # Check SSL configuration
        ssl_context = session.connector._ssl
        assert ssl_context is False or (
            isinstance(ssl_context, ssl.SSLContext) and
            ssl_context.check_hostname is False and
            ssl_context.verify_mode == ssl.CERT_NONE
        )
        
        await client.close()

    @pytest.mark.asyncio
    async def test_defillama_client_ssl_disabled(self):
        """Test that DeFiLlama client has SSL verification disabled"""
        
        client = DeFiLlamaClient()
        session = await client._get_session()
        
        # Check SSL configuration
        ssl_context = session.connector._ssl
        assert ssl_context is False or (
            isinstance(ssl_context, ssl.SSLContext) and
            ssl_context.check_hostname is False and
            ssl_context.verify_mode == ssl.CERT_NONE
        )
        
        await client.close()

    @pytest.mark.asyncio
    async def test_social_client_ssl_disabled(self):
        """Test that SocialSentiment client has SSL verification disabled"""
        
        # Mock the API key to avoid authentication error
        with patch.dict('os.environ', {'LUNARCRUSH_API_KEY': 'test_key'}):
            client = SocialSentimentClient()
            session = await client._get_session()
            
            # Check SSL configuration
            ssl_context = session.connector._ssl
            assert ssl_context is False or (
                isinstance(ssl_context, ssl.SSLContext) and
                ssl_context.check_hostname is False and
                ssl_context.verify_mode == ssl.CERT_NONE
            )
            
            await client.close()

    @pytest.mark.asyncio
    async def test_onchain_client_ssl_disabled(self):
        """Test that OnChainAnalytics client has SSL verification disabled"""
        
        client = OnChainAnalyticsClient()
        session = await client._get_session()
        
        # Check SSL configuration
        ssl_context = session.connector._ssl
        assert ssl_context is False or (
            isinstance(ssl_context, ssl.SSLContext) and
            ssl_context.check_hostname is False and
            ssl_context.verify_mode == ssl.CERT_NONE
        )
        
        await client.close()

    @pytest.mark.asyncio
    async def test_ssl_context_creation_utility(self):
        """Test that the SSL context creation utility function works correctly"""
        
        # This test will pass once we implement the _create_unverified_ssl_context method
        client = ConcreteMarketDataClient()
        
        # Check if we have the helper method
        if hasattr(client, '_create_unverified_ssl_context'):
            ssl_context = client._create_unverified_ssl_context()
            
            assert isinstance(ssl_context, ssl.SSLContext)
            assert ssl_context.check_hostname is False
            assert ssl_context.verify_mode == ssl.CERT_NONE

    @pytest.mark.asyncio
    async def test_make_request_with_ssl_disabled(self):
        """Test that _make_request uses SSL disabled session"""
        
        client = ConcreteMarketDataClient()
        
        # Mock the session's get method to avoid actual HTTP calls
        with patch.object(aiohttp.ClientSession, 'get') as mock_get:
            # Create a mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'test': 'data'})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_get.return_value = mock_response
            
            # Make the request
            result = await client._make_request('https://test.com')
            
            # Verify the request was made and returned data
            assert result == {'test': 'data'}
            
            # Verify that the session has SSL disabled
            session = await client._get_session()
            ssl_context = session.connector._ssl
            assert ssl_context is False or (
                isinstance(ssl_context, ssl.SSLContext) and
                ssl_context.check_hostname is False and
                ssl_context.verify_mode == ssl.CERT_NONE
            )
        
        await client.close()