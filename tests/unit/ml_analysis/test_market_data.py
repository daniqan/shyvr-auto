"""
Unit tests for market data API clients
"""

import pytest
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from src.ml_analysis.market_data import (
    FearGreedIndexClient,
    DeFiLlamaClient,
    CoinGeckoClient,
    OnChainAnalyticsClient,
    SocialSentimentClient,
    MarketSentimentData,
    DeFiMetrics,
    CorrelationMetrics,
    OnChainMetrics,
    SocialSentimentData,
    MarketDataError,
    APIRateLimitError,
    APIAuthenticationError,
    DataNotAvailableError
)
from src.utils.base import Chain


# Global fixtures for all test classes
@pytest.fixture
def fear_greed_client():
    """Create Fear & Greed Index client for testing"""
    return FearGreedIndexClient(cache_ttl=60)

@pytest.fixture
def defi_client():
    """Create DeFiLlama client for testing"""
    return DeFiLlamaClient(cache_ttl=60)

@pytest.fixture
def coingecko_client():
    """Create CoinGecko client for testing"""
    return CoinGeckoClient(api_key="test_key", cache_ttl=60)

@pytest.fixture
def onchain_client():
    """Create on-chain analytics client for testing"""
    return OnChainAnalyticsClient(cache_ttl=60)

@pytest.fixture
def social_client():
    """Create social sentiment client for testing"""
    return SocialSentimentClient(cache_ttl=60)


@pytest.fixture
def sample_fear_greed_response():
        """Sample Fear & Greed Index API response"""
        return {
            "name": "Fear and Greed Index",
            "data": [
                {
                    "value": "75",
                    "value_classification": "Greed",
                    "timestamp": "1640995200",
                    "time_until_update": "72000"
                }
            ],
            "metadata": {
                "error": None
            }
        }
    
@pytest.fixture
def sample_defi_tvl_response():
    """Sample DeFiLlama TVL API response"""
    return {
        "totalTvl": 50000000000.0,  # $50B
        "totalTvl24hAgo": 49000000000.0,
        "totalTvl7dAgo": 48000000000.0
    }

@pytest.fixture
def sample_defi_chains_response():
    """Sample DeFiLlama chains API response"""
    return [
        {"name": "Ethereum", "tvl": 25000000000.0},
        {"name": "BSC", "tvl": 5000000000.0},
        {"name": "Polygon", "tvl": 3000000000.0}
    ]

@pytest.fixture
def sample_defi_protocols_response():
    """Sample DeFiLlama protocols API response"""
    return [
        {"name": "Uniswap", "tvl": 5000000000.0},
        {"name": "Aave", "tvl": 3000000000.0},
        {"name": "Compound", "tvl": 2000000000.0},
        {"name": "Small Protocol", "tvl": 500000.0}  # Below $1M threshold
    ]

@pytest.fixture
def sample_coingecko_global_response():
    """Sample CoinGecko global market data response"""
    return {
        "data": {
            "active_cryptocurrencies": 10000,
            "market_cap_percentage": {
                "btc": 42.5,
                "eth": 18.2,
                "usdt": 5.1,
                "usdc": 2.8,
                "bnb": 3.2
            },
            "total_market_cap": {
                "usd": 1800000000000.0
            }
        }
    }

@pytest.fixture
def sample_coingecko_coins_response():
    """Sample CoinGecko coins market data response"""
    return [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": 45000.0,
            "market_cap": 850000000000.0,
            "price_change_percentage_24h": 2.5
        },
        {
            "id": "ethereum", 
            "symbol": "eth",
            "name": "Ethereum",
            "current_price": 3200.0,
            "market_cap": 380000000000.0,
            "price_change_percentage_24h": 3.1
        }
    ]


class TestMarketDataClients:
    """Test market data API clients"""


class TestFearGreedIndexClient:
    """Test Fear & Greed Index client"""
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, fear_greed_client, sample_fear_greed_response):
        """Test successful Fear & Greed Index retrieval"""
        with patch.object(fear_greed_client, '_make_request', return_value=sample_fear_greed_response):
            result = await fear_greed_client.get_market_data()
            
            assert isinstance(result, MarketSentimentData)
            assert result.fear_greed_index == 75.0
            assert result.fear_greed_classification == "Greed"
            assert result.market_trend == "bull"
            assert result.volatility_regime == "medium"
    
    @pytest.mark.asyncio
    async def test_get_market_data_extreme_fear(self, fear_greed_client):
        """Test Fear & Greed Index with extreme fear value"""
        extreme_fear_response = {
            "data": [{"value": "15", "timestamp": "1640995200"}]
        }
        
        with patch.object(fear_greed_client, '_make_request', return_value=extreme_fear_response):
            result = await fear_greed_client.get_market_data()
            
            assert result.fear_greed_index == 15.0
            assert result.fear_greed_classification == "Extreme Fear"
            assert result.market_trend == "bear"
            assert result.volatility_regime == "high"
    
    @pytest.mark.asyncio
    async def test_get_market_data_extreme_greed(self, fear_greed_client):
        """Test Fear & Greed Index with extreme greed value"""
        extreme_greed_response = {
            "data": [{"value": "85", "timestamp": "1640995200"}]
        }
        
        with patch.object(fear_greed_client, '_make_request', return_value=extreme_greed_response):
            result = await fear_greed_client.get_market_data()
            
            assert result.fear_greed_index == 85.0
            assert result.fear_greed_classification == "Extreme Greed"
            assert result.market_trend == "bull"
            assert result.volatility_regime == "high"
    
    @pytest.mark.asyncio
    async def test_get_market_data_no_data(self, fear_greed_client):
        """Test Fear & Greed Index with no data available"""
        empty_response = {"data": []}
        
        with patch.object(fear_greed_client, '_make_request', return_value=empty_response):
            with pytest.raises(DataNotAvailableError):
                await fear_greed_client.get_market_data()
    
    @pytest.mark.asyncio
    async def test_get_market_data_caching(self, fear_greed_client, sample_fear_greed_response):
        """Test caching behavior"""
        with patch.object(fear_greed_client, '_make_request', return_value=sample_fear_greed_response) as mock_request:
            # First call
            result1 = await fear_greed_client.get_market_data()
            # Second call should use cache
            result2 = await fear_greed_client.get_market_data()
            
            # Should only make one API request
            assert mock_request.call_count == 1
            assert result1 is result2  # Same cached object


class TestDeFiLlamaClient:
    """Test DeFiLlama client"""
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, defi_client, sample_defi_tvl_response, 
                                          sample_defi_chains_response, sample_defi_protocols_response):
        """Test successful DeFi metrics retrieval"""
        with patch.object(defi_client, '_make_request') as mock_request:
            # Mock different endpoints
            mock_request.side_effect = [
                sample_defi_tvl_response,
                sample_defi_chains_response,
                sample_defi_protocols_response
            ]
            
            result = await defi_client.get_market_data()
            
            assert isinstance(result, DeFiMetrics)
            assert result.total_value_locked == 50000000000.0
            assert result.tvl_change_24h == pytest.approx(2.04, rel=1e-2)  # ~2% increase
            assert result.tvl_change_7d == pytest.approx(4.17, rel=1e-2)   # ~4% increase
            assert result.protocols_count == 3  # Only protocols > $1M TVL
            assert "Ethereum" in result.chains_tvl
            assert result.chains_tvl["Ethereum"] == 25000000000.0
    
    @pytest.mark.asyncio
    async def test_get_market_data_api_error(self, defi_client):
        """Test DeFi client with API error"""
        with patch.object(defi_client, '_make_request', side_effect=MarketDataError("API error")):
            with pytest.raises(MarketDataError):
                await defi_client.get_market_data()
    
    @pytest.mark.asyncio
    async def test_get_market_data_empty_response(self, defi_client):
        """Test DeFi client with empty response"""
        with patch.object(defi_client, '_make_request', return_value={}):
            result = await defi_client.get_market_data()
            
            assert isinstance(result, DeFiMetrics)
            assert result.total_value_locked == 0.0
            assert result.tvl_change_24h == 0.0


class TestCoinGeckoClient:
    """Test CoinGecko client"""
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, coingecko_client, sample_coingecko_global_response, 
                                          sample_coingecko_coins_response):
        """Test successful correlation metrics retrieval"""
        with patch.object(coingecko_client, '_make_request') as mock_request:
            mock_request.side_effect = [
                sample_coingecko_global_response,
                sample_coingecko_coins_response
            ]
            
            result = await coingecko_client.get_market_data()
            
            assert isinstance(result, CorrelationMetrics)
            assert result.btc_dominance == 42.5
            assert result.eth_dominance == 18.2
            assert result.stablecoin_dominance == pytest.approx(7.9, rel=1e-2)  # USDT + USDC
            assert "BTC" in result.correlation_matrix
            assert "ETH" in result.correlation_matrix
    
    @pytest.mark.asyncio
    async def test_get_market_data_with_api_key(self, coingecko_client):
        """Test that API key is included in headers"""
        with patch.object(coingecko_client, '_make_request') as mock_request:
            mock_request.return_value = {"data": {"market_cap_percentage": {}}}
            
            await coingecko_client.get_market_data()
            
            # Check that headers were passed
            call_args = mock_request.call_args
            assert call_args[1]["headers"]["X-CG-Pro-API-Key"] == "test_key"
    
    @pytest.mark.asyncio
    async def test_get_market_data_no_data(self, coingecko_client):
        """Test CoinGecko client with missing data"""
        with patch.object(coingecko_client, '_make_request', return_value={}):
            result = await coingecko_client.get_market_data()
            
            assert isinstance(result, CorrelationMetrics)
            assert result.btc_dominance == 0.0
            assert result.eth_dominance == 0.0


class TestOnChainAnalyticsClient:
    """Test on-chain analytics client"""
    
    @pytest.mark.asyncio
    async def test_get_market_data_ethereum(self, onchain_client):
        """Test on-chain metrics for Ethereum"""
        result = await onchain_client.get_market_data(Chain.ETHEREUM)
        
        assert isinstance(result, OnChainMetrics)
        assert result.network_activity["chain"] == "ethereum"
        assert result.transaction_count_24h > 0
        assert result.active_addresses_24h > 0
        assert result.staking_ratio is not None  # Ethereum has staking
        assert result.hash_rate is None  # Ethereum is PoS, no hash rate
    
    @pytest.mark.asyncio
    async def test_get_market_data_bitcoin(self, onchain_client):
        """Test on-chain metrics for Bitcoin"""
        result = await onchain_client.get_market_data(Chain.BITCOIN)
        
        assert isinstance(result, OnChainMetrics)
        assert result.network_activity["chain"] == "bitcoin"
        assert result.hash_rate is not None  # Bitcoin has hash rate
        assert result.staking_ratio is None  # Bitcoin is PoW, no staking
    
    @pytest.mark.asyncio
    async def test_get_market_data_caching(self, onchain_client):
        """Test on-chain data caching"""
        # First call
        result1 = await onchain_client.get_market_data(Chain.ETHEREUM)
        # Second call should use cache
        result2 = await onchain_client.get_market_data(Chain.ETHEREUM)
        
        assert result1 is result2  # Same cached object


class TestSocialSentimentClient:
    """Test social sentiment client"""
    
    @pytest.mark.asyncio
    async def test_get_market_data_bitcoin(self, social_client):
        """Test social sentiment for Bitcoin"""
        result = await social_client.get_market_data("bitcoin")
        
        assert isinstance(result, SocialSentimentData)
        assert 0 <= result.social_score <= 1
        assert result.mention_volume > 0
        assert "twitter" in result.platform_mentions
        assert "reddit" in result.platform_mentions
        assert len(result.trending_keywords) > 0
    
    @pytest.mark.asyncio
    async def test_get_market_data_ethereum(self, social_client):
        """Test social sentiment for Ethereum"""
        result = await social_client.get_market_data("ethereum")
        
        assert isinstance(result, SocialSentimentData)
        assert result.social_score is not None
        assert result.influencer_sentiment is not None
    
    @pytest.mark.asyncio
    async def test_get_market_data_caching(self, social_client):
        """Test social sentiment caching"""
        # First call
        result1 = await social_client.get_market_data("bitcoin")
        # Second call should use cache
        result2 = await social_client.get_market_data("bitcoin")
        
        assert result1 is result2  # Same cached object


class TestMarketDataErrorHandling:
    """Test error handling across all clients"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self, fear_greed_client):
        """Test rate limit error handling"""
        mock_response = MagicMock()
        mock_response.status = 429
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(APIRateLimitError):
                await fear_greed_client._make_request("http://test.com")
    
    @pytest.mark.asyncio
    async def test_authentication_error(self, coingecko_client):
        """Test authentication error handling"""
        mock_response = MagicMock()
        mock_response.status = 401
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(APIAuthenticationError):
                await coingecko_client._make_request("http://test.com")
    
    @pytest.mark.asyncio
    async def test_not_found_error(self, defi_client):
        """Test data not found error handling"""
        mock_response = MagicMock()
        mock_response.status = 404
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(DataNotAvailableError):
                await defi_client._make_request("http://test.com")
    
    @pytest.mark.asyncio
    async def test_generic_api_error(self, social_client):
        """Test generic API error handling"""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(MarketDataError):
                await social_client._make_request("http://test.com")
    
    @pytest.mark.asyncio
    async def test_network_error(self, onchain_client):
        """Test network error handling"""
        with patch('aiohttp.ClientSession.get', side_effect=aiohttp.ClientError("Network error")):
            with pytest.raises(MarketDataError):
                await onchain_client._make_request("http://test.com")


class TestCacheValidation:
    """Test cache validation across all clients"""
    
    def test_cache_validation_valid(self, fear_greed_client):
        """Test cache validation with valid cache"""
        cache_key = "test_key"
        fear_greed_client._cache_timestamps[cache_key] = datetime.now()
        
        assert fear_greed_client._is_cache_valid(cache_key)
    
    def test_cache_validation_expired(self, fear_greed_client):
        """Test cache validation with expired cache"""
        cache_key = "test_key"
        fear_greed_client._cache_timestamps[cache_key] = datetime.now() - timedelta(seconds=400)
        
        assert not fear_greed_client._is_cache_valid(cache_key)
    
    def test_cache_validation_missing(self, fear_greed_client):
        """Test cache validation with missing cache entry"""
        assert not fear_greed_client._is_cache_valid("missing_key")


class TestDataValidation:
    """Test data validation and edge cases"""
    
    @pytest.mark.asyncio
    async def test_fear_greed_invalid_value(self, fear_greed_client):
        """Test Fear & Greed Index with invalid value"""
        invalid_response = {
            "data": [{"value": "invalid", "timestamp": "1640995200"}]
        }
        
        with patch.object(fear_greed_client, '_make_request', return_value=invalid_response):
            with pytest.raises(MarketDataError):
                await fear_greed_client.get_market_data()
    
    def test_defi_tvl_calculation_edge_cases(self, defi_client):
        """Test DeFi TVL calculation with edge cases"""
        # Test zero values
        current_tvl = 1000000.0
        tvl_24h_ago = 0.0  # Edge case: zero historical value
        
        change_24h = ((current_tvl - tvl_24h_ago) / tvl_24h_ago * 100) if tvl_24h_ago > 0 else 0
        assert change_24h == 0  # Should not divide by zero
    
    @pytest.mark.asyncio
    async def test_correlation_calculation_edge_cases(self, coingecko_client):
        """Test correlation calculation with edge cases"""
        single_coin_response = {
            "data": {"market_cap_percentage": {"btc": 100.0}},  # 100% dominance edge case
        }
        
        with patch.object(coingecko_client, '_make_request') as mock_request:
            mock_request.side_effect = [single_coin_response, []]
            
            result = await coingecko_client.get_market_data()
            
            assert result.btc_dominance == 100.0
            assert result.eth_dominance == 0.0


class TestCoinGeckoOHLCVMethods:
    """Test CoinGecko OHLCV data fetching methods"""
    
    @pytest.fixture
    def sample_ohlcv_response(self):
        """Sample OHLCV response from CoinGecko API"""
        return [
            [1640995200000, 47000.0, 48000.0, 46000.0, 47500.0, 1000000000.0],  # timestamp, o, h, l, c, v
            [1641081600000, 47500.0, 49000.0, 47000.0, 48200.0, 1100000000.0],
            [1641168000000, 48200.0, 48500.0, 47800.0, 48000.0, 950000000.0],
        ]
    
    @pytest.fixture
    def sample_coin_list_response(self):
        """Sample coin list response from CoinGecko API"""
        return [
            {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
            {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
        ]
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_data_success(self, coingecko_client, sample_ohlcv_response):
        """Test successful OHLCV data retrieval"""
        with patch.object(coingecko_client, '_make_request', return_value=sample_ohlcv_response):
            result = await coingecko_client.get_ohlcv_data("bitcoin", days=7)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert list(result.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            assert result['open'].iloc[0] == 47000.0
            assert result['high'].iloc[0] == 48000.0
            assert result['low'].iloc[0] == 46000.0
            assert result['close'].iloc[0] == 47500.0
            assert result['volume'].iloc[0] == 1000000000.0
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_data_with_custom_range(self, coingecko_client, sample_ohlcv_response):
        """Test OHLCV data retrieval with custom date range"""
        with patch.object(coingecko_client, '_make_request', return_value=sample_ohlcv_response):
            from_date = datetime(2022, 1, 1)
            to_date = datetime(2022, 1, 7)
            
            result = await coingecko_client.get_ohlcv_data(
                "bitcoin", 
                from_date=from_date, 
                to_date=to_date
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_data_rate_limiting(self, coingecko_client):
        """Test rate limiting for OHLCV requests"""
        with patch.object(coingecko_client, '_make_request', side_effect=APIRateLimitError("Rate limit exceeded")):
            with pytest.raises(APIRateLimitError):
                await coingecko_client.get_ohlcv_data("bitcoin", days=7)
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_data_invalid_coin(self, coingecko_client):
        """Test OHLCV data retrieval with invalid coin ID"""
        with patch.object(coingecko_client, '_make_request', side_effect=DataNotAvailableError("Coin not found")):
            with pytest.raises(DataNotAvailableError):
                await coingecko_client.get_ohlcv_data("invalid-coin", days=7)
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_data_empty_response(self, coingecko_client):
        """Test OHLCV data retrieval with empty response"""
        with patch.object(coingecko_client, '_make_request', return_value=[]):
            result = await coingecko_client.get_ohlcv_data("bitcoin", days=7)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_data_malformed_data(self, coingecko_client):
        """Test OHLCV data retrieval with malformed data"""
        malformed_data = [
            [1640995200000, 47000.0, 48000.0],  # Missing low, close, volume
            [1641081600000, 47500.0, 49000.0, 47000.0, 48200.0, 1100000000.0],
        ]
        
        with patch.object(coingecko_client, '_make_request', return_value=malformed_data):
            with pytest.raises(MarketDataError):
                await coingecko_client.get_ohlcv_data("bitcoin", days=7)
    
    @pytest.mark.asyncio
    async def test_search_coin_id_success(self, coingecko_client, sample_coin_list_response):
        """Test successful coin ID search"""
        with patch.object(coingecko_client, '_make_request', return_value=sample_coin_list_response):
            result = await coingecko_client.search_coin_id("bitcoin")
            
            assert result == "bitcoin"
    
    @pytest.mark.asyncio
    async def test_search_coin_id_by_symbol(self, coingecko_client, sample_coin_list_response):
        """Test coin ID search by symbol"""
        with patch.object(coingecko_client, '_make_request', return_value=sample_coin_list_response):
            result = await coingecko_client.search_coin_id("BTC")
            
            assert result == "bitcoin"
    
    @pytest.mark.asyncio
    async def test_search_coin_id_not_found(self, coingecko_client, sample_coin_list_response):
        """Test coin ID search for non-existent coin"""
        with patch.object(coingecko_client, '_make_request', return_value=sample_coin_list_response):
            with pytest.raises(DataNotAvailableError):
                await coingecko_client.search_coin_id("unknown-coin")
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_data_caching(self, coingecko_client, sample_ohlcv_response):
        """Test OHLCV data caching"""
        with patch.object(coingecko_client, '_make_request', return_value=sample_ohlcv_response) as mock_request:
            # First call
            result1 = await coingecko_client.get_ohlcv_data("bitcoin", days=7)
            # Second call should use cache
            result2 = await coingecko_client.get_ohlcv_data("bitcoin", days=7)
            
            # Should only make one API request
            assert mock_request.call_count == 1
            pd.testing.assert_frame_equal(result1, result2)
    
    @pytest.mark.asyncio
    async def test_get_ohlcv_data_cache_key_uniqueness(self, coingecko_client, sample_ohlcv_response):
        """Test that different parameters create different cache keys"""
        with patch.object(coingecko_client, '_make_request', return_value=sample_ohlcv_response) as mock_request:
            # Different coins should create separate cache entries
            await coingecko_client.get_ohlcv_data("bitcoin", days=7)
            await coingecko_client.get_ohlcv_data("ethereum", days=7)
            
            # Should make two API requests
            assert mock_request.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_historical_data_for_token_address(self, coingecko_client, sample_ohlcv_response):
        """Test getting historical data using token address"""
        # Mock the coin ID resolution
        coin_list_response = [{"id": "some-token", "symbol": "token", "name": "Some Token"}]
        
        with patch.object(coingecko_client, '_make_request') as mock_request:
            # First call returns coin list, second returns OHLCV data
            mock_request.side_effect = [coin_list_response, sample_ohlcv_response]
            
            result = await coingecko_client.get_historical_data_for_token("0x1234567890abcdef", days=7)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert mock_request.call_count == 2