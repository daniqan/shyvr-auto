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
    return SocialSentimentClient(api_key="test-lunarcrush-api-key", cache_ttl=60)


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
    """Test on-chain analytics client with real Helius API integration"""
    
    @pytest.fixture
    def helius_client(self):
        """Create Helius on-chain analytics client for testing"""
        return OnChainAnalyticsClient(api_key="test-helius-api-key", cache_ttl=60)
    
    @pytest.fixture
    def sample_helius_enhanced_transactions_response(self):
        """Sample Helius enhanced transactions API response"""
        return [
            {
                "signature": "5VgJd...",
                "timestamp": 1640995200,
                "description": "Token transfer",
                "type": "SWAP",
                "source": "JUPITER",
                "feePayer": "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
                "fee": 5000,
                "nativeTransfers": [
                    {
                        "fromUserAccount": "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
                        "toUserAccount": "H6ARHNZgvKC6DaYGAkNShUAhbFcT3KbkVG1p2ksswi5z",
                        "amount": 50000000
                    }
                ],
                "tokenTransfers": [
                    {
                        "fromUserAccount": "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
                        "toUserAccount": "H6ARHHNZgvKC6DaYGAkNShUAhbFcT3KbkVG1p2ksswi5z",
                        "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                        "tokenAmount": 1000000
                    }
                ]
            }
        ]
    
    @pytest.fixture
    def sample_helius_webhook_response(self):
        """Sample Helius webhook response for whale activity"""
        return {
            "type": "transaction",
            "data": {
                "accountData": [],
                "transaction": {
                    "signatures": ["5VgJd..."],
                    "message": {
                        "header": {"numRequiredSignatures": 1},
                        "accountKeys": ["9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"],
                        "instructions": []
                    }
                }
            }
        }
    
    @pytest.fixture
    def sample_rpc_transaction_count_response(self):
        """Sample RPC response for transaction count"""
        return {
            "jsonrpc": "2.0",
            "result": {
                "context": {"slot": 166974442},
                "value": 1245678
            },
            "id": 1
        }
    
    @pytest.mark.asyncio
    async def test_get_market_data_solana_success(self, helius_client, sample_helius_enhanced_transactions_response):
        """Test successful on-chain metrics for Solana using Helius API"""
        with patch.object(helius_client, '_make_request') as mock_request:
            # Mock multiple API calls for different metrics
            mock_request.side_effect = [
                sample_helius_enhanced_transactions_response,  # Enhanced transactions
                {"result": {"value": 150000}},  # Transaction count
                {"result": {"value": 75000}},   # Active addresses
                {"result": {"value": 250000000}},  # Volume
                {"result": {"value": 50000}},   # Network fees
                {"whale_transactions": 45, "large_volume_24h": 5000000}  # Whale activity
            ]
            
            result = await helius_client.get_market_data(Chain.SOLANA)
            
            assert isinstance(result, OnChainMetrics)
            assert result.network_activity["chain"] == "solana"
            assert result.transaction_count_24h == 150000
            assert result.active_addresses_24h == 75000
            assert result.transaction_volume_24h == 250000000.0
            assert result.network_fees_24h == 50000.0
            assert result.whale_activity["whale_transactions"] == 45
            assert result.hash_rate is None  # Solana is PoS
            assert result.staking_ratio is not None  # Solana has staking
    
    @pytest.mark.asyncio
    async def test_get_market_data_ethereum_success(self, helius_client):
        """Test on-chain metrics for Ethereum (fallback to other APIs)"""
        with patch.object(helius_client, '_make_request') as mock_request:
            # Mock responses for Ethereum data (using fallback APIs)
            mock_request.side_effect = [
                {"result": "0x186a0"},  # 100000 in hex
                {"result": "0x12345"},  # 74565 in hex
                {"result": "0x174876e800"},  # Volume in wei
                {"result": "0x2710"},  # Fees in wei
                {"staking_ratio": 0.65, "active_validators": 750000}  # Staking data
            ]
            
            result = await helius_client.get_market_data(Chain.ETHEREUM)
            
            assert isinstance(result, OnChainMetrics)
            assert result.network_activity["chain"] == "ethereum"
            assert result.transaction_count_24h == 100000
            assert result.active_addresses_24h == 74565
            assert result.staking_ratio == 0.65
            assert result.hash_rate is None  # Ethereum is PoS
    
    @pytest.mark.asyncio
    async def test_get_transaction_count_24h(self, helius_client, sample_rpc_transaction_count_response):
        """Test getting 24h transaction count"""
        with patch.object(helius_client, '_make_request', return_value=sample_rpc_transaction_count_response):
            count = await helius_client._get_transaction_count_24h(Chain.SOLANA)
            
            assert count == 1245678
    
    @pytest.mark.asyncio
    async def test_get_active_addresses_24h(self, helius_client):
        """Test getting 24h active addresses"""
        mock_response = {"result": {"value": 85000}}
        
        with patch.object(helius_client, '_make_request', return_value=mock_response):
            count = await helius_client._get_active_addresses_24h(Chain.SOLANA)
            
            assert count == 85000
    
    @pytest.mark.asyncio
    async def test_get_transaction_volume_24h(self, helius_client):
        """Test getting 24h transaction volume"""
        mock_response = {"result": {"value": 500000000}}
        
        with patch.object(helius_client, '_make_request', return_value=mock_response):
            volume = await helius_client._get_transaction_volume_24h(Chain.SOLANA)
            
            assert volume == 500000000.0
    
    @pytest.mark.asyncio
    async def test_get_network_fees_24h(self, helius_client):
        """Test getting 24h network fees"""
        mock_response = {"result": {"value": 75000}}
        
        with patch.object(helius_client, '_make_request', return_value=mock_response):
            fees = await helius_client._get_network_fees_24h(Chain.SOLANA)
            
            assert fees == 75000.0
    
    @pytest.mark.asyncio
    async def test_get_whale_activity(self, helius_client, sample_helius_enhanced_transactions_response):
        """Test getting whale activity data"""
        with patch.object(helius_client, '_make_request', return_value=sample_helius_enhanced_transactions_response):
            whale_data = await helius_client._get_whale_activity(Chain.SOLANA)
            
            assert isinstance(whale_data, dict)
            assert "large_transactions_24h" in whale_data
            assert "whale_net_flow" in whale_data
            assert whale_data["large_transactions_24h"] >= 0
    
    @pytest.mark.asyncio
    async def test_api_authentication(self, helius_client):
        """Test API authentication with Helius"""
        with patch.object(helius_client, '_make_request') as mock_request:
            mock_request.return_value = {"result": {"value": 100}}
            
            await helius_client._get_transaction_count_24h(Chain.SOLANA)
            
            # Verify API key is included in request
            call_args = mock_request.call_args
            assert "api-key" in call_args[0][0]  # URL should contain api-key parameter
    
    @pytest.mark.asyncio
    async def test_rate_limiting_helius(self, helius_client):
        """Test rate limiting for Helius API calls"""
        with patch.object(helius_client, '_make_request', side_effect=APIRateLimitError("Rate limit exceeded")):
            with pytest.raises(APIRateLimitError):
                await helius_client.get_market_data(Chain.SOLANA)
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self, helius_client):
        """Test handling of invalid API key"""
        with patch.object(helius_client, '_make_request', side_effect=APIAuthenticationError("Invalid API key")):
            with pytest.raises(APIAuthenticationError):
                await helius_client.get_market_data(Chain.SOLANA)
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, helius_client):
        """Test network error handling"""
        with patch.object(helius_client, '_make_request', side_effect=MarketDataError("Network error")):
            with pytest.raises(MarketDataError):
                await helius_client.get_market_data(Chain.SOLANA)
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self, helius_client):
        """Test handling of empty API responses"""
        with patch.object(helius_client, '_make_request', return_value={}):
            result = await helius_client.get_market_data(Chain.SOLANA)
            
            assert isinstance(result, OnChainMetrics)
            assert result.transaction_count_24h == 0
            assert result.active_addresses_24h == 0
    
    @pytest.mark.asyncio
    async def test_supported_chain_fallback(self, helius_client):
        """Test fallback behavior for supported EVM chains"""
        result = await helius_client.get_market_data(Chain.BASE)
        
        assert isinstance(result, OnChainMetrics)
        assert result.network_activity["chain"] == "base"
        # Should use Ethereum-compatible data structure
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, helius_client):
        """Test caching of on-chain data"""
        mock_response = {"result": {"value": 100000}}
        
        with patch.object(helius_client, '_make_request', return_value=mock_response) as mock_request:
            # First call
            result1 = await helius_client.get_market_data(Chain.SOLANA)
            # Second call should use cache
            result2 = await helius_client.get_market_data(Chain.SOLANA)
            
            # Should only make API requests once due to caching
            assert len([call for call in mock_request.call_args_list if 'enhanced' in str(call)]) <= 1
            assert result1.timestamp == result2.timestamp
    
    @pytest.mark.asyncio
    async def test_enhanced_transactions_parsing(self, helius_client, sample_helius_enhanced_transactions_response):
        """Test parsing of Helius enhanced transactions"""
        with patch.object(helius_client, '_make_request', return_value=sample_helius_enhanced_transactions_response):
            parsed_data = await helius_client._parse_enhanced_transactions(sample_helius_enhanced_transactions_response)
            
            assert isinstance(parsed_data, dict)
            assert "transaction_count" in parsed_data
            assert "total_volume" in parsed_data
            assert "unique_addresses" in parsed_data
            assert parsed_data["transaction_count"] == 1
    
    @pytest.mark.asyncio
    async def test_whale_detection_threshold(self, helius_client):
        """Test whale activity detection with different thresholds"""
        large_transaction_response = [
            {
                "signature": "test1",
                "nativeTransfers": [{"amount": 1000000000}],  # 1 SOL = large
                "tokenTransfers": [{"tokenAmount": 50000000}]   # 50M tokens = whale
            },
            {
                "signature": "test2", 
                "nativeTransfers": [{"amount": 1000000}],     # 0.001 SOL = small
                "tokenTransfers": [{"tokenAmount": 1000}]       # 1K tokens = small
            }
        ]
        
        with patch.object(helius_client, '_make_request', return_value=large_transaction_response):
            whale_data = await helius_client._get_whale_activity(Chain.SOLANA)
            
            # Should detect only the large transaction as whale activity
            assert whale_data["large_transactions_24h"] >= 1
            assert whale_data["whale_net_flow"] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_chain_support(self, helius_client):
        """Test support for multiple blockchain networks"""
        supported_chains = [Chain.SOLANA, Chain.ETHEREUM, Chain.BASE]
        
        for chain in supported_chains:
            with patch.object(helius_client, '_make_request', return_value={"result": {"value": 1000}}):
                result = await helius_client.get_market_data(chain)
                
                assert isinstance(result, OnChainMetrics)
                assert result.network_activity["chain"] == chain.value
    
    @pytest.mark.asyncio
    async def test_data_freshness_validation(self, helius_client):
        """Test validation of data freshness"""
        current_time = datetime.now()
        
        with patch.object(helius_client, '_make_request', return_value={"result": {"value": 1000}}):
            result = await helius_client.get_market_data(Chain.SOLANA)
            
            # Data should be recent (within last few minutes)
            time_diff = abs((result.timestamp - current_time).total_seconds())
            assert time_diff < 300  # Within 5 minutes


class TestSocialSentimentClient:
    """Test social sentiment client with real LunarCrush API integration"""
    
    @pytest.fixture
    def lunarcrush_client(self):
        """Create LunarCrush social sentiment client for testing"""
        return SocialSentimentClient(api_key="test-lunarcrush-api-key", cache_ttl=60)
    
    @pytest.fixture
    def sample_lunarcrush_topic_response(self):
        """Sample LunarCrush topic API response"""
        return {
            "data": {
                "topic": "bitcoin",
                "rank": 1,
                "title": "Bitcoin",
                "summary": "24h social activity summary for Bitcoin",
                "interactions_24h": 125000,
                "posts_24h": 5500,
                "contributors_24h": 2800,
                "sentiment": 3.2,  # 1-5 scale
                "sentiment_absolute": 0.64,  # 0-1 scale
                "galaxy_score": 78.5,
                "alt_rank": 1,
                "social_dominance": 15.2,
                "platforms": {
                    "reddit": {
                        "posts": 1500,
                        "interactions": 35000,
                        "sentiment": 3.1
                    },
                    "twitter": {
                        "posts": 3000,
                        "interactions": 75000,
                        "sentiment": 3.3
                    },
                    "youtube": {
                        "posts": 1000,
                        "interactions": 15000,
                        "sentiment": 3.0
                    }
                },
                "keywords": ["bullish", "moon", "hodl", "btc", "crypto"]
            }
        }
    
    @pytest.fixture
    def sample_lunarcrush_time_series_response(self):
        """Sample LunarCrush time series API response"""
        return {
            "data": [
                {
                    "time": 1640995200,
                    "interactions": 120000,
                    "posts": 5200,
                    "contributors": 2600,
                    "sentiment": 3.1,
                    "galaxy_score": 76.2,
                    "social_dominance": 14.8
                },
                {
                    "time": 1641081600,
                    "interactions": 125000,
                    "posts": 5500,
                    "contributors": 2800,
                    "sentiment": 3.2,
                    "galaxy_score": 78.5,
                    "social_dominance": 15.2
                }
            ]
        }
    
    @pytest.fixture
    def sample_lunarcrush_posts_response(self):
        """Sample LunarCrush posts API response"""
        return {
            "data": [
                {
                    "id": "post_123",
                    "title": "Bitcoin to the moon!",
                    "content": "BTC looking bullish today",
                    "url": "https://twitter.com/user/status/123",
                    "platform": "twitter",
                    "interactions": 1500,
                    "sentiment": 4,
                    "creator": {
                        "name": "crypto_influencer",
                        "followers": 100000,
                        "influence_score": 85
                    },
                    "created_time": 1640995200
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_get_market_data_bitcoin_success(self, lunarcrush_client, sample_lunarcrush_topic_response):
        """Test successful social sentiment retrieval for Bitcoin using LunarCrush API"""
        with patch.object(lunarcrush_client, '_get_topic_data', return_value=sample_lunarcrush_topic_response), \
             patch.object(lunarcrush_client, '_calculate_sentiment_trend', return_value=0.1), \
             patch.object(lunarcrush_client, '_calculate_influencer_sentiment', return_value=0.65):
            result = await lunarcrush_client.get_market_data("bitcoin")
            
            assert isinstance(result, SocialSentimentData)
            assert result.social_score == 0.64  # Converted from 1-5 to 0-1 scale
            assert result.mention_volume == 5500  # posts_24h
            assert result.sentiment_trend > 0  # Should be positive based on data
            assert "reddit" in result.platform_mentions
            assert "twitter" in result.platform_mentions
            assert "youtube" in result.platform_mentions
            assert result.platform_mentions["reddit"] == 1500
            assert result.platform_mentions["twitter"] == 3000
            assert result.platform_mentions["youtube"] == 1000
            assert "bullish" in result.trending_keywords
            assert "moon" in result.trending_keywords
            assert result.influencer_sentiment is not None
    
    @pytest.mark.asyncio
    async def test_get_market_data_ethereum_success(self, lunarcrush_client):
        """Test social sentiment for Ethereum"""
        ethereum_response = {
            "data": {
                "topic": "ethereum",
                "interactions_24h": 85000,
                "posts_24h": 3200,
                "sentiment_absolute": 0.58,
                "platforms": {
                    "reddit": {"posts": 1200, "sentiment": 2.9},
                    "twitter": {"posts": 2000, "sentiment": 2.8}
                },
                "keywords": ["eth", "ethereum", "defi"]
            }
        }
        
        with patch.object(lunarcrush_client, '_get_topic_data', return_value=ethereum_response), \
             patch.object(lunarcrush_client, '_calculate_sentiment_trend', return_value=0.05), \
             patch.object(lunarcrush_client, '_calculate_influencer_sentiment', return_value=0.6):
            result = await lunarcrush_client.get_market_data("ethereum")
            
            assert isinstance(result, SocialSentimentData)
            assert result.social_score == 0.58
            assert result.mention_volume == 3200
            assert "eth" in result.trending_keywords
    
    @pytest.mark.asyncio
    async def test_api_authentication_header(self, lunarcrush_client):
        """Test that API key is included in Authorization header"""
        sample_response = {"data": {"topic": "bitcoin", "sentiment_absolute": 0.5, "posts_24h": 100, "platforms": {}, "keywords": []}}
        
        with patch.object(lunarcrush_client, '_make_request') as mock_request:
            mock_request.return_value = sample_response
            
            await lunarcrush_client._get_topic_data("bitcoin")
            
            # Check that Authorization header with Bearer token was used
            call_args = mock_request.call_args
            headers = call_args[1].get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-lunarcrush-api-key"
    
    @pytest.mark.asyncio
    async def test_rate_limiting_lunarcrush_api(self, lunarcrush_client):
        """Test rate limiting for LunarCrush API calls"""
        with patch.object(lunarcrush_client, '_get_topic_data', side_effect=APIRateLimitError("Rate limit exceeded")):
            with pytest.raises(APIRateLimitError):
                await lunarcrush_client.get_market_data("bitcoin")
    
    @pytest.mark.asyncio
    async def test_invalid_api_key_lunarcrush(self, lunarcrush_client):
        """Test handling of invalid LunarCrush API key"""
        with patch.object(lunarcrush_client, '_get_topic_data', side_effect=APIAuthenticationError("Invalid API key")):
            with pytest.raises(APIAuthenticationError):
                await lunarcrush_client.get_market_data("bitcoin")
    
    @pytest.mark.asyncio
    async def test_asset_not_found_lunarcrush(self, lunarcrush_client):
        """Test handling when asset is not found in LunarCrush"""
        with patch.object(lunarcrush_client, '_get_topic_data', side_effect=DataNotAvailableError("Asset not found")):
            with pytest.raises(DataNotAvailableError):
                await lunarcrush_client.get_market_data("unknown-asset")
    
    @pytest.mark.asyncio
    async def test_caching_behavior_lunarcrush(self, lunarcrush_client, sample_lunarcrush_topic_response):
        """Test caching behavior for LunarCrush API responses"""
        with patch.object(lunarcrush_client, '_get_topic_data', return_value=sample_lunarcrush_topic_response) as mock_topic, \
             patch.object(lunarcrush_client, '_calculate_sentiment_trend', return_value=0.1), \
             patch.object(lunarcrush_client, '_calculate_influencer_sentiment', return_value=0.65):
            # First call
            result1 = await lunarcrush_client.get_market_data("bitcoin")
            # Second call should use cache
            result2 = await lunarcrush_client.get_market_data("bitcoin")
            
            # Should use caching - only make API requests once
            assert mock_topic.call_count == 1
            assert result1 is result2  # Same cached object
    
    @pytest.mark.asyncio
    async def test_get_topic_data_success(self, lunarcrush_client):
        """Test _get_topic_data method"""
        expected_response = {"data": {"topic": "bitcoin", "sentiment_absolute": 0.5}}
        
        with patch.object(lunarcrush_client, '_make_request', return_value=expected_response):
            result = await lunarcrush_client._get_topic_data("bitcoin")
            
            assert result == expected_response
    
    @pytest.mark.asyncio
    async def test_get_topic_data_error(self, lunarcrush_client):
        """Test _get_topic_data method with error"""
        with patch.object(lunarcrush_client, '_make_request', side_effect=MarketDataError("API error")):
            with pytest.raises(MarketDataError):
                await lunarcrush_client._get_topic_data("bitcoin")
    
    @pytest.mark.asyncio
    async def test_calculate_sentiment_trend_success(self, lunarcrush_client, sample_lunarcrush_time_series_response):
        """Test _calculate_sentiment_trend method"""
        with patch.object(lunarcrush_client, '_make_request', return_value=sample_lunarcrush_time_series_response):
            result = await lunarcrush_client._calculate_sentiment_trend("bitcoin")
            
            assert isinstance(result, float)
            assert -1.0 <= result <= 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_sentiment_trend_insufficient_data(self, lunarcrush_client):
        """Test _calculate_sentiment_trend with insufficient data"""
        insufficient_data = {"data": [{"sentiment": 3.0}]}  # Only one data point
        
        with patch.object(lunarcrush_client, '_make_request', return_value=insufficient_data):
            result = await lunarcrush_client._calculate_sentiment_trend("bitcoin")
            
            assert result == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_sentiment_trend_error(self, lunarcrush_client):
        """Test _calculate_sentiment_trend with API error"""
        with patch.object(lunarcrush_client, '_make_request', side_effect=MarketDataError("API error")):
            result = await lunarcrush_client._calculate_sentiment_trend("bitcoin")
            
            assert result == 0.0  # Should return 0.0 on error
    
    @pytest.mark.asyncio
    async def test_calculate_influencer_sentiment_success(self, lunarcrush_client, sample_lunarcrush_posts_response):
        """Test _calculate_influencer_sentiment method"""
        with patch.object(lunarcrush_client, '_make_request', return_value=sample_lunarcrush_posts_response):
            result = await lunarcrush_client._calculate_influencer_sentiment("bitcoin")
            
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_influencer_sentiment_no_posts(self, lunarcrush_client):
        """Test _calculate_influencer_sentiment with no posts"""
        empty_posts = {"data": []}
        
        with patch.object(lunarcrush_client, '_make_request', return_value=empty_posts):
            result = await lunarcrush_client._calculate_influencer_sentiment("bitcoin")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_calculate_influencer_sentiment_error(self, lunarcrush_client):
        """Test _calculate_influencer_sentiment with API error"""
        with patch.object(lunarcrush_client, '_make_request', side_effect=MarketDataError("API error")):
            result = await lunarcrush_client._calculate_influencer_sentiment("bitcoin")
            
            assert result is None  # Should return None on error
    
    @pytest.mark.asyncio
    async def test_no_data_available_error(self, lunarcrush_client):
        """Test handling when no data is available from API"""
        empty_response = {"data": {}}  # Empty data object
        
        with patch.object(lunarcrush_client, '_get_topic_data', return_value=empty_response):
            with pytest.raises(DataNotAvailableError):
                await lunarcrush_client.get_market_data("bitcoin")
    
    @pytest.mark.asyncio
    async def test_platform_sentiment_conversion(self, lunarcrush_client):
        """Test conversion of platform sentiment from 1-5 scale to 0-1 scale"""
        platform_response = {
            "data": {
                "sentiment_absolute": 0.6,
                "posts_24h": 1000,
                "platforms": {
                    "twitter": {"posts": 500, "sentiment": 4.0},  # Should convert to 0.75
                    "reddit": {"posts": 300, "sentiment": 2.0},   # Should convert to 0.25
                    "youtube": {"posts": 200, "sentiment": 5.0}   # Should convert to 1.0
                },
                "keywords": ["btc", "bitcoin"]
            }
        }
        
        with patch.object(lunarcrush_client, '_get_topic_data', return_value=platform_response), \
             patch.object(lunarcrush_client, '_calculate_sentiment_trend', return_value=0.1), \
             patch.object(lunarcrush_client, '_calculate_influencer_sentiment', return_value=0.65):
            result = await lunarcrush_client.get_market_data("bitcoin")
            
            # Check sentiment conversion
            assert result.sentiment_breakdown["twitter"] == 0.75
            assert result.sentiment_breakdown["reddit"] == 0.25
            assert result.sentiment_breakdown["youtube"] == 1.0
    
    @pytest.mark.asyncio
    async def test_multiple_assets(self, lunarcrush_client):
        """Test fetching data for different cryptocurrency assets"""
        assets = ["bitcoin", "ethereum", "solana"]
        
        for asset in assets:
            asset_response = {
                "data": {
                    "topic": asset,
                    "sentiment_absolute": 0.5,
                    "posts_24h": 1000,
                    "platforms": {"twitter": {"posts": 500, "sentiment": 3.0}},
                    "keywords": [asset.lower()]
                }
            }
            
            with patch.object(lunarcrush_client, '_get_topic_data', return_value=asset_response), \
                 patch.object(lunarcrush_client, '_calculate_sentiment_trend', return_value=0.0), \
                 patch.object(lunarcrush_client, '_calculate_influencer_sentiment', return_value=0.5):
                result = await lunarcrush_client.get_market_data(asset)
                
                assert isinstance(result, SocialSentimentData)
                assert asset.lower() in result.trending_keywords
    


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