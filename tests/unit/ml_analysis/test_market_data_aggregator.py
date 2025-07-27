"""
Unit tests for market data aggregator
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from src.ml_analysis.market_data_aggregator import MarketDataAggregator
from src.ml_analysis.market_data import (
    MarketSentimentData,
    DeFiMetrics,
    CorrelationMetrics,
    OnChainMetrics,
    SocialSentimentData,
    MarketDataError
)
from src.ml_analysis.base import MarketFeatures
from src.utils.base import Chain


class TestMarketDataAggregator:
    """Test market data aggregator"""
    
    @pytest.fixture
    def aggregator(self):
        """Create market data aggregator for testing"""
        return MarketDataAggregator(
            coingecko_api_key="test_key",
            cache_ttl=60,
            max_retries=2,
            retry_delay=0.1  # Fast retries for testing
        )
    
    @pytest.fixture
    def sample_sentiment_data(self):
        """Sample sentiment data"""
        return MarketSentimentData(
            fear_greed_index=75.0,
            fear_greed_classification="Greed",
            market_trend="bull",
            volatility_regime="medium"
        )
    
    @pytest.fixture
    def sample_defi_data(self):
        """Sample DeFi data"""
        return DeFiMetrics(
            total_value_locked=50e9,
            tvl_change_24h=2.5,
            tvl_change_7d=5.0,
            defi_dominance=0.05,
            protocols_count=300,
            chains_tvl={"Ethereum": 25e9, "BSC": 5e9}
        )
    
    @pytest.fixture
    def sample_correlation_data(self):
        """Sample correlation data"""
        return CorrelationMetrics(
            btc_correlation=1.0,
            eth_correlation=0.8,
            btc_dominance=42.5,
            eth_dominance=18.2,
            stablecoin_dominance=8.0,
            market_beta=1.0,
            correlation_matrix={"BTC": 2.5, "ETH": 3.1}
        )
    
    @pytest.fixture
    def sample_onchain_data(self):
        """Sample on-chain data"""
        return OnChainMetrics(
            network_activity={"chain": "ethereum", "gas_price": 20.0},
            transaction_count_24h=1200000,
            active_addresses_24h=600000,
            transaction_volume_24h=8e9,
            network_fees_24h=2e6,
            whale_activity={"large_transactions_24h": 150, "whale_net_flow": 1500.0}
        )
    
    @pytest.fixture
    def sample_social_data(self):
        """Sample social data"""
        return SocialSentimentData(
            social_score=0.7,
            mention_volume=5000,
            sentiment_trend=0.1,
            platform_mentions={"twitter": 3000, "reddit": 2000},
            sentiment_breakdown={"twitter": 0.75, "reddit": 0.65},
            trending_keywords=["bullish", "moon", "hodl"],
            influencer_sentiment=0.8
        )
    
    @pytest.mark.asyncio
    async def test_get_market_features_success(self, aggregator, sample_sentiment_data, 
                                              sample_defi_data, sample_correlation_data,
                                              sample_onchain_data, sample_social_data):
        """Test successful market features aggregation"""
        # Mock all client responses
        aggregator.fear_greed_client.get_market_data = AsyncMock(return_value=sample_sentiment_data)
        aggregator.defi_client.get_market_data = AsyncMock(return_value=sample_defi_data)
        aggregator.coingecko_client.get_market_data = AsyncMock(return_value=sample_correlation_data)
        aggregator.onchain_client.get_market_data = AsyncMock(return_value=sample_onchain_data)
        aggregator.social_client.get_market_data = AsyncMock(return_value=sample_social_data)
        
        result = await aggregator.get_market_features()
        
        assert isinstance(result, MarketFeatures)
        
        # Check sentiment data mapping
        assert result.fear_greed_index == 75.0
        assert result.fear_greed_classification == "Greed"
        assert result.market_trend == "bull"
        assert result.volatility_regime == "medium"
        
        # Check correlation data mapping
        assert result.btc_correlation == 1.0
        assert result.eth_correlation == 0.8
        assert result.btc_dominance == 42.5
        assert result.eth_dominance == 18.2
        assert result.stablecoin_dominance == 8.0
        
        # Check DeFi data mapping
        assert result.total_value_locked == 50e9
        assert result.tvl_change_24h == 2.5
        assert result.tvl_change_7d == 5.0
        assert result.active_protocols == 300
        
        # Check on-chain data mapping
        assert result.transaction_count_24h == 1200000
        assert result.active_addresses_24h == 600000
        assert result.transaction_volume_24h == 8e9
        assert result.network_fees_24h == 2e6
        assert result.whale_activity_score is not None
        
        # Check social data mapping
        assert result.social_score == 0.7
        assert result.mention_volume == 5000
        assert result.sentiment_trend == 0.1
        assert result.influencer_sentiment == 0.8
    
    @pytest.mark.asyncio
    async def test_get_market_features_partial_failure(self, aggregator, sample_sentiment_data):
        """Test market features aggregation with some API failures"""
        # Mock some clients to succeed and others to fail
        aggregator.fear_greed_client.get_market_data = AsyncMock(return_value=sample_sentiment_data)
        aggregator.defi_client.get_market_data = AsyncMock(side_effect=MarketDataError("DeFi API error"))
        aggregator.coingecko_client.get_market_data = AsyncMock(side_effect=MarketDataError("CoinGecko API error"))
        aggregator.onchain_client.get_market_data = AsyncMock(side_effect=MarketDataError("On-chain API error"))
        aggregator.social_client.get_market_data = AsyncMock(side_effect=MarketDataError("Social API error"))
        
        result = await aggregator.get_market_features()
        
        assert isinstance(result, MarketFeatures)
        
        # Should have sentiment data
        assert result.fear_greed_index == 75.0
        assert result.market_trend == "bull"
        
        # Should have None for failed APIs
        assert result.total_value_locked is None
        assert result.btc_dominance is None
        assert result.transaction_count_24h is None
        assert result.social_score is None
    
    @pytest.mark.asyncio
    async def test_get_market_features_all_failures(self, aggregator):
        """Test market features aggregation with all API failures"""
        # Mock all clients to fail
        error = MarketDataError("API error")
        aggregator.fear_greed_client.get_market_data = AsyncMock(side_effect=error)
        aggregator.defi_client.get_market_data = AsyncMock(side_effect=error)
        aggregator.coingecko_client.get_market_data = AsyncMock(side_effect=error)
        aggregator.onchain_client.get_market_data = AsyncMock(side_effect=error)
        aggregator.social_client.get_market_data = AsyncMock(side_effect=error)
        
        result = await aggregator.get_market_features()
        
        # Should return default market features
        assert isinstance(result, MarketFeatures)
        assert result.fear_greed_index == 50.0
        assert result.market_trend == "sideways"
        assert result.total_value_locked == 100e9
    
    @pytest.mark.asyncio
    async def test_get_market_features_caching(self, aggregator, sample_sentiment_data):
        """Test market features caching"""
        aggregator.fear_greed_client.get_market_data = AsyncMock(return_value=sample_sentiment_data)
        aggregator.defi_client.get_market_data = AsyncMock(return_value=None)
        aggregator.coingecko_client.get_market_data = AsyncMock(return_value=None)
        aggregator.onchain_client.get_market_data = AsyncMock(return_value=None)
        aggregator.social_client.get_market_data = AsyncMock(return_value=None)
        
        # First call
        result1 = await aggregator.get_market_features()
        # Second call should use cache
        result2 = await aggregator.get_market_features()
        
        # Should only call APIs once
        aggregator.fear_greed_client.get_market_data.assert_called_once()
        assert result1 is result2  # Same cached object
    
    @pytest.mark.asyncio
    async def test_get_market_features_cache_expiry(self, aggregator, sample_sentiment_data):
        """Test market features cache expiry"""
        aggregator.cache_ttl = 1  # 1 second cache
        aggregator.fear_greed_client.get_market_data = AsyncMock(return_value=sample_sentiment_data)
        aggregator.defi_client.get_market_data = AsyncMock(return_value=None)
        aggregator.coingecko_client.get_market_data = AsyncMock(return_value=None)
        aggregator.onchain_client.get_market_data = AsyncMock(return_value=None)
        aggregator.social_client.get_market_data = AsyncMock(return_value=None)
        
        # First call
        await aggregator.get_market_features()
        
        # Wait for cache to expire
        await asyncio.sleep(1.1)
        
        # Second call should not use cache
        await aggregator.get_market_features()
        
        # Should call APIs twice
        assert aggregator.fear_greed_client.get_market_data.call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_logic_success_after_failure(self, aggregator, sample_sentiment_data):
        """Test retry logic succeeds after initial failure"""
        # Mock client to fail once then succeed
        call_count = 0
        async def mock_get_data():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MarketDataError("Temporary error")
            return sample_sentiment_data
        
        aggregator.fear_greed_client.get_market_data = mock_get_data
        aggregator.defi_client.get_market_data = AsyncMock(return_value=None)
        aggregator.coingecko_client.get_market_data = AsyncMock(return_value=None)
        aggregator.onchain_client.get_market_data = AsyncMock(return_value=None)
        aggregator.social_client.get_market_data = AsyncMock(return_value=None)
        
        result = await aggregator.get_market_features()
        
        assert call_count == 2  # Should retry once
        assert result.fear_greed_index == 75.0  # Should succeed on retry
    
    @pytest.mark.asyncio
    async def test_retry_logic_max_retries_exceeded(self, aggregator):
        """Test retry logic when max retries exceeded"""
        # Mock client to always fail
        aggregator.fear_greed_client.get_market_data = AsyncMock(side_effect=MarketDataError("Persistent error"))
        aggregator.defi_client.get_market_data = AsyncMock(return_value=None)
        aggregator.coingecko_client.get_market_data = AsyncMock(return_value=None)
        aggregator.onchain_client.get_market_data = AsyncMock(return_value=None)
        aggregator.social_client.get_market_data = AsyncMock(return_value=None)
        
        result = await aggregator.get_market_features()
        
        # Should call 3 times (initial + 2 retries)
        assert aggregator.fear_greed_client.get_market_data.call_count == 3
        # Should have None for failed data
        assert result.fear_greed_index is None
    
    def test_calculate_whale_activity_score_normal(self, aggregator, sample_onchain_data):
        """Test whale activity score calculation with normal data"""
        score = aggregator._calculate_whale_activity_score(sample_onchain_data)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_calculate_whale_activity_score_high_activity(self, aggregator):
        """Test whale activity score with high activity"""
        onchain_data = OnChainMetrics(
            network_activity={},
            transaction_count_24h=0,
            active_addresses_24h=0,
            transaction_volume_24h=0,
            network_fees_24h=0,
            whale_activity={"large_transactions_24h": 300, "whale_net_flow": 15000.0}  # High activity
        )
        
        score = aggregator._calculate_whale_activity_score(onchain_data)
        
        assert score > 0.7  # Should be high score
    
    def test_calculate_whale_activity_score_negative_flow(self, aggregator):
        """Test whale activity score with negative net flow (distribution)"""
        onchain_data = OnChainMetrics(
            network_activity={},
            transaction_count_24h=0,
            active_addresses_24h=0,
            transaction_volume_24h=0,
            network_fees_24h=0,
            whale_activity={"large_transactions_24h": 100, "whale_net_flow": -5000.0}  # Distribution
        )
        
        score = aggregator._calculate_whale_activity_score(onchain_data)
        
        assert 0 <= score < 0.5  # Should be below neutral for distribution
    
    def test_calculate_whale_activity_score_no_data(self, aggregator):
        """Test whale activity score with no data"""
        score = aggregator._calculate_whale_activity_score(None)
        assert score == 0.5  # Default neutral score
    
    def test_get_default_market_features(self, aggregator):
        """Test default market features generation"""
        features = aggregator._get_default_market_features()
        
        assert isinstance(features, MarketFeatures)
        assert features.fear_greed_index == 50.0
        assert features.market_trend == "sideways"
        assert features.volatility_regime == "medium"
        assert features.btc_correlation == 0.5
        assert features.total_value_locked == 100e9
        assert features.social_score == 0.5
    
    @pytest.mark.asyncio
    async def test_get_sentiment_summary(self, aggregator, sample_sentiment_data, sample_defi_data, 
                                        sample_correlation_data):
        """Test sentiment summary generation"""
        aggregator.fear_greed_client.get_market_data = AsyncMock(return_value=sample_sentiment_data)
        aggregator.defi_client.get_market_data = AsyncMock(return_value=sample_defi_data)
        aggregator.coingecko_client.get_market_data = AsyncMock(return_value=sample_correlation_data)
        aggregator.onchain_client.get_market_data = AsyncMock(return_value=None)
        aggregator.social_client.get_market_data = AsyncMock(return_value=None)
        
        summary = await aggregator.get_sentiment_summary()
        
        assert isinstance(summary, dict)
        assert summary["fear_greed_index"] == 75.0
        assert summary["fear_greed_classification"] == "Greed"
        assert summary["market_trend"] == "bull"
        assert summary["btc_dominance"] == 42.5
        assert summary["defi_tvl"] == 50e9
        assert "timestamp" in summary
    
    @pytest.mark.asyncio
    async def test_get_sentiment_summary_error(self, aggregator):
        """Test sentiment summary with error"""
        aggregator.get_market_features = AsyncMock(side_effect=Exception("Test error"))
        
        summary = await aggregator.get_sentiment_summary()
        
        assert "error" in summary
        assert summary["error"] == "Test error"
        assert "timestamp" in summary
    
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, aggregator):
        """Test health check with all services healthy"""
        # Mock all clients to succeed
        aggregator.fear_greed_client.get_market_data = AsyncMock(return_value=MagicMock())
        aggregator.defi_client.get_market_data = AsyncMock(return_value=MagicMock())
        aggregator.coingecko_client.get_market_data = AsyncMock(return_value=MagicMock())
        aggregator.onchain_client.get_market_data = AsyncMock(return_value=MagicMock())
        aggregator.social_client.get_market_data = AsyncMock(return_value=MagicMock())
        
        health = await aggregator.health_check()
        
        assert isinstance(health, dict)
        assert health["fear_greed"] is True
        assert health["defi_llama"] is True
        assert health["coingecko"] is True
        assert health["onchain"] is True
        assert health["social"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_some_unhealthy(self, aggregator):
        """Test health check with some services unhealthy"""
        # Mock some clients to succeed and others to fail
        aggregator.fear_greed_client.get_market_data = AsyncMock(return_value=MagicMock())
        aggregator.defi_client.get_market_data = AsyncMock(side_effect=Exception("Service down"))
        aggregator.coingecko_client.get_market_data = AsyncMock(return_value=MagicMock())
        aggregator.onchain_client.get_market_data = AsyncMock(side_effect=Exception("Service down"))
        aggregator.social_client.get_market_data = AsyncMock(return_value=MagicMock())
        
        health = await aggregator.health_check()
        
        assert health["fear_greed"] is True
        assert health["defi_llama"] is False
        assert health["coingecko"] is True
        assert health["onchain"] is False
        assert health["social"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self, aggregator):
        """Test health check with timeout"""
        # Mock client to hang
        async def hanging_call():
            await asyncio.sleep(15)  # Longer than 10s timeout
            return MagicMock()
        
        aggregator.fear_greed_client.get_market_data = hanging_call
        aggregator.defi_client.get_market_data = AsyncMock(return_value=MagicMock())
        aggregator.coingecko_client.get_market_data = AsyncMock(return_value=MagicMock())
        aggregator.onchain_client.get_market_data = AsyncMock(return_value=MagicMock())
        aggregator.social_client.get_market_data = AsyncMock(return_value=MagicMock())
        
        health = await aggregator.health_check()
        
        assert health["fear_greed"] is False  # Should timeout
        assert health["defi_llama"] is True
    
    def test_clear_cache(self, aggregator):
        """Test cache clearing"""
        # Set up cache
        aggregator._aggregated_cache = MarketFeatures()
        aggregator._cache_timestamp = datetime.now()
        
        # Mock individual client caches
        for client in [aggregator.fear_greed_client, aggregator.defi_client]:
            client._cache = {"test": "data"}
            client._cache_timestamps = {"test": datetime.now()}
        
        aggregator.clear_cache()
        
        # Check aggregator cache cleared
        assert aggregator._aggregated_cache is None
        assert aggregator._cache_timestamp is None
        
        # Check individual client caches cleared
        assert len(aggregator.fear_greed_client._cache) == 0
        assert len(aggregator.defi_client._cache) == 0
    
    @pytest.mark.asyncio
    async def test_close(self, aggregator):
        """Test resource cleanup"""
        # Mock client close methods
        aggregator.fear_greed_client.close = AsyncMock()
        aggregator.defi_client.close = AsyncMock()
        aggregator.coingecko_client.close = AsyncMock()
        aggregator.onchain_client.close = AsyncMock()
        aggregator.social_client.close = AsyncMock()
        
        await aggregator.close()
        
        # Verify all clients were closed
        aggregator.fear_greed_client.close.assert_called_once()
        aggregator.defi_client.close.assert_called_once()
        aggregator.coingecko_client.close.assert_called_once()
        aggregator.onchain_client.close.assert_called_once()
        aggregator.social_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager(self, aggregator):
        """Test async context manager usage"""
        aggregator.close = AsyncMock()
        
        async with aggregator as agg:
            assert agg is aggregator
        
        aggregator.close.assert_called_once()
    
    def test_is_cache_valid_valid(self, aggregator):
        """Test cache validation with valid cache"""
        aggregator._cache_timestamp = datetime.now()
        assert aggregator._is_cache_valid()
    
    def test_is_cache_valid_expired(self, aggregator):
        """Test cache validation with expired cache"""
        aggregator.cache_ttl = 60
        aggregator._cache_timestamp = datetime.now() - timedelta(seconds=120)
        assert not aggregator._is_cache_valid()
    
    def test_is_cache_valid_no_timestamp(self, aggregator):
        """Test cache validation with no timestamp"""
        assert not aggregator._is_cache_valid()