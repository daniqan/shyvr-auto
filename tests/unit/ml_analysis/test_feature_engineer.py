"""
Unit tests for feature engineering module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from src.utils.base import Chain
from src.discovery.base import DiscoveredToken, TokenStatus
from src.ml_analysis.feature_engineer import FeatureEngineer
from src.ml_analysis.base import TechnicalIndicators, MarketFeatures, FeatureEngineeringError
from unittest.mock import AsyncMock, patch


class TestFeatureEngineer:
    """Test FeatureEngineer class"""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer for testing"""
        return FeatureEngineer(cache_ttl_minutes=5, enable_live_data=False)  # Disable live data for tests
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token for testing"""
        return DiscoveredToken(
            address="0x123456789abcdef",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=100.0,
            market_cap=1000000.0,
            volume_24h=50000.0
        )
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range(start='2025-01-01', periods=100, freq='1H')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible tests
        price_base = 100.0
        returns = np.random.normal(0, 0.02, 100)  # 2% volatility
        prices = [price_base]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        volumes = np.random.uniform(1000, 5000, 100)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],  # Slightly higher highs
            'low': [p * 0.99 for p in prices],   # Slightly lower lows
            'close': prices,
            'volume': volumes
        })
    
    def test_initialization(self, feature_engineer):
        """Test feature engineer initialization"""
        assert feature_engineer.cache_ttl_minutes == 5
        assert len(feature_engineer._indicator_cache) == 0
        assert feature_engineer._market_cache is None
    
    @pytest.mark.asyncio
    async def test_calculate_technical_indicators_success(self, feature_engineer, sample_token, sample_price_data):
        """Test successful technical indicator calculation"""
        indicators = await feature_engineer.calculate_technical_indicators(sample_token, sample_price_data)
        
        assert isinstance(indicators, TechnicalIndicators)
        
        # Check that indicators were calculated
        assert indicators.sma_20 is not None
        assert indicators.sma_50 is not None
        assert indicators.ema_12 is not None
        assert indicators.ema_26 is not None
        assert indicators.rsi is not None
        assert indicators.macd is not None
        assert indicators.volume_ratio is not None
        
        # Verify RSI is in valid range
        assert 0 <= indicators.rsi <= 100
        
        # Verify volume ratio is positive
        assert indicators.volume_ratio > 0
    
    @pytest.mark.asyncio
    async def test_calculate_technical_indicators_insufficient_data(self, feature_engineer, sample_token):
        """Test technical indicators with insufficient data"""
        # Create minimal price data
        short_data = pd.DataFrame({
            'close': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'volume': [1000, 1100, 1200]
        })
        
        indicators = await feature_engineer.calculate_technical_indicators(sample_token, short_data)
        
        # Should return default indicators
        assert isinstance(indicators, TechnicalIndicators)
        assert indicators.rsi == 50.0  # Default RSI
        assert indicators.volume_ratio == 1.0  # Default volume ratio
    
    @pytest.mark.asyncio
    async def test_calculate_technical_indicators_missing_columns(self, feature_engineer, sample_token):
        """Test technical indicators with missing required columns"""
        invalid_data = pd.DataFrame({
            'price': [100, 101, 102],  # Wrong column name
            'vol': [1000, 1100, 1200]  # Wrong column name
        })
        
        with pytest.raises(FeatureEngineeringError):
            await feature_engineer.calculate_technical_indicators(sample_token, invalid_data)
    
    @pytest.mark.asyncio
    async def test_calculate_technical_indicators_caching(self, feature_engineer, sample_token, sample_price_data):
        """Test technical indicators caching"""
        # First call
        indicators1 = await feature_engineer.calculate_technical_indicators(sample_token, sample_price_data)
        
        # Second call should return cached result
        indicators2 = await feature_engineer.calculate_technical_indicators(sample_token, sample_price_data)
        
        # Should be the same object (cached)
        assert indicators1 is indicators2
        
        # Cache should contain the result
        cache_key = f"{sample_token.chain_address}"
        assert cache_key in feature_engineer._indicator_cache
    
    @pytest.mark.asyncio
    async def test_calculate_market_features(self, feature_engineer):
        """Test market features calculation"""
        features = await feature_engineer.calculate_market_features()
        
        assert isinstance(features, MarketFeatures)
        
        # Check placeholder values are set (since live_data=False)
        assert features.fear_greed_index == 50.0
        assert features.fear_greed_classification == "Neutral"
        assert features.market_trend == "sideways"
        assert features.volatility_regime == "medium"
        assert features.btc_correlation == 0.5
        assert features.eth_correlation == 0.4
        assert features.btc_dominance == 40.0
        assert features.eth_dominance == 15.0
        assert features.stablecoin_dominance == 10.0
        assert features.market_beta == 1.0
        assert features.total_value_locked == 100e9
        assert features.tvl_change_24h == 0.0
        assert features.active_protocols == 300
        assert features.social_score == 0.5
        assert features.mention_volume == 1000
        assert features.sentiment_trend == 0.0
        assert features.influencer_sentiment == 0.5
    
    @pytest.mark.asyncio
    async def test_calculate_market_features_caching(self, feature_engineer):
        """Test market features caching"""
        # First call
        features1 = await feature_engineer.calculate_market_features()
        
        # Second call should return cached result
        features2 = await feature_engineer.calculate_market_features()
        
        # Should be the same object (cached)
        assert features1 is features2
        assert feature_engineer._market_cache is not None
    
    def test_calculate_sma(self, feature_engineer):
        """Test Simple Moving Average calculation"""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        sma_5 = feature_engineer._calculate_sma(series, 5)
        assert sma_5 == 8.0  # Average of [6, 7, 8, 9, 10]
        
        # Insufficient data
        short_series = pd.Series([1, 2, 3])
        sma_5_short = feature_engineer._calculate_sma(short_series, 5)
        assert sma_5_short is None
    
    def test_calculate_ema(self, feature_engineer):
        """Test Exponential Moving Average calculation"""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        ema_5 = feature_engineer._calculate_ema(series, 5)
        assert ema_5 is not None
        assert isinstance(ema_5, float)
        
        # EMA should be closer to recent values than SMA
        sma_5 = feature_engineer._calculate_sma(series, 5)
        assert ema_5 > sma_5  # For increasing series
    
    def test_calculate_rsi(self, feature_engineer):
        """Test Relative Strength Index calculation"""
        # Create series with clear trend
        increasing_series = pd.Series([50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80])
        decreasing_series = pd.Series([80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50])
        
        rsi_up = feature_engineer._calculate_rsi(increasing_series)
        rsi_down = feature_engineer._calculate_rsi(decreasing_series)
        
        # RSI should be in valid range
        assert 0 <= rsi_up <= 100
        assert 0 <= rsi_down <= 100
        
        # Increasing prices should have higher RSI
        assert rsi_up > rsi_down
        
        # Insufficient data
        short_series = pd.Series([1, 2, 3])
        rsi_short = feature_engineer._calculate_rsi(short_series)
        assert rsi_short == 50.0  # Default value
    
    def test_calculate_macd(self, feature_engineer):
        """Test MACD calculation"""
        series = pd.Series(range(1, 51))  # Increasing trend
        
        macd_result = feature_engineer._calculate_macd(series)
        
        assert isinstance(macd_result, dict)
        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result
        
        # All values should be floats
        assert isinstance(macd_result['macd'], float)
        assert isinstance(macd_result['signal'], float)
        assert isinstance(macd_result['histogram'], float)
        
        # For increasing series, MACD should be positive
        assert macd_result['macd'] > 0
    
    def test_calculate_bollinger_bands(self, feature_engineer):
        """Test Bollinger Bands calculation"""
        series = pd.Series([100] * 10 + [110] * 10 + [90] * 10)  # Some volatility
        
        bollinger = feature_engineer._calculate_bollinger_bands(series)
        
        assert isinstance(bollinger, dict)
        assert 'upper' in bollinger
        assert 'lower' in bollinger
        assert 'width' in bollinger
        
        if bollinger['upper'] and bollinger['lower']:
            # Upper should be greater than lower
            assert bollinger['upper'] > bollinger['lower']
            
            # Width should be positive
            assert bollinger['width'] > 0
    
    def test_calculate_atr(self, feature_engineer):
        """Test Average True Range calculation"""
        # Create price data with some volatility
        price_data = pd.DataFrame({
            'high': [102, 104, 103, 105, 107],
            'low': [98, 96, 97, 99, 101],
            'close': [100, 102, 101, 103, 105]
        })
        
        atr = feature_engineer._calculate_atr(price_data)
        
        if atr is not None:
            assert isinstance(atr, float)
            assert atr > 0  # ATR should be positive
    
    def test_calculate_volume_ratio(self, feature_engineer):
        """Test volume ratio calculation"""
        volume_series = pd.Series([1000] * 19 + [2000])  # Last volume is 2x average
        
        ratio = feature_engineer._calculate_volume_ratio(volume_series)
        
        assert isinstance(ratio, float)
        assert ratio == 2.0  # Should be exactly 2x
        
        # Test with insufficient data
        short_series = pd.Series([1000, 1100])
        ratio_short = feature_engineer._calculate_volume_ratio(short_series)
        assert ratio_short == 1.0  # Default
    
    def test_calculate_obv(self, feature_engineer):
        """Test On-Balance Volume calculation"""
        close_series = pd.Series([100, 102, 101, 103, 105])  # Mixed price movement
        volume_series = pd.Series([1000, 1500, 1200, 1800, 2000])
        
        obv = feature_engineer._calculate_obv(close_series, volume_series)
        
        assert isinstance(obv, float)
        # OBV calculation should complete without error
    
    def test_calculate_price_momentum(self, feature_engineer):
        """Test price momentum calculation"""
        increasing_series = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        decreasing_series = pd.Series([110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100])
        
        momentum_up = feature_engineer._calculate_price_momentum(increasing_series)
        momentum_down = feature_engineer._calculate_price_momentum(decreasing_series)
        
        assert isinstance(momentum_up, float)
        assert isinstance(momentum_down, float)
        
        # Increasing series should have positive momentum
        assert momentum_up > 0
        # Decreasing series should have negative momentum
        assert momentum_down < 0
    
    def test_calculate_volatility_score(self, feature_engineer):
        """Test volatility score calculation"""
        # Low volatility series
        stable_series = pd.Series([100] * 25)
        
        # High volatility series
        volatile_series = pd.Series([100, 110, 90, 120, 80, 115, 95, 105, 85, 125] * 3)
        
        vol_stable = feature_engineer._calculate_volatility_score(stable_series)
        vol_volatile = feature_engineer._calculate_volatility_score(volatile_series)
        
        assert isinstance(vol_stable, float)
        assert isinstance(vol_volatile, float)
        
        # Volatility scores should be in 0-1 range
        assert 0 <= vol_stable <= 1
        assert 0 <= vol_volatile <= 1
        
        # Volatile series should have higher score
        assert vol_volatile > vol_stable
    
    @pytest.mark.asyncio
    async def test_create_feature_matrix(self, feature_engineer, sample_price_data):
        """Test feature matrix creation"""
        tokens = [
            DiscoveredToken(
                address="0x123",
                chain=Chain.ETHEREUM,
                symbol="TEST1",
                name="Test Token 1",
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=100.0,
                market_cap=1000000.0
            ),
            DiscoveredToken(
                address="0x456",
                chain=Chain.SOLANA,
                symbol="TEST2",
                name="Test Token 2",
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=50.0,
                market_cap=500000.0
            )
        ]
        
        price_data = {
            "0x123": sample_price_data,
            "0x456": sample_price_data.copy()
        }
        
        feature_matrix, feature_names = await feature_engineer.create_feature_matrix(tokens, price_data)
        
        assert isinstance(feature_matrix, np.ndarray)
        assert isinstance(feature_names, list)
        
        # Should have 2 rows (one per token)
        assert feature_matrix.shape[0] == 2
        
        # Should have many features
        assert feature_matrix.shape[1] > 20
        
        # Feature names should match matrix columns
        assert len(feature_names) == feature_matrix.shape[1]
    
    @pytest.mark.asyncio
    async def test_create_feature_matrix_missing_data(self, feature_engineer):
        """Test feature matrix creation with missing price data"""
        tokens = [
            DiscoveredToken(
                address="0x123",
                chain=Chain.ETHEREUM,
                symbol="TEST1",
                name="Test Token 1",
                discovered_at=datetime.now(),
                discovery_source="test"
            )
        ]
        
        # Empty price data
        price_data = {}
        
        with pytest.raises(FeatureEngineeringError):
            await feature_engineer.create_feature_matrix(tokens, price_data)
    
    def test_extract_token_features(self, feature_engineer):
        """Test token-specific feature extraction"""
        token = DiscoveredToken(
            address="0x123",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now() - timedelta(days=30),
            discovery_source="test",
            price_usd=100.0,
            market_cap=1000000.0,
            volume_24h=50000.0,
            price_change_24h=5.0,
            tags=["verified", "trending"]
        )
        
        features = feature_engineer._extract_token_features(token)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) == 9  # Expected number of token features
        
        # Check specific features
        assert features[0] > 0  # log_price should be positive for price > 1
        assert features[4] == 1.0  # is_ethereum should be 1
        assert features[5] == 0.0  # is_solana should be 0
        assert features[7] == 30.0  # token_age_days should be 30
    
    def test_calculate_token_score(self, feature_engineer):
        """Test token quality score calculation"""
        # High quality token
        good_token = DiscoveredToken(
            address="0x123",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=100.0,
            market_cap=1000000.0,
            volume_24h=100000.0,  # Good volume/market cap ratio
            price_change_24h=2.0,  # Stable price change
            tags=["verified", "trending"]
        )
        
        # Low quality token
        poor_token = DiscoveredToken(
            address="0x456",
            chain=Chain.ETHEREUM,
            symbol="POOR",
            name="Poor Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=0.001,
            market_cap=1000.0,
            volume_24h=10.0,      # Poor volume/market cap ratio
            price_change_24h=-50.0,  # Highly volatile
            tags=[]
        )
        
        good_score = feature_engineer._calculate_token_score(good_token)
        poor_score = feature_engineer._calculate_token_score(poor_token)
        
        assert isinstance(good_score, float)
        assert isinstance(poor_score, float)
        
        # Scores should be in 0-1 range
        assert 0 <= good_score <= 1
        assert 0 <= poor_score <= 1
        
        # Good token should have higher score
        assert good_score > poor_score
    
    def test_get_feature_names(self, feature_engineer):
        """Test feature names generation"""
        feature_names = feature_engineer._get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 30  # Should have many features
        
        # Check for expected feature categories
        tech_features = [name for name in feature_names if any(
            tech in name for tech in ['sma', 'ema', 'rsi', 'macd', 'volume', 'bollinger']
        )]
        market_features = [name for name in feature_names if any(
            market in name for market in ['fear_greed', 'market', 'correlation', 'social']
        )]
        token_features = [name for name in feature_names if any(
            token in name for token in ['log_price', 'is_ethereum', 'token_age']
        )]
        
        assert len(tech_features) > 0
        assert len(market_features) > 0
        assert len(token_features) > 0
    
    def test_clear_cache(self, feature_engineer):
        """Test cache clearing"""
        # Add something to cache
        feature_engineer._indicator_cache["test"] = (datetime.now(), TechnicalIndicators())
        feature_engineer._market_cache = (datetime.now(), MarketFeatures())
        
        assert len(feature_engineer._indicator_cache) == 1
        assert feature_engineer._market_cache is not None
        
        # Clear cache
        feature_engineer.clear_cache()
        
        assert len(feature_engineer._indicator_cache) == 0
        assert feature_engineer._market_cache is None


class TestFeatureEngineerLiveData:
    """Test FeatureEngineer with live data enabled"""
    
    @pytest.fixture
    def live_feature_engineer(self):
        """Create feature engineer with live data enabled"""
        return FeatureEngineer(
            cache_ttl_minutes=5, 
            coingecko_api_key="test_key",
            enable_live_data=True
        )
    
    @pytest.mark.asyncio
    async def test_calculate_market_features_live_data(self, live_feature_engineer):
        """Test market features calculation with live data"""
        # Mock the market aggregator
        mock_features = MarketFeatures(
            fear_greed_index=65.0,
            fear_greed_classification="Greed",
            market_trend="bull",
            volatility_regime="medium",
            btc_dominance=42.0,
            eth_dominance=18.0,
            total_value_locked=80e9,
            social_score=0.7
        )
        
        live_feature_engineer.market_aggregator.get_market_features = AsyncMock(return_value=mock_features)
        
        result = await live_feature_engineer.calculate_market_features()
        
        assert isinstance(result, MarketFeatures)
        assert result.fear_greed_index == 65.0
        assert result.fear_greed_classification == "Greed"
        assert result.market_trend == "bull"
        assert result.btc_dominance == 42.0
        assert result.total_value_locked == 80e9
        assert result.social_score == 0.7
    
    @pytest.mark.asyncio
    async def test_calculate_market_features_live_data_fallback(self, live_feature_engineer):
        """Test market features calculation with live data API failure fallback"""
        # Mock the market aggregator to fail
        live_feature_engineer.market_aggregator.get_market_features = AsyncMock(
            side_effect=Exception("API error")
        )
        
        result = await live_feature_engineer.calculate_market_features()
        
        # Should fallback to placeholder values
        assert isinstance(result, MarketFeatures)
        assert result.fear_greed_index == 50.0
        assert result.market_trend == "sideways"
        assert result.total_value_locked == 100e9
    
    @pytest.mark.asyncio
    async def test_calculate_market_features_with_chain_param(self, live_feature_engineer):
        """Test market features calculation with specific chain parameter"""
        mock_features = MarketFeatures(fear_greed_index=60.0)
        live_feature_engineer.market_aggregator.get_market_features = AsyncMock(return_value=mock_features)
        
        result = await live_feature_engineer.calculate_market_features(primary_chain="solana")
        
        # Verify the chain parameter was passed
        live_feature_engineer.market_aggregator.get_market_features.assert_called_once()
        call_args = live_feature_engineer.market_aggregator.get_market_features.call_args
        assert call_args[1]["primary_chain"].value == "solana"
    
    @pytest.mark.asyncio
    async def test_calculate_market_features_invalid_chain(self, live_feature_engineer):
        """Test market features calculation with invalid chain parameter"""
        mock_features = MarketFeatures(fear_greed_index=60.0)
        live_feature_engineer.market_aggregator.get_market_features = AsyncMock(return_value=mock_features)
        
        result = await live_feature_engineer.calculate_market_features(primary_chain="invalid_chain")
        
        # Should default to Ethereum
        call_args = live_feature_engineer.market_aggregator.get_market_features.call_args
        assert call_args[1]["primary_chain"].value == "ethereum"
    
    @pytest.mark.asyncio
    async def test_health_check(self, live_feature_engineer):
        """Test health check functionality"""
        # Mock market aggregator health check
        mock_health = {
            "fear_greed": True,
            "defi_llama": True,
            "coingecko": False,
            "onchain": True,
            "social": True
        }
        live_feature_engineer.market_aggregator.health_check = AsyncMock(return_value=mock_health)
        
        health = await live_feature_engineer.health_check()
        
        assert isinstance(health, dict)
        assert "cache_size" in health
        assert "market_cache_valid" in health
        assert "live_data_enabled" in health
        assert health["live_data_enabled"] is True
        assert "market_data_sources" in health
        assert health["market_data_sources"] == mock_health
        assert health["market_data_healthy"] is False  # One service is down
    
    @pytest.mark.asyncio
    async def test_health_check_aggregator_error(self, live_feature_engineer):
        """Test health check with aggregator error"""
        live_feature_engineer.market_aggregator.health_check = AsyncMock(
            side_effect=Exception("Health check failed")
        )
        
        health = await live_feature_engineer.health_check()
        
        assert "market_data_error" in health
        assert health["market_data_error"] == "Health check failed"
        assert health["market_data_healthy"] is False
    
    @pytest.mark.asyncio
    async def test_close(self, live_feature_engineer):
        """Test resource cleanup"""
        live_feature_engineer.market_aggregator.close = AsyncMock()
        
        await live_feature_engineer.close()
        
        live_feature_engineer.market_aggregator.close.assert_called_once()
    
    def test_clear_cache_with_aggregator(self, live_feature_engineer):
        """Test cache clearing with market aggregator"""
        from unittest.mock import MagicMock
        
        # Mock aggregator clear_cache
        live_feature_engineer.market_aggregator.clear_cache = MagicMock()
        
        # Add something to cache
        live_feature_engineer._indicator_cache["test"] = (datetime.now(), TechnicalIndicators())
        live_feature_engineer._market_cache = (datetime.now(), MarketFeatures())
        
        live_feature_engineer.clear_cache()
        
        # Verify all caches were cleared
        assert len(live_feature_engineer._indicator_cache) == 0
        assert live_feature_engineer._market_cache is None
        live_feature_engineer.market_aggregator.clear_cache.assert_called_once()


class TestFeatureEngineerDisabledLiveData:
    """Test FeatureEngineer with live data disabled"""
    
    @pytest.fixture
    def disabled_feature_engineer(self):
        """Create feature engineer with live data disabled"""
        return FeatureEngineer(enable_live_data=False)
    
    @pytest.mark.asyncio
    async def test_health_check_disabled(self, disabled_feature_engineer):
        """Test health check with live data disabled"""
        health = await disabled_feature_engineer.health_check()
        
        assert health["live_data_enabled"] is False
        assert health["market_data_healthy"] is True  # Should be True in placeholder mode
        assert "market_data_sources" in health
    
    @pytest.mark.asyncio
    async def test_close_disabled(self, disabled_feature_engineer):
        """Test close with live data disabled"""
        # Should not raise an error
        await disabled_feature_engineer.close()
    
    def test_clear_cache_disabled(self, disabled_feature_engineer):
        """Test cache clearing with live data disabled"""
        disabled_feature_engineer._indicator_cache["test"] = (datetime.now(), TechnicalIndicators())
        disabled_feature_engineer._market_cache = (datetime.now(), MarketFeatures())
        
        # Should not raise an error
        disabled_feature_engineer.clear_cache()
        
        assert len(disabled_feature_engineer._indicator_cache) == 0
        assert disabled_feature_engineer._market_cache is None