"""
Unit tests for DEX base classes and data structures.

Tests the core DEX interfaces, configuration, and data structures
that provide the foundation for all DEX implementations.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

from src.dex.base import (
    DEXBase,
    DEXConfig,
    SwapQuote,
    SwapResult,
    SwapStatus,
    SwapType,
    PriceImpact,
    DEXError,
    DEXConnectionError,
    DEXTransactionError,
    DEXRateLimitError,
)
from src.utils.base import Chain


class MockDEX(DEXBase):
    """Mock DEX implementation for testing base functionality."""
    
    async def connect(self) -> bool:
        self._connected = True
        return True
    
    async def disconnect(self) -> None:
        self._connected = False
    
    async def get_quote(self, input_token, output_token, amount, swap_type=SwapType.EXACT_INPUT, slippage_bps=None):
        return SwapQuote(
            input_token=input_token,
            output_token=output_token,
            input_amount=amount,
            output_amount=amount * Decimal('1.1'),
            price=Decimal('1.1'),
            price_impact_bps=50,
            slippage_bps=slippage_bps or 50,
            dex_name=self.name
        )
    
    async def execute_swap(self, quote, wallet_address, signature_data=None):
        return SwapResult(
            transaction_hash="test_hash",
            status=SwapStatus.PENDING,
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            dex_name=self.name
        )
    
    async def get_token_price(self, token_address, base_token=None):
        return Decimal('1.0')
    
    async def get_supported_tokens(self):
        return [{"address": "token1", "symbol": "TEST1"}]
    
    async def estimate_gas(self, input_token, output_token, amount, wallet_address):
        return 100000
    
    async def get_transaction_status(self, transaction_hash):
        return SwapResult(
            transaction_hash=transaction_hash,
            status=SwapStatus.CONFIRMED,
            input_token="token1",
            output_token="token2",
            input_amount=Decimal('100'),
            dex_name=self.name
        )


class TestDEXConfig:
    """Test DEX configuration class."""
    
    def test_valid_config(self):
        """Test creating valid DEX configuration."""
        config = DEXConfig(
            chain=Chain.SOLANA,
            name="test_dex",
            max_slippage_bps=100,
            max_price_impact_bps=500
        )
        
        assert config.chain == Chain.SOLANA
        assert config.name == "test_dex"
        assert config.max_slippage_bps == 100
        assert config.max_price_impact_bps == 500
        assert config.timeout_seconds == 30  # Default value
    
    def test_invalid_slippage(self):
        """Test invalid slippage configuration."""
        with pytest.raises(ValueError, match="max_slippage_bps must be between 0 and 10000"):
            DEXConfig(
                chain=Chain.SOLANA,
                name="test_dex",
                max_slippage_bps=-1
            )
        
        with pytest.raises(ValueError, match="max_slippage_bps must be between 0 and 10000"):
            DEXConfig(
                chain=Chain.SOLANA,
                name="test_dex",
                max_slippage_bps=10001
            )
    
    def test_invalid_price_impact(self):
        """Test invalid price impact configuration."""
        with pytest.raises(ValueError, match="max_price_impact_bps must be between 0 and 10000"):
            DEXConfig(
                chain=Chain.SOLANA,
                name="test_dex",
                max_price_impact_bps=-1
            )


class TestSwapQuote:
    """Test SwapQuote data structure."""
    
    def test_basic_quote(self):
        """Test basic quote creation and properties."""
        quote = SwapQuote(
            input_token="token1",
            output_token="token2", 
            input_amount=Decimal('100'),
            output_amount=Decimal('110'),
            price=Decimal('1.1'),
            price_impact_bps=50,
            slippage_bps=100,
            dex_name="test_dex"
        )
        
        assert quote.input_token == "token1"
        assert quote.output_token == "token2"
        assert quote.input_amount == Decimal('100')
        assert quote.output_amount == Decimal('110')
        assert quote.price == Decimal('1.1')
        assert quote.price_impact_bps == 50
        assert quote.slippage_bps == 100
    
    def test_price_impact_levels(self):
        """Test price impact severity classification."""
        # Low impact (< 1%)
        quote_low = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=50, slippage_bps=50
        )
        assert quote_low.price_impact == PriceImpact.LOW
        
        # Medium impact (1-3%)
        quote_medium = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=200, slippage_bps=50
        )
        assert quote_medium.price_impact == PriceImpact.MEDIUM
        
        # High impact (3-10%)
        quote_high = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=500, slippage_bps=50
        )
        assert quote_high.price_impact == PriceImpact.HIGH
        
        # Extreme impact (> 10%)
        quote_extreme = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=1500, slippage_bps=50
        )
        assert quote_extreme.price_impact == PriceImpact.EXTREME
    
    def test_effective_price(self):
        """Test effective price calculation with slippage."""
        quote = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1.0'), price_impact_bps=0,
            slippage_bps=100  # 1% slippage
        )
        
        # Effective price should be 1.0 * (1 - 0.01) = 0.99
        expected_effective_price = Decimal('0.99')
        assert quote.effective_price == expected_effective_price
    
    def test_quote_expiry(self):
        """Test quote expiry functionality."""
        # Quote with future expiry
        future_expiry = datetime.now() + timedelta(minutes=5)
        quote_valid = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=0, slippage_bps=50,
            valid_until=future_expiry
        )
        assert not quote_valid.is_expired
        
        # Quote with past expiry
        past_expiry = datetime.now() - timedelta(minutes=5)
        quote_expired = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=0, slippage_bps=50,
            valid_until=past_expiry
        )
        assert quote_expired.is_expired
        
        # Quote without expiry
        quote_no_expiry = SwapQuote(
            input_token="token1", output_token="token2", 
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=0, slippage_bps=50
        )
        assert not quote_no_expiry.is_expired


class TestSwapResult:
    """Test SwapResult data structure."""
    
    def test_successful_swap(self):
        """Test successful swap result."""
        result = SwapResult(
            transaction_hash="0x123",
            status=SwapStatus.CONFIRMED,
            input_token="token1",
            output_token="token2", 
            input_amount=Decimal('100'),
            actual_output_amount=Decimal('110'),
            gas_used=21000,
            gas_price=Decimal('20'),
            dex_name="test_dex"
        )
        
        assert result.is_successful
        assert result.transaction_fee == Decimal('420000')  # 21000 * 20
        assert result.actual_price == Decimal('1.1')  # 110 / 100
    
    def test_failed_swap(self):
        """Test failed swap result."""
        result = SwapResult(
            transaction_hash="0x456",
            status=SwapStatus.FAILED,
            input_token="token1",
            output_token="token2",
            input_amount=Decimal('100'),
            error_message="Insufficient liquidity"
        )
        
        assert not result.is_successful
        assert result.transaction_fee is None
        assert result.actual_price is None
    
    def test_slippage_calculation(self):
        """Test slippage calculation vs quote."""
        quote = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('110'),
            price=Decimal('1.1'), price_impact_bps=0, slippage_bps=50
        )
        
        result = SwapResult(
            transaction_hash="0x789",
            status=SwapStatus.CONFIRMED,
            input_token="token1",
            output_token="token2",
            input_amount=Decimal('100'),
            actual_output_amount=Decimal('108'),  # Less than expected 110
            quote_used=quote
        )
        
        # Slippage = (110 - 108) / 110 = 0.0181... ≈ 1.82%
        expected_slippage = Decimal('2') / Decimal('110')  # (110-108)/110
        assert abs(result.slippage_vs_quote - expected_slippage) < Decimal('0.001')


class TestDEXBase:
    """Test DEX base class functionality."""
    
    @pytest.fixture
    def mock_dex(self):
        """Create mock DEX for testing."""
        config = DEXConfig(chain=Chain.SOLANA, name="mock_dex")
        return MockDEX(config)
    
    def test_initialization(self, mock_dex):
        """Test DEX initialization."""
        assert mock_dex.chain == Chain.SOLANA
        assert mock_dex.name == "mock_dex"
        assert not mock_dex.is_connected
    
    @pytest.mark.asyncio
    async def test_connection(self, mock_dex):
        """Test DEX connection."""
        assert not mock_dex.is_connected
        
        success = await mock_dex.connect()
        assert success
        assert mock_dex.is_connected
        
        await mock_dex.disconnect()
        assert not mock_dex.is_connected
    
    @pytest.mark.asyncio
    async def test_quote_validation(self, mock_dex):
        """Test quote validation."""
        # Valid quote
        valid_quote = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=50, slippage_bps=30,
            valid_until=datetime.now() + timedelta(minutes=5)
        )
        assert await mock_dex.validate_quote(valid_quote)
        
        # Expired quote
        expired_quote = SwapQuote(
            input_token="token1", output_token="token2", 
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=50, slippage_bps=30,
            valid_until=datetime.now() - timedelta(minutes=5)
        )
        assert not await mock_dex.validate_quote(expired_quote)
        
        # High price impact quote
        high_impact_quote = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=2000, slippage_bps=30  # 20% impact
        )
        assert not await mock_dex.validate_quote(high_impact_quote)
        
        # High slippage quote
        high_slippage_quote = SwapQuote(
            input_token="token1", output_token="token2",
            input_amount=Decimal('100'), output_amount=Decimal('100'),
            price=Decimal('1'), price_impact_bps=50, slippage_bps=200  # 2% slippage
        )
        assert not await mock_dex.validate_quote(high_slippage_quote)
    
    @pytest.mark.asyncio  
    async def test_health_check(self, mock_dex):
        """Test DEX health check."""
        health = await mock_dex.health_check()
        
        assert health["connected"]
        assert health["dex_name"] == "mock_dex"
        assert health["chain"] == "solana"
        assert health["status"] == "healthy"
        assert "supported_tokens" in health
    
    def test_string_representation(self, mock_dex):
        """Test string representation methods."""
        str_repr = str(mock_dex)
        assert "MockDEX" in str_repr
        assert "mock_dex" in str_repr
        assert "solana" in str_repr
        
        repr_str = repr(mock_dex)
        assert "MockDEX" in repr_str
        assert "name=mock_dex" in repr_str
        assert "chain=solana" in repr_str
        assert "connected=" in repr_str