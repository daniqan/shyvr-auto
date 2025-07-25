"""
Tests for P&L Calculator functionality.

Following TDD methodology - these tests define the expected behavior
for P&L calculations across different DEXs, position types, and time periods.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.portfolio.base import (
    Position,
    PositionType,
    PositionStatus,
    Transaction,
    TransactionType,
    Portfolio,
    PortfolioConfig,
    PerformanceMetrics,
)
from src.portfolio.pnl_calculator import (
    PnLCalculator,
    PnLCalculationResult,
    PnLAttribution,
    PnLTimeSeriesPoint,
    PnLSummary,
    PnLAggregationPeriod,
    PnLCalculationError,
)
from src.utils.base import Chain


class TestPnLCalculator:
    """Test P&L Calculator functionality."""
    
    @pytest.fixture
    def config(self):
        """Test portfolio configuration."""
        return PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC",
            stop_loss_pct=Decimal("0.08"),
            take_profit_pct=Decimal("0.4")
        )
    
    @pytest.fixture
    def portfolio(self, config):
        """Test portfolio instance."""
        return Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            config=config,
            cash_balance=Decimal("10000"),
            total_value=Decimal("10000")
        )
    
    @pytest.fixture
    def pnl_calculator(self, portfolio):
        """Test P&L calculator instance."""
        return PnLCalculator(portfolio)
    
    @pytest.fixture
    def spot_position(self):
        """Test spot position on Jupiter (Solana)."""
        return Position(
            position_id=uuid4(),
            symbol="SOL/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("100"),
            entry_price=Decimal("80"),
            current_price=Decimal("85"),
            status=PositionStatus.OPEN,
            created_at=datetime.now() - timedelta(hours=2)
        )
    
    @pytest.fixture
    def perpetual_position(self):
        """Test perpetual position on Hyperliquid."""
        return Position(
            position_id=uuid4(),
            symbol="BTC-USD",
            position_type=PositionType.PERPETUAL,
            chain=Chain.HYPERLIQUID,
            dex_name="hyperliquid",
            size=Decimal("0.5"),
            entry_price=Decimal("45000"),
            current_price=Decimal("46000"),
            status=PositionStatus.OPEN,
            leverage=Decimal("5"),
            side="LONG",
            unrealized_funding=Decimal("2.5"),
            created_at=datetime.now() - timedelta(hours=4)
        )
    
    @pytest.fixture
    def closed_position(self):
        """Test closed position for historical P&L."""
        return Position(
            position_id=uuid4(),
            symbol="ETH/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2100"),
            status=PositionStatus.CLOSED,
            created_at=datetime.now() - timedelta(days=2)
        )
    
    @pytest.fixture
    def buy_transaction(self, spot_position):
        """Test buy transaction."""
        return Transaction(
            transaction_id=uuid4(),
            position_id=spot_position.position_id,
            transaction_type=TransactionType.BUY,
            symbol="SOL/USDC",
            size=Decimal("100"),
            price=Decimal("80"),
            timestamp=datetime.now() - timedelta(hours=2),
            chain=Chain.SOLANA,
            dex_name="jupiter",
            transaction_hash="0xabcd1234",
            fees=Decimal("2.0"),
            gas_fees=Decimal("0.5")
        )
    
    @pytest.fixture
    def sell_transaction(self, spot_position):
        """Test sell transaction."""
        return Transaction(
            transaction_id=uuid4(),
            position_id=spot_position.position_id,
            transaction_type=TransactionType.SELL,
            symbol="SOL/USDC",
            size=Decimal("50"),
            price=Decimal("85"),
            timestamp=datetime.now() - timedelta(hours=1),
            chain=Chain.SOLANA,
            dex_name="jupiter",
            transaction_hash="0xefgh5678",
            fees=Decimal("1.5"),
            gas_fees=Decimal("0.3")
        )
    
    @pytest.fixture
    def funding_transaction(self, perpetual_position):
        """Test funding payment transaction."""
        return Transaction(
            transaction_id=uuid4(),
            position_id=perpetual_position.position_id,
            transaction_type=TransactionType.FUNDING,
            symbol="BTC-USD",
            size=Decimal("0"),
            price=Decimal("0"),
            timestamp=datetime.now() - timedelta(minutes=30),
            chain=Chain.HYPERLIQUID,
            dex_name="hyperliquid",
            transaction_hash="0xijkl9012",
            fees=Decimal("2.5")
        )

    # Core P&L Calculator Tests
    
    def test_pnl_calculator_creation(self, pnl_calculator, portfolio):
        """Test creating P&L calculator."""
        assert pnl_calculator.portfolio == portfolio
        assert pnl_calculator.cache_ttl_seconds == 300  # 5 minutes default
        assert len(pnl_calculator._cache) == 0
    
    def test_pnl_calculator_with_custom_config(self, portfolio):
        """Test creating P&L calculator with custom configuration."""
        calculator = PnLCalculator(
            portfolio=portfolio,
            cache_ttl_seconds=600,
            enable_attribution=True,
            enable_time_series=True
        )
        
        assert calculator.cache_ttl_seconds == 600
        assert calculator.enable_attribution is True
        assert calculator.enable_time_series is True

    # Real-time P&L Calculation Tests
    
    @pytest.mark.asyncio
    async def test_calculate_position_pnl_spot(self, pnl_calculator, spot_position):
        """Test calculating P&L for spot position."""
        # Add position to portfolio
        pnl_calculator.portfolio.add_position(spot_position)
        
        # Calculate P&L
        result = await pnl_calculator.calculate_position_pnl(spot_position.position_id)
        
        assert result.success is True
        assert result.position_id == spot_position.position_id
        assert result.unrealized_pnl == Decimal("500")  # (85 - 80) * 100
        assert result.realized_pnl == Decimal("0")  # No sales yet
        assert result.total_pnl == Decimal("500")
        assert result.position_type == PositionType.SPOT
        assert result.chain == Chain.SOLANA
        assert result.dex_name == "jupiter"
    
    @pytest.mark.asyncio
    async def test_calculate_position_pnl_perpetual(self, pnl_calculator, perpetual_position):
        """Test calculating P&L for perpetual position with leverage and funding."""
        # Add position to portfolio
        pnl_calculator.portfolio.add_position(perpetual_position)
        
        # Calculate P&L
        result = await pnl_calculator.calculate_position_pnl(perpetual_position.position_id)
        
        assert result.success is True
        assert result.position_id == perpetual_position.position_id
        
        # Expected P&L: (46000 - 45000) * 0.5 * 5 (leverage) + 2.5 (funding)
        expected_pnl = Decimal("1000") * Decimal("0.5") * Decimal("5") + Decimal("2.5")
        assert result.unrealized_pnl == expected_pnl
        assert result.total_pnl == expected_pnl
        assert result.position_type == PositionType.PERPETUAL
        assert result.leverage == Decimal("5")
        assert result.funding_pnl == Decimal("2.5")
    
    @pytest.mark.asyncio
    async def test_calculate_position_pnl_short(self, pnl_calculator):
        """Test calculating P&L for short perpetual position."""
        short_position = Position(
            position_id=uuid4(),
            symbol="BTC-USD",
            position_type=PositionType.PERPETUAL,
            chain=Chain.HYPERLIQUID,
            dex_name="hyperliquid",
            size=Decimal("0.3"),
            entry_price=Decimal("45000"),
            current_price=Decimal("44000"),  # Price down - profitable for short
            status=PositionStatus.OPEN,
            leverage=Decimal("3"),
            side="SHORT",
            unrealized_funding=Decimal("-1.2")  # Negative funding for short
        )
        
        pnl_calculator.portfolio.add_position(short_position)
        result = await pnl_calculator.calculate_position_pnl(short_position.position_id)
        
        assert result.success is True
        # Expected P&L: -(44000 - 45000) * 0.3 * 3 + (-1.2)
        expected_pnl = -(-Decimal("1000")) * Decimal("0.3") * Decimal("3") + Decimal("-1.2")
        assert result.unrealized_pnl == expected_pnl
        assert result.funding_pnl == Decimal("-1.2")
    
    @pytest.mark.asyncio
    async def test_calculate_position_pnl_closed(self, pnl_calculator, closed_position):
        """Test calculating P&L for closed position."""
        pnl_calculator.portfolio.add_position(closed_position)
        
        result = await pnl_calculator.calculate_position_pnl(closed_position.position_id)
        
        assert result.success is True
        assert result.unrealized_pnl == Decimal("0")  # Closed position
        # Need transactions to calculate realized P&L properly
        assert result.position_status == PositionStatus.CLOSED
    
    @pytest.mark.asyncio
    async def test_calculate_position_pnl_nonexistent(self, pnl_calculator):
        """Test calculating P&L for non-existent position."""
        result = await pnl_calculator.calculate_position_pnl(uuid4())
        
        assert result.success is False
        assert "Position not found" in result.error_message

    # Historical P&L Calculation Tests
    
    @pytest.mark.asyncio
    async def test_calculate_historical_pnl_with_transactions(self, pnl_calculator, spot_position, buy_transaction, sell_transaction):
        """Test calculating historical P&L with actual transactions."""
        # Add position and transactions
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_transaction(buy_transaction)
        pnl_calculator.portfolio.add_transaction(sell_transaction)
        
        result = await pnl_calculator.calculate_historical_pnl(
            position_id=spot_position.position_id,
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        assert result.success is True
        assert result.position_id == spot_position.position_id
        
        # Realized P&L: (85 - 80) * 50 = 250
        # Fees: 2.0 + 0.5 + 1.5 + 0.3 = 4.3
        expected_realized = Decimal("250")
        assert result.realized_pnl == expected_realized
        
        # Total fees from transactions
        expected_fees = Decimal("4.3")
        assert result.total_fees == expected_fees
        
        # Net P&L after fees
        assert result.net_pnl == expected_realized - expected_fees
    
    @pytest.mark.asyncio
    async def test_calculate_historical_pnl_date_range(self, pnl_calculator, spot_position, buy_transaction):
        """Test calculating historical P&L for specific date range."""
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_transaction(buy_transaction)
        
        # Query for date range that excludes the transaction
        result = await pnl_calculator.calculate_historical_pnl(
            position_id=spot_position.position_id,
            start_date=datetime.now() - timedelta(days=2),
            end_date=datetime.now() - timedelta(days=1)
        )
        
        assert result.success is True
        assert result.realized_pnl == Decimal("0")  # No transactions in range
        assert result.total_fees == Decimal("0")

    # Portfolio-level P&L Tests
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_pnl(self, pnl_calculator, spot_position, perpetual_position):
        """Test calculating total portfolio P&L."""
        # Add positions
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_position(perpetual_position)
        
        result = await pnl_calculator.calculate_portfolio_pnl()
        
        assert result.success is True
        assert len(result.position_pnls) == 2
        
        # Total unrealized P&L should be sum of both positions
        spot_pnl = Decimal("500")  # (85 - 80) * 100
        perp_pnl = Decimal("1000") * Decimal("0.5") * Decimal("5") + Decimal("2.5")  # With leverage and funding
        expected_total = spot_pnl + perp_pnl
        
        assert result.total_unrealized_pnl == expected_total
        assert result.total_positions == 2
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_pnl_empty(self, pnl_calculator):
        """Test calculating P&L for empty portfolio."""
        result = await pnl_calculator.calculate_portfolio_pnl()
        
        assert result.success is True
        assert result.total_unrealized_pnl == Decimal("0")
        assert result.total_realized_pnl == Decimal("0")
        assert result.total_positions == 0
        assert len(result.position_pnls) == 0

    # Multi-chain and Multi-DEX Tests
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_by_chain(self, pnl_calculator, spot_position, perpetual_position):
        """Test calculating P&L aggregated by blockchain chain."""
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_position(perpetual_position)
        
        result = await pnl_calculator.calculate_pnl_by_chain()
        
        assert result.success is True
        assert Chain.SOLANA in result.chain_pnls
        assert Chain.HYPERLIQUID in result.chain_pnls
        
        solana_pnl = result.chain_pnls[Chain.SOLANA]
        assert solana_pnl.total_pnl == Decimal("500")
        assert solana_pnl.position_count == 1
        
        hyperliquid_pnl = result.chain_pnls[Chain.HYPERLIQUID]
        expected_hyperliquid = Decimal("1000") * Decimal("0.5") * Decimal("5") + Decimal("2.5")
        assert hyperliquid_pnl.total_pnl == expected_hyperliquid
        assert hyperliquid_pnl.position_count == 1
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_by_dex(self, pnl_calculator, spot_position, perpetual_position):
        """Test calculating P&L aggregated by DEX."""
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_position(perpetual_position)
        
        result = await pnl_calculator.calculate_pnl_by_dex()
        
        assert result.success is True
        assert "jupiter" in result.dex_pnls
        assert "hyperliquid" in result.dex_pnls
        
        jupiter_pnl = result.dex_pnls["jupiter"]
        assert jupiter_pnl.total_pnl == Decimal("500")
        assert jupiter_pnl.position_count == 1
        
        hyperliquid_pnl = result.dex_pnls["hyperliquid"]
        expected_hyperliquid = Decimal("1000") * Decimal("0.5") * Decimal("5") + Decimal("2.5")
        assert hyperliquid_pnl.total_pnl == expected_hyperliquid
        assert hyperliquid_pnl.position_count == 1

    # Time-based Aggregation Tests
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_time_series_hourly(self, pnl_calculator, spot_position):
        """Test calculating P&L time series with hourly aggregation."""
        pnl_calculator.portfolio.add_position(spot_position)
        
        result = await pnl_calculator.calculate_pnl_time_series(
            start_date=datetime.now() - timedelta(hours=6),
            end_date=datetime.now(),
            period=PnLAggregationPeriod.HOURLY
        )
        
        assert result.success is True
        assert len(result.time_series) > 0
        
        # Check that time series points have correct structure
        for point in result.time_series:
            assert isinstance(point.timestamp, datetime)
            assert isinstance(point.total_pnl, Decimal)
            assert isinstance(point.position_count, int)
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_time_series_daily(self, pnl_calculator, spot_position, perpetual_position):
        """Test calculating P&L time series with daily aggregation."""
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_position(perpetual_position)
        
        result = await pnl_calculator.calculate_pnl_time_series(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            period=PnLAggregationPeriod.DAILY
        )
        
        assert result.success is True
        assert len(result.time_series) > 0
        
        # Should have multiple days of data
        total_pnl = sum(point.total_pnl for point in result.time_series)
        assert total_pnl > 0

    # Funding Payment Tests
    
    @pytest.mark.asyncio
    async def test_calculate_funding_pnl(self, pnl_calculator, perpetual_position, funding_transaction):
        """Test calculating funding payments for perpetual positions."""
        pnl_calculator.portfolio.add_position(perpetual_position)
        pnl_calculator.portfolio.add_transaction(funding_transaction)
        
        result = await pnl_calculator.calculate_funding_pnl(
            position_id=perpetual_position.position_id,
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now()
        )
        
        assert result.success is True
        assert result.position_id == perpetual_position.position_id
        assert result.total_funding == Decimal("2.5")  # From funding transaction
        assert result.funding_count == 1
    
    @pytest.mark.asyncio
    async def test_calculate_funding_pnl_spot_position(self, pnl_calculator, spot_position):
        """Test calculating funding P&L for spot position (should return zero)."""
        pnl_calculator.portfolio.add_position(spot_position)
        
        result = await pnl_calculator.calculate_funding_pnl(
            position_id=spot_position.position_id,
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now()
        )
        
        assert result.success is True
        assert result.total_funding == Decimal("0")
        assert result.funding_count == 0

    # Transaction Fees and Costs Tests
    
    @pytest.mark.asyncio
    async def test_calculate_fees_and_costs(self, pnl_calculator, spot_position, buy_transaction, sell_transaction):
        """Test calculating total fees and costs."""
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_transaction(buy_transaction)
        pnl_calculator.portfolio.add_transaction(sell_transaction)
        
        result = await pnl_calculator.calculate_fees_and_costs(
            position_id=spot_position.position_id,
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        assert result.success is True
        assert result.position_id == spot_position.position_id
        
        # Total fees: buy (2.0 + 0.5) + sell (1.5 + 0.3) = 4.3
        expected_total_fees = Decimal("4.3")
        assert result.total_fees == expected_total_fees
        assert result.trading_fees == Decimal("3.5")  # 2.0 + 1.5
        assert result.gas_fees == Decimal("0.8")  # 0.5 + 0.3
        assert result.transaction_count == 2
    
    @pytest.mark.asyncio
    async def test_calculate_fees_by_dex(self, pnl_calculator, spot_position, buy_transaction):
        """Test calculating fees aggregated by DEX."""
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_transaction(buy_transaction)
        
        result = await pnl_calculator.calculate_fees_by_dex(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        assert result.success is True
        assert "jupiter" in result.dex_fees
        
        jupiter_fees = result.dex_fees["jupiter"]
        assert jupiter_fees.total_fees == Decimal("2.5")  # 2.0 + 0.5
        assert jupiter_fees.transaction_count == 1

    # P&L Attribution Analysis Tests
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_attribution(self, pnl_calculator, perpetual_position, funding_transaction):
        """Test P&L attribution analysis (price vs funding vs fees)."""
        pnl_calculator.portfolio.add_position(perpetual_position)
        pnl_calculator.portfolio.add_transaction(funding_transaction)
        
        result = await pnl_calculator.calculate_pnl_attribution(
            position_id=perpetual_position.position_id
        )
        
        assert result.success is True
        assert result.position_id == perpetual_position.position_id
        
        # Price P&L: (46000 - 45000) * 0.5 * 5 = 2500
        assert result.attribution.price_pnl == Decimal("2500")
        
        # Funding P&L: 2.5 (from position unrealized_funding)
        assert result.attribution.funding_pnl == Decimal("2.5")
        
        # Fees should be from transactions
        assert result.attribution.fees_paid >= Decimal("0")
        
        # Total P&L should match price + funding (before fees)
        expected_total = result.attribution.price_pnl + result.attribution.funding_pnl
        assert abs(expected_total - result.total_pnl) < Decimal("0.01")  # Allow small rounding
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_attribution_spot(self, pnl_calculator, spot_position, buy_transaction):
        """Test P&L attribution for spot position (no funding)."""
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_transaction(buy_transaction)
        
        result = await pnl_calculator.calculate_pnl_attribution(
            position_id=spot_position.position_id
        )
        
        assert result.success is True
        
        # Only price P&L for spot positions
        assert result.attribution.price_pnl == Decimal("500")  # (85 - 80) * 100
        assert result.attribution.funding_pnl == Decimal("0")
        assert result.attribution.fees_paid > Decimal("0")

    # Caching Tests
    
    @pytest.mark.asyncio
    async def test_pnl_calculation_caching(self, pnl_calculator, spot_position):
        """Test that P&L calculations are cached for performance."""
        pnl_calculator.portfolio.add_position(spot_position)
        
        # First calculation
        result1 = await pnl_calculator.calculate_position_pnl(spot_position.position_id)
        assert result1.success is True
        
        # Check cache
        cache_key = f"position_pnl_{spot_position.position_id}"
        assert cache_key in pnl_calculator._cache
        
        # Second calculation should use cache
        result2 = await pnl_calculator.calculate_position_pnl(spot_position.position_id)
        assert result2.success is True
        assert result2.total_pnl == result1.total_pnl
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, pnl_calculator, spot_position):
        """Test that cache entries expire correctly."""
        # Use short TTL for testing
        pnl_calculator.cache_ttl_seconds = 1
        pnl_calculator.portfolio.add_position(spot_position)
        
        # First calculation
        result1 = await pnl_calculator.calculate_position_pnl(spot_position.position_id)
        assert result1.success is True
        
        # Wait for cache to expire
        import asyncio
        await asyncio.sleep(1.1)
        
        # Update position price
        spot_position.current_price = Decimal("90")
        
        # Second calculation should recalculate
        result2 = await pnl_calculator.calculate_position_pnl(spot_position.position_id)
        assert result2.success is True
        assert result2.total_pnl != result1.total_pnl
        assert result2.total_pnl == Decimal("1000")  # (90 - 80) * 100
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, pnl_calculator, spot_position):
        """Test clearing P&L calculation cache."""
        pnl_calculator.portfolio.add_position(spot_position)
        
        # Calculate P&L to populate cache
        await pnl_calculator.calculate_position_pnl(spot_position.position_id)
        assert len(pnl_calculator._cache) > 0
        
        # Clear cache
        pnl_calculator.clear_cache()
        assert len(pnl_calculator._cache) == 0

    # Performance Tests
    
    @pytest.mark.asyncio
    async def test_batch_pnl_calculation_performance(self, pnl_calculator):
        """Test P&L calculation performance with multiple positions."""
        import time
        
        # Create multiple positions
        positions = []
        for i in range(50):
            position = Position(
                position_id=uuid4(),
                symbol=f"TOKEN{i}/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.SOLANA,
                dex_name="jupiter",
                size=Decimal("100"),
                entry_price=Decimal("10"),
                current_price=Decimal("11"),
                status=PositionStatus.OPEN
            )
            positions.append(position)
            pnl_calculator.portfolio.add_position(position)
        
        # Measure calculation time
        start_time = time.time()
        result = await pnl_calculator.calculate_portfolio_pnl()
        end_time = time.time()
        
        assert result.success is True
        assert len(result.position_pnls) == 50
        
        # Should complete within reasonable time (< 1 second)
        calculation_time = end_time - start_time
        assert calculation_time < 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_pnl_calculations(self, pnl_calculator, spot_position, perpetual_position):
        """Test concurrent P&L calculations."""
        import asyncio
        
        pnl_calculator.portfolio.add_position(spot_position)
        pnl_calculator.portfolio.add_position(perpetual_position)
        
        # Run multiple calculations concurrently
        tasks = [
            pnl_calculator.calculate_position_pnl(spot_position.position_id),
            pnl_calculator.calculate_position_pnl(perpetual_position.position_id),
            pnl_calculator.calculate_portfolio_pnl(),
            pnl_calculator.calculate_pnl_by_chain(),
            pnl_calculator.calculate_pnl_by_dex()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All calculations should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert result.success is True

    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_invalid_position_id(self, pnl_calculator):
        """Test error handling for invalid position ID."""
        result = await pnl_calculator.calculate_position_pnl(uuid4())
        
        assert result.success is False
        assert "Position not found" in result.error_message
        assert isinstance(result.error_message, str)
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_invalid_date_range(self, pnl_calculator, spot_position):
        """Test error handling for invalid date range."""
        pnl_calculator.portfolio.add_position(spot_position)
        
        # End date before start date
        result = await pnl_calculator.calculate_historical_pnl(
            position_id=spot_position.position_id,
            start_date=datetime.now(),
            end_date=datetime.now() - timedelta(days=1)
        )
        
        assert result.success is False
        assert "Invalid date range" in result.error_message
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_with_exceptions(self, pnl_calculator, spot_position):
        """Test error handling when calculations raise exceptions."""
        pnl_calculator.portfolio.add_position(spot_position)
        
        # Mock position to raise exception when accessing unrealized_pnl property
        with patch.object(type(spot_position), 'unrealized_pnl', new_callable=lambda: property(lambda self: (_ for _ in ()).throw(Exception("Test error")))):
            result = await pnl_calculator.calculate_position_pnl(spot_position.position_id)
            
            assert result.success is False
            assert "Test error" in result.error_message
    
    def test_pnl_calculation_error_creation(self):
        """Test creating P&L calculation error."""
        error = PnLCalculationError("Test error", uuid4())
        
        assert str(error) == "Test error"
        assert error.position_id is not None

    # Edge Cases
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_zero_size_position(self, pnl_calculator):
        """Test P&L calculation for position with zero size."""
        zero_position = Position(
            position_id=uuid4(),
            symbol="ZERO/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("0"),
            entry_price=Decimal("100"),
            current_price=Decimal("110"),
            status=PositionStatus.OPEN
        )
        
        pnl_calculator.portfolio.add_position(zero_position)
        result = await pnl_calculator.calculate_position_pnl(zero_position.position_id)
        
        assert result.success is True
        assert result.total_pnl == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_extreme_leverage(self, pnl_calculator):
        """Test P&L calculation with extreme leverage."""
        high_leverage_position = Position(
            position_id=uuid4(),
            symbol="BTC-USD",
            position_type=PositionType.PERPETUAL,
            chain=Chain.HYPERLIQUID,
            dex_name="hyperliquid",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),  # Small price move
            status=PositionStatus.OPEN,
            leverage=Decimal("100"),  # 100x leverage
            side="LONG"
        )
        
        pnl_calculator.portfolio.add_position(high_leverage_position)
        result = await pnl_calculator.calculate_position_pnl(high_leverage_position.position_id)
        
        assert result.success is True
        # P&L: (50100 - 50000) * 0.1 * 100 = 1000
        assert result.total_pnl == Decimal("1000")
        assert result.leverage == Decimal("100")
    
    @pytest.mark.asyncio
    async def test_calculate_pnl_very_old_position(self, pnl_calculator):
        """Test P&L calculation for very old position."""
        old_position = Position(
            position_id=uuid4(),
            symbol="OLD/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            size=Decimal("1000"),
            entry_price=Decimal("1"),
            current_price=Decimal("100"),
            status=PositionStatus.OPEN,
            created_at=datetime.now() - timedelta(days=365)  # 1 year old
        )
        
        pnl_calculator.portfolio.add_position(old_position)
        result = await pnl_calculator.calculate_position_pnl(old_position.position_id)
        
        assert result.success is True
        assert result.total_pnl == Decimal("99000")  # (100 - 1) * 1000
        
        # Calculate time series for the old position
        time_series_result = await pnl_calculator.calculate_pnl_time_series(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            period=PnLAggregationPeriod.DAILY
        )
        
        assert time_series_result.success is True
        assert len(time_series_result.time_series) > 0