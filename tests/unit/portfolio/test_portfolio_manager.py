"""
Tests for Portfolio Manager functionality.

Following TDD methodology - these tests define the expected behavior
for portfolio management operations across multiple DEXs and chains.
The PortfolioManager serves as the central orchestrator for all 
portfolio operations.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from typing import Dict, List, Optional

from src.portfolio.base import (
    Portfolio,
    Position,
    PositionType,
    PositionStatus,
    Transaction,
    TransactionType,
    PortfolioConfig,
    PerformanceMetrics,
    RiskMetrics,
    DrawdownMetrics,
    PortfolioError,
    InsufficientFundsError,
    RiskLimitExceededError,
    PositionNotFoundError,
)
from src.portfolio.position_tracker import (
    PositionTracker,
    PositionUpdateResult,
    PositionCloseResult,
)
from src.utils.base import Chain


# Shared fixtures for all test classes
@pytest.fixture
def config():
    """Test portfolio configuration."""
    return PortfolioConfig(
        initial_balance=Decimal("10000"),
        base_currency="USDC",
        max_position_size_pct=Decimal("0.1"),  # 10%
        max_daily_loss_pct=Decimal("0.05"),    # 5%
        max_drawdown_pct=Decimal("0.15"),      # 15%
        stop_loss_pct=Decimal("0.08"),         # 8%
        take_profit_pct=Decimal("0.4"),        # 40%
        max_open_positions=10,
        min_trade_amount_usd=Decimal("10"),
        enable_risk_management=True,
        enable_auto_rebalancing=False
    )


@pytest.fixture
def portfolio_manager(config):
    """Test portfolio manager instance."""
    from src.portfolio.portfolio_manager import PortfolioManager
    return PortfolioManager(config)


@pytest.fixture
def sample_portfolio(config):
    """Sample portfolio for testing."""
    portfolio_id = uuid4()
    return Portfolio(
        portfolio_id=portfolio_id,
        name="Test Portfolio",
        config=config,
        cash_balance=Decimal("10000"),
        total_value=Decimal("10000")
    )


@pytest.fixture
def sample_position():
    """Sample position for testing."""
    return Position(
        position_id=uuid4(),
        symbol="BTC/USDC",
        position_type=PositionType.SPOT,
        chain=Chain.SOLANA,
        dex_name="jupiter",
        size=Decimal("0.01"),  # Smaller size: 0.01 BTC * $50k = $500 (under $1000 limit)
        entry_price=Decimal("50000"),
        current_price=Decimal("52000"),
        status=PositionStatus.OPEN,
        leverage=Decimal("1"),
        side="LONG"
    )


class TestPortfolioManager:
    """Test Portfolio Manager functionality with comprehensive coverage."""


class TestPortfolioManagerInitialization:
    """Test portfolio manager initialization and configuration."""
    
    def test_portfolio_manager_init_with_config(self, config):
        """Test portfolio manager initialization with valid config."""
        from src.portfolio.portfolio_manager import PortfolioManager
        
        manager = PortfolioManager(config)
        
        assert manager.config == config
        assert isinstance(manager.position_tracker, PositionTracker)
        assert len(manager.portfolios) == 0
        assert manager.default_portfolio_id is None
    
    def test_portfolio_manager_init_with_invalid_config(self):
        """Test portfolio manager initialization with invalid config."""
        from src.portfolio.portfolio_manager import PortfolioManager
        
        with pytest.raises(ValueError):
            PortfolioManager(None)
    
    def test_portfolio_manager_properties(self, portfolio_manager):
        """Test portfolio manager computed properties."""
        assert portfolio_manager.total_portfolios == 0
        assert portfolio_manager.total_value == Decimal("0")
        assert portfolio_manager.total_cash == Decimal("0")
        assert portfolio_manager.total_positions == 0
        assert portfolio_manager.total_unrealized_pnl == Decimal("0")


class TestPortfolioCreationAndManagement:
    """Test portfolio creation and management operations."""
    
    @pytest.mark.asyncio
    async def test_create_portfolio_success(self, portfolio_manager, config):
        """Test successful portfolio creation."""
        portfolio_name = "Test Portfolio"
        
        result = await portfolio_manager.create_portfolio(
            name=portfolio_name,
            initial_balance=config.initial_balance
        )
        
        assert result.success is True
        assert result.portfolio_id is not None
        assert result.message == "Portfolio created successfully"
        assert len(portfolio_manager.portfolios) == 1
        
        # Verify portfolio details
        portfolio = portfolio_manager.get_portfolio(result.portfolio_id)
        assert portfolio is not None
        assert portfolio.name == portfolio_name
        assert portfolio.cash_balance == config.initial_balance
        assert portfolio.config == config
    
    @pytest.mark.asyncio
    async def test_create_portfolio_with_zero_balance(self, portfolio_manager):
        """Test portfolio creation with zero initial balance fails."""
        result = await portfolio_manager.create_portfolio(
            name="Zero Balance Portfolio",
            initial_balance=Decimal("0")
        )
        
        assert result.success is False
        assert "Initial balance must be positive" in result.message
        assert len(portfolio_manager.portfolios) == 0
    
    @pytest.mark.asyncio
    async def test_create_portfolio_with_duplicate_name(self, portfolio_manager):
        """Test portfolio creation with duplicate name."""
        portfolio_name = "Duplicate Portfolio"
        
        # Create first portfolio
        result1 = await portfolio_manager.create_portfolio(
            name=portfolio_name,
            initial_balance=Decimal("1000")
        )
        assert result1.success is True
        
        # Try to create duplicate
        result2 = await portfolio_manager.create_portfolio(
            name=portfolio_name,
            initial_balance=Decimal("2000")
        )
        assert result2.success is False
        assert "Portfolio with name" in result2.message
        assert len(portfolio_manager.portfolios) == 1
    
    @pytest.mark.asyncio
    async def test_set_default_portfolio(self, portfolio_manager):
        """Test setting default portfolio."""
        # Create portfolio
        result = await portfolio_manager.create_portfolio(
            name="Default Portfolio",
            initial_balance=Decimal("5000")
        )
        portfolio_id = result.portfolio_id
        
        # Set as default
        set_result = await portfolio_manager.set_default_portfolio(portfolio_id)
        assert set_result.success is True
        assert portfolio_manager.default_portfolio_id == portfolio_id
    
    @pytest.mark.asyncio
    async def test_set_default_portfolio_invalid_id(self, portfolio_manager):
        """Test setting default portfolio with invalid ID."""
        invalid_id = uuid4()
        
        result = await portfolio_manager.set_default_portfolio(invalid_id)
        assert result.success is False
        assert "Portfolio not found" in result.message
    
    def test_get_portfolio_by_id(self, portfolio_manager, sample_portfolio):
        """Test getting portfolio by ID."""
        # Add portfolio manually for testing
        portfolio_manager.portfolios[sample_portfolio.portfolio_id] = sample_portfolio
        
        retrieved = portfolio_manager.get_portfolio(sample_portfolio.portfolio_id)
        assert retrieved == sample_portfolio
    
    def test_get_portfolio_by_name(self, portfolio_manager, sample_portfolio):
        """Test getting portfolio by name."""
        # Add portfolio manually for testing
        portfolio_manager.portfolios[sample_portfolio.portfolio_id] = sample_portfolio
        
        retrieved = portfolio_manager.get_portfolio_by_name(sample_portfolio.name)
        assert retrieved == sample_portfolio
    
    def test_get_nonexistent_portfolio(self, portfolio_manager):
        """Test getting non-existent portfolio returns None."""
        invalid_id = uuid4()
        result = portfolio_manager.get_portfolio(invalid_id)
        assert result is None
        
        result = portfolio_manager.get_portfolio_by_name("Non-existent")
        assert result is None


class TestPositionManagement:
    """Test position management operations through PortfolioManager."""
    
    @pytest.mark.asyncio
    async def test_add_position_to_portfolio_success(self, portfolio_manager, sample_position):
        """Test successfully adding position to portfolio."""
        # Create portfolio first
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Position Test Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Add position
        result = await portfolio_manager.add_position(
            portfolio_id=portfolio_id,
            position=sample_position
        )
        
        assert result.success is True
        assert result.position_id == sample_position.position_id
        assert "Position added successfully" in result.message
        
        # Verify position was added to both portfolio and tracker
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        assert sample_position.position_id in portfolio.positions
        
        # Verify position is tracked
        tracked_position = portfolio_manager.position_tracker.get_position(sample_position.position_id)
        assert tracked_position == sample_position
    
    @pytest.mark.asyncio
    async def test_add_position_to_nonexistent_portfolio(self, portfolio_manager, sample_position):
        """Test adding position to non-existent portfolio fails."""
        invalid_portfolio_id = uuid4()
        
        result = await portfolio_manager.add_position(
            portfolio_id=invalid_portfolio_id,
            position=sample_position
        )
        
        assert result.success is False
        assert "Portfolio not found" in result.message
    
    @pytest.mark.asyncio
    async def test_add_position_exceeds_max_positions(self, portfolio_manager):
        """Test adding position when max positions limit is reached."""
        # Create portfolio with low max positions limit
        config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC",
            max_open_positions=1  # Only allow 1 position
        )
        manager = portfolio_manager.__class__(config)
        
        # Create portfolio
        portfolio_result = await manager.create_portfolio(
            name="Limited Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Add first position (should succeed)
        position1 = Position(
            position_id=uuid4(),
            symbol="BTC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("0.01"),  # Small position within limits
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            status=PositionStatus.OPEN
        )
        
        result1 = await manager.add_position(portfolio_id, position1)
        assert result1.success is True
        
        # Add second position (should fail due to limit)
        position2 = Position(
            position_id=uuid4(),
            symbol="ETH/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("0.1"),  # Small position but should hit max positions limit
            entry_price=Decimal("3000"),
            current_price=Decimal("3000"),
            status=PositionStatus.OPEN
        )
        
        result2 = await manager.add_position(portfolio_id, position2)
        assert result2.success is False
        assert "Maximum open positions limit" in result2.message
    
    @pytest.mark.asyncio
    async def test_close_position_success(self, portfolio_manager, sample_position):
        """Test successfully closing a position."""
        # Setup portfolio and position
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Close Test Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        await portfolio_manager.add_position(portfolio_id, sample_position)
        
        # Close position
        result = await portfolio_manager.close_position(
            portfolio_id=portfolio_id,
            position_id=sample_position.position_id,
            close_size=sample_position.size,
            close_price=Decimal("55000"),
            reason="profit_taking"
        )
        
        assert result.success is True
        assert result.position_id == sample_position.position_id
        assert result.close_size == sample_position.size
        assert result.close_price == Decimal("55000")
        assert result.realized_pnl > 0  # Position was profitable
        
        # Verify position status updated
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        position = portfolio.positions[sample_position.position_id]
        assert position.status == PositionStatus.CLOSED
    
    @pytest.mark.asyncio
    async def test_close_position_partial(self, portfolio_manager, sample_position):
        """Test partially closing a position."""
        # Setup portfolio and position
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Partial Close Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        await portfolio_manager.add_position(portfolio_id, sample_position)
        original_size = sample_position.size
        close_size = original_size / 2
        
        # Partially close position
        result = await portfolio_manager.close_position(
            portfolio_id=portfolio_id,
            position_id=sample_position.position_id,
            close_size=close_size,
            close_price=Decimal("55000")
        )
        
        assert result.success is True
        assert result.close_size == close_size
        assert result.remaining_size == original_size - close_size
        
        # Verify position status and size updated
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        position = portfolio.positions[sample_position.position_id]
        assert position.status == PositionStatus.PARTIAL
        assert position.size == original_size - close_size
    
    @pytest.mark.asyncio
    async def test_update_position_prices_single(self, portfolio_manager, sample_position):
        """Test updating single position price."""
        # Setup portfolio and position
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Price Update Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        await portfolio_manager.add_position(portfolio_id, sample_position)
        
        # Update price
        new_price = Decimal("60000")
        result = await portfolio_manager.update_position_price(
            position_id=sample_position.position_id,
            new_price=new_price
        )
        
        assert result.success is True
        assert result.new_price == new_price
        assert result.price_change == new_price - sample_position.current_price
        
        # Verify position updated
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        position = portfolio.positions[sample_position.position_id]
        assert position.current_price == new_price


class TestBatchOperations:
    """Test batch operations for multiple positions."""
    
    @pytest.fixture
    def multiple_positions(self):
        """Multiple positions for batch testing."""
        positions = []
        symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        prices = [Decimal("50000"), Decimal("3000"), Decimal("100")]
        
        for i, (symbol, price) in enumerate(zip(symbols, prices)):
            position = Position(
                position_id=uuid4(),
                symbol=symbol,
                position_type=PositionType.SPOT,
                chain=Chain.SOLANA,
                dex_name="jupiter",
                size=Decimal("0.1") * (i + 1),  # Different sizes
                entry_price=price,
                current_price=price,
                status=PositionStatus.OPEN
            )
            positions.append(position)
        
        return positions
    
    @pytest.mark.asyncio
    async def test_batch_add_positions(self, portfolio_manager, multiple_positions):
        """Test adding multiple positions in batch."""
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Batch Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Batch add positions
        results = await portfolio_manager.batch_add_positions(
            portfolio_id=portfolio_id,
            positions=multiple_positions
        )
        
        assert len(results) == len(multiple_positions)
        assert all(result.success for result in results)
        
        # Verify all positions added
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        assert len(portfolio.positions) == len(multiple_positions)
        
        for position in multiple_positions:
            assert position.position_id in portfolio.positions
    
    @pytest.mark.asyncio
    async def test_batch_update_prices(self, portfolio_manager, multiple_positions):
        """Test updating multiple position prices in batch."""
        # Setup portfolio and positions
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Batch Update Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        await portfolio_manager.batch_add_positions(portfolio_id, multiple_positions)
        
        # Prepare price updates
        price_updates = {}
        for position in multiple_positions:
            # Increase each price by 10%
            new_price = position.current_price * Decimal("1.1")
            price_updates[position.position_id] = new_price
        
        # Batch update prices
        results = await portfolio_manager.batch_update_prices(price_updates)
        
        assert len(results) == len(multiple_positions)
        assert all(result.success for result in results)
        
        # Verify all prices updated
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        for position_id, expected_price in price_updates.items():
            position = portfolio.positions[position_id]
            assert position.current_price == expected_price
    
    @pytest.mark.asyncio
    async def test_batch_close_positions(self, portfolio_manager, multiple_positions):
        """Test closing multiple positions in batch."""
        # Setup portfolio and positions
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Batch Close Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        await portfolio_manager.batch_add_positions(portfolio_id, multiple_positions)
        
        # Prepare close operations
        close_requests = []
        for position in multiple_positions:
            close_requests.append({
                'position_id': position.position_id,
                'close_size': position.size,  # Close full position
                'close_price': position.current_price * Decimal("1.05"),  # 5% profit
                'reason': 'batch_close'
            })
        
        # Batch close positions
        results = await portfolio_manager.batch_close_positions(
            portfolio_id=portfolio_id,
            close_requests=close_requests
        )
        
        assert len(results) == len(multiple_positions)
        assert all(result.success for result in results)
        
        # Verify all positions closed
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        for position in portfolio.positions.values():
            assert position.status == PositionStatus.CLOSED


class TestRiskManagement:
    """Test risk management and validation."""
    
    @pytest.mark.asyncio
    async def test_position_size_validation(self, portfolio_manager):
        """Test position size validation against portfolio limits."""
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Risk Test Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Try to add position exceeding max size (10% of portfolio = $1000)
        large_position = Position(
            position_id=uuid4(),
            symbol="BTC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("0.03"),  # 0.03 BTC at $50k = $1500 > $1000 limit
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            status=PositionStatus.OPEN
        )
        
        result = await portfolio_manager.add_position(portfolio_id, large_position)
        assert result.success is False
        assert "exceeds maximum position size" in result.message
    
    @pytest.mark.asyncio
    async def test_insufficient_funds_validation(self, portfolio_manager):
        """Test insufficient funds validation."""
        # Create portfolio with limited funds
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Limited Funds Portfolio",
            initial_balance=Decimal("1000")  # Only $1000
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Try to add position requiring more funds than available
        expensive_position = Position(
            position_id=uuid4(),
            symbol="BTC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("0.03"),  # 0.03 BTC at $50k = $1500 > $1000 available
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            status=PositionStatus.OPEN
        )
        
        result = await portfolio_manager.add_position(portfolio_id, expensive_position)
        assert result.success is False
        assert "Insufficient funds" in result.message
    
    @pytest.mark.asyncio
    async def test_min_trade_amount_validation(self, portfolio_manager):
        """Test minimum trade amount validation."""
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Min Trade Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Try to add position below minimum trade amount ($10)
        tiny_position = Position(
            position_id=uuid4(),
            symbol="BTC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("0.0001"),  # 0.0001 BTC at $50k = $5 < $10 minimum
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            status=PositionStatus.OPEN
        )
        
        result = await portfolio_manager.add_position(portfolio_id, tiny_position)
        assert result.success is False
        assert "below minimum trade amount" in result.message
    
    @pytest.mark.asyncio
    async def test_drawdown_monitoring(self, portfolio_manager, sample_position):
        """Test portfolio drawdown monitoring."""
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Drawdown Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Add position
        await portfolio_manager.add_position(portfolio_id, sample_position)
        
        # Simulate large loss (update price to create 20% drawdown > 15% limit)
        loss_price = sample_position.entry_price * Decimal("0.6")  # 40% loss
        
        result = await portfolio_manager.update_position_price(
            sample_position.position_id,
            loss_price
        )
        
        # Should trigger drawdown warning
        assert result.success is True
        # Check if portfolio manager detects excessive drawdown
        drawdown_risk = await portfolio_manager.check_drawdown_risk(portfolio_id)
        assert drawdown_risk.at_risk is True
        assert drawdown_risk.current_drawdown > portfolio_manager.config.max_drawdown_pct


class TestMultiChainSupport:
    """Test multi-chain and multi-DEX support."""
    
    @pytest.mark.asyncio
    async def test_positions_across_multiple_chains(self, portfolio_manager):
        """Test managing positions across different blockchain networks."""
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Multi-Chain Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Create positions on different chains
        positions = [
            # Solana position
            Position(
                position_id=uuid4(),
                symbol="SOL/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.SOLANA,
                dex_name="jupiter",
                size=Decimal("10"),
                entry_price=Decimal("100"),
                current_price=Decimal("100"),
                status=PositionStatus.OPEN
            ),
            # Ethereum position
            Position(
                position_id=uuid4(),
                symbol="ETH/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.ETHEREUM,
                dex_name="uniswap_v3",
                size=Decimal("1"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3000"),
                status=PositionStatus.OPEN
            ),
            # Hyperliquid perpetual
            Position(
                position_id=uuid4(),
                symbol="BTC-USD",
                position_type=PositionType.PERPETUAL,
                chain=Chain.HYPERLIQUID,
                dex_name="hyperliquid",
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("50000"),
                status=PositionStatus.OPEN,
                leverage=Decimal("10"),
                side="LONG"
            )
        ]
        
        # Add all positions
        for position in positions:
            result = await portfolio_manager.add_position(portfolio_id, position)
            assert result.success is True
        
        # Verify positions grouped by chain
        chain_summary = await portfolio_manager.get_positions_by_chain(portfolio_id)
        assert Chain.SOLANA in chain_summary
        assert Chain.ETHEREUM in chain_summary
        assert Chain.HYPERLIQUID in chain_summary
        
        assert len(chain_summary[Chain.SOLANA]) == 1
        assert len(chain_summary[Chain.ETHEREUM]) == 1
        assert len(chain_summary[Chain.HYPERLIQUID]) == 1
    
    @pytest.mark.asyncio
    async def test_positions_grouped_by_dex(self, portfolio_manager):
        """Test getting positions grouped by DEX."""
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Multi-DEX Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Create positions on different DEXs
        jupiter_position = Position(
            position_id=uuid4(),
            symbol="SOL/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("10"),
            entry_price=Decimal("100"),
            current_price=Decimal("100"),
            status=PositionStatus.OPEN
        )
        
        hyperliquid_position = Position(
            position_id=uuid4(),
            symbol="BTC-USD",
            position_type=PositionType.PERPETUAL,
            chain=Chain.HYPERLIQUID,
            dex_name="hyperliquid",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            status=PositionStatus.OPEN
        )
        
        # Add positions
        await portfolio_manager.add_position(portfolio_id, jupiter_position)
        await portfolio_manager.add_position(portfolio_id, hyperliquid_position)
        
        # Get positions by DEX
        dex_summary = await portfolio_manager.get_positions_by_dex(portfolio_id)
        assert "jupiter" in dex_summary
        assert "hyperliquid" in dex_summary
        assert len(dex_summary["jupiter"]) == 1
        assert len(dex_summary["hyperliquid"]) == 1


class TestPerformanceMetrics:
    """Test portfolio performance calculations and metrics."""
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_performance(self, portfolio_manager, multiple_positions):
        """Test portfolio performance metrics calculation."""
        # Create portfolio and add positions
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Performance Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        await portfolio_manager.batch_add_positions(portfolio_id, multiple_positions)
        
        # Update prices to create some P&L
        price_updates = {}
        for i, position in enumerate(multiple_positions):
            # Mix of winners and losers
            multiplier = Decimal("1.2") if i % 2 == 0 else Decimal("0.9")
            price_updates[position.position_id] = position.current_price * multiplier
        
        await portfolio_manager.batch_update_prices(price_updates)
        
        # Calculate performance metrics
        metrics = await portfolio_manager.calculate_performance_metrics(portfolio_id)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_pnl != Decimal("0")  # Should have some P&L
        assert metrics.unrealized_pnl != Decimal("0")
        assert metrics.initial_balance == Decimal("50000")
        assert metrics.roi is not None
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, portfolio_manager, multiple_positions):
        """Test portfolio risk metrics calculation."""
        # Create portfolio and add positions
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Risk Metrics Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        await portfolio_manager.batch_add_positions(portfolio_id, multiple_positions)
        
        # Calculate risk metrics
        risk_metrics = await portfolio_manager.calculate_risk_metrics(portfolio_id)
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.volatility >= Decimal("0")
        assert risk_metrics.var_95 is not None
        assert risk_metrics.var_99 is not None
        assert risk_metrics.concentration_risk >= Decimal("0")
    
    @pytest.mark.asyncio
    async def test_portfolio_summary_statistics(self, portfolio_manager, multiple_positions):
        """Test comprehensive portfolio summary statistics."""
        # Create portfolio and add positions
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Summary Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        await portfolio_manager.batch_add_positions(portfolio_id, multiple_positions)
        
        # Get summary
        summary = await portfolio_manager.get_portfolio_summary(portfolio_id)
        
        assert summary is not None
        assert 'total_value' in summary
        assert 'total_positions' in summary
        assert 'open_positions' in summary
        assert 'unrealized_pnl' in summary
        assert 'performance_metrics' in summary
        assert 'risk_metrics' in summary
        assert 'positions_by_chain' in summary
        assert 'positions_by_dex' in summary


class TestRebalancingOperations:
    """Test portfolio rebalancing functionality."""
    
    @pytest.mark.asyncio
    async def test_auto_rebalancing_disabled_by_default(self, portfolio_manager):
        """Test that auto-rebalancing is disabled by default."""
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="No Rebalance Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Check rebalancing status
        rebalance_needed = await portfolio_manager.check_rebalancing_needed(portfolio_id)
        assert rebalance_needed.auto_rebalancing_enabled is False
        assert rebalance_needed.rebalancing_needed is False
    
    @pytest.mark.asyncio
    async def test_manual_rebalancing_suggestions(self, portfolio_manager, multiple_positions):
        """Test manual rebalancing suggestions."""
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Rebalance Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        await portfolio_manager.batch_add_positions(portfolio_id, multiple_positions)
        
        # Create imbalance by changing prices significantly
        price_updates = {}
        for i, position in enumerate(multiple_positions):
            # Make first position very large relative to others
            multiplier = Decimal("3.0") if i == 0 else Decimal("0.8")
            price_updates[position.position_id] = position.current_price * multiplier
        
        await portfolio_manager.batch_update_prices(price_updates)
        
        # Get rebalancing suggestions
        suggestions = await portfolio_manager.get_rebalancing_suggestions(portfolio_id)
        
        assert suggestions is not None
        assert len(suggestions.suggested_adjustments) > 0
        assert suggestions.total_imbalance > Decimal("0")


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_handle_invalid_portfolio_operations(self, portfolio_manager):
        """Test operations on invalid/non-existent portfolios."""
        invalid_id = uuid4()
        
        # Test various operations with invalid portfolio ID
        with pytest.raises((PortfolioError, PositionNotFoundError)):
            await portfolio_manager.add_position(invalid_id, Position(
                position_id=uuid4(),
                symbol="BTC/USDC",
                position_type=PositionType.SPOT,
                chain=Chain.SOLANA,
                dex_name="jupiter",
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("50000"),
                status=PositionStatus.OPEN
            ))
    
    @pytest.mark.asyncio
    async def test_handle_concurrent_operations(self, portfolio_manager, multiple_positions):
        """Test handling concurrent operations on same portfolio."""
        import asyncio
        
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Concurrent Portfolio",
            initial_balance=Decimal("50000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Attempt concurrent position additions
        tasks = []
        for position in multiple_positions:
            task = portfolio_manager.add_position(portfolio_id, position)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed or handle gracefully
        success_count = sum(1 for r in results if hasattr(r, 'success') and r.success)
        assert success_count == len(multiple_positions)
    
    @pytest.mark.asyncio
    async def test_handle_invalid_position_data(self, portfolio_manager):
        """Test handling invalid position data."""
        # Create portfolio
        portfolio_result = await portfolio_manager.create_portfolio(
            name="Invalid Data Portfolio",
            initial_balance=Decimal("10000")
        )
        portfolio_id = portfolio_result.portfolio_id
        
        # Test with invalid position data
        invalid_position = Position(
            position_id=uuid4(),
            symbol="",  # Empty symbol
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("-1"),  # Negative size
            entry_price=Decimal("0"),  # Zero price
            current_price=Decimal("0"),
            status=PositionStatus.OPEN
        )
        
        result = await portfolio_manager.add_position(portfolio_id, invalid_position)
        assert result.success is False
        assert "Invalid position data" in result.message