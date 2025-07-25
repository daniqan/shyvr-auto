"""
Tests for position tracker functionality.

Following TDD methodology - these tests define the expected behavior
for position tracking across different DEXs and position types.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.portfolio.base import (
    Position,
    PositionType,
    PositionStatus,
    Transaction,
    TransactionType,
    PortfolioConfig,
    PortfolioError,
    InvalidPositionError,
)
from src.portfolio.position_tracker import (
    PositionTracker,
    PositionUpdateResult,
    PositionCloseResult,
)
from src.utils.base import Chain


class TestPositionTracker:
    """Test position tracker functionality."""
    
    @pytest.fixture
    def config(self):
        """Test portfolio configuration."""
        return PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC",
            stop_loss_pct=Decimal("0.08"),  # 8%
            take_profit_pct=Decimal("0.4")  # 40%
        )
    
    @pytest.fixture
    def position_tracker(self, config):
        """Test position tracker instance."""
        return PositionTracker(config)
    
    @pytest.fixture
    def spot_position(self):
        """Test spot position."""
        return Position(
            position_id=uuid4(),
            symbol="SOL/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("100"),
            entry_price=Decimal("80"),
            current_price=Decimal("80"),
            status=PositionStatus.OPEN
        )
    
    @pytest.fixture
    def perpetual_position(self):
        """Test perpetual position."""
        return Position(
            position_id=uuid4(),
            symbol="BTC-USD",
            position_type=PositionType.PERPETUAL,
            chain=Chain.HYPERLIQUID,
            dex_name="hyperliquid",
            size=Decimal("0.5"),
            entry_price=Decimal("45000"),
            current_price=Decimal("45000"),
            status=PositionStatus.OPEN,
            leverage=Decimal("5"),
            side="LONG"
        )
    
    def test_position_tracker_creation(self, position_tracker, config):
        """Test creating position tracker."""
        assert position_tracker.config == config
        assert len(position_tracker.positions) == 0
        assert position_tracker.total_positions == 0
        assert position_tracker.open_positions_count == 0
    
    def test_add_position_spot(self, position_tracker, spot_position):
        """Test adding spot position."""
        result = position_tracker.add_position(spot_position)
        
        assert result.success is True
        assert result.position_id == spot_position.position_id
        assert result.message == "Position added successfully"
        
        assert len(position_tracker.positions) == 1
        assert spot_position.position_id in position_tracker.positions
        assert position_tracker.total_positions == 1
        assert position_tracker.open_positions_count == 1
    
    def test_add_position_perpetual(self, position_tracker, perpetual_position):
        """Test adding perpetual position."""
        result = position_tracker.add_position(perpetual_position)
        
        assert result.success is True
        assert result.position_id == perpetual_position.position_id
        
        # Check that stop loss and take profit were set
        tracked_position = position_tracker.get_position(perpetual_position.position_id)
        assert tracked_position.stop_loss_price is not None
        assert tracked_position.take_profit_price is not None
    
    def test_add_duplicate_position(self, position_tracker, spot_position):
        """Test adding duplicate position fails."""
        # Add position first time
        position_tracker.add_position(spot_position)
        
        # Try to add same position again
        result = position_tracker.add_position(spot_position)
        
        assert result.success is False
        assert "already exists" in result.message.lower()
        assert len(position_tracker.positions) == 1
    
    def test_get_position_exists(self, position_tracker, spot_position):
        """Test getting existing position."""
        position_tracker.add_position(spot_position)
        
        retrieved = position_tracker.get_position(spot_position.position_id)
        assert retrieved == spot_position
    
    def test_get_position_not_exists(self, position_tracker):
        """Test getting non-existent position."""
        non_existent_id = uuid4()
        retrieved = position_tracker.get_position(non_existent_id)
        assert retrieved is None
    
    def test_update_position_price(self, position_tracker, spot_position):
        """Test updating position price."""
        position_tracker.add_position(spot_position)
        
        new_price = Decimal("90")  # +$10 profit
        result = position_tracker.update_position_price(
            spot_position.position_id, 
            new_price
        )
        
        assert result.success is True
        assert result.price_change == new_price - spot_position.entry_price
        assert result.pnl_change == (new_price - spot_position.entry_price) * spot_position.size
        
        # Check position was updated
        updated_position = position_tracker.get_position(spot_position.position_id)
        assert updated_position.current_price == new_price
        assert updated_position.unrealized_pnl == Decimal("1000")  # $10 * 100 shares
    
    def test_update_position_price_nonexistent(self, position_tracker):
        """Test updating price for non-existent position."""
        result = position_tracker.update_position_price(uuid4(), Decimal("100"))
        
        assert result.success is False
        assert "not found" in result.message.lower()
    
    def test_check_stop_loss_triggered(self, position_tracker, spot_position):
        """Test stop loss trigger detection."""
        position_tracker.add_position(spot_position)
        
        # Set stop loss at 8% below entry ($73.60)
        stop_loss_price = spot_position.entry_price * (1 - position_tracker.config.stop_loss_pct)
        
        # Update to price that triggers stop loss
        trigger_price = Decimal("70")  # Below stop loss
        result = position_tracker.update_position_price(
            spot_position.position_id,
            trigger_price
        )
        
        assert result.success is True
        assert result.stop_loss_triggered is True
        assert result.stop_loss_price is not None
    
    def test_check_take_profit_triggered(self, position_tracker, spot_position):
        """Test take profit trigger detection."""
        position_tracker.add_position(spot_position)
        
        # Update to price that triggers take profit (40% gain = $112)
        profit_price = Decimal("120")  # Above take profit
        result = position_tracker.update_position_price(
            spot_position.position_id,
            profit_price
        )
        
        assert result.success is True
        assert result.take_profit_triggered is True
        assert result.take_profit_price is not None
    
    def test_close_position_full(self, position_tracker, spot_position):
        """Test closing position completely."""
        position_tracker.add_position(spot_position)
        
        # Update price first to create some P&L
        position_tracker.update_position_price(spot_position.position_id, Decimal("85"))
        
        original_size = spot_position.size
        close_result = position_tracker.close_position(
            spot_position.position_id,
            close_size=original_size,
            close_price=Decimal("85"),
            reason="manual_close"
        )
        
        assert close_result.success is True
        assert close_result.position_id == spot_position.position_id
        assert close_result.close_size == original_size
        assert close_result.realized_pnl == Decimal("500")  # (85-80) * 100
        assert close_result.remaining_size == Decimal("0")
        
        # Check position status
        closed_position = position_tracker.get_position(spot_position.position_id)
        assert closed_position.status == PositionStatus.CLOSED
        assert position_tracker.open_positions_count == 0
    
    def test_close_position_partial(self, position_tracker, spot_position):
        """Test closing position partially."""
        position_tracker.add_position(spot_position)
        
        partial_size = Decimal("30")  # Close 30 out of 100 shares
        close_result = position_tracker.close_position(
            spot_position.position_id,
            close_size=partial_size,
            close_price=Decimal("90"),
            reason="partial_profit"
        )
        
        assert close_result.success is True
        assert close_result.close_size == partial_size
        assert close_result.realized_pnl == Decimal("300")  # (90-80) * 30
        assert close_result.remaining_size == Decimal("70")
        
        # Check position status and size
        partial_position = position_tracker.get_position(spot_position.position_id)
        assert partial_position.status == PositionStatus.PARTIAL
        assert partial_position.size == Decimal("70")  # Remaining size
        assert position_tracker.open_positions_count == 1  # Still open
    
    def test_close_position_invalid_size(self, position_tracker, spot_position):
        """Test closing position with invalid size."""
        position_tracker.add_position(spot_position)
        
        # Try to close more than position size
        close_result = position_tracker.close_position(
            spot_position.position_id,
            close_size=Decimal("150"),  # More than 100 shares
            close_price=Decimal("90")
        )
        
        assert close_result.success is False
        assert "exceeds position size" in close_result.message.lower()
    
    def test_close_position_nonexistent(self, position_tracker):
        """Test closing non-existent position."""
        close_result = position_tracker.close_position(
            uuid4(),
            close_size=Decimal("50"),
            close_price=Decimal("100")
        )
        
        assert close_result.success is False
        assert "not found" in close_result.message.lower()
    
    def test_get_positions_by_status(self, position_tracker, spot_position):
        """Test filtering positions by status."""
        position_tracker.add_position(spot_position)
        
        # All positions should be open initially
        open_positions = position_tracker.get_positions_by_status(PositionStatus.OPEN)
        assert len(open_positions) == 1
        assert spot_position.position_id in open_positions
        
        closed_positions = position_tracker.get_positions_by_status(PositionStatus.CLOSED)
        assert len(closed_positions) == 0
        
        # Close the position
        position_tracker.close_position(
            spot_position.position_id,
            close_size=spot_position.size,
            close_price=Decimal("85")
        )
        
        # Check status counts
        open_positions = position_tracker.get_positions_by_status(PositionStatus.OPEN)
        assert len(open_positions) == 0
        
        closed_positions = position_tracker.get_positions_by_status(PositionStatus.CLOSED)
        assert len(closed_positions) == 1
    
    def test_get_positions_by_chain(self, position_tracker, spot_position, perpetual_position):
        """Test filtering positions by blockchain chain."""
        position_tracker.add_position(spot_position)
        position_tracker.add_position(perpetual_position)
        
        solana_positions = position_tracker.get_positions_by_chain(Chain.SOLANA)
        assert len(solana_positions) == 1
        assert spot_position.position_id in solana_positions
        
        hyperliquid_positions = position_tracker.get_positions_by_chain(Chain.HYPERLIQUID)
        assert len(hyperliquid_positions) == 1
        assert perpetual_position.position_id in hyperliquid_positions
        
        ethereum_positions = position_tracker.get_positions_by_chain(Chain.ETHEREUM)
        assert len(ethereum_positions) == 0
    
    def test_get_positions_by_dex(self, position_tracker, spot_position, perpetual_position):
        """Test filtering positions by DEX."""
        position_tracker.add_position(spot_position)
        position_tracker.add_position(perpetual_position)
        
        jupiter_positions = position_tracker.get_positions_by_dex("jupiter")
        assert len(jupiter_positions) == 1
        assert spot_position.position_id in jupiter_positions
        
        hyperliquid_positions = position_tracker.get_positions_by_dex("hyperliquid")
        assert len(hyperliquid_positions) == 1
        assert perpetual_position.position_id in hyperliquid_positions
        
        uniswap_positions = position_tracker.get_positions_by_dex("uniswap_v3")
        assert len(uniswap_positions) == 0
    
    def test_total_unrealized_pnl(self, position_tracker, spot_position, perpetual_position):
        """Test calculating total unrealized P&L across all positions."""
        position_tracker.add_position(spot_position)
        position_tracker.add_position(perpetual_position)
        
        # Update prices to create P&L
        position_tracker.update_position_price(spot_position.position_id, Decimal("90"))  # +$1000
        position_tracker.update_position_price(perpetual_position.position_id, Decimal("50000"))  # +$2500 (with 5x leverage = $12500)
        
        total_pnl = position_tracker.total_unrealized_pnl
        expected_pnl = Decimal("1000") + (Decimal("5000") * Decimal("0.5") * Decimal("5"))  # Spot + Perpetual with leverage
        assert total_pnl == expected_pnl
    
    def test_total_market_value(self, position_tracker, spot_position, perpetual_position):
        """Test calculating total market value of all positions."""
        position_tracker.add_position(spot_position)
        position_tracker.add_position(perpetual_position)
        
        # Update prices
        position_tracker.update_position_price(spot_position.position_id, Decimal("85"))
        position_tracker.update_position_price(perpetual_position.position_id, Decimal("50000"))
        
        total_value = position_tracker.total_market_value
        expected_value = (Decimal("85") * Decimal("100")) + (Decimal("50000") * Decimal("0.5"))
        assert total_value == expected_value
    
    def test_position_summary(self, position_tracker, spot_position, perpetual_position):
        """Test getting position summary statistics."""
        position_tracker.add_position(spot_position)
        position_tracker.add_position(perpetual_position)
        
        summary = position_tracker.get_position_summary()
        
        assert summary["total_positions"] == 2
        assert summary["open_positions"] == 2
        assert summary["closed_positions"] == 0
        assert summary["partial_positions"] == 0
        assert summary["chains"] == {"SOLANA": 1, "HYPERLIQUID": 1}
        assert summary["dexs"] == {"jupiter": 1, "hyperliquid": 1}
        assert summary["position_types"] == {"SPOT": 1, "PERPETUAL": 1}
    
    def test_liquidation_check_perpetual(self, position_tracker, perpetual_position):
        """Test liquidation check for perpetual positions."""
        # Set liquidation price
        perpetual_position.liquidation_price = Decimal("36000")  # 20% below entry at 5x leverage
        position_tracker.add_position(perpetual_position)
        
        # Update to price near liquidation
        result = position_tracker.update_position_price(
            perpetual_position.position_id,
            Decimal("35000")  # Below liquidation price
        )
        
        assert result.success is True
        assert result.liquidation_risk is True
        assert result.liquidation_price == Decimal("36000")
    
    def test_funding_update_perpetual(self, position_tracker, perpetual_position):
        """Test updating funding payments for perpetual positions."""
        position_tracker.add_position(perpetual_position)
        
        funding_payment = Decimal("-5.50")  # Negative funding payment
        result = position_tracker.update_funding(
            perpetual_position.position_id,
            funding_payment
        )
        
        assert result.success is True
        assert result.funding_payment == funding_payment
        
        # Check position was updated
        updated_position = position_tracker.get_position(perpetual_position.position_id)
        assert updated_position.unrealized_funding == funding_payment


class TestPositionUpdateResult:
    """Test position update result data structure."""
    
    def test_position_update_result_creation(self):
        """Test creating position update result."""
        result = PositionUpdateResult(
            success=True,
            position_id=uuid4(),
            old_price=Decimal("100"),
            new_price=Decimal("110"),
            price_change=Decimal("10"),
            pnl_change=Decimal("100"),
            stop_loss_triggered=False,
            take_profit_triggered=True,
            take_profit_price=Decimal("140")
        )
        
        assert result.success is True
        assert result.price_change == Decimal("10")
        assert result.pnl_change == Decimal("100")
        assert result.take_profit_triggered is True
        assert result.stop_loss_triggered is False


class TestPositionCloseResult:
    """Test position close result data structure."""
    
    def test_position_close_result_creation(self):
        """Test creating position close result."""
        result = PositionCloseResult(
            success=True,
            position_id=uuid4(),
            close_size=Decimal("50"),
            close_price=Decimal("110"),
            realized_pnl=Decimal("500"),
            remaining_size=Decimal("50"),
            fees=Decimal("5"),
            reason="take_profit"
        )
        
        assert result.success is True
        assert result.close_size == Decimal("50")
        assert result.realized_pnl == Decimal("500")
        assert result.remaining_size == Decimal("50")
        assert result.reason == "take_profit"