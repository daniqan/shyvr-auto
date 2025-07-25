"""
Tests for portfolio base data structures and enums.

Following TDD methodology - these tests define the expected behavior
before implementation.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any
from uuid import UUID, uuid4

from src.portfolio.base import (
    Portfolio,
    Position,
    PositionType,
    PositionStatus,
    Transaction,
    TransactionType,
    PerformanceMetrics,
    RiskMetrics,
    DrawdownMetrics,
    PortfolioConfig,
    PortfolioError,
    InsufficientFundsError,
    RiskLimitExceededError,
)
from src.utils.base import Chain


class TestPortfolioConfig:
    """Test portfolio configuration data structure."""
    
    def test_portfolio_config_creation(self):
        """Test creating portfolio configuration with default values."""
        config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC"
        )
        
        assert config.initial_balance == Decimal("10000")
        assert config.base_currency == "USDC"
        assert config.max_position_size_pct == Decimal("0.1")  # 10% default
        assert config.max_daily_loss_pct == Decimal("0.05")  # 5% default
        assert config.max_drawdown_pct == Decimal("0.15")  # 15% default
        assert config.max_open_positions == 10
        assert config.min_trade_amount_usd == Decimal("10")
    
    def test_portfolio_config_validation(self):
        """Test portfolio configuration validation."""
        # Valid configuration
        config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC",
            max_position_size_pct=Decimal("0.05")  # 5%
        )
        assert config.max_position_size_pct == Decimal("0.05")
        
        # Invalid percentage values should raise error
        with pytest.raises(ValueError, match="max_position_size_pct must be between 0 and 1"):
            PortfolioConfig(
                initial_balance=Decimal("10000"),
                base_currency="USDC",
                max_position_size_pct=Decimal("1.5")  # 150% - invalid
            )


class TestPosition:
    """Test position data structure."""
    
    def test_position_creation_spot(self):
        """Test creating a spot position."""
        position = Position(
            position_id=uuid4(),
            symbol="SOL/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("100"),
            entry_price=Decimal("80.50"),
            current_price=Decimal("82.00"),
            status=PositionStatus.OPEN
        )
        
        assert position.symbol == "SOL/USDC"
        assert position.position_type == PositionType.SPOT
        assert position.chain == Chain.SOLANA
        assert position.dex_name == "jupiter"
        assert position.size == Decimal("100")
        assert position.entry_price == Decimal("80.50")
        assert position.current_price == Decimal("82.00")
        assert position.status == PositionStatus.OPEN
        assert position.leverage == Decimal("1")  # Default for spot
    
    def test_position_creation_perpetual(self):
        """Test creating a perpetual futures position."""
        position = Position(
            position_id=uuid4(),
            symbol="BTC-USD",
            position_type=PositionType.PERPETUAL,
            chain=Chain.HYPERLIQUID,
            dex_name="hyperliquid",
            size=Decimal("0.5"),
            entry_price=Decimal("45000"),
            current_price=Decimal("46000"),
            status=PositionStatus.OPEN,
            leverage=Decimal("10"),
            side="LONG"
        )
        
        assert position.position_type == PositionType.PERPETUAL
        assert position.leverage == Decimal("10")
        assert position.side == "LONG"
    
    def test_position_pnl_calculation(self):
        """Test position P&L calculation methods."""
        position = Position(
            position_id=uuid4(),
            symbol="ETH/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            size=Decimal("10"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2100"),
            status=PositionStatus.OPEN
        )
        
        # Test unrealized P&L
        unrealized_pnl = position.unrealized_pnl
        expected_pnl = (Decimal("2100") - Decimal("2000")) * Decimal("10")
        assert unrealized_pnl == expected_pnl  # $1000 profit
        
        # Test unrealized P&L percentage
        pnl_pct = position.unrealized_pnl_pct
        expected_pct = (Decimal("2100") - Decimal("2000")) / Decimal("2000")
        assert pnl_pct == expected_pct  # 5% gain
    
    def test_position_market_value(self):
        """Test position market value calculation."""
        position = Position(
            position_id=uuid4(),
            symbol="AVAX/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            size=Decimal("50"),
            entry_price=Decimal("25"),
            current_price=Decimal("30"),
            status=PositionStatus.OPEN
        )
        
        market_value = position.market_value
        expected_value = Decimal("50") * Decimal("30")
        assert market_value == expected_value  # $1500
    
    def test_position_cost_basis(self):
        """Test position cost basis calculation."""
        position = Position(
            position_id=uuid4(),
            symbol="MATIC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            size=Decimal("1000"),
            entry_price=Decimal("0.75"),
            current_price=Decimal("0.80"),
            status=PositionStatus.OPEN
        )
        
        cost_basis = position.cost_basis
        expected_cost = Decimal("1000") * Decimal("0.75")
        assert cost_basis == expected_cost  # $750


class TestTransaction:
    """Test transaction data structure."""
    
    def test_transaction_creation_buy(self):
        """Test creating a buy transaction."""
        transaction = Transaction(
            transaction_id=uuid4(),
            position_id=uuid4(),
            transaction_type=TransactionType.BUY,
            symbol="BTC/USDC",
            size=Decimal("0.1"),
            price=Decimal("50000"),
            timestamp=datetime.now(),
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            transaction_hash="0x123abc",
            fees=Decimal("15")
        )
        
        assert transaction.transaction_type == TransactionType.BUY
        assert transaction.symbol == "BTC/USDC"
        assert transaction.size == Decimal("0.1")
        assert transaction.price == Decimal("50000")
        assert transaction.fees == Decimal("15")
        assert transaction.chain == Chain.ETHEREUM
        assert transaction.dex_name == "uniswap_v3"
    
    def test_transaction_value_calculation(self):
        """Test transaction value calculation."""
        transaction = Transaction(
            transaction_id=uuid4(),
            position_id=uuid4(),
            transaction_type=TransactionType.SELL,
            symbol="SOL/USDC",
            size=Decimal("20"),
            price=Decimal("100"),
            timestamp=datetime.now(),
            chain=Chain.SOLANA,
            dex_name="jupiter",
            transaction_hash="abc123def",
            fees=Decimal("2")
        )
        
        value = transaction.value
        expected_value = Decimal("20") * Decimal("100")
        assert value == expected_value  # $2000
        
        net_value = transaction.net_value
        expected_net = expected_value - Decimal("2")  # Subtract fees
        assert net_value == expected_net  # $1998


class TestPerformanceMetrics:
    """Test performance metrics data structure."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            total_pnl=Decimal("1500"),
            realized_pnl=Decimal("800"),
            unrealized_pnl=Decimal("700"),
            total_fees=Decimal("50"),
            win_rate=Decimal("0.65"),  # 65%
            avg_win=Decimal("120"),
            avg_loss=Decimal("-80"),
            profit_factor=Decimal("1.8"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown=Decimal("0.12"),  # 12%
            total_trades=25
        )
        
        assert metrics.total_pnl == Decimal("1500")
        assert metrics.realized_pnl == Decimal("800")
        assert metrics.unrealized_pnl == Decimal("700")
        assert metrics.win_rate == Decimal("0.65")
        assert metrics.total_trades == 25
        assert metrics.sharpe_ratio == Decimal("1.2")
    
    def test_performance_metrics_roi_calculation(self):
        """Test ROI calculation in performance metrics."""
        metrics = PerformanceMetrics(
            total_pnl=Decimal("2000"),
            realized_pnl=Decimal("2000"),
            unrealized_pnl=Decimal("0"),
            total_fees=Decimal("100"),
            win_rate=Decimal("0.6"),
            avg_win=Decimal("150"),
            avg_loss=Decimal("-100"),
            profit_factor=Decimal("2.0"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.08"),
            total_trades=20,
            initial_balance=Decimal("10000")
        )
        
        roi = metrics.roi
        expected_roi = Decimal("2000") / Decimal("10000")  # 20%
        assert roi == expected_roi


class TestPortfolio:
    """Test portfolio data structure."""
    
    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        config = PortfolioConfig(
            initial_balance=Decimal("50000"),
            base_currency="USDC"
        )
        
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            config=config,
            cash_balance=Decimal("45000"),
            total_value=Decimal("52000")
        )
        
        assert portfolio.name == "Test Portfolio"
        assert portfolio.config.initial_balance == Decimal("50000")
        assert portfolio.cash_balance == Decimal("45000")
        assert portfolio.total_value == Decimal("52000")
        assert len(portfolio.positions) == 0
        assert len(portfolio.transactions) == 0
    
    def test_portfolio_add_position(self):
        """Test adding positions to portfolio."""
        config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC"
        )
        
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            config=config,
            cash_balance=Decimal("8000"),
            total_value=Decimal("10000")
        )
        
        position = Position(
            position_id=uuid4(),
            symbol="BTC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            size=Decimal("0.05"),
            entry_price=Decimal("40000"),
            current_price=Decimal("42000"),
            status=PositionStatus.OPEN
        )
        
        portfolio.add_position(position)
        
        assert len(portfolio.positions) == 1
        assert portfolio.positions[position.position_id] == position
    
    def test_portfolio_total_unrealized_pnl(self):
        """Test portfolio total unrealized P&L calculation."""
        config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC"
        )
        
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio", 
            config=config,
            cash_balance=Decimal("5000"),
            total_value=Decimal("10000")
        )
        
        # Add positions with different P&L
        position1 = Position(
            position_id=uuid4(),
            symbol="ETH/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            size=Decimal("2"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2200"),  # +$400 profit
            status=PositionStatus.OPEN
        )
        
        position2 = Position(
            position_id=uuid4(),
            symbol="SOL/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("20"),
            entry_price=Decimal("100"),
            current_price=Decimal("90"),  # -$200 loss
            status=PositionStatus.OPEN
        )
        
        portfolio.add_position(position1)
        portfolio.add_position(position2)
        
        total_unrealized_pnl = portfolio.total_unrealized_pnl
        expected_pnl = Decimal("400") + Decimal("-200")  # $200 net profit
        assert total_unrealized_pnl == expected_pnl


class TestEnums:
    """Test portfolio enums."""
    
    def test_position_type_enum(self):
        """Test PositionType enum values."""
        assert PositionType.SPOT.value == "spot"
        assert PositionType.PERPETUAL.value == "perpetual"
        assert PositionType.FUTURES.value == "futures"
        assert PositionType.OPTION.value == "option"
    
    def test_position_status_enum(self):
        """Test PositionStatus enum values."""
        assert PositionStatus.OPEN.value == "open"
        assert PositionStatus.CLOSED.value == "closed"
        assert PositionStatus.PARTIAL.value == "partial"
        assert PositionStatus.LIQUIDATED.value == "liquidated"
    
    def test_transaction_type_enum(self):
        """Test TransactionType enum values."""
        assert TransactionType.BUY.value == "buy"
        assert TransactionType.SELL.value == "sell"
        assert TransactionType.DEPOSIT.value == "deposit"
        assert TransactionType.WITHDRAWAL.value == "withdrawal"
        assert TransactionType.FEE.value == "fee"
        assert TransactionType.FUNDING.value == "funding"


class TestExceptions:
    """Test portfolio exception classes."""
    
    def test_portfolio_error(self):
        """Test base PortfolioError exception."""
        with pytest.raises(PortfolioError, match="Test portfolio error"):
            raise PortfolioError("Test portfolio error")
    
    def test_insufficient_funds_error(self):
        """Test InsufficientFundsError exception."""
        with pytest.raises(InsufficientFundsError, match="Insufficient funds"):
            raise InsufficientFundsError("Insufficient funds")
    
    def test_risk_limit_exceeded_error(self):
        """Test RiskLimitExceededError exception."""
        with pytest.raises(RiskLimitExceededError, match="Risk limit exceeded"):
            raise RiskLimitExceededError("Risk limit exceeded")