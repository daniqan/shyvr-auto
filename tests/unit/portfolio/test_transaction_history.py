"""
Tests for TransactionHistory component following TDD methodology.

This module provides comprehensive test coverage for transaction history
tracking across all DEXs (Jupiter, Hyperliquid, Uniswap V3) including:
- Transaction CRUD operations
- Multi-chain and multi-DEX support
- Filtering and querying capabilities
- Aggregation and reporting features
- Export functionality
- Integration with existing portfolio components
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from unittest.mock import Mock, AsyncMock, patch
import csv
import json
import io

from src.portfolio.base import (
    Transaction,
    TransactionType,
    Position,
    PositionType,
    PositionStatus,
    Portfolio,
    PortfolioConfig,
)
from src.portfolio.transaction_history import (
    TransactionHistory,
    TransactionFilter,
    TransactionQuery,
    TransactionStatus,
    TransactionAggregation,
    TransactionReport,
    ExportFormat,
    TransactionHistoryError,
    TransactionNotFoundError,
    InvalidTransactionError,
)
from src.utils.base import Chain


# Module-level fixtures available to all test classes
@pytest.fixture
def portfolio_config() -> PortfolioConfig:
    """Create test portfolio configuration."""
    return PortfolioConfig(
        initial_balance=Decimal("10000"),
        base_currency="USDC"
    )


@pytest.fixture
def portfolio(portfolio_config) -> Portfolio:
    """Create test portfolio."""
    return Portfolio(
        portfolio_id=uuid4(),
        name="Test Portfolio",
        config=portfolio_config,
        cash_balance=Decimal("10000"),
        total_value=Decimal("10000")
    )


@pytest.fixture
def transaction_history(portfolio) -> TransactionHistory:
    """Create TransactionHistory instance."""
    return TransactionHistory(portfolio=portfolio)


@pytest.fixture
def sample_transaction() -> Transaction:
    """Create sample transaction for testing."""
    return Transaction(
        transaction_id=uuid4(),
        position_id=uuid4(),
        transaction_type=TransactionType.BUY,
        symbol="BTC/USDC",
        size=Decimal("0.1"),
        price=Decimal("50000"),
        timestamp=datetime.now(),
        chain=Chain.SOLANA,
        dex_name="jupiter",
        transaction_hash="abc123",
        fees=Decimal("5"),
        gas_fees=None,
        slippage_bps=25
    )


class TestTransactionHistory:
    """Test TransactionHistory core functionality."""
    
    def test_transaction_history_initialization(self, transaction_history, portfolio):
        """Test TransactionHistory initialization."""
        assert transaction_history.portfolio == portfolio
        assert transaction_history.portfolio_id == portfolio.portfolio_id
        assert len(transaction_history.transactions) == 0
        assert transaction_history.total_transactions == 0
        assert isinstance(transaction_history.created_at, datetime)
    
    async def test_add_transaction(self, transaction_history, sample_transaction):
        """Test adding a transaction to history."""
        result = await transaction_history.add_transaction(sample_transaction)
        
        assert result is True
        assert transaction_history.total_transactions == 1
        assert sample_transaction.transaction_id in transaction_history._transaction_index
        
        # Verify transaction was added to portfolio
        assert len(transaction_history.portfolio.transactions) == 1
        assert transaction_history.portfolio.transactions[0] == sample_transaction
    
    async def test_add_duplicate_transaction(self, transaction_history, sample_transaction):
        """Test adding duplicate transaction raises error."""
        await transaction_history.add_transaction(sample_transaction)
        
        with pytest.raises(InvalidTransactionError):
            await transaction_history.add_transaction(sample_transaction)
    
    async def test_get_transaction(self, transaction_history, sample_transaction):
        """Test retrieving transaction by ID."""
        await transaction_history.add_transaction(sample_transaction)
        
        retrieved = await transaction_history.get_transaction(sample_transaction.transaction_id)
        assert retrieved == sample_transaction
    
    async def test_get_nonexistent_transaction(self, transaction_history):
        """Test retrieving non-existent transaction returns None."""
        non_existent_id = uuid4()
        result = await transaction_history.get_transaction(non_existent_id)
        assert result is None
    
    async def test_update_transaction_status(self, transaction_history, sample_transaction):
        """Test updating transaction status."""
        await transaction_history.add_transaction(sample_transaction)
        
        success = await transaction_history.update_transaction_status(
            sample_transaction.transaction_id,
            TransactionStatus.CONFIRMED
        )
        
        assert success is True
        updated = await transaction_history.get_transaction(sample_transaction.transaction_id)
        assert updated.metadata.get("status") == TransactionStatus.CONFIRMED
    
    async def test_delete_transaction(self, transaction_history, sample_transaction):
        """Test deleting transaction from history."""
        await transaction_history.add_transaction(sample_transaction)
        
        success = await transaction_history.delete_transaction(sample_transaction.transaction_id)
        assert success is True
        assert transaction_history.total_transactions == 0
        
        result = await transaction_history.get_transaction(sample_transaction.transaction_id)
        assert result is None


class TestTransactionFiltering:
    """Test transaction filtering and querying capabilities."""
    
    async def setup_populated_history(self, transaction_history) -> TransactionHistory:
        """Create transaction history with multiple transactions."""
        transactions = [
            Transaction(
                transaction_id=uuid4(),
                position_id=uuid4(),
                transaction_type=TransactionType.BUY,
                symbol="BTC/USDC",
                size=Decimal("0.1"),
                price=Decimal("50000"),
                timestamp=datetime.now() - timedelta(days=1),
                chain=Chain.SOLANA,
                dex_name="jupiter",
                transaction_hash="hash1",
                fees=Decimal("5")
            ),
            Transaction(
                transaction_id=uuid4(),
                position_id=uuid4(),
                transaction_type=TransactionType.SELL,
                symbol="ETH/USDC",
                size=Decimal("1.0"),
                price=Decimal("3000"),
                timestamp=datetime.now() - timedelta(hours=12),
                chain=Chain.ETHEREUM,
                dex_name="uniswap_v3",
                transaction_hash="hash2",
                fees=Decimal("15"),
                gas_fees=Decimal("25")
            ),
            Transaction(
                transaction_id=uuid4(),
                position_id=uuid4(),
                transaction_type=TransactionType.BUY,
                symbol="SOL/USDC",
                size=Decimal("10"),
                price=Decimal("100"),
                timestamp=datetime.now(),
                chain=Chain.SOLANA,
                dex_name="jupiter",
                transaction_hash="hash3",
                fees=Decimal("2")
            ),
            Transaction(
                transaction_id=uuid4(),
                position_id=None,
                transaction_type=TransactionType.FUNDING,
                symbol="BTC-PERP",
                size=Decimal("0"),
                price=Decimal("0"),
                timestamp=datetime.now() - timedelta(hours=6),
                chain=Chain.HYPERLIQUID,
                dex_name="hyperliquid",
                transaction_hash="hash4",
                fees=Decimal("0.5")
            )
        ]
        
        for tx in transactions:
            await transaction_history.add_transaction(tx)
        
        return transaction_history
    
    async def test_filter_by_transaction_type(self, transaction_history):
        """Test filtering transactions by type."""
        populated_history = await self.setup_populated_history(transaction_history)
        buy_filter = TransactionFilter(transaction_type=TransactionType.BUY)
        buy_transactions = await populated_history.filter_transactions(buy_filter)
        
        assert len(buy_transactions) == 2
        assert all(tx.transaction_type == TransactionType.BUY for tx in buy_transactions)
    
    async def test_filter_by_chain(self, transaction_history):
        """Test filtering transactions by blockchain."""
        populated_history = await self.setup_populated_history(transaction_history)
        solana_filter = TransactionFilter(chain=Chain.SOLANA)
        solana_transactions = await populated_history.filter_transactions(solana_filter)
        
        assert len(solana_transactions) == 2
        assert all(tx.chain == Chain.SOLANA for tx in solana_transactions)
    
    async def test_filter_by_dex(self, transaction_history):
        """Test filtering transactions by DEX."""
        populated_history = await self.setup_populated_history(transaction_history)
        jupiter_filter = TransactionFilter(dex_name="jupiter")
        jupiter_transactions = await populated_history.filter_transactions(jupiter_filter)
        
        assert len(jupiter_transactions) == 2
        assert all(tx.dex_name == "jupiter" for tx in jupiter_transactions)
    
    async def test_filter_by_symbol(self, transaction_history):
        """Test filtering transactions by trading symbol."""
        populated_history = await self.setup_populated_history(transaction_history)
        btc_filter = TransactionFilter(symbol="BTC/USDC")
        btc_transactions = await populated_history.filter_transactions(btc_filter)
        
        assert len(btc_transactions) == 1
        assert btc_transactions[0].symbol == "BTC/USDC"
    
    async def test_filter_by_date_range(self, transaction_history):
        """Test filtering transactions by date range."""
        populated_history = await self.setup_populated_history(transaction_history)
        start_date = datetime.now() - timedelta(hours=24)
        end_date = datetime.now() - timedelta(hours=1)
        
        date_filter = TransactionFilter(
            start_date=start_date,
            end_date=end_date
        )
        filtered = await populated_history.filter_transactions(date_filter)
        
        assert len(filtered) == 2  # Should include ETH and funding transactions
    
    async def test_filter_by_amount_range(self, transaction_history):
        """Test filtering transactions by amount range."""
        populated_history = await self.setup_populated_history(transaction_history)
        amount_filter = TransactionFilter(
            min_amount=Decimal("1000"),  # ETH transaction (1.0 * 3000 = 3000)
            max_amount=Decimal("10000")
        )
        filtered = await populated_history.filter_transactions(amount_filter)
        
        assert len(filtered) == 3  # ETH, BTC, and SOL transactions
        for tx in filtered:
            amount = tx.size * tx.price
            assert Decimal("1000") <= amount <= Decimal("10000")
    
    async def test_complex_filter(self, transaction_history):
        """Test filtering with multiple criteria."""
        populated_history = await self.setup_populated_history(transaction_history)
        complex_filter = TransactionFilter(
            transaction_type=TransactionType.BUY,
            chain=Chain.SOLANA,
            start_date=datetime.now() - timedelta(days=2)
        )
        filtered = await populated_history.filter_transactions(complex_filter)
        
        assert len(filtered) == 2
        for tx in filtered:
            assert tx.transaction_type == TransactionType.BUY
            assert tx.chain == Chain.SOLANA
    
    async def test_query_with_pagination(self, transaction_history):
        """Test querying transactions with pagination."""
        populated_history = await self.setup_populated_history(transaction_history)
        query = TransactionQuery(
            page=1,
            page_size=2,
            sort_by="timestamp",
            sort_order="desc"
        )
        
        result = await populated_history.query_transactions(query)
        
        assert len(result.transactions) == 2
        assert result.total_count == 4
        assert result.page == 1
        assert result.total_pages == 2
        
        # Check sorting (most recent first)
        assert result.transactions[0].timestamp > result.transactions[1].timestamp
    
    async def test_query_with_sorting(self, transaction_history):
        """Test querying transactions with sorting."""
        populated_history = await self.setup_populated_history(transaction_history)
        # Sort by amount (ascending)
        query = TransactionQuery(
            sort_by="amount",
            sort_order="asc"
        )
        
        result = await populated_history.query_transactions(query)
        
        # Should be sorted by transaction value (size * price)
        amounts = [tx.size * tx.price for tx in result.transactions]
        assert amounts == sorted(amounts)


class TestTransactionAggregation:
    """Test transaction aggregation and reporting."""
    
    async def setup_trading_history(self, transaction_history) -> TransactionHistory:
        """Create history with diverse trading data."""
        # Add transactions for aggregation testing
        transactions = [
            # BTC trades
            Transaction(
                transaction_id=uuid4(), position_id=uuid4(),
                transaction_type=TransactionType.BUY, symbol="BTC/USDC",
                size=Decimal("0.1"), price=Decimal("50000"),
                timestamp=datetime.now() - timedelta(days=30),
                chain=Chain.SOLANA, dex_name="jupiter",
                transaction_hash="hash1", fees=Decimal("5")
            ),
            Transaction(
                transaction_id=uuid4(), position_id=uuid4(),
                transaction_type=TransactionType.SELL, symbol="BTC/USDC",
                size=Decimal("0.05"), price=Decimal("52000"),
                timestamp=datetime.now() - timedelta(days=15),
                chain=Chain.SOLANA, dex_name="jupiter",
                transaction_hash="hash2", fees=Decimal("2.5")
            ),
            # ETH trades
            Transaction(
                transaction_id=uuid4(), position_id=uuid4(),
                transaction_type=TransactionType.BUY, symbol="ETH/USDC",
                size=Decimal("2.0"), price=Decimal("3000"),
                timestamp=datetime.now() - timedelta(days=20),
                chain=Chain.ETHEREUM, dex_name="uniswap_v3",
                transaction_hash="hash3", fees=Decimal("20"), gas_fees=Decimal("30")
            ),
            # Fees and funding
            Transaction(
                transaction_id=uuid4(), position_id=None,
                transaction_type=TransactionType.FEE, symbol="",
                size=Decimal("0"), price=Decimal("0"),
                timestamp=datetime.now() - timedelta(days=10),
                chain=Chain.HYPERLIQUID, dex_name="hyperliquid",
                transaction_hash="hash4", fees=Decimal("1")
            ),
        ]
        
        for tx in transactions:
            await transaction_history.add_transaction(tx)
        
        return transaction_history
    
    async def test_aggregate_by_symbol(self, transaction_history):
        """Test aggregating transactions by trading symbol."""
        trading_history = await self.setup_trading_history(transaction_history)
        aggregation = await trading_history.aggregate_transactions(
            group_by="symbol",
            start_date=datetime.now() - timedelta(days=35)
        )
        
        assert isinstance(aggregation, TransactionAggregation)
        assert len(aggregation.groups) >= 2  # BTC/USDC and ETH/USDC
        
        # Check BTC aggregation
        btc_group = next(g for g in aggregation.groups if g.group_key == "BTC/USDC")
        assert btc_group.transaction_count == 2
        assert btc_group.total_volume > Decimal("0")
        assert btc_group.total_fees == Decimal("7.5")  # 5 + 2.5
    
    async def test_aggregate_by_dex(self, transaction_history):
        """Test aggregating transactions by DEX."""
        trading_history = await self.setup_trading_history(transaction_history)
        aggregation = await trading_history.aggregate_transactions(
            group_by="dex_name"
        )
        
        jupiter_group = next(g for g in aggregation.groups if g.group_key == "jupiter")
        assert jupiter_group.transaction_count == 2  # 2 BTC transactions
        
        uniswap_group = next(g for g in aggregation.groups if g.group_key == "uniswap_v3")
        assert uniswap_group.transaction_count == 1  # 1 ETH transaction
    
    async def test_aggregate_by_chain(self, transaction_history):
        """Test aggregating transactions by blockchain."""
        trading_history = await self.setup_trading_history(transaction_history)
        aggregation = await trading_history.aggregate_transactions(
            group_by="chain"
        )
        
        solana_group = next(g for g in aggregation.groups if g.group_key == Chain.SOLANA.value)
        assert solana_group.transaction_count == 2
        
        ethereum_group = next(g for g in aggregation.groups if g.group_key == Chain.ETHEREUM.value)
        assert ethereum_group.transaction_count == 1
    
    async def test_aggregate_by_transaction_type(self, transaction_history):
        """Test aggregating transactions by type."""
        trading_history = await self.setup_trading_history(transaction_history)
        aggregation = await trading_history.aggregate_transactions(
            group_by="transaction_type"
        )
        
        buy_group = next(g for g in aggregation.groups if g.group_key == TransactionType.BUY.value)
        assert buy_group.transaction_count == 2
        
        sell_group = next(g for g in aggregation.groups if g.group_key == TransactionType.SELL.value)
        assert sell_group.transaction_count == 1
    
    async def test_time_series_aggregation(self, transaction_history):
        """Test time-based aggregation of transactions."""
        trading_history = await self.setup_trading_history(transaction_history)
        aggregation = await trading_history.aggregate_transactions(
            group_by="daily",
            start_date=datetime.now() - timedelta(days=35)
        )
        
        # Should have daily groupings
        assert len(aggregation.groups) > 0
        
        # Check that time series data is present
        for group in aggregation.groups:
            assert group.time_period is not None
            assert isinstance(group.total_volume, Decimal)
    
    async def test_generate_report(self, transaction_history):
        """Test generating comprehensive transaction report."""
        trading_history = await self.setup_trading_history(transaction_history)
        report = await trading_history.generate_report(
            start_date=datetime.now() - timedelta(days=35),
            end_date=datetime.now(),
            include_aggregations=True
        )
        
        assert isinstance(report, TransactionReport)
        assert report.total_transactions == 4
        assert report.total_volume > Decimal("0")
        assert report.total_fees > Decimal("0")
        assert len(report.by_symbol) > 0  # Symbol aggregations
        assert len(report.by_dex) > 0     # DEX aggregations
        assert len(report.by_chain) > 0   # Chain aggregations
        
        # Check performance metrics
        assert report.buy_count >= 0
        assert report.sell_count >= 0
        assert isinstance(report.net_flow, Decimal)


class TestTransactionExport:
    """Test transaction export functionality."""
    
    async def setup_export_history(self, transaction_history) -> TransactionHistory:
        """Create history for export testing."""
        transactions = [
            Transaction(
                transaction_id=uuid4(), position_id=uuid4(),
                transaction_type=TransactionType.BUY, symbol="BTC/USDC",
                size=Decimal("0.1"), price=Decimal("50000"),
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                chain=Chain.SOLANA, dex_name="jupiter",
                transaction_hash="hash1", fees=Decimal("5"),
                metadata={"order_id": "ord123"}
            ),
            Transaction(
                transaction_id=uuid4(), position_id=uuid4(),
                transaction_type=TransactionType.SELL, symbol="ETH/USDC",
                size=Decimal("1.0"), price=Decimal("3000"),
                timestamp=datetime(2024, 1, 16, 14, 45, 0),
                chain=Chain.ETHEREUM, dex_name="uniswap_v3",
                transaction_hash="hash2", fees=Decimal("15"),
                gas_fees=Decimal("25")
            )
        ]
        
        for tx in transactions:
            await transaction_history.add_transaction(tx)
        
        return transaction_history
    
    async def test_export_to_csv(self, transaction_history):
        """Test exporting transactions to CSV format."""
        export_history = await self.setup_export_history(transaction_history)
        csv_data = await export_history.export_transactions(
            format=ExportFormat.CSV,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1)
        )
        
        assert isinstance(csv_data, str)
        
        # Parse CSV and validate
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        rows = list(csv_reader)
        
        assert len(rows) == 2
        
        # Check CSV headers
        expected_headers = [
            "transaction_id", "transaction_type", "symbol", "size", "price",
            "timestamp", "chain", "dex_name", "fees", "gas_fees", "transaction_hash"
        ]
        for header in expected_headers:
            assert header in rows[0].keys()
        
        # Validate first row
        assert rows[0]["symbol"] == "BTC/USDC"
        assert rows[0]["transaction_type"] == "buy"
        assert rows[0]["chain"] == "solana"
    
    async def test_export_to_json(self, transaction_history):
        """Test exporting transactions to JSON format."""
        export_history = await self.setup_export_history(transaction_history)
        json_data = await export_history.export_transactions(
            format=ExportFormat.JSON,
            include_metadata=True
        )
        
        data = json.loads(json_data)
        
        assert "transactions" in data
        assert "summary" in data
        assert len(data["transactions"]) == 2
        
        # Check transaction structure
        tx = data["transactions"][0]
        assert "transaction_id" in tx
        assert "symbol" in tx
        assert "timestamp" in tx
        assert "metadata" in tx  # Should include metadata
    
    async def test_export_filtered_transactions(self, transaction_history):
        """Test exporting filtered subset of transactions."""
        export_history = await self.setup_export_history(transaction_history)
        # Export only BTC transactions
        csv_data = await export_history.export_transactions(
            format=ExportFormat.CSV,
            transaction_filter=TransactionFilter(symbol="BTC/USDC")
        )
        
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        rows = list(csv_reader)
        
        assert len(rows) == 1
        assert rows[0]["symbol"] == "BTC/USDC"
    
    async def test_export_with_tax_format(self, transaction_history):
        """Test exporting transactions in tax-friendly format."""
        export_history = await self.setup_export_history(transaction_history)
        tax_data = await export_history.export_for_taxes(
            tax_year=2024,
            include_fees=True
        )
        
        assert isinstance(tax_data, dict)
        assert "capital_gains" in tax_data
        assert "trading_fees" in tax_data
        assert "summary" in tax_data
        
        # Should calculate gains/losses
        assert tax_data["summary"]["total_fees"] > Decimal("0")
        assert "transactions" in tax_data


class TestTransactionStatus:
    """Test transaction status tracking and updates."""
    
    async def test_track_pending_transaction(self, transaction_history, sample_transaction):
        """Test tracking pending transaction."""
        # Add transaction with pending status
        sample_transaction.metadata["status"] = TransactionStatus.PENDING
        await transaction_history.add_transaction(sample_transaction)
        
        pending_txs = await transaction_history.get_transactions_by_status(
            TransactionStatus.PENDING
        )
        
        assert len(pending_txs) == 1
        assert pending_txs[0].transaction_id == sample_transaction.transaction_id
    
    async def test_update_transaction_status_batch(self, transaction_history):
        """Test batch updating transaction statuses."""
        # Add multiple pending transactions
        tx_ids = []
        for i in range(3):
            tx = Transaction(
                transaction_id=uuid4(), position_id=uuid4(),
                transaction_type=TransactionType.BUY, symbol=f"TOKEN{i}/USDC",
                size=Decimal("1"), price=Decimal("100"),
                timestamp=datetime.now(), chain=Chain.SOLANA, dex_name="jupiter",
                transaction_hash=f"hash{i}", fees=Decimal("1"),
                metadata={"status": TransactionStatus.PENDING}
            )
            await transaction_history.add_transaction(tx)
            tx_ids.append(tx.transaction_id)
        
        # Update all to confirmed
        success = await transaction_history.update_transaction_status_batch(
            tx_ids, TransactionStatus.CONFIRMED
        )
        
        assert success is True
        
        confirmed_txs = await transaction_history.get_transactions_by_status(
            TransactionStatus.CONFIRMED
        )
        assert len(confirmed_txs) == 3
    
    async def test_get_failed_transactions(self, transaction_history):
        """Test retrieving failed transactions."""
        # Add failed transaction
        failed_tx = Transaction(
            transaction_id=uuid4(), position_id=uuid4(),
            transaction_type=TransactionType.BUY, symbol="FAIL/USDC",
            size=Decimal("1"), price=Decimal("100"),
            timestamp=datetime.now(), chain=Chain.SOLANA, dex_name="jupiter",
            transaction_hash="fail_hash", fees=Decimal("0"),
            metadata={"status": TransactionStatus.FAILED, "error": "Insufficient funds"}
        )
        await transaction_history.add_transaction(failed_tx)
        
        failed_txs = await transaction_history.get_failed_transactions()
        
        assert len(failed_txs) == 1
        assert failed_txs[0].metadata["error"] == "Insufficient funds"


class TestTransactionIntegration:
    """Test integration with Portfolio, Position, and DEX components."""
    
    async def test_integration_with_portfolio(self, transaction_history, sample_transaction):
        """Test transaction history integration with portfolio."""
        await transaction_history.add_transaction(sample_transaction)
        
        # Check that portfolio was updated
        portfolio_txs = transaction_history.portfolio.transactions
        assert len(portfolio_txs) == 1
        assert portfolio_txs[0] == sample_transaction
    
    async def test_link_transaction_to_position(self, transaction_history):
        """Test linking transaction to existing position."""
        # Create position first
        position_id = uuid4()
        position = Position(
            position_id=position_id, symbol="BTC/USDC",
            position_type=PositionType.SPOT, chain=Chain.SOLANA,
            dex_name="jupiter", size=Decimal("0.1"),
            entry_price=Decimal("50000"), current_price=Decimal("50000"),
            status=PositionStatus.OPEN
        )
        transaction_history.portfolio.add_position(position)
        
        # Create transaction linked to position
        linked_tx = Transaction(
            transaction_id=uuid4(), position_id=position_id,
            transaction_type=TransactionType.BUY, symbol="BTC/USDC",
            size=Decimal("0.1"), price=Decimal("50000"),
            timestamp=datetime.now(), chain=Chain.SOLANA, dex_name="jupiter",
            transaction_hash="linked_hash", fees=Decimal("5")
        )
        
        await transaction_history.add_transaction(linked_tx)
        
        # Test getting transactions for position
        position_txs = await transaction_history.get_transactions_for_position(position_id)
        assert len(position_txs) == 1
        assert position_txs[0].position_id == position_id
    
    async def test_multi_chain_transaction_support(self, transaction_history):
        """Test transactions across multiple chains."""
        chains_txs = [
            (Chain.SOLANA, "jupiter"),
            (Chain.ETHEREUM, "uniswap_v3"),
            (Chain.BASE, "uniswap_v3"),
            (Chain.HYPERLIQUID, "hyperliquid")
        ]
        
        for chain, dex in chains_txs:
            tx = Transaction(
                transaction_id=uuid4(), position_id=uuid4(),
                transaction_type=TransactionType.BUY, symbol=f"TOKEN/{chain.value}",
                size=Decimal("1"), price=Decimal("100"),
                timestamp=datetime.now(), chain=chain, dex_name=dex,
                transaction_hash=f"hash_{chain.value}", fees=Decimal("1")
            )
            await transaction_history.add_transaction(tx)
        
        # Test chain-specific filtering
        for chain, _ in chains_txs:
            chain_txs = await transaction_history.filter_transactions(
                TransactionFilter(chain=chain)
            )
            assert len(chain_txs) == 1
            assert chain_txs[0].chain == chain
    
    async def test_real_time_transaction_monitoring(self, transaction_history):
        """Test real-time transaction monitoring and updates."""
        # Mock real-time updates
        callbacks = []
        
        def mock_callback(transaction: Transaction):
            callbacks.append(transaction)
        
        transaction_history.add_update_callback(mock_callback)
        
        # Add transaction
        tx = Transaction(
            transaction_id=uuid4(), position_id=uuid4(),
            transaction_type=TransactionType.BUY, symbol="MONITOR/USDC",
            size=Decimal("1"), price=Decimal("100"),
            timestamp=datetime.now(), chain=Chain.SOLANA, dex_name="jupiter",
            transaction_hash="monitor_hash", fees=Decimal("1"),
            metadata={"status": TransactionStatus.PENDING}
        )
        
        await transaction_history.add_transaction(tx)
        
        # Update status
        await transaction_history.update_transaction_status(
            tx.transaction_id, TransactionStatus.CONFIRMED
        )
        
        # Check callbacks were triggered
        assert len(callbacks) >= 1  # At least one callback for the update


class TestTransactionHistoryEdgeCases:
    """Test edge cases and error conditions."""
    
    async def test_empty_transaction_history(self, transaction_history):
        """Test operations on empty transaction history."""
        assert transaction_history.total_transactions == 0
        
        # Aggregations should work with empty data
        aggregation = await transaction_history.aggregate_transactions(group_by="symbol")
        assert len(aggregation.groups) == 0
        
        # Exports should work with empty data
        csv_data = await transaction_history.export_transactions(format=ExportFormat.CSV)
        assert isinstance(csv_data, str)
        assert "transaction_id" in csv_data  # Header should be present
    
    async def test_invalid_transaction_data(self, transaction_history):
        """Test adding invalid transaction data."""
        invalid_tx = Transaction(
            transaction_id=uuid4(), position_id=uuid4(),
            transaction_type=TransactionType.BUY, symbol="",  # Invalid empty symbol
            size=Decimal("-1"),  # Invalid negative size
            price=Decimal("0"),   # Invalid zero price
            timestamp=datetime.now(), chain=Chain.SOLANA, dex_name="jupiter",
            transaction_hash="", fees=Decimal("1")
        )
        
        with pytest.raises(InvalidTransactionError):
            await transaction_history.add_transaction(invalid_tx)
    
    async def test_large_dataset_performance(self, transaction_history):
        """Test performance with large number of transactions."""
        # Add many transactions
        batch_size = 1000
        start_time = datetime.now()
        
        for i in range(batch_size):
            tx = Transaction(
                transaction_id=uuid4(), position_id=uuid4(),
                transaction_type=TransactionType.BUY, symbol=f"TOKEN{i%10}/USDC",
                size=Decimal("1"), price=Decimal("100"),
                timestamp=datetime.now() - timedelta(seconds=i),
                chain=Chain.SOLANA, dex_name="jupiter",
                transaction_hash=f"hash{i}", fees=Decimal("1")
            )
            await transaction_history.add_transaction(tx)
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        # Should be reasonably fast (less than 10 seconds for 1000 transactions)
        assert elapsed < 10.0
        assert transaction_history.total_transactions == batch_size
        
        # Test querying large dataset
        query_start = datetime.now()
        results = await transaction_history.filter_transactions(
            TransactionFilter(transaction_type=TransactionType.BUY)
        )
        query_end = datetime.now()
        query_elapsed = (query_end - query_start).total_seconds()
        
        assert len(results) == batch_size
        assert query_elapsed < 1.0  # Query should be fast
    
    async def test_concurrent_transaction_operations(self, transaction_history):
        """Test concurrent transaction operations."""
        async def add_transactions(start_idx: int, count: int):
            for i in range(start_idx, start_idx + count):
                tx = Transaction(
                    transaction_id=uuid4(), position_id=uuid4(),
                    transaction_type=TransactionType.BUY, symbol=f"CONCURRENT{i}/USDC",
                    size=Decimal("1"), price=Decimal("100"),
                    timestamp=datetime.now(), chain=Chain.SOLANA, dex_name="jupiter",
                    transaction_hash=f"concurrent{i}", fees=Decimal("1")
                )
                await transaction_history.add_transaction(tx)
        
        # Run concurrent adds
        await asyncio.gather(
            add_transactions(0, 50),
            add_transactions(50, 50),
            add_transactions(100, 50)
        )
        
        assert transaction_history.total_transactions == 150
    
    async def test_memory_usage_optimization(self, transaction_history):
        """Test memory usage optimization for large datasets."""
        # This would test that the transaction history doesn't hold
        # all transactions in memory unnecessarily
        import sys
        
        initial_size = sys.getsizeof(transaction_history)
        
        # Add transactions
        for i in range(100):
            tx = Transaction(
                transaction_id=uuid4(), position_id=uuid4(),
                transaction_type=TransactionType.BUY, symbol=f"MEM{i}/USDC",
                size=Decimal("1"), price=Decimal("100"),
                timestamp=datetime.now(), chain=Chain.SOLANA, dex_name="jupiter",
                transaction_hash=f"mem{i}", fees=Decimal("1")
            )
            await transaction_history.add_transaction(tx)
        
        final_size = sys.getsizeof(transaction_history)
        
        # Size growth should be reasonable (not linear with number of transactions)
        # This tests that we're not holding all data in memory
        size_growth = final_size - initial_size
        assert size_growth < 100000  # Less than 100KB growth for 100 transactions