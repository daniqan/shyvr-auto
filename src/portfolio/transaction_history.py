"""
Transaction History tracking for all DEXs (Jupiter, Hyperliquid, Uniswap V3).

This module provides comprehensive transaction history management across multiple
blockchains and DEXs, supporting:
- Transaction CRUD operations with async support
- Multi-chain and multi-DEX transaction tracking  
- Advanced filtering and querying capabilities
- Transaction aggregation and reporting
- Export functionality for CSV, JSON, and tax formats
- Transaction status tracking (pending, confirmed, failed)
- Integration with existing portfolio components
"""

import asyncio
import csv
import json
import io
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import UUID, uuid4
import structlog

from src.portfolio.base import (
    Transaction,
    TransactionType,
    Portfolio,
    PortfolioError
)
from src.utils.base import Chain


logger = structlog.get_logger()


class TransactionStatus(Enum):
    """Transaction status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExportFormat(Enum):
    """Export format enumeration."""
    CSV = "csv"
    JSON = "json"
    TSV = "tsv"


@dataclass
class TransactionFilter:
    """Filter criteria for transaction queries."""
    transaction_type: Optional[TransactionType] = None
    symbol: Optional[str] = None
    chain: Optional[Chain] = None
    dex_name: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_amount: Optional[Decimal] = None
    max_amount: Optional[Decimal] = None
    status: Optional[TransactionStatus] = None
    position_id: Optional[UUID] = None


@dataclass
class TransactionQuery:
    """Transaction query with pagination and sorting."""
    page: int = 1
    page_size: int = 50
    sort_by: str = "timestamp"  # timestamp, amount, symbol, type
    sort_order: str = "desc"    # asc, desc
    transaction_filter: Optional[TransactionFilter] = None


@dataclass
class TransactionQueryResult:
    """Result of transaction query with pagination info."""
    transactions: List[Transaction]
    total_count: int
    page: int
    page_size: int
    total_pages: int


@dataclass
class AggregationGroup:
    """Aggregation group for transaction reporting."""
    group_key: str
    transaction_count: int
    total_volume: Decimal
    total_fees: Decimal
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")
    net_volume: Decimal = Decimal("0")
    time_period: Optional[str] = None


@dataclass
class TransactionAggregation:
    """Transaction aggregation result."""
    groups: List[AggregationGroup]
    total_transactions: int
    total_volume: Decimal
    total_fees: Decimal
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class TransactionReport:
    """Comprehensive transaction report."""
    total_transactions: int
    total_volume: Decimal
    total_fees: Decimal
    buy_count: int
    sell_count: int
    net_flow: Decimal
    by_symbol: List[AggregationGroup]
    by_dex: List[AggregationGroup]
    by_chain: List[AggregationGroup]
    by_type: List[AggregationGroup]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    generated_at: datetime = field(default_factory=datetime.now)


class TransactionHistoryError(PortfolioError):
    """Base transaction history error."""
    pass


class TransactionNotFoundError(TransactionHistoryError):
    """Error when transaction is not found."""
    pass


class InvalidTransactionError(TransactionHistoryError):
    """Error when transaction data is invalid."""
    pass


class TransactionHistory:
    """
    Comprehensive transaction history management for multi-chain, multi-DEX trading.
    
    This class provides full transaction lifecycle management including:
    - CRUD operations with validation and indexing
    - Advanced filtering and querying with pagination
    - Aggregation and reporting across multiple dimensions
    - Export functionality for various formats
    - Real-time status tracking and updates
    - Integration with portfolio and position management
    """
    
    def __init__(self, portfolio: Portfolio):
        """
        Initialize TransactionHistory with portfolio reference.
        
        Args:
            portfolio: Portfolio instance to track transactions for
        """
        self.portfolio = portfolio
        self.portfolio_id = portfolio.portfolio_id
        self.created_at = datetime.now()
        self.logger = logger.bind(
            portfolio_id=str(self.portfolio_id),
            component="transaction_history"
        )
        
        # Transaction indexing for fast lookups
        self._transaction_index: Dict[UUID, Transaction] = {}
        self._status_index: Dict[TransactionStatus, List[UUID]] = defaultdict(list)
        self._symbol_index: Dict[str, List[UUID]] = defaultdict(list)
        self._chain_index: Dict[Chain, List[UUID]] = defaultdict(list)
        self._dex_index: Dict[str, List[UUID]] = defaultdict(list)
        self._position_index: Dict[UUID, List[UUID]] = defaultdict(list)
        
        # Callbacks for real-time updates
        self._update_callbacks: List[Callable[[Transaction], None]] = []
        
        # Build initial indexes from existing portfolio transactions
        self._rebuild_indexes()
        
        self.logger.info("Transaction history initialized")
    
    @property
    def transactions(self) -> List[Transaction]:
        """Get all transactions from portfolio."""
        return self.portfolio.transactions
    
    @property
    def total_transactions(self) -> int:
        """Get total number of transactions."""
        return len(self.portfolio.transactions)
    
    def _rebuild_indexes(self) -> None:
        """Rebuild all indexes from current portfolio transactions."""
        self._transaction_index.clear()
        self._status_index.clear()
        self._symbol_index.clear()
        self._chain_index.clear()
        self._dex_index.clear()
        self._position_index.clear()
        
        for transaction in self.portfolio.transactions:
            self._add_to_indexes(transaction)
    
    def _add_to_indexes(self, transaction: Transaction) -> None:
        """Add transaction to all relevant indexes."""
        tx_id = transaction.transaction_id
        
        # Main index
        self._transaction_index[tx_id] = transaction
        
        # Status index
        status = TransactionStatus(transaction.metadata.get("status", TransactionStatus.CONFIRMED.value))
        self._status_index[status].append(tx_id)
        
        # Symbol index
        self._symbol_index[transaction.symbol].append(tx_id)
        
        # Chain index
        self._chain_index[transaction.chain].append(tx_id)
        
        # DEX index
        self._dex_index[transaction.dex_name].append(tx_id)
        
        # Position index
        if transaction.position_id:
            self._position_index[transaction.position_id].append(tx_id)
    
    def _remove_from_indexes(self, transaction: Transaction) -> None:
        """Remove transaction from all indexes."""
        tx_id = transaction.transaction_id
        
        # Main index
        self._transaction_index.pop(tx_id, None)
        
        # Status index
        status = TransactionStatus(transaction.metadata.get("status", TransactionStatus.CONFIRMED.value))
        if tx_id in self._status_index[status]:
            self._status_index[status].remove(tx_id)
        
        # Symbol index
        if tx_id in self._symbol_index[transaction.symbol]:
            self._symbol_index[transaction.symbol].remove(tx_id)
        
        # Chain index
        if tx_id in self._chain_index[transaction.chain]:
            self._chain_index[transaction.chain].remove(tx_id)
        
        # DEX index
        if tx_id in self._dex_index[transaction.dex_name]:
            self._dex_index[transaction.dex_name].remove(tx_id)
        
        # Position index
        if transaction.position_id and tx_id in self._position_index[transaction.position_id]:
            self._position_index[transaction.position_id].remove(tx_id)
    
    def _validate_transaction(self, transaction: Transaction) -> None:
        """Validate transaction data."""
        # Allow empty symbol for certain transaction types like fees and funding
        if (transaction.transaction_type not in [TransactionType.FEE, TransactionType.FUNDING] and 
            (not transaction.symbol or transaction.symbol.strip() == "")):
            raise InvalidTransactionError("Transaction symbol cannot be empty")
        
        if transaction.size < 0:
            raise InvalidTransactionError("Transaction size cannot be negative")
        
        if transaction.price < 0:
            raise InvalidTransactionError("Transaction price cannot be negative")
        
        if not transaction.transaction_hash or transaction.transaction_hash.strip() == "":
            raise InvalidTransactionError("Transaction hash cannot be empty")
    
    async def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add a new transaction to the history.
        
        Args:
            transaction: Transaction to add
            
        Returns:
            True if transaction was added successfully
            
        Raises:
            InvalidTransactionError: If transaction data is invalid
        """
        try:
            # Check for duplicate
            if transaction.transaction_id in self._transaction_index:
                raise InvalidTransactionError(
                    f"Transaction {transaction.transaction_id} already exists"
                )
            
            # Validate transaction data
            self._validate_transaction(transaction)
            
            # Add to portfolio and indexes
            self.portfolio.add_transaction(transaction)
            self._add_to_indexes(transaction)
            
            # Trigger callbacks
            for callback in self._update_callbacks:
                try:
                    callback(transaction)
                except Exception as e:
                    self.logger.warning("Update callback failed", error=str(e))
            
            self.logger.info(
                "Transaction added to history",
                transaction_id=str(transaction.transaction_id),
                symbol=transaction.symbol,
                type=transaction.transaction_type.value
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to add transaction",
                transaction_id=str(transaction.transaction_id),
                error=str(e)
            )
            raise
    
    async def get_transaction(self, transaction_id: UUID) -> Optional[Transaction]:
        """
        Get transaction by ID.
        
        Args:
            transaction_id: Transaction ID to retrieve
            
        Returns:
            Transaction if found, None otherwise
        """
        return self._transaction_index.get(transaction_id)
    
    async def update_transaction_status(
        self, 
        transaction_id: UUID, 
        status: TransactionStatus
    ) -> bool:
        """
        Update transaction status.
        
        Args:
            transaction_id: Transaction ID to update
            status: New transaction status
            
        Returns:
            True if status was updated successfully
        """
        transaction = await self.get_transaction(transaction_id)
        if not transaction:
            return False
        
        # Remove from old status index
        old_status = TransactionStatus(
            transaction.metadata.get("status", TransactionStatus.CONFIRMED.value)
        )
        if transaction_id in self._status_index[old_status]:
            self._status_index[old_status].remove(transaction_id)
        
        # Update status and add to new index
        transaction.metadata["status"] = status
        self._status_index[status].append(transaction_id)
        
        # Trigger callbacks
        for callback in self._update_callbacks:
            try:
                callback(transaction)
            except Exception as e:
                self.logger.warning("Update callback failed", error=str(e))
        
        self.logger.info(
            "Transaction status updated",
            transaction_id=str(transaction_id),
            old_status=old_status.value,
            new_status=status.value
        )
        
        return True
    
    async def update_transaction_status_batch(
        self, 
        transaction_ids: List[UUID], 
        status: TransactionStatus
    ) -> bool:
        """
        Update status for multiple transactions.
        
        Args:
            transaction_ids: List of transaction IDs to update
            status: New transaction status
            
        Returns:
            True if all updates were successful
        """
        success_count = 0
        for tx_id in transaction_ids:
            if await self.update_transaction_status(tx_id, status):
                success_count += 1
        
        return success_count == len(transaction_ids)
    
    async def delete_transaction(self, transaction_id: UUID) -> bool:
        """
        Delete transaction from history.
        
        Args:
            transaction_id: Transaction ID to delete
            
        Returns:
            True if transaction was deleted successfully
        """
        transaction = await self.get_transaction(transaction_id)
        if not transaction:
            return False
        
        # Remove from portfolio
        self.portfolio.transactions = [
            tx for tx in self.portfolio.transactions 
            if tx.transaction_id != transaction_id
        ]
        
        # Remove from indexes
        self._remove_from_indexes(transaction)
        
        self.logger.info(
            "Transaction deleted from history",
            transaction_id=str(transaction_id)
        )
        
        return True
    
    async def filter_transactions(
        self, 
        transaction_filter: TransactionFilter
    ) -> List[Transaction]:
        """
        Filter transactions based on criteria.
        
        Args:
            transaction_filter: Filter criteria
            
        Returns:
            List of transactions matching filter
        """
        transactions = self.portfolio.transactions.copy()
        
        if transaction_filter.transaction_type:
            transactions = [
                tx for tx in transactions 
                if tx.transaction_type == transaction_filter.transaction_type
            ]
        
        if transaction_filter.symbol:
            transactions = [
                tx for tx in transactions 
                if tx.symbol == transaction_filter.symbol
            ]
        
        if transaction_filter.chain:
            transactions = [
                tx for tx in transactions 
                if tx.chain == transaction_filter.chain
            ]
        
        if transaction_filter.dex_name:
            transactions = [
                tx for tx in transactions 
                if tx.dex_name == transaction_filter.dex_name
            ]
        
        if transaction_filter.start_date:
            transactions = [
                tx for tx in transactions 
                if tx.timestamp >= transaction_filter.start_date
            ]
        
        if transaction_filter.end_date:
            transactions = [
                tx for tx in transactions 
                if tx.timestamp <= transaction_filter.end_date
            ]
        
        if transaction_filter.min_amount is not None:
            transactions = [
                tx for tx in transactions 
                if tx.size * tx.price >= transaction_filter.min_amount
            ]
        
        if transaction_filter.max_amount is not None:
            transactions = [
                tx for tx in transactions 
                if tx.size * tx.price <= transaction_filter.max_amount
            ]
        
        if transaction_filter.status:
            transactions = [
                tx for tx in transactions 
                if TransactionStatus(tx.metadata.get("status", TransactionStatus.CONFIRMED.value)) == transaction_filter.status
            ]
        
        if transaction_filter.position_id:
            transactions = [
                tx for tx in transactions 
                if tx.position_id == transaction_filter.position_id
            ]
        
        return transactions
    
    async def query_transactions(self, query: TransactionQuery) -> TransactionQueryResult:
        """
        Query transactions with pagination and sorting.
        
        Args:
            query: Transaction query parameters
            
        Returns:
            TransactionQueryResult with paginated results
        """
        # Apply filter if provided
        if query.transaction_filter:
            transactions = await self.filter_transactions(query.transaction_filter)
        else:
            transactions = self.portfolio.transactions.copy()
        
        # Sort transactions
        reverse = query.sort_order.lower() == "desc"
        
        if query.sort_by == "timestamp":
            transactions.sort(key=lambda tx: tx.timestamp, reverse=reverse)
        elif query.sort_by == "amount":
            transactions.sort(key=lambda tx: tx.size * tx.price, reverse=reverse)
        elif query.sort_by == "symbol":
            transactions.sort(key=lambda tx: tx.symbol, reverse=reverse)
        elif query.sort_by == "type":
            transactions.sort(key=lambda tx: tx.transaction_type.value, reverse=reverse)
        
        # Apply pagination
        total_count = len(transactions)
        total_pages = (total_count + query.page_size - 1) // query.page_size
        
        start_idx = (query.page - 1) * query.page_size
        end_idx = start_idx + query.page_size
        paginated_transactions = transactions[start_idx:end_idx]
        
        return TransactionQueryResult(
            transactions=paginated_transactions,
            total_count=total_count,
            page=query.page,
            page_size=query.page_size,
            total_pages=total_pages
        )
    
    async def get_transactions_by_status(
        self, 
        status: TransactionStatus
    ) -> List[Transaction]:
        """
        Get all transactions with specific status.
        
        Args:
            status: Transaction status to filter by
            
        Returns:
            List of transactions with the specified status
        """
        tx_ids = self._status_index.get(status, [])
        return [
            self._transaction_index[tx_id] 
            for tx_id in tx_ids 
            if tx_id in self._transaction_index
        ]
    
    async def get_failed_transactions(self) -> List[Transaction]:
        """Get all failed transactions."""
        return await self.get_transactions_by_status(TransactionStatus.FAILED)
    
    async def get_transactions_for_position(self, position_id: UUID) -> List[Transaction]:
        """
        Get all transactions for a specific position.
        
        Args:
            position_id: Position ID to get transactions for
            
        Returns:
            List of transactions linked to the position
        """
        tx_ids = self._position_index.get(position_id, [])
        return [
            self._transaction_index[tx_id] 
            for tx_id in tx_ids 
            if tx_id in self._transaction_index
        ]
    
    async def aggregate_transactions(
        self,
        group_by: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> TransactionAggregation:
        """
        Aggregate transactions by specified dimension.
        
        Args:
            group_by: Grouping dimension (symbol, dex_name, chain, transaction_type, daily)
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            TransactionAggregation with grouped results
        """
        # Filter by date range if provided
        transactions = self.portfolio.transactions
        if start_date:
            transactions = [tx for tx in transactions if tx.timestamp >= start_date]
        if end_date:
            transactions = [tx for tx in transactions if tx.timestamp <= end_date]
        
        # Group transactions
        groups_data = defaultdict(lambda: {
            "transactions": [],
            "total_volume": Decimal("0"),
            "total_fees": Decimal("0"),
            "buy_volume": Decimal("0"),
            "sell_volume": Decimal("0")
        })
        
        for tx in transactions:
            if group_by == "symbol":
                key = tx.symbol
            elif group_by == "dex_name":
                key = tx.dex_name
            elif group_by == "chain":
                key = tx.chain.value
            elif group_by == "transaction_type":
                key = tx.transaction_type.value
            elif group_by == "daily":
                key = tx.timestamp.strftime("%Y-%m-%d")
            else:
                raise ValueError(f"Unsupported group_by: {group_by}")
            
            groups_data[key]["transactions"].append(tx)
            
            volume = tx.size * tx.price
            groups_data[key]["total_volume"] += volume
            groups_data[key]["total_fees"] += tx.total_fees
            
            if tx.transaction_type == TransactionType.BUY:
                groups_data[key]["buy_volume"] += volume
            elif tx.transaction_type == TransactionType.SELL:
                groups_data[key]["sell_volume"] += volume
        
        # Create aggregation groups
        groups = []
        total_volume = Decimal("0")
        total_fees = Decimal("0")
        
        for key, data in groups_data.items():
            net_volume = data["buy_volume"] - data["sell_volume"]
            
            group = AggregationGroup(
                group_key=key,
                transaction_count=len(data["transactions"]),
                total_volume=data["total_volume"],
                total_fees=data["total_fees"],
                buy_volume=data["buy_volume"],
                sell_volume=data["sell_volume"],
                net_volume=net_volume,
                time_period=key if group_by == "daily" else None
            )
            groups.append(group)
            
            total_volume += data["total_volume"]
            total_fees += data["total_fees"]
        
        return TransactionAggregation(
            groups=groups,
            total_transactions=len(transactions),
            total_volume=total_volume,
            total_fees=total_fees,
            start_date=start_date,
            end_date=end_date
        )
    
    async def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_aggregations: bool = True
    ) -> TransactionReport:
        """
        Generate comprehensive transaction report.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            include_aggregations: Whether to include aggregation breakdowns
            
        Returns:
            TransactionReport with comprehensive analysis
        """
        # Filter transactions by date
        transactions = self.portfolio.transactions
        if start_date:
            transactions = [tx for tx in transactions if tx.timestamp >= start_date]
        if end_date:
            transactions = [tx for tx in transactions if tx.timestamp <= end_date]
        
        # Calculate basic metrics
        total_volume = sum(tx.size * tx.price for tx in transactions)
        total_fees = sum(tx.total_fees for tx in transactions)
        
        buy_transactions = [tx for tx in transactions if tx.transaction_type == TransactionType.BUY]
        sell_transactions = [tx for tx in transactions if tx.transaction_type == TransactionType.SELL]
        
        buy_volume = sum(tx.size * tx.price for tx in buy_transactions)
        sell_volume = sum(tx.size * tx.price for tx in sell_transactions)
        net_flow = buy_volume - sell_volume
        
        # Generate aggregations if requested
        by_symbol = []
        by_dex = []
        by_chain = []
        by_type = []
        
        if include_aggregations:
            symbol_agg = await self.aggregate_transactions("symbol", start_date, end_date)
            by_symbol = symbol_agg.groups
            
            dex_agg = await self.aggregate_transactions("dex_name", start_date, end_date)
            by_dex = dex_agg.groups
            
            chain_agg = await self.aggregate_transactions("chain", start_date, end_date)
            by_chain = chain_agg.groups
            
            type_agg = await self.aggregate_transactions("transaction_type", start_date, end_date)
            by_type = type_agg.groups
        
        return TransactionReport(
            total_transactions=len(transactions),
            total_volume=total_volume,
            total_fees=total_fees,
            buy_count=len(buy_transactions),
            sell_count=len(sell_transactions),
            net_flow=net_flow,
            by_symbol=by_symbol,
            by_dex=by_dex,
            by_chain=by_chain,
            by_type=by_type,
            start_date=start_date,
            end_date=end_date
        )
    
    async def export_transactions(
        self,
        format: ExportFormat = ExportFormat.CSV,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        transaction_filter: Optional[TransactionFilter] = None,
        include_metadata: bool = False
    ) -> str:
        """
        Export transactions to specified format.
        
        Args:
            format: Export format (CSV, JSON, TSV)
            start_date: Optional start date filter
            end_date: Optional end date filter
            transaction_filter: Optional additional filters
            include_metadata: Whether to include metadata in export
            
        Returns:
            Formatted export data as string
        """
        # Apply filters
        transactions = self.portfolio.transactions
        
        if start_date:
            transactions = [tx for tx in transactions if tx.timestamp >= start_date]
        if end_date:
            transactions = [tx for tx in transactions if tx.timestamp <= end_date]
        
        if transaction_filter:
            filter_obj = TransactionFilter(
                transaction_type=transaction_filter.transaction_type,
                symbol=transaction_filter.symbol,
                chain=transaction_filter.chain,
                dex_name=transaction_filter.dex_name,
                status=transaction_filter.status,
                position_id=transaction_filter.position_id
            )
            transactions = await self.filter_transactions(filter_obj)
        
        if format == ExportFormat.CSV:
            return self._export_csv(transactions, include_metadata)
        elif format == ExportFormat.JSON:
            return self._export_json(transactions, include_metadata)
        elif format == ExportFormat.TSV:
            return self._export_tsv(transactions, include_metadata)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_csv(self, transactions: List[Transaction], include_metadata: bool) -> str:
        """Export transactions to CSV format."""
        output = io.StringIO()
        
        fieldnames = [
            "transaction_id", "transaction_type", "symbol", "size", "price",
            "timestamp", "chain", "dex_name", "fees", "gas_fees", "transaction_hash"
        ]
        
        if include_metadata:
            fieldnames.append("metadata")
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for tx in transactions:
            row = {
                "transaction_id": str(tx.transaction_id),
                "transaction_type": tx.transaction_type.value,
                "symbol": tx.symbol,
                "size": str(tx.size),
                "price": str(tx.price),
                "timestamp": tx.timestamp.isoformat(),
                "chain": tx.chain.value,
                "dex_name": tx.dex_name,
                "fees": str(tx.fees),
                "gas_fees": str(tx.gas_fees) if tx.gas_fees else "",
                "transaction_hash": tx.transaction_hash
            }
            
            if include_metadata:
                row["metadata"] = json.dumps(tx.metadata)
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def _export_json(self, transactions: List[Transaction], include_metadata: bool) -> str:
        """Export transactions to JSON format."""
        data = {
            "transactions": [],
            "summary": {
                "total_count": len(transactions),
                "total_volume": str(sum(tx.size * tx.price for tx in transactions)),
                "total_fees": str(sum(tx.total_fees for tx in transactions)),
                "exported_at": datetime.now().isoformat()
            }
        }
        
        for tx in transactions:
            tx_data = {
                "transaction_id": str(tx.transaction_id),
                "transaction_type": tx.transaction_type.value,
                "symbol": tx.symbol,
                "size": str(tx.size),
                "price": str(tx.price),
                "timestamp": tx.timestamp.isoformat(),
                "chain": tx.chain.value,
                "dex_name": tx.dex_name,
                "fees": str(tx.fees),
                "gas_fees": str(tx.gas_fees) if tx.gas_fees else None,
                "transaction_hash": tx.transaction_hash
            }
            
            if include_metadata:
                tx_data["metadata"] = tx.metadata
            
            data["transactions"].append(tx_data)
        
        return json.dumps(data, indent=2)
    
    def _export_tsv(self, transactions: List[Transaction], include_metadata: bool) -> str:
        """Export transactions to TSV format."""
        # Reuse CSV logic but with tab delimiter
        output = io.StringIO()
        
        fieldnames = [
            "transaction_id", "transaction_type", "symbol", "size", "price",
            "timestamp", "chain", "dex_name", "fees", "gas_fees", "transaction_hash"
        ]
        
        if include_metadata:
            fieldnames.append("metadata")
        
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for tx in transactions:
            row = {
                "transaction_id": str(tx.transaction_id),
                "transaction_type": tx.transaction_type.value,
                "symbol": tx.symbol,
                "size": str(tx.size),
                "price": str(tx.price),
                "timestamp": tx.timestamp.isoformat(),
                "chain": tx.chain.value,
                "dex_name": tx.dex_name,
                "fees": str(tx.fees),
                "gas_fees": str(tx.gas_fees) if tx.gas_fees else "",
                "transaction_hash": tx.transaction_hash
            }
            
            if include_metadata:
                row["metadata"] = json.dumps(tx.metadata)
            
            writer.writerow(row)
        
        return output.getvalue()
    
    async def export_for_taxes(
        self,
        tax_year: int,
        include_fees: bool = True
    ) -> Dict[str, Any]:
        """
        Export transactions in tax-friendly format.
        
        Args:
            tax_year: Tax year to export for
            include_fees: Whether to include fee calculations
            
        Returns:
            Dictionary with tax-relevant transaction data
        """
        # Filter transactions for tax year
        start_date = datetime(tax_year, 1, 1)
        end_date = datetime(tax_year, 12, 31, 23, 59, 59)
        
        transactions = [
            tx for tx in self.portfolio.transactions 
            if start_date <= tx.timestamp <= end_date
        ]
        
        # Calculate capital gains/losses (simplified)
        buy_transactions = [tx for tx in transactions if tx.transaction_type == TransactionType.BUY]
        sell_transactions = [tx for tx in transactions if tx.transaction_type == TransactionType.SELL]
        
        total_fees = sum(tx.total_fees for tx in transactions) if include_fees else Decimal("0")
        
        tax_data = {
            "tax_year": tax_year,
            "capital_gains": {
                "total_buys": len(buy_transactions),
                "total_sells": len(sell_transactions),
                "buy_volume": str(sum(tx.size * tx.price for tx in buy_transactions)),
                "sell_volume": str(sum(tx.size * tx.price for tx in sell_transactions))
            },
            "trading_fees": str(total_fees),
            "summary": {
                "total_transactions": len(transactions),
                "total_fees": str(total_fees),
                "generated_at": datetime.now().isoformat()
            },
            "transactions": [
                {
                    "date": tx.timestamp.strftime("%Y-%m-%d"),
                    "type": tx.transaction_type.value,
                    "symbol": tx.symbol,
                    "amount": str(tx.size),
                    "price": str(tx.price),
                    "value": str(tx.size * tx.price),
                    "fees": str(tx.total_fees),
                    "dex": tx.dex_name,
                    "chain": tx.chain.value
                }
                for tx in transactions
            ]
        }
        
        return tax_data
    
    def add_update_callback(self, callback: Callable[[Transaction], None]) -> None:
        """
        Add callback for transaction updates.
        
        Args:
            callback: Function to call on transaction updates
        """
        self._update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable[[Transaction], None]) -> None:
        """
        Remove update callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)