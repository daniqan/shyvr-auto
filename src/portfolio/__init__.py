"""
Portfolio Management Module

This module provides comprehensive portfolio management and P&L tracking
capabilities for the Shyvr AI RLTE trading system. It supports multi-chain,
multi-DEX position tracking and real-time performance analytics.

Key Components:
- Portfolio: Core portfolio data structure and operations
- Position: Individual position tracking (spot + perpetual futures)
- PositionTracker: Position lifecycle management
- PnLCalculator: Real-time and historical P&L calculations  
- PortfolioManager: Central orchestrator for portfolio operations
- RiskManager: Position sizing, limits, and risk controls
- TransactionHistory: Complete transaction tracking across DEXs

Supported DEXs:
- Jupiter (Solana): Spot trading
- Hyperliquid: Perpetual futures
- Uniswap V3 (Ethereum/Base): Spot trading
"""

from .base import (
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

# Import core classes as they become available
try:
    from .position_tracker import PositionTracker, PositionUpdateResult, PositionCloseResult
except ImportError:
    PositionTracker = None
    PositionUpdateResult = None
    PositionCloseResult = None

try:
    from .pnl_calculator import (
        PnLCalculator,
        PnLCalculationResult,
        PnLAttribution,
        PnLTimeSeriesPoint,
        PnLSummary,
        PnLAggregationPeriod,
        PortfolioPnLResult,
        ChainPnLResult,
        DexPnLResult,
        TimeSeriesPnLResult,
        HistoricalPnLResult,
        FundingPnLResult,
        FeesResult,
        DexFeesResult,
        PnLAttributionResult,
        PnLCalculationError,
    )
except ImportError:
    PnLCalculator = None
    PnLCalculationResult = None
    PnLAttribution = None
    PnLTimeSeriesPoint = None
    PnLSummary = None
    PnLAggregationPeriod = None
    PortfolioPnLResult = None
    ChainPnLResult = None
    DexPnLResult = None
    TimeSeriesPnLResult = None
    HistoricalPnLResult = None
    FundingPnLResult = None
    FeesResult = None
    DexFeesResult = None
    PnLAttributionResult = None
    PnLCalculationError = None

try:
    from .portfolio_manager import (
        PortfolioManager, 
        PortfolioCreateResult, 
        PortfolioOperationResult,
        DrawdownRiskResult,
        RebalancingCheck,
        RebalancingSuggestions
    )
except ImportError:
    PortfolioManager = None
    PortfolioCreateResult = None
    PortfolioOperationResult = None
    DrawdownRiskResult = None
    RebalancingCheck = None
    RebalancingSuggestions = None

try:
    from .risk_manager import RiskManager
except ImportError:
    RiskManager = None

try:
    from .transaction_history import TransactionHistory
except ImportError:
    TransactionHistory = None

__all__ = [
    # Base data structures
    "Portfolio",
    "Position",
    "PositionType",
    "PositionStatus", 
    "Transaction",
    "TransactionType",
    "PerformanceMetrics",
    "RiskMetrics",
    "DrawdownMetrics",
    "PortfolioConfig",
    
    # Exceptions
    "PortfolioError",
    "InsufficientFundsError",
    "RiskLimitExceededError",
]

# Add core classes to __all__ if they're available
if PositionTracker is not None:
    __all__.extend(["PositionTracker", "PositionUpdateResult", "PositionCloseResult"])
if PnLCalculator is not None:
    __all__.extend([
        "PnLCalculator",
        "PnLCalculationResult",
        "PnLAttribution",
        "PnLTimeSeriesPoint",
        "PnLSummary",
        "PnLAggregationPeriod",
        "PortfolioPnLResult",
        "ChainPnLResult",
        "DexPnLResult",
        "TimeSeriesPnLResult",
        "HistoricalPnLResult",
        "FundingPnLResult",
        "FeesResult",
        "DexFeesResult",
        "PnLAttributionResult",
        "PnLCalculationError",
    ])
if PortfolioManager is not None:
    __all__.extend([
        "PortfolioManager", 
        "PortfolioCreateResult", 
        "PortfolioOperationResult",
        "DrawdownRiskResult",
        "RebalancingCheck",
        "RebalancingSuggestions"
    ])
if RiskManager is not None:
    __all__.append("RiskManager")
if TransactionHistory is not None:
    __all__.append("TransactionHistory")

# Module version
__version__ = "0.1.0"