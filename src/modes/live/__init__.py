"""
Live Trading Mode Modular Components

This package contains the modularized components of the live trading mode,
extracted from the monolithic live_mode.py for better maintainability.

Components:
- config: LiveModeConfig and related configuration classes
- safety: Safety systems (EmergencyStopSystem, SafetyInterlocks, LiveRiskManager)
- pnl_tracking: Real-time P&L tracking system
- portfolio_sync: Portfolio synchronization with on-chain data
- trading_executor: Live trading execution through DEX integration
- dex_integration: DEX client initialization and management
- position_management: Position and order management utilities
"""

# Import all main components for easy access
from .config import (
    LiveModeConfig, LiveModeMetrics, EmergencyStopReason, SafetyCheckResult,
    RiskValidationResult, EmergencyStopResult, PnLAlert, LiveModeError, EmergencyStopError
)
from .safety import EmergencyStopSystem, SafetyInterlocks, LiveRiskManager
from .pnl_tracking import RealTimePnLTracker
from .portfolio_sync import PortfolioSynchronizer, DiscrepancyReport, SyncHealthMetrics
from .trading_executor import LiveTradingExecutor, TradingSessionManager
from .dex_integration import DEXIntegrationManager
from .position_management import PositionManager

__all__ = [
    # Configuration classes
    'LiveModeConfig',
    'LiveModeMetrics',
    'EmergencyStopReason',
    'SafetyCheckResult',
    'RiskValidationResult',
    'EmergencyStopResult',
    'PnLAlert',
    'LiveModeError',
    'EmergencyStopError',
    
    # Safety systems
    'EmergencyStopSystem',
    'SafetyInterlocks',
    'LiveRiskManager',
    
    # Core components
    'RealTimePnLTracker',
    'PortfolioSynchronizer',
    'LiveTradingExecutor',
    'TradingSessionManager',
    'DEXIntegrationManager',
    'PositionManager',
    
    # Supporting classes
    'DiscrepancyReport',
    'SyncHealthMetrics'
]