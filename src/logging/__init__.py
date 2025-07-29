"""
Shyvr RLTE Logging Package

This package provides structured activity logging capabilities for the Shyvr RLTE system.
It's designed to coexist safely with Python's built-in logging module.

Main components:
- ActivityLogger: Production-ready activity logger for dashboard integration
- ActivityLogEntry: Structured log entry data class
- Enums for standardized logging categories, actions, and severity levels

Usage:
    from src.logging.activity_logger import ActivityLogger, ActivityCategory, ActivityAction
    
    # Create logger instance
    logger = ActivityLogger()
    await logger.start()
    
    # Log activities
    await logger.log_activity(
        category=ActivityCategory.SYSTEM,
        action=ActivityAction.START,
        source="system",
        event_type="startup",
        title="System startup completed"
    )
"""

# Import main components for convenient access
from .activity_logger import (
    ActivityLogger,
    ActivityLogEntry,
    ActivityCategory,
    ActivityAction,
    ActivitySeverity,
    TradingMode,
    ChainType,
    activity_logger,
    log_system_event,
    log_dashboard_action,
    log_trade_execution,
    log_ml_prediction,
    performance_tracker,
)

__all__ = [
    'ActivityLogger',
    'ActivityLogEntry',
    'ActivityCategory',
    'ActivityAction', 
    'ActivitySeverity',
    'TradingMode',
    'ChainType',
    'activity_logger',
    'log_system_event',
    'log_dashboard_action',
    'log_trade_execution',
    'log_ml_prediction',
    'performance_tracker',
]