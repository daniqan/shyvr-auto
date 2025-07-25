"""
Trading modes module for the Shyvr AI RLTE system.

This module provides the complete mode switching foundation including:
- Base mode classes and data structures
- Mode manager for coordinating multiple modes
- Concrete mode implementations (Analysis, Simulation, Trading)
- Mode scheduling and lifecycle management

Key Components:
- ModeBase: Abstract base class for all trading modes
- ModeManager: Central coordinator for mode lifecycle
- ModeScheduler: Time-based mode execution scheduling
- Trading modes: AnalysisMode, SimulationMode, TradingMode

Usage:
    from src.modes import ModeManager, ModeType, ModeConfig
    
    # Create mode configuration
    config = ModeConfig(mode_type=ModeType.ANALYSIS, enabled=True)
    
    # Initialize mode manager
    manager = ModeManager(config=manager_config, portfolio=portfolio)
    await manager.initialize()
    
    # Register and start a mode
    mode_id = await manager.register_mode(config)
    await manager.start_mode(mode_id)
"""

# Base mode classes and enums
from .base import (
    ModeType,
    ModeStatus,
    ModeBase,
    ModeConfig,
    ModeResult,
    TradingMode,
    SimulationMode,
    ModeError,
    ModeNotFoundError,
    ModeConfigError,
    InvalidModeTransitionError,
)

# Import base AnalysisMode for compatibility
from .base import AnalysisMode as BaseAnalysisMode

# Import enhanced AnalysisMode
from .analysis_mode import AnalysisMode

# Mode manager and scheduling
from .mode_manager import (
    ModeManager,
    ModeManagerConfig,
    ModeScheduler,
    ScheduledMode,
    ModeManagerError,
    ModeTransitionError,
    DuplicateModeError,
)

# Export version for compatibility
__version__ = "1.0.0"

# Define public API
__all__ = [
    # Base mode classes
    "ModeType",
    "ModeStatus", 
    "ModeBase",
    "ModeConfig",
    "ModeResult",
    "TradingMode",
    "AnalysisMode",
    "BaseAnalysisMode",
    "SimulationMode",
    
    # Mode manager classes
    "ModeManager",
    "ModeManagerConfig",
    "ModeScheduler",
    "ScheduledMode",
    
    # Exception classes
    "ModeError",
    "ModeNotFoundError",
    "ModeConfigError",
    "InvalidModeTransitionError",
    "ModeManagerError",
    "ModeTransitionError",
    "DuplicateModeError",
    
    # Version
    "__version__",
]


# Convenience functions for common operations
def create_analysis_mode(portfolio, **kwargs):
    """Create a configured analysis mode.
    
    Args:
        portfolio: Portfolio instance for the mode
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (mode_config, mode_id) for easy registration
    """
    config = ModeConfig(
        mode_type=ModeType.ANALYSIS,
        enabled=True,
        **kwargs
    )
    return config


def create_simulation_mode(portfolio, initial_balance=10000, **kwargs):
    """Create a configured simulation mode.
    
    Args:
        portfolio: Portfolio instance for the mode
        initial_balance: Starting balance for simulation
        **kwargs: Additional configuration parameters
        
    Returns:
        ModeConfig configured for simulation
    """
    parameters = {
        "initial_balance": initial_balance,
        "enable_fees": True,
        "slippage_bps": 10,
        **kwargs.get("parameters", {})
    }
    
    config = ModeConfig(
        mode_type=ModeType.SIMULATION,
        enabled=True,
        parameters=parameters,
        **{k: v for k, v in kwargs.items() if k != "parameters"}
    )
    return config


def create_trading_mode(portfolio, **kwargs):
    """Create a configured live trading mode.
    
    Args:
        portfolio: Portfolio instance for the mode
        **kwargs: Additional configuration parameters
        
    Returns:
        ModeConfig configured for live trading
    """
    parameters = {
        "max_position_size": 1000,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.15,
        **kwargs.get("parameters", {})
    }
    
    config = ModeConfig(
        mode_type=ModeType.LIVE_TRADING,
        enabled=True,
        parameters=parameters,
        **{k: v for k, v in kwargs.items() if k != "parameters"}
    )
    return config


# Add convenience functions to public API
__all__.extend([
    "create_analysis_mode",
    "create_simulation_mode", 
    "create_trading_mode",
])