"""
Trading Safety Management Module

This module provides comprehensive trading safety mechanisms including:
- Pre-trade validation
- Position size limits
- Risk exposure checks
- Rate limiting
- Emergency stop mechanisms
- Balance verification

Key Components:
- TradingSafetyManager: Main safety validation coordinator
- TradingSafetyConfig: Configuration for safety parameters
- PreTradeValidationResult: Validation result data structure
- Various validation utilities and enums
"""

# Import safety event logging components first (no complex dependencies)
from .safety_event_logging import (
    SafetyEventLogger,
    SafetyEventType,
    SafetyEventSeverity,
    SafetyEvent,
    SafetyEventConfig,
    SafetyEventError,
    SafetyEventValidationError,
    SafetyEventStorageError,
)

# Import trading safety components (may have dependencies)
try:
    from .trading_safety_manager import (
        TradingSafetyManager,
        TradingSafetyConfig,
        PreTradeValidationResult,
        ValidationStatus,
        ValidationReason,
        RateLimitState,
        PositionSizeValidation,
        RiskExposureValidation,
    )
except ImportError:
    # Skip trading safety imports if dependencies not available
    pass

__all__ = [
    # Safety Event Logging
    "SafetyEventLogger",
    "SafetyEventType",
    "SafetyEventSeverity",
    "SafetyEvent",
    "SafetyEventConfig",
    "SafetyEventError",
    "SafetyEventValidationError",
    "SafetyEventStorageError",
    # Trading Safety (if available)
    "TradingSafetyManager",
    "TradingSafetyConfig",
    "PreTradeValidationResult",
    "ValidationStatus",
    "ValidationReason",
    "RateLimitState",
    "PositionSizeValidation",
    "RiskExposureValidation",
]