"""
Live Trading Mode Configuration

This module contains configuration classes and enums for live trading mode.
Extracted from live_mode.py for better maintainability.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional
from datetime import datetime
from uuid import UUID


class EmergencyStopReason(Enum):
    """Reasons for emergency stop activation."""
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    SYSTEM_ERROR = "system_error"
    MANUAL_STOP = "manual_stop"
    DEX_FAILURES = "dex_failures"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    PORTFOLIO_SYNC_FAILURE = "portfolio_sync_failure"
    MARKET_VOLATILITY = "market_volatility"
    FLASH_CRASH_DETECTED = "flash_crash_detected"
    MIN_PORTFOLIO_VALUE = "min_portfolio_value"
    EXTREME_PORTFOLIO_RISK = "extreme_portfolio_risk"
    ATTENTION_ANOMALY = "attention_anomaly"


class SafetyCheckResult(Enum):
    """Results of safety checks."""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


@dataclass
class LiveModeConfig:
    """Configuration for live trading mode."""
    initial_balance: Decimal = Decimal("50000")
    enable_real_trading: bool = True
    max_position_size_pct: Decimal = Decimal("0.1")
    max_daily_loss_pct: Decimal = Decimal("0.05")
    max_drawdown_pct: Decimal = Decimal("0.15")
    emergency_drawdown_pct: Decimal = Decimal("0.25")
    stop_loss_pct: Decimal = Decimal("0.08")
    take_profit_pct: Decimal = Decimal("0.4")
    max_slippage_bps: int = 100
    max_open_positions: int = 15
    min_trade_interval_seconds: int = 5
    position_timeout_minutes: int = 60
    order_timeout_seconds: int = 30
    max_concurrent_orders: int = 5
    
    # Safety and monitoring
    enable_emergency_stop: bool = True
    safety_check_frequency_seconds: int = 10
    enable_risk_monitoring: bool = True
    
    # Portfolio synchronization
    enable_portfolio_sync: bool = True
    sync_frequency_seconds: int = 30
    
    # P&L tracking
    enable_real_time_pnl: bool = True
    pnl_update_frequency_seconds: int = 5
    loss_alert_threshold: Decimal = Decimal("0.05")
    
    # Trading hours
    trading_hours_start: int = 0
    trading_hours_end: int = 24
    enable_weekends: bool = True
    
    # DEX configuration
    dex_preference_order: List[str] = field(default_factory=lambda: ["jupiter", "uniswap_v3", "hyperliquid"])
    enable_cross_dex_arbitrage: bool = False
    
    # Experience collection
    enable_experience_collection: bool = True
    experience_buffer_size: int = 10000
    enable_rl_feedback: bool = True
    
    # ML-RL integration
    enable_ml_rl_integration: bool = True
    ml_rl_weight: Decimal = Decimal("0.6")  # 60% ML-RL, 40% traditional signals
    
    # Continuous learning integration
    enable_continuous_learning: bool = True
    enable_model_hot_swapping: bool = True
    enable_automated_deployment: bool = True
    enable_performance_monitoring: bool = True
    enable_performance_feedback: bool = True
    learning_check_frequency_seconds: int = 30
    learning_trigger_threshold: int = 1000
    deployment_safety_threshold: Decimal = Decimal("0.05")
    performance_rollback_threshold: Decimal = Decimal("-0.10")
    
    # Enhanced safety parameters
    warning_drawdown_pct: Decimal = Decimal("0.08")      # 8% warning threshold
    critical_drawdown_pct: Decimal = Decimal("0.12")     # 12% critical threshold
    volatility_circuit_breaker_pct: Decimal = Decimal("0.20")  # 20% volatility CB
    consecutive_failure_limit: int = 3                   # 3 consecutive failures
    portfolio_value_check_frequency: int = 5             # Check every 5 seconds
    emergency_liquidation_enabled: bool = True
    min_portfolio_value_pct: Decimal = Decimal("0.50")   # Stop if < 50% of initial
    
    # Enhanced risk parameters
    max_sector_exposure_pct: Decimal = Decimal("0.30")      # 30% max sector exposure
    max_token_concentration_pct: Decimal = Decimal("0.15")  # 15% max single token
    max_correlated_exposure_pct: Decimal = Decimal("0.25")  # 25% max correlated exposure
    correlation_threshold: Decimal = Decimal("0.70")        # 70% correlation threshold
    max_leverage_ratio: Decimal = Decimal("2.0")            # 2x max leverage
    min_liquidity_requirement: Decimal = Decimal("1000000") # $1M minimum liquidity
    
    # Risk monitoring
    risk_check_frequency_seconds: int = 5                   # Check every 5 seconds
    position_rebalance_threshold: Decimal = Decimal("0.20") # 20% rebalance threshold
    volatility_adjustment_factor: Decimal = Decimal("0.5")  # 50% volatility adjustment
    
    # Alert thresholds
    risk_warning_threshold: Decimal = Decimal("0.75")       # 75% of limit
    risk_critical_threshold: Decimal = Decimal("0.90")      # 90% of limit
    
    # Advanced risk features
    enable_dynamic_position_sizing: bool = True
    enable_correlation_monitoring: bool = True
    enable_sector_limits: bool = True
    enable_stress_testing: bool = True
    stress_test_scenarios: List[str] = field(default_factory=lambda: ["flash_crash", "market_dump", "high_volatility"])
    
    # Safety system integration
    enable_pre_trade_safety_checks: bool = True
    liquidation_trigger_threshold: Decimal = Decimal("0.12")  # 12% drawdown triggers liquidation
    partial_liquidation_percentage: Decimal = Decimal("0.50")  # Liquidate 50% initially
    full_liquidation_threshold: Decimal = Decimal("0.18")     # 18% drawdown triggers full liquidation
    safety_system_priority_order: List[str] = field(default_factory=lambda: ["emergency_stop", "risk_manager", "liquidity_check"])
    enable_safety_coordination: bool = True
    safety_override_threshold: Decimal = Decimal("0.95")     # 95% risk threshold for override

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not (0 < self.max_position_size_pct <= 1):
            raise ValueError("max_position_size_pct must be between 0 and 1")
        
        if not (0 < self.max_daily_loss_pct <= 1):
            raise ValueError("max_daily_loss_pct must be between 0 and 1")
        
        if not (0 < self.emergency_drawdown_pct <= 1):
            raise ValueError("emergency_drawdown_pct must be between 0 and 1")
        
        if not self.dex_preference_order:
            raise ValueError("At least one DEX must be configured")
        
        if self.trading_hours_start < 0 or self.trading_hours_start > 23:
            raise ValueError("trading_hours_start must be between 0 and 23")
        
        if self.trading_hours_end < 0 or self.trading_hours_end > 24:
            raise ValueError("trading_hours_end must be between 0 and 24")


@dataclass
class LiveModeMetrics:
    """Live trading mode performance metrics."""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_volume_usd: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    average_trade_size: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    current_drawdown: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    current_portfolio_value: Decimal = Decimal("0")
    active_positions: int = 0
    pending_orders: int = 0
    emergency_stops_triggered: int = 0
    dex_failures: int = 0
    avg_execution_latency_ms: float = 0.0
    last_trade_time: Optional[datetime] = None
    session_start_time: Optional[datetime] = None


@dataclass
class RiskValidationResult:
    """Result of risk validation check."""
    is_valid: bool
    reason: str = ""
    risk_level: SafetyCheckResult = SafetyCheckResult.SAFE
    recommended_action: str = ""


@dataclass
class EmergencyStopResult:
    """Result of emergency stop check."""
    should_stop: bool
    reason: EmergencyStopReason
    message: str = ""
    portfolio_value: Decimal = Decimal("0")
    triggered_at: Optional[datetime] = None


@dataclass
class PnLAlert:
    """P&L alert notification."""
    alert_id: str
    message: str
    severity: str
    timestamp: datetime
    pnl_amount: Decimal
    threshold_breached: str


class LiveModeError(Exception):
    """Base live mode error."""
    pass


class EmergencyStopError(LiveModeError):
    """Error during emergency stop execution."""
    pass