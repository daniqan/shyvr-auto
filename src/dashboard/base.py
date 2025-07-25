"""
Dashboard Base Models and Data Structures
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import structlog

logger = structlog.get_logger()


class DashboardError(Exception):
    """Base exception for dashboard operations"""
    pass


class SystemStatus(Enum):
    """System status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    OFFLINE = "offline"


class TradingMode(Enum):
    """Trading mode enumeration"""
    ANALYSIS = "analysis"
    SIMULATION = "simulation"
    LIVE = "live"
    STOPPED = "stopped"


@dataclass
class SystemMetrics:
    """System health and performance metrics"""
    status: SystemStatus
    uptime_seconds: float
    cpu_usage_pct: float
    memory_usage_mb: float
    memory_usage_pct: float
    active_connections: int
    requests_per_minute: float
    error_rate_pct: float
    response_time_ms: float
    last_updated: datetime
    
    # Component statuses
    database_status: SystemStatus
    redis_status: SystemStatus
    ml_models_status: SystemStatus
    rl_agent_status: SystemStatus
    dex_connections_status: SystemStatus
    
    # Performance counters
    total_requests: int
    total_errors: int
    cache_hit_rate_pct: float
    
    @classmethod
    def create_default(cls) -> "SystemMetrics":
        """Create default system metrics"""
        return cls(
            status=SystemStatus.HEALTHY,
            uptime_seconds=0.0,
            cpu_usage_pct=0.0,
            memory_usage_mb=0.0,
            memory_usage_pct=0.0,
            active_connections=0,
            requests_per_minute=0.0,
            error_rate_pct=0.0,
            response_time_ms=0.0,
            last_updated=datetime.utcnow(),
            database_status=SystemStatus.HEALTHY,
            redis_status=SystemStatus.HEALTHY,
            ml_models_status=SystemStatus.HEALTHY,
            rl_agent_status=SystemStatus.HEALTHY,
            dex_connections_status=SystemStatus.HEALTHY,
            total_requests=0,
            total_errors=0,
            cache_hit_rate_pct=0.0
        )


@dataclass
class Position:
    """Trading position information"""
    symbol: str
    chain: str
    side: str  # "long" or "short"
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    margin_used: Decimal
    leverage: float
    entry_time: datetime
    last_updated: datetime


@dataclass
class Trade:
    """Individual trade information"""
    id: str
    symbol: str
    chain: str
    dex: str
    side: str  # "buy" or "sell"
    size: Decimal
    price: Decimal
    value_usd: Decimal
    fee: Decimal
    slippage_pct: Decimal
    execution_time_ms: float
    timestamp: datetime
    status: str  # "completed", "pending", "failed"


@dataclass
class PortfolioStatus:
    """Portfolio status and performance"""
    total_value_usd: Decimal
    available_balance_usd: Decimal
    margin_used_usd: Decimal
    unrealized_pnl_usd: Decimal
    realized_pnl_usd: Decimal
    daily_pnl_usd: Decimal
    daily_pnl_pct: Decimal
    total_return_pct: Decimal
    
    # Risk metrics
    max_drawdown_pct: Decimal
    current_drawdown_pct: Decimal
    sharpe_ratio: Optional[float]
    volatility_pct: Optional[float]
    var_95_usd: Optional[Decimal]
    
    # Position information
    active_positions: List[Position]
    position_count: int
    recent_trades: List[Trade]
    
    # Chain balances
    chain_balances: Dict[str, Decimal]
    
    last_updated: datetime
    
    @classmethod
    def create_default(cls) -> "PortfolioStatus":
        """Create default portfolio status"""
        return cls(
            total_value_usd=Decimal("0"),
            available_balance_usd=Decimal("0"),
            margin_used_usd=Decimal("0"),
            unrealized_pnl_usd=Decimal("0"),
            realized_pnl_usd=Decimal("0"),
            daily_pnl_usd=Decimal("0"),
            daily_pnl_pct=Decimal("0"),
            total_return_pct=Decimal("0"),
            max_drawdown_pct=Decimal("0"),
            current_drawdown_pct=Decimal("0"),
            sharpe_ratio=None,
            volatility_pct=None,
            var_95_usd=None,
            active_positions=[],
            position_count=0,
            recent_trades=[],
            chain_balances={},
            last_updated=datetime.utcnow()
        )


@dataclass
class TradingStatus:
    """Current trading status and activity"""
    mode: TradingMode
    is_trading_active: bool
    last_trade_time: Optional[datetime]
    trades_today: int
    volume_today_usd: Decimal
    
    # Mode-specific status
    analysis_running: bool
    simulation_running: bool
    live_trading_enabled: bool
    
    # Safety and controls
    emergency_stop_active: bool
    risk_limits_active: bool
    max_position_size_usd: Decimal
    max_daily_loss_usd: Decimal
    
    # Performance metrics
    win_rate_pct: float
    avg_trade_duration_hours: float
    avg_profit_per_trade_usd: Decimal
    
    # Token analysis
    tokens_analyzed_today: int
    tokens_in_watchlist: int
    high_confidence_signals: int
    
    last_updated: datetime
    
    @classmethod
    def create_default(cls) -> "TradingStatus":
        """Create default trading status"""
        return cls(
            mode=TradingMode.ANALYSIS,
            is_trading_active=False,
            last_trade_time=None,
            trades_today=0,
            volume_today_usd=Decimal("0"),
            analysis_running=True,
            simulation_running=False,
            live_trading_enabled=False,
            emergency_stop_active=False,
            risk_limits_active=True,
            max_position_size_usd=Decimal("1000"),
            max_daily_loss_usd=Decimal("500"),
            win_rate_pct=0.0,
            avg_trade_duration_hours=0.0,
            avg_profit_per_trade_usd=Decimal("0"),
            tokens_analyzed_today=0,
            tokens_in_watchlist=0,
            high_confidence_signals=0,
            last_updated=datetime.utcnow()
        )


@dataclass
class MLModel:
    """ML model status"""
    name: str
    type: str
    status: SystemStatus
    accuracy: Optional[float]
    last_training_time: Optional[datetime]
    predictions_today: int
    avg_prediction_time_ms: float
    model_size_mb: float
    version: str


@dataclass
class RLAgent:
    """RL agent status"""
    name: str
    algorithm: str
    status: SystemStatus
    episode: int
    epsilon: float
    avg_reward: float
    win_rate_pct: float
    experience_buffer_size: int
    last_training_time: Optional[datetime]
    actions_today: int
    avg_decision_time_ms: float


@dataclass
class MLRLStatus:
    """ML and RL system status"""
    ml_models: List[MLModel]
    rl_agents: List[RLAgent]
    
    # Integration metrics
    ml_rl_integration_active: bool
    ml_rl_decision_latency_ms: float
    ml_confidence_threshold: float
    rl_action_confidence: float
    
    # Training status
    ml_training_active: bool
    rl_training_active: bool
    continuous_learning_active: bool
    
    # Performance metrics
    ml_prediction_accuracy_pct: float
    rl_action_success_rate_pct: float
    ensemble_agreement_pct: float
    
    last_updated: datetime
    
    @classmethod
    def create_default(cls) -> "MLRLStatus":
        """Create default ML/RL status"""
        return cls(
            ml_models=[],
            rl_agents=[],
            ml_rl_integration_active=True,
            ml_rl_decision_latency_ms=10.0,
            ml_confidence_threshold=0.7,
            rl_action_confidence=0.8,
            ml_training_active=False,
            rl_training_active=False,
            continuous_learning_active=True,
            ml_prediction_accuracy_pct=75.0,
            rl_action_success_rate_pct=60.0,
            ensemble_agreement_pct=80.0,
            last_updated=datetime.utcnow()
        )


@dataclass
class DashboardData:
    """Complete dashboard data structure"""
    system_metrics: SystemMetrics
    portfolio_status: PortfolioStatus
    trading_status: TradingStatus
    ml_rl_status: MLRLStatus
    
    # Alert and notification counts
    active_alerts: int
    warnings_count: int
    errors_count: int
    
    # Recent activity
    recent_logs: List[str]
    recent_notifications: List[str]
    
    timestamp: datetime
    
    @classmethod
    def create_default(cls) -> "DashboardData":
        """Create default dashboard data"""
        return cls(
            system_metrics=SystemMetrics.create_default(),
            portfolio_status=PortfolioStatus.create_default(),
            trading_status=TradingStatus.create_default(),
            ml_rl_status=MLRLStatus.create_default(),
            active_alerts=0,
            warnings_count=0,
            errors_count=0,
            recent_logs=[],
            recent_notifications=[],
            timestamp=datetime.utcnow()
        )


class DashboardDataCache:
    """Thread-safe cache for dashboard data"""
    
    def __init__(self):
        self._data: Optional[DashboardData] = None
        self._lock = asyncio.Lock()
        self._last_update: Optional[datetime] = None
    
    async def get_data(self) -> DashboardData:
        """Get current dashboard data"""
        async with self._lock:
            if self._data is None:
                self._data = DashboardData.create_default()
                self._last_update = datetime.utcnow()
            return self._data
    
    async def update_data(self, data: DashboardData) -> None:
        """Update dashboard data"""
        async with self._lock:
            self._data = data
            self._last_update = datetime.utcnow()
    
    async def update_partial(self, **kwargs) -> None:
        """Partially update dashboard data"""
        async with self._lock:
            if self._data is None:
                self._data = DashboardData.create_default()
            
            # Update specific fields
            for key, value in kwargs.items():
                if hasattr(self._data, key):
                    setattr(self._data, key, value)
            
            self._data.timestamp = datetime.utcnow()
            self._last_update = datetime.utcnow()
    
    async def get_last_update(self) -> Optional[datetime]:
        """Get timestamp of last update"""
        async with self._lock:
            return self._last_update


# Global dashboard data cache instance
dashboard_cache = DashboardDataCache()