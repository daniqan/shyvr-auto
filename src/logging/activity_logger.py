"""
Activity Logger Implementation
Production-ready activity logging for Shyvr RLTE dashboard
"""

import asyncio
import json
import uuid
import hashlib
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import asyncpg
import structlog

from ..utils.config import get_config
from ..utils.database import get_database_pool

logger = structlog.get_logger()


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class ActivityCategory(Enum):
    """Activity categories matching database enum"""
    SYSTEM = "system"
    TRADING = "trading"
    USER = "user"
    ML_RL = "ml_rl"
    SECURITY = "security"
    API = "api"
    PERFORMANCE = "performance"
    DATA = "data"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"
    DASHBOARD = "dashboard"


class ActivityAction(Enum):
    """Activity actions matching database enum"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    ERROR = "error"
    ALERT = "alert"
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    VIOLATION = "violation"
    TIMEOUT = "timeout"
    RETRY = "retry"
    SUCCESS = "success"
    FAILURE = "failure"


class ActivitySeverity(Enum):
    """Activity severity levels matching database enum"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"
    EMERGENCY = "emergency"


class TradingMode(Enum):
    """Trading modes matching database enum"""
    ANALYSIS = "analysis"
    SIMULATION = "simulation"
    LIVE = "live"


class ChainType(Enum):
    """Chain types matching database enum"""
    ETHEREUM = "ethereum"
    SOLANA = "solana"
    BASE = "base"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ActivityLogEntry:
    """Structured activity log entry"""
    
    # Required fields
    category: ActivityCategory
    action: ActivityAction
    source: str
    event_type: str
    title: str
    
    # Optional identification
    activity_id: Optional[str] = None
    severity: ActivitySeverity = ActivitySeverity.INFO
    description: Optional[str] = None
    
    # Context
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    parent_activity_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Trading context
    trading_mode: Optional[TradingMode] = None
    token_address: Optional[str] = None
    chain: Optional[ChainType] = None
    
    # Financial context
    amount_usd: Optional[Decimal] = None
    fee_usd: Optional[Decimal] = None
    
    # Technical context
    execution_time_ms: Optional[int] = None
    memory_usage_mb: Optional[int] = None
    cpu_usage_pct: Optional[Decimal] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    # Error context
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # API context
    api_endpoint: Optional[str] = None
    http_method: Optional[str] = None
    http_status: Optional[int] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Performance metrics
    response_time_ms: Optional[int] = None
    throughput_ops_per_sec: Optional[Decimal] = None
    
    # Security context
    security_level: str = "normal"
    risk_score: Optional[int] = None
    
    # Dashboard context
    dashboard_component: Optional[str] = None
    dashboard_action: Optional[str] = None
    
    # System fields (auto-generated)
    created_at: Optional[datetime] = None
    checksum: Optional[str] = None
    version: int = 1

    def __post_init__(self):
        """Auto-generate required fields"""
        if self.activity_id is None:
            self.activity_id = str(uuid.uuid4())
        
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        
        # Auto-generate request_id if not provided and we have a session
        if self.request_id is None and self.session_id:
            self.request_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        data = {}
        
        for field, value in asdict(self).items():
            if value is None:
                continue
                
            # Handle enums
            if isinstance(value, Enum):
                data[field] = value.value
            # Handle datetime
            elif isinstance(value, datetime):
                data[field] = value
            # Handle Decimal
            elif isinstance(value, Decimal):
                data[field] = value
            # Handle lists and dicts (convert to JSON for arrays/JSONB)
            elif isinstance(value, (list, dict)):
                data[field] = value
            else:
                data[field] = value
        
        return data
    
    def generate_checksum(self) -> str:
        """Generate data integrity checksum"""
        # Create a deterministic string representation
        checksum_data = {
            'activity_id': self.activity_id,
            'category': self.category.value,
            'action': self.action.value,
            'source': self.source,
            'event_type': self.event_type,
            'title': self.title,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
        
        checksum_string = json.dumps(checksum_data, sort_keys=True, default=str)
        return hashlib.sha256(checksum_string.encode()).hexdigest()


# =============================================================================
# ACTIVITY LOGGER CLASS
# =============================================================================

class ActivityLogger:
    """Production-ready activity logger for dashboard integration"""
    
    def __init__(self):
        self.config = get_config()
        self._pool: Optional[asyncpg.Pool] = None
        self._batch_size = 100
        self._batch_timeout = 5.0  # seconds
        self._buffer: List[ActivityLogEntry] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the activity logger"""
        if self._running:
            return
        
        self._pool = await get_database_pool()
        self._running = True
        
        # Start background flush task
        self._flush_task = asyncio.create_task(self._flush_loop())
        
        logger.info("Activity logger started")
    
    async def stop(self) -> None:
        """Stop the activity logger"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining buffer
        await self._flush_buffer()
        
        logger.info("Activity logger stopped")
    
    async def log_activity(
        self,
        category: ActivityCategory,
        action: ActivityAction,
        source: str,
        event_type: str,
        title: str,
        severity: ActivitySeverity = ActivitySeverity.INFO,
        **kwargs
    ) -> str:
        """
        Log an activity entry
        
        Args:
            category: Activity category
            action: Activity action
            source: Source component
            event_type: Specific event type
            title: Human-readable title
            severity: Log severity level
            **kwargs: Additional fields for ActivityLogEntry
        
        Returns:
            Activity ID for reference
        """
        entry = ActivityLogEntry(
            category=category,
            action=action,
            source=source,
            event_type=event_type,
            title=title,
            severity=severity,
            **kwargs
        )
        
        # Generate checksum
        entry.checksum = entry.generate_checksum()
        
        # Add to buffer for batch processing
        async with self._buffer_lock:
            self._buffer.append(entry)
            
            # Immediate flush for high-severity events
            if severity in (ActivitySeverity.CRITICAL, ActivitySeverity.ALERT, ActivitySeverity.EMERGENCY):
                await self._flush_buffer()
        
        # Also log to structured logger for immediate visibility
        structlog_data = {
            'activity_id': entry.activity_id,
            'category': category.value,
            'action': action.value,
            'source': source,
            'event_type': event_type,
            'user_id': entry.user_id,
            'session_id': entry.session_id,
        }
        
        if severity == ActivitySeverity.ERROR:
            logger.error(title, **structlog_data, error_code=entry.error_code)
        elif severity == ActivitySeverity.WARNING:
            logger.warning(title, **structlog_data)
        elif severity in (ActivitySeverity.CRITICAL, ActivitySeverity.ALERT, ActivitySeverity.EMERGENCY):
            logger.critical(title, **structlog_data)
        else:
            logger.info(title, **structlog_data)
        
        return entry.activity_id
    
    async def log_error(
        self,
        category: ActivityCategory,
        source: str,
        event_type: str,
        title: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        exception: Optional[Exception] = None,
        **kwargs
    ) -> str:
        """
        Log an error activity with automatic stack trace capture
        
        Args:
            category: Activity category
            source: Source component
            event_type: Specific event type
            title: Human-readable title
            error_code: Standardized error code
            error_message: Error message
            exception: Exception object (auto-captures stack trace)
            **kwargs: Additional fields
        
        Returns:
            Activity ID for reference
        """
        stack_trace = None
        if exception:
            stack_trace = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
            if not error_message:
                error_message = str(exception)
        elif not exception and not error_message:
            # Capture current stack trace if no exception provided
            stack_trace = ''.join(traceback.format_stack())
        
        return await self.log_activity(
            category=category,
            action=ActivityAction.ERROR,
            source=source,
            event_type=event_type,
            title=title,
            severity=ActivitySeverity.ERROR,
            error_code=error_code,
            error_message=error_message,
            stack_trace=stack_trace,
            **kwargs
        )
    
    async def log_performance(
        self,
        source: str,
        operation: str,
        execution_time_ms: int,
        success: bool = True,
        **kwargs
    ) -> str:
        """
        Log a performance metric
        
        Args:
            source: Source component
            operation: Operation name
            execution_time_ms: Execution time in milliseconds
            success: Whether operation was successful
            **kwargs: Additional fields
        
        Returns:
            Activity ID for reference
        """
        action = ActivityAction.SUCCESS if success else ActivityAction.FAILURE
        severity = ActivitySeverity.INFO if success else ActivitySeverity.WARNING
        
        return await self.log_activity(
            category=ActivityCategory.PERFORMANCE,
            action=action,
            source=source,
            event_type=f"performance_{operation}",
            title=f"{operation} performance metric",
            severity=severity,
            execution_time_ms=execution_time_ms,
            **kwargs
        )
    
    async def log_user_action(
        self,
        user_id: int,
        action: ActivityAction,
        component: str,
        title: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Log a user action
        
        Args:
            user_id: User ID
            action: User action
            component: Component or page
            title: Action description
            session_id: User session ID
            **kwargs: Additional fields
        
        Returns:
            Activity ID for reference
        """
        return await self.log_activity(
            category=ActivityCategory.USER,
            action=action,
            source="dashboard",
            event_type=f"user_{action.value}",
            title=title,
            user_id=user_id,
            session_id=session_id,
            dashboard_component=component,
            **kwargs
        )
    
    async def log_trading_activity(
        self,
        action: ActivityAction,
        title: str,
        trading_mode: TradingMode,
        token_address: Optional[str] = None,
        chain: Optional[ChainType] = None,
        amount_usd: Optional[Decimal] = None,
        **kwargs
    ) -> str:
        """
        Log a trading activity
        
        Args:
            action: Trading action
            title: Activity description
            trading_mode: Current trading mode
            token_address: Related token address
            chain: Blockchain
            amount_usd: Transaction amount
            **kwargs: Additional fields
        
        Returns:
            Activity ID for reference
        """
        return await self.log_activity(
            category=ActivityCategory.TRADING,
            action=action,
            source="trading_engine",
            event_type=f"trading_{action.value}",
            title=title,
            trading_mode=trading_mode,
            token_address=token_address,
            chain=chain,
            amount_usd=amount_usd,
            **kwargs
        )
    
    async def log_api_call(
        self,
        api_name: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
        success: bool = True,
        **kwargs
    ) -> str:
        """
        Log an API call
        
        Args:
            api_name: API service name
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            response_time_ms: Response time
            success: Whether call was successful
            **kwargs: Additional fields
        
        Returns:
            Activity ID for reference
        """
        severity = ActivitySeverity.INFO if success else ActivitySeverity.WARNING
        action = ActivityAction.SUCCESS if success else ActivityAction.FAILURE
        
        return await self.log_activity(
            category=ActivityCategory.API,
            action=action,
            source=api_name,
            event_type="api_call",
            title=f"{method} {endpoint}",
            severity=severity,
            api_endpoint=endpoint,
            http_method=method,
            http_status=status_code,
            response_time_ms=response_time_ms,
            **kwargs
        )
    
    async def _flush_loop(self) -> None:
        """Background task to flush buffer periodically"""
        while self._running:
            try:
                await asyncio.sleep(self._batch_timeout)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in flush loop", error=str(e))
    
    async def _flush_buffer(self) -> None:
        """Flush buffered entries to database"""
        if not self._buffer:
            return
        
        async with self._buffer_lock:
            entries_to_flush = self._buffer.copy()
            self._buffer.clear()
        
        if not entries_to_flush:
            return
        
        try:
            await self._insert_activities(entries_to_flush)
            logger.debug(f"Flushed {len(entries_to_flush)} activity entries")
        except Exception as e:
            logger.error(f"Failed to flush activity entries", error=str(e), count=len(entries_to_flush))
            # Re-add to buffer for retry (with some limit to prevent infinite growth)
            async with self._buffer_lock:
                if len(self._buffer) < self._batch_size * 10:  # Max 10 batches in buffer
                    self._buffer.extend(entries_to_flush)
    
    async def _insert_activities(self, entries: List[ActivityLogEntry]) -> None:
        """Insert activity entries to database"""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        # Prepare data for insertion
        insert_data = []
        for entry in entries:
            data = entry.to_dict()
            insert_data.append(data)
        
        # Build SQL query with all possible fields
        fields = [
            'activity_id', 'created_at', 'category', 'action', 'severity',
            'source', 'event_type', 'title', 'description',
            'user_id', 'session_id', 'request_id', 'parent_activity_id',
            'trading_mode', 'token_address', 'chain',
            'amount_usd', 'fee_usd',
            'execution_time_ms', 'memory_usage_mb', 'cpu_usage_pct',
            'metadata', 'tags', 'correlation_id',
            'error_code', 'error_message', 'stack_trace',
            'api_endpoint', 'http_method', 'http_status', 'user_agent', 'ip_address',
            'response_time_ms', 'throughput_ops_per_sec',
            'security_level', 'risk_score',
            'dashboard_component', 'dashboard_action',
            'checksum', 'version'
        ]
        
        placeholders = ', '.join([f'${i+1}' for i in range(len(fields))])
        field_names = ', '.join(fields)
        
        query = f"""
            INSERT INTO activity_logs ({field_names})
            VALUES ({placeholders})
        """
        
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for data in insert_data:
                    # Prepare values in correct order
                    values = []
                    for field in fields:
                        value = data.get(field)
                        # Convert lists to PostgreSQL arrays
                        if field == 'tags' and value:
                            value = value
                        values.append(value)
                    
                    await conn.execute(query, *values)


# =============================================================================
# GLOBAL INSTANCE AND CONVENIENCE FUNCTIONS
# =============================================================================

# Global activity logger instance
activity_logger = ActivityLogger()


# Convenience functions for common logging patterns
async def log_system_event(event_type: str, title: str, **kwargs) -> str:
    """Log a system event"""
    return await activity_logger.log_activity(
        category=ActivityCategory.SYSTEM,
        action=ActivityAction.EXECUTE,
        source="system",
        event_type=event_type,
        title=title,
        **kwargs
    )


async def log_dashboard_action(user_id: int, component: str, action: str, **kwargs) -> str:
    """Log a dashboard user action"""
    return await activity_logger.log_user_action(
        user_id=user_id,
        action=ActivityAction.ACCESS,
        component=component,
        title=f"User accessed {component}: {action}",
        dashboard_component=component,
        dashboard_action=action,
        **kwargs
    )


async def log_trade_execution(
    trading_mode: TradingMode,
    token_address: str,
    chain: ChainType,
    amount_usd: Decimal,
    success: bool = True,
    **kwargs
) -> str:
    """Log a trade execution"""
    action = ActivityAction.SUCCESS if success else ActivityAction.FAILURE
    severity = ActivitySeverity.INFO if success else ActivitySeverity.ERROR
    title = f"Trade {'executed' if success else 'failed'}: {token_address}"
    
    return await activity_logger.log_trading_activity(
        action=action,
        title=title,
        trading_mode=trading_mode,
        token_address=token_address,
        chain=chain,
        amount_usd=amount_usd,
        severity=severity,
        **kwargs
    )


async def log_ml_prediction(
    model_name: str,
    prediction_type: str,
    execution_time_ms: int,
    confidence: Optional[float] = None,
    **kwargs
) -> str:
    """Log an ML model prediction"""
    return await activity_logger.log_activity(
        category=ActivityCategory.ML_RL,
        action=ActivityAction.EXECUTE,
        source=f"ml_model_{model_name}",
        event_type=f"prediction_{prediction_type}",
        title=f"ML prediction: {prediction_type}",
        execution_time_ms=execution_time_ms,
        metadata={'confidence': confidence} if confidence else None,
        **kwargs
    )


# Context managers for automatic performance logging
class performance_tracker:
    """Context manager for automatic performance tracking"""
    
    def __init__(
        self,
        source: str,
        operation: str,
        category: ActivityCategory = ActivityCategory.PERFORMANCE,
        **kwargs
    ):
        self.source = source
        self.operation = operation
        self.category = category
        self.kwargs = kwargs
        self.start_time = None
        self.activity_id = None
    
    async def __aenter__(self):
        self.start_time = datetime.now(timezone.utc)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            execution_time = datetime.now(timezone.utc) - self.start_time
            execution_time_ms = int(execution_time.total_seconds() * 1000)
            
            success = exc_type is None
            self.activity_id = await activity_logger.log_performance(
                source=self.source,
                operation=self.operation,
                execution_time_ms=execution_time_ms,
                success=success,
                **self.kwargs
            )
            
            if not success and exc_val:
                await activity_logger.log_error(
                    category=self.category,
                    source=self.source,
                    event_type=f"error_{self.operation}",
                    title=f"Error in {self.operation}",
                    exception=exc_val,
                    parent_activity_id=self.activity_id
                )