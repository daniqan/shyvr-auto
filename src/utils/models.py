"""
SQLAlchemy models for RLTE Activity Logging
Provides ORM models for database entities with proper relationships
"""

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Integer, String, Text, 
    Numeric, ARRAY, JSON, ForeignKey, CheckConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


# =============================================================================
# ENUM CLASSES (matching database enums)
# =============================================================================

class ActivityCategory(enum.Enum):
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


class ActivityAction(enum.Enum):
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


class ActivitySeverity(enum.Enum):
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


class TradingMode(enum.Enum):
    """Trading modes matching database enum"""
    ANALYSIS = "analysis"
    SIMULATION = "simulation"
    LIVE = "live"


class ChainType(enum.Enum):
    """Chain types matching database enum"""
    ETHEREUM = "ethereum"
    SOLANA = "solana"
    BASE = "base"


# =============================================================================
# DATABASE MODELS
# =============================================================================

class User(Base):
    """User model for Telegram users"""
    __tablename__ = "users"
    
    telegram_user_id = Column(BigInteger, primary_key=True)
    username = Column(String(100), nullable=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    activity_logs = relationship("ActivityLog", back_populates="user")
    activity_sessions = relationship("UserActivitySession", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index('idx_users_username', 'username', postgresql_where=username.isnot(None)),
        Index('idx_users_active', 'is_active', 'last_seen_at'),
    )
    
    def __repr__(self):
        return f"<User(id={self.telegram_user_id}, username={self.username})>"


class ActivityLog(Base):
    """Main activity log model"""
    __tablename__ = "activity_logs"
    
    # Primary identifiers
    id = Column(BigInteger, primary_key=True)
    activity_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Activity classification
    category = Column(String(20), nullable=False)  # Using String instead of Enum for flexibility
    action = Column(String(20), nullable=False)
    severity = Column(String(20), default='info', nullable=False)
    
    # Basic activity information
    source = Column(String(100), nullable=False)
    event_type = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Context and relationships
    user_id = Column(BigInteger, ForeignKey('users.telegram_user_id'), nullable=True)
    session_id = Column(UUID(as_uuid=True), nullable=True)
    request_id = Column(UUID(as_uuid=True), nullable=True)
    parent_activity_id = Column(UUID(as_uuid=True), ForeignKey('activity_logs.activity_id'), nullable=True)
    
    # Trading context
    trading_mode = Column(String(20), nullable=True)
    token_address = Column(String(64), nullable=True)
    chain = Column(String(20), nullable=True)
    
    # Financial context
    amount_usd = Column(Numeric(20, 8), nullable=True)
    fee_usd = Column(Numeric(20, 8), nullable=True)
    
    # Technical context
    execution_time_ms = Column(Integer, nullable=True)
    memory_usage_mb = Column(Integer, nullable=True)
    cpu_usage_pct = Column(Numeric(5, 2), nullable=True)
    
    # Metadata and additional context  
    metadata_json = Column('metadata', JSONB, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    correlation_id = Column(String(100), nullable=True)
    
    # Error context
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # API context
    api_endpoint = Column(String(200), nullable=True)
    http_method = Column(String(10), nullable=True)
    http_status = Column(Integer, nullable=True)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(INET, nullable=True)
    
    # Performance metrics
    response_time_ms = Column(Integer, nullable=True)
    throughput_ops_per_sec = Column(Numeric(10, 2), nullable=True)
    
    # Security context
    security_level = Column(String(20), default='normal', nullable=True)
    risk_score = Column(Integer, nullable=True)
    
    # Dashboard specific
    dashboard_component = Column(String(100), nullable=True)
    dashboard_action = Column(String(100), nullable=True)
    
    # Data integrity
    checksum = Column(String(64), nullable=True)
    version = Column(Integer, default=1, nullable=False)
    
    # Indexing hints
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="activity_logs")
    parent_activity = relationship("ActivityLog", remote_side=[activity_id])
    child_activities = relationship("ActivityLog", back_populates="parent_activity")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('risk_score >= 0 AND risk_score <= 100', name='activity_logs_risk_score_check'),
        
        # Primary time-based indexes
        Index('idx_activity_logs_created_at_desc', 'created_at'),
        Index('idx_activity_logs_created_at_category', 'created_at', 'category'),
        Index('idx_activity_logs_created_at_severity', 'created_at', 'severity'),
        
        # Category and action filtering
        Index('idx_activity_logs_category_action', 'category', 'action'),
        Index('idx_activity_logs_category_created', 'category', 'created_at'),
        Index('idx_activity_logs_action_created', 'action', 'created_at'),
        
        # User and session tracking
        Index('idx_activity_logs_user_id_created', 'user_id', 'created_at', 
              postgresql_where=user_id.isnot(None)),
        Index('idx_activity_logs_session_id', 'session_id', 
              postgresql_where=session_id.isnot(None)),
        Index('idx_activity_logs_request_id', 'request_id', 
              postgresql_where=request_id.isnot(None)),
        
        # Source and component tracking
        Index('idx_activity_logs_source_created', 'source', 'created_at'),
        Index('idx_activity_logs_event_type', 'event_type'),
        Index('idx_activity_logs_dashboard_component', 'dashboard_component',
              postgresql_where=dashboard_component.isnot(None)),
        
        # Trading context
        Index('idx_activity_logs_trading_mode', 'trading_mode',
              postgresql_where=trading_mode.isnot(None)),
        Index('idx_activity_logs_token_chain', 'token_address', 'chain',
              postgresql_where=token_address.isnot(None)),
        
        # Error and performance tracking
        Index('idx_activity_logs_error_code', 'error_code',
              postgresql_where=error_code.isnot(None)),
        Index('idx_activity_logs_execution_time', 'execution_time_ms',
              postgresql_where=execution_time_ms.isnot(None)),
        Index('idx_activity_logs_response_time', 'response_time_ms',
              postgresql_where=response_time_ms.isnot(None)),
        
        # API tracking
        Index('idx_activity_logs_api_endpoint', 'api_endpoint', 'http_status',
              postgresql_where=api_endpoint.isnot(None)),
        
        # Security and risk
        Index('idx_activity_logs_risk_score', 'risk_score',
              postgresql_where=risk_score.isnot(None)),
        Index('idx_activity_logs_ip_address', 'ip_address',
              postgresql_where=ip_address.isnot(None)),
        
        # Hierarchy and correlation
        Index('idx_activity_logs_parent_activity', 'parent_activity_id',
              postgresql_where=parent_activity_id.isnot(None)),
        Index('idx_activity_logs_correlation_id', 'correlation_id',
              postgresql_where=correlation_id.isnot(None)),
        
        # GIN indexes for JSONB and array data
        Index('idx_activity_logs_metadata_gin', 'metadata', postgresql_using='gin'),
        Index('idx_activity_logs_tags_gin', 'tags', postgresql_using='gin'),
    )
    
    def __repr__(self):
        return f"<ActivityLog(id={self.id}, category={self.category}, action={self.action})>"


class ActivitySummary(Base):
    """Activity summary model for dashboard performance"""
    __tablename__ = "activity_summaries"
    
    id = Column(BigInteger, primary_key=True)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    period_type = Column(String(20), nullable=False)  # 'minute', 'hour', 'day', 'week'
    
    # Aggregated metrics
    category = Column(String(20), nullable=False)
    total_count = Column(Integer, default=0, nullable=False)
    error_count = Column(Integer, default=0, nullable=False)
    warning_count = Column(Integer, default=0, nullable=False)
    
    # Performance metrics
    avg_execution_time_ms = Column(Numeric(10, 2), nullable=True)
    max_execution_time_ms = Column(Integer, nullable=True)
    avg_response_time_ms = Column(Numeric(10, 2), nullable=True)
    max_response_time_ms = Column(Integer, nullable=True)
    
    # System metrics
    avg_memory_usage_mb = Column(Numeric(10, 2), nullable=True)
    max_memory_usage_mb = Column(Integer, nullable=True)
    avg_cpu_usage_pct = Column(Numeric(5, 2), nullable=True)
    max_cpu_usage_pct = Column(Numeric(5, 2), nullable=True)
    
    # Top sources and events
    top_sources = Column(JSONB, nullable=True)
    top_events = Column(JSONB, nullable=True)
    top_errors = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_activity_summaries_period', 'period_start', 'period_end', 'period_type'),
        Index('idx_activity_summaries_category_period', 'category', 'period_start'),
        # Unique constraint
        {'extend_existing': True}
    )
    
    def __repr__(self):
        return f"<ActivitySummary(id={self.id}, category={self.category}, period={self.period_type})>"


class UserActivitySession(Base):
    """User activity session model"""
    __tablename__ = "user_activity_sessions"
    
    id = Column(BigInteger, primary_key=True)
    session_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    user_id = Column(BigInteger, ForeignKey('users.telegram_user_id'), nullable=False)
    
    # Session lifecycle
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Session context
    source = Column(String(50), nullable=False)  # 'dashboard', 'api', 'telegram'
    user_agent = Column(Text, nullable=True)
    ip_address = Column(INET, nullable=True)
    
    # Session metrics
    total_activities = Column(Integer, default=0, nullable=False)
    unique_components_accessed = Column(Integer, default=0, nullable=False)
    errors_encountered = Column(Integer, default=0, nullable=False)
    
    # Session metadata
    metadata_json = Column('metadata', JSONB, nullable=True)
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="activity_sessions")
    
    __table_args__ = (
        Index('idx_user_sessions_user_started', 'user_id', 'started_at'),
        Index('idx_user_sessions_session_id', 'session_id'),
        Index('idx_user_sessions_active', 'last_activity_at', 
              postgresql_where=ended_at.is_(None)),
    )
    
    def __repr__(self):
        return f"<UserActivitySession(id={self.id}, user_id={self.user_id}, source={self.source})>"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_all_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """Drop all tables from the database"""
    Base.metadata.drop_all(engine)


def get_table_names():
    """Get list of all table names"""
    return list(Base.metadata.tables.keys())


# =============================================================================
# QUERY HELPERS
# =============================================================================

class ActivityLogQueries:
    """Helper class for common ActivityLog queries"""
    
    @staticmethod
    def recent_activities(session, hours: int = 24, limit: int = 100):
        """Get recent activities"""
        return session.query(ActivityLog).filter(
            ActivityLog.created_at > func.now() - func.interval(f'{hours} hours')
        ).order_by(ActivityLog.created_at.desc()).limit(limit)
    
    @staticmethod
    def activities_by_category(session, category: str, limit: int = 100):
        """Get activities by category"""
        return session.query(ActivityLog).filter(
            ActivityLog.category == category
        ).order_by(ActivityLog.created_at.desc()).limit(limit)
    
    @staticmethod
    def activities_by_user(session, user_id: int, limit: int = 100):
        """Get activities by user"""
        return session.query(ActivityLog).filter(
            ActivityLog.user_id == user_id
        ).order_by(ActivityLog.created_at.desc()).limit(limit)
    
    @staticmethod
    def error_activities(session, hours: int = 24):
        """Get error activities"""
        return session.query(ActivityLog).filter(
            ActivityLog.severity.in_(['error', 'critical', 'alert', 'emergency']),
            ActivityLog.created_at > func.now() - func.interval(f'{hours} hours')
        ).order_by(ActivityLog.created_at.desc())
    
    @staticmethod
    def performance_activities(session, min_execution_time: int = 1000):
        """Get slow performance activities"""
        return session.query(ActivityLog).filter(
            ActivityLog.execution_time_ms >= min_execution_time
        ).order_by(ActivityLog.execution_time_ms.desc())


class UserQueries:
    """Helper class for common User queries"""
    
    @staticmethod
    def active_users(session):
        """Get active users"""
        return session.query(User).filter(User.is_active == True)
    
    @staticmethod
    def admin_users(session):
        """Get admin users"""
        return session.query(User).filter(User.is_admin == True)
    
    @staticmethod
    def recent_users(session, days: int = 30):
        """Get recently active users"""
        return session.query(User).filter(
            User.last_seen_at > func.now() - func.interval(f'{days} days')
        ).order_by(User.last_seen_at.desc())