"""
Enhanced Logging System for Phase 6.2

This module provides production-ready structured logging, audit trails,
and compliance features for the Shyvr RLTE system.
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import uuid

import structlog
from structlog import configure
from structlog.stdlib import LoggerFactory, BoundLogger
from structlog.processors import JSONRenderer, TimeStamper

logger = structlog.get_logger()


class LogLevel(Enum):
    """Enhanced log levels for financial trading systems."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    AUDIT = "audit"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class LogCategory(Enum):
    """Log categories for better organization and filtering."""
    SYSTEM = "system"
    TRADING = "trading"
    FINANCIAL = "financial"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    COMPLIANCE = "compliance"
    ML_MODEL = "ml_model"
    RISK_MANAGEMENT = "risk_management"
    USER_ACTION = "user_action"
    API = "api"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class LogContext:
    """Enhanced log context for structured logging."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    component: Optional[str] = None
    version: str = "1.0.0"
    environment: str = "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "component": self.component,
            "version": self.version,
            "environment": self.environment
        }


class EnhancedLoggingFormatter:
    """Enhanced formatter for production logging."""
    
    def __init__(self, include_source: bool = True, include_context: bool = True):
        """Initialize enhanced formatter."""
        self.include_source = include_source
        self.include_context = include_context
    
    def __call__(self, logger, method_name, event_dict):
        """Format log entry with enhanced information."""
        # Add timestamp
        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add log level
        event_dict["level"] = method_name.upper()
        
        # Add source information if enabled
        if self.include_source:
            frame = sys._getframe(1)
            event_dict["source"] = {
                "file": os.path.basename(frame.f_code.co_filename),
                "function": frame.f_code.co_name,
                "line": frame.f_lineno
            }
        
        # Add process and thread info
        event_dict["process_id"] = os.getpid()
        
        # Format the event
        return event_dict


class ComplianceProcessor:
    """Processor for compliance-specific log formatting."""
    
    def __init__(self, enable_pii_redaction: bool = True):
        """Initialize compliance processor."""
        self.enable_pii_redaction = enable_pii_redaction
        self.pii_fields = {
            'password', 'secret', 'token', 'key', 'private_key',
            'ssn', 'social_security', 'credit_card', 'card_number',
            'bank_account', 'routing_number', 'api_key', 'session_token'
        }
    
    def __call__(self, logger, method_name, event_dict):
        """Process log entry for compliance requirements."""
        # Add compliance metadata
        compliance_metadata = {
            "audit_trail_id": str(uuid.uuid4()),
            "retention_policy": "financial_records",  # 7 years for financial data
            "classification": self._classify_log_entry(event_dict),
            "redacted": False
        }
        
        # Redact PII if enabled
        if self.enable_pii_redaction:
            original_dict = event_dict.copy()
            event_dict = self._redact_pii(event_dict)
            if event_dict != original_dict:
                compliance_metadata["redacted"] = True
        
        event_dict["compliance"] = compliance_metadata
        return event_dict
    
    def _classify_log_entry(self, event_dict: Dict[str, Any]) -> str:
        """Classify log entry for compliance purposes."""
        event_text = str(event_dict.get("event", "")).lower()
        
        if any(word in event_text for word in ["trade", "order", "position", "transaction"]):
            return "financial_transaction"
        elif any(word in event_text for word in ["login", "auth", "security", "access"]):
            return "security_event"
        elif any(word in event_text for word in ["model", "prediction", "ml", "ai"]):
            return "algorithmic_decision"
        else:
            return "general_operations"
    
    def _redact_pii(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Redact personally identifiable information."""
        def redact_recursive(obj):
            if isinstance(obj, dict):
                return {
                    key: "[REDACTED]" if key.lower() in self.pii_fields 
                    else redact_recursive(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [redact_recursive(item) for item in obj]
            elif isinstance(obj, str) and any(field in obj.lower() for field in self.pii_fields):
                return "[REDACTED]"
            return obj
        
        return redact_recursive(event_dict)


class SecurityEventProcessor:
    """Processor for security event logging."""
    
    def __init__(self, alert_threshold: int = 10):
        """Initialize security event processor."""
        self.alert_threshold = alert_threshold
        self.security_events = []
    
    def __call__(self, logger, method_name, event_dict):
        """Process security events."""
        if event_dict.get("category") == LogCategory.SECURITY.value:
            # Add security metadata
            event_dict["security"] = {
                "event_id": str(uuid.uuid4()),
                "severity": self._determine_severity(event_dict),
                "requires_investigation": self._requires_investigation(event_dict),
                "ip_address": event_dict.get("ip_address"),
                "user_agent": event_dict.get("user_agent")
            }
            
            # Track security events for alerting
            self.security_events.append({
                "timestamp": datetime.now(timezone.utc),
                "event": event_dict
            })
            
            # Clean old events (keep last hour)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            self.security_events = [
                e for e in self.security_events 
                if e["timestamp"] > cutoff
            ]
            
            # Check if alert threshold is exceeded
            if len(self.security_events) >= self.alert_threshold:
                event_dict["security"]["alert"] = "HIGH_FREQUENCY_SECURITY_EVENTS"
        
        return event_dict
    
    def _determine_severity(self, event_dict: Dict[str, Any]) -> str:
        """Determine security event severity."""
        event_text = str(event_dict.get("event", "")).lower()
        
        if any(word in event_text for word in ["breach", "attack", "exploit", "unauthorized"]):
            return "critical"
        elif any(word in event_text for word in ["failed", "blocked", "suspicious"]):
            return "high"
        elif any(word in event_text for word in ["warning", "unusual"]):
            return "medium"
        else:
            return "low"
    
    def _requires_investigation(self, event_dict: Dict[str, Any]) -> bool:
        """Determine if security event requires investigation."""
        severity = self._determine_severity(event_dict)
        return severity in ["critical", "high"]


class LogAggregator:
    """Log aggregation and buffering system."""
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 flush_interval: int = 30,
                 output_handlers: Optional[List[Callable]] = None):
        """Initialize log aggregator."""
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.output_handlers = output_handlers or []
        self.buffer: List[Dict[str, Any]] = []
        self.last_flush = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()
    
    async def add_log_entry(self, entry: Dict[str, Any]):
        """Add log entry to buffer."""
        async with self._lock:
            self.buffer.append(entry)
            
            # Check if we need to flush
            should_flush = (
                len(self.buffer) >= self.buffer_size or
                datetime.now(timezone.utc) - self.last_flush > timedelta(seconds=self.flush_interval)
            )
            
            if should_flush:
                await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush buffer to output handlers."""
        if not self.buffer:
            return
        
        entries_to_flush = self.buffer.copy()
        self.buffer.clear()
        self.last_flush = datetime.now(timezone.utc)
        
        # Process entries with all handlers
        for handler in self.output_handlers:
            try:
                await handler(entries_to_flush)
            except Exception as e:
                # Use standard logging to avoid recursion
                logging.error(f"Error in log output handler: {e}")
    
    async def force_flush(self):
        """Force flush of buffer."""
        async with self._lock:
            await self._flush_buffer()


class FileRotationHandler:
    """Handler for file-based log rotation."""
    
    def __init__(self,
                 base_path: Path,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 max_files: int = 10,
                 compress_old: bool = True):
        """Initialize file rotation handler."""
        self.base_path = Path(base_path)
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.compress_old = compress_old
        self.current_file = None
        self.current_size = 0
        
        # Ensure log directory exists
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def __call__(self, entries: List[Dict[str, Any]]):
        """Handle log entries with rotation."""
        for entry in entries:
            await self._write_entry(entry)
    
    async def _write_entry(self, entry: Dict[str, Any]):
        """Write single log entry with rotation check."""
        # Check if we need to rotate
        if self._should_rotate():
            await self._rotate_files()
        
        # Ensure we have a current file
        if self.current_file is None:
            self.current_file = open(self._get_current_filename(), 'a', encoding='utf-8')
        
        # Write entry
        entry_json = json.dumps(entry, default=str)
        self.current_file.write(entry_json + '\n')
        self.current_file.flush()
        self.current_size += len(entry_json) + 1
    
    def _should_rotate(self) -> bool:
        """Check if log rotation is needed."""
        if self.current_file is None:
            return False
        return self.current_size >= self.max_file_size
    
    async def _rotate_files(self):
        """Rotate log files."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None
            self.current_size = 0
        
        # Rotate existing files
        for i in range(self.max_files - 1, 0, -1):
            old_file = self.base_path.with_suffix(f'.{i}.log')
            new_file = self.base_path.with_suffix(f'.{i + 1}.log')
            
            if old_file.exists():
                if i == self.max_files - 1:
                    # Delete oldest file
                    old_file.unlink()
                else:
                    old_file.rename(new_file)
        
        # Move current to .1
        current_file = self._get_current_filename()
        if Path(current_file).exists():
            Path(current_file).rename(self.base_path.with_suffix('.1.log'))
    
    def _get_current_filename(self) -> str:
        """Get current log filename."""
        return str(self.base_path.with_suffix('.log'))


class EnhancedLoggingSystem:
    """Main enhanced logging system for Phase 6.2."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced logging system."""
        self.config = config or {}
        self.context = LogContext()
        self.aggregator = None
        self._is_initialized = False
    
    def initialize(self,
                   log_level: LogLevel = LogLevel.INFO,
                   enable_file_logging: bool = True,
                   enable_console_logging: bool = True,
                   enable_aggregation: bool = True,
                   log_directory: Optional[Path] = None):
        """Initialize the enhanced logging system."""
        if self._is_initialized:
            return
        
        # Set up log directory
        if log_directory is None:
            log_directory = Path("logs")
        log_directory.mkdir(exist_ok=True)
        
        # Configure processors
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            EnhancedLoggingFormatter(include_source=True),
            ComplianceProcessor(enable_pii_redaction=True),
            SecurityEventProcessor(),
            structlog.processors.UnicodeDecoder(),
        ]
        
        # Add JSON renderer for structured output
        if enable_file_logging or enable_aggregation:
            processors.append(JSONRenderer())
        else:
            processors.append(structlog.processors.KeyValueRenderer())
        
        # Configure structlog
        configure(
            processors=processors,
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure standard logging
        logging.basicConfig(
            level=getattr(logging, log_level.value.upper()),
            format='%(message)s',
            handlers=[]
        )
        
        # Set up aggregation if enabled
        if enable_aggregation:
            handlers = []
            
            if enable_file_logging:
                file_handler = FileRotationHandler(
                    base_path=log_directory / "shyvr-rlte",
                    max_file_size=100 * 1024 * 1024,  # 100MB
                    max_files=10
                )
                handlers.append(file_handler)
            
            self.aggregator = LogAggregator(
                buffer_size=1000,
                flush_interval=30,
                output_handlers=handlers
            )
        
        self._is_initialized = True
    
    def get_logger(self, name: str, **context) -> structlog.BoundLogger:
        """Get enhanced logger with context."""
        if not self._is_initialized:
            self.initialize()
        
        # Merge context
        full_context = {**self.context.to_dict(), **context}
        
        return structlog.get_logger(name).bind(**full_context)
    
    def set_context(self, **context):
        """Update global context."""
        for key, value in context.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    async def shutdown(self):
        """Shutdown logging system."""
        if self.aggregator:
            await self.aggregator.force_flush()
    
    async def force_flush(self):
        """Force flush of aggregator."""
        if self.aggregator:
            await self.aggregator.force_flush()


# Global logging system instance
_logging_system = EnhancedLoggingSystem()


def initialize_logging(**kwargs):
    """Initialize the global logging system."""
    _logging_system.initialize(**kwargs)


def get_enhanced_logger(name: str, **context) -> structlog.BoundLogger:
    """Get enhanced logger with context."""
    return _logging_system.get_logger(name, **context)


def set_logging_context(**context):
    """Set global logging context."""
    _logging_system.set_context(**context)


async def shutdown_logging():
    """Shutdown logging system."""
    await _logging_system.shutdown()


def audit_log(action: str, 
              resource: str,
              user_id: Optional[str] = None,
              success: bool = True,
              details: Optional[Dict[str, Any]] = None,
              **kwargs):
    """Create audit log entry."""
    audit_logger = get_enhanced_logger("audit", category=LogCategory.AUDIT.value)
    
    audit_entry = {
        "action": action,
        "resource": resource,
        "user_id": user_id,
        "success": success,
        "details": details or {},
        **kwargs
    }
    
    if success:
        audit_logger.info("Audit event", **audit_entry)
    else:
        audit_logger.warning("Failed audit event", **audit_entry)


def security_log(event: str,
                 severity: str = "medium",
                 user_id: Optional[str] = None,
                 ip_address: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 **kwargs):
    """Create security log entry."""
    security_logger = get_enhanced_logger("security", category=LogCategory.SECURITY.value)
    
    security_entry = {
        "security_event": event,
        "severity": severity,
        "user_id": user_id,
        "ip_address": ip_address,
        "details": details or {},
        **kwargs
    }
    
    if severity in ["critical", "high"]:
        security_logger.error("Security event", **security_entry)
    elif severity == "medium":
        security_logger.warning("Security event", **security_entry)
    else:
        security_logger.info("Security event", **security_entry)


def compliance_log(regulation: str,
                   requirement: str,
                   status: str = "compliant",
                   evidence: Optional[Dict[str, Any]] = None,
                   **kwargs):
    """Create compliance log entry."""
    compliance_logger = get_enhanced_logger("compliance", category=LogCategory.COMPLIANCE.value)
    
    compliance_entry = {
        "regulation": regulation,
        "requirement": requirement,
        "status": status,
        "evidence": evidence or {},
        **kwargs
    }
    
    compliance_logger.info("Compliance event", **compliance_entry)


def financial_log(transaction_type: str,
                  amount: Optional[float] = None,
                  symbol: Optional[str] = None,
                  order_id: Optional[str] = None,
                  success: bool = True,
                  details: Optional[Dict[str, Any]] = None,
                  **kwargs):
    """Create financial transaction log entry."""
    financial_logger = get_enhanced_logger("financial", category=LogCategory.FINANCIAL.value)
    
    financial_entry = {
        "transaction_type": transaction_type,
        "amount": amount,
        "symbol": symbol,
        "order_id": order_id,
        "success": success,
        "details": details or {},
        **kwargs
    }
    
    if success:
        financial_logger.info("Financial transaction", **financial_entry)
    else:
        financial_logger.error("Failed financial transaction", **financial_entry)


def performance_log(operation: str,
                    duration_ms: float,
                    success: bool = True,
                    resource_usage: Optional[Dict[str, Any]] = None,
                    **kwargs):
    """Create performance log entry."""
    performance_logger = get_enhanced_logger("performance", category=LogCategory.PERFORMANCE.value)
    
    performance_entry = {
        "operation": operation,
        "duration_ms": duration_ms,
        "success": success,
        "resource_usage": resource_usage or {},
        **kwargs
    }
    
    performance_logger.info("Performance metric", **performance_entry)