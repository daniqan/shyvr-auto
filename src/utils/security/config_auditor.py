"""
Configuration audit logging utilities
Provides comprehensive audit logging for configuration access and modifications
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import os

logger = logging.getLogger(__name__)


class ConfigAuditor:
    """Handles audit logging for configuration access and modifications"""
    
    # Sensitive configuration keys that trigger alerts
    SENSITIVE_KEYS = {
        'password', 'secret', 'token', 'key', 'credentials',
        'private_key', 'webhook_secret', 'api_key', 'secret_key'
    }
    
    def __init__(self, 
                 audit_file: Optional[str] = None,
                 enable_alerts: bool = True,
                 alert_threshold: int = 5):
        """Initialize configuration auditor"""
        self.audit_file = audit_file or "/tmp/config_audit.log"
        self.enable_alerts = enable_alerts
        self.alert_threshold = alert_threshold
        self._access_counts = {}
        self._setup_audit_logger()
        logger.info(f"Configuration auditor initialized - audit file: {self.audit_file}")
    
    def _setup_audit_logger(self) -> None:
        """Setup dedicated audit logger"""
        self.audit_logger = logging.getLogger("config_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.audit_logger.handlers.clear()
        
        # Create file handler for audit logs
        handler = logging.FileHandler(self.audit_file)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        self.audit_logger.propagate = False
    
    def log_config_access(self, key: str, value: Any, user: str = "system") -> None:
        """Log configuration value access"""
        # Redact sensitive values
        redacted_value = self._redact_sensitive_value(key, value)
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'ACCESS',
            'key': key,
            'value': redacted_value,
            'user': user,
            'is_sensitive': self._is_sensitive_key(key)
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
        
        # Track access count for alert threshold
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        
        # Check for sensitive access patterns
        if self._is_sensitive_key(key):
            self._check_sensitive_access_patterns(key, user)
    
    def log_config_modification(self, key: str, old_value: Any, new_value: Any, user: str = "system") -> None:
        """Log configuration value modification"""
        redacted_old = self._redact_sensitive_value(key, old_value)
        redacted_new = self._redact_sensitive_value(key, new_value)
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'MODIFY',
            'key': key,
            'old_value': redacted_old,
            'new_value': redacted_new,
            'user': user,
            'is_sensitive': self._is_sensitive_key(key)
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
        
        # Always alert on sensitive modifications
        if self._is_sensitive_key(key):
            self.alert_sensitive_access(key, user, action="MODIFY")
    
    def alert_sensitive_access(self, key: str, user: str, action: str = "ACCESS") -> None:
        """Alert on sensitive configuration access"""
        if not self.enable_alerts:
            return
        
        alert_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'ALERT',
            'alert_type': 'SENSITIVE_ACCESS',
            'key': key,
            'user': user,
            'triggering_action': action,
            'severity': 'HIGH' if action == 'MODIFY' else 'MEDIUM'
        }
        
        self.audit_logger.warning(json.dumps(alert_entry))
        logger.warning(f"SECURITY ALERT: Sensitive configuration access - Key: {key}, User: {user}, Action: {action}")
    
    def _redact_sensitive_value(self, key: str, value: Any) -> str:
        """Redact sensitive configuration values for logging"""
        if not self._is_sensitive_key(key):
            return str(value)
        
        if value is None:
            return "***NULL***"
        
        value_str = str(value)
        if len(value_str) <= 4:
            return "***REDACTED***"
        
        # Show first 2 and last 2 characters for debugging
        return f"{value_str[:2]}***{value_str[-2:]}"
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if configuration key contains sensitive data"""
        key_lower = key.lower()
        return any(sensitive_word in key_lower for sensitive_word in self.SENSITIVE_KEYS)
    
    def _check_sensitive_access_patterns(self, key: str, user: str) -> None:
        """Check for suspicious access patterns"""
        access_count = self._access_counts.get(key, 0)
        
        if access_count > self.alert_threshold:
            alert_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'ALERT',
                'alert_type': 'EXCESSIVE_ACCESS',
                'key': key,
                'user': user,
                'access_count': access_count,
                'threshold': self.alert_threshold,
                'severity': 'HIGH'
            }
            
            self.audit_logger.error(json.dumps(alert_entry))
            logger.error(f"SECURITY ALERT: Excessive access to sensitive key {key} by {user} ({access_count} times)")
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for the specified time period"""
        # This would parse the audit log file and provide summary statistics
        summary = {
            'total_accesses': sum(self._access_counts.values()),
            'sensitive_accesses': sum(1 for key in self._access_counts.keys() if self._is_sensitive_key(key)),
            'most_accessed_keys': sorted(self._access_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'alert_count': self._count_alerts_in_period(hours)
        }
        return summary
    
    def _count_alerts_in_period(self, hours: int) -> int:
        """Count alerts in the specified time period"""
        # Simplified implementation - in production would parse the audit file
        return 0
    
    def clear_access_counts(self) -> None:
        """Clear access count tracking (for testing or periodic cleanup)"""
        self._access_counts.clear()
        logger.info("Access count tracking cleared")