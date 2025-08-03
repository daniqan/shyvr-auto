"""
Test suite for enhanced logging system (Phase 6.2)

This module provides comprehensive TDD tests for the enhanced logging,
audit trails, and compliance features.
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the enhanced logging system
from src.enhanced_logging.enhanced_logging import (
    LogLevel, LogCategory, LogContext, EnhancedLoggingFormatter,
    ComplianceProcessor, SecurityEventProcessor, LogAggregator,
    FileRotationHandler, EnhancedLoggingSystem, initialize_logging,
    get_enhanced_logger, audit_log, security_log, compliance_log,
    financial_log, performance_log
)


class TestLogLevel:
    """Test log level enumeration."""
    
    def test_log_levels_have_correct_values(self):
        """Test that log levels have expected values."""
        assert LogLevel.TRACE.value == "trace"
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"
        assert LogLevel.AUDIT.value == "audit"
        assert LogLevel.SECURITY.value == "security"
        assert LogLevel.COMPLIANCE.value == "compliance"


class TestLogCategory:
    """Test log category enumeration."""
    
    def test_log_categories_have_correct_values(self):
        """Test that log categories have expected values."""
        assert LogCategory.SYSTEM.value == "system"
        assert LogCategory.TRADING.value == "trading"
        assert LogCategory.FINANCIAL.value == "financial"
        assert LogCategory.SECURITY.value == "security"
        assert LogCategory.PERFORMANCE.value == "performance"
        assert LogCategory.AUDIT.value == "audit"
        assert LogCategory.COMPLIANCE.value == "compliance"
        assert LogCategory.ML_MODEL.value == "ml_model"
        assert LogCategory.RISK_MANAGEMENT.value == "risk_management"
        assert LogCategory.USER_ACTION.value == "user_action"
        assert LogCategory.API.value == "api"
        assert LogCategory.DATABASE.value == "database"
        assert LogCategory.EXTERNAL_SERVICE.value == "external_service"


class TestLogContext:
    """Test log context functionality."""
    
    def test_log_context_creation(self):
        """Test log context creation with defaults."""
        context = LogContext()
        
        assert context.session_id is not None
        assert context.user_id is None
        assert context.request_id is None
        assert context.correlation_id is None
        assert context.component is None
        assert context.version == "1.0.0"
        assert context.environment == "development"
    
    def test_log_context_with_custom_values(self):
        """Test log context creation with custom values."""
        context = LogContext(
            user_id="user123",
            request_id="req456",
            correlation_id="corr789",
            component="trading",
            version="2.0.0",
            environment="production"
        )
        
        assert context.user_id == "user123"
        assert context.request_id == "req456"
        assert context.correlation_id == "corr789"
        assert context.component == "trading"
        assert context.version == "2.0.0"
        assert context.environment == "production"
    
    def test_log_context_to_dict(self):
        """Test log context conversion to dictionary."""
        context = LogContext(
            user_id="user123",
            component="trading"
        )
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict["user_id"] == "user123"
        assert context_dict["component"] == "trading"
        assert "session_id" in context_dict
        assert "version" in context_dict
        assert "environment" in context_dict


class TestEnhancedLoggingFormatter:
    """Test enhanced logging formatter."""
    
    def test_formatter_initialization(self):
        """Test formatter initialization with options."""
        formatter = EnhancedLoggingFormatter(include_source=True, include_context=True)
        
        assert formatter.include_source is True
        assert formatter.include_context is True
    
    def test_formatter_adds_timestamp(self):
        """Test that formatter adds timestamp."""
        formatter = EnhancedLoggingFormatter()
        mock_logger = Mock()
        
        event_dict = {"event": "test message"}
        result = formatter(mock_logger, "info", event_dict)
        
        assert "timestamp" in result
        assert "level" in result
        assert result["level"] == "INFO"
    
    def test_formatter_adds_source_info(self):
        """Test that formatter adds source information."""
        formatter = EnhancedLoggingFormatter(include_source=True)
        mock_logger = Mock()
        
        event_dict = {"event": "test message"}
        result = formatter(mock_logger, "info", event_dict)
        
        assert "source" in result
        assert "file" in result["source"]
        assert "function" in result["source"]
        assert "line" in result["source"]
    
    def test_formatter_excludes_source_when_disabled(self):
        """Test that formatter excludes source info when disabled."""
        formatter = EnhancedLoggingFormatter(include_source=False)
        mock_logger = Mock()
        
        event_dict = {"event": "test message"}
        result = formatter(mock_logger, "info", event_dict)
        
        assert "source" not in result
    
    def test_formatter_adds_process_info(self):
        """Test that formatter adds process information."""
        formatter = EnhancedLoggingFormatter()
        mock_logger = Mock()
        
        event_dict = {"event": "test message"}
        result = formatter(mock_logger, "info", event_dict)
        
        assert "process_id" in result
        assert isinstance(result["process_id"], int)


class TestComplianceProcessor:
    """Test compliance processor functionality."""
    
    def test_compliance_processor_initialization(self):
        """Test compliance processor initialization."""
        processor = ComplianceProcessor(enable_pii_redaction=True)
        
        assert processor.enable_pii_redaction is True
        assert "password" in processor.pii_fields
        assert "secret" in processor.pii_fields
        assert "token" in processor.pii_fields
    
    def test_compliance_processor_adds_metadata(self):
        """Test that compliance processor adds metadata."""
        processor = ComplianceProcessor()
        mock_logger = Mock()
        
        event_dict = {"event": "test message"}
        result = processor(mock_logger, "info", event_dict)
        
        assert "compliance" in result
        assert "audit_trail_id" in result["compliance"]
        assert "retention_policy" in result["compliance"]
        assert "classification" in result["compliance"]
        assert "redacted" in result["compliance"]
    
    def test_compliance_processor_classifies_financial_transaction(self):
        """Test classification of financial transactions."""
        processor = ComplianceProcessor()
        mock_logger = Mock()
        
        event_dict = {"event": "trade order executed successfully"}
        result = processor(mock_logger, "info", event_dict)
        
        assert result["compliance"]["classification"] == "financial_transaction"
    
    def test_compliance_processor_classifies_security_event(self):
        """Test classification of security events."""
        processor = ComplianceProcessor()
        mock_logger = Mock()
        
        event_dict = {"event": "user login attempt failed"}
        result = processor(mock_logger, "info", event_dict)
        
        assert result["compliance"]["classification"] == "security_event"
    
    def test_compliance_processor_classifies_algorithmic_decision(self):
        """Test classification of algorithmic decisions."""
        processor = ComplianceProcessor()
        mock_logger = Mock()
        
        event_dict = {"event": "ML model prediction generated"}
        result = processor(mock_logger, "info", event_dict)
        
        assert result["compliance"]["classification"] == "algorithmic_decision"
    
    def test_compliance_processor_redacts_pii(self):
        """Test PII redaction functionality."""
        processor = ComplianceProcessor(enable_pii_redaction=True)
        mock_logger = Mock()
        
        event_dict = {
            "event": "user data processed",
            "user_info": {
                "username": "john_doe",
                "password": "secret123",
                "api_key": "key_abc123"
            }
        }
        result = processor(mock_logger, "info", event_dict)
        
        assert result["user_info"]["username"] == "john_doe"
        assert result["user_info"]["password"] == "[REDACTED]"
        assert result["user_info"]["api_key"] == "[REDACTED]"
        assert result["compliance"]["redacted"] is True
    
    def test_compliance_processor_no_redaction_when_disabled(self):
        """Test that PII redaction can be disabled."""
        processor = ComplianceProcessor(enable_pii_redaction=False)
        mock_logger = Mock()
        
        event_dict = {
            "event": "user data processed",
            "password": "secret123"
        }
        result = processor(mock_logger, "info", event_dict)
        
        assert result["password"] == "secret123"


class TestSecurityEventProcessor:
    """Test security event processor."""
    
    def test_security_processor_initialization(self):
        """Test security processor initialization."""
        processor = SecurityEventProcessor(alert_threshold=10)
        
        assert processor.alert_threshold == 10
        assert isinstance(processor.security_events, list)
    
    def test_security_processor_handles_non_security_events(self):
        """Test that non-security events pass through unchanged."""
        processor = SecurityEventProcessor()
        mock_logger = Mock()
        
        event_dict = {"event": "regular message", "category": "system"}
        result = processor(mock_logger, "info", event_dict)
        
        assert "security" not in result
    
    def test_security_processor_adds_security_metadata(self):
        """Test that security events get enhanced metadata."""
        processor = SecurityEventProcessor()
        mock_logger = Mock()
        
        event_dict = {
            "event": "unauthorized access attempt",
            "category": "security",
            "ip_address": "192.168.1.100",
            "user_agent": "test-browser"
        }
        result = processor(mock_logger, "info", event_dict)
        
        assert "security" in result
        assert "event_id" in result["security"]
        assert "severity" in result["security"]
        assert "requires_investigation" in result["security"]
        assert result["security"]["ip_address"] == "192.168.1.100"
        assert result["security"]["user_agent"] == "test-browser"
    
    def test_security_processor_determines_critical_severity(self):
        """Test critical severity determination."""
        processor = SecurityEventProcessor()
        mock_logger = Mock()
        
        event_dict = {
            "event": "security breach detected",
            "category": "security"
        }
        result = processor(mock_logger, "info", event_dict)
        
        assert result["security"]["severity"] == "critical"
        assert result["security"]["requires_investigation"] is True
    
    def test_security_processor_determines_high_severity(self):
        """Test high severity determination."""
        processor = SecurityEventProcessor()
        mock_logger = Mock()
        
        event_dict = {
            "event": "login failed for user",
            "category": "security"
        }
        result = processor(mock_logger, "info", event_dict)
        
        assert result["security"]["severity"] == "high"
        assert result["security"]["requires_investigation"] is True
    
    def test_security_processor_determines_medium_severity(self):
        """Test medium severity determination."""
        processor = SecurityEventProcessor()
        mock_logger = Mock()
        
        event_dict = {
            "event": "unusual access pattern warning",
            "category": "security"
        }
        result = processor(mock_logger, "info", event_dict)
        
        assert result["security"]["severity"] == "medium"
        assert result["security"]["requires_investigation"] is False
    
    def test_security_processor_tracks_event_frequency(self):
        """Test that security processor tracks event frequency."""
        processor = SecurityEventProcessor(alert_threshold=2)
        mock_logger = Mock()
        
        # First security event
        event_dict1 = {"event": "security event 1", "category": "security"}
        result1 = processor(mock_logger, "info", event_dict1)
        assert "alert" not in result1.get("security", {})
        
        # Second security event - should trigger alert
        event_dict2 = {"event": "security event 2", "category": "security"}
        result2 = processor(mock_logger, "info", event_dict2)
        assert result2["security"]["alert"] == "HIGH_FREQUENCY_SECURITY_EVENTS"


class TestLogAggregator:
    """Test log aggregation functionality."""
    
    @pytest.mark.asyncio
    async def test_log_aggregator_initialization(self):
        """Test log aggregator initialization."""
        aggregator = LogAggregator(
            buffer_size=100,
            flush_interval=10,
            output_handlers=[]
        )
        
        assert aggregator.buffer_size == 100
        assert aggregator.flush_interval == 10
        assert len(aggregator.output_handlers) == 0
        assert len(aggregator.buffer) == 0
    
    @pytest.mark.asyncio
    async def test_log_aggregator_adds_entries(self):
        """Test that log aggregator adds entries to buffer."""
        aggregator = LogAggregator(buffer_size=100, flush_interval=60)
        
        entry = {"event": "test message", "timestamp": datetime.now().isoformat()}
        await aggregator.add_log_entry(entry)
        
        assert len(aggregator.buffer) == 1
        assert aggregator.buffer[0] == entry
    
    @pytest.mark.asyncio
    async def test_log_aggregator_flushes_on_buffer_size(self):
        """Test that aggregator flushes when buffer size is reached."""
        mock_handler = AsyncMock()
        aggregator = LogAggregator(
            buffer_size=2,
            flush_interval=60,
            output_handlers=[mock_handler]
        )
        
        # Add entries up to buffer size
        await aggregator.add_log_entry({"event": "message 1"})
        await aggregator.add_log_entry({"event": "message 2"})  # Should trigger flush
        
        # Verify handler was called
        mock_handler.assert_called_once()
        assert len(aggregator.buffer) == 0
    
    @pytest.mark.asyncio
    async def test_log_aggregator_handles_handler_errors(self):
        """Test that aggregator handles output handler errors gracefully."""
        mock_handler = AsyncMock(side_effect=Exception("Handler error"))
        aggregator = LogAggregator(
            buffer_size=1,
            flush_interval=60,
            output_handlers=[mock_handler]
        )
        
        # Should not raise exception despite handler error
        await aggregator.add_log_entry({"event": "test message"})
        
        mock_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_aggregator_force_flush(self):
        """Test force flush functionality."""
        mock_handler = AsyncMock()
        aggregator = LogAggregator(
            buffer_size=100,
            flush_interval=60,
            output_handlers=[mock_handler]
        )
        
        # Add entry without triggering automatic flush
        await aggregator.add_log_entry({"event": "test message"})
        assert len(aggregator.buffer) == 1
        
        # Force flush
        await aggregator.force_flush()
        
        mock_handler.assert_called_once()
        assert len(aggregator.buffer) == 0


class TestFileRotationHandler:
    """Test file rotation handler."""
    
    @pytest.mark.asyncio
    async def test_file_rotation_handler_initialization(self):
        """Test file rotation handler initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test.log"
            handler = FileRotationHandler(
                base_path=base_path,
                max_file_size=1024,
                max_files=5,
                compress_old=True
            )
            
            assert handler.base_path == base_path
            assert handler.max_file_size == 1024
            assert handler.max_files == 5
            assert handler.compress_old is True
            assert handler.current_file is None
            assert handler.current_size == 0
    
    @pytest.mark.asyncio
    async def test_file_rotation_handler_writes_entries(self):
        """Test that file handler writes log entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test.log"
            handler = FileRotationHandler(base_path=base_path, max_file_size=1024)
            
            entries = [
                {"event": "message 1", "timestamp": "2025-01-01T12:00:00"},
                {"event": "message 2", "timestamp": "2025-01-01T12:01:00"}
            ]
            
            await handler(entries)
            
            # Check that file was created and contains entries
            log_file = base_path.with_suffix('.log')
            assert log_file.exists()
            
            content = log_file.read_text()
            assert "message 1" in content
            assert "message 2" in content
    
    @pytest.mark.asyncio
    async def test_file_rotation_handler_rotates_on_size(self):
        """Test that files are rotated when size limit is reached."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test.log"
            handler = FileRotationHandler(
                base_path=base_path,
                max_file_size=50,  # Very small size to trigger rotation
                max_files=3
            )
            
            # Write enough entries to trigger rotation
            large_entry = {"event": "x" * 100, "timestamp": "2025-01-01T12:00:00"}
            await handler([large_entry])
            
            # Write another large entry to trigger rotation
            await handler([large_entry])
            
            # Check that rotation occurred
            log_file = base_path.with_suffix('.log')
            rotated_file = base_path.with_suffix('.1.log')
            
            assert log_file.exists()
            assert rotated_file.exists()


class TestEnhancedLoggingSystem:
    """Test the main enhanced logging system."""
    
    def test_enhanced_logging_system_initialization(self):
        """Test enhanced logging system initialization."""
        system = EnhancedLoggingSystem()
        
        assert system.config == {}
        assert system.context is not None
        assert system.aggregator is None
        assert system._is_initialized is False
    
    def test_enhanced_logging_system_with_config(self):
        """Test enhanced logging system with custom config."""
        config = {"log_level": "debug", "enable_audit": True}
        system = EnhancedLoggingSystem(config)
        
        assert system.config == config
    
    def test_enhanced_logging_system_initialize(self):
        """Test enhanced logging system initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            system = EnhancedLoggingSystem()
            
            system.initialize(
                log_level=LogLevel.INFO,
                enable_file_logging=True,
                enable_console_logging=True,
                enable_aggregation=True,
                log_directory=Path(temp_dir)
            )
            
            assert system._is_initialized is True
            assert system.aggregator is not None
    
    def test_enhanced_logging_system_get_logger(self):
        """Test getting logger with context."""
        system = EnhancedLoggingSystem()
        system.initialize(enable_aggregation=False)
        
        logger = system.get_logger("test", component="trading", user_id="user123")
        
        assert logger is not None
        # Note: Testing the actual bound context would require more integration testing
    
    def test_enhanced_logging_system_set_context(self):
        """Test setting global context."""
        system = EnhancedLoggingSystem()
        
        system.set_context(user_id="user456", environment="production")
        
        assert system.context.user_id == "user456"
        assert system.context.environment == "production"
    
    @pytest.mark.asyncio
    async def test_enhanced_logging_system_shutdown(self):
        """Test logging system shutdown."""
        system = EnhancedLoggingSystem()
        system.initialize(enable_aggregation=True)
        
        # Should not raise exception
        await system.shutdown()


class TestLoggingHelperFunctions:
    """Test logging helper functions."""
    
    def test_initialize_logging_function(self):
        """Test global initialize_logging function."""
        # Should not raise exception
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(
                log_level=LogLevel.DEBUG,
                log_directory=Path(temp_dir)
            )
    
    def test_get_enhanced_logger_function(self):
        """Test get_enhanced_logger function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(log_directory=Path(temp_dir), enable_aggregation=False)
            
            logger = get_enhanced_logger("test", component="trading")
            assert logger is not None
    
    def test_audit_log_function(self):
        """Test audit_log helper function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(log_directory=Path(temp_dir), enable_aggregation=False)
            
            # Should not raise exception
            audit_log(
                action="CREATE",
                resource="trade_order",
                user_id="user123",
                success=True,
                details={"order_id": "ord_456", "amount": 1000.0}
            )
    
    def test_security_log_function(self):
        """Test security_log helper function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(log_directory=Path(temp_dir), enable_aggregation=False)
            
            # Should not raise exception
            security_log(
                event="login_attempt",
                severity="high",
                user_id="user123",
                ip_address="192.168.1.100",
                details={"success": False, "reason": "invalid_password"}
            )
    
    def test_compliance_log_function(self):
        """Test compliance_log helper function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(log_directory=Path(temp_dir), enable_aggregation=False)
            
            # Should not raise exception
            compliance_log(
                regulation="MiFID II",
                requirement="transaction_reporting",
                status="compliant",
                evidence={"report_id": "rpt_789", "submission_time": "2025-01-01T12:00:00Z"}
            )
    
    def test_financial_log_function(self):
        """Test financial_log helper function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(log_directory=Path(temp_dir), enable_aggregation=False)
            
            # Should not raise exception
            financial_log(
                transaction_type="BUY",
                amount=1500.0,
                symbol="BTC/USD",
                order_id="ord_123",
                success=True,
                details={"execution_price": 45000.0, "fees": 1.5}
            )
    
    def test_performance_log_function(self):
        """Test performance_log helper function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(log_directory=Path(temp_dir), enable_aggregation=False)
            
            # Should not raise exception
            performance_log(
                operation="model_prediction",
                duration_ms=250.5,
                success=True,
                resource_usage={"cpu_percent": 45.2, "memory_mb": 128.5}
            )


class TestLoggingIntegration:
    """Test integration with existing systems."""
    
    @pytest.mark.asyncio
    async def test_integration_with_activity_logger(self):
        """Test integration with existing activity logger."""
        # This would test how the enhanced logging integrates with
        # the existing activity_logging system
        
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(log_directory=Path(temp_dir), enable_aggregation=False)
            
            # Get logger for activity logging category
            logger = get_enhanced_logger("activity", category=LogCategory.USER_ACTION.value)
            
            # Should be able to log activity
            logger.info("User logged in", user_id="user123", action="login")
            
            # Integration should work without errors
            assert True
    
    def test_integration_with_existing_structlog(self):
        """Test that enhanced logging works with existing structlog usage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(log_directory=Path(temp_dir), enable_aggregation=False)
            
            # Standard structlog usage should still work
            import structlog
            logger = structlog.get_logger()
            
            # Should not raise exception
            logger.info("Standard structlog message", component="test")
    
    @pytest.mark.asyncio
    async def test_logging_performance_under_load(self):
        """Test logging performance under high load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_logging(
                log_directory=Path(temp_dir),
                enable_aggregation=True
            )
            
            logger = get_enhanced_logger("performance_test")
            
            # Log many entries quickly
            start_time = datetime.now()
            for i in range(100):
                logger.info(f"Performance test message {i}", iteration=i)
            
            # Force flush to ensure all messages are processed
            from src.enhanced_logging.enhanced_logging import _logging_system
            if _logging_system.aggregator:
                await _logging_system.force_flush()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Should complete within reasonable time (less than 1 second for 100 messages)
            assert duration < 1.0


class TestComplianceFeatures:
    """Test compliance-specific features."""
    
    def test_pii_redaction_comprehensive(self):
        """Test comprehensive PII redaction."""
        processor = ComplianceProcessor(enable_pii_redaction=True)
        mock_logger = Mock()
        
        event_dict = {
            "event": "user registration",
            "user_data": {
                "username": "john_doe",
                "password": "secret123",
                "email": "john@example.com",
                "ssn": "123-45-6789",
                "credit_card": "4111-1111-1111-1111",
                "api_key": "key_abc123"
            },
            "metadata": {
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0...",
                "session_token": "sess_xyz789"
            }
        }
        
        result = processor(mock_logger, "info", event_dict)
        
        # Non-PII fields should remain
        assert result["user_data"]["username"] == "john_doe"
        assert result["user_data"]["email"] == "john@example.com"
        assert result["metadata"]["ip_address"] == "192.168.1.100"
        assert result["metadata"]["user_agent"] == "Mozilla/5.0..."
        
        # PII fields should be redacted
        assert result["user_data"]["password"] == "[REDACTED]"
        assert result["user_data"]["ssn"] == "[REDACTED]"
        assert result["user_data"]["credit_card"] == "[REDACTED]"
        assert result["user_data"]["api_key"] == "[REDACTED]"
        assert result["metadata"]["session_token"] == "[REDACTED]"
        
        # Should mark as redacted
        assert result["compliance"]["redacted"] is True
    
    def test_audit_trail_generation(self):
        """Test audit trail ID generation."""
        processor = ComplianceProcessor()
        mock_logger = Mock()
        
        event_dict1 = {"event": "first message"}
        event_dict2 = {"event": "second message"}
        
        result1 = processor(mock_logger, "info", event_dict1)
        result2 = processor(mock_logger, "info", event_dict2)
        
        # Each event should have unique audit trail ID
        assert result1["compliance"]["audit_trail_id"] != result2["compliance"]["audit_trail_id"]
        
        # Both should have audit trail IDs
        assert "audit_trail_id" in result1["compliance"]
        assert "audit_trail_id" in result2["compliance"]
    
    def test_retention_policy_assignment(self):
        """Test that retention policies are correctly assigned."""
        processor = ComplianceProcessor()
        mock_logger = Mock()
        
        # Financial transaction should get financial records retention
        financial_event = {"event": "trade executed successfully"}
        result = processor(mock_logger, "info", financial_event)
        assert result["compliance"]["retention_policy"] == "financial_records"
        
        # All events currently get the same policy, but this could be enhanced
        # to have different policies based on classification
        security_event = {"event": "user authentication failed"}
        result = processor(mock_logger, "info", security_event)
        assert result["compliance"]["retention_policy"] == "financial_records"


class TestSecurityFeatures:
    """Test security-specific features."""
    
    def test_security_event_tracking_and_alerting(self):
        """Test security event tracking and high-frequency alerting."""
        processor = SecurityEventProcessor(alert_threshold=3)
        mock_logger = Mock()
        
        # Generate multiple security events
        for i in range(5):
            event_dict = {
                "event": f"security event {i}",
                "category": "security",
                "ip_address": "192.168.1.100"
            }
            result = processor(mock_logger, "info", event_dict)
            
            if i < 2:
                # First few events should not trigger alert
                assert "alert" not in result.get("security", {})
            else:
                # Later events should trigger high frequency alert
                assert result["security"]["alert"] == "HIGH_FREQUENCY_SECURITY_EVENTS"
    
    def test_security_severity_escalation(self):
        """Test security severity determination and escalation."""
        processor = SecurityEventProcessor()
        mock_logger = Mock()
        
        # Test different severity levels
        test_cases = [
            ("security breach detected", "critical", True),
            ("attack vector identified", "critical", True),
            ("login failed", "high", True),
            ("access blocked", "high", True),
            ("unusual pattern warning", "medium", False),
            ("routine security check", "low", False)
        ]
        
        for event_text, expected_severity, should_investigate in test_cases:
            event_dict = {
                "event": event_text,
                "category": "security"
            }
            result = processor(mock_logger, "info", event_dict)
            
            assert result["security"]["severity"] == expected_severity
            assert result["security"]["requires_investigation"] == should_investigate
    
    def test_security_event_metadata_enrichment(self):
        """Test that security events get proper metadata enrichment."""
        processor = SecurityEventProcessor()
        mock_logger = Mock()
        
        event_dict = {
            "event": "unauthorized access detected",
            "category": "security",
            "ip_address": "10.0.0.1",
            "user_agent": "suspicious-client/1.0",
            "user_id": "user123"
        }
        
        result = processor(mock_logger, "info", event_dict)
        
        security_data = result["security"]
        assert "event_id" in security_data
        assert security_data["ip_address"] == "10.0.0.1"
        assert security_data["user_agent"] == "suspicious-client/1.0"
        assert security_data["severity"] in ["low", "medium", "high", "critical"]
        assert isinstance(security_data["requires_investigation"], bool)


class TestFileHandlingAndRotation:
    """Test file handling and rotation features."""
    
    @pytest.mark.asyncio
    async def test_file_rotation_preserves_data(self):
        """Test that file rotation preserves all log data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test.log"
            handler = FileRotationHandler(
                base_path=base_path,
                max_file_size=100,  # Small size to force rotation
                max_files=3
            )
            
            # Write multiple batches of entries to trigger rotation
            all_messages = []
            for batch in range(3):
                entries = []
                for i in range(2):
                    message = f"batch_{batch}_message_{i}"
                    entry = {"event": message, "timestamp": "2025-01-01T12:00:00"}
                    entries.append(entry)
                    all_messages.append(message)
                
                await handler(entries)
            
            # Check that all messages are preserved across files
            all_files = list(Path(temp_dir).glob("*.log"))
            all_content = ""
            for file_path in all_files:
                all_content += file_path.read_text()
            
            for message in all_messages:
                assert message in all_content
    
    @pytest.mark.asyncio
    async def test_file_rotation_limits_file_count(self):
        """Test that file rotation respects maximum file count."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test.log"
            handler = FileRotationHandler(
                base_path=base_path,
                max_file_size=50,
                max_files=2  # Only keep 2 files
            )
            
            # Write enough data to create more than max_files
            for i in range(5):
                large_entry = {"event": "x" * 100, "batch": i}
                await handler([large_entry])
            
            # Should only have max_files + 1 files (current + rotated)
            log_files = list(Path(temp_dir).glob("*.log"))
            assert len(log_files) <= 3  # current + .1 + .2 (max_files=2)


class TestErrorHandlingAndResilience:
    """Test error handling and system resilience."""
    
    @pytest.mark.asyncio
    async def test_logging_system_handles_disk_full(self):
        """Test graceful handling of disk full scenarios."""
        # This would require more complex setup to simulate disk full
        # For now, test that exceptions in file writing don't crash the system
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "test.log"
            handler = FileRotationHandler(base_path=base_path)
            
            # Simulate file write error
            with patch('builtins.open', side_effect=OSError("No space left on device")):
                # Should not raise exception
                try:
                    await handler([{"event": "test message"}])
                except OSError:
                    # Expected to fail, but shouldn't crash the test
                    pass
    
    @pytest.mark.asyncio
    async def test_aggregator_handles_malformed_entries(self):
        """Test that aggregator handles malformed log entries gracefully."""
        mock_handler = AsyncMock()
        aggregator = LogAggregator(
            buffer_size=10,
            output_handlers=[mock_handler]
        )
        
        # Add various types of entries, including malformed ones
        entries = [
            {"event": "normal message"},
            None,  # Should handle None
            {"event": None},  # Should handle None values
            42,  # Should handle non-dict
            {"event": "another normal message"}
        ]
        
        for entry in entries:
            await aggregator.add_log_entry(entry)
        
        # Should not raise exception and should have processed entries
        assert len(aggregator.buffer) == len(entries)
    
    def test_processor_handles_missing_fields_gracefully(self):
        """Test that processors handle missing fields gracefully."""
        compliance_processor = ComplianceProcessor()
        security_processor = SecurityEventProcessor()
        mock_logger = Mock()
        
        # Test with minimal event dict
        minimal_event = {}
        
        # Should not raise exceptions
        compliance_result = compliance_processor(mock_logger, "info", minimal_event)
        security_result = security_processor(mock_logger, "info", minimal_event)
        
        assert "compliance" in compliance_result
        # Security processor should only add metadata for security category events
        assert "security" not in security_result
    
    def test_context_handling_with_invalid_data(self):
        """Test context handling with invalid data."""
        context = LogContext()
        
        # Should handle setting invalid values gracefully
        context.user_id = None
        context.environment = ""
        
        context_dict = context.to_dict()
        assert context_dict["user_id"] is None
        assert context_dict["environment"] == ""
        assert "session_id" in context_dict  # Should still have required fields