"""
Phase 7.3: Security Monitoring & Audit Trail Tests
Following TDD methodology - these tests validate security monitoring and audit systems
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

from src.activity_logging.activity_logger import (
    ActivityLogger, ActivityCategory, ActivityAction, ActivitySeverity
)
from src.monitoring.alerting import AlertingSystem
from src.monitoring.notifications import NotificationSystem


class TestSecurityAuditLogging:
    """Test security audit logging following TDD methodology"""
    
    @pytest.fixture
    def activity_logger(self):
        """Create ActivityLogger for testing"""
        return ActivityLogger()
    
    @pytest.fixture
    def security_events(self):
        """Sample security events for testing"""
        return [
            {
                "category": ActivityCategory.SECURITY,
                "action": ActivityAction.LOGIN,
                "event_type": "user_login",
                "user_id": 12345,
                "metadata": {"ip_address": "192.168.1.100", "user_agent": "Chrome"}
            },
            {
                "category": ActivityCategory.SECURITY,
                "action": ActivityAction.VIOLATION,
                "event_type": "failed_authentication",
                "user_id": None,
                "metadata": {"ip_address": "10.0.0.50", "attempts": 5}
            },
            {
                "category": ActivityCategory.SECURITY,
                "action": ActivityAction.ACCESS,
                "event_type": "admin_panel_access",
                "user_id": 67890,
                "metadata": {"admin_action": "user_management", "target_user": "trader001"}
            }
        ]

    @pytest.mark.asyncio
    async def test_security_events_should_be_logged_with_complete_metadata(self, activity_logger, security_events):
        """Test security events are logged with complete metadata - TDD FAIL FIRST"""
        # Mock the database insert to capture logged data
        logged_events = []
        
        async def mock_insert_activity(data):
            logged_events.append(data)
        
        with patch.object(activity_logger, '_insert_activity_to_db', side_effect=mock_insert_activity):
            # Log each security event
            for event in security_events:
                await activity_logger.log_activity(
                    category=event["category"],
                    action=event["action"],
                    source="security_test",
                    event_type=event["event_type"],
                    title=f"Security event: {event['event_type']}",
                    user_id=event.get("user_id"),
                    metadata=event["metadata"]
                )
        
        # Verify all events were logged
        assert len(logged_events) == len(security_events), \
            "All security events should be logged"
        
        # Verify required security fields are present
        for logged_event in logged_events:
            assert "timestamp" in logged_event, "Security event should have timestamp"
            assert "category" in logged_event, "Security event should have category"
            assert "action" in logged_event, "Security event should have action"
            assert "source" in logged_event, "Security event should have source"
            assert "event_type" in logged_event, "Security event should have event type"
            assert "metadata" in logged_event, "Security event should have metadata"
            
            # Verify metadata contains security-relevant information
            metadata = logged_event["metadata"]
            if "ip_address" in metadata:
                assert metadata["ip_address"], "IP address should be logged for security events"

    @pytest.mark.asyncio
    async def test_failed_authentication_attempts_should_trigger_security_alerts(self, activity_logger):
        """Test failed authentication attempts trigger security alerts - TDD FAIL FIRST"""
        # Mock alerting system
        security_alerts = []
        
        async def mock_send_alert(alert_data):
            security_alerts.append(alert_data)
        
        with patch('src.monitoring.alerting.AlertingSystem.send_security_alert', side_effect=mock_send_alert):
            # Simulate multiple failed authentication attempts
            for i in range(5):  # 5 failed attempts
                await activity_logger.log_activity(
                    category=ActivityCategory.SECURITY,
                    action=ActivityAction.VIOLATION,
                    source="authentication_system",
                    event_type="failed_login_attempt",
                    title="Failed login attempt",
                    severity=ActivitySeverity.WARNING,
                    user_id=None,
                    metadata={
                        "ip_address": "192.168.1.100",
                        "username": "admin",
                        "attempt_number": i + 1
                    }
                )
        
        # Should trigger security alert for multiple failed attempts
        assert len(security_alerts) > 0, "Multiple failed authentication attempts should trigger security alert"
        
        # Verify alert contains relevant information
        alert = security_alerts[0]
        assert "failed_login" in alert.get("alert_type", "").lower(), \
            "Alert should indicate failed login attempts"
        assert "192.168.1.100" in str(alert.get("details", {})), \
            "Alert should include IP address"

    @pytest.mark.asyncio
    async def test_privileged_access_should_be_audited_with_elevated_logging(self, activity_logger):
        """Test privileged access is audited with elevated logging - TDD FAIL FIRST"""
        # Mock elevated logging for privileged actions
        elevated_logs = []
        
        async def mock_elevated_log(log_data):
            elevated_logs.append(log_data)
        
        with patch.object(activity_logger, '_log_privileged_action', side_effect=mock_elevated_log):
            # Simulate privileged actions
            privileged_actions = [
                {
                    "action": "user_creation",
                    "target": "new_trader_001",
                    "permissions": ["trading", "analysis"]
                },
                {
                    "action": "permission_grant",
                    "target": "trader_002", 
                    "new_permissions": ["admin"]
                },
                {
                    "action": "system_configuration_change",
                    "config_section": "trading_limits",
                    "old_value": "10000",
                    "new_value": "50000"
                }
            ]
            
            for action in privileged_actions:
                await activity_logger.log_activity(
                    category=ActivityCategory.SECURITY,
                    action=ActivityAction.MODIFY,
                    source="admin_panel",
                    event_type="privileged_action",
                    title=f"Privileged action: {action['action']}",
                    user_id=12345,
                    security_level="elevated",
                    metadata=action
                )
        
        # Verify privileged actions were logged with elevation
        assert len(elevated_logs) == len(privileged_actions), \
            "All privileged actions should have elevated logging"
        
        # Verify elevated log structure
        for log in elevated_logs:
            assert log.get("security_level") == "elevated", \
                "Privileged actions should have elevated security level"
            assert "admin_panel" in log.get("source", ""), \
                "Should track source of privileged action"

    @pytest.mark.asyncio
    async def test_security_log_tampering_detection_should_identify_modifications(self, activity_logger):
        """Test security log tampering detection identifies modifications - TDD FAIL FIRST"""
        # Mock log integrity checking
        tampered_logs = []
        
        def mock_integrity_check(log_entry):
            # Simulate tampered log detection
            if "tampered" in str(log_entry.get("metadata", {})):
                tampered_logs.append(log_entry)
                return False
            return True
        
        with patch.object(activity_logger, '_verify_log_integrity', side_effect=mock_integrity_check):
            # Log normal security event
            await activity_logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.ACCESS,
                source="security_system",
                event_type="normal_access",
                title="Normal security event",
                user_id=12345,
                metadata={"action": "file_access"}
            )
            
            # Simulate tampered log entry
            await activity_logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.ACCESS,
                source="security_system",
                event_type="suspicious_access",
                title="Potentially tampered log",
                user_id=12345,
                metadata={"action": "file_access", "tampered": True}
            )
        
        # Should detect tampered logs
        assert len(tampered_logs) == 1, "Should detect tampered log entries"
        assert tampered_logs[0]["metadata"]["tampered"] is True, \
            "Should identify the specific tampered entry"

    def test_audit_log_retention_should_maintain_logs_for_compliance_period(self):
        """Test audit log retention maintains logs for compliance period - TDD FAIL FIRST"""
        # Test log retention policy
        retention_policy = {
            "security_events": timedelta(days=2557),  # 7 years for security events
            "trading_events": timedelta(days=2557),   # 7 years for trading events
            "system_events": timedelta(days=1095),    # 3 years for system events
            "general_events": timedelta(days=365)     # 1 year for general events
        }
        
        # Mock current time and old log entries
        current_time = datetime.now()
        
        # Create logs of different ages
        test_logs = [
            {
                "category": "SECURITY",
                "timestamp": current_time - timedelta(days=1000),  # 2.7 years old
                "event_type": "security_violation"
            },
            {
                "category": "TRADING", 
                "timestamp": current_time - timedelta(days=3000),  # 8.2 years old
                "event_type": "trade_execution"
            },
            {
                "category": "SYSTEM",
                "timestamp": current_time - timedelta(days=500),   # 1.4 years old
                "event_type": "system_startup"
            }
        ]
        
        # Test retention logic
        logs_to_retain = []
        logs_to_archive = []
        
        for log in test_logs:
            log_age = current_time - log["timestamp"]
            category = log["category"].lower()
            
            if category in ["security", "trading"]:
                retention_period = retention_policy["security_events"]
            elif category == "system":
                retention_period = retention_policy["system_events"]
            else:
                retention_period = retention_policy["general_events"]
            
            if log_age <= retention_period:
                logs_to_retain.append(log)
            else:
                logs_to_archive.append(log)
        
        # Verify retention logic
        assert len(logs_to_retain) == 2, "Should retain security and system logs within retention period"
        assert len(logs_to_archive) == 1, "Should archive trading log exceeding retention period"
        
        # Verify correct logs are retained
        retained_types = [log["event_type"] for log in logs_to_retain]
        assert "security_violation" in retained_types, "Security logs should be retained for 7 years"
        assert "system_startup" in retained_types, "System logs should be retained for 3 years"


class TestSecurityMonitoring:
    """Test security monitoring and alerting systems"""
    
    @pytest.fixture
    def alerting_system(self):
        """Create AlertingSystem for testing"""
        return AlertingSystem()
    
    @pytest.fixture 
    def notification_system(self):
        """Create NotificationSystem for testing"""
        return NotificationSystem()

    @pytest.mark.asyncio
    async def test_suspicious_activity_detection_should_trigger_real_time_alerts(self, alerting_system):
        """Test suspicious activity detection triggers real-time alerts - TDD FAIL FIRST"""
        # Mock alert destinations
        sent_alerts = []
        
        async def mock_send_alert(alert_type, details, severity):
            sent_alerts.append({
                "type": alert_type,
                "details": details,
                "severity": severity,
                "timestamp": datetime.now()
            })
        
        with patch.object(alerting_system, 'send_alert', side_effect=mock_send_alert):
            # Simulate suspicious activities
            suspicious_activities = [
                {
                    "type": "multiple_failed_logins",
                    "ip_address": "192.168.1.100",
                    "attempts": 10,
                    "timeframe": "5 minutes"
                },
                {
                    "type": "privilege_escalation_attempt",
                    "user": "regular_user",
                    "attempted_action": "admin_panel_access",
                    "permission_level": "read_only"
                },
                {
                    "type": "unusual_trading_pattern",
                    "user": "trader_001",
                    "trade_volume": "1000x_normal",
                    "asset": "BTC/USD"
                },
                {
                    "type": "off_hours_access",
                    "user": "admin_user",
                    "access_time": "03:30 AM",
                    "location": "unusual_ip_range"
                }
            ]
            
            # Trigger alerts for each suspicious activity
            for activity in suspicious_activities:
                await alerting_system.detect_and_alert_suspicious_activity(activity)
        
        # Verify alerts were triggered
        assert len(sent_alerts) == len(suspicious_activities), \
            "All suspicious activities should trigger alerts"
        
        # Verify alert content
        alert_types = [alert["type"] for alert in sent_alerts]
        assert "security_violation" in alert_types or "suspicious_activity" in alert_types, \
            "Should categorize alerts as security violations"

    def test_security_metrics_collection_should_track_key_indicators(self):
        """Test security metrics collection tracks key indicators - TDD FAIL FIRST"""
        # Mock security metrics collector
        security_metrics = SecurityMetricsCollector()
        
        # Simulate collecting security metrics over time
        metrics_data = {
            "failed_login_attempts": 15,
            "successful_logins": 85,
            "admin_actions": 3,
            "trading_violations": 0,
            "unusual_access_patterns": 2,
            "api_key_usage_anomalies": 1,
            "database_access_violations": 0,
            "privileged_escalations": 0
        }
        
        # Process metrics
        processed_metrics = security_metrics.process_security_metrics(metrics_data)
        
        # Verify key security indicators are tracked
        assert "login_failure_rate" in processed_metrics, \
            "Should calculate login failure rate"
        assert "admin_activity_level" in processed_metrics, \
            "Should track administrative activity"
        assert "security_violation_count" in processed_metrics, \
            "Should count security violations"
        
        # Verify calculated metrics
        expected_failure_rate = 15 / (15 + 85)  # 15%
        assert abs(processed_metrics["login_failure_rate"] - expected_failure_rate) < 0.01, \
            "Login failure rate should be calculated correctly"

    @pytest.mark.asyncio
    async def test_automated_response_should_trigger_for_critical_threats(self, alerting_system):
        """Test automated response triggers for critical threats - TDD FAIL FIRST"""
        # Mock automated response actions
        automated_actions = []
        
        async def mock_execute_response(action_type, details):
            automated_actions.append({
                "action": action_type,
                "details": details,
                "timestamp": datetime.now()
            })
        
        with patch.object(alerting_system, 'execute_automated_response', side_effect=mock_execute_response):
            # Simulate critical security threats
            critical_threats = [
                {
                    "threat_type": "brute_force_attack",
                    "ip_address": "192.168.1.100",
                    "attempts": 50,
                    "automated_response": "block_ip"
                },
                {
                    "threat_type": "sql_injection_attempt",
                    "endpoint": "/api/trading/orders",
                    "payload": "'; DROP TABLE users; --",
                    "automated_response": "block_request_pattern"
                },
                {
                    "threat_type": "unauthorized_admin_access",
                    "user": "compromised_account",
                    "sensitive_action": "user_deletion",
                    "automated_response": "suspend_account"
                }
            ]
            
            # Trigger automated responses
            for threat in critical_threats:
                await alerting_system.handle_critical_threat(threat)
        
        # Verify automated responses were triggered
        assert len(automated_actions) == len(critical_threats), \
            "All critical threats should trigger automated responses"
        
        # Verify appropriate response actions
        response_actions = [action["action"] for action in automated_actions]
        expected_actions = ["block_ip", "block_request_pattern", "suspend_account"]
        
        for expected_action in expected_actions:
            assert expected_action in response_actions, \
                f"Should trigger {expected_action} automated response"

    def test_security_dashboard_should_provide_real_time_visibility(self):
        """Test security dashboard provides real-time visibility - TDD FAIL FIRST"""
        # Mock security dashboard data aggregator
        dashboard = SecurityDashboard()
        
        # Mock real-time security data
        current_security_state = {
            "active_sessions": 25,
            "failed_logins_last_hour": 3,
            "admin_actions_today": 7,
            "suspicious_ips": ["192.168.1.100", "10.0.0.50"],
            "trading_anomalies": 1,
            "system_health": "green",
            "last_security_scan": datetime.now() - timedelta(hours=2)
        }
        
        # Generate dashboard view
        dashboard_data = dashboard.generate_security_overview(current_security_state)
        
        # Verify dashboard provides comprehensive security view
        required_sections = [
            "authentication_status",
            "access_control_summary", 
            "threat_detection_status",
            "system_security_health",
            "recent_security_events"
        ]
        
        for section in required_sections:
            assert section in dashboard_data, \
                f"Security dashboard should include {section} section"
        
        # Verify threat indicators are highlighted
        assert "suspicious_activity_detected" in dashboard_data["threat_detection_status"], \
            "Dashboard should highlight suspicious activity"

    @pytest.mark.asyncio
    async def test_compliance_reporting_should_generate_audit_reports(self):
        """Test compliance reporting generates audit reports - TDD FAIL FIRST"""
        # Mock compliance report generator
        compliance_reporter = ComplianceReporter()
        
        # Mock audit data for compliance period
        audit_period = {
            "start_date": datetime.now() - timedelta(days=90),
            "end_date": datetime.now()
        }
        
        # Generate compliance report
        compliance_report = await compliance_reporter.generate_security_compliance_report(audit_period)
        
        # Verify report contains required compliance elements
        required_elements = [
            "executive_summary",
            "authentication_security",
            "access_control_audit",
            "data_protection_status",
            "incident_response_summary",
            "compliance_violations",
            "recommendations"
        ]
        
        for element in required_elements:
            assert element in compliance_report, \
                f"Compliance report should include {element}"
        
        # Verify report format is suitable for auditors
        assert "compliance_period" in compliance_report, \
            "Report should specify compliance period"
        assert "report_generation_timestamp" in compliance_report, \
            "Report should include generation timestamp"


# Mock classes for testing security monitoring
class SecurityMetricsCollector:
    """Mock security metrics collector"""
    
    def process_security_metrics(self, raw_metrics):
        """Process raw security metrics into analyzed data"""
        total_logins = raw_metrics["failed_login_attempts"] + raw_metrics["successful_logins"]
        login_failure_rate = raw_metrics["failed_login_attempts"] / total_logins if total_logins > 0 else 0
        
        return {
            "login_failure_rate": login_failure_rate,
            "admin_activity_level": raw_metrics["admin_actions"],
            "security_violation_count": raw_metrics["trading_violations"] + raw_metrics["unusual_access_patterns"],
            "total_security_events": sum(raw_metrics.values())
        }


class SecurityDashboard:
    """Mock security dashboard"""
    
    def generate_security_overview(self, security_state):
        """Generate security dashboard overview"""
        return {
            "authentication_status": {
                "active_sessions": security_state["active_sessions"],
                "failed_logins": security_state["failed_logins_last_hour"]
            },
            "access_control_summary": {
                "admin_actions": security_state["admin_actions_today"]
            },
            "threat_detection_status": {
                "suspicious_activity_detected": len(security_state["suspicious_ips"]) > 0,
                "suspicious_ips": security_state["suspicious_ips"]
            },
            "system_security_health": {
                "overall_status": security_state["system_health"],
                "last_scan": security_state["last_security_scan"]
            },
            "recent_security_events": {
                "trading_anomalies": security_state["trading_anomalies"]
            }
        }


class ComplianceReporter:
    """Mock compliance reporter"""
    
    async def generate_security_compliance_report(self, audit_period):
        """Generate security compliance report"""
        return {
            "executive_summary": "Security compliance summary for audit period",
            "compliance_period": {
                "start": audit_period["start_date"].isoformat(),
                "end": audit_period["end_date"].isoformat()
            },
            "authentication_security": {
                "mfa_compliance": "enabled",
                "password_policy": "compliant"
            },
            "access_control_audit": {
                "privilege_reviews": "completed",
                "access_violations": 0
            },
            "data_protection_status": {
                "encryption_status": "compliant",
                "backup_security": "verified"
            },
            "incident_response_summary": {
                "incidents_handled": 2,
                "response_time_avg": "15 minutes"
            },
            "compliance_violations": [],
            "recommendations": [
                "Continue regular security assessments",
                "Update incident response procedures"
            ],
            "report_generation_timestamp": datetime.now().isoformat()
        }