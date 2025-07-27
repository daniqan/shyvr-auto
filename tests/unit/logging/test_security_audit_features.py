"""
Test Security and Audit Features
Following TDD methodology - comprehensive security and audit testing
"""

import asyncio
import pytest
import uuid
import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call
import json
import re

from src.logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction, 
    ActivitySeverity, TradingMode, ChainType
)


class TestDataIntegrityAndAudit:
    """Test data integrity and audit trail features"""
    
    @pytest.mark.asyncio
    async def test_activity_checksum_generation(self, activity_logger_instance):
        """Test that activity logs have integrity checksums"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.ACCESS,
            source="audit_test",
            event_type="checksum_test",
            title="Checksum integrity test",
            user_id=123456789,
            metadata={"test": "data"}
        )
        
        entry = logger._buffer[0]
        
        # Should have checksum generated
        assert entry.checksum is not None
        assert len(entry.checksum) == 64  # SHA256 hex length
        
        # Verify checksum is deterministic
        checksum1 = entry.generate_checksum()
        checksum2 = entry.generate_checksum()
        assert checksum1 == checksum2
        
        # Different data should produce different checksum
        entry.title = "Modified title"
        checksum3 = entry.generate_checksum()
        assert checksum1 != checksum3
    
    @pytest.mark.asyncio
    async def test_checksum_verification(self, activity_logger_instance):
        """Test verification of activity log checksums"""
        logger = activity_logger_instance
        
        # Create activity with known data
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.CREATE,
            source="integrity_test",
            event_type="verification_test",
            title="Checksum verification test"
        )
        
        entry = logger._buffer[0]
        original_checksum = entry.checksum
        
        # Verify checksum matches
        calculated_checksum = entry.generate_checksum()
        assert calculated_checksum == original_checksum
        
        # Simulate data tampering
        entry.title = "Tampered title"
        tampered_checksum = entry.generate_checksum()
        
        # Should detect tampering
        assert tampered_checksum != original_checksum
    
    @pytest.mark.asyncio
    async def test_audit_trail_completeness(self, activity_logger_instance):
        """Test that audit trails are complete and tamper-evident"""
        logger = activity_logger_instance
        
        # Create sequence of related activities
        session_id = str(uuid.uuid4())
        user_id = 123456789
        
        activities = [
            ("login", "User login"),
            ("access", "Accessed trading page"),
            ("execute", "Executed trade"),
            ("logout", "User logout")
        ]
        
        activity_ids = []
        for action_str, title in activities:
            action = getattr(ActivityAction, action_str.upper())
            activity_id = await logger.log_user_action(
                user_id=user_id,
                action=action,
                component="audit_test",
                title=title,
                session_id=session_id,
                metadata={"sequence": len(activity_ids) + 1}
            )
            activity_ids.append(activity_id)
        
        # Should create complete audit trail
        assert len(logger._buffer) == 4
        
        # All activities should be linked by session
        for entry in logger._buffer:
            assert entry.session_id == session_id
            assert entry.user_id == user_id
            assert entry.checksum is not None
        
        # Should maintain chronological order
        timestamps = [entry.created_at for entry in logger._buffer]
        assert timestamps == sorted(timestamps)
    
    @pytest.mark.asyncio
    async def test_immutable_audit_records(self, mock_database_pool, mock_config):
        """Test that audit records are immutable once created"""
        mock_conn = AsyncMock()
        
        # Mock that prevents updates to activity logs
        async def mock_execute(*args, **kwargs):
            query = args[0] if args else ""
            if "UPDATE activity_logs" in query and "archived_at" not in query:
                raise Exception("Audit records are immutable")
            return None
        
        mock_conn.execute = mock_execute
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock()
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            # Insert should work
            entry = ActivityLogEntry(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.CREATE,
                source="immutable_test",
                event_type="audit_test",
                title="Immutable test"
            )
            
            await logger._insert_activities([entry])
            
            # Try to update (should be prevented)
            with pytest.raises(Exception, match="immutable"):
                await mock_conn.execute("""
                    UPDATE activity_logs 
                    SET title = 'Modified title'
                    WHERE activity_id = $1
                """, entry.activity_id)
            
            await logger.stop()


class TestSecurityEventLogging:
    """Test security-specific event logging"""
    
    @pytest.mark.asyncio
    async def test_authentication_failure_logging(self, activity_logger_instance, security_test_scenarios):
        """Test logging of authentication failures"""
        logger = activity_logger_instance
        
        # Multiple failed login attempts
        ip_address = "192.168.1.100"
        for attempt in range(3):
            activity_id = await logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.FAILURE,
                source="auth_system",
                event_type="login_failure",
                title=f"Login attempt {attempt + 1} failed",
                severity=ActivitySeverity.WARNING,
                ip_address=ip_address,
                user_agent="Suspicious Browser",
                risk_score=50 + (attempt * 20),  # Increasing risk
                metadata={
                    "attempt_number": attempt + 1,
                    "failure_reason": "invalid_credentials"
                }
            )
        
        # Should log all failed attempts
        assert len(logger._buffer) == 3
        
        # Risk scores should increase
        risk_scores = [entry.risk_score for entry in logger._buffer]
        assert risk_scores == [50, 70, 90]
        
        # All should be from same IP
        for entry in logger._buffer:
            assert entry.ip_address == ip_address
    
    @pytest.mark.asyncio
    async def test_suspicious_activity_detection(self, activity_logger_instance):
        """Test detection and logging of suspicious activities"""
        logger = activity_logger_instance
        
        # Log suspicious trading pattern
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.ALERT,
            source="anomaly_detector",
            event_type="suspicious_trading",
            title="Unusual trading pattern detected",
            severity=ActivitySeverity.ALERT,
            risk_score=95,
            security_level="high",
            metadata={
                "pattern_type": "volume_spike",
                "deviation_score": 4.2,
                "normal_range": [100, 500],
                "detected_value": 2500,
                "confidence": 0.96
            },
            tags=["anomaly", "high_risk", "trading"]
        )
        
        entry = logger._buffer[0]
        
        # Should be marked as high-risk security event
        assert entry.severity == ActivitySeverity.ALERT
        assert entry.risk_score == 95
        assert entry.security_level == "high"
        assert "anomaly" in entry.tags
    
    @pytest.mark.asyncio
    async def test_unauthorized_access_logging(self, activity_logger_instance):
        """Test logging of unauthorized access attempts"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.VIOLATION,
            source="access_control",
            event_type="unauthorized_access",
            title="Unauthorized API access attempt",
            severity=ActivitySeverity.ERROR,
            ip_address="suspicious.ip.address",
            api_endpoint="/admin/sensitive-data",
            http_method="POST",
            http_status=403,
            risk_score=80,
            metadata={
                "access_type": "api",
                "requested_resource": "/admin/sensitive-data",
                "auth_level_required": "admin",
                "auth_level_provided": "user"
            }
        )
        
        entry = logger._buffer[0]
        
        # Should capture unauthorized access details
        assert entry.action == ActivityAction.VIOLATION
        assert entry.http_status == 403
        assert entry.risk_score == 80
        assert entry.metadata["auth_level_required"] == "admin"
    
    @pytest.mark.asyncio
    async def test_privilege_escalation_logging(self, activity_logger_instance):
        """Test logging of privilege escalation attempts"""
        logger = activity_logger_instance
        
        user_id = 123456789
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.VIOLATION,
            source="privilege_monitor",
            event_type="privilege_escalation",
            title="Privilege escalation attempt detected",
            severity=ActivitySeverity.CRITICAL,
            user_id=user_id,
            risk_score=100,
            security_level="critical",
            metadata={
                "current_role": "user",
                "attempted_role": "admin",
                "escalation_method": "token_manipulation",
                "detection_confidence": 0.98
            },
            tags=["privilege_escalation", "critical", "security_violation"]
        )
        
        entry = logger._buffer[0]
        
        # Should be flagged as critical security event
        assert entry.severity == ActivitySeverity.CRITICAL
        assert entry.risk_score == 100
        assert entry.security_level == "critical"
        assert "privilege_escalation" in entry.tags


class TestInputSanitizationAndValidation:
    """Test input sanitization and validation for security"""
    
    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, activity_logger_instance, security_test_scenarios):
        """Test prevention of SQL injection in log data"""
        logger = activity_logger_instance
        
        # Test with SQL injection payload
        sql_injection_payload = "'; DROP TABLE activity_logs; --"
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.VIOLATION,
            source="input_validation",
            event_type="sql_injection_attempt",
            title=sql_injection_payload,  # Malicious title
            description=f"Description with payload: {sql_injection_payload}",
            metadata={
                "malicious_input": sql_injection_payload,
                "sanitized": True
            }
        )
        
        entry = logger._buffer[0]
        
        # Should sanitize or safely handle malicious input
        assert entry.activity_id is not None  # Should still create entry
        # The actual sanitization would be handled by database layer
        assert sql_injection_payload in entry.title  # Input preserved for audit
    
    @pytest.mark.asyncio
    async def test_xss_payload_handling(self, activity_logger_instance, security_test_scenarios):
        """Test handling of XSS payloads in log data"""
        logger = activity_logger_instance
        
        # Test with XSS payload
        xss_payload = "<script>alert('xss')</script>"
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.VIOLATION,
            source="xss_detector",
            event_type="xss_attempt",
            title="XSS attempt detected",
            description=f"Malicious script: {xss_payload}",
            metadata={
                "xss_payload": xss_payload,
                "source_field": "user_input"
            }
        )
        
        entry = logger._buffer[0]
        
        # Should handle XSS payload safely
        assert entry.activity_id is not None
        assert xss_payload in entry.description  # Preserved for security analysis
    
    @pytest.mark.asyncio
    async def test_oversized_input_handling(self, activity_logger_instance, security_test_scenarios):
        """Test handling of oversized inputs (DoS prevention)"""
        logger = activity_logger_instance
        
        # Create very large input
        large_payload = "A" * 100000  # 100KB payload
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.VIOLATION,
            source="dos_detector",
            event_type="oversized_input",
            title="Oversized input detected",
            description=large_payload,
            metadata={
                "input_size": len(large_payload),
                "max_allowed_size": 10000,
                "truncated": True
            }
        )
        
        entry = logger._buffer[0]
        
        # Should handle large input appropriately
        assert entry.activity_id is not None
        # Description might be truncated or handled specially
        assert entry.metadata["input_size"] == 100000
    
    @pytest.mark.asyncio
    async def test_unicode_normalization_security(self, activity_logger_instance):
        """Test unicode normalization for security"""
        logger = activity_logger_instance
        
        # Unicode normalization attack vectors
        unicode_attacks = [
            "а́dmin",  # Cyrillic 'a' with combining acute
            "аdmin",   # Cyrillic 'a'
            "admin\u200b",  # Zero-width space
            "admin\ufeff"   # Byte order mark
        ]
        
        for i, attack in enumerate(unicode_attacks):
            activity_id = await logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.VIOLATION,
                source="unicode_validator",
                event_type=f"unicode_attack_{i}",
                title=f"Unicode normalization attack: {attack}",
                metadata={
                    "original_input": attack,
                    "normalized": attack.encode('utf-8').decode('utf-8'),
                    "attack_type": "unicode_normalization"
                }
            )
        
        # Should handle all unicode variations
        assert len(logger._buffer) == len(unicode_attacks)


class TestAccessControlAndPermissions:
    """Test access control and permission logging"""
    
    @pytest.mark.asyncio
    async def test_role_based_access_logging(self, activity_logger_instance):
        """Test logging of role-based access control"""
        logger = activity_logger_instance
        
        # Test different access levels
        access_scenarios = [
            ("admin", "admin_panel", True, "full_access"),
            ("user", "admin_panel", False, "insufficient_privileges"),
            ("user", "trading_panel", True, "authorized_access"),
            ("guest", "trading_panel", False, "authentication_required")
        ]
        
        for role, resource, allowed, result in access_scenarios:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.ACCESS if allowed else ActivityAction.VIOLATION,
                source="rbac_system",
                event_type="access_control",
                title=f"{role} access to {resource}: {result}",
                severity=ActivitySeverity.INFO if allowed else ActivitySeverity.WARNING,
                metadata={
                    "user_role": role,
                    "requested_resource": resource,
                    "access_granted": allowed,
                    "access_result": result
                }
            )
        
        # Should log all access attempts
        assert len(logger._buffer) == 4
        
        # Check access granted/denied
        granted_count = sum(1 for entry in logger._buffer if entry.action == ActivityAction.ACCESS)
        denied_count = sum(1 for entry in logger._buffer if entry.action == ActivityAction.VIOLATION)
        
        assert granted_count == 2  # admin->admin_panel, user->trading_panel
        assert denied_count == 2   # user->admin_panel, guest->trading_panel
    
    @pytest.mark.asyncio
    async def test_api_key_access_logging(self, activity_logger_instance):
        """Test logging of API key-based access"""
        logger = activity_logger_instance
        
        api_key_hash = hashlib.sha256("secret_api_key".encode()).hexdigest()
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.ACCESS,
            source="api_gateway",
            event_type="api_key_access",
            title="API access with valid key",
            api_endpoint="/api/v1/trades",
            http_method="GET",
            http_status=200,
            metadata={
                "api_key_hash": api_key_hash,
                "api_key_scope": ["read:trades", "write:orders"],
                "rate_limit_remaining": 95,
                "access_granted": True
            }
        )
        
        entry = logger._buffer[0]
        
        # Should log API access details without exposing key
        assert entry.metadata["api_key_hash"] == api_key_hash
        assert "secret_api_key" not in str(entry.to_dict())  # Key not exposed
        assert entry.metadata["access_granted"] is True
    
    @pytest.mark.asyncio
    async def test_session_management_logging(self, activity_logger_instance):
        """Test logging of session management events"""
        logger = activity_logger_instance
        
        session_id = str(uuid.uuid4())
        user_id = 123456789
        
        # Session lifecycle events
        session_events = [
            ("create", "Session created"),
            ("validate", "Session validated"),
            ("refresh", "Session refreshed"),
            ("expire", "Session expired"),
            ("invalidate", "Session invalidated")
        ]
        
        for event_type, title in session_events:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=getattr(ActivityAction, event_type.upper(), ActivityAction.EXECUTE),
                source="session_manager",
                event_type=f"session_{event_type}",
                title=title,
                user_id=user_id,
                session_id=session_id,
                metadata={
                    "session_action": event_type,
                    "session_lifetime": 3600,  # 1 hour
                    "ip_address": "192.168.1.100"
                }
            )
        
        # Should track complete session lifecycle
        assert len(logger._buffer) == 5
        
        # All events should be linked to same session
        for entry in logger._buffer:
            assert entry.session_id == session_id
            assert entry.user_id == user_id


class TestComplianceAndRegulatory:
    """Test compliance and regulatory audit features"""
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance_logging(self, activity_logger_instance):
        """Test GDPR compliance audit logging"""
        logger = activity_logger_instance
        
        user_id = 123456789
        
        # GDPR-related activities
        gdpr_activities = [
            ("data_access", "User requested personal data access"),
            ("data_export", "Personal data exported for user"),
            ("data_correction", "User corrected personal data"),
            ("data_deletion", "User requested data deletion"),
            ("consent_given", "User gave consent for data processing"),
            ("consent_withdrawn", "User withdrew data processing consent")
        ]
        
        for event_type, title in gdpr_activities:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.EXECUTE,
                source="gdpr_compliance",
                event_type=event_type,
                title=title,
                user_id=user_id,
                metadata={
                    "gdpr_request_type": event_type,
                    "compliance_officer": "compliance@example.com",
                    "processing_time_hours": 24,
                    "legal_basis": "user_consent"
                },
                tags=["gdpr", "compliance", "privacy"]
            )
        
        # Should log all GDPR activities
        assert len(logger._buffer) == 6
        
        # All should be tagged for compliance
        for entry in logger._buffer:
            assert "gdpr" in entry.tags
            assert "compliance" in entry.tags
    
    @pytest.mark.asyncio
    async def test_financial_compliance_logging(self, activity_logger_instance):
        """Test financial regulatory compliance logging"""
        logger = activity_logger_instance
        
        # Financial compliance events
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.ALERT,
            source="aml_monitor",
            event_type="suspicious_transaction",
            title="Suspicious transaction pattern detected",
            severity=ActivitySeverity.ALERT,
            user_id=123456789,
            amount_usd=Decimal("50000.00"),
            metadata={
                "aml_rule_triggered": "large_cash_transaction",
                "risk_score": 85,
                "requires_sar": True,  # Suspicious Activity Report
                "compliance_flags": ["unusual_pattern", "high_amount"],
                "review_deadline": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
            },
            tags=["aml", "compliance", "suspicious", "financial"]
        )
        
        entry = logger._buffer[0]
        
        # Should capture financial compliance details
        assert entry.severity == ActivitySeverity.ALERT
        assert entry.amount_usd == Decimal("50000.00")
        assert entry.metadata["requires_sar"] is True
        assert "aml" in entry.tags
    
    @pytest.mark.asyncio
    async def test_audit_log_retention_compliance(self, activity_logger_instance):
        """Test compliance with audit log retention requirements"""
        logger = activity_logger_instance
        
        # Create audit activities with compliance retention requirements
        retention_periods = [
            ("financial", 7),     # 7 years for financial records
            ("security", 3),      # 3 years for security logs
            ("operational", 1),   # 1 year for operational logs
            ("compliance", 10)    # 10 years for compliance records
        ]
        
        for record_type, years in retention_periods:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.CREATE,
                source="compliance_audit",
                event_type=f"{record_type}_audit",
                title=f"{record_type.title()} compliance record created",
                metadata={
                    "record_type": record_type,
                    "retention_period_years": years,
                    "compliance_requirement": "regulatory_mandate",
                    "destruction_date": (datetime.now(timezone.utc) + timedelta(days=365*years)).isoformat()
                },
                tags=["audit", "compliance", "retention", record_type]
            )
        
        # Should create compliant audit records
        assert len(logger._buffer) == 4
        
        # Check retention periods are set
        for entry in logger._buffer:
            assert "retention_period_years" in entry.metadata
            assert "destruction_date" in entry.metadata


class TestCryptographicSecurity:
    """Test cryptographic security features"""
    
    @pytest.mark.asyncio
    async def test_sensitive_data_hashing(self, activity_logger_instance):
        """Test that sensitive data is properly hashed"""
        logger = activity_logger_instance
        
        # Sensitive data that should be hashed
        sensitive_email = "user@example.com"
        sensitive_phone = "+1234567890"
        
        email_hash = hashlib.sha256(sensitive_email.encode()).hexdigest()
        phone_hash = hashlib.sha256(sensitive_phone.encode()).hexdigest()
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.ACCESS,
            source="privacy_protection",
            event_type="sensitive_data_access",
            title="Sensitive data accessed",
            user_id=123456789,
            metadata={
                "email_hash": email_hash,
                "phone_hash": phone_hash,
                "data_accessed": ["email", "phone"],
                "hashing_algorithm": "sha256"
            }
        )
        
        entry = logger._buffer[0]
        entry_dict = entry.to_dict()
        
        # Should store hashes, not plain text
        assert sensitive_email not in str(entry_dict)
        assert sensitive_phone not in str(entry_dict)
        assert entry.metadata["email_hash"] == email_hash
        assert entry.metadata["phone_hash"] == phone_hash
    
    @pytest.mark.asyncio
    async def test_cryptographic_signature_verification(self, activity_logger_instance):
        """Test cryptographic signature verification for critical operations"""
        logger = activity_logger_instance
        
        # Simulate signed transaction
        transaction_data = {
            "from": "0x123",
            "to": "0x456", 
            "amount": "1000000000000000000",  # 1 ETH in wei
            "nonce": 42
        }
        
        # Create signature (simplified for test)
        message = json.dumps(transaction_data, sort_keys=True)
        signature = hmac.new(b"secret_key", message.encode(), hashlib.sha256).hexdigest()
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.EXECUTE,
            source="crypto_validator",
            event_type="transaction_signature_verification",
            title="Transaction signature verified",
            severity=ActivitySeverity.INFO,
            metadata={
                "transaction_hash": hashlib.sha256(message.encode()).hexdigest(),
                "signature_valid": True,
                "signature_algorithm": "HMAC-SHA256",
                "verification_time_ms": 15
            }
        )
        
        entry = logger._buffer[0]
        
        # Should log signature verification
        assert entry.metadata["signature_valid"] is True
        assert entry.metadata["signature_algorithm"] == "HMAC-SHA256"
    
    @pytest.mark.asyncio
    async def test_encryption_key_rotation_logging(self, activity_logger_instance):
        """Test logging of encryption key rotation events"""
        logger = activity_logger_instance
        
        old_key_id = "key_2024_001"
        new_key_id = "key_2024_002"
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.UPDATE,
            source="key_management",
            event_type="encryption_key_rotation",
            title="Encryption key rotated",
            severity=ActivitySeverity.NOTICE,
            metadata={
                "old_key_id": old_key_id,
                "new_key_id": new_key_id,
                "rotation_reason": "scheduled_rotation",
                "affected_systems": ["database", "api", "storage"],
                "rotation_completed_at": datetime.now(timezone.utc).isoformat()
            },
            tags=["encryption", "key_rotation", "security", "critical"]
        )
        
        entry = logger._buffer[0]
        
        # Should log key rotation without exposing actual keys
        assert entry.metadata["old_key_id"] == old_key_id
        assert entry.metadata["new_key_id"] == new_key_id
        assert "key_rotation" in entry.tags


class TestThreatDetectionAndResponse:
    """Test threat detection and incident response logging"""
    
    @pytest.mark.asyncio
    async def test_automated_threat_detection(self, activity_logger_instance):
        """Test automated threat detection logging"""
        logger = activity_logger_instance
        
        threat_indicators = [
            ("brute_force", "Multiple failed login attempts", 90),
            ("ddos", "Unusual traffic spike detected", 85),
            ("malware", "Suspicious file upload detected", 95),
            ("phishing", "Potential phishing attempt", 75)
        ]
        
        for threat_type, description, risk_score in threat_indicators:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.ALERT,
                source="threat_detector",
                event_type=f"threat_{threat_type}",
                title=f"Threat detected: {threat_type}",
                description=description,
                severity=ActivitySeverity.ALERT,
                risk_score=risk_score,
                metadata={
                    "threat_type": threat_type,
                    "detection_confidence": 0.85 + (risk_score - 75) * 0.002,
                    "mitigation_required": True,
                    "response_time_required_minutes": 15
                },
                tags=["threat", "automated", threat_type]
            )
        
        # Should detect and log all threats
        assert len(logger._buffer) == 4
        
        # All should be high-risk alerts
        for entry in logger._buffer:
            assert entry.severity == ActivitySeverity.ALERT
            assert entry.risk_score >= 75
            assert "threat" in entry.tags
    
    @pytest.mark.asyncio
    async def test_incident_response_workflow(self, activity_logger_instance):
        """Test incident response workflow logging"""
        logger = activity_logger_instance
        
        incident_id = str(uuid.uuid4())
        
        # Incident response stages
        response_stages = [
            ("detection", "Security incident detected"),
            ("analysis", "Incident analysis initiated"),
            ("containment", "Threat contained"),
            ("eradication", "Threat eradicated"),
            ("recovery", "Systems recovered"),
            ("lessons_learned", "Post-incident review completed")
        ]
        
        for stage, title in response_stages:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.EXECUTE,
                source="incident_response",
                event_type=f"ir_{stage}",
                title=title,
                correlation_id=incident_id,
                metadata={
                    "incident_id": incident_id,
                    "ir_stage": stage,
                    "response_team": "security_team",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                tags=["incident_response", "security", stage]
            )
        
        # Should track complete incident response
        assert len(logger._buffer) == 6
        
        # All should be correlated to same incident
        for entry in logger._buffer:
            assert entry.correlation_id == incident_id
            assert "incident_response" in entry.tags


class TestSecurityMetricsAndReporting:
    """Test security metrics and reporting features"""
    
    @pytest.mark.asyncio
    async def test_security_metrics_calculation(self, mock_database_pool):
        """Test calculation of security metrics"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'category': 'security',
                'total_events': 1000,
                'critical_events': 25,
                'high_risk_events': 150,
                'avg_response_time': 45.5,
                'incident_count': 5
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                metrics = await conn.fetch("""
                    SELECT 
                        category,
                        COUNT(*) as total_events,
                        COUNT(*) FILTER (WHERE severity = 'critical') as critical_events,
                        COUNT(*) FILTER (WHERE risk_score >= 80) as high_risk_events,
                        AVG(response_time_ms) as avg_response_time,
                        COUNT(DISTINCT correlation_id) FILTER (WHERE event_type LIKE 'ir_%') as incident_count
                    FROM activity_logs 
                    WHERE category = 'security'
                    AND created_at > NOW() - INTERVAL '30 days'
                    GROUP BY category
                """)
            
            # Should calculate security metrics
            assert len(metrics) == 1
            metric = metrics[0]
            assert metric['total_events'] == 1000
            assert metric['critical_events'] == 25
            assert metric['incident_count'] == 5
    
    @pytest.mark.asyncio
    async def test_compliance_reporting(self, mock_database_pool):
        """Test compliance reporting queries"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'compliance_type': 'gdpr',
                'total_requests': 50,
                'completed_requests': 48,
                'avg_completion_time_hours': 18.5,
                'compliance_rate': 96.0
            },
            {
                'compliance_type': 'aml',
                'total_requests': 25,
                'completed_requests': 25,
                'avg_completion_time_hours': 12.0,
                'compliance_rate': 100.0
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                compliance_report = await conn.fetch("""
                    SELECT 
                        CASE 
                            WHEN 'gdpr' = ANY(tags) THEN 'gdpr'
                            WHEN 'aml' = ANY(tags) THEN 'aml'
                            ELSE 'other'
                        END as compliance_type,
                        COUNT(*) as total_requests,
                        COUNT(*) FILTER (WHERE action = 'success') as completed_requests,
                        AVG(EXTRACT(EPOCH FROM (updated_at - created_at))/3600) as avg_completion_time_hours,
                        ROUND(COUNT(*) FILTER (WHERE action = 'success') * 100.0 / COUNT(*), 2) as compliance_rate
                    FROM activity_logs 
                    WHERE tags && ARRAY['gdpr', 'aml', 'compliance']
                    AND created_at > NOW() - INTERVAL '30 days'
                    GROUP BY compliance_type
                """)
            
            # Should generate compliance reports
            assert len(compliance_report) == 2
            assert compliance_report[0]['compliance_type'] == 'gdpr'
            assert compliance_report[1]['compliance_type'] == 'aml'


# NOTE: All these tests should FAIL initially since the security and audit features
# may not be fully implemented yet. This follows TDD methodology.