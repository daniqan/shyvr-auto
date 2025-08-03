"""
Phase 7.3: API Security & Input Validation Tests
Following TDD methodology - these tests validate API endpoint security
"""

import pytest
import json
import re
from unittest.mock import patch, MagicMock
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
from datetime import datetime
import html

# Mock imports to avoid dependency issues during testing
try:
    from src.dashboard.api import app
    from src.dashboard.auth import dashboard_auth
except ImportError:
    # Create mock objects for testing
    app = None
    dashboard_auth = None


class TestAPIInputValidation:
    """Test API input validation security following TDD methodology"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        if app is None:
            # Create mock FastAPI app for testing
            from fastapi import FastAPI
            mock_app = FastAPI()
            return TestClient(mock_app)
        return TestClient(app)
    
    @pytest.fixture 
    def auth_headers(self):
        """Create authentication headers for testing"""
        # Create test user and API key
        if dashboard_auth is None:
            # Return mock headers for testing
            return {"Authorization": "Bearer mock_api_key_for_testing"}
        api_key = dashboard_auth.create_user("testuser", "password", {"read", "write"})
        return {"Authorization": f"Bearer {api_key}"}

    def test_sql_injection_prevention_in_api_endpoints_should_fail_injection(self, client, auth_headers):
        """Test SQL injection prevention in API endpoints - TDD FAIL FIRST"""
        # Common SQL injection payloads
        sql_injections = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --", 
            "'; INSERT INTO users VALUES ('hacker', 'pass'); --",
            "admin'--",
            "' OR 1=1 #",
            "'; EXEC xp_cmdshell('dir'); --"
        ]
        
        for injection in sql_injections:
            # Test portfolio endpoint with SQL injection
            response = client.get(
                f"/api/portfolio/positions?symbol={injection}",
                headers=auth_headers
            )
            
            # Should not execute SQL injection
            assert response.status_code != 200 or "users" not in response.text.lower(), \
                f"SQL injection may have succeeded: {injection}"
            
            # Test trading signals endpoint
            response = client.get(
                f"/api/trading/signals?asset={injection}",
                headers=auth_headers
            )
            
            # Should return error or sanitized data, not execute injection
            assert response.status_code in [400, 422, 404] or "DROP" not in response.text, \
                f"SQL injection in trading signals: {injection}"

    def test_xss_prevention_in_api_responses_should_sanitize_output(self, client, auth_headers):
        """Test XSS prevention in API responses - TDD FAIL FIRST"""
        # XSS payloads 
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "';window.location='http://attacker.com';//"
        ]
        
        for payload in xss_payloads:
            # Test user creation with XSS payload
            response = client.post(
                "/api/admin/users",
                headers=auth_headers,
                json={
                    "username": payload,
                    "permissions": ["read"]
                }
            )
            
            # Response should not contain unescaped script tags
            if response.status_code == 200:
                response_text = response.text
                assert "<script" not in response_text, f"XSS payload not sanitized: {payload}"
                assert "javascript:" not in response_text, f"JavaScript URL not sanitized: {payload}"
                assert payload not in response_text or html.escape(payload) in response_text, \
                    f"XSS payload not properly escaped: {payload}"

    def test_input_length_validation_should_reject_oversized_inputs(self, client, auth_headers):
        """Test input length validation rejects oversized inputs - TDD FAIL FIRST"""
        # Test with extremely long strings
        long_string = "A" * 10000  # 10KB string
        very_long_string = "B" * 100000  # 100KB string
        
        # Test username length validation
        response = client.post(
            "/api/admin/users",
            headers=auth_headers,
            json={
                "username": long_string,
                "permissions": ["read"]
            }
        )
        
        # Should reject oversized username
        assert response.status_code in [400, 422], "Should reject oversized username"
        
        # Test trading signal with oversized data
        response = client.post(
            "/api/trading/manual_signal",
            headers=auth_headers,
            json={
                "symbol": "BTC/USD",
                "signal": "BUY",
                "notes": very_long_string
            }
        )
        
        # Should reject oversized notes
        assert response.status_code in [400, 422], "Should reject oversized notes field"

    def test_numeric_input_validation_should_prevent_overflow(self, client, auth_headers):
        """Test numeric input validation prevents overflow attacks - TDD FAIL FIRST"""
        # Test with extreme numeric values
        extreme_values = [
            9999999999999999999999999999999,  # Very large number
            -9999999999999999999999999999999, # Very large negative
            float('inf'),  # Infinity
            float('-inf'), # Negative infinity
            float('nan')   # Not a number
        ]
        
        for value in extreme_values:
            # Test portfolio position amount
            response = client.post(
                "/api/portfolio/positions",
                headers=auth_headers,
                json={
                    "symbol": "BTC/USD",
                    "amount": value,
                    "entry_price": 50000
                }
            )
            
            # Should reject extreme values
            assert response.status_code in [400, 422], f"Should reject extreme value: {value}"

    def test_json_structure_validation_should_reject_malformed_data(self, client, auth_headers):
        """Test JSON structure validation rejects malformed data - TDD FAIL FIRST"""
        # Test malformed JSON structures
        malformed_payloads = [
            '{"unclosed": "json"',  # Unclosed JSON
            '{"duplicate": 1, "duplicate": 2}',  # Duplicate keys
            '{"circular": {"ref": "circular"}}',  # Circular reference attempt
            '{"nested": {"very": {"deeply": {"nested": {"object": {"should": {"be": {"rejected": true}}}}}}}}'  # Deep nesting
        ]
        
        for payload in malformed_payloads:
            # Send raw malformed JSON
            response = client.post(
                "/api/trading/signals",
                headers={**auth_headers, "Content-Type": "application/json"},
                data=payload
            )
            
            # Should reject malformed JSON
            assert response.status_code in [400, 422], f"Should reject malformed JSON: {payload[:50]}..."

    def test_file_upload_validation_should_prevent_malicious_files(self, client, auth_headers):
        """Test file upload validation prevents malicious files - TDD FAIL FIRST"""
        # Test malicious file uploads if file upload endpoints exist
        malicious_files = [
            ("script.js", "text/javascript", "alert('XSS')"),
            ("malware.exe", "application/octet-stream", b"\x4d\x5a\x90\x00"),  # PE header
            ("shell.php", "text/php", "<?php system($_GET['cmd']); ?>"),
            ("huge.txt", "text/plain", "A" * 10000000),  # 10MB file
        ]
        
        for filename, content_type, content in malicious_files:
            # Test configuration file upload if endpoint exists
            response = client.post(
                "/api/admin/upload_config",
                headers=auth_headers,
                files={"file": (filename, content, content_type)}
            )
            
            # Should reject malicious files (404 if endpoint doesn't exist is also valid)
            assert response.status_code in [400, 403, 404, 422], \
                f"Should reject malicious file: {filename}"


class TestAPIAuthenticationSecurity:
    """Test API authentication security mechanisms"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        return TestClient(app)
    
    def test_unauthenticated_requests_should_be_rejected(self, client):
        """Test that unauthenticated requests are rejected - TDD FAIL FIRST"""
        protected_endpoints = [
            "/api/portfolio/positions",
            "/api/trading/signals", 
            "/api/admin/users",
            "/api/trading/manual_signal",
            "/api/portfolio/performance"
        ]
        
        for endpoint in protected_endpoints:
            # Test GET without authentication
            response = client.get(endpoint)
            assert response.status_code == 401, f"GET {endpoint} should require authentication"
            
            # Test POST without authentication
            response = client.post(endpoint, json={})
            assert response.status_code == 401, f"POST {endpoint} should require authentication"

    def test_invalid_bearer_token_should_be_rejected(self, client):
        """Test that invalid bearer tokens are rejected - TDD FAIL FIRST"""
        invalid_tokens = [
            "invalid-token",
            "Bearer invalid-token",
            "malformed.jwt.token",
            "",
            "null",
            "undefined"
        ]
        
        for token in invalid_tokens:
            headers = {"Authorization": f"Bearer {token}"}
            
            response = client.get("/api/portfolio/positions", headers=headers)
            assert response.status_code == 401, f"Invalid token should be rejected: {token}"

    def test_authorization_header_parsing_security(self, client):
        """Test authorization header parsing security - TDD FAIL FIRST"""
        malicious_headers = [
            "Bearer " + "A" * 10000,  # Extremely long token
            "Basic " + "malicious-basic-auth",  # Wrong auth type
            "Bearer token1 token2 token3",  # Multiple tokens
            "Bearer \n\r\t token",  # Tokens with control characters
            "Bearer token; DROP TABLE users;",  # SQL injection in header
        ]
        
        for header in malicious_headers:
            headers = {"Authorization": header}
            
            response = client.get("/api/portfolio/positions", headers=headers)
            assert response.status_code in [400, 401], f"Malicious auth header should be rejected: {header[:50]}..."


class TestAPIRateLimiting:
    """Test API rate limiting security"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers for testing"""
        api_key = dashboard_auth.create_user("ratelimituser", "password", {"read", "write"})
        return {"Authorization": f"Bearer {api_key}"}

    def test_rate_limiting_blocks_excessive_requests(self, client, auth_headers):
        """Test that rate limiting blocks excessive requests - TDD FAIL FIRST"""
        # Make many rapid requests to trigger rate limiting
        responses = []
        for i in range(150):  # Exceed typical rate limits
            response = client.get("/api/portfolio/positions", headers=auth_headers)
            responses.append(response.status_code)
            
            # Stop early if rate limited
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        assert 429 in responses, "Rate limiting should block excessive requests"

    def test_rate_limiting_per_user_isolation(self, client):
        """Test that rate limiting is properly isolated per user - TDD FAIL FIRST"""
        # Create two different users
        api_key1 = dashboard_auth.create_user("user1", "password", {"read"})
        api_key2 = dashboard_auth.create_user("user2", "password", {"read"})
        
        headers1 = {"Authorization": f"Bearer {api_key1}"}
        headers2 = {"Authorization": f"Bearer {api_key2}"}
        
        # Exhaust rate limit for user1
        for i in range(100):
            response = client.get("/api/portfolio/positions", headers=headers1)
            if response.status_code == 429:
                break
        
        # User2 should still be able to make requests
        response = client.get("/api/portfolio/positions", headers=headers2)
        assert response.status_code != 429, "Rate limiting should be isolated per user"


class TestAPIErrorHandling:
    """Test API error handling security"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers for testing"""
        api_key = dashboard_auth.create_user("erroruser", "password", {"read", "write"})
        return {"Authorization": f"Bearer {api_key}"}

    def test_error_responses_do_not_leak_sensitive_information(self, client, auth_headers):
        """Test that error responses don't leak sensitive info - TDD FAIL FIRST"""
        # Trigger various error conditions
        error_endpoints = [
            "/api/nonexistent/endpoint",
            "/api/portfolio/positions/999999",  # Non-existent position
            "/api/trading/signals/invalid_id",
        ]
        
        for endpoint in error_endpoints:
            response = client.get(endpoint, headers=auth_headers)
            
            if response.status_code in [400, 404, 500]:
                error_text = response.text.lower()
                
                # Should not leak sensitive information
                sensitive_patterns = [
                    "password",
                    "secret",
                    "token",
                    "database",
                    "connection",
                    "api_key",
                    "private_key",
                    "/usr/",
                    "/home/",
                    "traceback",
                    "stack trace"
                ]
                
                for pattern in sensitive_patterns:
                    assert pattern not in error_text, \
                        f"Error response leaks sensitive info '{pattern}' in {endpoint}"

    def test_exception_handling_prevents_information_disclosure(self, client, auth_headers):
        """Test exception handling prevents information disclosure - TDD FAIL FIRST"""
        # Send malformed requests to trigger exceptions
        malformed_requests = [
            {"endpoint": "/api/portfolio/positions", "json": {"invalid": "json structure"}},
            {"endpoint": "/api/trading/signals", "json": {"missing": "required fields"}},
        ]
        
        for request in malformed_requests:
            response = client.post(
                request["endpoint"],
                headers=auth_headers,
                json=request["json"]
            )
            
            # Should handle exceptions gracefully
            assert response.status_code in [400, 422], \
                f"Should handle malformed request gracefully: {request['endpoint']}"
            
            # Should not expose internal details
            response_text = response.text.lower()
            internal_details = ["traceback", "file", "line", "function", "module"]
            
            for detail in internal_details:
                assert detail not in response_text, \
                    f"Exception response exposes internal details: {detail}"


class TestAPICORSSecurity:
    """Test API CORS security configuration"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        return TestClient(app)

    def test_cors_configuration_restricts_origins(self, client):
        """Test CORS configuration restricts allowed origins - TDD FAIL FIRST"""
        # Test CORS with potentially malicious origins
        malicious_origins = [
            "http://evil.com",
            "https://attacker.site", 
            "null",
            "*",
            "data:text/html,<script>alert('XSS')</script>"
        ]
        
        for origin in malicious_origins:
            headers = {
                "Origin": origin,
                "Access-Control-Request-Method": "GET"
            }
            
            # Send preflight request
            response = client.options("/api/portfolio/positions", headers=headers)
            
            # Should not allow arbitrary origins
            cors_header = response.headers.get("Access-Control-Allow-Origin", "")
            if cors_header:
                assert cors_header != "*", "CORS should not allow all origins (*)"
                assert origin not in cors_header or origin in ["http://localhost", "https://localhost"], \
                    f"CORS should not allow malicious origin: {origin}"

    def test_cors_credentials_handling_security(self, client):
        """Test CORS credentials handling security - TDD FAIL FIRST"""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization"
        }
        
        response = client.options("/api/portfolio/positions", headers=headers)
        
        # If credentials are allowed, origin should not be *
        allow_credentials = response.headers.get("Access-Control-Allow-Credentials", "").lower()
        allow_origin = response.headers.get("Access-Control-Allow-Origin", "")
        
        if allow_credentials == "true":
            assert allow_origin != "*", \
                "CORS should not allow credentials with wildcard origin"


class TestAPIContentSecurityPolicy:
    """Test API Content Security Policy headers"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        return TestClient(app)

    def test_security_headers_are_present(self, client):
        """Test that security headers are present in API responses - TDD FAIL FIRST"""
        response = client.get("/api/health")  # Assuming health endpoint exists
        
        # Security headers that should be present
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        for header in expected_headers:
            assert header in response.headers, f"Security header missing: {header}"
        
        # Verify header values
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") in ["DENY", "SAMEORIGIN"]
        assert "max-age" in response.headers.get("Strict-Transport-Security", "").lower()

    def test_content_type_security_validation(self, client):
        """Test content type security validation - TDD FAIL FIRST"""
        # Test with malicious content types
        malicious_content_types = [
            "text/html",  # Should not serve HTML from API
            "application/javascript",
            "text/javascript", 
            "application/x-msdownload",
            "application/octet-stream"
        ]
        
        for content_type in malicious_content_types:
            headers = {"Accept": content_type}
            response = client.get("/api/portfolio/positions", headers=headers)
            
            # API should return JSON, not requested dangerous content type
            actual_content_type = response.headers.get("content-type", "").lower()
            assert "application/json" in actual_content_type, \
                f"API should not serve dangerous content type: {content_type}"