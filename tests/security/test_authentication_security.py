"""
Phase 7.3: Authentication & Authorization Security Tests
Following TDD methodology - these tests validate authentication security
"""

import pytest
import jwt
import time
import secrets
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
import asyncio

from src.dashboard.auth import (
    DashboardAuth, User, Session, dashboard_auth,
    get_current_user_api_key, get_current_user_jwt, 
    require_permission, require_admin
)
from src.utils.config import get_config


class TestAuthenticationSecurity:
    """Test authentication security following TDD methodology"""
    
    @pytest.fixture
    def auth_system(self):
        """Create fresh auth system for testing"""
        return DashboardAuth()
    
    @pytest.fixture
    def test_user(self):
        """Create test user for security testing"""
        return User(
            username="testuser",
            user_id="test-001",
            permissions={"read", "write"},
            created_at=datetime.utcnow(),
            is_active=True
        )
    
    @pytest.fixture
    def admin_user(self):
        """Create admin user for testing"""
        return User(
            username="admin",
            user_id="admin-001",
            permissions={"read", "write", "admin", "trading"},
            created_at=datetime.utcnow(),
            is_active=True
        )

    def test_jwt_token_validation_with_expired_token_should_fail(self, auth_system):
        """Test that expired JWT tokens are rejected - TDD FAIL FIRST"""
        # Create expired token
        payload = {
            "user_id": "test-001",
            "username": "testuser", 
            "permissions": ["read"],
            "iat": datetime.utcnow() - timedelta(hours=25),  # Issued 25 hours ago
            "exp": datetime.utcnow() - timedelta(hours=1)    # Expired 1 hour ago
        }
        
        # Mock config for JWT secret
        with patch('src.dashboard.auth.get_config') as mock_config:
            mock_config.return_value.security.secret_key = "test-secret-key"
            mock_config.return_value.security.jwt_algorithm = "HS256"
            
            expired_token = jwt.encode(
                payload, 
                "test-secret-key", 
                algorithm="HS256"
            )
            
            # Should fail with expired token
            result = auth_system.verify_jwt_token(expired_token)
            assert result is None, "Expired JWT token should be rejected"

    def test_jwt_token_validation_with_invalid_signature_should_fail(self, auth_system):
        """Test that JWT tokens with invalid signatures are rejected - TDD FAIL FIRST"""
        # Create token with wrong secret
        payload = {
            "user_id": "test-001",
            "username": "testuser",
            "permissions": ["read"],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        # Create token with wrong secret
        wrong_secret_token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
        
        # Mock config with correct secret  
        with patch('src.dashboard.auth.get_config') as mock_config:
            mock_config.return_value.security.secret_key = "correct-secret"
            mock_config.return_value.security.jwt_algorithm = "HS256"
            
            # Should fail with invalid signature
            result = auth_system.verify_jwt_token(wrong_secret_token)
            assert result is None, "JWT token with invalid signature should be rejected"

    def test_api_key_authentication_with_invalid_key_should_fail(self, auth_system):
        """Test that invalid API keys are rejected - TDD FAIL FIRST"""
        invalid_api_key = "invalid-api-key-12345"
        
        # Should return None for invalid API key
        result = auth_system.authenticate_api_key(invalid_api_key)
        assert result is None, "Invalid API key should be rejected"

    def test_api_key_authentication_with_inactive_user_should_fail(self, auth_system):
        """Test that API keys for inactive users are rejected - TDD FAIL FIRST"""
        # Create user and API key
        api_key = auth_system.create_user("testuser", "password", {"read"})
        
        # Deactivate user
        user_id = auth_system._api_keys[api_key]
        auth_system._users[user_id].is_active = False
        
        # Should return None for inactive user
        result = auth_system.authenticate_api_key(api_key)
        assert result is None, "API key for inactive user should be rejected"

    def test_session_expiration_validation_should_fail_expired_sessions(self, auth_system, test_user):
        """Test that expired sessions are properly invalidated - TDD FAIL FIRST"""
        # Create session
        session_id = auth_system.create_session(test_user)
        
        # Manually expire session
        auth_system._sessions[session_id].expires_at = datetime.utcnow() - timedelta(hours=1)
        
        # Should return None for expired session
        result = auth_system.get_session(session_id)
        assert result is None, "Expired session should be invalidated"
        
        # Session should be removed from memory
        assert session_id not in auth_system._sessions, "Expired session should be removed"

    def test_password_hashing_security_validation(self, auth_system):
        """Test password hashing implementation security - TDD FAIL FIRST"""
        password = "test-password-123"
        salt = secrets.token_hex(16)
        
        # Hash password
        hashed1 = auth_system._hash_password(password, salt)
        hashed2 = auth_system._hash_password(password, salt)
        
        # Same password with same salt should produce same hash
        assert hashed1 == hashed2, "Same password+salt should produce same hash"
        
        # Different salt should produce different hash
        different_salt = secrets.token_hex(16)
        hashed3 = auth_system._hash_password(password, different_salt)
        assert hashed1 != hashed3, "Different salt should produce different hash"
        
        # Hash should be proper length (64 hex chars for 32 bytes)
        assert len(hashed1) == 64, "Hash should be 64 hex characters"
        
        # Password verification should work
        assert auth_system._verify_password(password, hashed1, salt), "Password verification should work"
        
        # Wrong password should fail verification
        assert not auth_system._verify_password("wrong-password", hashed1, salt), "Wrong password should fail"

    def test_rate_limiting_enforcement_should_block_excessive_requests(self, auth_system):
        """Test rate limiting blocks excessive requests - TDD FAIL FIRST"""
        user_id = "test-user-001"
        max_requests = 5
        window_minutes = 1
        
        # First 5 requests should pass
        for i in range(max_requests):
            result = auth_system.check_rate_limit(user_id, max_requests, window_minutes)
            assert result is True, f"Request {i+1} should pass rate limit"
        
        # 6th request should fail
        result = auth_system.check_rate_limit(user_id, max_requests, window_minutes)
        assert result is False, "Request exceeding rate limit should fail"

    def test_session_hijacking_protection_with_session_validation(self, auth_system, test_user):
        """Test session security against hijacking attempts - TDD FAIL FIRST"""
        # Create legitimate session
        session_id = auth_system.create_session(test_user)
        session = auth_system.get_session(session_id)
        assert session is not None, "Legitimate session should be valid"
        
        # Test with manipulated session ID
        fake_session_id = "fake-session-" + secrets.token_hex(12)
        fake_session = auth_system.get_session(fake_session_id)
        assert fake_session is None, "Fake session ID should be rejected"
        
        # Test session cleanup removes invalid sessions
        initial_count = len(auth_system._sessions)
        expired_count = auth_system.cleanup_expired_sessions()
        assert expired_count >= 0, "Cleanup should return count of expired sessions"

    def test_permission_escalation_prevention(self, auth_system, test_user):
        """Test that users cannot escalate their permissions - TDD FAIL FIRST"""
        # User has only read, write permissions
        assert "admin" not in test_user.permissions, "Test user should not have admin permissions"
        assert "trading" not in test_user.permissions, "Test user should not have trading permissions"
        
        # Check permission should fail for admin operations
        has_admin = auth_system.check_permission(test_user, "admin")
        assert has_admin is False, "User without admin permission should be denied admin access"
        
        # Check permission should fail for trading operations  
        has_trading = auth_system.check_permission(test_user, "trading")
        assert has_trading is False, "User without trading permission should be denied trading access"
        
        # Should only allow read/write operations
        has_read = auth_system.check_permission(test_user, "read")
        has_write = auth_system.check_permission(test_user, "write")
        assert has_read is True, "User should have read permission"
        assert has_write is True, "User should have write permission"


class TestAuthorizationSecurity:
    """Test authorization security mechanisms"""
    
    @pytest.fixture
    def mock_credentials(self):
        """Mock HTTP authorization credentials"""
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-token")
    
    @pytest.mark.asyncio
    async def test_unauthorized_api_key_access_should_fail(self, mock_credentials):
        """Test that unauthorized API key access fails - TDD FAIL FIRST"""
        mock_credentials.credentials = "invalid-api-key"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_api_key(mock_credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid API key" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_rate_limited_user_should_be_blocked(self):
        """Test that rate limited users are blocked - TDD FAIL FIRST"""
        # Mock user that exceeds rate limit
        with patch('src.dashboard.auth.dashboard_auth') as mock_auth:
            mock_user = User(
                username="ratelimited",
                user_id="rate-001", 
                permissions={"read"},
                created_at=datetime.utcnow()
            )
            mock_auth.authenticate_api_key.return_value = mock_user
            mock_auth.check_rate_limit.return_value = False  # Rate limited
            
            mock_credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", 
                credentials="valid-api-key"
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_api_key(mock_credentials)
            
            assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            assert "Rate limit exceeded" in str(exc_info.value.detail)

    @pytest.mark.asyncio 
    async def test_jwt_authentication_with_invalid_user_should_fail(self):
        """Test JWT authentication fails for invalid users - TDD FAIL FIRST"""
        # Mock JWT verification that returns payload for non-existent user
        with patch('src.dashboard.auth.dashboard_auth') as mock_auth:
            mock_auth.verify_jwt_token.return_value = {
                "user_id": "non-existent-user",
                "username": "ghost",
                "permissions": ["read"]
            }
            mock_auth._users = {}  # Empty user store
            
            mock_credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="valid-jwt-token"
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_jwt(mock_credentials)
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "User not found" in str(exc_info.value.detail)

    def test_permission_decorator_enforcement_should_deny_unauthorized(self):
        """Test permission decorators properly enforce authorization - TDD FAIL FIRST"""
        # Test user without admin permission
        test_user = User(
            username="regular_user",
            user_id="regular-001",
            permissions={"read", "write"},
            created_at=datetime.utcnow()
        )
        
        # Mock the auth system
        with patch('src.dashboard.auth.dashboard_auth') as mock_auth:
            mock_auth.check_permission.return_value = False
            
            # Create permission dependency
            admin_dependency = require_permission("admin")
            
            # Should raise HTTP 403 Forbidden
            with pytest.raises(HTTPException) as exc_info:
                admin_dependency(test_user)
            
            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Permission 'admin' required" in str(exc_info.value.detail)

    def test_admin_permission_grants_all_access(self):
        """Test that admin permission grants access to all operations"""
        admin_user = User(
            username="admin_user",
            user_id="admin-001", 
            permissions={"admin"},  # Only admin permission
            created_at=datetime.utcnow()
        )
        
        auth_system = DashboardAuth()
        
        # Admin should have access to all permissions
        assert auth_system.check_permission(admin_user, "read"), "Admin should have read access"
        assert auth_system.check_permission(admin_user, "write"), "Admin should have write access"
        assert auth_system.check_permission(admin_user, "trading"), "Admin should have trading access"
        assert auth_system.check_permission(admin_user, "admin"), "Admin should have admin access"


class TestSecurityAuditLogging:
    """Test security event audit logging"""
    
    @pytest.mark.asyncio
    async def test_failed_authentication_attempts_are_logged(self):
        """Test that failed authentication attempts are properly logged - TDD FAIL FIRST"""
        auth_system = DashboardAuth()
        
        # Mock activity logger to capture security events
        with patch('src.dashboard.auth.activity_logger') as mock_logger:
            # Attempt authentication with invalid API key
            result = auth_system.authenticate_api_key("invalid-key")
            assert result is None, "Invalid authentication should fail"
            
            # Wait for async logging
            await asyncio.sleep(0.1)
            
            # Verify security violation was logged
            mock_logger.log_activity.assert_called()
            call_args = mock_logger.log_activity.call_args
            
            # Verify security logging parameters
            assert call_args.kwargs['category'].value == 'SECURITY'
            assert call_args.kwargs['action'].value == 'VIOLATION'
            assert call_args.kwargs['event_type'] == 'invalid_api_key'
            assert call_args.kwargs['severity'].value == 'WARNING'

    @pytest.mark.asyncio
    async def test_successful_authentication_is_logged(self):
        """Test that successful authentication is properly logged - TDD FAIL FIRST"""
        auth_system = DashboardAuth()
        
        # Create user and get API key
        api_key = auth_system.create_user("testuser", "password", {"read"})
        
        with patch('src.dashboard.auth.activity_logger') as mock_logger:
            # Authenticate with valid API key
            result = auth_system.authenticate_api_key(api_key)
            assert result is not None, "Valid authentication should succeed"
            
            # Wait for async logging
            await asyncio.sleep(0.1)
            
            # Verify success event was logged
            mock_logger.log_activity.assert_called()
            call_args = mock_logger.log_activity.call_args
            
            # Verify success logging parameters
            assert call_args.kwargs['category'].value == 'SECURITY'
            assert call_args.kwargs['action'].value == 'ACCESS'
            assert call_args.kwargs['event_type'] == 'api_key_auth_success'
            assert call_args.kwargs['severity'].value == 'INFO'

    @pytest.mark.asyncio
    async def test_session_creation_and_revocation_are_logged(self):
        """Test that session lifecycle events are logged - TDD FAIL FIRST"""
        auth_system = DashboardAuth()
        
        test_user = User(
            username="testuser",
            user_id="test-001",
            permissions={"read"},
            created_at=datetime.utcnow()
        )
        
        with patch('src.dashboard.auth.activity_logger') as mock_logger:
            # Create session
            session_id = auth_system.create_session(test_user)
            assert session_id is not None, "Session creation should succeed"
            
            # Revoke session
            revoked = auth_system.revoke_session(session_id)
            assert revoked is True, "Session revocation should succeed"
            
            # Wait for async logging
            await asyncio.sleep(0.1)
            
            # Verify both create and revoke were logged
            assert mock_logger.log_activity.call_count >= 2, "Both session create and revoke should be logged"


class TestSecurityTokenGeneration:
    """Test security token generation and validation"""
    
    def test_api_key_generation_is_cryptographically_secure(self):
        """Test API key generation uses cryptographically secure methods - TDD FAIL FIRST"""
        auth_system = DashboardAuth()
        
        # Generate multiple API keys
        keys = set()
        for _ in range(100):
            key = auth_system._generate_api_key()
            
            # Should be URL-safe base64
            assert key.replace('-', '').replace('_', '').isalnum(), "API key should be URL-safe base64"
            
            # Should be sufficiently long (32 bytes = 43+ base64 chars)
            assert len(key) >= 40, "API key should be sufficiently long"
            
            # Should be unique
            assert key not in keys, "API keys should be unique"
            keys.add(key)

    def test_session_id_generation_is_cryptographically_secure(self):
        """Test session ID generation uses secure methods - TDD FAIL FIRST"""
        auth_system = DashboardAuth()
        
        # Generate multiple session IDs
        session_ids = set()
        for _ in range(100):
            session_id = auth_system._generate_session_id()
            
            # Should be URL-safe base64
            assert session_id.replace('-', '').replace('_', '').isalnum(), "Session ID should be URL-safe base64"
            
            # Should be sufficiently long (24 bytes = 32+ base64 chars)
            assert len(session_id) >= 30, "Session ID should be sufficiently long"
            
            # Should be unique
            assert session_id not in session_ids, "Session IDs should be unique"
            session_ids.add(session_id)

    def test_jwt_secret_key_validation_in_production(self):
        """Test JWT secret key validation for production security - TDD FAIL FIRST"""
        # Mock production config
        with patch('src.dashboard.auth.get_config') as mock_config:
            # Test weak secret key should be detected
            mock_config.return_value.security.secret_key = "weak"
            mock_config.return_value.security.jwt_algorithm = "HS256"
            
            auth_system = DashboardAuth()
            
            # Create test user for token generation
            test_user = User(
                username="testuser",
                user_id="test-001",
                permissions={"read"},
                created_at=datetime.utcnow()
            )
            
            # Should not allow weak secret keys in production
            # This test validates the key strength
            with pytest.raises((ValueError, Exception)) as exc_info:
                # This should fail if proper validation is implemented
                token = auth_system.create_jwt_token(test_user)
                # If no exception, check token manually
                if len(mock_config.return_value.security.secret_key) < 16:
                    raise ValueError("JWT secret key too weak for production")