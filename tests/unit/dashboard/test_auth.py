"""
Tests for dashboard authentication and authorization
"""

import pytest
import asyncio
import jwt
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.dashboard.auth import (
    DashboardAuth, User, Session, dashboard_auth,
    get_current_user_api_key, get_current_user_jwt, require_permission
)


class TestUser:
    """Test suite for User dataclass"""
    
    def test_user_creation(self):
        """Test User creation with valid data"""
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read", "write", "admin"},
            created_at=datetime.utcnow()
        )
        
        assert user.username == "testuser"
        assert user.user_id == "user-123"
        assert len(user.permissions) == 3
        assert "read" in user.permissions
        assert "admin" in user.permissions
        assert user.is_active is True
    
    def test_user_with_last_login(self):
        """Test User with last_login set"""
        now = datetime.utcnow()
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read"},
            created_at=now,
            last_login=now,
            is_active=False
        )
        
        assert user.last_login == now
        assert user.is_active is False


class TestSession:
    """Test suite for Session dataclass"""
    
    def test_session_creation(self):
        """Test Session creation"""
        now = datetime.utcnow()
        expires = now + timedelta(hours=24)
        
        session = Session(
            session_id="session-123",
            user_id="user-123",
            username="testuser",
            permissions={"read", "write"},
            created_at=now,
            expires_at=expires,
            last_activity=now
        )
        
        assert session.session_id == "session-123"
        assert session.user_id == "user-123"
        assert session.username == "testuser"
        assert len(session.permissions) == 2
        assert session.expires_at == expires


class TestDashboardAuth:
    """Test suite for DashboardAuth"""
    
    @pytest.fixture
    def auth_instance(self):
        """DashboardAuth instance for testing"""
        with patch('src.dashboard.auth.get_config') as mock_config:
            config = MagicMock()
            config.security.secret_key = "test-secret-key"
            config.security.jwt_algorithm = "HS256"
            mock_config.return_value = config
            return DashboardAuth()
    
    def test_dashboard_auth_initialization(self, auth_instance):
        """Test DashboardAuth initialization"""
        auth = auth_instance
        
        assert auth._users is not None
        assert auth._sessions is not None
        assert auth._api_keys is not None
        assert auth._rate_limits is not None
        
        # Should have default admin user
        assert len(auth._users) == 1
        admin_user = list(auth._users.values())[0]
        assert admin_user.username == "admin"
        assert "admin" in admin_user.permissions
    
    def test_create_user(self, auth_instance):
        """Test user creation"""
        auth = auth_instance
        initial_user_count = len(auth._users)
        
        api_key = auth.create_user(
            username="testuser",
            password="testpass",
            permissions={"read", "write"}
        )
        
        assert len(auth._users) == initial_user_count + 1
        assert len(api_key) > 0
        assert api_key in auth._api_keys
        
        # Find the new user
        new_user = None
        for user in auth._users.values():
            if user.username == "testuser":
                new_user = user
                break
        
        assert new_user is not None
        assert new_user.username == "testuser"
        assert "read" in new_user.permissions
        assert "write" in new_user.permissions
    
    def test_authenticate_api_key_valid(self, auth_instance):
        """Test API key authentication with valid key"""
        auth = auth_instance
        
        # Create a user and get their API key
        api_key = auth.create_user(
            username="testuser",
            password="testpass",
            permissions={"read"}
        )
        
        # Authenticate with the API key
        user = auth.authenticate_api_key(api_key)
        
        assert user is not None
        assert user.username == "testuser"
        assert "read" in user.permissions
        assert user.is_active is True
    
    def test_authenticate_api_key_invalid(self, auth_instance):
        """Test API key authentication with invalid key"""
        auth = auth_instance
        
        # Test invalid key
        user = auth.authenticate_api_key("invalid-key")
        assert user is None
        
        # Test empty key
        user = auth.authenticate_api_key("")
        assert user is None
        
        # Test None key
        user = auth.authenticate_api_key(None)
        assert user is None
    
    def test_authenticate_api_key_inactive_user(self, auth_instance):
        """Test API key authentication with inactive user"""
        auth = auth_instance
        
        # Create user and get API key
        api_key = auth.create_user(
            username="testuser",
            password="testpass",
            permissions={"read"}
        )
        
        # Deactivate user
        user_id = auth._api_keys[api_key]
        auth._users[user_id].is_active = False
        
        # Should not authenticate
        user = auth.authenticate_api_key(api_key)
        assert user is None
    
    def test_create_session(self, auth_instance):
        """Test session creation"""
        auth = auth_instance
        
        # Create a user
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read"},
            created_at=datetime.utcnow()
        )
        auth._users[user.user_id] = user
        
        # Create session
        session_id = auth.create_session(user)
        
        assert len(session_id) > 0
        assert session_id in auth._sessions
        
        session = auth._sessions[session_id]
        assert session.user_id == user.user_id
        assert session.username == user.username
        assert session.permissions == user.permissions
        assert user.last_login is not None
    
    def test_get_session_valid(self, auth_instance):
        """Test getting valid session"""
        auth = auth_instance
        
        # Create user and session
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read"},
            created_at=datetime.utcnow()
        )
        auth._users[user.user_id] = user
        session_id = auth.create_session(user)
        
        # Get session
        session = auth.get_session(session_id)
        
        assert session is not None
        assert session.session_id == session_id
        assert session.user_id == user.user_id
    
    def test_get_session_expired(self, auth_instance):
        """Test getting expired session"""
        auth = auth_instance
        
        # Create expired session manually
        now = datetime.utcnow()
        expired_session = Session(
            session_id="expired-session",
            user_id="user-123",
            username="testuser",
            permissions={"read"},
            created_at=now - timedelta(hours=25),
            expires_at=now - timedelta(hours=1),  # Expired
            last_activity=now - timedelta(hours=2)
        )
        auth._sessions["expired-session"] = expired_session
        
        # Should return None and remove session
        session = auth.get_session("expired-session")
        assert session is None
        assert "expired-session" not in auth._sessions
    
    def test_revoke_session(self, auth_instance):
        """Test session revocation"""
        auth = auth_instance
        
        # Create user and session
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read"},
            created_at=datetime.utcnow()
        )
        auth._users[user.user_id] = user
        session_id = auth.create_session(user)
        
        # Revoke session
        result = auth.revoke_session(session_id)
        
        assert result is True
        assert session_id not in auth._sessions
        
        # Revoking again should return False
        result = auth.revoke_session(session_id)
        assert result is False
    
    def test_create_jwt_token(self, auth_instance):
        """Test JWT token creation"""
        auth = auth_instance
        
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read", "write"},
            created_at=datetime.utcnow()
        )
        
        token = auth.create_jwt_token(user)
        
        assert len(token) > 0
        
        # Verify token can be decoded
        payload = jwt.decode(
            token,
            "test-secret-key",
            algorithms=["HS256"]
        )
        
        assert payload["user_id"] == user.user_id
        assert payload["username"] == user.username
        assert set(payload["permissions"]) == user.permissions
    
    def test_verify_jwt_token_valid(self, auth_instance):
        """Test JWT token verification with valid token"""
        auth = auth_instance
        
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read"},
            created_at=datetime.utcnow()
        )
        
        # Create token
        token = auth.create_jwt_token(user)
        
        # Verify token
        payload = auth.verify_jwt_token(token)
        
        assert payload is not None
        assert payload["user_id"] == user.user_id
        assert payload["username"] == user.username
    
    def test_verify_jwt_token_invalid(self, auth_instance):
        """Test JWT token verification with invalid tokens"""
        auth = auth_instance
        
        # Test invalid token
        payload = auth.verify_jwt_token("invalid.jwt.token")
        assert payload is None
        
        # Test expired token
        expired_payload = {
            "user_id": "user-123",
            "username": "testuser",
            "permissions": ["read"],
            "iat": datetime.utcnow() - timedelta(hours=25),
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired
        }
        expired_token = jwt.encode(expired_payload, "test-secret-key", algorithm="HS256")
        
        payload = auth.verify_jwt_token(expired_token)
        assert payload is None
    
    def test_check_permission(self, auth_instance):
        """Test permission checking"""
        auth = auth_instance
        
        # User with specific permissions
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read", "write"},
            created_at=datetime.utcnow()
        )
        
        assert auth.check_permission(user, "read") is True
        assert auth.check_permission(user, "write") is True
        assert auth.check_permission(user, "admin") is False
        
        # Admin user should have all permissions
        admin_user = User(
            username="admin",
            user_id="admin-123",
            permissions={"admin"},
            created_at=datetime.utcnow()
        )
        
        assert auth.check_permission(admin_user, "read") is True
        assert auth.check_permission(admin_user, "write") is True
        assert auth.check_permission(admin_user, "any_permission") is True
    
    def test_check_rate_limit(self, auth_instance):
        """Test rate limiting"""
        auth = auth_instance
        user_id = "user-123"
        
        # Should allow requests initially
        for _ in range(10):
            result = auth.check_rate_limit(user_id, max_requests=10, window_minutes=60)
            assert result is True
        
        # Should block after limit
        result = auth.check_rate_limit(user_id, max_requests=10, window_minutes=60)
        assert result is False
    
    def test_cleanup_expired_sessions(self, auth_instance):
        """Test expired session cleanup"""
        auth = auth_instance
        
        # Create expired session
        now = datetime.utcnow()
        expired_session = Session(
            session_id="expired-session",
            user_id="user-123",
            username="testuser",
            permissions={"read"},
            created_at=now - timedelta(hours=25),
            expires_at=now - timedelta(hours=1),  # Expired
            last_activity=now - timedelta(hours=2)
        )
        auth._sessions["expired-session"] = expired_session
        
        # Create valid session
        valid_session = Session(
            session_id="valid-session",
            user_id="user-456",
            username="validuser",
            permissions={"read"},
            created_at=now,
            expires_at=now + timedelta(hours=23),  # Valid
            last_activity=now
        )
        auth._sessions["valid-session"] = valid_session
        
        # Cleanup
        cleaned_count = auth.cleanup_expired_sessions()
        
        assert cleaned_count == 1
        assert "expired-session" not in auth._sessions
        assert "valid-session" in auth._sessions
    
    def test_get_user_stats(self, auth_instance):
        """Test user statistics"""
        auth = auth_instance
        
        stats = auth.get_user_stats()
        
        assert "total_users" in stats
        assert "active_users" in stats
        assert "active_sessions" in stats
        assert "api_keys_issued" in stats
        
        # Should have at least the default admin user
        assert stats["total_users"] >= 1
        assert stats["active_users"] >= 1


class TestDashboardAuthDependencies:
    """Test authentication dependency functions"""
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.dashboard_auth')
    async def test_get_current_user_api_key_valid(self, mock_auth):
        """Test get_current_user_api_key with valid key"""
        # Setup mock
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read"},
            created_at=datetime.utcnow()
        )
        mock_auth.authenticate_api_key.return_value = user
        mock_auth.check_rate_limit.return_value = True
        
        # Mock credentials
        credentials = MagicMock()
        credentials.credentials = "valid-api-key"
        
        # Test function
        result = await get_current_user_api_key(credentials)
        
        assert result == user
        mock_auth.authenticate_api_key.assert_called_once_with("valid-api-key")
        mock_auth.check_rate_limit.assert_called_once_with(user.user_id)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.dashboard_auth')
    async def test_get_current_user_api_key_invalid(self, mock_auth):
        """Test get_current_user_api_key with invalid key"""
        from fastapi import HTTPException
        
        # Setup mock to return None (invalid key)
        mock_auth.authenticate_api_key.return_value = None
        
        # Mock credentials
        credentials = MagicMock()
        credentials.credentials = "invalid-api-key"
        
        # Test function - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_api_key(credentials)
        
        assert exc_info.value.status_code == 401
        assert "Authentication failed" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.dashboard_auth')
    async def test_get_current_user_api_key_rate_limited(self, mock_auth):
        """Test get_current_user_api_key with rate limit exceeded"""
        from fastapi import HTTPException
        
        # Setup mock
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read"},
            created_at=datetime.utcnow()
        )
        mock_auth.authenticate_api_key.return_value = user
        mock_auth.check_rate_limit.return_value = False  # Rate limited
        
        # Mock credentials
        credentials = MagicMock()
        credentials.credentials = "valid-api-key"
        
        # Test function - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_api_key(credentials)
        
        assert exc_info.value.status_code == 401
        assert "Authentication failed" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.dashboard_auth')
    async def test_get_current_user_jwt_valid(self, mock_auth):
        """Test get_current_user_jwt with valid token"""
        # Setup mock
        user = User(
            username="testuser",
            user_id="user-123",
            permissions={"read"},
            created_at=datetime.utcnow()
        )
        payload = {
            "user_id": "user-123",
            "username": "testuser",
            "permissions": ["read"]
        }
        mock_auth.verify_jwt_token.return_value = payload
        mock_auth._users = {"user-123": user}
        mock_auth.check_rate_limit.return_value = True
        
        # Mock credentials
        credentials = MagicMock()
        credentials.credentials = "valid-jwt-token"
        
        # Test function
        result = await get_current_user_jwt(credentials)
        
        assert result == user
        mock_auth.verify_jwt_token.assert_called_once_with("valid-jwt-token")
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.dashboard_auth')
    async def test_get_current_user_jwt_invalid(self, mock_auth):
        """Test get_current_user_jwt with invalid token"""
        from fastapi import HTTPException
        
        # Setup mock to return None (invalid token)
        mock_auth.verify_jwt_token.return_value = None
        
        # Mock credentials
        credentials = MagicMock()
        credentials.credentials = "invalid-jwt-token"
        
        # Test function - should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_jwt(credentials)
        
        assert exc_info.value.status_code == 401
        assert "Invalid or expired token" in str(exc_info.value.detail)
    
    def test_require_permission_decorator(self):
        """Test require_permission decorator"""
        from fastapi import HTTPException
        
        # Create decorator
        decorator = require_permission("read")
        
        # Create user with permission
        user_with_permission = User(
            username="testuser",
            user_id="user-123",
            permissions={"read"},
            created_at=datetime.utcnow()
        )
        
        # Should return user when permission exists
        result = decorator(user_with_permission)
        assert result == user_with_permission
        
        # Create user without permission
        user_without_permission = User(
            username="testuser2",
            user_id="user-456",
            permissions={"write"},
            created_at=datetime.utcnow()
        )
        
        # Should raise HTTPException when permission missing
        with pytest.raises(HTTPException) as exc_info:
            decorator(user_without_permission)
        
        assert exc_info.value.status_code == 403
        assert "Permission 'read' required" in str(exc_info.value.detail)


class TestGlobalDashboardAuth:
    """Test global dashboard_auth instance"""
    
    def test_global_instance_exists(self):
        """Test that global dashboard_auth instance exists"""
        assert dashboard_auth is not None
        assert isinstance(dashboard_auth, DashboardAuth)
    
    def test_global_instance_has_admin(self):
        """Test that global instance has default admin user"""
        assert len(dashboard_auth._users) >= 1
        
        # Find admin user
        admin_user = None
        for user in dashboard_auth._users.values():
            if user.username == "admin":
                admin_user = user
                break
        
        assert admin_user is not None
        assert "admin" in admin_user.permissions
    
    def test_global_instance_functionality(self):
        """Test that global instance works correctly"""
        # Should handle invalid API key gracefully
        user = dashboard_auth.authenticate_api_key("invalid-key")
        assert user is None
        
        # Should provide user stats
        stats = dashboard_auth.get_user_stats()
        assert isinstance(stats, dict)
        assert "total_users" in stats


class TestDashboardAuthIntegration:
    """Integration tests for DashboardAuth"""
    
    @pytest.fixture
    def fresh_auth(self):
        """Fresh DashboardAuth instance for integration tests"""
        with patch('src.dashboard.auth.get_config') as mock_config:
            config = MagicMock()
            config.security.secret_key = "integration-test-secret"
            config.security.jwt_algorithm = "HS256"
            mock_config.return_value = config
            return DashboardAuth()
    
    def test_full_user_lifecycle(self, fresh_auth):
        """Test complete user lifecycle"""
        auth = fresh_auth
        
        # 1. Create user
        api_key = auth.create_user(
            username="testuser",
            password="testpass",
            permissions={"read", "write"}
        )
        
        # 2. Authenticate with API key
        user = auth.authenticate_api_key(api_key)
        assert user is not None
        assert user.username == "testuser"
        
        # 3. Create session
        session_id = auth.create_session(user)
        
        # 4. Get session
        session = auth.get_session(session_id)
        assert session is not None
        assert session.user_id == user.user_id
        
        # 5. Create JWT token
        jwt_token = auth.create_jwt_token(user)
        
        # 6. Verify JWT token
        payload = auth.verify_jwt_token(jwt_token)
        assert payload is not None
        assert payload["user_id"] == user.user_id
        
        # 7. Check permission
        assert auth.check_permission(user, "read") is True
        assert auth.check_permission(user, "admin") is False
        
        # 8. Revoke session
        result = auth.revoke_session(session_id)
        assert result is True
        
        # 9. Session should be gone
        session = auth.get_session(session_id)
        assert session is None
    
    def test_concurrent_operations(self, fresh_auth):
        """Test concurrent authentication operations"""
        auth = fresh_auth
        
        # Create multiple users concurrently (simulated)
        api_keys = []
        for i in range(5):
            api_key = auth.create_user(
                username=f"user{i}",
                password=f"pass{i}",
                permissions={"read"}
            )
            api_keys.append(api_key)
        
        # Authenticate all users
        users = []
        for api_key in api_keys:
            user = auth.authenticate_api_key(api_key)
            assert user is not None
            users.append(user)
        
        # Create sessions for all users
        session_ids = []
        for user in users:
            session_id = auth.create_session(user)
            session_ids.append(session_id)
        
        # Verify all sessions exist
        for session_id in session_ids:
            session = auth.get_session(session_id)
            assert session is not None
        
        # Cleanup - revoke all sessions
        for session_id in session_ids:
            result = auth.revoke_session(session_id)
            assert result is True
    
    def test_error_resilience(self, fresh_auth):
        """Test error handling and resilience"""
        auth = fresh_auth
        
        # Test with None inputs
        assert auth.authenticate_api_key(None) is None
        assert auth.verify_jwt_token(None) is None
        assert auth.get_session(None) is None
        
        # Test with empty strings
        assert auth.authenticate_api_key("") is None
        assert auth.verify_jwt_token("") is None
        assert auth.get_session("") is None
        
        # Test invalid operations
        assert auth.revoke_session("nonexistent-session") is False
        
        # Test rate limiting with high load
        user_id = "test-user"
        for _ in range(200):  # Exceed any reasonable limit
            auth.check_rate_limit(user_id, max_requests=10, window_minutes=1)
        
        # Should still handle the call without crashing
        result = auth.check_rate_limit(user_id, max_requests=10, window_minutes=1)
        assert isinstance(result, bool)