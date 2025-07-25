"""
Tests for dashboard authentication and authorization
"""

import pytest
import asyncio
import jwt
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.dashboard.auth import (
    AuthService, Role, AuthToken, LoginRequest, TokenResponse,
    AuthenticationError, AuthorizationError, RateLimitError,
    auth_service, require_permission, verify_api_key, verify_jwt_token
)


class TestRole:
    """Test suite for Role enum"""
    
    def test_role_values(self):
        """Test Role enum values"""
        assert Role.ADMIN.value == "admin"
        assert Role.TRADER.value == "trader"
        assert Role.VIEWER.value == "viewer"
    
    def test_role_comparison(self):
        """Test Role comparison"""
        assert Role.ADMIN != Role.TRADER
        assert Role.TRADER != Role.VIEWER
        assert Role.ADMIN == Role.ADMIN


class TestAuthToken:
    """Test suite for AuthToken data structure"""
    
    def test_auth_token_creation(self):
        """Test AuthToken creation with valid data"""
        token = AuthToken(
            user_id="user-123",
            username="testuser",
            role=Role.ADMIN,
            permissions=["dashboard.read", "dashboard.write", "trading.control"]
        )
        
        assert token.user_id == "user-123"
        assert token.username == "testuser"
        assert token.role == Role.ADMIN
        assert len(token.permissions) == 3
        assert "dashboard.read" in token.permissions
        assert "trading.control" in token.permissions
    
    def test_auth_token_has_permission(self):
        """Test AuthToken permission checking"""
        token = AuthToken(
            user_id="user-123",
            username="testuser",
            role=Role.TRADER,
            permissions=["dashboard.read", "trading.control"]
        )
        
        assert token.has_permission("dashboard.read") is True
        assert token.has_permission("trading.control") is True
        assert token.has_permission("system.control") is False
        assert token.has_permission("nonexistent.permission") is False


class TestLoginRequest:
    """Test suite for LoginRequest data structure"""
    
    def test_login_request_creation(self):
        """Test LoginRequest creation"""
        request = LoginRequest(
            username="testuser",
            password="testpass"
        )
        
        assert request.username == "testuser"
        assert request.password == "testpass"


class TestTokenResponse:
    """Test suite for TokenResponse data structure"""
    
    def test_token_response_creation(self):
        """Test TokenResponse creation"""
        response = TokenResponse(
            access_token="jwt-token-here",
            token_type="bearer",
            expires_in=3600,
            user_id="user-123",
            username="testuser",
            role="admin",
            permissions=["dashboard.read", "trading.control"]
        )
        
        assert response.access_token == "jwt-token-here"
        assert response.token_type == "bearer"
        assert response.expires_in == 3600
        assert response.user_id == "user-123"
        assert response.username == "testuser"
        assert response.role == "admin"
        assert len(response.permissions) == 2


class TestAuthService:
    """Test suite for AuthService"""
    
    @pytest.fixture
    def auth_service_instance(self):
        """AuthService instance for testing"""
        return AuthService()
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        config = MagicMock()
        config.dashboard.jwt_secret = "test-secret-key"
        config.dashboard.jwt_algorithm = "HS256"
        config.dashboard.jwt_expiry_hours = 24
        config.dashboard.api_keys = {
            "admin-key": {
                "user_id": "admin-user",
                "username": "admin",
                "role": "admin",
                "permissions": ["dashboard.read", "dashboard.write", "trading.control", "system.control"]
            },
            "trader-key": {
                "user_id": "trader-user", 
                "username": "trader",
                "role": "trader",
                "permissions": ["dashboard.read", "trading.control"]
            },
            "viewer-key": {
                "user_id": "viewer-user",
                "username": "viewer",
                "role": "viewer",
                "permissions": ["dashboard.read"]
            }
        }
        config.dashboard.users = {
            "admin": {
                "password_hash": "$2b$12$hash.for.admin.password",
                "role": "admin",
                "permissions": ["dashboard.read", "dashboard.write", "trading.control", "system.control"]
            },
            "trader": {
                "password_hash": "$2b$12$hash.for.trader.password",
                "role": "trader", 
                "permissions": ["dashboard.read", "trading.control"]
            }
        }
        config.dashboard.rate_limit_requests_per_minute = 60
        return config
    
    def test_auth_service_initialization(self, auth_service_instance):
        """Test AuthService initialization"""
        service = auth_service_instance
        
        assert service._rate_limiter is not None
        assert service._failed_attempts is not None
        assert service._active_sessions is not None
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.get_config')
    async def test_verify_api_key_valid(self, mock_get_config, auth_service_instance, mock_config):
        """Test API key verification with valid key"""
        mock_get_config.return_value = mock_config
        service = auth_service_instance
        
        # Test admin key
        token = await service.verify_api_key("admin-key")
        
        assert token is not None
        assert token.user_id == "admin-user"
        assert token.username == "admin"
        assert token.role == Role.ADMIN
        assert "system.control" in token.permissions
        
        # Test trader key
        token = await service.verify_api_key("trader-key")
        
        assert token is not None
        assert token.user_id == "trader-user"
        assert token.role == Role.TRADER
        assert "trading.control" in token.permissions
        assert "system.control" not in token.permissions
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.get_config')
    async def test_verify_api_key_invalid(self, mock_get_config, auth_service_instance, mock_config):
        """Test API key verification with invalid key"""
        mock_get_config.return_value = mock_config
        service = auth_service_instance
        
        # Test invalid key
        token = await service.verify_api_key("invalid-key")
        assert token is None
        
        # Test empty key
        token = await service.verify_api_key("")
        assert token is None
        
        # Test None key
        token = await service.verify_api_key(None)
        assert token is None
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.get_config')
    @patch('src.dashboard.auth.bcrypt')
    async def test_login_success(self, mock_bcrypt, mock_get_config, auth_service_instance, mock_config):
        """Test successful user login"""
        mock_get_config.return_value = mock_config
        mock_bcrypt.checkpw.return_value = True
        service = auth_service_instance
        
        # Create login request
        login_request = LoginRequest(username="admin", password="correct-password")
        
        # Attempt login
        response = await service.login(login_request)
        
        assert response is not None
        assert response.user_id == "admin"
        assert response.username == "admin"
        assert response.role == "admin"
        assert response.token_type == "bearer"
        assert response.expires_in == 24 * 3600  # 24 hours in seconds
        assert len(response.access_token) > 0
        
        # Should have all admin permissions
        assert "system.control" in response.permissions
        assert "trading.control" in response.permissions
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.get_config')
    @patch('src.dashboard.auth.bcrypt')
    async def test_login_invalid_credentials(self, mock_bcrypt, mock_get_config, auth_service_instance, mock_config):
        """Test login with invalid credentials"""
        mock_get_config.return_value = mock_config
        mock_bcrypt.checkpw.return_value = False  # Wrong password
        service = auth_service_instance
        
        # Test wrong password
        login_request = LoginRequest(username="admin", password="wrong-password")
        
        with pytest.raises(AuthenticationError) as exc_info:
            await service.login(login_request)
        assert "Invalid credentials" in str(exc_info.value)
        
        # Test nonexistent user
        login_request = LoginRequest(username="nonexistent", password="password")
        
        with pytest.raises(AuthenticationError) as exc_info:
            await service.login(login_request)
        assert "Invalid credentials" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.get_config')
    async def test_verify_jwt_token_valid(self, mock_get_config, auth_service_instance, mock_config):
        """Test JWT token verification with valid token"""
        mock_get_config.return_value = mock_config
        service = auth_service_instance
        
        # Create a valid JWT token
        payload = {
            "user_id": "test-user",
            "username": "testuser",
            "role": "admin",
            "permissions": ["dashboard.read", "trading.control"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        # Verify token
        auth_token = await service.verify_jwt_token(token)
        
        assert auth_token is not None
        assert auth_token.user_id == "test-user"
        assert auth_token.username == "testuser"
        assert auth_token.role == Role.ADMIN
        assert "dashboard.read" in auth_token.permissions
        assert "trading.control" in auth_token.permissions
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.get_config')
    async def test_verify_jwt_token_invalid(self, mock_get_config, auth_service_instance, mock_config):
        """Test JWT token verification with invalid tokens"""
        mock_get_config.return_value = mock_config
        service = auth_service_instance
        
        # Test invalid token
        invalid_token = await service.verify_jwt_token("invalid.jwt.token")
        assert invalid_token is None
        
        # Test expired token
        payload = {
            "user_id": "test-user",
            "username": "testuser",
            "role": "admin",
            "permissions": ["dashboard.read"],
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired
        }
        expired_token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        invalid_token = await service.verify_jwt_token(expired_token)
        assert invalid_token is None
        
        # Test token with wrong secret
        wrong_secret_token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
        invalid_token = await service.verify_jwt_token(wrong_secret_token)
        assert invalid_token is None
    
    @pytest.mark.asyncio
    async def test_logout(self, auth_service_instance):
        """Test user logout"""
        service = auth_service_instance
        user_id = "test-user"
        
        # Add user to active sessions
        service._active_sessions[user_id] = datetime.utcnow()
        assert user_id in service._active_sessions
        
        # Logout
        await service.logout(user_id)
        
        # Should remove from active sessions
        assert user_id not in service._active_sessions
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, auth_service_instance):
        """Test rate limiting functionality"""
        service = auth_service_instance
        client_ip = "192.168.1.100"
        
        # Should allow first request
        is_allowed = await service.check_rate_limit(client_ip)
        assert is_allowed is True
        
        # Simulate many requests in short time
        for _ in range(100):  # Exceed rate limit
            await service.check_rate_limit(client_ip)
        
        # Should now be rate limited
        is_allowed = await service.check_rate_limit(client_ip)
        assert is_allowed is False
    
    @pytest.mark.asyncio
    async def test_failed_login_tracking(self, auth_service_instance):
        """Test failed login attempt tracking"""
        service = auth_service_instance
        username = "testuser"
        
        # Should allow login attempts initially
        is_allowed = await service.check_failed_attempts(username)
        assert is_allowed is True
        
        # Record failed attempts
        for _ in range(10):  # Exceed failed attempt limit
            await service.record_failed_attempt(username)
        
        # Should now block login attempts
        is_allowed = await service.check_failed_attempts(username)
        assert is_allowed is False
    
    @pytest.mark.asyncio
    async def test_session_management(self, auth_service_instance):
        """Test session management"""
        service = auth_service_instance
        user_id = "test-user"
        
        # Create session
        await service.create_session(user_id)
        
        # Should be in active sessions
        assert user_id in service._active_sessions
        
        # Check if session is valid
        is_valid = await service.is_session_valid(user_id)
        assert is_valid is True
        
        # End session
        await service.end_session(user_id)
        
        # Should no longer be valid
        is_valid = await service.is_session_valid(user_id)
        assert is_valid is False


class TestAuthDecorators:
    """Test suite for authentication decorators and functions"""
    
    @pytest.mark.asyncio
    async def test_require_permission_decorator_success(self):
        """Test require_permission decorator with valid permission"""
        # Mock function that requires permission
        @require_permission("dashboard.read")
        async def protected_function(auth_token: AuthToken):
            return {"success": True, "user": auth_token.username}
        
        # Create token with required permission
        token = AuthToken(
            user_id="user-123",
            username="testuser",
            role=Role.ADMIN,
            permissions=["dashboard.read", "trading.control"]
        )
        
        # Should execute successfully
        result = await protected_function(token)
        assert result["success"] is True
        assert result["user"] == "testuser"
    
    @pytest.mark.asyncio
    async def test_require_permission_decorator_failure(self):
        """Test require_permission decorator with missing permission"""
        # Mock function that requires permission
        @require_permission("system.control")
        async def protected_function(auth_token: AuthToken):
            return {"success": True}
        
        # Create token without required permission
        token = AuthToken(
            user_id="user-123",
            username="testuser",
            role=Role.VIEWER,
            permissions=["dashboard.read"]  # Missing system.control
        )
        
        # Should raise AuthorizationError
        with pytest.raises(AuthorizationError) as exc_info:
            await protected_function(token)
        assert "Insufficient permissions" in str(exc_info.value)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.auth_service')
    async def test_verify_api_key_function(self, mock_auth_service):
        """Test verify_api_key function"""
        # Setup mock
        expected_token = AuthToken(
            user_id="user-123",
            username="testuser",
            role=Role.ADMIN,
            permissions=["dashboard.read"]
        )
        mock_auth_service.verify_api_key = AsyncMock(return_value=expected_token)
        
        # Test function
        result = await verify_api_key("test-api-key")
        
        assert result == expected_token
        mock_auth_service.verify_api_key.assert_called_once_with("test-api-key")
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.auth_service')
    async def test_verify_jwt_token_function(self, mock_auth_service):
        """Test verify_jwt_token function"""
        # Setup mock
        expected_token = AuthToken(
            user_id="user-123",
            username="testuser",
            role=Role.ADMIN,
            permissions=["dashboard.read"]
        )
        mock_auth_service.verify_jwt_token = AsyncMock(return_value=expected_token)
        
        # Test function
        result = await verify_jwt_token("test-jwt-token")
        
        assert result == expected_token
        mock_auth_service.verify_jwt_token.assert_called_once_with("test-jwt-token")


class TestAuthExceptions:
    """Test suite for authentication exceptions"""
    
    def test_authentication_error(self):
        """Test AuthenticationError exception"""
        error = AuthenticationError("Invalid credentials")
        assert str(error) == "Invalid credentials"
        assert isinstance(error, Exception)
    
    def test_authorization_error(self):
        """Test AuthorizationError exception"""
        error = AuthorizationError("Insufficient permissions")
        assert str(error) == "Insufficient permissions"
        assert isinstance(error, Exception)
    
    def test_rate_limit_error(self):
        """Test RateLimitError exception"""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, Exception)


class TestAuthServiceIntegration:
    """Integration tests for AuthService"""
    
    @pytest.mark.asyncio
    @patch('src.dashboard.auth.get_config')
    @patch('src.dashboard.auth.bcrypt')
    async def test_full_authentication_flow(self, mock_bcrypt, mock_get_config):
        """Test complete authentication flow"""
        # Setup config
        config = MagicMock()
        config.dashboard.jwt_secret = "test-secret-key"
        config.dashboard.jwt_algorithm = "HS256"
        config.dashboard.jwt_expiry_hours = 24
        config.dashboard.api_keys = {
            "admin-key": {
                "user_id": "admin-user",
                "username": "admin",
                "role": "admin",
                "permissions": ["dashboard.read", "dashboard.write", "trading.control", "system.control"]
            }
        }
        config.dashboard.users = {
            "admin": {
                "password_hash": "$2b$12$hash.for.admin.password",
                "role": "admin",
                "permissions": ["dashboard.read", "dashboard.write", "trading.control", "system.control"]
            }
        }
        config.dashboard.rate_limit_requests_per_minute = 60
        mock_get_config.return_value = config
        mock_bcrypt.checkpw.return_value = True
        
        service = AuthService()
        
        # 1. Login with credentials
        login_request = LoginRequest(username="admin", password="correct-password")
        token_response = await service.login(login_request)
        
        assert token_response.username == "admin"
        assert token_response.role == "admin"
        jwt_token = token_response.access_token
        
        # 2. Verify JWT token
        auth_token = await service.verify_jwt_token(jwt_token)
        assert auth_token is not None
        assert auth_token.username == "admin"
        assert auth_token.role == Role.ADMIN
        
        # 3. Verify API key
        api_token = await service.verify_api_key("admin-key")
        assert api_token is not None
        assert api_token.username == "admin"
        assert api_token.role == Role.ADMIN
        
        # 4. Check permissions
        assert auth_token.has_permission("system.control") is True
        assert api_token.has_permission("trading.control") is True
        
        # 5. Logout
        await service.logout(auth_token.user_id)
        
        # Session should be ended
        is_valid = await service.is_session_valid(auth_token.user_id)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_concurrent_authentication(self):
        """Test concurrent authentication operations"""
        service = AuthService()
        
        # Concurrent API key verification
        tasks = []
        for i in range(10):
            task = asyncio.create_task(service.verify_api_key(f"key-{i}"))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should return None (invalid keys) but not crash
        assert all(result is None for result in results)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test authentication error handling"""
        service = AuthService()
        
        # Test with None inputs
        result = await service.verify_api_key(None)
        assert result is None
        
        result = await service.verify_jwt_token(None)
        assert result is None
        
        # Test with empty strings
        result = await service.verify_api_key("")
        assert result is None
        
        result = await service.verify_jwt_token("")
        assert result is None


class TestGlobalAuthService:
    """Test global auth_service instance"""
    
    def test_global_instance_exists(self):
        """Test that global auth_service instance exists"""
        assert auth_service is not None
        assert isinstance(auth_service, AuthService)
    
    @pytest.mark.asyncio
    async def test_global_instance_functionality(self):
        """Test that global instance works correctly"""
        # Should be able to verify invalid key (returns None)
        token = await auth_service.verify_api_key("invalid-key")
        assert token is None
        
        # Should handle rate limiting
        is_allowed = await auth_service.check_rate_limit("test-ip")
        assert isinstance(is_allowed, bool)