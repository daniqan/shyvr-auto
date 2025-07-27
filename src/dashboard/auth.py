"""
Dashboard Authentication System
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
from dataclasses import dataclass

import jwt
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..utils.config import get_config
from .base import DashboardError
from ..logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity
)

logger = structlog.get_logger()

security = HTTPBearer()


@dataclass
class User:
    """Dashboard user"""
    username: str
    user_id: str
    permissions: Set[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


@dataclass
class Session:
    """User session"""
    session_id: str
    user_id: str
    username: str
    permissions: Set[str]
    created_at: datetime
    expires_at: datetime
    last_activity: datetime


class DashboardAuth:
    """Dashboard authentication and authorization"""
    
    def __init__(self):
        self.config = get_config()
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._api_keys: Dict[str, str] = {}  # api_key -> user_id
        self._rate_limits: Dict[str, list] = {}  # user_id -> [timestamp, ...]
        
        # Default admin user (should be changed in production)
        self._create_default_admin()
    
    def _create_default_admin(self) -> None:
        """Create default admin user"""
        admin_user = User(
            username="admin",
            user_id="admin-001",
            permissions={"read", "write", "admin", "trading"},
            created_at=datetime.utcnow()
        )
        self._users[admin_user.user_id] = admin_user
        
        # Generate API key for admin
        api_key = self._generate_api_key()
        self._api_keys[api_key] = admin_user.user_id
        
        logger.info(
            "Default admin user created",
            username=admin_user.username,
            api_key=api_key
        )
    
    def _generate_api_key(self) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(32)
    
    def _generate_session_id(self) -> str:
        """Generate a secure session ID"""
        return secrets.token_urlsafe(24)
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash a password with salt"""
        return hashlib.pbkdf2_hex(
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000,  # iterations
            32       # length
        )
    
    def _verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify a password against its hash"""
        return hmac.compare_digest(
            self._hash_password(password, salt),
            hashed
        )
    
    def create_user(self, username: str, password: str, 
                    permissions: Set[str]) -> str:
        """Create a new user"""
        user_id = f"user-{secrets.token_hex(8)}"
        
        # In a real implementation, you'd store password hash + salt
        # For this demo, we'll use a simple approach
        user = User(
            username=username,
            user_id=user_id,
            permissions=permissions,
            created_at=datetime.utcnow()
        )
        
        self._users[user_id] = user
        
        # Generate API key
        api_key = self._generate_api_key()
        self._api_keys[api_key] = user_id
        
        # Log user creation
        import asyncio
        asyncio.create_task(activity_logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.CREATE,
            source="dashboard_auth",
            event_type="user_created",
            title=f"New user created: {username}",
            severity=ActivitySeverity.INFO,
            user_id=int(user_id.split('-')[-1], 16) % 10000,  # Convert hex to reasonable int
            metadata={
                "username": username,
                "user_id": user_id,
                "permissions": list(permissions),
                "api_key_generated": True
            }
        ))
        
        logger.info(
            "User created",
            username=username,
            user_id=user_id,
            permissions=list(permissions)
        )
        
        return api_key
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        user_id = self._api_keys.get(api_key)
        if not user_id:
            # Log failed API key authentication
            import asyncio
            asyncio.create_task(activity_logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.VIOLATION,
                source="dashboard_auth",
                event_type="invalid_api_key",
                title="Invalid API key authentication attempt",
                severity=ActivitySeverity.WARNING,
                security_level="elevated",
                metadata={
                    "api_key_prefix": api_key[:8] + "..." if len(api_key) > 8 else "short_key",
                    "reason": "api_key_not_found"
                }
            ))
            return None
        
        user = self._users.get(user_id)
        if not user or not user.is_active:
            # Log inactive user authentication attempt
            import asyncio
            asyncio.create_task(activity_logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.VIOLATION,
                source="dashboard_auth",
                event_type="inactive_user_access",
                title="Authentication attempt with inactive user",
                severity=ActivitySeverity.WARNING,
                user_id=int(user_id.split('-')[-1], 16) % 10000 if user_id else None,
                security_level="elevated",
                metadata={
                    "user_id": user_id,
                    "user_exists": user is not None,
                    "user_active": user.is_active if user else False,
                    "reason": "user_inactive_or_not_found"
                }
            ))
            return None
        
        # Log successful API key authentication
        import asyncio
        asyncio.create_task(activity_logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.ACCESS,
            source="dashboard_auth",
            event_type="api_key_auth_success",
            title=f"Successful API key authentication: {user.username}",
            severity=ActivitySeverity.INFO,
            user_id=int(user_id.split('-')[-1], 16) % 10000,
            metadata={
                "username": user.username,
                "user_id": user_id,
                "permissions": list(user.permissions),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
        ))
        
        return user
    
    def create_session(self, user: User) -> str:
        """Create a new session for user"""
        session_id = self._generate_session_id()
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            username=user.username,
            permissions=user.permissions,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            last_activity=datetime.utcnow()
        )
        
        self._sessions[session_id] = session
        
        # Update user last login
        user.last_login = datetime.utcnow()
        
        # Log session creation
        import asyncio
        asyncio.create_task(activity_logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.LOGIN,
            source="dashboard_auth",
            event_type="session_created",
            title=f"User session created: {user.username}",
            severity=ActivitySeverity.INFO,
            user_id=int(user.user_id.split('-')[-1], 16) % 10000,
            session_id=session_id,
            metadata={
                "username": user.username,
                "user_id": user.user_id,
                "session_id": session_id,
                "expires_at": expires_at.isoformat(),
                "permissions": list(user.permissions)
            }
        ))
        
        logger.info(
            "Session created",
            username=user.username,
            session_id=session_id,
            expires_at=expires_at.isoformat()
        )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        # Check if session is expired
        if datetime.utcnow() > session.expires_at:
            del self._sessions[session_id]
            return None
        
        # Update last activity
        session.last_activity = datetime.utcnow()
        
        return session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session"""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            del self._sessions[session_id]
            
            # Log session revocation
            import asyncio
            asyncio.create_task(activity_logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.LOGOUT,
                source="dashboard_auth",
                event_type="session_revoked",
                title=f"User session revoked: {session.username}",
                severity=ActivitySeverity.INFO,
                user_id=int(session.user_id.split('-')[-1], 16) % 10000,
                session_id=session_id,
                metadata={
                    "username": session.username,
                    "user_id": session.user_id,
                    "session_id": session_id,
                    "session_duration_minutes": int((datetime.utcnow() - session.created_at).total_seconds() / 60)
                }
            ))
            
            logger.info(
                "Session revoked",
                session_id=session_id,
                username=session.username
            )
            return True
        
        return False
    
    def create_jwt_token(self, user: User) -> str:
        """Create JWT token for user"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "permissions": list(user.permissions),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        
        token = jwt.encode(
            payload,
            self.config.security.secret_key,
            algorithm=self.config.security.jwt_algorithm
        )
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.security.secret_key,
                algorithms=[self.config.security.jwt_algorithm]
            )
            return payload
        
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", error=str(e))
            return None
    
    def check_permission(self, user: User, required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user.permissions or "admin" in user.permissions
    
    def check_rate_limit(self, user_id: str, max_requests: int = 100, 
                         window_minutes: int = 60) -> bool:
        """Check rate limit for user"""
        now = time.time()
        window_start = now - (window_minutes * 60)
        
        # Clean old requests
        if user_id in self._rate_limits:
            self._rate_limits[user_id] = [
                timestamp for timestamp in self._rate_limits[user_id]
                if timestamp > window_start
            ]
        else:
            self._rate_limits[user_id] = []
        
        # Check current count
        current_requests = len(self._rate_limits[user_id])
        if current_requests >= max_requests:
            return False
        
        # Add current request
        self._rate_limits[user_id].append(now)
        return True
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        now = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session in self._sessions.items()
            if session.expires_at < now
        ]
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
        
        if expired_sessions:
            logger.info(
                "Expired sessions cleaned up",
                count=len(expired_sessions)
            )
        
        return len(expired_sessions)
    
    def get_user_stats(self) -> Dict:
        """Get user and session statistics"""
        return {
            "total_users": len(self._users),
            "active_users": len([u for u in self._users.values() if u.is_active]),
            "active_sessions": len(self._sessions),
            "api_keys_issued": len(self._api_keys)
        }


# Global auth instance
dashboard_auth = DashboardAuth()


# Dependency functions for FastAPI
async def get_current_user_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current user from API key"""
    try:
        api_key = credentials.credentials
        user = dashboard_auth.authenticate_api_key(api_key)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Check rate limit
        if not dashboard_auth.check_rate_limit(user.user_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        return user
    
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_current_user_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current user from JWT token"""
    try:
        token = credentials.credentials
        payload = dashboard_auth.verify_jwt_token(token)
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        user_id = payload.get("user_id")
        if not user_id or user_id not in dashboard_auth._users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        user = dashboard_auth._users[user_id]
        
        # Check rate limit
        if not dashboard_auth.check_rate_limit(user.user_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        return user
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("JWT authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def dependency(user: User = Depends(get_current_user_api_key)) -> User:
        if not dashboard_auth.check_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return user
    
    return dependency


# Specific permission dependencies
require_read = require_permission("read")
require_write = require_permission("write")
require_admin = require_permission("admin")
require_trading = require_permission("trading")