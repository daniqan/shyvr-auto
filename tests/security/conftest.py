"""
Security tests configuration and fixtures
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with proper configuration"""
    # Set environment variables for testing
    os.environ.update({
        "SECRET_KEY": "test_secret_key_for_security_testing_at_least_16_chars",
        "DATABASE_URL": "sqlite:///:memory:",
        "ENVIRONMENT": "test"
    })


@pytest.fixture
def mock_app():
    """Create mock FastAPI application for testing"""
    app = FastAPI()
    
    # Add mock routes that would exist in real app
    @app.get("/api/health")
    def health():
        return {"status": "ok"}
    
    @app.get("/api/portfolio/positions")
    def get_positions():
        return {"positions": []}
    
    @app.post("/api/trading/execute")
    def execute_trade():
        return {"status": "executed"}
    
    @app.get("/api/admin/users")
    def get_admin_users():
        return {"users": []}
    
    return app


@pytest.fixture
def test_client(mock_app):
    """Create test client with mock app"""
    return TestClient(mock_app)


@pytest.fixture
def mock_auth_headers():
    """Create mock authentication headers"""
    return {"Authorization": "Bearer test_api_key_for_security_testing"}


@pytest.fixture
def mock_dashboard_auth():
    """Create mock dashboard auth system"""
    auth_mock = MagicMock()
    auth_mock.create_user.return_value = "test_api_key_123"
    auth_mock.authenticate_api_key.return_value = MagicMock()
    auth_mock.check_rate_limit.return_value = True
    return auth_mock


@pytest.fixture
def mock_activity_logger():
    """Create mock activity logger"""
    logger_mock = MagicMock()
    logger_mock.log_activity = MagicMock()
    return logger_mock


@pytest.fixture(autouse=True)
def patch_imports():
    """Patch imports to avoid configuration issues"""
    with patch.dict('sys.modules', {
        'src.dashboard.auth': MagicMock(),
        'src.dashboard.api': MagicMock(),
        'src.activity_logging.activity_logger': MagicMock(),
        'src.utils.config': MagicMock(),
    }):
        yield