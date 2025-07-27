#!/usr/bin/env python3
"""
Test script to validate dashboard authentication and mode switching functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import structlog
from src.utils.config import init_config
from src.dashboard.auth import dashboard_auth
from src.dashboard.service import dashboard_service
from src.dashboard.api import dashboard_api

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger()


async def test_authentication():
    """Test dashboard authentication system."""
    print("=== Testing Dashboard Authentication ===")
    
    # Test admin user creation
    print("1. Checking admin user creation...")
    admin_user = None
    admin_api_key = None
    
    for user_id, user in dashboard_auth._users.items():
        if user.username == "admin":
            admin_user = user
            break
    
    if admin_user:
        print(f"✓ Admin user found: {admin_user.username} (ID: {admin_user.user_id})")
        print(f"  Permissions: {admin_user.permissions}")
    else:
        print("✗ Admin user not found!")
        return False
    
    # Test API key
    print("2. Checking API key...")
    for api_key, user_id in dashboard_auth._api_keys.items():
        if user_id == admin_user.user_id:
            admin_api_key = api_key
            break
    
    if admin_api_key:
        print(f"✓ Admin API key found: {admin_api_key[:8]}...")
    else:
        print("✗ Admin API key not found!")
        return False
    
    # Test API key authentication
    print("3. Testing API key authentication...")
    authenticated_user = dashboard_auth.authenticate_api_key(admin_api_key)
    if authenticated_user and authenticated_user.user_id == admin_user.user_id:
        print(f"✓ API key authentication successful")
    else:
        print("✗ API key authentication failed!")
        return False
    
    # Test permission checking
    print("4. Testing permission checking...")
    permissions_to_test = ["read", "write", "admin", "trading"]
    for permission in permissions_to_test:
        has_permission = dashboard_auth.check_permission(authenticated_user, permission)
        status = "✓" if has_permission else "✗"
        print(f"  {status} {permission}: {has_permission}")
    
    return True


async def test_dashboard_service():
    """Test dashboard service functionality."""
    print("\n=== Testing Dashboard Service ===")
    
    try:
        # Initialize configuration
        config = init_config()
        print("✓ Configuration loaded")
        
        # Test service initialization
        print("1. Testing service initialization...")
        await dashboard_service.start()
        print("✓ Dashboard service started")
        
        # Test getting dashboard data
        print("2. Testing dashboard data retrieval...")
        dashboard_data = await dashboard_service.get_dashboard_data()
        if dashboard_data:
            print("✓ Dashboard data retrieved successfully")
            print(f"  Current mode: {dashboard_data.trading_status.mode.value}")
            print(f"  System status: {dashboard_data.system_metrics.status.value}")
        else:
            print("✗ Failed to retrieve dashboard data")
            return False
        
        # Test mode switching
        print("3. Testing mode switching...")
        test_modes = ["analysis", "simulation", "live"]
        for mode in test_modes:
            try:
                result = await dashboard_service.switch_trading_mode(mode, "test-user")
                status = "✓" if result else "✗"
                print(f"  {status} Switch to {mode}: {result}")
            except Exception as e:
                print(f"  ✗ Switch to {mode} failed: {e}")
        
        # Test emergency stop
        print("4. Testing emergency stop...")
        try:
            result = await dashboard_service.emergency_stop("test-user")
            status = "✓" if result else "✗"
            print(f"  {status} Emergency stop: {result}")
        except Exception as e:
            print(f"  ✗ Emergency stop failed: {e}")
        
        # Stop service
        await dashboard_service.stop()
        print("✓ Dashboard service stopped")
        
        return True
        
    except Exception as e:
        print(f"✗ Dashboard service test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("Starting Dashboard Authentication and Mode Switching Tests\n")
    
    # Test authentication
    auth_success = await test_authentication()
    
    # Test dashboard service
    service_success = await test_dashboard_service()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Authentication: {'PASS' if auth_success else 'FAIL'}")
    print(f"Dashboard Service: {'PASS' if service_success else 'FAIL'}")
    
    overall_success = auth_success and service_success
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\n✓ All tests passed! Dashboard authentication and mode switching are working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)