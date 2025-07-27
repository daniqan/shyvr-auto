#!/usr/bin/env python3
"""
Integration test for dashboard authentication and mode switching via HTTP API.
This simulates browser requests to test the full authentication flow.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path - adjust for new location in tests/integration/dashboard/
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
# Add root to path for main app import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
import structlog
from fastapi.testclient import TestClient

# Import the main app
from main import app

logger = structlog.get_logger()


class DashboardIntegrationTest:
    """Integration test for dashboard API."""
    
    def __init__(self):
        self.client = TestClient(app)
        self.api_key = None
        
    def test_api_key_retrieval(self):
        """Test retrieving API key from the auth endpoint."""
        print("1. Testing API key retrieval...")
        
        response = self.client.get("/api/auth/key")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "api_key" in data, "API key not found in response"
        
        self.api_key = data["api_key"]
        print(f"✓ API key retrieved: {self.api_key[:8]}...")
        
    def test_dashboard_data_with_auth(self):
        """Test fetching dashboard data with authentication."""
        print("2. Testing dashboard data retrieval with authentication...")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = self.client.get("/dashboard/data", headers=headers)
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "system_metrics" in data, "System metrics not found"
        assert "trading_status" in data, "Trading status not found"
        assert "portfolio_status" in data, "Portfolio status not found"
        
        print(f"✓ Dashboard data retrieved successfully")
        print(f"  Current mode: {data['trading_status']['mode']}")
        
    def test_dashboard_data_without_auth(self):
        """Test dashboard data retrieval fails without authentication."""
        print("3. Testing dashboard data retrieval without authentication...")
        
        response = self.client.get("/dashboard/data")
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        
        print("✓ Unauthenticated request properly rejected")
        
    def test_mode_switching_with_auth(self):
        """Test mode switching with proper authentication."""
        print("4. Testing mode switching with authentication...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        test_modes = ["analysis", "simulation", "live", "analysis"]
        
        for mode in test_modes:
            print(f"  Testing switch to {mode}...")
            
            response = self.client.post(
                "/dashboard/trading/mode", 
                headers=headers,
                json={"mode": mode}
            )
            
            assert response.status_code == 200, f"Mode switch to {mode} failed: {response.status_code}"
            
            data = response.json()
            assert data["success"] == True, f"Mode switch to {mode} returned success=False"
            
            print(f"    ✓ Successfully switched to {mode}")
            
    def test_mode_switching_without_auth(self):
        """Test mode switching fails without authentication."""
        print("5. Testing mode switching without authentication...")
        
        response = self.client.post(
            "/dashboard/trading/mode",
            json={"mode": "simulation"}
        )
        
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("✓ Unauthenticated mode switch properly rejected")
        
    def test_invalid_mode_switching(self):
        """Test invalid mode switching."""
        print("6. Testing invalid mode switching...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = self.client.post(
            "/dashboard/trading/mode",
            headers=headers,
            json={"mode": "invalid_mode"}
        )
        
        # Should return 422 for validation error
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"
        print("✓ Invalid mode properly rejected")
        
    def test_emergency_stop_with_auth(self):
        """Test emergency stop with authentication."""
        print("7. Testing emergency stop with authentication...")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = self.client.post("/dashboard/trading/emergency-stop", headers=headers)
        
        assert response.status_code == 200, f"Emergency stop failed: {response.status_code}"
        
        data = response.json()
        assert data["success"] == True, "Emergency stop returned success=False"
        
        print("✓ Emergency stop executed successfully")
        
    def test_risk_limits_update_with_auth(self):
        """Test risk limits update with authentication."""
        print("8. Testing risk limits update with authentication...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        risk_limits = {
            "max_position_size_usd": 1000.0,
            "max_daily_loss_usd": 500.0,
            "max_drawdown_pct": 10.0,
            "stop_loss_pct": 5.0
        }
        
        response = self.client.put(
            "/dashboard/trading/risk-limits",
            headers=headers,
            json=risk_limits
        )
        
        assert response.status_code == 200, f"Risk limits update failed: {response.status_code}"
        
        data = response.json()
        assert data["success"] == True, "Risk limits update returned success=False"
        
        print("✓ Risk limits updated successfully")
        
    def test_system_health_endpoint(self):
        """Test system health endpoint."""
        print("9. Testing system health endpoint...")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = self.client.get("/dashboard/system/health", headers=headers)
        
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        
        data = response.json()
        assert "overall_status" in data, "Overall status not found"
        assert "component_statuses" in data, "Component statuses not found"
        
        print(f"✓ System health check passed: {data['overall_status']}")
        
    def run_all_tests(self):
        """Run all integration tests."""
        print("=== Dashboard Integration Tests ===\n")
        
        try:
            self.test_api_key_retrieval()
            self.test_dashboard_data_with_auth()
            self.test_dashboard_data_without_auth()
            self.test_mode_switching_with_auth()
            self.test_mode_switching_without_auth()
            self.test_invalid_mode_switching()
            self.test_emergency_stop_with_auth()
            self.test_risk_limits_update_with_auth()
            self.test_system_health_endpoint()
            
            print("\n=== All Tests Passed! ===")
            print("✓ Authentication working correctly")
            print("✓ Mode switching functional")
            print("✓ API endpoints properly secured")
            print("✓ Error handling working as expected")
            
            return True
            
        except AssertionError as e:
            print(f"\n✗ Test failed: {e}")
            return False
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            return False


def main():
    """Run integration tests."""
    test_runner = DashboardIntegrationTest()
    success = test_runner.run_all_tests()
    
    if success:
        print("\n🎉 Dashboard authentication and mode switching are working perfectly!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)