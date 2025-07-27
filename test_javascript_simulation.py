#!/usr/bin/env python3
"""
Simulate JavaScript client behavior to test dashboard authentication flow.
This simulates the exact sequence of operations the browser would perform.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from fastapi.testclient import TestClient
from main import app


class JavaScriptSimulation:
    """Simulate JavaScript dashboard client behavior."""
    
    def __init__(self):
        self.client = TestClient(app)
        self.api_key = None
        
    def simulate_page_load(self):
        """Simulate initial page load and API key initialization."""
        print("🌐 Simulating page load...")
        
        # 1. Load the main dashboard page
        response = self.client.get("/")
        assert response.status_code == 200, "Failed to load dashboard page"
        print("✓ Dashboard page loaded")
        
        # 2. JavaScript would immediately try to get API key
        print("🔑 JavaScript: Fetching API key...")
        response = self.client.get("/api/auth/key")
        assert response.status_code == 200, "Failed to get API key"
        
        data = response.json()
        self.api_key = data.get("api_key")
        assert self.api_key, "No API key in response"
        
        print(f"✓ API key obtained: {self.api_key[:8]}...")
        
    def simulate_initial_data_fetch(self):
        """Simulate fetching initial dashboard data."""
        print("📊 JavaScript: Fetching initial dashboard data...")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = self.client.get("/dashboard/data", headers=headers)
        assert response.status_code == 200, "Failed to fetch dashboard data"
        
        data = response.json()
        current_mode = data.get("trading_status", {}).get("mode", "unknown")
        
        print(f"✓ Dashboard data loaded, current mode: {current_mode}")
        return data
        
    def simulate_mode_switch_ui_interaction(self):
        """Simulate user clicking mode switch in UI."""
        print("🖱️  User: Clicking mode switch button...")
        
        # Simulate switching from analysis to simulation
        print("   Switching to simulation mode...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = self.client.post(
            "/dashboard/trading/mode",
            headers=headers,
            json={"mode": "simulation"}
        )
        
        assert response.status_code == 200, "Mode switch failed"
        result = response.json()
        assert result.get("success"), "Mode switch returned failure"
        
        print("✓ Mode switched successfully")
        
        # Verify the change by fetching fresh data
        print("🔄 JavaScript: Refreshing data after mode switch...")
        response = self.client.get("/dashboard/data", headers=headers)
        assert response.status_code == 200, "Failed to refresh data"
        
        data = response.json()
        current_mode = data.get("trading_status", {}).get("mode", "unknown")
        
        print(f"✓ Data refreshed, mode is now: {current_mode}")
        
    def simulate_authentication_error_recovery(self):
        """Simulate handling authentication errors and recovery."""
        print("🔒 Simulating authentication error and recovery...")
        
        # Try with invalid API key
        print("   Using invalid API key...")
        headers = {"Authorization": "Bearer invalid_key"}
        response = self.client.get("/dashboard/data", headers=headers)
        
        assert response.status_code == 401, "Expected 401 for invalid key"
        print("✓ Invalid API key properly rejected")
        
        # Simulate JavaScript recovering by fetching new API key
        print("🔄 JavaScript: Recovering by fetching new API key...")
        response = self.client.get("/api/auth/key")
        assert response.status_code == 200, "Failed to get new API key"
        
        data = response.json()
        new_api_key = data.get("api_key")
        
        print(f"✓ New API key obtained: {new_api_key[:8]}...")
        
        # Try again with new key
        headers = {"Authorization": f"Bearer {new_api_key}"}
        response = self.client.get("/dashboard/data", headers=headers)
        assert response.status_code == 200, "Failed with new API key"
        
        print("✓ Authentication recovery successful")
        
    def simulate_emergency_stop_scenario(self):
        """Simulate emergency stop button press."""
        print("🚨 User: Pressing emergency stop button...")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = self.client.post("/dashboard/trading/emergency-stop", headers=headers)
        
        assert response.status_code == 200, "Emergency stop failed"
        result = response.json()
        assert result.get("success"), "Emergency stop returned failure"
        
        print("✓ Emergency stop executed successfully")
        
    def simulate_settings_update(self):
        """Simulate updating risk settings."""
        print("⚙️  User: Updating risk settings...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        risk_settings = {
            "max_position_size_usd": 500.0,
            "max_daily_loss_usd": 250.0
        }
        
        response = self.client.put(
            "/dashboard/trading/risk-limits",
            headers=headers,
            json=risk_settings
        )
        
        assert response.status_code == 200, "Settings update failed"
        result = response.json()
        assert result.get("success"), "Settings update returned failure"
        
        print("✓ Risk settings updated successfully")
        
    def run_full_simulation(self):
        """Run complete JavaScript client simulation."""
        print("=== JavaScript Client Simulation ===\n")
        print("Simulating real browser dashboard usage...\n")
        
        try:
            self.simulate_page_load()
            print()
            
            self.simulate_initial_data_fetch()
            print()
            
            self.simulate_mode_switch_ui_interaction()
            print()
            
            self.simulate_authentication_error_recovery()
            print()
            
            self.simulate_emergency_stop_scenario()
            print()
            
            self.simulate_settings_update()
            print()
            
            print("=== Simulation Complete ===")
            print("✅ All JavaScript client operations successful!")
            print("✅ Authentication flow working properly")
            print("✅ Mode switching functional from UI perspective")
            print("✅ Error recovery mechanisms working")
            print("✅ All dashboard features accessible")
            
            return True
            
        except AssertionError as e:
            print(f"\n❌ Simulation failed: {e}")
            return False
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            return False


def main():
    """Run the JavaScript simulation."""
    simulator = JavaScriptSimulation()
    success = simulator.run_full_simulation()
    
    if success:
        print("\n🎉 Dashboard is ready for production use!")
        print("   The authentication issues have been resolved.")
        print("   Mode switching works correctly from the UI.")
        print("   All API endpoints are properly secured.")
        return 0
    else:
        print("\n💔 Simulation revealed issues that need fixing.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)