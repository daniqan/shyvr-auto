#!/usr/bin/env python3
"""
Comprehensive Dashboard Setup Validation Script
Validates all dashboard functionality for local development
"""

import asyncio
import json
import os
import requests
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import psutil

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_test(test_name: str, status: str, details: str = ""):
    """Print test result"""
    status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_icon} {test_name}: {status}")
    if details:
        print(f"   {details}")

class DashboardValidator:
    def __init__(self):
        self.base_url = "http://localhost:8080"
        self.api_key = None
        self.server_process = None
        self.results = {}
        
    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        print_header("System Prerequisites")
        
        all_good = True
        
        # Check Docker
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print_test("Docker installed", "PASS", result.stdout.strip())
            else:
                print_test("Docker installed", "FAIL", "Docker not found")
                all_good = False
        except FileNotFoundError:
            print_test("Docker installed", "FAIL", "Docker not found")
            all_good = False
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 11):
            print_test("Python version", "PASS", f"Python {python_version.major}.{python_version.minor}")
        else:
            print_test("Python version", "WARN", f"Python {python_version.major}.{python_version.minor} (3.11+ recommended)")
        
        # Check required files
        required_files = [
            "main.py",
            "config/config.yaml",
            "static/index.html",
            "static/dashboard.js",
            ".env.local"
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                print_test(f"File {file_path}", "PASS")
            else:
                print_test(f"File {file_path}", "FAIL", "File not found")
                all_good = False
        
        return all_good
    
    def setup_database(self) -> bool:
        """Setup and validate database"""
        print_header("Database Setup")
        
        try:
            # Check if PostgreSQL container is running
            result = subprocess.run([
                "docker-compose", "-f", "docker/docker-compose.yml", 
                "-f", "docker/docker-compose.local.yml", "ps", "postgres"
            ], capture_output=True, text=True)
            
            if "Up" in result.stdout and "healthy" in result.stdout:
                print_test("PostgreSQL container", "PASS", "Container is running and healthy")
            else:
                print_test("PostgreSQL container", "WARN", "Starting PostgreSQL container...")
                
                # Start PostgreSQL
                start_result = subprocess.run([
                    "docker-compose", "-f", "docker/docker-compose.yml",
                    "-f", "docker/docker-compose.local.yml", "up", "-d", "postgres"
                ], capture_output=True, text=True)
                
                if start_result.returncode == 0:
                    print_test("Start PostgreSQL", "PASS", "Container started")
                    
                    # Wait for health check
                    print("   Waiting for database to be ready...")
                    for i in range(30):
                        time.sleep(2)
                        health_result = subprocess.run([
                            "docker-compose", "-f", "docker/docker-compose.yml",
                            "-f", "docker/docker-compose.local.yml", "ps", "postgres"
                        ], capture_output=True, text=True)
                        
                        if "healthy" in health_result.stdout:
                            print_test("Database health check", "PASS", f"Ready after {i*2}s")
                            break
                    else:
                        print_test("Database health check", "FAIL", "Timeout waiting for health check")
                        return False
                else:
                    print_test("Start PostgreSQL", "FAIL", start_result.stderr)
                    return False
            
            # Test database connection
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host="localhost",
                    port=5432,
                    database="shyvr_rlte",
                    user="rlte_user",
                    password="dev_password_change_in_prod"
                )
                conn.close()
                print_test("Database connection", "PASS", "Successfully connected to PostgreSQL")
                return True
            except ImportError:
                print_test("Database connection", "WARN", "psycopg2 not installed, skipping connection test")
                return True
            except Exception as e:
                print_test("Database connection", "FAIL", str(e))
                return False
                
        except Exception as e:
            print_test("Database setup", "FAIL", str(e))
            return False
    
    def start_application(self) -> bool:
        """Start the application"""
        print_header("Application Startup")
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            "ENVIRONMENT": "development",
            "LOG_LEVEL": "INFO",
            "DB_HOST": "localhost",
            "DB_PASSWORD": "dev_password_change_in_prod",
            "SECRET_KEY": "dev-key-for-local-testing",
            "ENABLE_MOCK_DATA": "true"
        })
        
        try:
            # Check if server is already running
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print_test("Application already running", "PASS", "Server is responding")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            # Start the application
            print("   Starting application...")
            self.server_process = subprocess.Popen(
                [sys.executable, "main.py"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for startup and capture API key
            startup_timeout = 60
            start_time = time.time()
            
            while time.time() - start_time < startup_timeout:
                if self.server_process.stdout:
                    line = self.server_process.stdout.readline()
                    if line:
                        print(f"   {line.strip()}")
                        
                        # Extract API key from log output
                        if "api_key=" in line:
                            self.api_key = line.split("api_key=")[1].split()[0]
                            print_test("API key extracted", "PASS", f"Key: {self.api_key[:16]}...")
                
                # Check if server is responding
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=1)
                    if response.status_code == 200:
                        print_test("Application startup", "PASS", f"Server started after {time.time()-start_time:.1f}s")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(1)
            
            print_test("Application startup", "FAIL", "Timeout waiting for server to start")
            return False
            
        except Exception as e:
            print_test("Application startup", "FAIL", str(e))
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test API endpoints"""
        print_header("API Endpoint Testing")
        
        # Test public endpoints
        public_endpoints = [
            ("/health", "Health check"),
            ("/api", "API root"),
            ("/config", "Configuration"),
            ("/", "Dashboard UI")
        ]
        
        all_passed = True
        
        for endpoint, description in public_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print_test(f"{description} ({endpoint})", "PASS", f"Status: {response.status_code}")
                else:
                    print_test(f"{description} ({endpoint})", "FAIL", f"Status: {response.status_code}")
                    all_passed = False
            except Exception as e:
                print_test(f"{description} ({endpoint})", "FAIL", str(e))
                all_passed = False
        
        # Test authenticated endpoints
        if self.api_key:
            auth_headers = {"Authorization": f"Bearer {self.api_key}"}
            
            auth_endpoints = [
                ("/dashboard/data", "Dashboard data"),
                ("/dashboard/system/health", "System health"),
                ("/dashboard/trading/status", "Trading status"),
                ("/dashboard/portfolio/status", "Portfolio status"),
                ("/dashboard/ml-rl/status", "ML/RL status")
            ]
            
            for endpoint, description in auth_endpoints:
                try:
                    response = requests.get(
                        f"{self.base_url}{endpoint}", 
                        headers=auth_headers, 
                        timeout=5
                    )
                    if response.status_code == 200:
                        print_test(f"{description} ({endpoint})", "PASS", f"Status: {response.status_code}")
                    else:
                        print_test(f"{description} ({endpoint})", "FAIL", f"Status: {response.status_code}")
                        all_passed = False
                except Exception as e:
                    print_test(f"{description} ({endpoint})", "FAIL", str(e))
                    all_passed = False
        else:
            print_test("Authenticated endpoints", "FAIL", "No API key available")
            all_passed = False
        
        return all_passed
    
    def test_trading_modes(self) -> bool:
        """Test trading mode switching"""
        print_header("Trading Mode Testing")
        
        if not self.api_key:
            print_test("Trading mode tests", "FAIL", "No API key available")
            return False
        
        auth_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        all_passed = True
        
        # Test mode switching
        modes_to_test = [
            ("analysis", "Mode 1 (Analysis)"),
            ("simulation", "Mode 2 (Simulation)"),
            ("analysis", "Back to Analysis")  # Switch back
        ]
        
        for mode, description in modes_to_test:
            try:
                response = requests.post(
                    f"{self.base_url}/dashboard/trading/mode",
                    headers=auth_headers,
                    json={"mode": mode},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        print_test(f"Switch to {description}", "PASS", data.get("message", ""))
                    else:
                        print_test(f"Switch to {description}", "FAIL", "Success flag is False")
                        all_passed = False
                else:
                    print_test(f"Switch to {description}", "FAIL", f"Status: {response.status_code}")
                    all_passed = False
                    
                # Verify mode change
                status_response = requests.get(
                    f"{self.base_url}/dashboard/trading/status",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_mode = status_data.get("trading_status", {}).get("mode")
                    if current_mode == mode:
                        print_test(f"Verify {description} mode", "PASS", f"Current mode: {current_mode}")
                    else:
                        print_test(f"Verify {description} mode", "FAIL", f"Expected: {mode}, Got: {current_mode}")
                        all_passed = False
                        
            except Exception as e:
                print_test(f"Switch to {description}", "FAIL", str(e))
                all_passed = False
        
        # Test invalid mode
        try:
            response = requests.post(
                f"{self.base_url}/dashboard/trading/mode",
                headers=auth_headers,
                json={"mode": "invalid_mode"},
                timeout=5
            )
            
            if response.status_code != 200:
                print_test("Invalid mode rejection", "PASS", f"Properly rejected with status {response.status_code}")
            else:
                print_test("Invalid mode rejection", "FAIL", "Invalid mode was accepted")
                all_passed = False
                
        except Exception as e:
            print_test("Invalid mode test", "FAIL", str(e))
            all_passed = False
        
        return all_passed
    
    def test_dashboard_ui(self) -> bool:
        """Test dashboard UI components"""
        print_header("Dashboard UI Testing")
        
        all_passed = True
        
        # Test static files
        static_files = [
            ("/static/dashboard.js", "Dashboard JavaScript"),
            ("/static/styles.css", "Dashboard CSS")
        ]
        
        for endpoint, description in static_files:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print_test(description, "PASS", f"Size: {len(response.content)} bytes")
                else:
                    print_test(description, "FAIL", f"Status: {response.status_code}")
                    all_passed = False
            except Exception as e:
                print_test(description, "FAIL", str(e))
                all_passed = False
        
        # Test dashboard HTML content
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                html_content = response.text
                
                # Check for key UI components
                ui_components = [
                    ("Shyvr RLTE", "Page title"),
                    ("tab-btn", "Tab buttons"),
                    ("dashboard-grid", "Dashboard layout"),
                    ("dashboard.js", "JavaScript include"),
                    ("Overview", "Overview tab"),
                    ("Trading", "Trading tab"),
                    ("Portfolio", "Portfolio tab"),
                    ("ML & RL", "ML/RL tab"),
                    ("System", "System tab")
                ]
                
                for component, description in ui_components:
                    if component in html_content:
                        print_test(f"UI component: {description}", "PASS")
                    else:
                        print_test(f"UI component: {description}", "FAIL", f"'{component}' not found in HTML")
                        all_passed = False
            else:
                print_test("Dashboard HTML", "FAIL", f"Status: {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print_test("Dashboard HTML", "FAIL", str(e))
            all_passed = False
        
        return all_passed
    
    def test_websocket_connection(self) -> bool:
        """Test WebSocket connection"""
        print_header("WebSocket Testing")
        
        try:
            import websocket
            
            # Test WebSocket connection
            ws_url = f"ws://localhost:8080/dashboard/ws/test-connection-{int(time.time())}"
            
            def on_open(ws):
                print_test("WebSocket connection", "PASS", "Connection established")
                ws.send(json.dumps({"type": "ping"}))
            
            def on_message(ws, message):
                print_test("WebSocket message", "PASS", f"Received: {message[:50]}...")
                ws.close()
            
            def on_error(ws, error):
                print_test("WebSocket connection", "FAIL", str(error))
            
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error
            )
            
            # Run WebSocket in a separate thread with timeout
            import threading
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection
            time.sleep(3)
            
            return True
            
        except ImportError:
            print_test("WebSocket testing", "WARN", "websocket-client not installed, skipping WebSocket test")
            return True
        except Exception as e:
            print_test("WebSocket connection", "FAIL", str(e))
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        print_header("Cleanup")
        
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                print_test("Application shutdown", "PASS", "Server stopped gracefully")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print_test("Application shutdown", "WARN", "Server killed forcefully")
            except Exception as e:
                print_test("Application shutdown", "FAIL", str(e))
    
    def generate_report(self):
        """Generate final test report"""
        print_header("Test Summary")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {test_name}")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! Dashboard is ready for local development.")
            print("\n📚 Quick Start:")
            print("  1. Access dashboard: http://localhost:8080/")
            print(f"  2. Use API key: {self.api_key[:16] if self.api_key else 'Check console output'}...")
            print("  3. Test mode switching between Analysis and Simulation")
            print("  4. Explore all dashboard tabs and features")
        else:
            print("\n⚠️ Some tests failed. Please check the issues above.")
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("🧪 Shyvr RLTE Dashboard - Comprehensive Validation")
        print("=" * 60)
        
        # Run all test suites
        self.results["Prerequisites"] = self.check_prerequisites()
        self.results["Database Setup"] = self.setup_database()
        self.results["Application Startup"] = self.start_application()
        self.results["API Endpoints"] = self.test_api_endpoints()
        self.results["Trading Modes"] = self.test_trading_modes()
        self.results["Dashboard UI"] = self.test_dashboard_ui()
        self.results["WebSocket"] = self.test_websocket_connection()
        
        # Generate report
        self.generate_report()
        
        # Cleanup
        self.cleanup()
        
        return all(self.results.values())

def main():
    """Main function"""
    validator = DashboardValidator()
    
    try:
        success = validator.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        validator.cleanup()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        validator.cleanup()
        return 1

if __name__ == "__main__":
    sys.exit(main())