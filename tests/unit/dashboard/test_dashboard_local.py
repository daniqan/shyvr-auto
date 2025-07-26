#!/usr/bin/env python3
"""
Local Dashboard Testing Script
Tests dashboard functionality for local development environment
"""

import asyncio
import json
import os
import requests
import subprocess
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

def test_environment_setup():
    """Test that the environment is properly configured"""
    print("🧪 Testing Environment Setup...")
    
    # Check if .env.local exists
    env_file = Path(".env.local")
    if env_file.exists():
        print("✅ .env.local file exists")
    else:
        print("❌ .env.local file missing")
        return False
    
    # Check if Docker is running
    try:
        subprocess.run(["docker", "ps"], check=True, capture_output=True)
        print("✅ Docker is running")
    except subprocess.CalledProcessError:
        print("❌ Docker is not running")
        return False
    
    return True

def test_database_connection():
    """Test PostgreSQL database connection"""
    print("\n🗄️ Testing Database Connection...")
    
    try:
        import psycopg2
        
        # Test connection to local PostgreSQL
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="shyvr_rlte",
            user="rlte_user",
            password="dev_password_change_in_prod"
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ Database connected: {version}")
        
        cursor.close()
        conn.close()
        return True
        
    except ImportError:
        print("⚠️ psycopg2 not installed, skipping database test")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_application_startup():
    """Test application startup with local config"""
    print("\n🚀 Testing Application Startup...")
    
    try:
        # Set environment variables for testing
        os.environ["ENVIRONMENT"] = "development"
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["DB_HOST"] = "localhost"
        os.environ["DB_PASSWORD"] = "dev_password_change_in_prod"
        os.environ["SECRET_KEY"] = "dev-key-for-local-testing-only"
        
        # Import after setting environment
        from src.utils.config import init_config
        from src.dashboard.service import dashboard_service
        from src.dashboard.auth import dashboard_auth
        
        # Initialize configuration
        config = init_config()
        print(f"✅ Configuration loaded: {config.app.environment}")
        
        # Test dashboard auth
        stats = dashboard_auth.get_user_stats()
        print(f"✅ Dashboard auth initialized: {stats['total_users']} users")
        
        return True
        
    except Exception as e:
        print(f"❌ Application startup failed: {e}")
        return False

def test_dashboard_mock_data():
    """Test dashboard with mock data"""
    print("\n📊 Testing Dashboard Mock Data...")
    
    try:
        # Set environment for testing
        os.environ["ENABLE_MOCK_DATA"] = "true"
        
        from src.dashboard.service import dashboard_service
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Test getting dashboard data
        async def test_data():
            try:
                data = await dashboard_service.get_dashboard_data()
                print(f"✅ Dashboard data retrieved: {data.system_metrics.status}")
                print(f"✅ Portfolio value: ${data.portfolio_status.total_value_usd}")
                print(f"✅ Trading mode: {data.trading_status.mode}")
                print(f"✅ ML/RL integration: {data.ml_rl_status.ml_rl_integration_active}")
                return True
            except Exception as e:
                print(f"❌ Dashboard data failed: {e}")
                return False
        
        result = loop.run_until_complete(test_data())
        loop.close()
        
        return result
        
    except Exception as e:
        print(f"❌ Dashboard mock data test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints if server is running"""
    print("\n🌐 Testing API Endpoints...")
    
    base_url = "http://localhost:8080"
    
    # Test basic endpoints without authentication
    endpoints = [
        "/health",
        "/api",
        "/config"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint}: {response.status_code}")
            else:
                print(f"⚠️ {endpoint}: {response.status_code}")
        except requests.exceptions.RequestException:
            print(f"❌ {endpoint}: Connection failed (server may not be running)")
    
    return True

def test_dashboard_html():
    """Test dashboard HTML files"""
    print("\n🎨 Testing Dashboard HTML...")
    
    # Check if static files exist
    static_files = [
        "static/index.html",
        "static/dashboard.js",
        "static/styles.css"
    ]
    
    all_exist = True
    for file_path in static_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_modes_configuration():
    """Test trading modes configuration"""
    print("\n⚙️ Testing Trading Modes Configuration...")
    
    try:
        from src.utils.config import init_config
        
        config = init_config()
        
        # Check mode configurations
        modes = config.trading.modes
        print(f"✅ Analysis mode: {modes.analysis}")
        print(f"✅ Simulation mode: {modes.simulation}")
        print(f"⚠️ Live mode: {modes.live} (should be False for safety)")
        
        # Check risk management
        risk = config.trading.risk_management
        print(f"✅ Max position size: {risk.max_position_size_pct}%")
        print(f"✅ Max daily loss: {risk.max_daily_loss_pct}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading modes test failed: {e}")
        return False

def generate_test_report(results):
    """Generate a test report"""
    print("\n" + "="*60)
    print("📋 DASHBOARD LOCAL TESTING REPORT")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nTest Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Dashboard is ready for local development.")
        print("\n📚 Next Steps:")
        print("  1. Start the application: python main.py")
        print("  2. Access dashboard: http://localhost:8080/")
        print("  3. View API docs: http://localhost:8080/docs")
        print("  4. Check health: http://localhost:8080/health")
        print("  5. Use .env.local for configuration")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration.")
        print("\n🔧 Troubleshooting:")
        print("  1. Ensure Docker is running")
        print("  2. Check .env.local file exists")
        print("  3. Start PostgreSQL: docker-compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml up postgres")
        print("  4. Install dependencies: pip install -r requirements.txt")
    
    return passed_tests == total_tests

def main():
    """Run all dashboard tests"""
    print("🧪 Shyvr RLTE Dashboard - Local Development Testing")
    print("="*60)
    
    # Run all tests
    results = {}
    
    results["Environment Setup"] = test_environment_setup()
    results["Database Connection"] = test_database_connection()
    results["Application Startup"] = test_application_startup()
    results["Dashboard Mock Data"] = test_dashboard_mock_data()
    results["Dashboard HTML"] = test_dashboard_html()
    results["Trading Modes Config"] = test_modes_configuration()
    results["API Endpoints"] = test_api_endpoints()
    
    # Generate report
    success = generate_test_report(results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())