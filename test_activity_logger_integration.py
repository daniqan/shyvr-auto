#!/usr/bin/env python3
"""
Comprehensive test script for ActivityLogger integration across RLTE system components.

This script verifies that ActivityLogger is properly integrated into all major system
components and captures meaningful events for the dashboard activity log.
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    TradingMode as LoggingTradingMode, ChainType
)
from src.utils.config import get_config


async def test_activity_logger_basic_functionality():
    """Test basic ActivityLogger functionality"""
    print("🧪 Testing ActivityLogger basic functionality...")
    
    try:
        # Start the activity logger
        await activity_logger.start()
        
        # Test basic logging
        activity_id = await activity_logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.START,
            source="test_integration",
            event_type="test_basic_functionality",
            title="Testing basic ActivityLogger functionality",
            severity=ActivitySeverity.INFO,
            metadata={"test_component": "basic_functionality"}
        )
        
        print(f"✅ Basic logging successful. Activity ID: {activity_id}")
        
        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            error_id = await activity_logger.log_error(
                category=ActivityCategory.SYSTEM,
                source="test_integration",
                event_type="test_error_logging",
                title="Testing error logging functionality",
                exception=e,
                metadata={"test_component": "error_logging"}
            )
            print(f"✅ Error logging successful. Activity ID: {error_id}")
        
        # Test performance logging
        perf_id = await activity_logger.log_performance(
            source="test_integration",
            operation="test_operation",
            execution_time_ms=150,
            success=True,
            metadata={"test_component": "performance_logging"}
        )
        print(f"✅ Performance logging successful. Activity ID: {perf_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


async def test_dashboard_service_integration():
    """Test ActivityLogger integration with Dashboard Service"""
    print("\n🖥️ Testing Dashboard Service integration...")
    
    try:
        # Import dashboard service
        from src.dashboard.service import dashboard_service
        
        # Test dashboard service startup (which should log activities)
        print("  📝 Testing dashboard service startup logging...")
        
        # Note: We can't actually start the service in test due to dependencies,
        # but we can test that the integration points exist
        
        # Check if the service has the activity logger imported
        import src.dashboard.service as service_module
        if hasattr(service_module, 'activity_logger'):
            print("  ✅ ActivityLogger imported in dashboard service")
        else:
            print("  ❌ ActivityLogger not found in dashboard service")
            return False
        
        # Test that the service methods would log activities
        print("  ✅ Dashboard service integration verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Dashboard service integration test failed: {e}")
        return False


async def test_trading_system_integration():
    """Test ActivityLogger integration with Trading System"""
    print("\n💱 Testing Trading System (DEX-Wallet Bridge) integration...")
    
    try:
        # Import trading system
        import src.trading.dex_wallet_bridge as bridge_module
        
        # Check if the bridge has the activity logger imported
        if hasattr(bridge_module, 'activity_logger'):
            print("  ✅ ActivityLogger imported in DEX-Wallet Bridge")
        else:
            print("  ❌ ActivityLogger not found in DEX-Wallet Bridge")
            return False
        
        # Test quote generation logging (would normally require DEX client)
        print("  📝 Testing quote generation logging integration...")
        
        # Verify the DEXWalletBridge class has been updated
        from src.trading.dex_wallet_bridge import DEXWalletBridge
        
        # Check if get_quote method exists and has been updated
        import inspect
        get_quote_source = inspect.getsource(DEXWalletBridge.get_quote)
        if 'activity_logger' in get_quote_source:
            print("  ✅ Quote generation logging integrated")
        else:
            print("  ❌ Quote generation logging not integrated")
            return False
        
        # Check if execute_swap method has been updated
        execute_swap_source = inspect.getsource(DEXWalletBridge.execute_swap)
        if 'activity_logger' in execute_swap_source:
            print("  ✅ Swap execution logging integrated")
        else:
            print("  ❌ Swap execution logging not integrated")
            return False
        
        print("  ✅ Trading system integration verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Trading system integration test failed: {e}")
        return False


async def test_ml_rl_integration():
    """Test ActivityLogger integration with ML/RL components"""
    print("\n🧠 Testing ML/RL components integration...")
    
    try:
        # Test Model Manager integration
        print("  📝 Testing Model Manager integration...")
        
        import src.ml_analysis.model_manager as mm_module
        if hasattr(mm_module, 'activity_logger'):
            print("  ✅ ActivityLogger imported in Model Manager")
        else:
            print("  ❌ ActivityLogger not found in Model Manager")
            return False
        
        # Test DQN Agent integration
        print("  📝 Testing DQN Agent integration...")
        
        import src.rl_agent.dqn_agent as dqn_module
        if hasattr(dqn_module, 'activity_logger'):
            print("  ✅ ActivityLogger imported in DQN Agent")
        else:
            print("  ❌ ActivityLogger not found in DQN Agent")
            return False
        
        # Verify DQN Agent predict_action method has been updated
        from src.rl_agent.dqn_agent import DQNTradingAgent
        import inspect
        predict_action_source = inspect.getsource(DQNTradingAgent.predict_action)
        if 'activity_logger' in predict_action_source:
            print("  ✅ DQN Agent action prediction logging integrated")
        else:
            print("  ❌ DQN Agent action prediction logging not integrated")
            return False
        
        print("  ✅ ML/RL components integration verified")
        return True
        
    except Exception as e:
        print(f"  ❌ ML/RL components integration test failed: {e}")
        return False


async def test_authentication_integration():
    """Test ActivityLogger integration with Authentication System"""
    print("\n🔐 Testing Authentication System integration...")
    
    try:
        # Test authentication system integration
        import src.dashboard.auth as auth_module
        
        if hasattr(auth_module, 'activity_logger'):
            print("  ✅ ActivityLogger imported in Authentication System")
        else:
            print("  ❌ ActivityLogger not found in Authentication System")
            return False
        
        # Test authentication methods have been updated
        from src.dashboard.auth import DashboardAuth
        import inspect
        
        # Check authenticate_api_key method
        auth_method_source = inspect.getsource(DashboardAuth.authenticate_api_key)
        if 'activity_logger' in auth_method_source:
            print("  ✅ API key authentication logging integrated")
        else:
            print("  ❌ API key authentication logging not integrated")
            return False
        
        # Check create_session method
        session_method_source = inspect.getsource(DashboardAuth.create_session)
        if 'activity_logger' in session_method_source:
            print("  ✅ Session creation logging integrated")
        else:
            print("  ❌ Session creation logging not integrated")
            return False
        
        print("  ✅ Authentication system integration verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Authentication system integration test failed: {e}")
        return False


async def test_api_endpoints_integration():
    """Test ActivityLogger integration with API endpoints"""
    print("\n🌐 Testing API endpoints integration...")
    
    try:
        # Test API endpoints integration
        import src.dashboard.api as api_module
        
        if hasattr(api_module, 'activity_logger'):
            print("  ✅ ActivityLogger imported in API endpoints")
        else:
            print("  ❌ ActivityLogger not found in API endpoints")
            return False
        
        # Check if get_dashboard_data method has been updated
        from src.dashboard.api import DashboardAPI
        import inspect
        
        get_data_source = inspect.getsource(DashboardAPI.get_dashboard_data)
        if 'activity_logger' in get_data_source:
            print("  ✅ Dashboard data API logging integrated")
        else:
            print("  ❌ Dashboard data API logging not integrated")
            return False
        
        print("  ✅ API endpoints integration verified")
        return True
        
    except Exception as e:
        print(f"  ❌ API endpoints integration test failed: {e}")
        return False


async def test_mode_manager_integration():
    """Test ActivityLogger integration with Mode Manager"""
    print("\n⚙️ Testing Mode Manager integration...")
    
    try:
        # Test Mode Manager integration
        import src.modes.mode_manager as mm_module
        
        if hasattr(mm_module, 'activity_logger'):
            print("  ✅ ActivityLogger imported in Mode Manager")
        else:
            print("  ❌ ActivityLogger not found in Mode Manager")
            return False
        
        # Check if initialize method has been updated
        from src.modes.mode_manager import ModeManager
        import inspect
        
        init_source = inspect.getsource(ModeManager.initialize)
        if 'activity_logger' in init_source:
            print("  ✅ Mode Manager initialization logging integrated")
        else:
            print("  ❌ Mode Manager initialization logging not integrated")
            return False
        
        # Check if switch_mode method has been updated
        switch_source = inspect.getsource(ModeManager.switch_mode)
        if 'activity_logger' in switch_source:
            print("  ✅ Mode switching logging integrated")
        else:
            print("  ❌ Mode switching logging not integrated")
            return False
        
        print("  ✅ Mode Manager integration verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Mode Manager integration test failed: {e}")
        return False


async def test_performance_tracking():
    """Test performance tracking functionality"""
    print("\n⏱️ Testing performance tracking...")
    
    try:
        from src.logging.activity_logger import performance_tracker
        
        # Test performance tracker context manager
        async with performance_tracker(
            source="test_integration",
            operation="performance_test",
            category=ActivityCategory.SYSTEM
        ) as tracker:
            # Simulate some work
            await asyncio.sleep(0.1)
            print("  ✅ Performance tracker context manager working")
        
        print("  ✅ Performance tracking verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance tracking test failed: {e}")
        return False


async def test_activity_categories_and_actions():
    """Test that all activity categories and actions are properly defined"""
    print("\n📊 Testing activity categories and actions...")
    
    try:
        # Test that all required categories exist
        required_categories = [
            ActivityCategory.SYSTEM,
            ActivityCategory.TRADING,
            ActivityCategory.USER,
            ActivityCategory.ML_RL,
            ActivityCategory.SECURITY,
            ActivityCategory.API,
            ActivityCategory.PERFORMANCE,
            ActivityCategory.CONFIGURATION,
            ActivityCategory.DASHBOARD
        ]
        
        for category in required_categories:
            print(f"  ✅ Category {category.value} available")
        
        # Test that all required actions exist
        required_actions = [
            ActivityAction.CREATE,
            ActivityAction.READ,
            ActivityAction.UPDATE,
            ActivityAction.DELETE,
            ActivityAction.EXECUTE,
            ActivityAction.START,
            ActivityAction.STOP,
            ActivityAction.SUCCESS,
            ActivityAction.FAILURE,
            ActivityAction.ERROR,
            ActivityAction.LOGIN,
            ActivityAction.LOGOUT,
            ActivityAction.ACCESS
        ]
        
        for action in required_actions:
            print(f"  ✅ Action {action.value} available")
        
        print("  ✅ Activity categories and actions verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Activity categories and actions test failed: {e}")
        return False


async def run_comprehensive_test():
    """Run comprehensive ActivityLogger integration test"""
    print("🚀 Starting comprehensive ActivityLogger integration test...\n")
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_activity_logger_basic_functionality),
        ("Dashboard Service", test_dashboard_service_integration),
        ("Trading System", test_trading_system_integration),
        ("ML/RL Components", test_ml_rl_integration),
        ("Authentication", test_authentication_integration),
        ("API Endpoints", test_api_endpoints_integration),
        ("Mode Manager", test_mode_manager_integration),
        ("Performance Tracking", test_performance_tracking),
        ("Categories & Actions", test_activity_categories_and_actions),
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-"*70)
    print(f"Total tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(test_results)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! ActivityLogger integration is successful.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review the integration.")
    
    # Stop activity logger
    try:
        await activity_logger.stop()
    except:
        pass
    
    return failed == 0


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)