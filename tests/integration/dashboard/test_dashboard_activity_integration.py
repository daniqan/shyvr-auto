#!/usr/bin/env python3
"""
Test script for dashboard activity integration
Demonstrates the new database-backed activity logging functionality
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path - adjust for new location in tests/integration/dashboard/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.activity_logging.activity_logger import (
    activity_logger,
    ActivityCategory,
    ActivityAction,
    ActivitySeverity,
    TradingMode,
    ChainType
)
from src.dashboard.activity_integration import dashboard_activity
from src.dashboard.service import dashboard_service
from src.dashboard.websocket_manager import websocket_manager


async def test_activity_logging():
    """Test basic activity logging functionality"""
    print("🧪 Testing Activity Logging...")
    
    try:
        # Start activity logger
        await activity_logger.start()
        await dashboard_activity.start()
        
        # Log some test activities
        activities = []
        
        # System activity
        activity_id = await activity_logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.START,
            source="test_system",
            event_type="test_startup",
            title="Test system started",
            description="Test system startup for dashboard integration",
            severity=ActivitySeverity.INFO,
            user_id=1001,
            metadata={"test": True, "version": "1.0.0"}
        )
        activities.append(activity_id)
        print(f"✅ Logged system activity: {activity_id}")
        
        # Trading activity
        activity_id = await activity_logger.log_trading_activity(
            action=ActivityAction.EXECUTE,
            title="Test trade executed",
            trading_mode=TradingMode.SIMULATION,
            token_address="0x123456789abcdef",
            chain=ChainType.ETHEREUM,
            amount_usd=1000.50,
            fee_usd=2.50,
            user_id=1001,
            metadata={"symbol": "ETH/USDC", "side": "buy"}
        )
        activities.append(activity_id)
        print(f"✅ Logged trading activity: {activity_id}")
        
        # API activity
        activity_id = await activity_logger.log_api_call(
            api_name="dashboard_api",
            endpoint="/dashboard/data",
            method="GET",
            status_code=200,
            response_time_ms=150,
            success=True,
            user_id=1001,
            metadata={"test_request": True}
        )
        activities.append(activity_id)
        print(f"✅ Logged API activity: {activity_id}")
        
        # Error activity
        activity_id = await activity_logger.log_error(
            category=ActivityCategory.DASHBOARD,
            source="test_component",
            event_type="test_error",
            title="Test error occurred",
            error_message="This is a test error",
            severity=ActivitySeverity.ERROR,
            user_id=1001,
            error_code="TEST_001"
        )
        activities.append(activity_id)
        print(f"✅ Logged error activity: {activity_id}")
        
        return activities
        
    except Exception as e:
        print(f"❌ Activity logging test failed: {e}")
        return []


async def test_activity_retrieval():
    """Test activity retrieval with filtering"""
    print("\n🔍 Testing Activity Retrieval...")
    
    try:
        # Get recent activities
        recent_activities = await dashboard_activity.get_recent_activity(
            limit=10,
            hours_back=1
        )
        print(f"✅ Retrieved {len(recent_activities)} recent activities")
        
        # Get activities by category
        system_activities = await dashboard_activity.get_recent_activity(
            limit=5,
            category="system",
            hours_back=1
        )
        print(f"✅ Retrieved {len(system_activities)} system activities")
        
        # Get error summary
        error_summary = await dashboard_activity.get_error_summary(hours_back=1)
        print(f"✅ Retrieved error summary with {len(error_summary)} error types")
        
        # Get performance metrics
        performance_metrics = await dashboard_activity.get_performance_metrics(hours_back=1)
        print(f"✅ Retrieved performance metrics for {len(performance_metrics)} sources")
        
        # Get user activity stats
        user_stats = await dashboard_activity.get_user_activity_stats(
            user_id=1001,
            days_back=1
        )
        print(f"✅ Retrieved user stats: {user_stats.get('total_activities', 0)} total activities")
        
        return recent_activities
        
    except Exception as e:
        print(f"❌ Activity retrieval test failed: {e}")
        return []


async def test_dashboard_service_integration():
    """Test dashboard service integration with real activity data"""
    print("\n📊 Testing Dashboard Service Integration...")
    
    try:
        # Start dashboard service
        await dashboard_service.start()
        
        # Get dashboard data (should include real activity data now)
        dashboard_data = await dashboard_service.get_dashboard_data()
        
        print(f"✅ Dashboard data retrieved")
        print(f"   - Active alerts: {dashboard_data.active_alerts}")
        print(f"   - Warnings: {dashboard_data.warnings_count}")
        print(f"   - Errors: {dashboard_data.errors_count}")
        print(f"   - Recent logs: {len(dashboard_data.recent_logs)}")
        
        # Test recent activity method
        recent_activities = await dashboard_service._get_recent_activity(limit=5)
        print(f"✅ Recent activities: {len(recent_activities)} items")
        
        # Test system logs (should use real data)
        system_logs = await dashboard_service.get_system_logs(limit=5)
        print(f"✅ System logs: {len(system_logs)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard service integration test failed: {e}")
        return False
    finally:
        await dashboard_service.stop()


async def test_websocket_integration():
    """Test WebSocket integration for real-time activity updates"""
    print("\n📡 Testing WebSocket Integration...")
    
    try:
        # Start WebSocket manager
        await websocket_manager.start()
        
        # Simulate activity update broadcast
        test_activity = {
            "activity_id": "test-activity-123",
            "created_at": datetime.utcnow().isoformat(),
            "category": "system",
            "action": "test",
            "severity": "info",
            "source": "test_websocket",
            "title": "Test WebSocket activity update",
            "description": "Testing real-time activity broadcasting"
        }
        
        # This would normally be called when new activities are logged
        await websocket_manager.send_activity_update(test_activity)
        print("✅ Activity update broadcast sent")
        
        # Test batch update
        test_activities = [test_activity, {**test_activity, "activity_id": "test-activity-124"}]
        await websocket_manager.send_activity_batch(test_activities)
        print("✅ Activity batch update broadcast sent")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket integration test failed: {e}")
        return False
    finally:
        await websocket_manager.stop()


async def main():
    """Run all tests"""
    print("🚀 Starting Dashboard Activity Integration Tests\n")
    
    # Test activity logging
    logged_activities = await test_activity_logging()
    
    # Small delay to ensure activities are stored
    await asyncio.sleep(1)
    
    # Test activity retrieval
    retrieved_activities = await test_activity_retrieval()
    
    # Test dashboard service integration
    dashboard_success = await test_dashboard_service_integration()
    
    # Test WebSocket integration
    websocket_success = await test_websocket_integration()
    
    # Cleanup
    try:
        await dashboard_activity.stop()
        await activity_logger.stop()
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"   - Activity Logging: {'✅ PASS' if logged_activities else '❌ FAIL'}")
    print(f"   - Activity Retrieval: {'✅ PASS' if retrieved_activities else '❌ FAIL'}")
    print(f"   - Dashboard Integration: {'✅ PASS' if dashboard_success else '❌ FAIL'}")
    print(f"   - WebSocket Integration: {'✅ PASS' if websocket_success else '❌ FAIL'}")
    
    overall_success = all([
        bool(logged_activities),
        bool(retrieved_activities),
        dashboard_success,
        websocket_success
    ])
    
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)