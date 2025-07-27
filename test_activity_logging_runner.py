#!/usr/bin/env python3
"""
Activity Logging Test Runner
Utility script to run activity logging tests with proper setup
"""

import asyncio
import os
import sys
import time
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def check_prerequisites():
    """Check that all prerequisites are met before running tests"""
    print("Checking prerequisites...")
    
    # Check database connection
    try:
        from src.utils.database import get_database_pool
        pool = await get_database_pool()
        
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
        
        print("✓ Database connection successful")
        
        # Check required tables exist
        async with pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name IN ('activity_logs', 'users')
            """)
            
            table_names = [t['table_name'] for t in tables]
            
            if 'activity_logs' not in table_names:
                print("❌ activity_logs table not found")
                return False
            
            print("✓ Required database tables exist")
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False
    
    # Check configuration
    try:
        from src.utils.config import get_config
        config = get_config()
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
        return False
    
    return True


async def run_basic_tests():
    """Run basic activity logging tests"""
    print("\n" + "="*50)
    print("RUNNING BASIC ACTIVITY LOGGING TESTS")
    print("="*50)
    
    from src.logging.activity_logger import (
        activity_logger, ActivityCategory, ActivityAction, ActivitySeverity
    )
    
    try:
        # Start activity logger
        await activity_logger.start()
        
        # Test 1: Basic activity logging
        print("\n--- Test 1: Basic Activity Logging ---")
        activity_id = await activity_logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="test_runner",
            event_type="basic_test",
            title="Basic activity logging test",
            description="Testing basic activity logging functionality"
        )
        
        print(f"✓ Activity logged with ID: {activity_id}")
        
        # Test 2: Error logging
        print("\n--- Test 2: Error Logging ---")
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            error_id = await activity_logger.log_error(
                category=ActivityCategory.SYSTEM,
                source="test_runner",
                event_type="error_test",
                title="Test error logging",
                exception=e
            )
            print(f"✓ Error logged with ID: {error_id}")
        
        # Test 3: Performance logging
        print("\n--- Test 3: Performance Logging ---")
        start_time = time.time()
        await asyncio.sleep(0.1)  # Simulate work
        execution_time = int((time.time() - start_time) * 1000)
        
        perf_id = await activity_logger.log_performance(
            source="test_runner",
            operation="test_operation",
            execution_time_ms=execution_time
        )
        print(f"✓ Performance metric logged with ID: {perf_id}")
        
        # Force flush to ensure all activities are stored
        await activity_logger._flush_buffer()
        
        # Verify activities were stored
        print("\n--- Verifying Storage ---")
        from src.utils.database import get_database_pool
        pool = await get_database_pool()
        
        async with pool.acquire() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM activity_logs 
                WHERE source = 'test_runner' 
                AND created_at > NOW() - INTERVAL '1 minute'
            """)
            
            print(f"✓ {count} test activities found in database")
            
            if count >= 3:
                print("✓ All basic tests passed")
                return True
            else:
                print("❌ Not all activities were stored")
                return False
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False
    
    finally:
        await activity_logger.stop()


async def run_dashboard_integration_tests():
    """Run dashboard integration tests"""
    print("\n" + "="*50)
    print("RUNNING DASHBOARD INTEGRATION TESTS")
    print("="*50)
    
    try:
        from src.dashboard.activity_integration import dashboard_activity
        
        # Start dashboard activity integration
        await dashboard_activity.start()
        
        # Test 1: Dashboard activity logging
        print("\n--- Test 1: Dashboard Activity Logging ---")
        activity_id = await dashboard_activity.log_dashboard_access(
            user_id=12345,
            component="test_component",
            session_id="test_session_123"
        )
        print(f"✓ Dashboard activity logged with ID: {activity_id}")
        
        # Test 2: Recent activity retrieval
        print("\n--- Test 2: Recent Activity Retrieval ---")
        recent_activities = await dashboard_activity.get_recent_activity(limit=10)
        print(f"✓ Retrieved {len(recent_activities)} recent activities")
        
        # Test 3: Performance metrics
        print("\n--- Test 3: Performance Metrics ---")
        metrics = await dashboard_activity.get_performance_metrics(hours_back=1)
        print(f"✓ Retrieved {len(metrics)} performance metrics")
        
        print("✓ Dashboard integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard integration tests failed: {e}")
        return False
    
    finally:
        await dashboard_activity.stop()


async def run_websocket_tests():
    """Run WebSocket tests"""
    print("\n" + "="*50)
    print("RUNNING WEBSOCKET TESTS")
    print("="*50)
    
    try:
        from src.dashboard.websocket_manager import websocket_manager
        
        # Start WebSocket manager
        await websocket_manager.start()
        
        # Test 1: WebSocket manager startup
        print("\n--- Test 1: WebSocket Manager Status ---")
        assert websocket_manager._running, "WebSocket manager should be running"
        print("✓ WebSocket manager is running")
        
        # Test 2: Activity broadcasting
        print("\n--- Test 2: Activity Broadcasting ---")
        test_activity = {
            'activity_id': 'test-ws-123',
            'category': 'system',
            'title': 'WebSocket test activity',
            'created_at': '2025-01-01T00:00:00Z'
        }
        
        await websocket_manager.send_activity_update(test_activity)
        print("✓ Activity broadcast successful")
        
        # Test 3: Connection management
        print("\n--- Test 3: Connection Management ---")
        connection_count = await websocket_manager.connection_manager.get_connection_count()
        print(f"✓ Current connections: {connection_count}")
        
        print("✓ WebSocket tests passed")
        return True
        
    except Exception as e:
        print(f"❌ WebSocket tests failed: {e}")
        return False
    
    finally:
        await websocket_manager.stop()


async def run_performance_tests():
    """Run performance tests"""
    print("\n" + "="*50)
    print("RUNNING PERFORMANCE TESTS")
    print("="*50)
    
    try:
        from src.logging.activity_logger import activity_logger, ActivityCategory, ActivityAction
        
        await activity_logger.start()
        
        # Test 1: High volume activity generation
        print("\n--- Test 1: High Volume Activity Generation ---")
        start_time = time.time()
        
        activities = []
        for i in range(100):
            activity_id = await activity_logger.log_activity(
                category=ActivityCategory.PERFORMANCE,
                action=ActivityAction.EXECUTE,
                source="performance_test",
                event_type="volume_test",
                title=f"Performance test activity {i}",
                execution_time_ms=10 + i
            )
            activities.append(activity_id)
        
        await activity_logger._flush_buffer()
        generation_time = time.time() - start_time
        
        print(f"✓ Generated 100 activities in {generation_time:.2f} seconds")
        print(f"✓ Rate: {100 / generation_time:.1f} activities/second")
        
        # Test 2: Query performance
        print("\n--- Test 2: Query Performance ---")
        from src.utils.database import get_database_pool
        pool = await get_database_pool()
        
        query_start = time.time()
        async with pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT * FROM activity_logs 
                WHERE source = 'performance_test'
                ORDER BY created_at DESC 
                LIMIT 50
            """)
        query_time = time.time() - query_start
        
        print(f"✓ Query returned {len(results)} results in {query_time:.3f} seconds")
        
        if generation_time < 5.0 and query_time < 1.0:
            print("✓ Performance tests passed")
            return True
        else:
            print("❌ Performance below expected thresholds")
            return False
            
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
        return False
    
    finally:
        await activity_logger.stop()


async def cleanup_test_data():
    """Clean up test data from database"""
    print("\n--- Cleaning up test data ---")
    
    try:
        from src.utils.database import get_database_pool
        pool = await get_database_pool()
        
        async with pool.acquire() as conn:
            # Delete test activities
            deleted = await conn.fetchval("""
                DELETE FROM activity_logs 
                WHERE source IN ('test_runner', 'performance_test', 'dashboard_test')
                OR event_type LIKE '%test%'
                RETURNING COUNT(*)
            """)
            
            print(f"✓ Cleaned up {deleted or 0} test activities")
            
    except Exception as e:
        print(f"⚠ Warning: Failed to clean up test data: {e}")


async def main():
    """Main test runner"""
    print("Activity Logging Test Runner")
    print("=" * 60)
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        print("\n\nTest interrupted by user")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check prerequisites
    if not await check_prerequisites():
        print("\n❌ Prerequisites not met. Please check your setup.")
        return False
    
    # Run test suites
    test_results = []
    
    # Basic tests
    basic_result = await run_basic_tests()
    test_results.append(("Basic Tests", basic_result))
    
    # Dashboard integration tests
    dashboard_result = await run_dashboard_integration_tests()
    test_results.append(("Dashboard Integration", dashboard_result))
    
    # WebSocket tests
    websocket_result = await run_websocket_tests()
    test_results.append(("WebSocket Tests", websocket_result))
    
    # Performance tests
    performance_result = await run_performance_tests()
    test_results.append(("Performance Tests", performance_result))
    
    # Clean up
    await cleanup_test_data()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        icon = "✓" if result else "❌"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    if success_rate >= 75:
        print("\n🎉 ACTIVITY LOGGING TESTS: SUCCESS")
        return True
    else:
        print("\n❌ ACTIVITY LOGGING TESTS: NEEDS IMPROVEMENT")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)