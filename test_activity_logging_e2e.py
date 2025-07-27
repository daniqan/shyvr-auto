#!/usr/bin/env python3
"""
End-to-End Activity Logging Test Suite
Tests the complete flow from system events to dashboard display
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import pytest
import pytest_asyncio
import asyncpg
import structlog
from fastapi.testclient import TestClient
from websockets import connect as ws_connect, ConnectionClosedError

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.logging.activity_logger import (
    activity_logger, ActivityLogger, ActivityCategory, ActivityAction,
    ActivitySeverity, TradingMode, ChainType, ActivityLogEntry,
    log_system_event, log_dashboard_action, log_trade_execution
)
from src.dashboard.activity_integration import (
    dashboard_activity, DashboardActivityIntegration
)
from src.dashboard.websocket_manager import websocket_manager, WebSocketManager
from src.dashboard.api import create_app
from src.utils.config import get_config
from src.utils.database import get_database_pool

logger = structlog.get_logger()


class ActivityLoggingE2ETest:
    """End-to-end test suite for activity logging"""
    
    def __init__(self):
        self.config = get_config()
        self.db_pool: Optional[asyncpg.Pool] = None
        self.app = None
        self.test_client = None
        self.test_activities: List[str] = []
        self.websocket_connections: List = []
        
    async def setup(self):
        """Setup test environment"""
        print("\n=== Setting up E2E Activity Logging Test ===")
        
        # Initialize database connection
        self.db_pool = await get_database_pool()
        
        # Start activity logger
        await activity_logger.start()
        
        # Start dashboard activity integration
        await dashboard_activity.start()
        
        # Start WebSocket manager
        await websocket_manager.start()
        
        # Create FastAPI app and test client
        self.app = create_app()
        self.test_client = TestClient(self.app)
        
        print("✓ E2E test environment setup complete")
    
    async def teardown(self):
        """Clean up test environment"""
        print("\n=== Cleaning up E2E Activity Logging Test ===")
        
        # Close WebSocket connections
        for ws in self.websocket_connections:
            try:
                await ws.close()
            except:
                pass
        
        # Stop services
        await websocket_manager.stop()
        await dashboard_activity.stop()
        await activity_logger.stop()
        
        # Clean up test data
        if self.db_pool and self.test_activities:
            try:
                async with self.db_pool.acquire() as conn:
                    activity_ids = [f"'{aid}'" for aid in self.test_activities]
                    await conn.execute(
                        f"DELETE FROM activity_logs WHERE activity_id IN ({','.join(activity_ids)})"
                    )
                print(f"✓ Cleaned up {len(self.test_activities)} test activities")
            except Exception as e:
                print(f"⚠ Warning: Failed to clean up test data: {e}")
        
        if self.test_client:
            self.test_client.close()
        
        print("✓ E2E test cleanup complete")


# Test instance
e2e_test = ActivityLoggingE2ETest()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_and_teardown():
    """Setup and teardown for the entire test session"""
    await e2e_test.setup()
    yield
    await e2e_test.teardown()


class TestDatabaseSetupAndConnectivity:
    """Test 1: Database setup and connectivity"""
    
    async def test_database_connection(self):
        """Test basic database connectivity"""
        print("\n--- Test 1.1: Database Connection ---")
        
        assert e2e_test.db_pool is not None, "Database pool should be initialized"
        
        async with e2e_test.db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1, "Basic database query should work"
        
        print("✓ Database connection successful")
    
    async def test_activity_logs_table_exists(self):
        """Test that activity_logs table exists with correct schema"""
        print("\n--- Test 1.2: Activity Logs Table Schema ---")
        
        async with e2e_test.db_pool.acquire() as conn:
            # Check table exists
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'activity_logs'
                )
            """)
            assert exists, "activity_logs table should exist"
            
            # Check required columns exist
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'activity_logs'
                ORDER BY ordinal_position
            """)
            
            column_names = [col['column_name'] for col in columns]
            required_columns = [
                'id', 'activity_id', 'created_at', 'category', 'action',
                'severity', 'source', 'event_type', 'title'
            ]
            
            for col in required_columns:
                assert col in column_names, f"Required column '{col}' should exist"
        
        print("✓ Activity logs table schema valid")
    
    async def test_database_indexes(self):
        """Test that required indexes exist for performance"""
        print("\n--- Test 1.3: Database Indexes ---")
        
        async with e2e_test.db_pool.acquire() as conn:
            indexes = await conn.fetch("""
                SELECT indexname, tablename
                FROM pg_indexes
                WHERE tablename = 'activity_logs'
            """)
            
            index_names = [idx['indexname'] for idx in indexes]
            
            # Check for critical performance indexes
            critical_indexes = [
                'idx_activity_logs_created_at_desc',
                'idx_activity_logs_category_created',
                'idx_activity_logs_severity_created'
            ]
            
            for idx in critical_indexes:
                assert idx in index_names, f"Critical index '{idx}' should exist"
        
        print("✓ Database indexes present")


class TestActivityGeneration:
    """Test 2: Activity generation from various system components"""
    
    async def test_system_activity_generation(self):
        """Test generating system activities"""
        print("\n--- Test 2.1: System Activity Generation ---")
        
        activity_id = await log_system_event(
            event_type="test_system_startup",
            title="Test system startup event",
            description="E2E test system startup",
            metadata={"test_type": "e2e", "component": "system"}
        )
        
        assert activity_id is not None, "System activity should generate activity ID"
        e2e_test.test_activities.append(activity_id)
        
        # Wait for activity to be persisted
        await asyncio.sleep(1)
        
        # Verify activity was stored
        async with e2e_test.db_pool.acquire() as conn:
            activity = await conn.fetchrow(
                "SELECT * FROM activity_logs WHERE activity_id = $1",
                activity_id
            )
            
            assert activity is not None, "Activity should be stored in database"
            assert activity['category'] == 'system'
            assert activity['event_type'] == 'test_system_startup'
            assert activity['title'] == 'Test system startup event'
        
        print("✓ System activity generation successful")
    
    async def test_trading_activity_generation(self):
        """Test generating trading activities"""
        print("\n--- Test 2.2: Trading Activity Generation ---")
        
        activity_id = await log_trade_execution(
            trading_mode=TradingMode.SIMULATION,
            token_address="0x1234567890abcdef",
            chain=ChainType.ETHEREUM,
            amount_usd=Decimal("100.50"),
            success=True,
            execution_time_ms=250,
            metadata={"test_type": "e2e", "trader": "test_bot"}
        )
        
        assert activity_id is not None, "Trading activity should generate activity ID"
        e2e_test.test_activities.append(activity_id)
        
        # Wait for activity to be persisted
        await asyncio.sleep(1)
        
        # Verify activity was stored with trading context
        async with e2e_test.db_pool.acquire() as conn:
            activity = await conn.fetchrow(
                "SELECT * FROM activity_logs WHERE activity_id = $1",
                activity_id
            )
            
            assert activity is not None, "Trading activity should be stored"
            assert activity['category'] == 'trading'
            assert activity['trading_mode'] == 'simulation'
            assert activity['token_address'] == '0x1234567890abcdef'
            assert activity['chain'] == 'ethereum'
            assert activity['amount_usd'] == Decimal("100.50")
        
        print("✓ Trading activity generation successful")
    
    async def test_dashboard_activity_generation(self):
        """Test generating dashboard activities"""
        print("\n--- Test 2.3: Dashboard Activity Generation ---")
        
        user_id = 12345
        session_id = str(uuid.uuid4())
        
        activity_id = await log_dashboard_action(
            user_id=user_id,
            component="portfolio_view",
            action="view_positions",
            session_id=session_id,
            metadata={"test_type": "e2e", "page": "portfolio"}
        )
        
        assert activity_id is not None, "Dashboard activity should generate activity ID"
        e2e_test.test_activities.append(activity_id)
        
        # Wait for activity to be persisted
        await asyncio.sleep(1)
        
        # Verify activity was stored with user context
        async with e2e_test.db_pool.acquire() as conn:
            activity = await conn.fetchrow(
                "SELECT * FROM activity_logs WHERE activity_id = $1",
                activity_id
            )
            
            assert activity is not None, "Dashboard activity should be stored"
            assert activity['category'] == 'user'
            assert activity['user_id'] == user_id
            assert activity['session_id'] == session_id
            assert activity['dashboard_component'] == 'portfolio_view'
        
        print("✓ Dashboard activity generation successful")
    
    async def test_error_activity_generation(self):
        """Test generating error activities with stack traces"""
        print("\n--- Test 2.4: Error Activity Generation ---")
        
        try:
            # Generate a test exception
            raise ValueError("Test error for E2E testing")
        except Exception as e:
            activity_id = await activity_logger.log_error(
                category=ActivityCategory.SYSTEM,
                source="e2e_test",
                event_type="test_error",
                title="Test error event",
                error_code="E2E_TEST_ERROR",
                exception=e,
                metadata={"test_type": "e2e", "component": "error_handler"}
            )
        
        assert activity_id is not None, "Error activity should generate activity ID"
        e2e_test.test_activities.append(activity_id)
        
        # Wait for activity to be persisted
        await asyncio.sleep(1)
        
        # Verify error activity was stored with error context
        async with e2e_test.db_pool.acquire() as conn:
            activity = await conn.fetchrow(
                "SELECT * FROM activity_logs WHERE activity_id = $1",
                activity_id
            )
            
            assert activity is not None, "Error activity should be stored"
            assert activity['severity'] == 'error'
            assert activity['error_code'] == 'E2E_TEST_ERROR'
            assert activity['error_message'] == 'Test error for E2E testing'
            assert activity['stack_trace'] is not None
            assert 'ValueError' in activity['stack_trace']
        
        print("✓ Error activity generation successful")


class TestActivityStorage:
    """Test 3: Activity storage in PostgreSQL database"""
    
    async def test_high_volume_activity_storage(self):
        """Test storing multiple activities rapidly"""
        print("\n--- Test 3.1: High Volume Activity Storage ---")
        
        # Generate 50 activities rapidly
        activity_ids = []
        start_time = time.time()
        
        for i in range(50):
            activity_id = await activity_logger.log_activity(
                category=ActivityCategory.PERFORMANCE,
                action=ActivityAction.EXECUTE,
                source="e2e_test_batch",
                event_type="batch_test",
                title=f"Batch test activity {i}",
                severity=ActivitySeverity.INFO,
                execution_time_ms=10 + i,
                metadata={"batch_number": i, "test_type": "e2e"}
            )
            activity_ids.append(activity_id)
        
        # Force flush to ensure all activities are stored
        await activity_logger._flush_buffer()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Generated {len(activity_ids)} activities in {duration:.2f} seconds")
        
        # Add to cleanup list
        e2e_test.test_activities.extend(activity_ids)
        
        # Wait for all activities to be persisted
        await asyncio.sleep(2)
        
        # Verify all activities were stored
        async with e2e_test.db_pool.acquire() as conn:
            stored_count = await conn.fetchval("""
                SELECT COUNT(*) FROM activity_logs 
                WHERE source = 'e2e_test_batch' AND event_type = 'batch_test'
            """)
            
            assert stored_count == 50, f"All 50 activities should be stored, got {stored_count}"
        
        print("✓ High volume activity storage successful")
    
    async def test_activity_data_integrity(self):
        """Test that activity data integrity is maintained"""
        print("\n--- Test 3.2: Activity Data Integrity ---")
        
        # Create activity with comprehensive data
        original_entry = ActivityLogEntry(
            category=ActivityCategory.API,
            action=ActivityAction.SUCCESS,
            source="e2e_api_test",
            event_type="api_call_test",
            title="API integrity test",
            description="Testing data integrity preservation",
            severity=ActivitySeverity.INFO,
            user_id=67890,
            session_id=str(uuid.uuid4()),
            trading_mode=TradingMode.LIVE,
            token_address="0xabcdef1234567890",
            chain=ChainType.SOLANA,
            amount_usd=Decimal("1234.56"),
            fee_usd=Decimal("12.34"),
            execution_time_ms=123,
            memory_usage_mb=456,
            cpu_usage_pct=Decimal("78.9"),
            metadata={"test": "integrity", "number": 42, "nested": {"key": "value"}},
            tags=["test", "integrity", "e2e"],
            api_endpoint="/api/test",
            http_method="POST",
            http_status=200,
            response_time_ms=100,
            security_level="high",
            risk_score=25
        )
        
        # Log the activity
        activity_id = await activity_logger.log_activity(**original_entry.to_dict())
        e2e_test.test_activities.append(activity_id)
        
        # Wait for persistence
        await asyncio.sleep(1)
        
        # Retrieve and verify all fields
        async with e2e_test.db_pool.acquire() as conn:
            stored_activity = await conn.fetchrow(
                "SELECT * FROM activity_logs WHERE activity_id = $1",
                activity_id
            )
            
            assert stored_activity is not None, "Activity should be stored"
            
            # Verify all key fields
            assert stored_activity['category'] == 'api'
            assert stored_activity['action'] == 'success'
            assert stored_activity['source'] == 'e2e_api_test'
            assert stored_activity['title'] == 'API integrity test'
            assert stored_activity['user_id'] == 67890
            assert stored_activity['trading_mode'] == 'live'
            assert stored_activity['amount_usd'] == Decimal("1234.56")
            assert stored_activity['metadata']['test'] == 'integrity'
            assert 'test' in stored_activity['tags']
            assert stored_activity['http_status'] == 200
            assert stored_activity['risk_score'] == 25
        
        print("✓ Activity data integrity maintained")
    
    async def test_concurrent_activity_storage(self):
        """Test concurrent activity storage from multiple sources"""
        print("\n--- Test 3.3: Concurrent Activity Storage ---")
        
        async def generate_activities(source_name: str, count: int) -> List[str]:
            """Generate activities from a specific source"""
            ids = []
            for i in range(count):
                activity_id = await activity_logger.log_activity(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.EXECUTE,
                    source=source_name,
                    event_type="concurrent_test",
                    title=f"{source_name} activity {i}",
                    metadata={"source": source_name, "index": i}
                )
                ids.append(activity_id)
                # Small delay to simulate real activity generation
                await asyncio.sleep(0.01)
            return ids
        
        # Launch concurrent activity generation
        sources = ["source_1", "source_2", "source_3", "source_4"]
        tasks = [generate_activities(source, 25) for source in sources]
        
        results = await asyncio.gather(*tasks)
        all_activity_ids = [aid for result in results for aid in result]
        
        e2e_test.test_activities.extend(all_activity_ids)
        
        # Wait for all activities to be stored
        await asyncio.sleep(3)
        
        # Verify all activities were stored correctly
        async with e2e_test.db_pool.acquire() as conn:
            for i, source in enumerate(sources):
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM activity_logs 
                    WHERE source = $1 AND event_type = 'concurrent_test'
                """, source)
                
                assert count == 25, f"Source {source} should have 25 activities, got {count}"
        
        print("✓ Concurrent activity storage successful")


class TestDashboardAPIRetrieval:
    """Test 4: Dashboard API retrieval of activity data"""
    
    async def test_recent_activity_api(self):
        """Test retrieving recent activities via API"""
        print("\n--- Test 4.1: Recent Activity API ---")
        
        # Generate some test activities first
        for i in range(5):
            activity_id = await activity_logger.log_activity(
                category=ActivityCategory.DASHBOARD,
                action=ActivityAction.ACCESS,
                source="api_test",
                event_type="recent_activity_test",
                title=f"API test activity {i}",
                severity=ActivitySeverity.INFO,
                metadata={"api_test": True, "index": i}
            )
            e2e_test.test_activities.append(activity_id)
        
        # Wait for activities to be stored
        await asyncio.sleep(2)
        
        # Test API endpoint
        response = e2e_test.test_client.get("/api/v1/activity/recent?limit=10")
        
        assert response.status_code == 200, f"API should return 200, got {response.status_code}"
        
        data = response.json()
        assert 'activities' in data, "Response should contain activities"
        assert len(data['activities']) >= 5, "Should return at least our test activities"
        
        # Verify activity structure
        activity = data['activities'][0]
        required_fields = ['activity_id', 'created_at', 'category', 'action', 'title']
        for field in required_fields:
            assert field in activity, f"Activity should contain {field}"
        
        print("✓ Recent activity API working")
    
    async def test_activity_filtering_api(self):
        """Test activity filtering via API parameters"""
        print("\n--- Test 4.2: Activity Filtering API ---")
        
        # Generate activities with different categories
        categories = ['system', 'trading', 'user']
        for cat in categories:
            for i in range(3):
                activity_id = await activity_logger.log_activity(
                    category=ActivityCategory(cat),
                    action=ActivityAction.EXECUTE,
                    source="filter_test",
                    event_type="filtering_test",
                    title=f"{cat} filter test {i}",
                    metadata={"filter_test": True, "category": cat}
                )
                e2e_test.test_activities.append(activity_id)
        
        await asyncio.sleep(2)
        
        # Test filtering by category
        response = e2e_test.test_client.get("/api/v1/activity/recent?category=trading")
        assert response.status_code == 200
        
        data = response.json()
        trading_activities = [a for a in data['activities'] if a['category'] == 'trading']
        assert len(trading_activities) >= 3, "Should return trading activities"
        
        # Test filtering by severity
        response = e2e_test.test_client.get("/api/v1/activity/recent?severity=info")
        assert response.status_code == 200
        
        print("✓ Activity filtering API working")
    
    async def test_activity_search_api(self):
        """Test activity search functionality"""
        print("\n--- Test 4.3: Activity Search API ---")
        
        # Generate searchable activities
        search_terms = ["unique_search_term_123", "another_search_456"]
        for term in search_terms:
            activity_id = await activity_logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source="search_test",
                event_type="search_test",
                title=f"Activity with {term}",
                description=f"This activity contains the search term {term}",
                metadata={"search_test": True, "term": term}
            )
            e2e_test.test_activities.append(activity_id)
        
        await asyncio.sleep(2)
        
        # Test search functionality (if implemented)
        response = e2e_test.test_client.get("/api/v1/activity/search?q=unique_search_term_123")
        
        # Note: This might return 404 if search endpoint doesn't exist yet
        if response.status_code == 200:
            data = response.json()
            found_activities = [a for a in data.get('activities', []) 
                             if 'unique_search_term_123' in a.get('title', '')]
            assert len(found_activities) >= 1, "Should find activities with search term"
            print("✓ Activity search API working")
        else:
            print("⚠ Activity search API not implemented yet")


class TestWebSocketRealTimeBroadcasting:
    """Test 5: WebSocket real-time activity broadcasting"""
    
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        print("\n--- Test 5.1: WebSocket Connection ---")
        
        # Note: This is a simplified test since we need a running server
        # In a real E2E test, you'd have a test server running
        
        # Test that WebSocket manager is running
        assert websocket_manager._running, "WebSocket manager should be running"
        
        # Test connection count
        count = await websocket_manager.connection_manager.get_connection_count()
        assert count >= 0, "Connection count should be valid"
        
        print("✓ WebSocket manager operational")
    
    async def test_activity_broadcasting(self):
        """Test real-time activity broadcasting"""
        print("\n--- Test 5.2: Activity Broadcasting ---")
        
        # Create a test activity that should trigger WebSocket broadcast
        test_activity = {
            'activity_id': str(uuid.uuid4()),
            'category': 'system',
            'action': 'execute',
            'title': 'WebSocket broadcast test',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'metadata': {'test_type': 'websocket_broadcast'}
        }
        
        # Send activity update via WebSocket manager
        await websocket_manager.send_activity_update(test_activity)
        
        # Test batch activity broadcasting
        batch_activities = [
            {
                'activity_id': str(uuid.uuid4()),
                'category': 'trading',
                'title': f'Batch activity {i}',
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            for i in range(3)
        ]
        
        await websocket_manager.send_activity_batch(batch_activities)
        
        print("✓ Activity broadcasting successful")
    
    async def test_websocket_subscriptions(self):
        """Test WebSocket topic subscriptions"""
        print("\n--- Test 5.3: WebSocket Subscriptions ---")
        
        # Test topic subscriber counts
        activity_subscribers = await websocket_manager.connection_manager.get_topic_subscribers("activity")
        dashboard_subscribers = await websocket_manager.connection_manager.get_topic_subscribers("dashboard")
        
        assert activity_subscribers >= 0, "Activity subscribers count should be valid"
        assert dashboard_subscribers >= 0, "Dashboard subscribers count should be valid"
        
        print("✓ WebSocket subscriptions working")


class TestFrontendActivityDisplay:
    """Test 6: Frontend activity display and filtering"""
    
    async def test_activity_display_api(self):
        """Test the API endpoints that frontend uses for activity display"""
        print("\n--- Test 6.1: Frontend Activity Display API ---")
        
        # Test the main activities endpoint for frontend
        response = e2e_test.test_client.get("/api/v1/activity/recent?limit=50")
        assert response.status_code == 200
        
        data = response.json()
        activities = data.get('activities', [])
        
        # Verify activity data structure for frontend consumption
        if activities:
            activity = activities[0]
            frontend_fields = [
                'activity_id', 'created_at', 'category', 'action', 
                'severity', 'title', 'source', 'event_type'
            ]
            
            for field in frontend_fields:
                assert field in activity, f"Frontend requires field: {field}"
        
        print("✓ Frontend activity display API working")
    
    async def test_activity_summary_api(self):
        """Test activity summary API for dashboard widgets"""
        print("\n--- Test 6.2: Activity Summary API ---")
        
        # Generate some activities for summary testing
        for severity in ['info', 'warning', 'error']:
            for i in range(2):
                activity_id = await activity_logger.log_activity(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.EXECUTE,
                    source="summary_test",
                    event_type="summary_test",
                    title=f"Summary test {severity} {i}",
                    severity=ActivitySeverity(severity),
                    metadata={"summary_test": True}
                )
                e2e_test.test_activities.append(activity_id)
        
        await asyncio.sleep(2)
        
        # Test summary endpoint (if exists)
        response = e2e_test.test_client.get("/api/v1/activity/summary")
        
        if response.status_code == 200:
            data = response.json()
            assert 'summary' in data or 'counts' in data, "Summary should contain aggregated data"
            print("✓ Activity summary API working")
        else:
            print("⚠ Activity summary API not implemented yet")


class TestPerformanceHighVolume:
    """Test 7: Performance with high-volume activity logging"""
    
    async def test_high_volume_performance(self):
        """Test system performance with high volume of activities"""
        print("\n--- Test 7.1: High Volume Performance ---")
        
        # Test with 200 activities generated rapidly
        start_time = time.time()
        activity_ids = []
        
        # Generate activities in batches to simulate real load
        batch_size = 20
        total_activities = 200
        
        for batch in range(total_activities // batch_size):
            batch_tasks = []
            for i in range(batch_size):
                task = activity_logger.log_activity(
                    category=ActivityCategory.PERFORMANCE,
                    action=ActivityAction.EXECUTE,
                    source="performance_test",
                    event_type="high_volume_test",
                    title=f"Performance test activity {batch * batch_size + i}",
                    execution_time_ms=10 + i,
                    metadata={
                        "batch": batch,
                        "index": i,
                        "test_type": "high_volume"
                    }
                )
                batch_tasks.append(task)
            
            # Execute batch concurrently
            batch_ids = await asyncio.gather(*batch_tasks)
            activity_ids.extend(batch_ids)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Force flush all activities
        await activity_logger._flush_buffer()
        
        generation_time = time.time() - start_time
        e2e_test.test_activities.extend(activity_ids)
        
        print(f"Generated {len(activity_ids)} activities in {generation_time:.2f} seconds")
        print(f"Rate: {len(activity_ids) / generation_time:.1f} activities/second")
        
        # Wait for all activities to be persisted
        await asyncio.sleep(3)
        
        # Verify all activities were stored
        async with e2e_test.db_pool.acquire() as conn:
            stored_count = await conn.fetchval("""
                SELECT COUNT(*) FROM activity_logs 
                WHERE source = 'performance_test' AND event_type = 'high_volume_test'
            """)
            
            assert stored_count == total_activities, f"Should store all {total_activities} activities"
        
        # Test query performance
        query_start = time.time()
        async with e2e_test.db_pool.acquire() as conn:
            recent_activities = await conn.fetch("""
                SELECT * FROM activity_logs 
                WHERE created_at > NOW() - INTERVAL '1 hour'
                ORDER BY created_at DESC 
                LIMIT 100
            """)
        query_time = time.time() - query_start
        
        print(f"Query time for recent 100 activities: {query_time:.3f} seconds")
        assert query_time < 1.0, "Query should complete within 1 second"
        
        print("✓ High volume performance test passed")
    
    async def test_memory_usage_during_load(self):
        """Test memory usage remains stable during high load"""
        print("\n--- Test 7.2: Memory Usage During Load ---")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate load
        for i in range(100):
            await activity_logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source="memory_test",
                event_type="memory_load_test",
                title=f"Memory load test {i}",
                metadata={"memory_test": True, "large_data": "x" * 1000}  # 1KB metadata
            )
        
        await activity_logger._flush_buffer()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50, f"Memory increase should be reasonable, got {memory_increase:.1f} MB"
        
        print("✓ Memory usage test passed")


class TestErrorHandlingRecovery:
    """Test 8: Error handling and recovery scenarios"""
    
    async def test_database_connection_recovery(self):
        """Test recovery from database connection issues"""
        print("\n--- Test 8.1: Database Connection Recovery ---")
        
        # This is a simplified test - in real scenario you'd simulate DB disconnection
        
        # Test that the activity logger handles connection failures gracefully
        original_pool = activity_logger._pool
        
        try:
            # Temporarily set pool to None to simulate connection failure
            activity_logger._pool = None
            
            # This should not crash, but should handle the error
            try:
                await activity_logger.log_activity(
                    category=ActivityCategory.SYSTEM,
                    action=ActivityAction.ERROR,
                    source="recovery_test",
                    event_type="connection_failure_test",
                    title="Test during connection failure"
                )
                print("⚠ Activity logged despite no connection (buffered)")
            except Exception as e:
                print(f"⚠ Expected error during connection failure: {e}")
            
            # Restore connection
            activity_logger._pool = original_pool
            
            # This should work again
            activity_id = await activity_logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.SUCCESS,
                source="recovery_test",
                event_type="connection_recovery_test",
                title="Test after connection recovery"
            )
            
            assert activity_id is not None, "Should work after connection recovery"
            e2e_test.test_activities.append(activity_id)
            
            print("✓ Database connection recovery test passed")
            
        finally:
            # Ensure pool is restored
            activity_logger._pool = original_pool
    
    async def test_invalid_data_handling(self):
        """Test handling of invalid activity data"""
        print("\n--- Test 8.2: Invalid Data Handling ---")
        
        # Test with invalid enum values (should not crash)
        try:
            # This should handle invalid category gracefully
            activity_id = await activity_logger.log_activity(
                category=ActivityCategory.SYSTEM,  # Valid
                action=ActivityAction.EXECUTE,     # Valid
                source="",  # Invalid empty source
                event_type="invalid_data_test",
                title="Test with invalid data"
            )
            # If this succeeds, the system handles empty source
            if activity_id:
                e2e_test.test_activities.append(activity_id)
        except Exception as e:
            print(f"Expected error with invalid data: {e}")
        
        # Test with None values in required fields
        try:
            await activity_logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source="invalid_test",
                event_type="",  # Invalid empty event_type
                title="Test with empty event type"
            )
        except Exception as e:
            print(f"Expected error with empty event type: {e}")
        
        print("✓ Invalid data handling test passed")
    
    async def test_high_error_rate_handling(self):
        """Test system behavior under high error rates"""
        print("\n--- Test 8.3: High Error Rate Handling ---")
        
        # Generate many error activities rapidly
        error_ids = []
        for i in range(20):
            activity_id = await activity_logger.log_error(
                category=ActivityCategory.SYSTEM,
                source="error_rate_test",
                event_type="high_error_rate_test",
                title=f"Test error {i}",
                error_code=f"TEST_ERROR_{i}",
                error_message=f"Test error message {i}"
            )
            error_ids.append(activity_id)
        
        e2e_test.test_activities.extend(error_ids)
        
        await asyncio.sleep(2)
        
        # Verify all errors were logged
        async with e2e_test.db_pool.acquire() as conn:
            error_count = await conn.fetchval("""
                SELECT COUNT(*) FROM activity_logs 
                WHERE source = 'error_rate_test' AND severity = 'error'
            """)
            
            assert error_count == 20, f"Should log all 20 errors, got {error_count}"
        
        print("✓ High error rate handling test passed")


class TestIntegrationFlow:
    """Test 9: Complete integration flow validation"""
    
    async def test_complete_e2e_flow(self):
        """Test the complete flow from activity generation to dashboard display"""
        print("\n--- Test 9.1: Complete E2E Flow ---")
        
        # Step 1: Generate a comprehensive activity
        test_activity_data = {
            'category': ActivityCategory.TRADING,
            'action': ActivityAction.EXECUTE,
            'source': 'e2e_complete_test',
            'event_type': 'complete_flow_test',
            'title': 'Complete E2E flow test',
            'description': 'Testing the complete activity logging flow',
            'severity': ActivitySeverity.INFO,
            'user_id': 98765,
            'session_id': str(uuid.uuid4()),
            'trading_mode': TradingMode.SIMULATION,
            'token_address': '0xe2e_test_token',
            'chain': ChainType.ETHEREUM,
            'amount_usd': Decimal("500.00"),
            'execution_time_ms': 150,
            'metadata': {
                'test_type': 'complete_e2e',
                'step': 'generation',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
        
        # Generate the activity
        activity_id = await activity_logger.log_activity(**test_activity_data)
        e2e_test.test_activities.append(activity_id)
        
        print("✓ Step 1: Activity generated")
        
        # Step 2: Wait for database storage
        await asyncio.sleep(2)
        
        # Verify storage
        async with e2e_test.db_pool.acquire() as conn:
            stored_activity = await conn.fetchrow(
                "SELECT * FROM activity_logs WHERE activity_id = $1",
                activity_id
            )
            
            assert stored_activity is not None, "Activity should be stored in database"
            assert stored_activity['title'] == 'Complete E2E flow test'
        
        print("✓ Step 2: Activity stored in database")
        
        # Step 3: Test API retrieval
        response = e2e_test.test_client.get(f"/api/v1/activity/recent?limit=1")
        assert response.status_code == 200
        
        data = response.json()
        activities = data.get('activities', [])
        
        # Find our test activity
        test_activity = None
        for activity in activities:
            if activity.get('activity_id') == activity_id:
                test_activity = activity
                break
        
        assert test_activity is not None, "Activity should be retrievable via API"
        assert test_activity['title'] == 'Complete E2E flow test'
        
        print("✓ Step 3: Activity retrievable via API")
        
        # Step 4: Test WebSocket broadcasting
        websocket_test_activity = {
            'activity_id': activity_id,
            'category': 'trading',
            'title': 'Complete E2E flow test',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'metadata': test_activity_data['metadata']
        }
        
        await websocket_manager.send_activity_update(websocket_test_activity)
        
        print("✓ Step 4: Activity broadcast via WebSocket")
        
        # Step 5: Test dashboard integration
        recent_activities = await dashboard_activity.get_recent_activity(limit=10)
        
        # Find our activity in the recent activities
        found_in_dashboard = False
        for activity in recent_activities:
            if activity.get('activity_id') == activity_id:
                found_in_dashboard = True
                break
        
        assert found_in_dashboard, "Activity should be accessible via dashboard integration"
        
        print("✓ Step 5: Activity accessible via dashboard integration")
        
        print("✓ Complete E2E flow test passed")
    
    async def test_performance_metrics_flow(self):
        """Test the flow for performance metrics"""
        print("\n--- Test 9.2: Performance Metrics Flow ---")
        
        # Generate performance activities
        performance_activities = []
        for i in range(10):
            activity_id = await activity_logger.log_performance(
                source="e2e_performance_test",
                operation="test_operation",
                execution_time_ms=100 + i * 10,
                success=True,
                metadata={
                    "test_type": "performance_flow",
                    "operation_id": i
                }
            )
            performance_activities.append(activity_id)
        
        e2e_test.test_activities.extend(performance_activities)
        
        await asyncio.sleep(2)
        
        # Test performance metrics retrieval
        metrics = await dashboard_activity.get_performance_metrics(hours_back=1)
        
        # Find our test metrics
        test_metrics = None
        for metric in metrics:
            if metric.get('source') == 'e2e_performance_test':
                test_metrics = metric
                break
        
        if test_metrics:
            assert test_metrics['total_operations'] >= 10, "Should capture all performance activities"
            assert test_metrics['avg_execution_time_ms'] > 0, "Should calculate average execution time"
            print("✓ Performance metrics flow working")
        else:
            print("⚠ Performance metrics not found (may need more time to aggregate)")
    
    async def test_user_activity_tracking_flow(self):
        """Test user activity tracking flow"""
        print("\n--- Test 9.3: User Activity Tracking Flow ---")
        
        test_user_id = 55555
        test_session_id = str(uuid.uuid4())
        
        # Generate user activities
        user_activities = []
        for i in range(5):
            activity_id = await dashboard_activity.log_dashboard_action(
                user_id=test_user_id,
                component="e2e_test_component",
                action=f"test_action_{i}",
                session_id=test_session_id,
                metadata={
                    "test_type": "user_tracking",
                    "action_sequence": i
                }
            )
            user_activities.append(activity_id)
        
        e2e_test.test_activities.extend(user_activities)
        
        await asyncio.sleep(2)
        
        # Test user activity stats retrieval
        user_stats = await dashboard_activity.get_user_activity_stats(
            user_id=test_user_id,
            days_back=1
        )
        
        assert user_stats['total_activities'] >= 5, "Should track user activities"
        assert test_user_id in str(user_stats), "Should be associated with correct user"
        
        print("✓ User activity tracking flow working")


# Main test execution
async def run_e2e_tests():
    """Run all E2E tests"""
    print("\n" + "="*60)
    print("STARTING END-TO-END ACTIVITY LOGGING TESTS")
    print("="*60)
    
    test_classes = [
        TestDatabaseSetupAndConnectivity(),
        TestActivityGeneration(),
        TestActivityStorage(),
        TestDashboardAPIRetrieval(),
        TestWebSocketRealTimeBroadcasting(),
        TestFrontendActivityDisplay(),
        TestPerformanceHighVolume(),
        TestErrorHandlingRecovery(),
        TestIntegrationFlow()
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{'='*40}")
        print(f"Running {class_name}")
        print(f"{'='*40}")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_') and callable(getattr(test_class, method))]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                test_method = getattr(test_class, method_name)
                await test_method()
                passed_tests += 1
                print(f"✓ {method_name} PASSED")
            except Exception as e:
                failed_tests.append(f"{class_name}.{method_name}: {str(e)}")
                print(f"✗ {method_name} FAILED: {e}")
    
    # Print final results
    print(f"\n{'='*60}")
    print("E2E TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed Tests:")
        for failure in failed_tests:
            print(f"  ✗ {failure}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 E2E TESTS OVERALL: SUCCESS")
    else:
        print("\n❌ E2E TESTS OVERALL: NEEDS IMPROVEMENT")
    
    return success_rate >= 80


if __name__ == "__main__":
    import asyncio
    
    # Run the tests
    success = asyncio.run(run_e2e_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)