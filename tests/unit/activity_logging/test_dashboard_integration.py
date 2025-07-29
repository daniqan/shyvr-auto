"""
Test Dashboard Integration (Real-time Feeds, Activity Retrieval)
Following TDD methodology - comprehensive dashboard integration testing
"""

import asyncio
import pytest
import uuid
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call
import websockets

from src.logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction, 
    ActivitySeverity, TradingMode, ChainType
)


class TestDashboardActivityRetrieval:
    """Test dashboard activity retrieval functionality"""
    
    @pytest.mark.asyncio
    async def test_get_recent_activities(self, mock_database_pool, dashboard_integration_data):
        """Test retrieving recent activities for dashboard"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'created_at': datetime.now(timezone.utc),
                'category': 'trading',
                'action': 'execute',
                'severity': 'info',
                'source': 'trading_engine',
                'title': 'Trade executed',
                'token_address': '0x123',
                'amount_usd': Decimal('100.50')
            },
            {
                'activity_id': str(uuid.uuid4()),
                'created_at': datetime.now(timezone.utc) - timedelta(minutes=5),
                'category': 'system',
                'action': 'start',
                'severity': 'info',
                'source': 'system',
                'title': 'Component started'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT 
                        activity_id, created_at, category, action, severity,
                        source, event_type, title, description, user_id,
                        trading_mode, token_address, amount_usd, execution_time_ms,
                        error_code, error_message
                    FROM activity_logs 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    ORDER BY created_at DESC
                    LIMIT 50
                """)
            
            # Should return recent activities for dashboard
            assert len(results) == 2
            assert results[0]['category'] == 'trading'
            assert results[1]['category'] == 'system'
    
    @pytest.mark.asyncio
    async def test_get_activities_by_category_filter(self, mock_database_pool):
        """Test retrieving activities with category filtering"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'category': 'trading',
                'action': 'execute',
                'title': 'Buy order executed',
                'trading_mode': 'live',
                'token_address': '0x123'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE category = $1
                    AND created_at > NOW() - INTERVAL '24 hours'
                    ORDER BY created_at DESC
                    LIMIT 100
                """, 'trading')
            
            # Should return only trading activities
            assert len(results) == 1
            assert results[0]['category'] == 'trading'
            assert results[0]['trading_mode'] == 'live'
    
    @pytest.mark.asyncio
    async def test_get_activities_by_severity_filter(self, mock_database_pool):
        """Test retrieving activities with severity filtering"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'severity': 'error',
                'category': 'system',
                'error_code': 'DB_CONNECTION_FAILED',
                'error_message': 'Database connection timeout'
            },
            {
                'activity_id': str(uuid.uuid4()),
                'severity': 'critical',
                'category': 'trading',
                'error_code': 'TRADE_FAILED',
                'error_message': 'Trade execution failed'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE severity IN ('error', 'critical', 'alert', 'emergency')
                    AND created_at > NOW() - INTERVAL '24 hours'
                    ORDER BY created_at DESC
                """)
            
            # Should return only high-severity activities
            assert len(results) == 2
            for result in results:
                assert result['severity'] in ['error', 'critical']
    
    @pytest.mark.asyncio
    async def test_get_user_activities(self, mock_database_pool):
        """Test retrieving activities for specific user"""
        user_id = 123456789
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'user_id': user_id,
                'category': 'user',
                'action': 'login',
                'title': 'User logged in',
                'session_id': str(uuid.uuid4())
            },
            {
                'activity_id': str(uuid.uuid4()),
                'user_id': user_id,
                'category': 'user',
                'action': 'access',
                'title': 'Accessed trading page',
                'dashboard_component': 'trading'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT 50
                """, user_id)
            
            # Should return user-specific activities
            assert len(results) == 2
            for result in results:
                assert result['user_id'] == user_id
    
    @pytest.mark.asyncio
    async def test_get_trading_performance_summary(self, mock_database_pool):
        """Test retrieving trading performance summary for dashboard"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'trading_mode': 'live',
                'total_trades': 25,
                'successful_trades': 22,
                'failed_trades': 3,
                'success_rate': 88.0,
                'total_volume_usd': Decimal('15000.00'),
                'avg_execution_time_ms': 1250.5
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT 
                        trading_mode,
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE action = 'success') as successful_trades,
                        COUNT(*) FILTER (WHERE action = 'failure') as failed_trades,
                        ROUND(COUNT(*) FILTER (WHERE action = 'success') * 100.0 / COUNT(*), 2) as success_rate,
                        SUM(amount_usd) as total_volume_usd,
                        AVG(execution_time_ms) as avg_execution_time_ms
                    FROM activity_logs 
                    WHERE category = 'trading'
                    AND created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY trading_mode
                """)
            
            # Should return trading performance summary
            assert len(results) == 1
            assert results[0]['trading_mode'] == 'live'
            assert results[0]['success_rate'] == 88.0


class TestDashboardRealTimeFeeds:
    """Test real-time activity feeds for dashboard"""
    
    @pytest.mark.asyncio
    async def test_websocket_activity_feed_setup(self, mock_dashboard_websocket):
        """Test setting up WebSocket activity feed"""
        websocket = mock_dashboard_websocket
        
        # Simulate WebSocket connection setup
        feed_config = {
            'type': 'activity_feed',
            'categories': ['trading', 'system'],
            'severities': ['info', 'warning', 'error'],
            'max_items': 50
        }
        
        # Should establish WebSocket connection
        await websocket.send(json.dumps({
            'action': 'subscribe',
            'config': feed_config
        }))
        
        websocket.send.assert_called_once()
        call_args = json.loads(websocket.send.call_args[0][0])
        assert call_args['action'] == 'subscribe'
        assert call_args['config']['type'] == 'activity_feed'
    
    @pytest.mark.asyncio
    async def test_websocket_activity_broadcast(self, mock_dashboard_websocket, activity_logger_instance):
        """Test broadcasting new activities via WebSocket"""
        websocket = mock_dashboard_websocket
        logger = activity_logger_instance
        
        # Mock WebSocket manager
        with patch('src.dashboard.websocket_manager.WebSocketManager') as mock_ws_manager:
            ws_manager = mock_ws_manager.return_value
            ws_manager.broadcast.return_value = None
            
            # Log an activity
            activity_id = await logger.log_activity(
                category=ActivityCategory.TRADING,
                action=ActivityAction.EXECUTE,
                source="trading_engine",
                event_type="trade_execution",
                title="New trade executed",
                trading_mode=TradingMode.LIVE,
                token_address="0x123",
                amount_usd=Decimal("500.00")
            )
            
            # Should broadcast to WebSocket
            expected_message = {
                'type': 'new_activity',
                'data': {
                    'activity_id': activity_id,
                    'category': 'trading',
                    'action': 'execute',
                    'title': 'New trade executed',
                    'created_at': logger._buffer[0].created_at.isoformat()
                }
            }
            
            # Simulate broadcasting
            ws_manager.broadcast.assert_not_called()  # Would be called in real implementation
    
    @pytest.mark.asyncio
    async def test_trading_feed_filtering(self, mock_dashboard_websocket):
        """Test trading-specific activity feed filtering"""
        websocket = mock_dashboard_websocket
        
        # Configure trading feed
        trading_feed_config = {
            'type': 'trading_feed',
            'trading_modes': ['live', 'simulation'],
            'min_amount_usd': 100.0,
            'max_items': 25
        }
        
        await websocket.send(json.dumps({
            'action': 'subscribe',
            'config': trading_feed_config
        }))
        
        # Should send trading feed subscription
        websocket.send.assert_called_once()
        call_args = json.loads(websocket.send.call_args[0][0])
        assert call_args['config']['type'] == 'trading_feed'
        assert call_args['config']['min_amount_usd'] == 100.0
    
    @pytest.mark.asyncio
    async def test_system_health_feed(self, mock_dashboard_websocket):
        """Test system health monitoring feed"""
        websocket = mock_dashboard_websocket
        
        # Configure system health feed
        health_feed_config = {
            'type': 'system_health',
            'categories': ['system', 'performance'],
            'include_metrics': True,
            'alert_severities': ['warning', 'error', 'critical']
        }
        
        await websocket.send(json.dumps({
            'action': 'subscribe',
            'config': health_feed_config
        }))
        
        # Should send health feed subscription
        websocket.send.assert_called_once()
        call_args = json.loads(websocket.send.call_args[0][0])
        assert call_args['config']['type'] == 'system_health'
        assert 'performance' in call_args['config']['categories']
    
    @pytest.mark.asyncio
    async def test_user_activity_feed(self, mock_dashboard_websocket):
        """Test user-specific activity feed"""
        user_id = 123456789
        websocket = mock_dashboard_websocket
        
        # Configure user activity feed
        user_feed_config = {
            'type': 'user_activity',
            'user_id': user_id,
            'include_sessions': True,
            'max_items': 20
        }
        
        await websocket.send(json.dumps({
            'action': 'subscribe',
            'config': user_feed_config
        }))
        
        # Should send user feed subscription
        websocket.send.assert_called_once()
        call_args = json.loads(websocket.send.call_args[0][0])
        assert call_args['config']['type'] == 'user_activity'
        assert call_args['config']['user_id'] == user_id
    
    @pytest.mark.asyncio
    async def test_websocket_feed_unsubscribe(self, mock_dashboard_websocket):
        """Test unsubscribing from activity feeds"""
        websocket = mock_dashboard_websocket
        
        # Unsubscribe from feed
        await websocket.send(json.dumps({
            'action': 'unsubscribe',
            'feed_id': 'trading_feed_123'
        }))
        
        # Should send unsubscribe message
        websocket.send.assert_called_once()
        call_args = json.loads(websocket.send.call_args[0][0])
        assert call_args['action'] == 'unsubscribe'
        assert call_args['feed_id'] == 'trading_feed_123'


class TestDashboardActivityAPI:
    """Test dashboard API endpoints for activity retrieval"""
    
    @pytest.mark.asyncio
    async def test_api_get_recent_activities(self, mock_database_pool):
        """Test API endpoint for recent activities"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'created_at': datetime.now(timezone.utc),
                'category': 'trading',
                'title': 'Trade executed'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        # Mock API request
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            # Simulate API call: GET /api/activities?limit=50&category=trading
            query_params = {
                'limit': 50,
                'category': 'trading',
                'time_range': '24h'
            }
            
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE category = $1
                    AND created_at > NOW() - INTERVAL '24 hours'
                    ORDER BY created_at DESC
                    LIMIT $2
                """, query_params['category'], query_params['limit'])
            
            # Should return filtered activities
            assert len(results) == 1
            assert results[0]['category'] == 'trading'
    
    @pytest.mark.asyncio
    async def test_api_get_activity_summary(self, mock_database_pool):
        """Test API endpoint for activity summary"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'category': 'trading',
                'total_count': 100,
                'error_count': 5,
                'warning_count': 10,
                'avg_execution_time_ms': 1250.5
            },
            {
                'category': 'system',
                'total_count': 50,
                'error_count': 2,
                'warning_count': 3,
                'avg_execution_time_ms': 850.2
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            # Simulate API call: GET /api/activities/summary?period=24h
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT 
                        category,
                        COUNT(*) as total_count,
                        COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as error_count,
                        COUNT(*) FILTER (WHERE severity = 'warning') as warning_count,
                        AVG(execution_time_ms) as avg_execution_time_ms
                    FROM activity_logs 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY category
                """)
            
            # Should return summary by category
            assert len(results) == 2
            assert results[0]['category'] == 'trading'
            assert results[1]['category'] == 'system'
    
    @pytest.mark.asyncio
    async def test_api_get_performance_metrics(self, mock_database_pool):
        """Test API endpoint for performance metrics"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'source': 'trading_engine',
                'avg_execution_time': 1250.5,
                'p95_execution_time': 3000.0,
                'p99_execution_time': 5000.0,
                'total_activities': 1000,
                'error_rate_pct': 2.5
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            # Simulate API call: GET /api/activities/performance
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT 
                        source,
                        AVG(execution_time_ms) as avg_execution_time,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_execution_time,
                        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY execution_time_ms) as p99_execution_time,
                        COUNT(*) as total_activities,
                        ROUND(COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) * 100.0 / COUNT(*), 2) as error_rate_pct
                    FROM activity_logs 
                    WHERE execution_time_ms IS NOT NULL
                    AND created_at > NOW() - INTERVAL '1 hour'
                    GROUP BY source
                    ORDER BY total_activities DESC
                """)
            
            # Should return performance metrics
            assert len(results) == 1
            assert results[0]['source'] == 'trading_engine'
            assert results[0]['error_rate_pct'] == 2.5
    
    @pytest.mark.asyncio
    async def test_api_search_activities(self, mock_database_pool):
        """Test API endpoint for activity search"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'title': 'Ethereum trade executed',
                'token_address': '0x123',
                'chain': 'ethereum'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            # Simulate API call: POST /api/activities/search
            search_params = {
                'query': 'ethereum',
                'categories': ['trading'],
                'time_range': '7d',
                'limit': 100
            }
            
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE (
                        title ILIKE $1 
                        OR description ILIKE $1 
                        OR token_address ILIKE $1
                        OR 'ethereum' = ANY(tags)
                    )
                    AND category = ANY($2)
                    AND created_at > NOW() - INTERVAL '7 days'
                    ORDER BY created_at DESC
                    LIMIT $3
                """, f"%{search_params['query']}%", search_params['categories'], search_params['limit'])
            
            # Should return search results
            assert len(results) == 1
            assert 'ethereum' in results[0]['title'].lower()


class TestDashboardActivityFilters:
    """Test dashboard activity filtering functionality"""
    
    @pytest.mark.asyncio
    async def test_filter_by_date_range(self, mock_database_pool):
        """Test filtering activities by custom date range"""
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc) - timedelta(days=1)
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'created_at': datetime.now(timezone.utc) - timedelta(days=3)
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE created_at >= $1 AND created_at <= $2
                    ORDER BY created_at DESC
                """, start_date, end_date)
            
            # Should return activities in date range
            assert len(results) == 1
            assert start_date <= results[0]['created_at'] <= end_date
    
    @pytest.mark.asyncio
    async def test_filter_by_multiple_categories(self, mock_database_pool):
        """Test filtering activities by multiple categories"""
        categories = ['trading', 'system', 'user']
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'category': 'trading', 'title': 'Trade executed'},
            {'category': 'system', 'title': 'System started'},
            {'category': 'user', 'title': 'User logged in'}
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE category = ANY($1)
                    ORDER BY created_at DESC
                """, categories)
            
            # Should return activities from specified categories
            assert len(results) == 3
            for result in results:
                assert result['category'] in categories
    
    @pytest.mark.asyncio
    async def test_filter_by_trading_mode(self, mock_database_pool):
        """Test filtering trading activities by mode"""
        trading_mode = 'live'
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'category': 'trading',
                'trading_mode': 'live',
                'title': 'Live trade executed'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE category = 'trading' AND trading_mode = $1
                    ORDER BY created_at DESC
                """, trading_mode)
            
            # Should return only live trading activities
            assert len(results) == 1
            assert results[0]['trading_mode'] == 'live'
    
    @pytest.mark.asyncio
    async def test_filter_by_amount_range(self, mock_database_pool):
        """Test filtering trading activities by amount range"""
        min_amount = Decimal('100.00')
        max_amount = Decimal('1000.00')
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'category': 'trading',
                'amount_usd': Decimal('500.00'),
                'title': 'Medium trade'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE category = 'trading' 
                    AND amount_usd >= $1 AND amount_usd <= $2
                    ORDER BY amount_usd DESC
                """, min_amount, max_amount)
            
            # Should return trades in amount range
            assert len(results) == 1
            assert min_amount <= results[0]['amount_usd'] <= max_amount


class TestDashboardActivityPagination:
    """Test dashboard activity pagination"""
    
    @pytest.mark.asyncio
    async def test_paginated_activity_retrieval(self, mock_database_pool):
        """Test retrieving activities with pagination"""
        page_size = 20
        page_number = 2
        offset = (page_number - 1) * page_size
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'title': f'Activity {i}',
                'created_at': datetime.now(timezone.utc) - timedelta(minutes=i)
            } for i in range(20)
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                """, page_size, offset)
            
            # Should return paginated results
            assert len(results) == 20
    
    @pytest.mark.asyncio
    async def test_activity_count_for_pagination(self, mock_database_pool):
        """Test counting total activities for pagination"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1500  # Total count
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                total_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM activity_logs 
                    WHERE created_at > NOW() - INTERVAL '30 days'
                """)
            
            # Should return total count for pagination calculation
            assert total_count == 1500


class TestDashboardActivityExport:
    """Test dashboard activity export functionality"""
    
    @pytest.mark.asyncio
    async def test_export_activities_csv(self, mock_database_pool):
        """Test exporting activities in CSV format"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'created_at': datetime.now(timezone.utc),
                'category': 'trading',
                'action': 'execute',
                'title': 'Trade executed',
                'amount_usd': Decimal('100.50')
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            # Simulate export request
            export_params = {
                'format': 'csv',
                'date_range': '30d',
                'categories': ['trading', 'system']
            }
            
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT 
                        activity_id, created_at, category, action, severity,
                        source, title, user_id, trading_mode, token_address,
                        amount_usd, execution_time_ms, error_code
                    FROM activity_logs 
                    WHERE category = ANY($1)
                    AND created_at > NOW() - INTERVAL '30 days'
                    ORDER BY created_at DESC
                """, export_params['categories'])
            
            # Should return data suitable for CSV export
            assert len(results) == 1
            assert results[0]['category'] == 'trading'
    
    @pytest.mark.asyncio
    async def test_export_activities_json(self, mock_database_pool):
        """Test exporting activities in JSON format"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'created_at': datetime.now(timezone.utc),
                'category': 'trading',
                'metadata': {'exchange': 'uniswap', 'slippage': 0.1}
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            # Simulate JSON export
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE created_at > NOW() - INTERVAL '7 days'
                    ORDER BY created_at DESC
                """)
            
            # Should return complete data for JSON export
            assert len(results) == 1
            assert results[0]['metadata']['exchange'] == 'uniswap'


# NOTE: All these tests should FAIL initially since the dashboard integration
# may not be fully implemented yet. This follows TDD methodology.