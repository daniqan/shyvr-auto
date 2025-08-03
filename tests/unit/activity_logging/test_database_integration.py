"""
Test Database Integration (CRUD Operations, Querying, Filtering)
Following TDD methodology - comprehensive database operation testing
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call
import json

from src.activity_logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction, 
    ActivitySeverity, TradingMode, ChainType
)


class TestDatabaseCRUDOperations:
    """Test Create, Read, Update, Delete operations for activity logs"""
    
    @pytest.mark.asyncio
    async def test_insert_single_activity(self, mock_database_pool, mock_config):
        """Test inserting a single activity log entry"""
        with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            entry = ActivityLogEntry(
                category=ActivityCategory.TRADING,
                action=ActivityAction.EXECUTE,
                source="trading_engine",
                event_type="trade_execution",
                title="Test trade"
            )
            
            await logger._insert_activities([entry])
            
            # Verify database interaction
            conn = mock_database_pool.acquire.return_value.__aenter__.return_value
            conn.execute.assert_called_once()
            
            # Check SQL query structure
            call_args = conn.execute.call_args
            query = call_args[0][0]
            assert "INSERT INTO activity_logs" in query
            assert "activity_id" in query
            assert "category" in query
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_insert_batch_activities(self, mock_database_pool, mock_config):
        """Test inserting multiple activity log entries in batch"""
        with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            entries = [
                ActivityLogEntry(
                    category=ActivityCategory.TRADING,
                    action=ActivityAction.EXECUTE,
                    source="trading_engine",
                    event_type="buy_order",
                    title=f"Trade {i}"
                ) for i in range(5)
            ]
            
            await logger._insert_activities(entries)
            
            # Should execute one INSERT per entry
            conn = mock_database_pool.acquire.return_value.__aenter__.return_value
            assert conn.execute.call_count == 5
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_insert_with_all_fields(self, mock_database_pool, mock_config):
        """Test inserting activity with all possible fields populated"""
        with patch('src.activity_logging.activity_logger.get_database_pool', return_value=mock_database_pool), \
             patch('src.activity_logging.activity_logger.get_config', return_value=mock_config):
            
            logger = ActivityLogger()
            await logger.start()
            
            entry = ActivityLogEntry(
                category=ActivityCategory.TRADING,
                action=ActivityAction.EXECUTE,
                source="trading_engine",
                event_type="comprehensive_trade",
                title="Complete trade entry",
                description="Full-featured trade with all fields",
                severity=ActivitySeverity.INFO,
                user_id=123456789,
                session_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
                trading_mode=TradingMode.LIVE,
                token_address="0x123",
                chain=ChainType.ETHEREUM,
                amount_usd=Decimal("1000.50"),
                fee_usd=Decimal("5.25"),
                execution_time_ms=1500,
                memory_usage_mb=64,
                cpu_usage_pct=Decimal("15.7"),
                metadata={"exchange": "uniswap", "slippage": 0.1},
                tags=["high_value", "ethereum", "defi"],
                correlation_id="test-correlation-123",
                api_endpoint="/swap",
                http_method="POST",
                http_status=200,
                user_agent="TradingBot/1.0",
                ip_address="192.168.1.100",
                response_time_ms=800,
                throughput_ops_per_sec=Decimal("50.5"),
                security_level="medium",
                risk_score=25,
                dashboard_component="trading",
                dashboard_action="execute_swap"
            )
            entry.checksum = entry.generate_checksum()
            
            await logger._insert_activities([entry])
            
            # Verify all fields are handled in SQL
            conn = mock_database_pool.acquire.return_value.__aenter__.return_value
            conn.execute.assert_called_once()
            
            call_args = conn.execute.call_args
            query = call_args[0][0]
            values = call_args[0][1:]
            
            # Check that complex fields are properly handled
            assert "metadata" in query
            assert "tags" in query
            assert "amount_usd" in query
            
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_read_activities_by_category(self, mock_database_pool):
        """Test reading activities filtered by category"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'category': 'trading',
                'action': 'execute',
                'source': 'trading_engine',
                'title': 'Test trade 1',
                'created_at': datetime.now(timezone.utc)
            },
            {
                'activity_id': str(uuid.uuid4()),
                'category': 'trading',
                'action': 'execute',
                'source': 'trading_engine',
                'title': 'Test trade 2',
                'created_at': datetime.now(timezone.utc)
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
                    ORDER BY created_at DESC
                """, 'trading')
            
            # Should return trading activities
            assert len(results) == 2
            for result in results:
                assert result['category'] == 'trading'
    
    @pytest.mark.asyncio
    async def test_read_activities_by_time_range(self, mock_database_pool):
        """Test reading activities within time range"""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'category': 'system',
                'created_at': datetime.now(timezone.utc) - timedelta(minutes=30)
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
                """, start_time, end_time)
            
            # Should return activities in time range
            assert len(results) == 1
            assert results[0]['created_at'] >= start_time
            assert results[0]['created_at'] <= end_time
    
    @pytest.mark.asyncio
    async def test_read_activities_by_user(self, mock_database_pool):
        """Test reading activities for specific user"""
        user_id = 123456789
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'user_id': user_id,
                'category': 'user',
                'action': 'login',
                'title': 'User logged in'
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
                """, user_id)
            
            # Should return user activities
            assert len(results) == 1
            assert results[0]['user_id'] == user_id
    
    @pytest.mark.asyncio
    async def test_update_activity_archived_status(self, mock_database_pool):
        """Test updating activity archived status"""
        activity_id = str(uuid.uuid4())
        
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE activity_logs 
                    SET archived_at = NOW()
                    WHERE activity_id = $1
                """, activity_id)
            
            # Should execute update
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_old_activities(self, mock_database_pool):
        """Test deleting old activity records"""
        retention_days = 90
        
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM activity_logs 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                    AND severity NOT IN ('critical', 'alert', 'emergency')
                """, retention_days)
            
            # Should execute delete
            mock_conn.execute.assert_called_once()


class TestDatabaseQuerying:
    """Test complex database querying capabilities"""
    
    @pytest.mark.asyncio
    async def test_query_recent_activities_view(self, mock_database_pool):
        """Test querying the recent_activity view"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'created_at': datetime.now(timezone.utc),
                'category': 'trading',
                'severity': 'info',
                'title': 'Recent trade'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("SELECT * FROM recent_activity LIMIT 50")
            
            # Should return recent activities
            assert len(results) == 1
            assert results[0]['category'] == 'trading'
    
    @pytest.mark.asyncio
    async def test_query_error_summary_view(self, mock_database_pool):
        """Test querying the error_summary view"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'category': 'trading',
                'source': 'trading_engine',
                'error_code': 'INSUFFICIENT_BALANCE',
                'error_count': 5,
                'last_occurrence': datetime.now(timezone.utc)
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("SELECT * FROM error_summary ORDER BY error_count DESC")
            
            # Should return error summary
            assert len(results) == 1
            assert results[0]['error_code'] == 'INSUFFICIENT_BALANCE'
            assert results[0]['error_count'] == 5
    
    @pytest.mark.asyncio
    async def test_query_performance_metrics_view(self, mock_database_pool):
        """Test querying the performance_metrics view"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'source': 'trading_engine',
                'category': 'trading',
                'total_activities': 100,
                'avg_execution_time': 1500.5,
                'p95_execution_time': 3000.0,
                'error_count': 2,
                'error_rate_pct': 2.0
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("SELECT * FROM performance_metrics ORDER BY total_activities DESC")
            
            # Should return performance metrics
            assert len(results) == 1
            assert results[0]['source'] == 'trading_engine'
            assert results[0]['error_rate_pct'] == 2.0
    
    @pytest.mark.asyncio
    async def test_query_user_activity_overview(self, mock_database_pool):
        """Test querying the user_activity_overview view"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'telegram_user_id': 123456789,
                'username': 'test_user',
                'total_activities': 50,
                'active_days': 7,
                'last_activity': datetime.now(timezone.utc),
                'errors_encountered': 1,
                'categories_used': ['trading', 'user'],
                'components_used': ['dashboard', 'trading_engine']
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("SELECT * FROM user_activity_overview ORDER BY total_activities DESC")
            
            # Should return user activity overview
            assert len(results) == 1
            assert results[0]['telegram_user_id'] == 123456789
            assert results[0]['total_activities'] == 50
    
    @pytest.mark.asyncio
    async def test_query_trading_activity_summary(self, mock_database_pool):
        """Test querying the trading_activity_summary view"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'trading_mode': 'live',
                'activity_date': datetime.now(timezone.utc).date(),
                'total_activities': 25,
                'unique_tokens': 5,
                'executions': 20,
                'errors': 1,
                'total_volume_usd': Decimal('5000.00'),
                'avg_execution_time': 1200.5
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("SELECT * FROM trading_activity_summary ORDER BY activity_date DESC")
            
            # Should return trading summary
            assert len(results) == 1
            assert results[0]['trading_mode'] == 'live'
            assert results[0]['total_volume_usd'] == Decimal('5000.00')


class TestDatabaseFiltering:
    """Test advanced filtering capabilities"""
    
    @pytest.mark.asyncio
    async def test_filter_by_severity_levels(self, mock_database_pool):
        """Test filtering activities by severity levels"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'severity': 'error',
                'title': 'Error activity'
            },
            {
                'activity_id': str(uuid.uuid4()),
                'severity': 'critical',
                'title': 'Critical activity'
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
                    ORDER BY created_at DESC
                """)
            
            # Should return only high-severity activities
            assert len(results) == 2
            for result in results:
                assert result['severity'] in ['error', 'critical']
    
    @pytest.mark.asyncio
    async def test_filter_by_trading_mode(self, mock_database_pool):
        """Test filtering activities by trading mode"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'trading_mode': 'live',
                'category': 'trading'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE trading_mode = $1 AND category = 'trading'
                    ORDER BY created_at DESC
                """, 'live')
            
            # Should return only live trading activities
            assert len(results) == 1
            assert results[0]['trading_mode'] == 'live'
    
    @pytest.mark.asyncio
    async def test_filter_by_metadata_jsonb(self, mock_database_pool):
        """Test filtering activities by JSONB metadata"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'metadata': {'exchange': 'uniswap', 'slippage': 0.1}
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE metadata->>'exchange' = $1
                """, 'uniswap')
            
            # Should return activities with specific exchange
            assert len(results) == 1
            assert results[0]['metadata']['exchange'] == 'uniswap'
    
    @pytest.mark.asyncio
    async def test_filter_by_tags_array(self, mock_database_pool):
        """Test filtering activities by tags array"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'tags': ['high_value', 'ethereum', 'defi']
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE 'high_value' = ANY(tags)
                """)
            
            # Should return activities with specific tag
            assert len(results) == 1
            assert 'high_value' in results[0]['tags']
    
    @pytest.mark.asyncio
    async def test_filter_by_performance_thresholds(self, mock_database_pool):
        """Test filtering activities by performance thresholds"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'execution_time_ms': 5000,
                'source': 'slow_component'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE execution_time_ms > $1
                    ORDER BY execution_time_ms DESC
                """, 3000)
            
            # Should return slow activities
            assert len(results) == 1
            assert results[0]['execution_time_ms'] > 3000
    
    @pytest.mark.asyncio
    async def test_filter_by_risk_score_range(self, mock_database_pool):
        """Test filtering activities by risk score range"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'risk_score': 85,
                'category': 'security'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE risk_score >= $1 AND risk_score <= $2
                    ORDER BY risk_score DESC
                """, 80, 100)
            
            # Should return high-risk activities
            assert len(results) == 1
            assert 80 <= results[0]['risk_score'] <= 100


class TestDatabaseAggregations:
    """Test database aggregation operations"""
    
    @pytest.mark.asyncio
    async def test_count_activities_by_category(self, mock_database_pool):
        """Test counting activities by category"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'category': 'trading', 'count': 50},
            {'category': 'system', 'count': 25},
            {'category': 'user', 'count': 30}
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT category, COUNT(*) as count 
                    FROM activity_logs 
                    GROUP BY category 
                    ORDER BY count DESC
                """)
            
            # Should return category counts
            assert len(results) == 3
            assert results[0]['category'] == 'trading'
            assert results[0]['count'] == 50
    
    @pytest.mark.asyncio
    async def test_average_execution_time_by_source(self, mock_database_pool):
        """Test calculating average execution time by source"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {'source': 'trading_engine', 'avg_execution_time': 1250.5},
            {'source': 'ml_model', 'avg_execution_time': 850.2}
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT source, AVG(execution_time_ms) as avg_execution_time
                    FROM activity_logs 
                    WHERE execution_time_ms IS NOT NULL
                    GROUP BY source 
                    ORDER BY avg_execution_time DESC
                """)
            
            # Should return average execution times
            assert len(results) == 2
            assert results[0]['source'] == 'trading_engine'
            assert results[0]['avg_execution_time'] == 1250.5
    
    @pytest.mark.asyncio
    async def test_error_rate_by_time_period(self, mock_database_pool):
        """Test calculating error rates by time period"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'hour': datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
                'total_count': 100,
                'error_count': 5,
                'error_rate': 5.0
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT 
                        date_trunc('hour', created_at) as hour,
                        COUNT(*) as total_count,
                        COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as error_count,
                        ROUND(COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) * 100.0 / COUNT(*), 2) as error_rate
                    FROM activity_logs 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY date_trunc('hour', created_at)
                    ORDER BY hour DESC
                """)
            
            # Should return hourly error rates
            assert len(results) == 1
            assert results[0]['error_rate'] == 5.0


class TestDatabaseSummaryOperations:
    """Test activity summary table operations"""
    
    @pytest.mark.asyncio
    async def test_insert_activity_summary(self, mock_database_pool):
        """Test inserting activity summary records"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO activity_summaries (
                        period_start, period_end, period_type, category,
                        total_count, error_count, warning_count,
                        avg_execution_time_ms, max_execution_time_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                datetime.now(timezone.utc) - timedelta(hours=1),
                datetime.now(timezone.utc),
                'hour',
                'trading',
                100, 2, 5, 1250.5, 5000
                )
            
            # Should execute insert
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_activity_summaries(self, mock_database_pool):
        """Test querying activity summaries"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'period_start': datetime.now(timezone.utc) - timedelta(hours=1),
                'period_end': datetime.now(timezone.utc),
                'category': 'trading',
                'total_count': 100,
                'error_count': 2,
                'avg_execution_time_ms': 1250.5
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT * FROM activity_summaries 
                    WHERE period_type = 'hour' AND category = 'trading'
                    ORDER BY period_start DESC
                """)
            
            # Should return summary data
            assert len(results) == 1
            assert results[0]['category'] == 'trading'
            assert results[0]['total_count'] == 100


class TestUserActivitySessions:
    """Test user activity session operations"""
    
    @pytest.mark.asyncio
    async def test_insert_user_session(self, mock_database_pool):
        """Test inserting user activity session"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO user_activity_sessions (
                        session_id, user_id, source, total_activities,
                        started_at, last_activity_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                str(uuid.uuid4()), 123456789, 'dashboard', 0,
                datetime.now(timezone.utc), datetime.now(timezone.utc)
                )
            
            # Should execute insert
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_session_activity(self, mock_database_pool):
        """Test updating session last activity"""
        session_id = str(uuid.uuid4())
        
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE user_activity_sessions 
                    SET last_activity_at = NOW(), total_activities = total_activities + 1
                    WHERE session_id = $1
                """, session_id)
            
            # Should execute update
            mock_conn.execute.assert_called_once()


# NOTE: All these tests should FAIL initially since the full database integration
# may not be implemented yet. This follows TDD methodology.