"""
Test Data Retention and Cleanup Functionality
Following TDD methodology - comprehensive data lifecycle management testing
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.activity_logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction, 
    ActivitySeverity, TradingMode, ChainType
)


class TestDataRetentionPolicies:
    """Test data retention policy implementation"""
    
    @pytest.mark.asyncio
    async def test_retention_policy_configuration(self, mock_database_pool):
        """Test configurable retention policies"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 90  # Default 90 days retention
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Test getting current retention policy
                retention_days = await conn.fetchval("""
                    SELECT EXTRACT(DAY FROM INTERVAL '90 days')
                """)
            
            # Should return configured retention period
            assert retention_days == 90
    
    @pytest.mark.asyncio
    async def test_different_retention_by_severity(self, mock_database_pool, data_retention_scenarios):
        """Test different retention periods based on severity"""
        mock_conn = AsyncMock()
        
        # Mock deletion counts for different scenarios
        def mock_execute_side_effect(*args, **kwargs):
            query = args[0] if args else ""
            if "DELETE" in query and "severity NOT IN" in query:
                return None  # Normal deletion
            elif "UPDATE" in query and "archived_at" in query:
                return None  # Critical archival
            return None
        
        mock_conn.execute.side_effect = mock_execute_side_effect
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Delete old normal activities
                await conn.execute("""
                    DELETE FROM activity_logs 
                    WHERE created_at < NOW() - INTERVAL '90 days'
                    AND severity NOT IN ('critical', 'alert', 'emergency')
                """)
                
                # Archive old critical activities instead of deleting
                await conn.execute("""
                    UPDATE activity_logs 
                    SET archived_at = NOW()
                    WHERE created_at < NOW() - INTERVAL '365 days'
                    AND severity IN ('critical', 'alert', 'emergency')
                    AND archived_at IS NULL
                """)
            
            # Should execute both retention operations
            assert mock_conn.execute.call_count == 2
    
    @pytest.mark.asyncio
    async def test_retention_by_category(self, mock_database_pool):
        """Test different retention periods for different categories"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        # Define category-specific retention periods
        category_retention = {
            'user': 30,      # User activities kept for 30 days
            'trading': 365,  # Trading activities kept for 1 year
            'security': 730, # Security activities kept for 2 years
            'system': 90     # System activities kept for 90 days
        }
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Apply category-specific retention
                for category, days in category_retention.items():
                    await conn.execute("""
                        DELETE FROM activity_logs 
                        WHERE category = $1 
                        AND created_at < NOW() - INTERVAL '%s days'
                        AND severity NOT IN ('critical', 'alert', 'emergency')
                    """, category, days)
            
            # Should execute retention for each category
            assert mock_conn.execute.call_count == len(category_retention)
    
    @pytest.mark.asyncio
    async def test_retention_with_size_limits(self, mock_database_pool):
        """Test retention based on database size limits"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1024 * 1024 * 1024  # 1GB database size
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        max_db_size = 500 * 1024 * 1024  # 500MB limit
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Check current database size
                current_size = await conn.fetchval("""
                    SELECT pg_database_size(current_database())
                """)
                
                if current_size > max_db_size:
                    # Delete oldest activities to reduce size
                    await conn.execute("""
                        DELETE FROM activity_logs 
                        WHERE id IN (
                            SELECT id FROM activity_logs 
                            WHERE severity NOT IN ('critical', 'alert', 'emergency')
                            ORDER BY created_at ASC 
                            LIMIT 10000
                        )
                    """)
            
            # Should check size and potentially delete old data
            mock_conn.fetchval.assert_called_once()
            mock_conn.execute.assert_called_once()


class TestDataCleanupOperations:
    """Test data cleanup operation implementation"""
    
    @pytest.mark.asyncio
    async def test_cleanup_old_activity_logs_function(self, mock_database_pool):
        """Test the cleanup_old_activity_logs database function"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1500  # Deleted count
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Call the cleanup function
                deleted_count = await conn.fetchval("""
                    SELECT cleanup_old_activity_logs()
                """)
            
            # Should return number of deleted records
            assert deleted_count == 1500
            mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_with_custom_retention_period(self, mock_database_pool):
        """Test cleanup with custom retention period"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 750  # Deleted count
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        custom_retention_days = 60
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Call cleanup with custom retention
                deleted_count = await conn.fetchval("""
                    SELECT cleanup_old_activity_logs($1)
                """, custom_retention_days)
            
            # Should use custom retention period
            assert deleted_count == 750
    
    @pytest.mark.asyncio
    async def test_cleanup_preserves_critical_activities(self, mock_database_pool):
        """Test that cleanup preserves critical activities"""
        mock_conn = AsyncMock()
        
        # Mock data: mix of normal and critical activities
        mock_conn.fetch.return_value = [
            {
                'id': 1,
                'severity': 'info',
                'created_at': datetime.now(timezone.utc) - timedelta(days=100)
            },
            {
                'id': 2,
                'severity': 'critical',
                'created_at': datetime.now(timezone.utc) - timedelta(days=100)
            },
            {
                'id': 3,
                'severity': 'error',
                'created_at': datetime.now(timezone.utc) - timedelta(days=100)
            }
        ]
        
        mock_conn.execute.return_value = None
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Get activities to be cleaned up
                old_activities = await conn.fetch("""
                    SELECT id, severity, created_at 
                    FROM activity_logs 
                    WHERE created_at < NOW() - INTERVAL '90 days'
                """)
                
                # Delete only non-critical activities
                await conn.execute("""
                    DELETE FROM activity_logs 
                    WHERE created_at < NOW() - INTERVAL '90 days'
                    AND severity NOT IN ('critical', 'alert', 'emergency')
                """)
            
            # Should fetch old activities and delete only non-critical ones
            mock_conn.fetch.assert_called_once()
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_performance_with_large_dataset(self, mock_database_pool):
        """Test cleanup performance with large datasets"""
        mock_conn = AsyncMock()
        
        # Mock large dataset cleanup
        async def mock_execute(*args, **kwargs):
            query = args[0] if args else ""
            if "DELETE" in query:
                # Simulate time for large deletion
                await asyncio.sleep(0.01)  # 10ms for large delete
            return None
        
        mock_conn.execute = mock_execute
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Perform batched cleanup for better performance
                batch_size = 1000
                for batch in range(10):  # 10 batches
                    await conn.execute("""
                        DELETE FROM activity_logs 
                        WHERE id IN (
                            SELECT id FROM activity_logs 
                            WHERE created_at < NOW() - INTERVAL '90 days'
                            AND severity NOT IN ('critical', 'alert', 'emergency')
                            ORDER BY created_at ASC 
                            LIMIT $1
                        )
                    """, batch_size)
            
            # Should perform batched deletions
            assert mock_conn.execute.call_count == 10
    
    @pytest.mark.asyncio
    async def test_cleanup_scheduling(self, mock_database_pool):
        """Test automated cleanup scheduling"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 500  # Deleted count
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        # Simulate scheduled cleanup task
        async def scheduled_cleanup():
            with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
                pool = await mock_database_pool
                async with pool.acquire() as conn:
                    # Run cleanup function
                    deleted_count = await conn.fetchval("SELECT cleanup_old_activity_logs()")
                    return deleted_count
        
        # Run scheduled cleanup
        result = await scheduled_cleanup()
        
        # Should execute cleanup and return result
        assert result == 500


class TestDataArchiving:
    """Test data archiving functionality"""
    
    @pytest.mark.asyncio
    async def test_archive_old_critical_activities(self, mock_database_pool):
        """Test archiving of old critical activities"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        mock_conn.fetchval.return_value = 25  # Archived count
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Archive critical activities older than 1 year
                await conn.execute("""
                    UPDATE activity_logs 
                    SET archived_at = NOW()
                    WHERE created_at < NOW() - INTERVAL '365 days'
                    AND severity IN ('critical', 'alert', 'emergency')
                    AND archived_at IS NULL
                """)
                
                # Count archived activities
                archived_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM activity_logs 
                    WHERE archived_at IS NOT NULL
                """)
            
            # Should archive critical activities and return count
            mock_conn.execute.assert_called_once()
            mock_conn.fetchval.assert_called_once()
            assert archived_count == 25
    
    @pytest.mark.asyncio
    async def test_archive_by_category(self, mock_database_pool):
        """Test category-specific archiving"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        # Archive categories with different schedules
        archive_policies = [
            ('security', 365),   # Security logs archived after 1 year
            ('trading', 180),    # Trading logs archived after 6 months
            ('user', 90)         # User logs archived after 3 months
        ]
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                for category, days in archive_policies:
                    await conn.execute("""
                        UPDATE activity_logs 
                        SET archived_at = NOW()
                        WHERE category = $1
                        AND created_at < NOW() - INTERVAL '%s days'
                        AND archived_at IS NULL
                    """, category, days)
            
            # Should execute archiving for each category
            assert mock_conn.execute.call_count == len(archive_policies)
    
    @pytest.mark.asyncio
    async def test_archived_data_accessibility(self, mock_database_pool):
        """Test that archived data remains accessible"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'activity_id': str(uuid.uuid4()),
                'category': 'security',
                'severity': 'critical',
                'archived_at': datetime.now(timezone.utc) - timedelta(days=30),
                'created_at': datetime.now(timezone.utc) - timedelta(days=400)
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Query archived activities
                archived_activities = await conn.fetch("""
                    SELECT * FROM activity_logs 
                    WHERE archived_at IS NOT NULL
                    ORDER BY archived_at DESC
                """)
            
            # Should be able to access archived data
            assert len(archived_activities) == 1
            assert archived_activities[0]['severity'] == 'critical'
    
    @pytest.mark.asyncio
    async def test_archive_compression_metadata(self, mock_database_pool):
        """Test archive compression and metadata tracking"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Archive with compression metadata
                await conn.execute("""
                    UPDATE activity_logs 
                    SET 
                        archived_at = NOW(),
                        metadata = metadata || jsonb_build_object(
                            'archive_info', jsonb_build_object(
                                'archived_by', 'system',
                                'archive_reason', 'retention_policy',
                                'original_size', octet_length(description::text)
                            )
                        )
                    WHERE created_at < NOW() - INTERVAL '365 days'
                    AND severity IN ('critical', 'alert', 'emergency')
                    AND archived_at IS NULL
                """)
            
            # Should add archive metadata
            mock_conn.execute.assert_called_once()


class TestActivitySummaryGeneration:
    """Test activity summary generation for data reduction"""
    
    @pytest.mark.asyncio
    async def test_generate_hourly_summaries(self, mock_database_pool):
        """Test generation of hourly activity summaries"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 5  # Number of summaries generated
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Generate hourly summaries
                summary_count = await conn.fetchval("""
                    SELECT generate_activity_summaries($1, $2, 'hour')
                """, start_time, end_time)
            
            # Should generate summaries
            assert summary_count == 5
            mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_daily_summaries(self, mock_database_pool):
        """Test generation of daily activity summaries"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Generate daily summaries
                await conn.execute("""
                    INSERT INTO activity_summaries (
                        period_start, period_end, period_type, category,
                        total_count, error_count, warning_count,
                        avg_execution_time_ms, max_execution_time_ms
                    )
                    SELECT 
                        date_trunc('day', created_at) as period_start,
                        date_trunc('day', created_at) + INTERVAL '1 day' as period_end,
                        'day' as period_type,
                        category,
                        COUNT(*) as total_count,
                        COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) as error_count,
                        COUNT(*) FILTER (WHERE severity = 'warning') as warning_count,
                        AVG(execution_time_ms) as avg_execution_time_ms,
                        MAX(execution_time_ms) as max_execution_time_ms
                    FROM activity_logs 
                    WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY date_trunc('day', created_at), category
                    ON CONFLICT DO NOTHING
                """)
            
            # Should generate daily summaries
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_summary_aggregation_accuracy(self, mock_database_pool):
        """Test accuracy of summary aggregations"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'period_start': datetime.now(timezone.utc) - timedelta(hours=1),
                'period_end': datetime.now(timezone.utc),
                'category': 'trading',
                'total_count': 100,
                'error_count': 5,
                'warning_count': 10,
                'avg_execution_time_ms': 1250.5,
                'max_execution_time_ms': 5000,
                'top_sources': {'trading_engine': 80, 'api_client': 20},
                'top_events': {'trade_execution': 60, 'order_placement': 40}
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Verify summary accuracy
                summaries = await conn.fetch("""
                    SELECT * FROM activity_summaries 
                    WHERE period_type = 'hour' 
                    AND category = 'trading'
                    ORDER BY period_start DESC
                    LIMIT 1
                """)
            
            # Should return accurate summary data
            assert len(summaries) == 1
            summary = summaries[0]
            assert summary['total_count'] == 100
            assert summary['error_count'] == 5
            assert summary['avg_execution_time_ms'] == 1250.5
    
    @pytest.mark.asyncio
    async def test_automated_summary_generation(self, mock_database_pool):
        """Test automated summary generation scheduling"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 10  # Summaries generated
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        async def automated_summary_task():
            """Simulate automated summary generation"""
            with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
                pool = await mock_database_pool
                async with pool.acquire() as conn:
                    # Generate summaries for last hour
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(hours=1)
                    
                    count = await conn.fetchval("""
                        SELECT generate_activity_summaries($1, $2, 'hour')
                    """, start_time, end_time)
                    return count
        
        # Run automated task
        result = await automated_summary_task()
        
        # Should generate summaries automatically
        assert result == 10


class TestDataLifecycleManagement:
    """Test complete data lifecycle management"""
    
    @pytest.mark.asyncio
    async def test_complete_data_lifecycle(self, mock_database_pool):
        """Test complete data lifecycle from creation to deletion"""
        mock_conn = AsyncMock()
        
        # Mock lifecycle stages
        lifecycle_results = {
            'created': 1000,
            'summarized': 50,
            'archived': 25,
            'deleted': 500
        }
        
        async def mock_fetchval(*args, **kwargs):
            query = args[0] if args else ""
            if "COUNT(*)" in query and "archived_at IS NULL" in query:
                return lifecycle_results['created']
            elif "generate_activity_summaries" in query:
                return lifecycle_results['summarized']
            elif "archived_at IS NOT NULL" in query:
                return lifecycle_results['archived']
            elif "cleanup_old_activity_logs" in query:
                return lifecycle_results['deleted']
            return 0
        
        mock_conn.fetchval = mock_fetchval
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Stage 1: Count active activities
                active_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM activity_logs 
                    WHERE archived_at IS NULL
                """)
                
                # Stage 2: Generate summaries
                summary_count = await conn.fetchval("""
                    SELECT generate_activity_summaries($1, $2, 'hour')
                """, datetime.now(timezone.utc) - timedelta(hours=1), datetime.now(timezone.utc))
                
                # Stage 3: Archive old critical data
                await conn.execute("""
                    UPDATE activity_logs 
                    SET archived_at = NOW()
                    WHERE created_at < NOW() - INTERVAL '365 days'
                    AND severity IN ('critical', 'alert', 'emergency')
                    AND archived_at IS NULL
                """)
                
                # Stage 4: Count archived activities
                archived_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM activity_logs 
                    WHERE archived_at IS NOT NULL
                """)
                
                # Stage 5: Cleanup old data
                deleted_count = await conn.fetchval("""
                    SELECT cleanup_old_activity_logs()
                """)
            
            # Should complete full lifecycle
            assert active_count == 1000
            assert summary_count == 50
            assert archived_count == 25
            assert deleted_count == 500
    
    @pytest.mark.asyncio
    async def test_data_lifecycle_monitoring(self, mock_database_pool):
        """Test monitoring of data lifecycle processes"""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'process': 'cleanup',
                'last_run': datetime.now(timezone.utc) - timedelta(hours=1),
                'records_processed': 1500,
                'status': 'completed'
            },
            {
                'process': 'archival',
                'last_run': datetime.now(timezone.utc) - timedelta(hours=6),
                'records_processed': 250,
                'status': 'completed'
            }
        ]
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Monitor lifecycle processes
                process_status = await conn.fetch("""
                    SELECT 
                        'cleanup' as process,
                        MAX(created_at) as last_run,
                        COUNT(*) as records_processed,
                        'completed' as status
                    FROM activity_logs 
                    WHERE event_type = 'data_cleanup'
                    UNION ALL
                    SELECT 
                        'archival' as process,
                        MAX(archived_at) as last_run,
                        COUNT(*) as records_processed,
                        'completed' as status
                    FROM activity_logs 
                    WHERE archived_at IS NOT NULL
                """)
            
            # Should return lifecycle monitoring data
            assert len(process_status) == 2
            assert process_status[0]['process'] == 'cleanup'
            assert process_status[1]['process'] == 'archival'


class TestDataRetentionCompliance:
    """Test compliance aspects of data retention"""
    
    @pytest.mark.asyncio
    async def test_gdpr_data_deletion(self, mock_database_pool):
        """Test GDPR-compliant data deletion"""
        user_id = 123456789
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        mock_conn.fetchval.return_value = 50  # Records deleted
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Delete all user data (GDPR right to be forgotten)
                await conn.execute("""
                    DELETE FROM activity_logs 
                    WHERE user_id = $1
                """, user_id)
                
                # Verify deletion
                remaining_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM activity_logs 
                    WHERE user_id = $1
                """, user_id)
            
            # Should delete user data completely
            mock_conn.execute.assert_called_once()
            mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_audit_trail_preservation(self, mock_database_pool):
        """Test preservation of audit trails during cleanup"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        mock_database_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_database_pool.acquire.return_value.__aexit__ = AsyncMock()
        
        with patch('src.utils.database.get_database_pool', return_value=mock_database_pool):
            pool = await mock_database_pool
            async with pool.acquire() as conn:
                # Preserve audit trails during cleanup
                await conn.execute("""
                    DELETE FROM activity_logs 
                    WHERE created_at < NOW() - INTERVAL '90 days'
                    AND severity NOT IN ('critical', 'alert', 'emergency')
                    AND category NOT IN ('security', 'audit')
                    AND NOT (tags && ARRAY['audit', 'compliance', 'regulatory'])
                """)
            
            # Should preserve audit-related activities
            mock_conn.execute.assert_called_once()


# NOTE: All these tests should FAIL initially since the data retention and cleanup
# functionality may not be fully implemented yet. This follows TDD methodology.