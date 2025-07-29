"""
Tests for enhanced dashboard experience service methods (Phase 5.2)
Following TDD methodology - tests written before implementation
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from typing import Dict, Any, List

from src.dashboard.service import DashboardService
from src.dashboard.base import DashboardError


class TestEnhancedExperienceService:
    """Test suite for Enhanced Experience Service methods (Phase 5.2)"""
    
    @pytest.fixture
    def dashboard_service(self):
        """Dashboard service instance"""
        service = DashboardService()
        # Mock the config to avoid initialization issues
        service.config = MagicMock()
        return service
    
    @pytest.fixture
    def mock_db_connection(self):
        """Mock database connection"""
        mock_conn = AsyncMock()
        return mock_conn
    
    @pytest.fixture
    def sample_experience_aggregation_data(self):
        """Sample aggregated experience data"""
        return {
            "total_experiences": 1500,
            "sessions_count": 25,  
            "avg_reward_per_session": 125.75,
            "success_rate": 0.68,
            "action_distribution": {
                "0": 450,  # HOLD
                "1": 380,  # BUY  
                "2": 370,  # SELL
                "3": 300   # WAIT
            },
            "hourly_distribution": [
                {"hour": 0, "count": 45},
                {"hour": 1, "count": 52},
                {"hour": 2, "count": 38}
            ]
        }
    
    @pytest.fixture 
    def sample_visualization_data(self):
        """Sample visualization preparation data"""
        return {
            "reward_distribution": {
                "bins": [-10, -5, 0, 5, 10, 15, 20],
                "counts": [12, 45, 78, 156, 89, 34, 8]
            },
            "performance_timeline": [
                {"timestamp": "2024-01-01T00:00:00", "cumulative_reward": 0},
                {"timestamp": "2024-01-01T01:00:00", "cumulative_reward": 15.5},
                {"timestamp": "2024-01-01T02:00:00", "cumulative_reward": 28.3}
            ],
            "action_heatmap": {
                "hours": list(range(24)),
                "actions": [0, 1, 2, 3],
                "data": [[10, 15, 12, 8] for _ in range(24)]
            }
        }

    # =============================================================================
    # EXPERIENCE AGGREGATION METHODS TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_get_experience_aggregation_by_session_success(self, dashboard_service, mock_db_connection):
        """Test successful experience aggregation by session"""
        # This test will initially fail - implementing TDD
        
        # Mock database response
        mock_rows = [
            {
                'session_id': uuid4(),
                'experience_count': 50,
                'total_reward': Decimal('125.50'),
                'avg_reward': Decimal('2.51'),
                'success_rate': 0.68,
                'duration_minutes': 45,
                'session_start': datetime.utcnow(),
                'session_end': datetime.utcnow(),
                'trading_mode': 'simulation',
                'unique_tokens': 3
            }
        ]
        mock_db_connection.fetch.return_value = mock_rows
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            
            async def async_pool():
                return mock_pool
            
            mock_get_pool.return_value = async_pool()
            
            # This should fail initially as method doesn't exist
            result = await dashboard_service.get_experience_aggregation_by_session(
                hours_back=24,
                min_experiences=10
            )
            
            assert result is not None
            assert 'sessions' in result
            assert len(result['sessions']) == 1
            assert result['sessions'][0]['experience_count'] == 50
            assert result['sessions'][0]['total_reward'] == 125.50
    
    @pytest.mark.asyncio
    async def test_get_experience_aggregation_by_action_success(self, dashboard_service, mock_db_connection):
        """Test successful experience aggregation by action"""
        # This test will initially fail - implementing TDD
        
        # Mock database response
        mock_rows = [
            {'action': 0, 'count': 450, 'avg_reward': Decimal('1.25'), 'success_rate': 0.65},
            {'action': 1, 'count': 380, 'avg_reward': Decimal('2.10'), 'success_rate': 0.72}, 
            {'action': 2, 'count': 370, 'avg_reward': Decimal('1.85'), 'success_rate': 0.69},
            {'action': 3, 'count': 300, 'avg_reward': Decimal('0.95'), 'success_rate': 0.58}
        ]
        mock_db_connection.fetch.return_value = mock_rows
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            mock_get_pool.return_value = mock_pool
            
            # This should fail initially as method doesn't exist
            result = await dashboard_service.get_experience_aggregation_by_action(hours_back=24)
            
            assert result is not None
            assert 'actions' in result
            assert len(result['actions']) == 4
            assert result['actions'][0]['action'] == 0
            assert result['actions'][0]['count'] == 450
    
    @pytest.mark.asyncio
    async def test_get_experience_aggregation_by_time_success(self, dashboard_service, mock_db_connection):
        """Test successful experience aggregation by time periods"""
        # This test will initially fail - implementing TDD
        
        # Mock database response
        mock_rows = [
            {'time_bucket': datetime(2024, 1, 1, 10, 0), 'count': 45, 'avg_reward': Decimal('1.5')},
            {'time_bucket': datetime(2024, 1, 1, 11, 0), 'count': 52, 'avg_reward': Decimal('2.1')},
            {'time_bucket': datetime(2024, 1, 1, 12, 0), 'count': 38, 'avg_reward': Decimal('1.8')}
        ]
        mock_db_connection.fetch.return_value = mock_rows
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            mock_get_pool.return_value = mock_pool
            
            # This should fail initially as method doesn't exist
            result = await dashboard_service.get_experience_aggregation_by_time(
                hours_back=24,
                bucket_size='hour'
            )
            
            assert result is not None
            assert 'time_buckets' in result
            assert len(result['time_buckets']) == 3
            assert result['time_buckets'][0]['count'] == 45

    # =============================================================================
    # VISUALIZATION DATA PREPARATION TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_prepare_experience_visualization_data_success(self, dashboard_service, mock_db_connection):
        """Test successful experience visualization data preparation"""
        # This test will initially fail - implementing TDD
        
        # Mock various database responses for visualization
        mock_db_connection.fetch.side_effect = [
            # Reward distribution
            [{'reward_bin': i, 'count': 10 + i * 5} for i in range(7)],
            # Performance timeline
            [
                {'time_bucket': datetime(2024, 1, 1, i), 'cumulative_reward': i * 10.5}
                for i in range(24)
            ],
            # Action heatmap
            [
                {'hour': h, 'action': a, 'count': 5 + h + a}
                for h in range(24) for a in range(4)
            ]
        ]
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            mock_get_pool.return_value = mock_pool
            
            # This should fail initially as method doesn't exist
            result = await dashboard_service.prepare_experience_visualization_data(hours_back=24)
            
            assert result is not None
            assert 'reward_distribution' in result
            assert 'performance_timeline' in result
            assert 'action_heatmap' in result
    
    @pytest.mark.asyncio
    async def test_prepare_experience_correlation_matrix_success(self, dashboard_service, mock_db_connection):
        """Test successful experience correlation matrix preparation"""
        # This test will initially fail - implementing TDD
        
        # Mock correlation data
        mock_rows = [
            {'feature_1': 'reward', 'feature_2': 'priority', 'correlation': 0.75},
            {'feature_1': 'reward', 'feature_2': 'action', 'correlation': 0.42},
            {'feature_1': 'priority', 'feature_2': 'action', 'correlation': -0.15}
        ]
        mock_db_connection.fetch.return_value = mock_rows
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            mock_get_pool.return_value = mock_pool
            
            # This should fail initially as method doesn't exist
            result = await dashboard_service.prepare_experience_correlation_matrix(hours_back=24)
            
            assert result is not None
            assert 'correlations' in result
            assert len(result['correlations']) == 3

    # =============================================================================
    # REAL-TIME MONITORING TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_get_real_time_experience_metrics_success(self, dashboard_service, mock_db_connection):
        """Test successful real-time experience metrics retrieval"""
        # This test will initially fail - implementing TDD
        
        # Mock real-time metrics
        mock_db_connection.fetchrow.return_value = {
            'experiences_last_minute': 15,
            'experiences_last_hour': 420,
            'avg_reward_last_minute': Decimal('2.15'),
            'success_rate_last_minute': 0.73,
            'current_session_count': 3,
            'active_trading_modes': ['simulation', 'live']
        }
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            mock_get_pool.return_value = mock_pool
            
            # This should fail initially as method doesn't exist
            result = await dashboard_service.get_real_time_experience_metrics()
            
            assert result is not None
            assert result['experiences_last_minute'] == 15
            assert result['experiences_last_hour'] == 420
            assert result['avg_reward_last_minute'] == 2.15
    
    @pytest.mark.asyncio
    async def test_get_experience_stream_status_success(self, dashboard_service, mock_db_connection):
        """Test successful experience stream status monitoring"""
        # This test will initially fail - implementing TDD
        
        # Mock stream status data
        mock_rows = [
            {
                'session_id': str(uuid4()),
                'last_experience_at': datetime.utcnow() - timedelta(seconds=30),
                'experiences_per_minute': 5.2,
                'is_active': True,
                'trading_mode': 'live'
            }
        ]
        mock_db_connection.fetch.return_value = mock_rows
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            mock_get_pool.return_value = mock_pool
            
            # This should fail initially as method doesn't exist
            result = await dashboard_service.get_experience_stream_status()
            
            assert result is not None
            assert 'active_sessions' in result
            assert len(result['active_sessions']) == 1
            assert result['active_sessions'][0]['is_active'] is True

    # =============================================================================
    # CACHING AND OPTIMIZATION TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_get_cached_experience_summary_success(self, dashboard_service):
        """Test successful cached experience summary retrieval"""
        # This test will initially fail - implementing TDD
        
        with patch('src.dashboard.service.dashboard_cache') as mock_cache:
            mock_cache.get_experience_summary.return_value = {
                'total_experiences': 10000,
                'last_updated': datetime.utcnow().isoformat(),
                'cache_hit': True
            }
            
            # This should fail initially as method doesn't exist
            result = await dashboard_service.get_cached_experience_summary()
            
            assert result is not None
            assert result['cache_hit'] is True
            assert result['total_experiences'] == 10000
    
    @pytest.mark.asyncio
    async def test_invalidate_experience_cache_success(self, dashboard_service):
        """Test successful experience cache invalidation"""
        # This test will initially fail - implementing TDD
        
        with patch('src.dashboard.service.dashboard_cache') as mock_cache:
            mock_cache.invalidate_experience_data.return_value = True
            
            # This should fail initially as method doesn't exist
            result = await dashboard_service.invalidate_experience_cache()
            
            assert result is True

    # =============================================================================
    # ERROR HANDLING TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_get_experience_aggregation_database_error(self, dashboard_service, mock_db_connection):
        """Test database error handling in experience aggregation"""
        mock_db_connection.fetch.side_effect = Exception("Database connection failed")
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            mock_get_pool.return_value = mock_pool
            
            with pytest.raises(DashboardError, match="Failed to aggregate experiences"):
                await dashboard_service.get_experience_aggregation_by_session(hours_back=24)
    
    @pytest.mark.asyncio
    async def test_prepare_visualization_data_error(self, dashboard_service, mock_db_connection):
        """Test error handling in visualization data preparation"""  
        mock_db_connection.fetch.side_effect = Exception("Query timeout")
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            mock_get_pool.return_value = mock_pool
            
            with pytest.raises(DashboardError, match="Failed to prepare visualization data"):
                await dashboard_service.prepare_experience_visualization_data(hours_back=24)
    
    @pytest.mark.asyncio
    async def test_real_time_metrics_error(self, dashboard_service, mock_db_connection):
        """Test error handling in real-time metrics"""
        mock_db_connection.fetchrow.side_effect = Exception("Connection timeout")
        
        with patch('src.utils.database.get_database_pool') as mock_get_pool:
            mock_pool = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_db_connection
            mock_get_pool.return_value = mock_pool
            
            with pytest.raises(DashboardError, match="Failed to get real-time metrics"):
                await dashboard_service.get_real_time_experience_metrics()

    # =============================================================================
    # PARAMETER VALIDATION TESTS  
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_aggregation_invalid_hours_back(self, dashboard_service):
        """Test invalid hours_back parameter validation"""
        with pytest.raises(ValueError, match="hours_back must be positive"):
            await dashboard_service.get_experience_aggregation_by_session(hours_back=-1)
    
    @pytest.mark.asyncio
    async def test_aggregation_invalid_bucket_size(self, dashboard_service):
        """Test invalid bucket_size parameter validation"""
        with pytest.raises(ValueError, match="Invalid bucket_size"):
            await dashboard_service.get_experience_aggregation_by_time(
                hours_back=24,
                bucket_size='invalid'
            )
    
    @pytest.mark.asyncio
    async def test_visualization_invalid_parameters(self, dashboard_service):
        """Test invalid visualization parameters"""
        with pytest.raises(ValueError, match="hours_back must be positive"):
            await dashboard_service.prepare_experience_visualization_data(hours_back=0)