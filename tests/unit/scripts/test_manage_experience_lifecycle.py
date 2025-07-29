"""
Unit tests for experience lifecycle management script

Tests the experience lifecycle management functionality using TDD methodology.
Tests verify the actual implementation works correctly.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import json
import os

# Import from the absolute path since we're running in the test environment
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.manage_experience_lifecycle import (
    ExperienceLifecycleManager,
    LifecycleConfig,
    ExperienceLifecycleError,
    CloudStorageError,
    compress_experience_data,
    validate_cloud_storage_config,
    calculate_optimal_batch_size,
    format_storage_report,
    main
)


class TestExperienceLifecycleManager:
    """Test ExperienceLifecycleManager class with TDD approach"""
    
    @pytest.fixture
    def config(self):
        """Test configuration for lifecycle management"""
        return {
            'retention_days': 30,
            'archive_threshold_days': 7,
            'cleanup_batch_size': 1000,
            'cloud_storage_bucket': 'test-experience-archive',
            'max_storage_gb': 100,
            'min_experiences_to_keep': 1000,
            'performance_optimization_interval_hours': 24,
            'monitoring_enabled': True
        }
    
    @pytest.fixture
    def mock_db_pool(self):
        """Mock database pool for testing"""
        pool = AsyncMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        conn.fetchval = AsyncMock()
        conn.fetch = AsyncMock()
        conn.execute = AsyncMock()
        conn.transaction.return_value.__aenter__ = AsyncMock()
        conn.transaction.return_value.__aexit__ = AsyncMock()
        return pool, conn
    
    @pytest.fixture 
    def mock_gcs_client(self):
        """Mock Google Cloud Storage client"""
        client = Mock()
        bucket = Mock()
        blob = Mock()
        client.bucket.return_value = bucket
        bucket.blob.return_value = blob
        bucket.exists.return_value = True
        blob.upload_from_string = Mock()
        blob.patch = Mock()
        blob.name = "test-archive.json.gz"
        return client
    
    @pytest.fixture
    def sample_experiences(self):
        """Sample experience data for testing"""
        base_time = datetime.now(timezone.utc)
        return [
            {
                'id': i,
                'experience_id': str(uuid.uuid4()),
                'session_id': str(uuid.uuid4()),
                'state_data': {'price': 0.001, 'volume': 1000},
                'action': 1,
                'reward': 0.1,
                'created_at': base_time - timedelta(days=i),
                'metadata': {'episode': 1, 'step': i}
            }
            for i in range(100)
        ]

    def test_lifecycle_manager_initialization(self, config):
        """Test that lifecycle manager initializes correctly"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        assert manager.config == lifecycle_config
        assert manager.config.retention_days == 30
        assert manager.config.archive_threshold_days == 7

    @patch('scripts.manage_experience_lifecycle.get_database_pool')
    @pytest.mark.asyncio
    async def test_calculate_storage_usage(self, mock_get_pool, config):
        """Test storage usage calculation"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        # Mock database response
        pool, conn = Mock(), AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        conn.fetchval.return_value = 5000000  # 5MB in bytes
        mock_get_pool.return_value = pool
        
        # Test storage calculation
        storage_gb = await manager.calculate_storage_usage()
        assert abs(storage_gb - 0.005) < 0.001  # 5MB = ~0.005GB

    @patch('scripts.manage_experience_lifecycle.get_database_pool')
    @pytest.mark.asyncio
    async def test_get_experience_counts(self, mock_get_pool, config):
        """Test experience counts retrieval"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        # Mock database responses
        pool, conn = Mock(), AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        conn.fetchval.side_effect = [5000, 4000, 500, 500]  # total, active, archivable, old
        mock_get_pool.return_value = pool
        
        counts = await manager.get_experience_counts()
        assert counts['total'] == 5000
        assert counts['active'] == 4000
        assert counts['archivable'] == 500
        assert counts['old'] == 500

    @patch('scripts.manage_experience_lifecycle.get_database_pool')
    @pytest.mark.asyncio
    async def test_identify_old_experiences(self, mock_get_pool, config, sample_experiences):
        """Test identification of old experiences for cleanup"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        # Mock database to return old experiences
        pool, conn = Mock(), AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        
        old_experiences = [exp for exp in sample_experiences if 
                         (datetime.now(timezone.utc) - exp['created_at']).days > 30]
        
        # Convert to database row format
        mock_rows = []
        for exp in old_experiences:
            mock_row = {
                'id': exp['id'],
                'experience_id': uuid.UUID(exp['experience_id']),
                'session_id': uuid.UUID(exp['session_id']),
                'created_at': exp['created_at'],
                'experience_size': 1024
            }
            mock_rows.append(mock_row)
        
        conn.fetch.return_value = mock_rows
        mock_get_pool.return_value = pool
        
        identified = await manager.identify_old_experiences()
        assert len(identified) > 0
        assert all((datetime.now(timezone.utc) - 
                   datetime.fromisoformat(exp['created_at'].replace('Z', '+00:00'))).days > 30 
                  for exp in identified if 'created_at' in exp)

    @pytest.mark.asyncio
    async def test_archive_experiences_to_cloud_storage(self, config, mock_gcs_client, sample_experiences):
        """Test archiving experiences to cloud storage"""
        lifecycle_config = LifecycleConfig(**config)
        lifecycle_config.cloud_storage_enabled = True
        manager = ExperienceLifecycleManager(lifecycle_config)
        manager.gcs_client = mock_gcs_client
        
        # Test archiving
        archive_path = await manager.archive_experiences(sample_experiences[:10])
        
        # Verify GCS client was called
        mock_gcs_client.bucket.assert_called_once_with('test-experience-archive')
        assert archive_path == "test-archive.json.gz"

    @patch('scripts.manage_experience_lifecycle.get_database_pool')
    @pytest.mark.asyncio
    async def test_cleanup_old_experiences(self, mock_get_pool, config):
        """Test cleanup of old experiences from database"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        experience_ids = [1, 2, 3, 4, 5]
        
        # Mock database
        pool, conn = Mock(), AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        conn.execute.return_value = "DELETE 5"
        conn.transaction.return_value.__aenter__ = AsyncMock()
        conn.transaction.return_value.__aexit__ = AsyncMock()
        mock_get_pool.return_value = pool
        
        deleted_count = await manager.cleanup_experiences(experience_ids)
        assert deleted_count == 5
        
        # Verify database delete was called
        conn.execute.assert_called()

    @patch('scripts.manage_experience_lifecycle.get_database_pool')
    @pytest.mark.asyncio
    async def test_enforce_retention_policy(self, mock_get_pool, config):
        """Test data retention policy enforcement"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        # Mock database and methods
        pool, conn = Mock(), AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        mock_get_pool.return_value = pool
        
        with patch.object(manager, 'get_experience_counts') as mock_counts, \
             patch.object(manager, 'identify_old_experiences') as mock_identify, \
             patch.object(manager, 'get_experiences_for_archive') as mock_archive_data, \
             patch.object(manager, 'archive_experiences') as mock_archive, \
             patch.object(manager, 'cleanup_experiences') as mock_cleanup:
            
            mock_counts.return_value = {
                'total': 5000, 'active': 4000, 'archivable': 500, 'old': 500
            }
            mock_identify.return_value = [{'id': i} for i in range(10)]
            mock_archive_data.return_value = [{'experience_id': str(uuid.uuid4())}] * 10
            mock_archive.return_value = "test-archive.json.gz"
            mock_cleanup.return_value = 10
            
            result = await manager.enforce_retention_policy()
            
            assert result['total_experiences'] == 5000
            assert result['active_experiences'] == 4000
            assert result['old_experiences'] == 10
            assert result['archived_count'] == 10
            assert result['deleted_count'] == 10

    @patch('scripts.manage_experience_lifecycle.get_database_pool')
    @pytest.mark.asyncio
    async def test_optimize_database_performance(self, mock_get_pool, config):
        """Test database performance optimization"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        # Mock database
        pool, conn = Mock(), AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        conn.execute.return_value = None
        conn.fetch.return_value = [{'schemaname': 'public', 'tablename': 'rl_experiences'}]
        mock_get_pool.return_value = pool
        
        result = await manager.optimize_database_performance()
        
        # Should run VACUUM, ANALYZE, and reindex operations
        assert conn.execute.call_count >= 3
        assert result['vacuum_completed'] is True
        assert result['analyze_completed'] is True
        assert result['reindex_completed'] is True

    @patch('scripts.manage_experience_lifecycle.get_database_pool')
    @pytest.mark.asyncio
    async def test_monitor_storage_usage(self, mock_get_pool, config):
        """Test storage usage monitoring and alerting"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        with patch.object(manager, 'calculate_storage_usage') as mock_storage, \
             patch.object(manager, 'get_experience_counts') as mock_counts:
            
            mock_storage.return_value = 50.0  # 50GB
            mock_counts.return_value = {
                'total': 100000, 'active': 75000, 'archivable': 15000, 'old': 10000
            }
            
            report = await manager.monitor_storage_usage()
            
            assert report['total_storage_gb'] == 50.0
            assert report['total_experiences'] == 100000
            assert report['storage_utilization'] == 0.5  # 50/100
            assert 'recommendations' in report

    @patch('scripts.manage_experience_lifecycle.get_database_pool')
    @pytest.mark.asyncio
    async def test_generate_lifecycle_report(self, mock_get_pool, config):
        """Test comprehensive lifecycle management report generation"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        # Mock database
        pool, conn = Mock(), AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        conn.fetch.side_effect = [
            [{'total_sessions': 10, 'active_sessions': 5, 'completed_sessions': 5}],
            [{'date': datetime.now().date(), 'experiences_created': 1000}]
        ]
        mock_get_pool.return_value = pool
        
        with patch.object(manager, 'monitor_storage_usage') as mock_monitor:
            mock_monitor.return_value = {
                'total_storage_gb': 25.0,
                'total_experiences': 50000,
                'storage_status': 'normal'
            }
            
            report = await manager.generate_lifecycle_report()
            
            assert 'report_id' in report
            assert 'storage_metrics' in report
            assert 'performance_metrics' in report
            assert report['storage_metrics']['total_storage_gb'] == 25.0

    @pytest.mark.asyncio
    async def test_scheduled_maintenance(self, config):
        """Test scheduled maintenance execution"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        with patch.object(manager, 'enforce_retention_policy') as mock_retention, \
             patch.object(manager, 'optimize_database_performance') as mock_optimize, \
             patch.object(manager, 'monitor_storage_usage') as mock_monitor:
            
            mock_retention.return_value = {'archived_count': 100, 'deleted_count': 50}
            mock_optimize.return_value = {'vacuum_completed': True}
            mock_monitor.return_value = {'total_storage_gb': 1.5}
            
            result = await manager.run_scheduled_maintenance()
            
            mock_retention.assert_called_once()
            mock_optimize.assert_called_once()
            mock_monitor.assert_called_once()
            
            assert result['maintenance_completed'] is True
            assert result['retention_policy_enforced'] is True
            assert result['performance_optimized'] is True

    @pytest.mark.asyncio
    async def test_cloud_storage_error_handling(self, config, mock_gcs_client):
        """Test error handling for cloud storage operations"""
        lifecycle_config = LifecycleConfig(**config)
        lifecycle_config.cloud_storage_enabled = True
        manager = ExperienceLifecycleManager(lifecycle_config)
        manager.gcs_client = mock_gcs_client
        
        # Mock GCS error
        mock_gcs_client.bucket.side_effect = Exception("Storage service unavailable")
        
        with pytest.raises(CloudStorageError) as exc_info:
            await manager.archive_experiences([{'test': 'data'}])
        
        assert "Storage service unavailable" in str(exc_info.value)

    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test invalid configurations
        invalid_configs = [
            {'retention_days': 0},  # Invalid retention days
            {'archive_threshold_days': -1},  # Invalid threshold
            {'cleanup_batch_size': 0},  # Invalid batch size
            {'max_storage_gb': -1},  # Invalid storage limit
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                LifecycleConfig(**invalid_config)

    @pytest.mark.asyncio
    async def test_concurrent_operation_safety(self, config):
        """Test that concurrent operations are handled safely"""
        lifecycle_config = LifecycleConfig(**config)
        manager = ExperienceLifecycleManager(lifecycle_config)
        
        # Simulate lock being held
        manager._maintenance_lock._locked = True
        
        result = await manager.run_scheduled_maintenance()
        assert result['skipped_due_to_concurrent_operation'] is True


class TestLifecycleUtilities:
    """Test utility functions for lifecycle management"""
    
    def test_compress_experience_data(self):
        """Test experience data compression"""
        sample_data = [{'id': 1, 'state': [1, 2, 3], 'action': 1}] * 100
        
        compressed = compress_experience_data(sample_data)
        assert len(compressed) < len(json.dumps(sample_data))

    def test_validate_cloud_storage_config(self):
        """Test cloud storage configuration validation"""
        valid_config = {
            'bucket_name': 'valid-bucket-name',
            'project_id': 'test-project',
            'credentials_path': '/path/to/credentials.json'
        }
        
        assert validate_cloud_storage_config(valid_config) is True
        
        # Test invalid config
        invalid_config = {'bucket_name': ''}
        assert validate_cloud_storage_config(invalid_config) is False

    def test_calculate_optimal_batch_size(self):
        """Test optimal batch size calculation"""
        batch_size = calculate_optimal_batch_size(
            total_experiences=100000,
            available_memory_gb=4,
            target_processing_time_minutes=10
        )
        
        assert isinstance(batch_size, int)
        assert 100 <= batch_size <= 10000  # Reasonable range

    def test_format_storage_report(self):
        """Test storage report formatting"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_experiences': 50000,
            'storage_usage_gb': 2.5,
            'old_experiences': 10000,
            'archived_count': 5000,
            'total_storage_gb': 2.5,
            'max_storage_gb': 100.0,
            'storage_utilization': 0.025,
            'storage_status': 'normal',
            'active_experiences': 40000,
            'recommendations': ['Test recommendation']
        }
        
        formatted = format_storage_report(report_data)
        assert 'Total Experiences: 50,000' in formatted
        assert 'Storage Usage: 2.50 GB' in formatted
        assert isinstance(formatted, str)


class TestMainScriptExecution:
    """Test main script execution functionality"""
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('scripts.manage_experience_lifecycle.load_config_from_file')
    @pytest.mark.asyncio
    async def test_main_with_cleanup_command(self, mock_load_config, mock_args):
        """Test main script with cleanup command"""
        mock_args.return_value = Mock(
            command='cleanup',
            dry_run=False,
            config_file='config/config.yaml',
            verbose=False,
            output_format='text'
        )
        
        mock_config = LifecycleConfig()
        mock_load_config.return_value = mock_config
        
        with patch('scripts.manage_experience_lifecycle.ExperienceLifecycleManager') as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance
            mock_instance.enforce_retention_policy = AsyncMock(
                return_value={'deleted_count': 100}
            )
            
            result = await main()
            assert result == 0  # Success exit code

    @patch('argparse.ArgumentParser.parse_args')
    @patch('scripts.manage_experience_lifecycle.load_config_from_file')
    @pytest.mark.asyncio
    async def test_main_with_report_command(self, mock_load_config, mock_args):
        """Test main script with report command"""
        mock_args.return_value = Mock(
            command='report',
            dry_run=False,
            config_file='config/config.yaml',
            verbose=True,
            output_format='text'
        )
        
        mock_config = LifecycleConfig()
        mock_load_config.return_value = mock_config
        
        with patch('scripts.manage_experience_lifecycle.ExperienceLifecycleManager') as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance
            mock_instance.generate_lifecycle_report = AsyncMock(return_value={
                'storage_metrics': {
                    'total_experiences': 25000,
                    'total_storage_gb': 1.2,
                    'storage_status': 'normal',
                    'recommendations': []
                }
            })
            
            result = await main()
            assert result == 0  # Success exit code

    @patch('argparse.ArgumentParser.parse_args')
    @patch('scripts.manage_experience_lifecycle.load_config_from_file')
    @pytest.mark.asyncio
    async def test_main_with_dry_run(self, mock_load_config, mock_args):
        """Test main script with dry run option"""
        mock_args.return_value = Mock(
            command='cleanup',
            dry_run=True,
            config_file='config/config.yaml',
            verbose=False,
            output_format='text'
        )
        
        mock_config = LifecycleConfig()
        mock_load_config.return_value = mock_config
        
        with patch('scripts.manage_experience_lifecycle.ExperienceLifecycleManager') as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance
            mock_instance.enforce_retention_policy = AsyncMock(
                return_value={'deleted_count': 0, 'dry_run': True}
            )
            
            result = await main()
            assert result == 0  # Success exit code

    @patch('argparse.ArgumentParser.parse_args')
    @patch('scripts.manage_experience_lifecycle.load_config_from_file')
    @pytest.mark.asyncio
    async def test_error_handling_in_main(self, mock_load_config, mock_args):
        """Test error handling in main execution"""
        mock_args.return_value = Mock(
            command='cleanup',
            dry_run=False,
            config_file='config/config.yaml',
            verbose=False,
            output_format='text'
        )
        
        mock_load_config.side_effect = Exception("Configuration error")
        
        result = await main()
        assert result == 1  # Error exit code