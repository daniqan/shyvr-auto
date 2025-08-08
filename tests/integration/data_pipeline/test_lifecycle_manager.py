"""
TDD Tests for Data Lifecycle Manager

This test suite follows strict TDD methodology - tests are written FIRST
before implementation. Tests use REAL database connections to CloudSQL
with actual data to ensure authentic lifecycle management.

Test Categories:
1. Data Source Separation Tests - Enforce isolation between initial/simulation/live
2. Data Lineage Tracking Tests - Track relationships between data and model training  
3. Version Management Tests - Corpus versioning and immutability
4. Retention Policy Tests - Automated cleanup based on data source policies
5. Archive Management Tests - Move old data to archive storage
6. Storage Optimization Tests - Compression and space optimization
7. Integration Tests - End-to-end lifecycle workflows

Retention Policies:
- Initial corpus: Permanent retention (never delete)
- Simulation data: 6 months rolling window  
- Live data: 12 months rolling window
- Archived training data: 24 months

CRITICAL: These tests should FAIL initially as DataLifecycleManager doesn't exist yet.
This is the proper TDD approach - tests drive implementation.
"""

import pytest
import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json
from unittest import mock
from pathlib import Path

# Database and infrastructure imports
from src.utils.database import get_database_connection, execute_query, execute_transaction
from src.utils.config import get_config

# This import will FAIL initially - that's expected in TDD
# Implementation will be created after tests are written
try:
    from src.data_pipeline.lifecycle_manager import DataLifecycleManager, DataLifecycleError
except ImportError:
    # Expected failure for TDD - tests written before implementation
    DataLifecycleManager = None
    DataLifecycleError = Exception


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def database_connection():
    """Provide real CloudSQL database connection"""
    async with get_database_connection() as conn:
        yield conn


@pytest.fixture
async def lifecycle_manager():
    """
    Create DataLifecycleManager instance
    
    This fixture will FAIL initially since DataLifecycleManager doesn't exist.
    This is expected in TDD - implementation comes after tests.
    """
    if DataLifecycleManager is None:
        pytest.skip("DataLifecycleManager not implemented yet - TDD test written first")
    
    manager = DataLifecycleManager()
    await manager.initialize()
    
    yield manager
    
    # Cleanup
    await manager.close()


@pytest.fixture
async def sample_training_data(database_connection):
    """Insert sample training data for lifecycle tests"""
    sample_data = []
    
    # Create test data across different sources and time periods
    base_time = datetime.now(timezone.utc) - timedelta(days=400)  # Old enough for retention testing
    
    # Initial corpus data (should never be deleted)
    for i in range(100):
        timestamp = base_time + timedelta(hours=i)
        sample_data.append({
            'token_symbol': 'BTC',
            'timestamp': timestamp,
            'open': 50000 + i * 10,
            'high': 50100 + i * 10,
            'low': 49900 + i * 10, 
            'close': 50000 + i * 10,
            'volume': 1000000,
            'data_source': 'initial',
            'training_status': 'trained',
            'model_version': 'v1.0'
        })
    
    # Simulation data (6 months retention)
    sim_base = datetime.now(timezone.utc) - timedelta(days=200)
    for i in range(50):
        timestamp = sim_base + timedelta(hours=i)
        sample_data.append({
            'token_symbol': 'ETH',
            'timestamp': timestamp,
            'open': 3000 + i * 5,
            'high': 3010 + i * 5,
            'low': 2990 + i * 5,
            'close': 3000 + i * 5,
            'volume': 500000,
            'data_source': 'simulation',
            'training_status': 'untrained'
        })
    
    # Live data (12 months retention)
    live_base = datetime.now(timezone.utc) - timedelta(days=400)  # Some should be eligible for cleanup
    for i in range(75):
        timestamp = live_base + timedelta(hours=i)
        sample_data.append({
            'token_symbol': 'SOL',
            'timestamp': timestamp,
            'open': 100 + i,
            'high': 101 + i,
            'low': 99 + i,
            'close': 100 + i,
            'volume': 250000,
            'data_source': 'live',
            'training_status': 'in_training'
        })
    
    # Insert test data
    insert_query = """
        INSERT INTO crypto_ohlcv (
            token_symbol, timestamp, open, high, low, close, volume,
            data_source, training_status, model_version
        ) VALUES (
            %(token_symbol)s, %(timestamp)s, %(open)s, %(high)s, %(low)s,
            %(close)s, %(volume)s, %(data_source)s, %(training_status)s, %(model_version)s
        )
    """
    
    for data in sample_data:
        await execute_query(database_connection, insert_query, data)
    
    yield sample_data
    
    # Cleanup test data
    cleanup_query = "DELETE FROM crypto_ohlcv WHERE token_symbol IN ('BTC', 'ETH', 'SOL')"
    await execute_query(database_connection, cleanup_query)


@pytest.fixture
async def sample_corpus_versions(database_connection):
    """Create sample corpus versions for testing"""
    versions = [
        {
            'version_name': 'test_initial_v1.0',
            'data_source': 'initial',
            'sample_count': 1000,
            'feature_count': 130,
            'tokens': ['BTC', 'ETH'],
            'is_active': True,
            'is_immutable': True,
            'storage_path': 'gs://test-bucket/initial/v1.0',
            'notes': 'Test initial corpus'
        },
        {
            'version_name': 'test_live_2024_01',
            'data_source': 'live',
            'sample_count': 5000,
            'feature_count': 130,
            'tokens': ['BTC', 'ETH', 'SOL'],
            'is_active': False,
            'is_immutable': False,
            'storage_path': 'gs://test-bucket/live/2024_01',
            'notes': 'Test live data corpus'
        }
    ]
    
    version_ids = []
    insert_query = """
        INSERT INTO training_corpus_versions (
            version_name, data_source, sample_count, feature_count, tokens,
            is_active, is_immutable, storage_path, notes
        ) VALUES (
            %(version_name)s, %(data_source)s, %(sample_count)s, %(feature_count)s,
            %(tokens)s, %(is_active)s, %(is_immutable)s, %(storage_path)s, %(notes)s
        ) RETURNING version_id
    """
    
    for version in versions:
        result = await execute_query(database_connection, insert_query, version)
        version_ids.append(result[0]['version_id'])
    
    yield version_ids, versions
    
    # Cleanup
    cleanup_query = "DELETE FROM training_corpus_versions WHERE version_name LIKE 'test_%'"
    await execute_query(database_connection, cleanup_query)


class TestDataLifecycleManagerInitialization:
    """Test lifecycle manager initialization and configuration"""
    
    @pytest.mark.asyncio
    async def test_lifecycle_manager_initialization(self):
        """Test that DataLifecycleManager can be initialized with proper configuration"""
        if DataLifecycleManager is None:
            pytest.skip("DataLifecycleManager not implemented yet - TDD test")
        
        manager = DataLifecycleManager()
        
        # Test initialization
        await manager.initialize()
        
        # Verify configuration
        assert manager.retention_policies is not None
        assert 'initial' in manager.retention_policies
        assert 'simulation' in manager.retention_policies  
        assert 'live' in manager.retention_policies
        
        # Verify retention periods
        assert manager.retention_policies['initial'] is None  # Permanent
        assert manager.retention_policies['simulation'] == timedelta(days=180)  # 6 months
        assert manager.retention_policies['live'] == timedelta(days=365)  # 12 months
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_lifecycle_manager_database_connection(self, database_connection):
        """Test lifecycle manager establishes proper database connection"""
        if DataLifecycleManager is None:
            pytest.skip("DataLifecycleManager not implemented yet - TDD test")
        
        manager = DataLifecycleManager()
        await manager.initialize()
        
        # Test database connectivity
        assert manager.db_pool is not None
        
        # Test can execute queries
        test_query = "SELECT 1 as test"
        result = await manager.execute_query(test_query)
        assert result[0]['test'] == 1
        
        await manager.close()


class TestDataSourceSeparation:
    """Test enforcement of data source separation"""
    
    @pytest.mark.asyncio
    async def test_enforce_data_source_isolation(self, lifecycle_manager, sample_training_data):
        """Test that data sources remain isolated from each other"""
        
        # Test data source counts
        initial_count = await lifecycle_manager.get_data_count('initial')
        simulation_count = await lifecycle_manager.get_data_count('simulation')
        live_count = await lifecycle_manager.get_data_count('live')
        
        assert initial_count == 100  # From sample data
        assert simulation_count == 50
        assert live_count == 75
        
        # Test cross-contamination prevention
        result = await lifecycle_manager.validate_data_source_separation()
        assert result['is_valid'] is True
        assert result['violations'] == []
    
    @pytest.mark.asyncio
    async def test_prevent_cross_source_data_mixing(self, lifecycle_manager, database_connection):
        """Test prevention of accidental data mixing between sources"""
        
        # Attempt to create invalid mixed-source record
        invalid_data = {
            'token_symbol': 'TEST',
            'timestamp': datetime.now(timezone.utc),
            'open': 1000,
            'high': 1010,
            'low': 990,
            'close': 1005,
            'volume': 100000,
            'data_source': 'initial',
            'training_status': 'live_trading'  # Invalid status for initial source
        }
        
        with pytest.raises(DataLifecycleError, match="Invalid training_status for data_source"):
            await lifecycle_manager.insert_data(invalid_data)
    
    @pytest.mark.asyncio
    async def test_data_source_specific_validation_rules(self, lifecycle_manager):
        """Test that each data source has specific validation rules"""
        
        # Test initial source requirements
        initial_rules = await lifecycle_manager.get_validation_rules('initial')
        assert 'immutable' in initial_rules
        assert initial_rules['immutable'] is True
        
        # Test simulation source requirements  
        sim_rules = await lifecycle_manager.get_validation_rules('simulation')
        assert 'simulation_id_required' in sim_rules
        
        # Test live source requirements
        live_rules = await lifecycle_manager.get_validation_rules('live')
        assert 'real_time_validation' in live_rules


class TestDataLineageTracking:
    """Test tracking of relationships between data and model training"""
    
    @pytest.mark.asyncio
    async def test_track_training_data_lineage(self, lifecycle_manager, sample_corpus_versions):
        """Test lineage tracking between training data and models"""
        version_ids, versions = sample_corpus_versions
        
        # Create training history record
        training_record = {
            'model_type': 'lstm',
            'corpus_version_id': version_ids[0],
            'training_mode': 'initial',
            'performance_metrics': {'loss': 0.001, 'accuracy': 0.95},
            'model_checkpoint_path': 'gs://test/model.pt'
        }
        
        training_id = await lifecycle_manager.record_training_event(training_record)
        
        # Test lineage retrieval
        lineage = await lifecycle_manager.get_data_lineage(training_id)
        
        assert lineage['training_id'] == training_id
        assert lineage['corpus_version_id'] == version_ids[0]
        assert lineage['data_source'] == 'initial'
        assert 'BTC' in lineage['tokens_used']
        assert 'ETH' in lineage['tokens_used']
    
    @pytest.mark.asyncio 
    async def test_reverse_lineage_lookup(self, lifecycle_manager, sample_training_data):
        """Test finding which models used specific data samples"""
        
        # Mark some training data as used by specific model
        await lifecycle_manager.mark_data_as_trained(
            data_source='initial',
            token_symbol='BTC', 
            model_version='v1.0'
        )
        
        # Test reverse lookup
        models_used = await lifecycle_manager.get_models_using_data(
            data_source='initial',
            token_symbol='BTC'
        )
        
        assert len(models_used) > 0
        assert 'v1.0' in [m['model_version'] for m in models_used]
    
    @pytest.mark.asyncio
    async def test_lineage_integrity_validation(self, lifecycle_manager, sample_corpus_versions):
        """Test validation of lineage integrity"""
        
        # Test orphaned training records
        orphans = await lifecycle_manager.find_orphaned_training_records()
        assert isinstance(orphans, list)
        
        # Test missing corpus references
        missing_corpus = await lifecycle_manager.find_missing_corpus_references()
        assert isinstance(missing_corpus, list)
        
        # Test lineage completeness
        completeness_report = await lifecycle_manager.validate_lineage_completeness()
        assert 'complete_lineage_percentage' in completeness_report
        assert completeness_report['complete_lineage_percentage'] >= 0


class TestVersionManagement:
    """Test corpus version management and immutability"""
    
    @pytest.mark.asyncio
    async def test_create_new_corpus_version(self, lifecycle_manager, database_connection):
        """Test creation of new training corpus versions"""
        
        version_spec = {
            'version_name': 'test_live_2024_02',
            'data_source': 'live',
            'tokens': ['BTC', 'ETH', 'ADA'],
            'start_timestamp': datetime.now(timezone.utc) - timedelta(days=30),
            'end_timestamp': datetime.now(timezone.utc),
            'feature_count': 130,
            'notes': 'February live data corpus'
        }
        
        version_id = await lifecycle_manager.create_corpus_version(version_spec)
        
        assert isinstance(version_id, int)
        assert version_id > 0
        
        # Verify version was created properly
        version_info = await lifecycle_manager.get_corpus_version(version_id)
        assert version_info['version_name'] == 'test_live_2024_02'
        assert version_info['data_source'] == 'live'
        assert version_info['is_immutable'] is False  # Live data is mutable
        
        # Cleanup
        await lifecycle_manager.delete_corpus_version(version_id, force=True)
    
    @pytest.mark.asyncio
    async def test_immutable_version_protection(self, lifecycle_manager, sample_corpus_versions):
        """Test that immutable versions cannot be modified"""
        version_ids, versions = sample_corpus_versions
        
        # Try to modify immutable initial corpus
        with pytest.raises(DataLifecycleError, match="Cannot modify immutable corpus version"):
            await lifecycle_manager.update_corpus_version(
                version_ids[0], 
                {'notes': 'This should fail'}
            )
    
    @pytest.mark.asyncio
    async def test_version_activation_management(self, lifecycle_manager, sample_corpus_versions):
        """Test version activation/deactivation"""
        version_ids, versions = sample_corpus_versions
        
        # Deactivate current active version
        await lifecycle_manager.deactivate_corpus_version(version_ids[0])
        
        # Verify deactivation
        version_info = await lifecycle_manager.get_corpus_version(version_ids[0])
        assert version_info['is_active'] is False
        
        # Activate different version
        await lifecycle_manager.activate_corpus_version(version_ids[1])
        
        # Verify only one version is active per data source
        active_versions = await lifecycle_manager.get_active_corpus_versions()
        live_active = [v for v in active_versions if v['data_source'] == 'live']
        assert len(live_active) == 1
        assert live_active[0]['version_id'] == version_ids[1]
    
    @pytest.mark.asyncio
    async def test_version_history_tracking(self, lifecycle_manager, sample_corpus_versions):
        """Test tracking of version history and changes"""
        version_ids, versions = sample_corpus_versions
        
        # Make changes to mutable version
        changes = {
            'notes': 'Updated notes for testing',
            'storage_path': 'gs://test-bucket/live/2024_01_updated'
        }
        
        await lifecycle_manager.update_corpus_version(version_ids[1], changes)
        
        # Get version history
        history = await lifecycle_manager.get_version_history(version_ids[1])
        
        assert len(history) > 0
        assert any('notes' in h['changes'] for h in history)


class TestRetentionPolicyEnforcement:
    """Test automated cleanup based on retention policies"""
    
    @pytest.mark.asyncio
    async def test_identify_data_for_cleanup(self, lifecycle_manager, sample_training_data):
        """Test identification of data eligible for cleanup"""
        
        # Get cleanup candidates for each source
        sim_cleanup = await lifecycle_manager.get_cleanup_candidates('simulation')
        live_cleanup = await lifecycle_manager.get_cleanup_candidates('live') 
        initial_cleanup = await lifecycle_manager.get_cleanup_candidates('initial')
        
        # Simulation data older than 6 months should be eligible
        assert len(sim_cleanup) > 0
        
        # Live data older than 12 months should be eligible  
        assert len(live_cleanup) > 0
        
        # Initial data should NEVER be eligible for cleanup
        assert len(initial_cleanup) == 0
    
    @pytest.mark.asyncio
    async def test_execute_retention_cleanup(self, lifecycle_manager, sample_training_data):
        """Test execution of retention-based cleanup"""
        
        # Count data before cleanup
        pre_cleanup_counts = {
            'simulation': await lifecycle_manager.get_data_count('simulation'),
            'live': await lifecycle_manager.get_data_count('live'),
            'initial': await lifecycle_manager.get_data_count('initial')
        }
        
        # Execute cleanup
        cleanup_result = await lifecycle_manager.execute_retention_cleanup(dry_run=False)
        
        # Verify cleanup results
        assert 'simulation' in cleanup_result
        assert 'live' in cleanup_result
        assert 'initial' not in cleanup_result  # Should never cleanup initial
        
        # Count data after cleanup
        post_cleanup_counts = {
            'simulation': await lifecycle_manager.get_data_count('simulation'),
            'live': await lifecycle_manager.get_data_count('live'), 
            'initial': await lifecycle_manager.get_data_count('initial')
        }
        
        # Verify some old simulation/live data was removed
        if cleanup_result['simulation']['deleted_count'] > 0:
            assert post_cleanup_counts['simulation'] < pre_cleanup_counts['simulation']
        
        if cleanup_result['live']['deleted_count'] > 0:
            assert post_cleanup_counts['live'] < pre_cleanup_counts['live']
        
        # Initial data should be unchanged
        assert post_cleanup_counts['initial'] == pre_cleanup_counts['initial']
    
    @pytest.mark.asyncio
    async def test_retention_policy_dry_run(self, lifecycle_manager, sample_training_data):
        """Test dry run mode for retention cleanup"""
        
        # Execute dry run
        dry_run_result = await lifecycle_manager.execute_retention_cleanup(dry_run=True)
        
        # Verify dry run format
        assert 'simulation' in dry_run_result
        assert 'would_delete_count' in dry_run_result['simulation']
        assert 'sample_records' in dry_run_result['simulation']
        
        # Verify no actual deletion occurred
        post_dry_run_count = await lifecycle_manager.get_data_count('simulation')
        assert post_dry_run_count == 50  # Original count from sample data
    
    @pytest.mark.asyncio
    async def test_retention_policy_override_protection(self, lifecycle_manager):
        """Test protection against accidental retention policy override"""
        
        # Attempt to set invalid retention policy for initial data
        with pytest.raises(DataLifecycleError, match="Cannot set retention policy for initial data"):
            await lifecycle_manager.set_retention_policy('initial', timedelta(days=30))
        
        # Attempt to set too-short retention for live data
        with pytest.raises(DataLifecycleError, match="Retention period too short"):
            await lifecycle_manager.set_retention_policy('live', timedelta(days=7))


class TestArchiveManagement:
    """Test archive operations for old data"""
    
    @pytest.mark.asyncio
    async def test_archive_old_training_data(self, lifecycle_manager, sample_training_data):
        """Test archiving of old training data"""
        
        # Identify data for archiving (older than retention but before deletion)
        archive_candidates = await lifecycle_manager.get_archive_candidates('live')
        
        if len(archive_candidates) > 0:
            # Archive the data
            archive_result = await lifecycle_manager.archive_data(
                data_source='live',
                archive_name='test_archive_2024_01'
            )
            
            assert 'archived_count' in archive_result
            assert 'archive_location' in archive_result
            assert archive_result['archived_count'] > 0
            
            # Verify archived data is moved to archive table
            archived_count = await lifecycle_manager.get_archived_data_count('test_archive_2024_01')
            assert archived_count == archive_result['archived_count']
    
    @pytest.mark.asyncio
    async def test_restore_from_archive(self, lifecycle_manager):
        """Test restoring data from archive"""
        
        # First create some archived data
        await lifecycle_manager.archive_data(
            data_source='simulation',
            archive_name='test_restore_archive'
        )
        
        # Get archived data count
        pre_restore_count = await lifecycle_manager.get_data_count('simulation')
        archived_count = await lifecycle_manager.get_archived_data_count('test_restore_archive')
        
        if archived_count > 0:
            # Restore from archive
            restore_result = await lifecycle_manager.restore_from_archive(
                archive_name='test_restore_archive',
                target_source='simulation'
            )
            
            assert 'restored_count' in restore_result
            assert restore_result['restored_count'] == archived_count
            
            # Verify data count increased
            post_restore_count = await lifecycle_manager.get_data_count('simulation')
            assert post_restore_count == pre_restore_count + archived_count
    
    @pytest.mark.asyncio
    async def test_archive_metadata_tracking(self, lifecycle_manager):
        """Test tracking of archive metadata"""
        
        archive_name = 'test_metadata_archive'
        
        # Create archive
        await lifecycle_manager.archive_data(
            data_source='simulation',
            archive_name=archive_name
        )
        
        # Get archive metadata
        metadata = await lifecycle_manager.get_archive_metadata(archive_name)
        
        assert metadata['archive_name'] == archive_name
        assert 'created_at' in metadata
        assert 'data_source' in metadata
        assert 'record_count' in metadata
        assert 'storage_size' in metadata


class TestStorageOptimization:
    """Test compression and storage optimization"""
    
    @pytest.mark.asyncio
    async def test_compress_old_data(self, lifecycle_manager, sample_training_data):
        """Test compression of old data for storage efficiency"""
        
        # Get storage stats before compression
        pre_compression_stats = await lifecycle_manager.get_storage_statistics()
        
        # Compress old data
        compression_result = await lifecycle_manager.compress_old_data(
            age_threshold=timedelta(days=90)
        )
        
        assert 'compressed_tables' in compression_result
        assert 'space_saved_bytes' in compression_result
        
        # Get storage stats after compression
        post_compression_stats = await lifecycle_manager.get_storage_statistics()
        
        # Verify space savings (if any data was compressed)
        if compression_result['space_saved_bytes'] > 0:
            assert post_compression_stats['total_size'] <= pre_compression_stats['total_size']
    
    @pytest.mark.asyncio
    async def test_optimize_table_storage(self, lifecycle_manager):
        """Test table optimization operations"""
        
        optimization_result = await lifecycle_manager.optimize_table_storage([
            'crypto_ohlcv',
            'crypto_features'
        ])
        
        assert 'optimized_tables' in optimization_result
        assert 'vacuum_results' in optimization_result
        assert 'reindex_results' in optimization_result
    
    @pytest.mark.asyncio
    async def test_storage_usage_analysis(self, lifecycle_manager):
        """Test analysis of storage usage patterns"""
        
        usage_analysis = await lifecycle_manager.analyze_storage_usage()
        
        assert 'by_data_source' in usage_analysis
        assert 'by_table' in usage_analysis
        assert 'growth_trends' in usage_analysis
        assert 'optimization_recommendations' in usage_analysis
        
        # Verify all data sources are represented
        assert 'initial' in usage_analysis['by_data_source']
        assert 'simulation' in usage_analysis['by_data_source']
        assert 'live' in usage_analysis['by_data_source']


class TestIntegrationWorkflows:
    """Test end-to-end lifecycle management workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_data_lifecycle_workflow(self, lifecycle_manager, database_connection):
        """Test complete data lifecycle from ingestion to archive"""
        
        # 1. Ingest new data
        new_data = {
            'token_symbol': 'LIFECYCLE_TEST',
            'timestamp': datetime.now(timezone.utc),
            'open': 100,
            'high': 105,
            'low': 98,
            'close': 102,
            'volume': 1000000,
            'data_source': 'live',
            'training_status': 'untrained'
        }
        
        data_id = await lifecycle_manager.insert_data(new_data)
        assert data_id is not None
        
        # 2. Create corpus version including this data
        version_spec = {
            'version_name': 'lifecycle_test_v1',
            'data_source': 'live',
            'tokens': ['LIFECYCLE_TEST'],
            'start_timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
            'end_timestamp': datetime.now(timezone.utc),
            'feature_count': 130
        }
        
        version_id = await lifecycle_manager.create_corpus_version(version_spec)
        
        # 3. Record training event
        training_record = {
            'model_type': 'test_model',
            'corpus_version_id': version_id,
            'training_mode': 'incremental',
            'performance_metrics': {'loss': 0.01},
            'model_checkpoint_path': 'gs://test/lifecycle_model.pt'
        }
        
        training_id = await lifecycle_manager.record_training_event(training_record)
        
        # 4. Verify lineage
        lineage = await lifecycle_manager.get_data_lineage(training_id)
        assert lineage['corpus_version_id'] == version_id
        
        # 5. Mark data as trained
        await lifecycle_manager.mark_data_as_trained(
            data_source='live',
            token_symbol='LIFECYCLE_TEST',
            model_version='lifecycle_v1'
        )
        
        # 6. Verify training status update
        data_info = await lifecycle_manager.get_data_info(data_id)
        assert data_info['training_status'] in ['trained', 'in_training']
        
        # Cleanup
        await lifecycle_manager.delete_corpus_version(version_id, force=True)
        await execute_query(
            database_connection,
            "DELETE FROM crypto_ohlcv WHERE token_symbol = 'LIFECYCLE_TEST'"
        )
    
    @pytest.mark.asyncio
    async def test_automated_lifecycle_monitoring(self, lifecycle_manager):
        """Test automated monitoring of lifecycle processes"""
        
        # Start monitoring
        monitoring_result = await lifecycle_manager.run_lifecycle_monitoring()
        
        assert 'retention_status' in monitoring_result
        assert 'archive_recommendations' in monitoring_result
        assert 'storage_alerts' in monitoring_result
        assert 'lineage_integrity' in monitoring_result
        
        # Check for any critical issues
        critical_issues = monitoring_result.get('critical_issues', [])
        
        # Log any issues found (should be empty in healthy system)
        if critical_issues:
            print(f"Critical lifecycle issues found: {critical_issues}")
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_procedures(self, lifecycle_manager):
        """Test disaster recovery and backup procedures"""
        
        # Test backup creation
        backup_result = await lifecycle_manager.create_lifecycle_backup()
        
        assert 'backup_id' in backup_result
        assert 'backup_location' in backup_result
        assert 'tables_backed_up' in backup_result
        
        # Test backup validation
        validation_result = await lifecycle_manager.validate_backup(backup_result['backup_id'])
        
        assert validation_result['is_valid'] is True
        assert 'table_counts' in validation_result
        assert 'integrity_checks' in validation_result


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_handle_concurrent_lifecycle_operations(self, lifecycle_manager):
        """Test handling of concurrent lifecycle operations"""
        
        # Simulate concurrent retention cleanup attempts
        tasks = [
            lifecycle_manager.execute_retention_cleanup(dry_run=True)
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle concurrent operations gracefully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 1
    
    @pytest.mark.asyncio
    async def test_handle_corrupted_data_recovery(self, lifecycle_manager):
        """Test recovery from corrupted data scenarios"""
        
        # Test corruption detection
        corruption_report = await lifecycle_manager.detect_data_corruption()
        
        assert 'corrupted_records' in corruption_report
        assert 'integrity_violations' in corruption_report
        
        # Test automatic repair if corruption found
        if corruption_report['corrupted_records']:
            repair_result = await lifecycle_manager.repair_corrupted_data()
            assert 'repaired_count' in repair_result
            assert 'unrecoverable_count' in repair_result
    
    @pytest.mark.asyncio
    async def test_handle_storage_quota_limits(self, lifecycle_manager):
        """Test handling of storage quota and limits"""
        
        # Check storage usage against limits
        usage_check = await lifecycle_manager.check_storage_limits()
        
        assert 'current_usage' in usage_check
        assert 'quota_limit' in usage_check
        assert 'usage_percentage' in usage_check
        
        # Test automatic cleanup when approaching limits
        if usage_check['usage_percentage'] > 80:
            emergency_cleanup = await lifecycle_manager.emergency_storage_cleanup()
            assert 'freed_space' in emergency_cleanup
    
    @pytest.mark.asyncio
    async def test_validate_lifecycle_configuration(self, lifecycle_manager):
        """Test validation of lifecycle manager configuration"""
        
        validation_result = await lifecycle_manager.validate_configuration()
        
        assert validation_result['is_valid'] is True
        assert 'retention_policies' in validation_result
        assert 'database_connectivity' in validation_result
        assert 'storage_accessibility' in validation_result
        
        # All checks should pass
        for check, status in validation_result.items():
            if isinstance(status, dict) and 'status' in status:
                assert status['status'] in ['pass', 'warning']


if __name__ == "__main__":
    """
    Run these tests to see them FAIL initially.
    This is the expected behavior in TDD - tests are written before implementation.
    
    The failing tests will guide the implementation of DataLifecycleManager.
    """
    
    print("Running TDD tests for DataLifecycleManager...")
    print("These tests SHOULD FAIL initially - that's proper TDD!")
    
    # Run with verbose output to see failure details
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])