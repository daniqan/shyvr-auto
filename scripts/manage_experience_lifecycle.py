#!/usr/bin/env python3
"""
Experience Lifecycle Management Script

Comprehensive lifecycle management for RL training experiences including:
- Automated experience archival to cloud storage
- Old experience cleanup procedures  
- Data retention policy enforcement
- Performance optimization maintenance
- Storage usage monitoring and reporting

This script implements Phase 6.1 of the RL experience storage system.
"""

import asyncio
import argparse
import gzip
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import yaml

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from google.cloud import storage
    CLOUD_STORAGE_AVAILABLE = True
except ImportError:
    CLOUD_STORAGE_AVAILABLE = False
    storage = None

from src.utils.database import get_database_pool, DatabaseError, DatabaseConnectionError
from src.utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass 
class LifecycleConfig:
    """Configuration for experience lifecycle management"""
    
    # Retention policy settings
    retention_days: int = 30
    archive_threshold_days: int = 7
    min_experiences_to_keep: int = 1000
    
    # Performance settings
    cleanup_batch_size: int = 1000
    max_concurrent_operations: int = 5
    operation_timeout_seconds: int = 300
    
    # Cloud storage settings
    cloud_storage_enabled: bool = True
    cloud_storage_bucket: str = ""
    cloud_storage_project_id: str = ""
    archive_compression_enabled: bool = True
    
    # Storage monitoring
    max_storage_gb: float = 100.0
    storage_warning_threshold: float = 0.8  # 80%
    storage_critical_threshold: float = 0.95  # 95%
    
    # Performance optimization
    performance_optimization_enabled: bool = True
    performance_optimization_interval_hours: int = 24
    vacuum_threshold_percent: float = 20.0
    
    # Monitoring and reporting
    monitoring_enabled: bool = True
    report_format: str = "text"  # text, json, yaml
    alert_on_high_usage: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Convert string values to appropriate types if needed
        if isinstance(self.retention_days, str):
            self.retention_days = int(self.retention_days)
        if isinstance(self.archive_threshold_days, str):
            self.archive_threshold_days = int(self.archive_threshold_days)
        if isinstance(self.cleanup_batch_size, str):
            self.cleanup_batch_size = int(self.cleanup_batch_size)
        if isinstance(self.min_experiences_to_keep, str):
            self.min_experiences_to_keep = int(self.min_experiences_to_keep)
        if isinstance(self.max_storage_gb, str):
            self.max_storage_gb = float(self.max_storage_gb)
        if isinstance(self.storage_warning_threshold, str):
            self.storage_warning_threshold = float(self.storage_warning_threshold)
        if isinstance(self.storage_critical_threshold, str):
            self.storage_critical_threshold = float(self.storage_critical_threshold)
        if isinstance(self.vacuum_threshold_percent, str):
            self.vacuum_threshold_percent = float(self.vacuum_threshold_percent)
            
        # Validate ranges
        if self.retention_days <= 0:
            raise ValueError("retention_days must be positive")
        if self.archive_threshold_days < 0:
            raise ValueError("archive_threshold_days must be non-negative")
        if self.archive_threshold_days >= self.retention_days:
            raise ValueError("archive_threshold_days must be less than retention_days")
        if self.cleanup_batch_size <= 0:
            raise ValueError("cleanup_batch_size must be positive")
        if self.min_experiences_to_keep < 0:
            raise ValueError("min_experiences_to_keep must be non-negative")
        if self.max_storage_gb <= 0:
            raise ValueError("max_storage_gb must be positive")


class ExperienceLifecycleError(Exception):
    """Base exception for lifecycle management errors"""
    pass


class CloudStorageError(ExperienceLifecycleError):
    """Exception for cloud storage operations"""
    pass


class ExperienceLifecycleManager:
    """Main class for managing RL experience lifecycle"""
    
    def __init__(self, config: LifecycleConfig):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self._maintenance_lock = asyncio.Lock()
        self._gcs_client = None
        
        # Initialize cloud storage if enabled
        if config.cloud_storage_enabled and CLOUD_STORAGE_AVAILABLE:
            self._init_cloud_storage()
        elif config.cloud_storage_enabled and not CLOUD_STORAGE_AVAILABLE:
            self.logger.warning("Cloud storage enabled but google-cloud-storage not available")
    
    def _init_cloud_storage(self):
        """Initialize Google Cloud Storage client"""
        try:
            if self.config.cloud_storage_project_id:
                self._gcs_client = storage.Client(project=self.config.cloud_storage_project_id)
            else:
                self._gcs_client = storage.Client()
            
            # Verify bucket access
            if self.config.cloud_storage_bucket:
                bucket = self._gcs_client.bucket(self.config.cloud_storage_bucket)
                bucket.exists()  # Test access
                
            self.logger.info("Cloud storage initialized", 
                           bucket=self.config.cloud_storage_bucket)
        except Exception as e:
            self.logger.error(f"Failed to initialize cloud storage: {e}")
            raise CloudStorageError(f"Cloud storage initialization failed: {e}")
    
    @property
    def gcs_client(self):
        """Get Google Cloud Storage client"""
        return self._gcs_client
    
    @gcs_client.setter
    def gcs_client(self, client):
        """Set Google Cloud Storage client (for testing)"""
        self._gcs_client = client
    
    async def calculate_storage_usage(self) -> float:
        """Calculate current storage usage in GB"""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                # Calculate total size of RL experience tables
                query = """
                SELECT 
                    COALESCE(
                        pg_total_relation_size('rl_experiences') +
                        pg_total_relation_size('rl_training_sessions') +
                        pg_total_relation_size('rl_performance_metrics'),
                        0
                    ) as total_size_bytes
                """
                total_bytes = await conn.fetchval(query)
                return (total_bytes or 0) / (1024 ** 3)  # Convert to GB
                
        except Exception as e:
            self.logger.error(f"Failed to calculate storage usage: {e}")
            raise ExperienceLifecycleError(f"Storage calculation failed: {e}")
    
    async def get_experience_counts(self) -> Dict[str, int]:
        """Get experience counts by age"""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                # Get total experiences
                total_query = "SELECT COUNT(*) FROM rl_experiences"
                total_count = await conn.fetchval(total_query)
                
                # Get experiences within retention period
                retention_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.retention_days)
                active_query = """
                SELECT COUNT(*) FROM rl_experiences 
                WHERE created_at > $1
                """
                active_count = await conn.fetchval(active_query, retention_cutoff)
                
                # Get experiences older than archive threshold
                archive_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.archive_threshold_days)
                archivable_query = """
                SELECT COUNT(*) FROM rl_experiences 
                WHERE created_at <= $1 AND created_at > $2
                """
                archivable_count = await conn.fetchval(archivable_query, archive_cutoff, retention_cutoff)
                
                # Get experiences older than retention period
                old_query = """
                SELECT COUNT(*) FROM rl_experiences 
                WHERE created_at <= $1
                """
                old_count = await conn.fetchval(old_query, retention_cutoff)
                
                return {
                    'total': total_count or 0,
                    'active': active_count or 0,
                    'archivable': archivable_count or 0,
                    'old': old_count or 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get experience counts: {e}")
            raise ExperienceLifecycleError(f"Experience counting failed: {e}")
    
    async def identify_old_experiences(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Identify experiences older than retention period"""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                retention_cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.retention_days)
                
                query = """
                SELECT id, experience_id, session_id, created_at, 
                       pg_column_size(state_data) + pg_column_size(next_state_data) + 
                       pg_column_size(metadata) as experience_size
                FROM rl_experiences 
                WHERE created_at <= $1
                ORDER BY created_at ASC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                rows = await conn.fetch(query, retention_cutoff)
                
                experiences = []
                for row in rows:
                    experiences.append({
                        'id': row['id'],
                        'experience_id': str(row['experience_id']),
                        'session_id': str(row['session_id']),
                        'created_at': row['created_at'],
                        'size_bytes': row['experience_size'] or 0
                    })
                
                return experiences
                
        except Exception as e:
            self.logger.error(f"Failed to identify old experiences: {e}")
            raise ExperienceLifecycleError(f"Experience identification failed: {e}")
    
    async def get_experiences_for_archive(self, experience_ids: List[int]) -> List[Dict[str, Any]]:
        """Get full experience data for archival"""
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                query = """
                SELECT experience_id, session_id, user_id, state_data, action, reward,
                       next_state_data, done, priority, trading_mode, token_address,
                       chain, market_conditions, performance_metrics, error_data,
                       metadata, created_at, updated_at
                FROM rl_experiences 
                WHERE id = ANY($1)
                ORDER BY created_at
                """
                
                rows = await conn.fetch(query, experience_ids)
                
                experiences = []
                for row in rows:
                    exp_data = {
                        'experience_id': str(row['experience_id']),
                        'session_id': str(row['session_id']),
                        'user_id': row['user_id'],
                        'state_data': row['state_data'],
                        'action': row['action'],
                        'reward': float(row['reward']),
                        'next_state_data': row['next_state_data'],
                        'done': row['done'],
                        'priority': float(row['priority']),
                        'trading_mode': row['trading_mode'],
                        'token_address': row['token_address'],
                        'chain': row['chain'],
                        'market_conditions': row['market_conditions'],
                        'performance_metrics': row['performance_metrics'],
                        'error_data': row['error_data'],
                        'metadata': row['metadata'],
                        'created_at': row['created_at'].isoformat(),
                        'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None
                    }
                    experiences.append(exp_data)
                
                return experiences
                
        except Exception as e:
            self.logger.error(f"Failed to get experiences for archive: {e}")
            raise ExperienceLifecycleError(f"Archive data retrieval failed: {e}")
    
    async def archive_experiences(self, experiences: List[Dict[str, Any]]) -> str:
        """Archive experiences to cloud storage"""
        if not self.config.cloud_storage_enabled or not self._gcs_client:
            raise CloudStorageError("Cloud storage not available")
        
        if not experiences:
            return ""
        
        try:
            # Create archive metadata
            archive_metadata = {
                'archive_id': str(uuid.uuid4()),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'experience_count': len(experiences),
                'total_size_bytes': sum(len(json.dumps(exp).encode()) for exp in experiences),
                'retention_cutoff_days': self.config.retention_days,
                'experiences': experiences
            }
            
            # Generate archive filename
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            archive_filename = f"experiences_archive_{timestamp}_{len(experiences)}.json"
            
            # Compress if enabled
            archive_data = json.dumps(archive_metadata, indent=2).encode()
            if self.config.archive_compression_enabled:
                archive_data = gzip.compress(archive_data)
                archive_filename += ".gz"
            
            # Upload to cloud storage
            bucket = self._gcs_client.bucket(self.config.cloud_storage_bucket)
            blob = bucket.blob(f"rl_experiences/{datetime.now().year}/{archive_filename}")
            
            blob.upload_from_string(archive_data)
            
            # Set metadata
            blob.metadata = {
                'archive_id': archive_metadata['archive_id'],
                'experience_count': str(len(experiences)),
                'created_at': archive_metadata['created_at'],
                'compressed': str(self.config.archive_compression_enabled)
            }
            blob.patch()
            
            self.logger.info("Experiences archived to cloud storage",
                           archive_path=blob.name,
                           experience_count=len(experiences),
                           compressed=self.config.archive_compression_enabled)
            
            return blob.name
            
        except Exception as e:
            self.logger.error(f"Failed to archive experiences: {e}")
            raise CloudStorageError(f"Archive operation failed: {e}")
    
    async def cleanup_experiences(self, experience_ids: List[int]) -> int:
        """Delete experiences from database"""
        if not experience_ids:
            return 0
        
        try:
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Delete in batches to avoid locks
                    total_deleted = 0
                    batch_size = self.config.cleanup_batch_size
                    
                    for i in range(0, len(experience_ids), batch_size):
                        batch_ids = experience_ids[i:i + batch_size]
                        
                        result = await conn.execute("""
                            DELETE FROM rl_experiences 
                            WHERE id = ANY($1)
                        """, batch_ids)
                        
                        # Extract number from "DELETE N" result
                        if result.startswith("DELETE "):
                            batch_deleted = int(result.split()[1])
                            total_deleted += batch_deleted
                    
                    self.logger.info("Experiences deleted from database",
                                   deleted_count=total_deleted,
                                   batches=len(range(0, len(experience_ids), batch_size)))
                    
                    return total_deleted
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup experiences: {e}")
            raise ExperienceLifecycleError(f"Cleanup operation failed: {e}")
    
    async def enforce_retention_policy(self, dry_run: bool = False) -> Dict[str, Any]:
        """Enforce data retention policy"""
        try:
            self.logger.info("Starting retention policy enforcement", 
                           retention_days=self.config.retention_days,
                           dry_run=dry_run)
            
            # Get current experience counts
            counts = await self.get_experience_counts()
            
            # Check if we have minimum experiences to keep
            if counts['active'] < self.config.min_experiences_to_keep:
                self.logger.warning("Not enough active experiences to enforce retention policy",
                                  active_count=counts['active'],
                                  minimum_required=self.config.min_experiences_to_keep)
                return {
                    'skipped': True,
                    'reason': 'insufficient_active_experiences',
                    'active_count': counts['active'],
                    'minimum_required': self.config.min_experiences_to_keep
                }
            
            # Identify old experiences for cleanup
            old_experiences = await self.identify_old_experiences()
            
            if not old_experiences:
                self.logger.info("No old experiences found for cleanup")
                return {
                    'total_experiences': counts['total'],
                    'active_experiences': counts['active'],
                    'old_experiences': 0,
                    'archived_count': 0,
                    'deleted_count': 0,
                    'cleanup_skipped': True
                }
            
            archived_count = 0
            deleted_count = 0
            
            if not dry_run:
                # Archive experiences if cloud storage is enabled
                if self.config.cloud_storage_enabled and old_experiences:
                    # Get full experience data for archival
                    experience_ids = [exp['id'] for exp in old_experiences]
                    archive_data = await self.get_experiences_for_archive(experience_ids)
                    
                    if archive_data:
                        archive_path = await self.archive_experiences(archive_data)
                        archived_count = len(archive_data)
                        self.logger.info("Experiences archived", 
                                       count=archived_count,
                                       archive_path=archive_path)
                
                # Delete old experiences from database
                experience_ids = [exp['id'] for exp in old_experiences]
                deleted_count = await self.cleanup_experiences(experience_ids)
            else:
                # Dry run - just report what would be done
                archived_count = len(old_experiences) if self.config.cloud_storage_enabled else 0
                deleted_count = len(old_experiences)
                
                self.logger.info("DRY RUN: Would archive and delete experiences",
                               would_archive=archived_count,
                               would_delete=deleted_count)
            
            result = {
                'total_experiences': counts['total'],
                'active_experiences': counts['active'], 
                'old_experiences': len(old_experiences),
                'archived_count': archived_count,
                'deleted_count': deleted_count,
                'dry_run': dry_run,
                'cleanup_completed': not dry_run
            }
            
            self.logger.info("Retention policy enforcement completed", **result)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to enforce retention policy: {e}")
            raise ExperienceLifecycleError(f"Retention policy enforcement failed: {e}")
    
    async def optimize_database_performance(self, dry_run: bool = False) -> Dict[str, Any]:
        """Optimize database performance through maintenance operations"""
        if not self.config.performance_optimization_enabled:
            return {'skipped': True, 'reason': 'performance_optimization_disabled'}
        
        try:
            self.logger.info("Starting database performance optimization", dry_run=dry_run)
            
            results = {
                'vacuum_completed': False,
                'analyze_completed': False,
                'reindex_completed': False,
                'dry_run': dry_run
            }
            
            if not dry_run:
                pool = await get_database_pool()
                async with pool.acquire() as conn:
                    # Run VACUUM on experience tables
                    try:
                        self.logger.info("Running VACUUM on rl_experiences table")
                        await conn.execute("VACUUM ANALYZE rl_experiences")
                        results['vacuum_completed'] = True
                        
                        await conn.execute("VACUUM ANALYZE rl_training_sessions") 
                        await conn.execute("VACUUM ANALYZE rl_performance_metrics")
                        
                    except Exception as e:
                        self.logger.warning("VACUUM operation failed", error=str(e))
                    
                    # Run ANALYZE for statistics update
                    try:
                        self.logger.info("Running ANALYZE on experience tables")
                        await conn.execute("ANALYZE rl_experiences")
                        await conn.execute("ANALYZE rl_training_sessions")
                        await conn.execute("ANALYZE rl_performance_metrics")
                        results['analyze_completed'] = True
                        
                    except Exception as e:
                        self.logger.warning("ANALYZE operation failed", error=str(e))
                    
                    # Reindex if needed (check for index bloat)
                    try:
                        # Check index usage and bloat
                        index_stats = await conn.fetch("""
                            SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
                            FROM pg_stat_user_indexes 
                            WHERE schemaname = 'public' 
                            AND tablename IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                        """)
                        
                        # For now, just reindex the main indexes
                        await conn.execute("REINDEX INDEX idx_rl_experiences_session_id")
                        await conn.execute("REINDEX INDEX idx_rl_experiences_priority_desc")
                        await conn.execute("REINDEX INDEX idx_rl_experiences_user_created")
                        results['reindex_completed'] = True
                        
                    except Exception as e:
                        self.logger.warning("REINDEX operation failed", error=str(e))
            else:
                # Dry run - report what would be done
                self.logger.info("DRY RUN: Would run VACUUM, ANALYZE, and REINDEX operations")
                results.update({
                    'vacuum_completed': True,
                    'analyze_completed': True, 
                    'reindex_completed': True
                })
            
            self.logger.info("Database performance optimization completed", **results)
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to optimize database performance: {e}")
            raise ExperienceLifecycleError(f"Performance optimization failed: {e}")
    
    async def monitor_storage_usage(self) -> Dict[str, Any]:
        """Monitor storage usage and generate recommendations"""
        try:
            # Get storage metrics
            storage_gb = await self.calculate_storage_usage()
            counts = await self.get_experience_counts()
            
            # Calculate utilization
            storage_utilization = storage_gb / self.config.max_storage_gb
            
            # Generate status and recommendations
            status = "normal"
            recommendations = []
            
            if storage_utilization >= self.config.storage_critical_threshold:
                status = "critical"
                recommendations.extend([
                    "Immediate cleanup required - storage usage is critical",
                    "Consider reducing retention period temporarily",
                    "Archive old experiences to cloud storage immediately"
                ])
            elif storage_utilization >= self.config.storage_warning_threshold:
                status = "warning"
                recommendations.extend([
                    "Storage usage is high - schedule cleanup soon",
                    "Review retention policy settings",
                    "Consider archiving older experiences"
                ])
            
            if counts['old'] > 0:
                recommendations.append(f"Found {counts['old']:,} experiences older than retention period")
            
            if counts['archivable'] > 1000:
                recommendations.append(f"Found {counts['archivable']:,} experiences ready for archival")
            
            # Calculate average experience size
            avg_experience_size = 0
            if counts['total'] > 0:
                avg_experience_size = (storage_gb * 1024 ** 3) / counts['total']  # bytes per experience
            
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_storage_gb': round(storage_gb, 3),
                'max_storage_gb': self.config.max_storage_gb,
                'storage_utilization': round(storage_utilization, 3),
                'storage_status': status,
                'total_experiences': counts['total'],
                'active_experiences': counts['active'],
                'archivable_experiences': counts['archivable'],
                'old_experiences': counts['old'],
                'avg_experience_size_bytes': round(avg_experience_size, 2),
                'recommendations': recommendations
            }
            
            # Log warnings for high usage
            if status == "critical":
                self.logger.critical("CRITICAL: Storage usage is at critical level",
                                   utilization=storage_utilization,
                                   storage_gb=storage_gb)
            elif status == "warning":
                self.logger.warning("WARNING: Storage usage is high",
                                  utilization=storage_utilization,
                                  storage_gb=storage_gb)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to monitor storage usage: {e}")
            raise ExperienceLifecycleError(f"Storage monitoring failed: {e}")
    
    async def generate_lifecycle_report(self) -> Dict[str, Any]:
        """Generate comprehensive lifecycle management report"""
        try:
            self.logger.info("Generating lifecycle management report")
            
            # Get storage monitoring data
            storage_report = await self.monitor_storage_usage()
            
            # Get additional performance metrics
            pool = await get_database_pool()
            async with pool.acquire() as conn:
                # Get session statistics
                session_stats = await conn.fetch("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN session_status = 'running' THEN 1 END) as active_sessions,
                        COUNT(CASE WHEN session_status = 'completed' THEN 1 END) as completed_sessions,
                        AVG(total_experiences) as avg_experiences_per_session,
                        MAX(total_experiences) as max_experiences_per_session
                    FROM rl_training_sessions
                    WHERE created_at > NOW() - INTERVAL '30 days'
                """)
                
                # Get recent activity metrics
                activity_stats = await conn.fetch("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as experiences_created,
                        AVG(reward) as avg_reward,
                        COUNT(CASE WHEN reward > 0 THEN 1 END) as positive_rewards
                    FROM rl_experiences
                    WHERE created_at > NOW() - INTERVAL '7 days'
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                """)
            
            # Compile comprehensive report
            report = {
                'report_id': str(uuid.uuid4()),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'config': {
                    'retention_days': self.config.retention_days,
                    'archive_threshold_days': self.config.archive_threshold_days,
                    'max_storage_gb': self.config.max_storage_gb,
                    'cloud_storage_enabled': self.config.cloud_storage_enabled
                },
                'storage_metrics': storage_report,
                'performance_metrics': {
                    'session_stats': dict(session_stats[0]) if session_stats else {},
                    'recent_activity': [dict(row) for row in activity_stats]
                },
                'system_health': {
                    'lifecycle_manager_version': '1.0.0',
                    'database_connection_healthy': True,
                    'cloud_storage_available': self.config.cloud_storage_enabled and self._gcs_client is not None
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate lifecycle report: {e}")
            raise ExperienceLifecycleError(f"Report generation failed: {e}")
    
    async def run_scheduled_maintenance(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run complete scheduled maintenance cycle"""
        # Check if another maintenance operation is running
        if self._maintenance_lock.locked():
            self.logger.warning("Maintenance already running, skipping")
            return {'skipped_due_to_concurrent_operation': True}
        
        async with self._maintenance_lock:
            try:
                self.logger.info("Starting scheduled maintenance cycle", dry_run=dry_run)
                start_time = time.time()
                
                results = {
                    'maintenance_started_at': datetime.now(timezone.utc).isoformat(),
                    'dry_run': dry_run,
                    'maintenance_completed': False,
                    'retention_policy_enforced': False,
                    'performance_optimized': False,
                    'storage_monitored': False,
                    'errors': []
                }
                
                # Enforce retention policy
                try:
                    retention_result = await self.enforce_retention_policy(dry_run=dry_run)
                    results['retention_policy_result'] = retention_result
                    results['retention_policy_enforced'] = True
                except Exception as e:
                    error_msg = f"Retention policy enforcement failed: {e}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                
                # Optimize database performance
                try:
                    perf_result = await self.optimize_database_performance(dry_run=dry_run)
                    results['performance_optimization_result'] = perf_result
                    results['performance_optimized'] = True
                except Exception as e:
                    error_msg = f"Performance optimization failed: {e}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                
                # Monitor storage usage
                try:
                    storage_result = await self.monitor_storage_usage()
                    results['storage_monitoring_result'] = storage_result
                    results['storage_monitored'] = True
                except Exception as e:
                    error_msg = f"Storage monitoring failed: {e}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
                
                # Calculate maintenance duration
                duration_seconds = time.time() - start_time
                results['maintenance_duration_seconds'] = round(duration_seconds, 2)
                results['maintenance_completed'] = len(results['errors']) == 0
                results['maintenance_completed_at'] = datetime.now(timezone.utc).isoformat()
                
                self.logger.info("Scheduled maintenance cycle completed",
                               duration_seconds=duration_seconds,
                               errors=len(results['errors']),
                               dry_run=dry_run)
                
                return results
                
            except Exception as e:
                self.logger.error(f"Failed to run scheduled maintenance: {e}")
                raise ExperienceLifecycleError(f"Scheduled maintenance failed: {e}")


# Utility functions

def compress_experience_data(data: List[Dict[str, Any]]) -> bytes:
    """Compress experience data using gzip"""
    json_data = json.dumps(data).encode()
    return gzip.compress(json_data)


def validate_cloud_storage_config(config: Dict[str, Any]) -> bool:
    """Validate cloud storage configuration"""
    required_fields = ['bucket_name']
    return all(config.get(field) for field in required_fields)


def calculate_optimal_batch_size(total_experiences: int, 
                                available_memory_gb: float,
                                target_processing_time_minutes: int) -> int:
    """Calculate optimal batch size for processing"""
    # Simple heuristic based on memory and time constraints
    memory_based_batch = int(available_memory_gb * 1000)  # 1000 experiences per GB
    time_based_batch = min(10000, total_experiences // target_processing_time_minutes)
    
    return max(100, min(memory_based_batch, time_based_batch))


def format_storage_report(report_data: Dict[str, Any]) -> str:
    """Format storage report as human-readable text"""
    lines = [
        "=== Experience Storage Report ===",
        f"Generated: {report_data.get('timestamp', 'Unknown')}",
        "",
        f"Total Experiences: {report_data.get('total_experiences', 0):,}",
        f"Active Experiences: {report_data.get('active_experiences', 0):,}",
        f"Old Experiences: {report_data.get('old_experiences', 0):,}",
        "",
        f"Storage Usage: {report_data.get('total_storage_gb', 0):.2f} GB",
        f"Storage Limit: {report_data.get('max_storage_gb', 0):.2f} GB",
        f"Utilization: {report_data.get('storage_utilization', 0):.1%}",
        f"Status: {report_data.get('storage_status', 'unknown').upper()}",
        ""
    ]
    
    recommendations = report_data.get('recommendations', [])
    if recommendations:
        lines.append("Recommendations:")
        for rec in recommendations:
            lines.append(f"- {rec}")
    
    return "\n".join(lines)


def load_config_from_file(config_path: str) -> LifecycleConfig:
    """Load lifecycle configuration from YAML file"""
    try:
        # Use the application's config loading to resolve environment variables
        app_config = get_config()
        
        # Extract RL experience storage lifecycle config
        if hasattr(app_config, 'rl') and hasattr(app_config.rl, 'experience_storage'):
            experience_config = app_config.rl.experience_storage
            lifecycle_config = getattr(experience_config, 'lifecycle', {})
            
            # Convert to dict if it's a config object
            if hasattr(lifecycle_config, '__dict__'):
                lifecycle_dict = lifecycle_config.__dict__
            else:
                lifecycle_dict = lifecycle_config if isinstance(lifecycle_config, dict) else {}
        else:
            # Fallback to direct YAML loading with defaults
            lifecycle_dict = {}
        
        # Map config to LifecycleConfig with safe defaults
        return LifecycleConfig(
            retention_days=lifecycle_dict.get('max_age_days', 30),
            archive_threshold_days=lifecycle_dict.get('archive_threshold_days', 7),
            cleanup_batch_size=lifecycle_dict.get('cleanup_batch_size', 1000),
            min_experiences_to_keep=lifecycle_dict.get('min_experiences_to_keep', 1000),
            cloud_storage_enabled=lifecycle_dict.get('archive_old_experiences', False),
            cloud_storage_bucket=lifecycle_dict.get('cloud_storage_bucket', ''),
            max_storage_gb=lifecycle_dict.get('max_storage_gb', 100.0),
            performance_optimization_enabled=lifecycle_dict.get('performance_optimization_enabled', True)
        )
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        # Return default config on error
        logger.warning("Using default lifecycle configuration")
        return LifecycleConfig()


async def main() -> int:
    """Main entry point for lifecycle management script"""
    parser = argparse.ArgumentParser(
        description="RL Experience Lifecycle Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run cleanup with default config
  python manage_experience_lifecycle.py cleanup
  
  # Generate report
  python manage_experience_lifecycle.py report
  
  # Run full maintenance cycle
  python manage_experience_lifecycle.py maintenance
  
  # Dry run to see what would be done
  python manage_experience_lifecycle.py cleanup --dry-run
  
  # Use custom config file
  python manage_experience_lifecycle.py cleanup --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        'command',
        choices=['cleanup', 'archive', 'optimize', 'monitor', 'report', 'maintenance'],
        help='Operation to perform'
    )
    
    parser.add_argument(
        '--config-file',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['text', 'json', 'yaml'],
        default='text',
        help='Output format for reports (default: text)'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = load_config_from_file(args.config_file)
        config.report_format = args.output_format
        
        # Create lifecycle manager
        manager = ExperienceLifecycleManager(config)
        
        # Execute requested command
        if args.command == 'cleanup':
            result = await manager.enforce_retention_policy(dry_run=args.dry_run)
            print(f"Cleanup completed: {result['deleted_count']} experiences deleted")
            
        elif args.command == 'archive':
            # Archive old experiences without deleting
            old_experiences = await manager.identify_old_experiences()
            if old_experiences and not args.dry_run:
                experience_ids = [exp['id'] for exp in old_experiences]
                archive_data = await manager.get_experiences_for_archive(experience_ids)
                archive_path = await manager.archive_experiences(archive_data)
                print(f"Archived {len(archive_data)} experiences to {archive_path}")
            else:
                print(f"Would archive {len(old_experiences)} experiences" if args.dry_run 
                      else "No experiences to archive")
                
        elif args.command == 'optimize':
            result = await manager.optimize_database_performance(dry_run=args.dry_run)
            print(f"Database optimization: VACUUM={result['vacuum_completed']}, "
                  f"ANALYZE={result['analyze_completed']}, REINDEX={result['reindex_completed']}")
            
        elif args.command == 'monitor':
            result = await manager.monitor_storage_usage()
            if args.output_format == 'json':
                print(json.dumps(result, indent=2))
            elif args.output_format == 'yaml':
                print(yaml.dump(result, default_flow_style=False))
            else:
                print(format_storage_report(result))
                
        elif args.command == 'report':
            result = await manager.generate_lifecycle_report()
            if args.output_format == 'json':
                print(json.dumps(result, indent=2))
            elif args.output_format == 'yaml':
                print(yaml.dump(result, default_flow_style=False))
            else:
                # Text format - show key metrics
                storage = result['storage_metrics']
                print(format_storage_report(storage))
                
        elif args.command == 'maintenance':
            result = await manager.run_scheduled_maintenance(dry_run=args.dry_run)
            print(f"Maintenance completed: {result['maintenance_completed']}")
            if result.get('errors'):
                print(f"Errors encountered: {len(result['errors'])}")
                for error in result['errors']:
                    print(f"  - {error}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)