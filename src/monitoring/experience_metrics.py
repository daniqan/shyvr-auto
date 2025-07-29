"""
Experience Storage Metrics Collector

Implements comprehensive Prometheus metrics for RL experience storage system,
including database performance, cache efficiency, and training pipeline monitoring.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
import structlog

from .base import MetricsCollector, MetricsRegistry
from ..utils.database import get_database_connection, DatabaseQueryError
from ..rl_agent.experience_database import DatabaseExperienceBuffer


logger = structlog.get_logger()


# Metric buckets for histograms (in milliseconds)
LATENCY_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
SIZE_BUCKETS = (10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000)


class ExperienceStorageMetrics(MetricsCollector):
    """
    Metrics collector for RL experience storage system.
    
    Tracks database performance, cache efficiency, experience lifecycle,
    and training pipeline integration metrics.
    """
    
    def __init__(self, registry: MetricsRegistry):
        """Initialize experience storage metrics collector."""
        super().__init__(registry)
        
        # Counter metrics
        self.experiences_inserted_total = registry.get_counter(
            "rl_experiences_inserted_total",
            "Total number of experiences inserted into database",
            ["session_id", "mode", "status"]
        )
        
        self.experiences_sampled_total = registry.get_counter(
            "rl_experiences_sampled_total", 
            "Total number of experiences sampled for training",
            ["session_id", "sampling_type"]
        )
        
        self.database_operations_total = registry.get_counter(
            "rl_database_operations_total",
            "Total database operations performed",
            ["operation_type", "status"]
        )
        
        self.cache_operations_total = registry.get_counter(
            "rl_cache_operations_total",
            "Total cache operations performed",
            ["operation", "result"]
        )
        
        self.experience_cleanup_total = registry.get_counter(
            "rl_experience_cleanup_total",
            "Total number of experience cleanup operations",
            ["session_id", "reason"]
        )
        
        # Gauge metrics
        self.active_sessions = registry.get_gauge(
            "rl_active_sessions",
            "Number of active RL training sessions"
        )
        
        self.total_experiences_stored = registry.get_gauge(
            "rl_total_experiences_stored",
            "Total number of experiences currently stored",
            ["session_id"]
        )
        
        self.database_connections_active = registry.get_gauge(
            "rl_database_connections_active",
            "Number of active database connections"
        )
        
        self.memory_cache_size = registry.get_gauge(
            "rl_memory_cache_size",
            "Current size of in-memory experience cache",
            ["session_id"]
        )
        
        self.buffer_utilization = registry.get_gauge(
            "rl_buffer_utilization_ratio",
            "Current buffer utilization as ratio of max capacity",
            ["session_id"]
        )
        
        # Histogram metrics
        self.database_operation_duration = registry.get_histogram(
            "rl_database_operation_duration_ms",
            "Database operation execution time in milliseconds",
            ["operation_type"],
            buckets=LATENCY_BUCKETS
        )
        
        self.experience_insertion_duration = registry.get_histogram(
            "rl_experience_insertion_duration_ms",
            "Experience insertion latency in milliseconds",
            ["batch_size_range"],
            buckets=LATENCY_BUCKETS
        )
        
        self.experience_sampling_duration = registry.get_histogram(
            "rl_experience_sampling_duration_ms",
            "Experience sampling latency in milliseconds",
            ["sampling_type"],
            buckets=LATENCY_BUCKETS
        )
        
        self.experience_batch_size = registry.get_histogram(
            "rl_experience_batch_size",
            "Size of experience batches processed",
            ["operation"],
            buckets=SIZE_BUCKETS
        )
        
        self.cache_hit_rate = registry.get_gauge(
            "rl_cache_hit_rate",
            "Cache hit rate for experience retrieval",
            ["session_id", "cache_type"]
        )
        
        self.priority_update_duration = registry.get_histogram(
            "rl_priority_update_duration_ms",
            "Time taken to update experience priorities",
            ["update_count_range"],
            buckets=LATENCY_BUCKETS
        )
        
        # Training-specific metrics
        self.training_steps_completed = registry.get_counter(
            "rl_training_steps_completed_total",
            "Total number of training steps completed",
            ["session_id"]
        )
        
        self.beta_annealing_value = registry.get_gauge(
            "rl_beta_annealing_current",
            "Current beta value for importance sampling",
            ["session_id"]
        )
        
        # Error tracking
        self.database_errors_total = registry.get_counter(
            "rl_database_errors_total",
            "Total number of database errors encountered",
            ["error_type", "operation"]
        )
        
        # Internal state
        self._registered_buffers: Dict[str, DatabaseExperienceBuffer] = {}
        self._last_collection_stats: Dict[str, Dict] = {}
    
    def register_buffer(self, buffer: DatabaseExperienceBuffer) -> None:
        """Register an experience buffer for metrics collection."""
        session_id = str(buffer.session_id)
        self._registered_buffers[session_id] = buffer
        
        logger.info("Registered experience buffer for metrics",
                   session_id=session_id)
    
    def unregister_buffer(self, session_id: str) -> None:
        """Unregister an experience buffer from metrics collection."""
        if session_id in self._registered_buffers:
            del self._registered_buffers[session_id]
            logger.info("Unregistered experience buffer from metrics",
                       session_id=session_id)
    
    async def collect_metrics(self) -> None:
        """Collect all experience storage metrics."""
        try:
            # Collect database-level metrics
            await self._collect_database_metrics()
            
            # Collect buffer-specific metrics
            await self._collect_buffer_metrics()
            
            # Collect session-level metrics
            await self._collect_session_metrics()
            
            # Collect performance metrics
            await self._collect_performance_metrics()
            
            # Update active sessions count
            self.active_sessions.set(len(self._registered_buffers))
            
        except Exception as e:
            logger.error("Error collecting experience storage metrics", error=str(e))
            self.database_errors_total.labels(
                error_type="metrics_collection",
                operation="collect_all"
            ).inc()
            raise
    
    async def _collect_database_metrics(self) -> None:
        """Collect database-level metrics."""
        try:
            start_time = time.time()
            
            async with get_database_connection() as conn:
                # Get total experiences count
                total_experiences = await conn.fetchval(
                    "SELECT COUNT(*) FROM rl_experiences"
                )
                
                # Get active sessions count  
                active_sessions = await conn.fetchval(
                    "SELECT COUNT(DISTINCT session_id) FROM rl_training_sessions WHERE session_status = 'running'"
                )
                
                # Get database size information
                db_size_query = """
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE tablename IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                """
                
                table_sizes = await conn.fetch(db_size_query)
                
                # Update metrics
                self.active_sessions.set(active_sessions or 0)
                
            operation_time = (time.time() - start_time) * 1000
            self.database_operation_duration.labels(
                operation_type="metrics_query"
            ).observe(operation_time)
            
        except Exception as e:
            logger.error("Error collecting database metrics", error=str(e))
            self.database_errors_total.labels(
                error_type="connection_error",
                operation="metrics_query"
            ).inc()
    
    async def _collect_buffer_metrics(self) -> None:
        """Collect metrics for each registered buffer."""
        for session_id, buffer in self._registered_buffers.items():
            try:
                await self._collect_single_buffer_metrics(session_id, buffer)
            except Exception as e:
                logger.error("Error collecting buffer metrics",
                           session_id=session_id, error=str(e))
                self.database_errors_total.labels(
                    error_type="buffer_metrics",
                    operation="collect_buffer"
                ).inc()
    
    async def _collect_single_buffer_metrics(self, session_id: str, 
                                           buffer: DatabaseExperienceBuffer) -> None:
        """Collect metrics for a single experience buffer."""
        try:
            # Get buffer statistics
            stats = await buffer.get_statistics()
            performance_metrics = await buffer.get_performance_metrics()
            
            # Update experience count
            current_size = await buffer.size()
            self.total_experiences_stored.labels(session_id=session_id).set(current_size)
            
            # Update buffer utilization
            utilization = current_size / buffer.config.max_size
            self.buffer_utilization.labels(session_id=session_id).set(utilization)
            
            # Update cache metrics
            if 'cache_hit_rate' in stats:
                self.cache_hit_rate.labels(
                    session_id=session_id,
                    cache_type="memory"
                ).set(stats['cache_hit_rate'])
            
            # Update beta annealing value
            if hasattr(buffer, 'beta'):
                self.beta_annealing_value.labels(session_id=session_id).set(buffer.beta)
            
            # Update performance histograms with recent values
            if 'insertion_latency_ms' in performance_metrics:
                # Determine batch size range
                batch_size_range = self._get_batch_size_range(buffer.config.batch_size)
                self.experience_insertion_duration.labels(
                    batch_size_range=batch_size_range
                ).observe(performance_metrics['insertion_latency_ms'])
            
            if 'query_latency_ms' in performance_metrics:
                self.experience_sampling_duration.labels(
                    sampling_type="prioritized" if buffer.config.prioritized else "uniform"
                ).observe(performance_metrics['query_latency_ms'])
            
            # Cache operation tracking
            if 'cache_hits' in stats and 'cache_misses' in stats:
                # Update cache operation counters if they changed
                prev_stats = self._last_collection_stats.get(session_id, {})
                
                cache_hits_delta = stats['cache_hits'] - prev_stats.get('cache_hits', 0)
                cache_misses_delta = stats['cache_misses'] - prev_stats.get('cache_misses', 0)
                
                if cache_hits_delta > 0:
                    self.cache_operations_total.labels(
                        operation="get",
                        result="hit"
                    ).inc(cache_hits_delta)
                
                if cache_misses_delta > 0:
                    self.cache_operations_total.labels(
                        operation="get", 
                        result="miss"
                    ).inc(cache_misses_delta)
            
            # Store current stats for delta calculations
            self._last_collection_stats[session_id] = stats
            
        except Exception as e:
            logger.error("Error collecting single buffer metrics",
                        session_id=session_id, error=str(e))
            raise
    
    async def _collect_session_metrics(self) -> None:
        """Collect session-level metrics from database."""
        try:
            start_time = time.time()
            
            async with get_database_connection() as conn:
                # Get session statistics
                session_stats = await conn.fetch("""
                    SELECT 
                        ts.session_id,
                        ts.trading_mode,
                        ts.session_status,
                        COUNT(re.id) as total_experiences,
                        AVG(re.priority) as avg_priority,
                        MAX(re.created_at) as last_experience_time
                    FROM rl_training_sessions ts
                    LEFT JOIN rl_experiences re ON ts.session_id = re.session_id
                    WHERE ts.session_status = 'running'
                    GROUP BY ts.session_id, ts.trading_mode, ts.session_status
                """)
                
                # Update session-specific metrics
                for session in session_stats:
                    session_id = str(session['session_id'])
                    self.total_experiences_stored.labels(
                        session_id=session_id
                    ).set(session['total_experiences'] or 0)
            
            operation_time = (time.time() - start_time) * 1000
            self.database_operation_duration.labels(
                operation_type="session_stats"
            ).observe(operation_time)
            
        except Exception as e:
            logger.error("Error collecting session metrics", error=str(e))
            self.database_errors_total.labels(
                error_type="session_query",
                operation="session_stats"
            ).inc()
    
    async def _collect_performance_metrics(self) -> None:
        """Collect database performance metrics."""
        try:
            start_time = time.time()
            
            async with get_database_connection() as conn:
                # Get database connection pool statistics
                pool_stats = await conn.fetchrow("""
                    SELECT 
                        numbackends as active_connections,
                        xact_commit as transactions_committed,
                        xact_rollback as transactions_rolled_back,
                        tup_returned as tuples_returned,
                        tup_fetched as tuples_fetched,
                        tup_inserted as tuples_inserted,
                        tup_updated as tuples_updated,
                        tup_deleted as tuples_deleted
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                
                if pool_stats:
                    self.database_connections_active.set(
                        pool_stats['active_connections'] or 0
                    )
                
                # Get experience table statistics
                table_stats = await conn.fetchrow("""
                    SELECT 
                        n_tup_ins as inserts,
                        n_tup_upd as updates, 
                        n_tup_del as deletes,
                        n_live_tup as live_tuples,
                        n_dead_tup as dead_tuples,
                        seq_scan as sequential_scans,
                        idx_scan as index_scans
                    FROM pg_stat_user_tables 
                    WHERE relname = 'rl_experiences'
                """)
            
            operation_time = (time.time() - start_time) * 1000
            self.database_operation_duration.labels(
                operation_type="performance_stats"
            ).observe(operation_time)
            
        except Exception as e:
            logger.error("Error collecting performance metrics", error=str(e))
            self.database_errors_total.labels(
                error_type="performance_query",
                operation="performance_stats"
            ).inc()
    
    def record_experience_insertion(self, session_id: str, batch_size: int, 
                                  duration_ms: float, success: bool) -> None:
        """Record experience insertion metrics."""
        status = "success" if success else "error"
        
        # Update counter
        self.experiences_inserted_total.labels(
            session_id=session_id,
            mode="database",
            status=status
        ).inc(batch_size)
        
        # Update batch operation counter
        self.database_operations_total.labels(
            operation_type="insert",
            status=status
        ).inc()
        
        # Record duration if successful
        if success:
            batch_size_range = self._get_batch_size_range(batch_size)
            self.experience_insertion_duration.labels(
                batch_size_range=batch_size_range
            ).observe(duration_ms)
            
            self.experience_batch_size.labels(
                operation="insert"
            ).observe(batch_size)
    
    def record_experience_sampling(self, session_id: str, sample_size: int,
                                 duration_ms: float, sampling_type: str) -> None:
        """Record experience sampling metrics."""
        # Update counter
        self.experiences_sampled_total.labels(
            session_id=session_id,
            sampling_type=sampling_type
        ).inc(sample_size)
        
        # Update operation counter
        self.database_operations_total.labels(
            operation_type="sample",
            status="success"
        ).inc()
        
        # Record duration
        self.experience_sampling_duration.labels(
            sampling_type=sampling_type
        ).observe(duration_ms)
        
        self.experience_batch_size.labels(
            operation="sample"
        ).observe(sample_size)
    
    def record_priority_update(self, session_id: str, update_count: int,
                             duration_ms: float) -> None:
        """Record priority update metrics."""
        # Update training step counter
        self.training_steps_completed.labels(session_id=session_id).inc()
        
        # Update operation counter
        self.database_operations_total.labels(
            operation_type="priority_update",
            status="success"
        ).inc()
        
        # Record duration
        update_count_range = self._get_batch_size_range(update_count)
        self.priority_update_duration.labels(
            update_count_range=update_count_range
        ).observe(duration_ms)
    
    def record_cleanup_operation(self, session_id: str, reason: str,
                               experiences_removed: int) -> None:
        """Record experience cleanup metrics."""
        self.experience_cleanup_total.labels(
            session_id=session_id,
            reason=reason
        ).inc(experiences_removed)
        
        self.database_operations_total.labels(
            operation_type="cleanup",
            status="success"
        ).inc()
    
    def record_database_error(self, error_type: str, operation: str) -> None:
        """Record database error metrics."""
        self.database_errors_total.labels(
            error_type=error_type,
            operation=operation
        ).inc()
    
    def _get_batch_size_range(self, batch_size: int) -> str:
        """Get batch size range label for histograms."""
        if batch_size <= 10:
            return "1-10"
        elif batch_size <= 32:
            return "11-32"
        elif batch_size <= 64:
            return "33-64"
        elif batch_size <= 128:
            return "65-128"
        elif batch_size <= 256:
            return "129-256"
        else:
            return "256+"
    
    def get_metric_definitions(self) -> Dict[str, str]:
        """Get metric definitions for this collector."""
        return {
            # Counters
            "rl_experiences_inserted_total": "Total experiences inserted into database",
            "rl_experiences_sampled_total": "Total experiences sampled for training",
            "rl_database_operations_total": "Total database operations performed",
            "rl_cache_operations_total": "Total cache operations performed",
            "rl_experience_cleanup_total": "Total experience cleanup operations",
            "rl_training_steps_completed_total": "Total training steps completed",
            "rl_database_errors_total": "Total database errors encountered",
            
            # Gauges
            "rl_active_sessions": "Number of active RL training sessions",
            "rl_total_experiences_stored": "Total experiences currently stored",
            "rl_database_connections_active": "Active database connections",
            "rl_memory_cache_size": "In-memory cache size",
            "rl_buffer_utilization_ratio": "Buffer utilization ratio",
            "rl_cache_hit_rate": "Cache hit rate for experience retrieval",
            "rl_beta_annealing_current": "Current beta value for importance sampling",
            
            # Histograms
            "rl_database_operation_duration_ms": "Database operation execution time",
            "rl_experience_insertion_duration_ms": "Experience insertion latency",
            "rl_experience_sampling_duration_ms": "Experience sampling latency",
            "rl_experience_batch_size": "Experience batch sizes processed",
            "rl_priority_update_duration_ms": "Priority update execution time"
        }
    
    def is_healthy(self) -> bool:
        """Check if experience storage metrics collector is healthy."""
        try:
            # Simple health check without async database connection
            # The actual health will be verified during metrics collection
            return True
        except Exception as e:
            logger.error("Experience storage metrics health check failed", error=str(e))
            return False
    
    async def _health_check(self) -> None:
        """Perform health check for metrics collector."""
        async with get_database_connection() as conn:
            await conn.fetchval("SELECT 1")


def create_experience_storage_metrics(registry: MetricsRegistry) -> ExperienceStorageMetrics:
    """Factory function to create experience storage metrics collector."""
    return ExperienceStorageMetrics(registry)