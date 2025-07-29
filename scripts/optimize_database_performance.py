#!/usr/bin/env python3
"""
Database Performance Optimization for Experience Storage

Analyzes database query performance, optimizes indexes, and tunes
database parameters for optimal RL experience storage performance.
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import get_database_connection
from src.utils.config import get_config

logger = structlog.get_logger()


class DatabasePerformanceOptimizer:
    """
    Database performance optimizer for RL experience storage.
    
    Analyzes query performance, identifies bottlenecks, and applies
    optimizations to improve database performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the database performance optimizer."""
        self.config = config
        self.optimization_results: Dict[str, Any] = {}
        self.applied_optimizations: List[Dict[str, Any]] = []
        
    async def analyze_query_performance(self) -> Dict[str, Any]:
        """Analyze database query performance and identify bottlenecks."""
        try:
            logger.info("Starting database query performance analysis")
            
            async with get_database_connection() as conn:
                # Get slow query statistics
                slow_queries = await self._analyze_slow_queries(conn)
                
                # Analyze index usage and effectiveness
                index_analysis = await self._analyze_index_usage(conn)
                
                # Check table statistics and bloat
                table_stats = await self._analyze_table_statistics(conn)
                
                # Analyze query patterns and frequency
                query_patterns = await self._analyze_query_patterns(conn)
                
                # Check database configuration
                db_config = await self._analyze_database_configuration(conn)
                
                # Identify missing indexes
                missing_indexes = await self._identify_missing_indexes(conn)
                
                # Check connection pool efficiency
                connection_stats = await self._analyze_connection_performance(conn)
                
                results = {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "slow_queries": slow_queries,
                    "index_analysis": index_analysis,
                    "table_statistics": table_stats,
                    "query_patterns": query_patterns,
                    "database_configuration": db_config,
                    "missing_indexes": missing_indexes,
                    "connection_performance": connection_stats
                }
                
                return results
                
        except Exception as e:
            logger.error("Failed to analyze query performance", error=str(e))
            raise
    
    async def _analyze_slow_queries(self, conn) -> List[Dict[str, Any]]:
        """Analyze slow queries and their performance impact."""
        # Enable pg_stat_statements if available
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
        except Exception:
            logger.warning("pg_stat_statements extension not available")
            return []
        
        query = """
        SELECT 
            query,
            calls,
            total_time,
            mean_time,
            max_time,
            rows,
            100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
        FROM pg_stat_statements
        WHERE query LIKE '%rl_experiences%' OR query LIKE '%rl_training_sessions%'
        ORDER BY total_time DESC
        LIMIT 20
        """
        
        try:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.warning("Could not analyze slow queries", error=str(e))
            return []
    
    async def _analyze_index_usage(self, conn) -> Dict[str, Any]:
        """Analyze index usage and effectiveness."""
        queries = {
            "index_usage_stats": """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_size_pretty(pg_relation_size(indexrelid)) as size
                FROM pg_stat_user_indexes 
                WHERE tablename IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                ORDER BY idx_scan DESC
            """,
            
            "unused_indexes": """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    pg_size_pretty(pg_relation_size(indexrelid)) as size,
                    idx_scan
                FROM pg_stat_user_indexes 
                WHERE tablename IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                AND idx_scan = 0
                ORDER BY pg_relation_size(indexrelid) DESC
            """,
            
            "index_efficiency": """
                WITH index_ratios AS (
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch,
                        CASE WHEN idx_tup_read > 0 
                             THEN idx_tup_fetch::float / idx_tup_read 
                             ELSE 0 
                        END as efficiency_ratio
                    FROM pg_stat_user_indexes 
                    WHERE tablename IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                    AND idx_scan > 0
                )
                SELECT *
                FROM index_ratios
                ORDER BY efficiency_ratio ASC
            """
        }
        
        results = {}
        for analysis_name, query in queries.items():
            try:
                rows = await conn.fetch(query)
                results[analysis_name] = [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to analyze {analysis_name}", error=str(e))
                results[analysis_name] = []
        
        return results
    
    async def _analyze_table_statistics(self, conn) -> Dict[str, Any]:
        """Analyze table statistics and potential bloat."""
        queries = {
            "table_sizes": """
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
                    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size,
                    pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes
                FROM pg_tables 
                WHERE tablename IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """,
            
            "table_stats": """
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples,
                    n_dead_tup::float / GREATEST(n_live_tup + n_dead_tup, 1) as dead_tuple_ratio,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze
                FROM pg_stat_user_tables 
                WHERE tablename IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
            """,
            
            "bloat_estimation": """
                WITH table_stats AS (
                    SELECT 
                        schemaname,
                        tablename,
                        n_live_tup,
                        n_dead_tup,
                        pg_total_relation_size(schemaname||'.'||tablename) as total_size
                    FROM pg_stat_user_tables 
                    WHERE tablename IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                )
                SELECT 
                    schemaname,
                    tablename,
                    n_live_tup,
                    n_dead_tup,
                    pg_size_pretty(total_size) as total_size,
                    CASE 
                        WHEN n_live_tup + n_dead_tup > 0 
                        THEN ROUND(n_dead_tup::numeric / (n_live_tup + n_dead_tup) * 100, 2)
                        ELSE 0 
                    END as estimated_bloat_percent
                FROM table_stats
            """
        }
        
        results = {}
        for analysis_name, query in queries.items():
            try:
                rows = await conn.fetch(query)
                results[analysis_name] = [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to analyze {analysis_name}", error=str(e))
                results[analysis_name] = []
        
        return results
    
    async def _analyze_query_patterns(self, conn) -> Dict[str, Any]:
        """Analyze common query patterns and their frequency."""
        # This would analyze actual query logs in production
        # For now, we'll analyze expected patterns based on the schema
        
        patterns = {
            "experience_insertion_pattern": {
                "description": "Bulk experience insertion operations",
                "query_template": "INSERT INTO rl_experiences (...) VALUES (...)",
                "expected_frequency": "high",
                "optimization_target": "write_performance"
            },
            
            "prioritized_sampling_pattern": {
                "description": "Priority-based experience sampling",
                "query_template": "SELECT * FROM rl_experiences WHERE ... ORDER BY priority DESC",
                "expected_frequency": "high", 
                "optimization_target": "read_performance"
            },
            
            "session_filtering_pattern": {
                "description": "Experience queries filtered by session",
                "query_template": "SELECT * FROM rl_experiences WHERE session_id = ?",
                "expected_frequency": "medium",
                "optimization_target": "index_efficiency"
            },
            
            "time_range_queries": {
                "description": "Experience queries within time ranges",
                "query_template": "SELECT * FROM rl_experiences WHERE created_at BETWEEN ? AND ?",
                "expected_frequency": "medium",
                "optimization_target": "index_efficiency"
            },
            
            "priority_updates": {
                "description": "Batch priority updates after training",
                "query_template": "UPDATE rl_experiences SET priority = ? WHERE id IN (...)",
                "expected_frequency": "medium",
                "optimization_target": "update_performance"
            }
        }
        
        return patterns
    
    async def _analyze_database_configuration(self, conn) -> Dict[str, Any]:
        """Analyze database configuration for optimization opportunities."""
        config_queries = {
            "memory_settings": """
                SELECT name, setting, unit, short_desc
                FROM pg_settings 
                WHERE name IN (
                    'shared_buffers',
                    'effective_cache_size', 
                    'work_mem',
                    'maintenance_work_mem',
                    'wal_buffers'
                )
            """,
            
            "performance_settings": """
                SELECT name, setting, unit, short_desc
                FROM pg_settings 
                WHERE name IN (
                    'checkpoint_completion_target',
                    'wal_sync_method',
                    'synchronous_commit',
                    'random_page_cost',
                    'seq_page_cost',
                    'effective_io_concurrency'
                )
            """,
            
            "connection_settings": """
                SELECT name, setting, unit, short_desc
                FROM pg_settings 
                WHERE name IN (
                    'max_connections',
                    'shared_preload_libraries'
                )
            """
        }
        
        results = {}
        for category, query in config_queries.items():
            try:
                rows = await conn.fetch(query)
                results[category] = [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to analyze {category}", error=str(e))
                results[category] = []
        
        return results
    
    async def _identify_missing_indexes(self, conn) -> List[Dict[str, Any]]:
        """Identify potentially missing indexes based on query patterns."""
        # Common patterns that would benefit from indexes
        missing_indexes = []
        
        # Check if we have the expected indexes
        existing_indexes_query = """
            SELECT 
                t.relname as table_name,
                i.relname as index_name,
                pg_get_indexdef(i.oid) as index_definition
            FROM pg_class t, pg_class i, pg_index ix
            WHERE t.oid = ix.indrelid 
            AND i.oid = ix.indexrelid
            AND t.relname IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
        """
        
        try:
            existing_indexes = await conn.fetch(existing_indexes_query)
            existing_defs = [row['index_definition'] for row in existing_indexes]
            
            # Define recommended indexes
            recommended_indexes = [
                {
                    "table": "rl_experiences",
                    "index_name": "idx_rl_experiences_session_priority",
                    "definition": "CREATE INDEX CONCURRENTLY idx_rl_experiences_session_priority ON rl_experiences (session_id, priority DESC)",
                    "justification": "Optimizes prioritized sampling queries filtered by session"
                },
                {
                    "table": "rl_experiences", 
                    "index_name": "idx_rl_experiences_created_at",
                    "definition": "CREATE INDEX CONCURRENTLY idx_rl_experiences_created_at ON rl_experiences (created_at)",
                    "justification": "Optimizes time-range queries and cleanup operations"
                },
                {
                    "table": "rl_experiences",
                    "index_name": "idx_rl_experiences_session_step",
                    "definition": "CREATE INDEX CONCURRENTLY idx_rl_experiences_session_step ON rl_experiences (session_id, step_number)",
                    "justification": "Optimizes sequential experience retrieval"
                },
                {
                    "table": "rl_training_sessions",
                    "index_name": "idx_rl_training_sessions_status",
                    "definition": "CREATE INDEX CONCURRENTLY idx_rl_training_sessions_status ON rl_training_sessions (session_status)",
                    "justification": "Optimizes queries for active sessions"
                }
            ]
            
            # Check which indexes are missing
            for rec_idx in recommended_indexes:
                index_exists = any(rec_idx["index_name"] in existing_def for existing_def in existing_defs)
                if not index_exists:
                    missing_indexes.append(rec_idx)
            
        except Exception as e:
            logger.error("Failed to identify missing indexes", error=str(e))
        
        return missing_indexes
    
    async def _analyze_connection_performance(self, conn) -> Dict[str, Any]:
        """Analyze database connection performance."""
        connection_queries = {
            "active_connections": """
                SELECT 
                    state,
                    COUNT(*) as count
                FROM pg_stat_activity 
                WHERE datname = current_database()
                GROUP BY state
            """,
            
            "connection_age": """
                SELECT 
                    state,
                    AVG(EXTRACT(EPOCH FROM (now() - backend_start))) as avg_connection_age_seconds,
                    MAX(EXTRACT(EPOCH FROM (now() - backend_start))) as max_connection_age_seconds
                FROM pg_stat_activity 
                WHERE datname = current_database()
                GROUP BY state
            """,
            
            "waiting_queries": """
                SELECT 
                    wait_event_type,
                    wait_event,
                    COUNT(*) as count
                FROM pg_stat_activity 
                WHERE datname = current_database()
                AND wait_event IS NOT NULL
                GROUP BY wait_event_type, wait_event
                ORDER BY count DESC
            """
        }
        
        results = {}
        for analysis_name, query in connection_queries.items():
            try:
                rows = await conn.fetch(query)
                results[analysis_name] = [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to analyze {analysis_name}", error=str(e))
                results[analysis_name] = []
        
        return results
    
    async def apply_optimizations(self, dry_run: bool = True) -> List[Dict[str, Any]]:
        """Apply database optimizations based on analysis."""
        try:
            logger.info("Applying database optimizations", dry_run=dry_run)
            
            optimizations_applied = []
            
            async with get_database_connection() as conn:
                # Apply missing indexes
                missing_indexes = await self._identify_missing_indexes(conn)
                
                for idx in missing_indexes:
                    optimization = {
                        "type": "create_index",
                        "table": idx["table"],
                        "index_name": idx["index_name"],
                        "sql": idx["definition"],
                        "justification": idx["justification"],
                        "applied": False,
                        "error": None
                    }
                    
                    if not dry_run:
                        try:
                            start_time = time.time()
                            await conn.execute(idx["definition"])
                            duration = time.time() - start_time
                            
                            optimization.update({
                                "applied": True,
                                "duration_seconds": duration
                            })
                            
                            logger.info("Created index", 
                                      index_name=idx["index_name"],
                                      duration=duration)
                            
                        except Exception as e:
                            optimization["error"] = str(e)
                            logger.error("Failed to create index", 
                                       index_name=idx["index_name"],
                                       error=str(e))
                    
                    optimizations_applied.append(optimization)
                
                # Update table statistics
                table_stats_optimization = {
                    "type": "update_statistics",
                    "sql": "ANALYZE rl_experiences, rl_training_sessions, rl_performance_metrics",
                    "justification": "Update table statistics for better query planning",
                    "applied": False,
                    "error": None
                }
                
                if not dry_run:
                    try:
                        start_time = time.time()
                        await conn.execute("ANALYZE rl_experiences")
                        await conn.execute("ANALYZE rl_training_sessions") 
                        await conn.execute("ANALYZE rl_performance_metrics")
                        duration = time.time() - start_time
                        
                        table_stats_optimization.update({
                            "applied": True,
                            "duration_seconds": duration
                        })
                        
                        logger.info("Updated table statistics", duration=duration)
                        
                    except Exception as e:
                        table_stats_optimization["error"] = str(e)
                        logger.error("Failed to update statistics", error=str(e))
                
                optimizations_applied.append(table_stats_optimization)
                
                # Check for tables that need vacuuming
                table_stats = await self._analyze_table_statistics(conn)
                bloat_data = table_stats.get("bloat_estimation", [])
                
                for table in bloat_data:
                    bloat_percent = table.get("estimated_bloat_percent", 0)
                    if bloat_percent > 20:  # More than 20% bloat
                        vacuum_optimization = {
                            "type": "vacuum_table",
                            "table": table["tablename"],
                            "sql": f"VACUUM ANALYZE {table['tablename']}",
                            "justification": f"Table has {bloat_percent}% estimated bloat",
                            "applied": False,
                            "error": None
                        }
                        
                        if not dry_run:
                            try:
                                start_time = time.time()
                                await conn.execute(f"VACUUM ANALYZE {table['tablename']}")
                                duration = time.time() - start_time
                                
                                vacuum_optimization.update({
                                    "applied": True,
                                    "duration_seconds": duration
                                })
                                
                                logger.info("Vacuumed table", 
                                          table=table["tablename"],
                                          duration=duration)
                                
                            except Exception as e:
                                vacuum_optimization["error"] = str(e)
                                logger.error("Failed to vacuum table",
                                           table=table["tablename"],
                                           error=str(e))
                        
                        optimizations_applied.append(vacuum_optimization)
            
            self.applied_optimizations = optimizations_applied
            return optimizations_applied
            
        except Exception as e:
            logger.error("Failed to apply optimizations", error=str(e))
            raise
    
    async def benchmark_performance(self, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark database performance with common queries."""
        try:
            logger.info("Starting database performance benchmark", iterations=iterations)
            
            benchmarks = {}
            
            async with get_database_connection() as conn:
                # Common query patterns to benchmark
                queries = {
                    "experience_count": "SELECT COUNT(*) FROM rl_experiences",
                    
                    "recent_experiences": """
                        SELECT * FROM rl_experiences 
                        ORDER BY created_at DESC 
                        LIMIT 100
                    """,
                    
                    "prioritized_sample": """
                        SELECT * FROM rl_experiences 
                        WHERE session_id = (SELECT session_id FROM rl_training_sessions LIMIT 1)
                        ORDER BY priority DESC 
                        LIMIT 32
                    """,
                    
                    "session_stats": """
                        SELECT 
                            session_id,
                            COUNT(*) as experience_count,
                            AVG(reward) as avg_reward,
                            MAX(step_number) as max_step
                        FROM rl_experiences 
                        GROUP BY session_id
                    """,
                    
                    "time_range_query": """
                        SELECT COUNT(*) FROM rl_experiences 
                        WHERE created_at >= NOW() - INTERVAL '24 hours'
                    """
                }
                
                for query_name, sql in queries.items():
                    times = []
                    
                    for i in range(iterations):
                        start_time = time.time()
                        try:
                            await conn.fetch(sql)
                            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
                            times.append(duration)
                        except Exception as e:
                            logger.error(f"Benchmark query failed: {query_name}", error=str(e))
                            continue
                    
                    if times:
                        benchmarks[query_name] = {
                            "avg_time_ms": sum(times) / len(times),
                            "min_time_ms": min(times),
                            "max_time_ms": max(times),
                            "iterations": len(times),
                            "sql": sql
                        }
            
            return benchmarks
            
        except Exception as e:
            logger.error("Failed to benchmark performance", error=str(e))
            raise
    
    async def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        try:
            logger.info("Generating database optimization report")
            
            # Analyze current performance
            performance_analysis = await self.analyze_query_performance()
            
            # Run benchmarks
            benchmarks = await self.benchmark_performance()
            
            # Identify optimizations (dry run)
            potential_optimizations = await self.apply_optimizations(dry_run=True)
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "performance_analysis": performance_analysis,
                "benchmarks": benchmarks,
                "potential_optimizations": potential_optimizations,
                "recommendations": self._generate_recommendations(
                    performance_analysis, benchmarks, potential_optimizations
                )
            }
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate optimization report", error=str(e))
            raise
    
    def _generate_recommendations(self, performance_analysis: Dict[str, Any],
                                benchmarks: Dict[str, Any],
                                optimizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Check for slow queries
        slow_queries = performance_analysis.get("slow_queries", [])
        if slow_queries:
            high_impact_queries = [q for q in slow_queries if q.get("mean_time", 0) > 100]
            if high_impact_queries:
                recommendations.append({
                    "priority": "high",
                    "category": "query_optimization",
                    "issue": "slow_queries_detected",
                    "description": f"Found {len(high_impact_queries)} queries with >100ms average execution time",
                    "action": "Review and optimize slow queries, consider adding indexes"
                })
        
        # Check for unused indexes
        index_analysis = performance_analysis.get("index_analysis", {})
        unused_indexes = index_analysis.get("unused_indexes", [])
        if unused_indexes:
            recommendations.append({
                "priority": "medium",
                "category": "index_optimization", 
                "issue": "unused_indexes",
                "description": f"Found {len(unused_indexes)} unused indexes consuming space",
                "action": "Consider dropping unused indexes to improve write performance"
            })
        
        # Check for missing indexes
        missing_indexes = [opt for opt in optimizations if opt["type"] == "create_index"]
        if missing_indexes:
            recommendations.append({
                "priority": "high",
                "category": "index_optimization",
                "issue": "missing_indexes",
                "description": f"Found {len(missing_indexes)} recommended indexes that could improve performance",
                "action": "Create missing indexes to optimize query performance"
            })
        
        # Check for table bloat
        table_stats = performance_analysis.get("table_statistics", {})
        bloat_data = table_stats.get("bloat_estimation", [])
        high_bloat_tables = [t for t in bloat_data if t.get("estimated_bloat_percent", 0) > 20]
        if high_bloat_tables:
            recommendations.append({
                "priority": "medium",
                "category": "maintenance",
                "issue": "table_bloat",
                "description": f"Found {len(high_bloat_tables)} tables with >20% estimated bloat",
                "action": "Run VACUUM on bloated tables to reclaim space"
            })
        
        # Check benchmark performance
        slow_benchmarks = {k: v for k, v in benchmarks.items() if v.get("avg_time_ms", 0) > 50}
        if slow_benchmarks:
            recommendations.append({
                "priority": "medium",
                "category": "performance_tuning",
                "issue": "slow_benchmark_queries",
                "description": f"Found {len(slow_benchmarks)} benchmark queries with >50ms average time",
                "action": "Investigate and optimize slow query patterns"
            })
        
        return recommendations


async def main():
    """Main entry point for database performance optimization."""
    try:
        logger.info("Starting database performance optimization")
        
        # Load configuration
        config = get_config()
        
        # Create optimizer
        optimizer = DatabasePerformanceOptimizer(config)
        
        # Parse command line arguments
        mode = "analyze"  # Default mode
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
        
        if mode == "analyze":
            # Generate optimization report
            report = await optimizer.generate_optimization_report()
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"database_optimization_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"\nDatabase Optimization Report generated: {report_file}")
            
            # Print summary
            recommendations = report.get("recommendations", [])
            if recommendations:
                print(f"\nKey Recommendations ({len(recommendations)}):")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"{i}. [{rec['priority'].upper()}] {rec['description']}")
            
        elif mode == "optimize":
            # Apply optimizations
            print("Applying database optimizations...")
            optimizations = await optimizer.apply_optimizations(dry_run=False)
            
            successful = len([opt for opt in optimizations if opt.get("applied", False)])
            failed = len([opt for opt in optimizations if opt.get("error")])
            
            print(f"\nOptimization Results:")
            print(f"  Successfully applied: {successful}")
            print(f"  Failed: {failed}")
            
            if failed > 0:
                print("\nErrors:")
                for opt in optimizations:
                    if opt.get("error"):
                        print(f"  - {opt['type']}: {opt['error']}")
        
        elif mode == "benchmark":
            # Run performance benchmarks
            print("Running database performance benchmarks...")
            benchmarks = await optimizer.benchmark_performance(iterations=20)
            
            print(f"\nBenchmark Results:")
            for query_name, results in benchmarks.items():
                print(f"  {query_name}: {results['avg_time_ms']:.2f}ms avg ({results['min_time_ms']:.2f}-{results['max_time_ms']:.2f}ms)")
        
        else:
            print("Usage: python optimize_database_performance.py [analyze|optimize|benchmark]")
            print("  analyze   - Generate optimization report (default)")
            print("  optimize  - Apply optimizations") 
            print("  benchmark - Run performance benchmarks")
            sys.exit(1)
            
    except Exception as e:
        logger.error("Database optimization failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())