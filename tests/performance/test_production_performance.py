#!/usr/bin/env python3
"""
Production Database Performance and Reliability Testing
Comprehensive testing of RL experience database under production conditions
"""

import asyncio
import logging
import time
import json
import sys
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple
import asyncpg
from concurrent.futures import ThreadPoolExecutor
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_production_migrations import ProductionMigrationRunner

logger = logging.getLogger(__name__)


class ProductionPerformanceTesting:
    """Production database performance and reliability testing"""
    
    def __init__(self):
        self.project_id = "shvyr-ai-bots"
        self.connection_name = "shvyr-ai-bots:us-central1:shyvr-rlte-db-prod"
        self.proxy_port = 5433
        self.migration_runner = ProductionMigrationRunner()
        self.test_session_id = str(uuid.uuid4())
        
    async def get_connection(self) -> asyncpg.Connection:
        """Get database connection"""
        return await self.migration_runner.get_database_connection()
    
    async def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic database connectivity"""
        logger.info("Testing basic database connectivity...")
        
        results = {
            "test_name": "basic_connectivity",
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        try:
            # Start proxy
            if not self.migration_runner.start_cloud_sql_proxy():
                results["status"] = "failed"
                results["error"] = "Failed to start Cloud SQL proxy"
                return results
            
            start_time = time.time()
            conn = await self.get_connection()
            connection_time = time.time() - start_time
            
            try:
                # Test basic query
                query_start = time.time()
                result = await conn.fetchval("SELECT 1")
                query_time = time.time() - query_start
                
                # Test RL tables exist
                tables_query_start = time.time()
                tables = await conn.fetch("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('rl_experiences', 'rl_training_sessions', 'rl_performance_metrics')
                    ORDER BY table_name
                """)
                tables_query_time = time.time() - tables_query_start
                
                results.update({
                    "status": "passed",
                    "connection_time_ms": connection_time * 1000,
                    "basic_query_time_ms": query_time * 1000,
                    "tables_query_time_ms": tables_query_time * 1000,
                    "tables_found": len(tables),
                    "expected_tables": 3,
                    "query_result": result
                })
                
                if result != 1:
                    results["status"] = "failed"
                    results["error"] = f"Unexpected query result: {result}"
                elif len(tables) != 3:
                    results["status"] = "failed"
                    results["error"] = f"Missing tables: found {len(tables)}, expected 3"
                
            finally:
                await conn.close()
                
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Basic connectivity test failed: {e}")
        finally:
            self.migration_runner.stop_cloud_sql_proxy()
            results["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def test_query_performance(self) -> Dict[str, Any]:
        """Test query performance for RL operations"""
        logger.info("Testing RL query performance...")
        
        results = {
            "test_name": "query_performance",
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "queries": []
        }
        
        try:
            if not self.migration_runner.start_cloud_sql_proxy():
                results["status"] = "failed"
                results["error"] = "Failed to start Cloud SQL proxy"
                return results
            
            conn = await self.get_connection()
            
            try:
                # Test queries for RL experience operations
                test_queries = [
                    {
                        "name": "count_experiences",
                        "query": "SELECT COUNT(*) FROM rl_experiences",
                        "expected_time_ms": 100
                    },
                    {
                        "name": "count_sessions",
                        "query": "SELECT COUNT(*) FROM rl_training_sessions",
                        "expected_time_ms": 50
                    },
                    {
                        "name": "recent_experiences",
                        "query": "SELECT * FROM rl_experiences ORDER BY created_at DESC LIMIT 10",
                        "expected_time_ms": 200
                    },
                    {
                        "name": "session_stats",
                        "query": """
                            SELECT session_id, COUNT(*) as experience_count, AVG(reward) as avg_reward
                            FROM rl_experiences 
                            GROUP BY session_id 
                            LIMIT 5
                        """,
                        "expected_time_ms": 300
                    },
                    {
                        "name": "priority_sampling",
                        "query": "SELECT * FROM rl_experiences ORDER BY priority DESC LIMIT 32",
                        "expected_time_ms": 150
                    },
                    {
                        "name": "trading_mode_filter",
                        "query": "SELECT * FROM rl_experiences WHERE trading_mode = 'simulation' LIMIT 10",
                        "expected_time_ms": 200
                    }
                ]
                
                total_queries = 0
                passed_queries = 0
                
                for query_test in test_queries:
                    logger.info(f"Running query test: {query_test['name']}")
                    
                    query_result = {
                        "name": query_test["name"],
                        "query": query_test["query"].strip(),
                        "expected_time_ms": query_test["expected_time_ms"]
                    }
                    
                    try:
                        # Run query multiple times for accuracy
                        times = []
                        for _ in range(3):
                            start_time = time.time()
                            result = await conn.fetch(query_test["query"])
                            query_time = (time.time() - start_time) * 1000
                            times.append(query_time)
                        
                        query_result.update({
                            "times_ms": times,
                            "avg_time_ms": statistics.mean(times),
                            "min_time_ms": min(times),
                            "max_time_ms": max(times),
                            "result_count": len(result),
                            "status": "passed" if statistics.mean(times) <= query_test["expected_time_ms"] else "warning"
                        })
                        
                        if query_result["status"] == "passed":
                            passed_queries += 1
                        
                        total_queries += 1
                        
                    except Exception as e:
                        query_result.update({
                            "status": "failed",
                            "error": str(e)
                        })
                        total_queries += 1
                    
                    results["queries"].append(query_result)
                
                # Overall performance assessment
                results.update({
                    "status": "passed" if passed_queries == total_queries else "partial",
                    "total_queries": total_queries,
                    "passed_queries": passed_queries,
                    "failed_queries": total_queries - passed_queries,
                    "success_rate": passed_queries / total_queries if total_queries > 0 else 0
                })
                
            finally:
                await conn.close()
                
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Query performance test failed: {e}")
        finally:
            self.migration_runner.stop_cloud_sql_proxy()
            results["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def test_concurrent_connections(self, num_connections: int = 5) -> Dict[str, Any]:
        """Test concurrent database connections"""
        logger.info(f"Testing {num_connections} concurrent connections...")
        
        results = {
            "test_name": "concurrent_connections",
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "num_connections": num_connections,
            "connections": []
        }
        
        try:
            if not self.migration_runner.start_cloud_sql_proxy():
                results["status"] = "failed"
                results["error"] = "Failed to start Cloud SQL proxy"
                return results
            
            async def test_single_connection(connection_id: int):
                """Test a single connection"""
                conn_result = {
                    "connection_id": connection_id,
                    "start_time": time.time()
                }
                
                try:
                    conn = await self.get_connection()
                    conn_result["connection_time"] = time.time() - conn_result["start_time"]
                    
                    try:
                        # Perform a simple query
                        query_start = time.time()
                        result = await conn.fetchval("SELECT COUNT(*) FROM rl_experiences")
                        conn_result["query_time"] = time.time() - query_start
                        conn_result["query_result"] = result
                        conn_result["status"] = "success"
                        
                    finally:
                        await conn.close()
                        
                except Exception as e:
                    conn_result["status"] = "failed"
                    conn_result["error"] = str(e)
                
                conn_result["total_time"] = time.time() - conn_result["start_time"]
                return conn_result
            
            # Create concurrent connections
            tasks = [test_single_connection(i) for i in range(num_connections)]
            connection_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_connections = 0
            total_connection_time = 0
            total_query_time = 0
            
            for i, conn_result in enumerate(connection_results):
                if isinstance(conn_result, Exception):
                    results["connections"].append({
                        "connection_id": i,
                        "status": "failed",
                        "error": str(conn_result)
                    })
                else:
                    results["connections"].append(conn_result)
                    if conn_result.get("status") == "success":
                        successful_connections += 1
                        total_connection_time += conn_result.get("connection_time", 0)
                        total_query_time += conn_result.get("query_time", 0)
            
            # Calculate averages
            if successful_connections > 0:
                avg_connection_time = total_connection_time / successful_connections
                avg_query_time = total_query_time / successful_connections
            else:
                avg_connection_time = 0
                avg_query_time = 0
            
            results.update({
                "status": "passed" if successful_connections == num_connections else "partial",
                "successful_connections": successful_connections,
                "failed_connections": num_connections - successful_connections,
                "success_rate": successful_connections / num_connections,
                "avg_connection_time_ms": avg_connection_time * 1000,
                "avg_query_time_ms": avg_query_time * 1000
            })
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Concurrent connections test failed: {e}")
        finally:
            self.migration_runner.stop_cloud_sql_proxy()
            results["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def test_data_operations(self) -> Dict[str, Any]:
        """Test data insertion, update, and deletion operations"""
        logger.info("Testing data operations (insert, update, delete)...")
        
        results = {
            "test_name": "data_operations",
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "operations": []
        }
        
        try:
            if not self.migration_runner.start_cloud_sql_proxy():
                results["status"] = "failed"
                results["error"] = "Failed to start Cloud SQL proxy"
                return results
            
            conn = await self.get_connection()
            
            try:
                # Test 1: Insert RL training session
                session_insert_start = time.time()
                session_insert_query = """
                    INSERT INTO rl_training_sessions (
                        session_id, session_name, trading_mode, agent_config, environment_config
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING id, session_id
                """
                
                session_result = await conn.fetchrow(
                    session_insert_query,
                    self.test_session_id,
                    "Performance Test Session",
                    "simulation",
                    '{"algorithm": "DQN", "learning_rate": 0.001}',
                    '{"initial_balance": 1000.0}'
                )
                session_insert_time = time.time() - session_insert_start
                
                results["operations"].append({
                    "name": "insert_training_session",
                    "time_ms": session_insert_time * 1000,
                    "status": "success",
                    "session_id": str(session_result["session_id"])
                })
                
                # Test 2: Insert multiple RL experiences
                experiences_insert_start = time.time()
                experience_data = []
                for i in range(10):
                    experience_data.append((
                        str(uuid.uuid4()),  # experience_id
                        self.test_session_id,  # session_id
                        f'{{"price": {1.0 + i * 0.01}, "position": 0.{i}}}',  # state_data
                        i % 3,  # action
                        0.1 + (i * 0.01),  # reward
                        f'{{"price": {1.0 + (i+1) * 0.01}, "position": 0.{i+1}}}',  # next_state_data
                        False,  # done
                        min(0.9, 0.5 + (i * 0.05)),  # priority (ensure within 0-1 range)
                        "simulation",  # trading_mode
                        "ETH",  # token_address
                        "ethereum",  # chain
                        '{"volatility": 0.15}',  # market_conditions
                        '{"latency_ms": 100}',  # performance_metrics
                        f'{{"test_run": {i}}}'  # metadata
                    ))
                
                experience_insert_query = """
                    INSERT INTO rl_experiences (
                        experience_id, session_id, state_data, action, reward, next_state_data,
                        done, priority, trading_mode, token_address, chain, market_conditions,
                        performance_metrics, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """
                
                for exp_data in experience_data:
                    await conn.execute(experience_insert_query, *exp_data)
                
                experiences_insert_time = time.time() - experiences_insert_start
                
                results["operations"].append({
                    "name": "insert_experiences",
                    "time_ms": experiences_insert_time * 1000,
                    "status": "success",
                    "experiences_inserted": len(experience_data)
                })
                
                # Test 3: Batch query performance
                batch_query_start = time.time()
                batch_result = await conn.fetch("""
                    SELECT session_id, COUNT(*) as count, AVG(reward) as avg_reward
                    FROM rl_experiences 
                    WHERE session_id = $1
                    GROUP BY session_id
                """, self.test_session_id)
                batch_query_time = time.time() - batch_query_start
                
                results["operations"].append({
                    "name": "batch_query",
                    "time_ms": batch_query_time * 1000,
                    "status": "success",
                    "result_count": len(batch_result),
                    "avg_reward": float(batch_result[0]["avg_reward"]) if batch_result else 0
                })
                
                # Test 4: Update operations
                update_start = time.time()
                update_result = await conn.execute("""
                    UPDATE rl_experiences 
                    SET priority = priority * 1.1 
                    WHERE session_id = $1
                """, self.test_session_id)
                update_time = time.time() - update_start
                
                results["operations"].append({
                    "name": "batch_update",
                    "time_ms": update_time * 1000,
                    "status": "success",
                    "rows_updated": int(update_result.split()[-1]) if "UPDATE" in update_result else 0
                })
                
                # Test 5: Cleanup (delete test data)
                cleanup_start = time.time()
                await conn.execute("DELETE FROM rl_experiences WHERE session_id = $1", self.test_session_id)
                await conn.execute("DELETE FROM rl_training_sessions WHERE session_id = $1", self.test_session_id)
                cleanup_time = time.time() - cleanup_start
                
                results["operations"].append({
                    "name": "cleanup",
                    "time_ms": cleanup_time * 1000,
                    "status": "success"
                })
                
                # Overall assessment
                total_time = sum(op["time_ms"] for op in results["operations"])
                successful_ops = sum(1 for op in results["operations"] if op["status"] == "success")
                
                results.update({
                    "status": "passed" if successful_ops == len(results["operations"]) else "partial",
                    "total_operations": len(results["operations"]),
                    "successful_operations": successful_ops,
                    "total_time_ms": total_time
                })
                
            finally:
                await conn.close()
                
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Data operations test failed: {e}")
        finally:
            self.migration_runner.stop_cloud_sql_proxy()
            results["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def test_reliability_under_load(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Test database reliability under sustained load"""
        logger.info(f"Testing reliability under load for {duration_seconds} seconds...")
        
        results = {
            "test_name": "reliability_under_load",
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "duration_seconds": duration_seconds,
            "load_stats": {
                "queries_executed": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "total_query_time": 0,
                "errors": []
            }
        }
        
        try:
            if not self.migration_runner.start_cloud_sql_proxy():
                results["status"] = "failed"
                results["error"] = "Failed to start Cloud SQL proxy"
                return results
            
            end_time = time.time() + duration_seconds
            query_times = []
            
            while time.time() < end_time:
                try:
                    conn = await self.get_connection()
                    try:
                        query_start = time.time()
                        await conn.fetchval("SELECT COUNT(*) FROM rl_experiences")
                        query_time = time.time() - query_start
                        
                        results["load_stats"]["queries_executed"] += 1
                        results["load_stats"]["successful_queries"] += 1
                        results["load_stats"]["total_query_time"] += query_time
                        query_times.append(query_time)
                        
                    finally:
                        await conn.close()
                        
                except Exception as e:
                    results["load_stats"]["queries_executed"] += 1
                    results["load_stats"]["failed_queries"] += 1
                    results["load_stats"]["errors"].append(str(e))
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            # Calculate statistics
            if query_times:
                results["load_stats"].update({
                    "avg_query_time_ms": statistics.mean(query_times) * 1000,
                    "min_query_time_ms": min(query_times) * 1000,
                    "max_query_time_ms": max(query_times) * 1000,
                    "median_query_time_ms": statistics.median(query_times) * 1000,
                    "queries_per_second": len(query_times) / duration_seconds
                })
            
            success_rate = (results["load_stats"]["successful_queries"] / 
                          results["load_stats"]["queries_executed"] 
                          if results["load_stats"]["queries_executed"] > 0 else 0)
            
            results.update({
                "status": "passed" if success_rate >= 0.95 else "warning" if success_rate >= 0.8 else "failed",
                "success_rate": success_rate
            })
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(f"Reliability test failed: {e}")
        finally:
            self.migration_runner.stop_cloud_sql_proxy()
            results["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive performance and reliability test suite"""
        logger.info("Starting comprehensive production performance test suite...")
        
        suite_results = {
            "test_suite": "production_performance_reliability",
            "start_time": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
        
        # Define test sequence
        tests = [
            ("basic_connectivity", self.test_basic_connectivity),
            ("query_performance", self.test_query_performance),
            ("concurrent_connections", lambda: self.test_concurrent_connections(5)),
            ("data_operations", self.test_data_operations),
            ("reliability_under_load", lambda: self.test_reliability_under_load(30))
        ]
        
        passed_tests = 0
        failed_tests = 0
        warning_tests = 0
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                test_result = await test_func()
                suite_results["tests"].append(test_result)
                
                if test_result["status"] == "passed":
                    passed_tests += 1
                elif test_result["status"] == "failed":
                    failed_tests += 1
                else:
                    warning_tests += 1
                    
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                suite_results["tests"].append({
                    "test_name": test_name,
                    "status": "failed",
                    "error": str(e)
                })
                failed_tests += 1
        
        # Generate summary
        total_tests = len(tests)
        suite_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "warning_tests": warning_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_status": (
                "passed" if failed_tests == 0 and warning_tests <= 1 else
                "warning" if failed_tests == 0 else
                "failed"
            )
        }
        
        suite_results["end_time"] = datetime.now().isoformat()
        
        # Save detailed results
        report_path = self.save_test_results(suite_results)
        suite_results["report_path"] = report_path
        
        return suite_results
    
    def save_test_results(self, results: Dict[str, Any]) -> str:
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"production_performance_test_{timestamp}.json"
        
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved: {report_path}")
        return str(report_path)


async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Database Performance Testing")
    parser.add_argument("--test", choices=[
        "connectivity", "performance", "concurrent", "operations", "load", "all"
    ], default="all", help="Test to run")
    parser.add_argument("--connections", type=int, default=5, help="Number of concurrent connections")
    parser.add_argument("--duration", type=int, default=30, help="Load test duration in seconds")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    tester = ProductionPerformanceTesting()
    
    try:
        if args.test == "all":
            results = await tester.run_comprehensive_test_suite()
            
            print(f"\n🧪 Production Performance Test Suite Results")
            print(f"=" * 50)
            print(f"Overall Status: {results['summary']['overall_status'].upper()}")
            print(f"Total Tests: {results['summary']['total_tests']}")
            print(f"Passed: {results['summary']['passed_tests']}")
            print(f"Warnings: {results['summary']['warning_tests']}")
            print(f"Failed: {results['summary']['failed_tests']}")
            print(f"Success Rate: {results['summary']['success_rate']:.1%}")
            
            print(f"\n📋 Individual Test Results:")
            for test in results["tests"]:
                status_emoji = {
                    "passed": "✅",
                    "warning": "⚠️",
                    "failed": "❌",
                    "partial": "🟡"
                }.get(test["status"], "❓")
                
                print(f"  {status_emoji} {test['test_name']}: {test['status']}")
                if "error" in test:
                    print(f"    Error: {test['error']}")
            
            print(f"\n📄 Detailed report: {results['report_path']}")
            
            # Exit with appropriate code
            if results['summary']['overall_status'] == "failed":
                sys.exit(1)
            elif results['summary']['overall_status'] == "warning":
                sys.exit(2)
            else:
                sys.exit(0)
                
        else:
            # Run individual test
            test_map = {
                "connectivity": tester.test_basic_connectivity,
                "performance": tester.test_query_performance,
                "concurrent": lambda: tester.test_concurrent_connections(args.connections),
                "operations": tester.test_data_operations,
                "load": lambda: tester.test_reliability_under_load(args.duration)
            }
            
            if args.test in test_map:
                result = await test_map[args.test]()
                print(f"\n🧪 Test Result: {result['test_name']}")
                print(f"Status: {result['status']}")
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print(json.dumps(result, indent=2, default=str))
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())