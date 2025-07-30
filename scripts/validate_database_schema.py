#!/usr/bin/env python3
"""
Database Schema and Connectivity Validation Script for Shyvr RLTE
Performs comprehensive database validation including schema, connectivity, and performance
"""

import asyncio
import asyncpg
import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from google.cloud import secretmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseValidationResult:
    """Database validation result container"""
    test_category: str
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

class DatabaseValidator:
    """Comprehensive database validation system"""
    
    def __init__(self, project_id: str = "shvyr-ai-bots"):
        self.project_id = project_id
        self.connection: Optional[asyncpg.Connection] = None
        self.results: List[DatabaseValidationResult] = []
        
    async def get_database_connection(self) -> Tuple[bool, Optional[asyncpg.Connection], str]:
        """Get database connection with proper credentials"""
        try:
            # Get database credentials from Secret Manager
            client = secretmanager.SecretManagerServiceClient()
            
            # Get database password
            password_secret = f"projects/{self.project_id}/secrets/db-password-production/versions/latest"
            password_response = client.access_secret_version(request={"name": password_secret})
            password = password_response.payload.data.decode("UTF-8")
            
            # Get database URL or construct from components
            try:
                db_url_secret = f"projects/{self.project_id}/secrets/DATABASE_URL/versions/latest"
                db_url_response = client.access_secret_version(request={"name": db_url_secret})
                database_url = db_url_response.payload.data.decode("UTF-8")
                
                # Connect using URL
                conn = await asyncpg.connect(database_url)
                return True, conn, "Connected using DATABASE_URL"
                
            except Exception:
                # Fallback to component-based connection
                conn = await asyncpg.connect(
                    host="localhost",  # Cloud SQL proxy
                    port=5433,
                    database="shyvr_rlte_prod",
                    user="rlte_prod_user",
                    password=password
                )
                return True, conn, "Connected using component credentials"
                
        except Exception as e:
            return False, None, f"Connection failed: {str(e)}"
    
    def add_result(self, result: DatabaseValidationResult):
        """Add validation result"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        logger.info(f"{status} {result.test_category}.{result.test_name}: {result.message}")
    
    async def validate_connectivity(self) -> List[DatabaseValidationResult]:
        """Validate database connectivity"""
        results = []
        
        # Test connection establishment
        start_time = time.time()
        success, conn, message = await self.get_database_connection()
        connection_time = (time.time() - start_time) * 1000
        
        results.append(DatabaseValidationResult(
            test_category="Connectivity",
            test_name="Connection Establishment",
            passed=success,
            message=message,
            execution_time_ms=connection_time,
            details={"connection_time_ms": connection_time}
        ))
        
        if success and conn:
            self.connection = conn
            
            # Test basic query execution
            try:
                start_time = time.time()
                result = await conn.fetchval("SELECT 1")
                query_time = (time.time() - start_time) * 1000
                
                results.append(DatabaseValidationResult(
                    test_category="Connectivity",
                    test_name="Basic Query",
                    passed=result == 1,
                    message=f"Basic query executed successfully in {query_time:.2f}ms",
                    execution_time_ms=query_time,
                    details={"query_result": result}
                ))
                
            except Exception as e:
                results.append(DatabaseValidationResult(
                    test_category="Connectivity",
                    test_name="Basic Query",
                    passed=False,
                    message=f"Basic query failed: {str(e)}",
                    details={"error": str(e)}
                ))
        
        return results
    
    async def validate_schema_structure(self) -> List[DatabaseValidationResult]:
        """Validate database schema structure"""
        results = []
        
        if not self.connection:
            results.append(DatabaseValidationResult(
                test_category="Schema",
                test_name="Schema Validation",
                passed=False,
                message="No database connection available for schema validation"
            ))
            return results
        
        # Check for core tables
        core_tables = [
            "activity_logs",
            "rl_experiences",
            "rl_training_sessions",
            "rl_performance_metrics"
        ]
        
        try:
            # Get all tables in public schema
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """
            
            start_time = time.time()
            existing_tables = await self.connection.fetch(tables_query)
            query_time = (time.time() - start_time) * 1000
            
            existing_table_names = [row['table_name'] for row in existing_tables]
            
            results.append(DatabaseValidationResult(
                test_category="Schema",
                test_name="Table Discovery",
                passed=len(existing_table_names) > 0,
                message=f"Found {len(existing_table_names)} tables in database",
                execution_time_ms=query_time,
                details={"tables": existing_table_names}
            ))
            
            # Check each core table
            for table_name in core_tables:
                table_exists = table_name in existing_table_names
                results.append(DatabaseValidationResult(
                    test_category="Schema",
                    test_name=f"Table {table_name}",
                    passed=table_exists,
                    message=f"Table {table_name} {'exists' if table_exists else 'missing'}",
                    details={"table_name": table_name, "exists": table_exists}
                ))
                
        except Exception as e:
            results.append(DatabaseValidationResult(
                test_category="Schema",
                test_name="Schema Structure",
                passed=False,
                message=f"Schema validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_table_columns(self) -> List[DatabaseValidationResult]:
        """Validate table column structure"""
        results = []
        
        if not self.connection:
            return results
        
        # Expected column structures for core tables
        expected_columns = {
            "activity_logs": [
                "id", "timestamp", "level", "component", "action", 
                "details", "user_id", "session_id", "metadata"
            ],
            "rl_experiences": [
                "id", "timestamp", "state", "action", "reward", 
                "next_state", "done", "episode_id", "step"
            ],
            "rl_training_sessions": [
                "id", "start_time", "end_time", "episode_count", 
                "total_reward", "configuration", "status"
            ],
            "rl_performance_metrics": [
                "id", "timestamp", "session_id", "metric_name", 
                "metric_value", "episode_id"
            ]
        }
        
        for table_name, expected_cols in expected_columns.items():
            try:
                # Get column information
                columns_query = """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = $1
                    ORDER BY ordinal_position
                """
                
                start_time = time.time()
                columns = await self.connection.fetch(columns_query, table_name)
                query_time = (time.time() - start_time) * 1000
                
                if not columns:
                    results.append(DatabaseValidationResult(
                        test_category="Schema",
                        test_name=f"Columns {table_name}",
                        passed=False,
                        message=f"Table {table_name} not found or has no columns",
                        execution_time_ms=query_time
                    ))
                    continue
                
                actual_cols = [row['column_name'] for row in columns]
                missing_cols = set(expected_cols) - set(actual_cols)
                extra_cols = set(actual_cols) - set(expected_cols)
                
                # Check if all expected columns exist
                all_expected_present = len(missing_cols) == 0
                
                results.append(DatabaseValidationResult(
                    test_category="Schema",
                    test_name=f"Columns {table_name}",
                    passed=all_expected_present,
                    message=f"Table {table_name}: {len(actual_cols)} columns, {len(missing_cols)} missing, {len(extra_cols)} extra",
                    execution_time_ms=query_time,
                    details={
                        "expected_columns": expected_cols,
                        "actual_columns": actual_cols,
                        "missing_columns": list(missing_cols),
                        "extra_columns": list(extra_cols),
                        "column_details": [dict(row) for row in columns]
                    }
                ))
                
            except Exception as e:
                results.append(DatabaseValidationResult(
                    test_category="Schema",
                    test_name=f"Columns {table_name}",
                    passed=False,
                    message=f"Column validation failed for {table_name}: {str(e)}",
                    details={"error": str(e), "table_name": table_name}
                ))
        
        return results
    
    async def validate_indexes_and_constraints(self) -> List[DatabaseValidationResult]:
        """Validate database indexes and constraints"""
        results = []
        
        if not self.connection:
            return results
        
        try:
            # Check for indexes
            indexes_query = """
                SELECT schemaname, tablename, indexname, indexdef
                FROM pg_indexes 
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """
            
            start_time = time.time()
            indexes = await self.connection.fetch(indexes_query)
            query_time = (time.time() - start_time) * 1000
            
            # Group indexes by table
            indexes_by_table = {}
            for idx in indexes:
                table = idx['tablename']
                if table not in indexes_by_table:
                    indexes_by_table[table] = []
                indexes_by_table[table].append({
                    'name': idx['indexname'],
                    'definition': idx['indexdef']
                })
            
            results.append(DatabaseValidationResult(
                test_category="Schema",
                test_name="Index Discovery",
                passed=len(indexes) > 0,
                message=f"Found {len(indexes)} indexes across {len(indexes_by_table)} tables",
                execution_time_ms=query_time,
                details={"indexes_by_table": indexes_by_table}
            ))
            
            # Check for critical indexes on RL tables
            critical_indexes = {
                "rl_experiences": ["episode_id", "timestamp"],
                "rl_performance_metrics": ["session_id", "timestamp"],
                "activity_logs": ["timestamp", "component"]
            }
            
            for table_name, index_columns in critical_indexes.items():
                table_indexes = indexes_by_table.get(table_name, [])
                table_index_defs = [idx['definition'].lower() for idx in table_indexes]
                
                for column in index_columns:
                    has_index = any(column.lower() in idx_def for idx_def in table_index_defs)
                    results.append(DatabaseValidationResult(
                        test_category="Schema",
                        test_name=f"Index {table_name}.{column}",
                        passed=has_index,
                        message=f"Index on {table_name}.{column} {'exists' if has_index else 'missing (performance impact possible)'}",
                        details={"table": table_name, "column": column, "has_index": has_index}
                    ))
            
        except Exception as e:
            results.append(DatabaseValidationResult(
                test_category="Schema",
                test_name="Indexes and Constraints",
                passed=False,
                message=f"Index validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_data_integrity(self) -> List[DatabaseValidationResult]:
        """Validate data integrity and constraints"""
        results = []
        
        if not self.connection:
            return results
        
        try:
            # Check record counts in main tables
            tables_to_check = ["activity_logs", "rl_experiences", "rl_training_sessions", "rl_performance_metrics"]
            
            for table_name in tables_to_check:
                try:
                    start_time = time.time()
                    count = await self.connection.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                    query_time = (time.time() - start_time) * 1000
                    
                    results.append(DatabaseValidationResult(
                        test_category="Data Integrity",
                        test_name=f"Record Count {table_name}",
                        passed=True,  # Always pass, just informational
                        message=f"Table {table_name} contains {count} records",
                        execution_time_ms=query_time,
                        details={"table": table_name, "record_count": count}
                    ))
                    
                except Exception as e:
                    results.append(DatabaseValidationResult(
                        test_category="Data Integrity",
                        test_name=f"Record Count {table_name}",
                        passed=False,
                        message=f"Failed to count records in {table_name}: {str(e)}",
                        details={"error": str(e), "table": table_name}
                    ))
            
            # Check for recent data (if any exists)
            try:
                recent_activity = await self.connection.fetchval("""
                    SELECT COUNT(*) FROM activity_logs 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                """)
                
                results.append(DatabaseValidationResult(
                    test_category="Data Integrity",
                    test_name="Recent Activity",
                    passed=True,  # Informational
                    message=f"Found {recent_activity} activity log entries in last 24 hours",
                    details={"recent_records": recent_activity}
                ))
                
            except Exception as e:
                results.append(DatabaseValidationResult(
                    test_category="Data Integrity",
                    test_name="Recent Activity",
                    passed=False,
                    message=f"Failed to check recent activity: {str(e)}",
                    details={"error": str(e)}
                ))
            
        except Exception as e:
            results.append(DatabaseValidationResult(
                test_category="Data Integrity",
                test_name="Data Integrity Check",
                passed=False,
                message=f"Data integrity validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_performance(self) -> List[DatabaseValidationResult]:
        """Validate database performance characteristics"""
        results = []
        
        if not self.connection:
            return results
        
        try:
            # Test query performance on main tables
            performance_tests = [
                ("SELECT COUNT(*) FROM activity_logs", "activity_logs count"),
                ("SELECT COUNT(*) FROM rl_experiences", "rl_experiences count"),
                ("SELECT * FROM activity_logs ORDER BY timestamp DESC LIMIT 10", "recent activity logs"),
                ("SELECT * FROM rl_experiences ORDER BY timestamp DESC LIMIT 100", "recent rl experiences")
            ]
            
            for query, test_name in performance_tests:
                try:
                    start_time = time.time()
                    result = await self.connection.fetch(query)
                    query_time = (time.time() - start_time) * 1000
                    
                    # Consider queries over 5 seconds as slow
                    is_fast = query_time < 5000
                    
                    results.append(DatabaseValidationResult(
                        test_category="Performance",
                        test_name=f"Query Performance: {test_name}",
                        passed=is_fast,
                        message=f"Query completed in {query_time:.2f}ms {'(fast)' if is_fast else '(slow - consider optimization)'}",
                        execution_time_ms=query_time,
                        details={"query": query, "result_count": len(result)}
                    ))
                    
                except Exception as e:
                    results.append(DatabaseValidationResult(
                        test_category="Performance",
                        test_name=f"Query Performance: {test_name}",
                        passed=False,
                        message=f"Performance test failed: {str(e)}",
                        details={"error": str(e), "query": query}
                    ))
            
            # Check connection pool performance
            start_time = time.time()
            await self.connection.fetchval("SELECT 1")
            connection_reuse_time = (time.time() - start_time) * 1000
            
            results.append(DatabaseValidationResult(
                test_category="Performance",
                test_name="Connection Reuse",
                passed=connection_reuse_time < 100,  # Should be very fast
                message=f"Connection reuse test: {connection_reuse_time:.2f}ms",
                execution_time_ms=connection_reuse_time,
                details={"connection_reuse_time_ms": connection_reuse_time}
            ))
            
        except Exception as e:
            results.append(DatabaseValidationResult(
                test_category="Performance",
                test_name="Performance Validation",
                passed=False,
                message=f"Performance validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def validate_rl_specific_features(self) -> List[DatabaseValidationResult]:
        """Validate RL-specific database features"""
        results = []
        
        if not self.connection:
            return results
        
        try:
            # Test RL experience storage functionality
            test_experience = {
                'state': json.dumps([1.0, 2.0, 3.0]),
                'action': 1,
                'reward': 0.5,
                'next_state': json.dumps([1.1, 2.1, 3.1]),
                'done': False,
                'episode_id': 'test_episode_validation',
                'step': 1
            }
            
            # Insert test experience
            start_time = time.time()
            await self.connection.execute("""
                INSERT INTO rl_experiences 
                (timestamp, state, action, reward, next_state, done, episode_id, step)
                VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7)
            """, test_experience['state'], test_experience['action'], 
                test_experience['reward'], test_experience['next_state'],
                test_experience['done'], test_experience['episode_id'], 
                test_experience['step'])
            insert_time = (time.time() - start_time) * 1000
            
            results.append(DatabaseValidationResult(
                test_category="RL Features",
                test_name="Experience Insert",
                passed=True,
                message=f"Successfully inserted test RL experience in {insert_time:.2f}ms",
                execution_time_ms=insert_time,
                details={"test_experience": test_experience}
            ))
            
            # Query test experience
            start_time = time.time()
            retrieved_exp = await self.connection.fetchrow("""
                SELECT * FROM rl_experiences 
                WHERE episode_id = $1 AND step = $2
            """, test_experience['episode_id'], test_experience['step'])
            query_time = (time.time() - start_time) * 1000
            
            results.append(DatabaseValidationResult(
                test_category="RL Features",
                test_name="Experience Query",
                passed=retrieved_exp is not None,
                message=f"Successfully queried RL experience in {query_time:.2f}ms",
                execution_time_ms=query_time,
                details={"found_experience": retrieved_exp is not None}
            ))
            
            # Clean up test data
            await self.connection.execute("""
                DELETE FROM rl_experiences 
                WHERE episode_id = $1 AND step = $2
            """, test_experience['episode_id'], test_experience['step'])
            
            # Test batch operations (important for RL)
            start_time = time.time()
            await self.connection.executemany("""
                INSERT INTO rl_experiences 
                (timestamp, state, action, reward, next_state, done, episode_id, step)
                VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7)
            """, [
                (json.dumps([i*1.0]), i, i*0.1, json.dumps([i*1.1]), False, f'batch_test_{i}', i)
                for i in range(10)
            ])
            batch_time = (time.time() - start_time) * 1000
            
            results.append(DatabaseValidationResult(
                test_category="RL Features",
                test_name="Batch Insert",
                passed=batch_time < 1000,  # Should complete in under 1 second
                message=f"Batch inserted 10 experiences in {batch_time:.2f}ms",
                execution_time_ms=batch_time,
                details={"batch_size": 10}
            ))
            
            # Clean up batch test data
            await self.connection.execute("""
                DELETE FROM rl_experiences 
                WHERE episode_id LIKE 'batch_test_%'
            """)
            
        except Exception as e:
            results.append(DatabaseValidationResult(
                test_category="RL Features",
                test_name="RL Feature Validation",
                passed=False,
                message=f"RL feature validation failed: {str(e)}",
                details={"error": str(e)}
            ))
        
        return results
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all database validations"""
        logger.info("🔍 Starting comprehensive database validation")
        
        validation_categories = [
            ("Connectivity", self.validate_connectivity()),
            ("Schema Structure", self.validate_schema_structure()),
            ("Table Columns", self.validate_table_columns()),
            ("Indexes and Constraints", self.validate_indexes_and_constraints()),
            ("Data Integrity", self.validate_data_integrity()),
            ("Performance", self.validate_performance()),
            ("RL Features", self.validate_rl_specific_features())
        ]
        
        start_time = time.time()
        
        for category_name, validation_task in validation_categories:
            logger.info(f"🧪 Validating {category_name}...")
            try:
                category_results = await validation_task
                for result in category_results:
                    self.add_result(result)
            except Exception as e:
                error_result = DatabaseValidationResult(
                    test_category=category_name,
                    test_name="Category Validation",
                    passed=False,
                    message=f"Category validation failed: {str(e)}",
                    details={"error": str(e)}
                )
                self.add_result(error_result)
        
        total_time = time.time() - start_time
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Categorize results
        results_by_category = {}
        for result in self.results:
            if result.test_category not in results_by_category:
                results_by_category[result.test_category] = []
            results_by_category[result.test_category].append(asdict(result))
        
        summary = {
            "validation_complete": True,
            "timestamp": datetime.utcnow().isoformat(),
            "execution_time_seconds": total_time,
            "database_ready": failed_tests == 0,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate_percent": round(success_rate, 2)
            },
            "results_by_category": results_by_category,
            "critical_failures": [
                asdict(r) for r in self.results 
                if not r.passed and r.test_category in ["Connectivity", "Schema Structure"]
            ]
        }
        
        # Close connection
        if self.connection:
            await self.connection.close()
        
        # Log summary
        logger.info(f"✅ Database validation complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        if failed_tests > 0:
            logger.error(f"❌ {failed_tests} database tests failed")
        else:
            logger.info("🎉 All database validations passed - database ready for production")
        
        return summary

async def main():
    """Main database validation runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Shyvr RLTE database")
    parser.add_argument("--project-id", default="shvyr-ai-bots", help="GCP project ID")
    parser.add_argument("--output-file", help="Output file for validation results")
    
    args = parser.parse_args()
    
    validator = DatabaseValidator(project_id=args.project_id)
    
    try:
        results = await validator.run_all_validations()
        
        # Save results
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_file = Path(__file__).parent.parent / "reports" / f"database_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📄 Database validation report saved to: {output_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results["database_ready"] else 1)
        
    except Exception as e:
        logger.error(f"💥 Database validation failed with exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())