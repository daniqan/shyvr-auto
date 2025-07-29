#!/usr/bin/env python3
"""
Full Integration Test Suite for RL Experience Storage
Executes comprehensive integration tests for production deployment validation
"""

import asyncio
import logging
import subprocess
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """Full integration test suite runner"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = []
        self.start_time = datetime.now()
        
    def run_command(self, command: List[str], description: str, timeout: int = 300) -> Dict[str, Any]:
        """Run command with timeout and logging"""
        logger.info(f"Running: {description}")
        logger.debug(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "description": description
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "timeout",
                "description": description,
                "timeout": timeout
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "description": description
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for RL experience storage"""
        logger.info("Running unit tests...")
        
        test_result = {
            "test_category": "unit_tests",
            "start_time": datetime.now().isoformat(),
            "tests": []
        }
        
        # Unit test categories
        unit_test_categories = [
            {
                "name": "rl_agent_tests",
                "path": "tests/unit/rl_agent/",
                "description": "RL Agent unit tests"
            },
            {
                "name": "database_tests", 
                "path": "tests/unit/database/",
                "description": "Database unit tests"
            },
            {
                "name": "activity_logging_tests",
                "path": "tests/unit/activity_logging/",
                "description": "Activity logging unit tests"
            },
            {
                "name": "dashboard_tests",
                "path": "tests/unit/dashboard/",
                "description": "Dashboard unit tests"
            },
            {
                "name": "modes_tests",
                "path": "tests/unit/modes/",
                "description": "Trading modes unit tests"
            }
        ]
        
        passed_categories = 0
        total_tests_run = 0
        total_failures = 0
        
        for category in unit_test_categories:
            logger.info(f"Running {category['description']}...")
            
            command = [
                "uv", "run", "python", "-m", "pytest",
                category["path"],
                "-v",
                "--tb=short",
                "--junit-xml=test_results.xml",
                "--cov=src",
                "--cov-report=term-missing"
            ]
            
            result = self.run_command(command, category["description"])
            
            category_result = {
                "name": category["name"],
                "description": category["description"],
                "success": result["success"],
                "output": result.get("stdout", ""),
                "errors": result.get("stderr", "")
            }
            
            if result["success"]:
                passed_categories += 1
                # Parse pytest output for test counts
                output = result.get("stdout", "")
                if "passed" in output:
                    try:
                        # Extract test counts from pytest output
                        lines = output.split("\n")
                        for line in lines:
                            if " passed" in line and "failed" in line:
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if part == "passed":
                                        passed = int(parts[i-1]) if i > 0 else 0
                                        total_tests_run += passed
                                    elif part == "failed":
                                        failed = int(parts[i-1]) if i > 0 else 0
                                        total_failures += failed
                    except (ValueError, IndexError):
                        pass
            
            test_result["tests"].append(category_result)
        
        test_result.update({
            "end_time": datetime.now().isoformat(),
            "summary": {
                "total_categories": len(unit_test_categories),
                "passed_categories": passed_categories,
                "failed_categories": len(unit_test_categories) - passed_categories,
                "total_tests_run": total_tests_run,
                "total_failures": total_failures,
                "success_rate": passed_categories / len(unit_test_categories) if unit_test_categories else 0
            },
            "overall_status": "passed" if passed_categories == len(unit_test_categories) else "failed"
        })
        
        return test_result
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        logger.info("Running integration tests...")
        
        test_result = {
            "test_category": "integration_tests",
            "start_time": datetime.now().isoformat(),
            "tests": []
        }
        
        # Integration test categories
        integration_tests = [
            {
                "name": "rl_integration",
                "path": "tests/integration/test_real_rl_integration.py",
                "description": "RL system integration tests"
            },
            {
                "name": "dashboard_integration",
                "path": "tests/integration/dashboard/",
                "description": "Dashboard integration tests"
            },
            {
                "name": "mode_integration",
                "path": "tests/integration/test_mode_integration_e2e.py",
                "description": "Trading mode integration tests"
            },
            {
                "name": "monitoring_integration",
                "path": "tests/integration/test_monitoring_integration.py",
                "description": "Monitoring integration tests"
            }
        ]
        
        passed_tests = 0
        
        for test in integration_tests:
            logger.info(f"Running {test['description']}...")
            
            command = [
                "uv", "run", "python", "-m", "pytest",
                test["path"],
                "-v",
                "--tb=short",
                "-x"  # Stop on first failure for integration tests
            ]
            
            result = self.run_command(command, test["description"], timeout=600)  # 10 min timeout
            
            test_result["tests"].append({
                "name": test["name"],
                "description": test["description"],
                "success": result["success"],
                "output": result.get("stdout", ""),
                "errors": result.get("stderr", "")
            })
            
            if result["success"]:
                passed_tests += 1
        
        test_result.update({
            "end_time": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(integration_tests),
                "passed_tests": passed_tests,
                "failed_tests": len(integration_tests) - passed_tests,
                "success_rate": passed_tests / len(integration_tests) if integration_tests else 0
            },
            "overall_status": "passed" if passed_tests == len(integration_tests) else "failed"
        })
        
        return test_result
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        logger.info("Running performance tests...")
        
        test_result = {
            "test_category": "performance_tests", 
            "start_time": datetime.now().isoformat(),
            "tests": []
        }
        
        # Performance test categories
        performance_tests = [
            {
                "name": "ml_rl_performance",
                "path": "tests/performance/test_ml_rl_performance.py",
                "description": "ML/RL performance benchmarks"
            },
            {
                "name": "experience_storage_performance",
                "path": "tests/performance/test_real_vs_mock_benchmarks.py", 
                "description": "Experience storage performance tests"
            },
            {
                "name": "production_database_performance",
                "script": "scripts/test_production_performance.py",
                "description": "Production database performance validation"
            }
        ]
        
        passed_tests = 0
        
        for test in performance_tests:
            logger.info(f"Running {test['description']}...")
            
            if "script" in test:
                # Run custom script
                command = ["uv", "run", "python", test["script"], "--test", "all"]
            else:
                # Run pytest
                command = [
                    "uv", "run", "python", "-m", "pytest", 
                    test["path"],
                    "-v",
                    "--tb=short"
                ]
            
            result = self.run_command(command, test["description"], timeout=900)  # 15 min timeout
            
            test_result["tests"].append({
                "name": test["name"],
                "description": test["description"],
                "success": result["success"],
                "output": result.get("stdout", ""),
                "errors": result.get("stderr", "")
            })
            
            if result["success"]:
                passed_tests += 1
        
        test_result.update({
            "end_time": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(performance_tests),
                "passed_tests": passed_tests,
                "failed_tests": len(performance_tests) - passed_tests,
                "success_rate": passed_tests / len(performance_tests) if performance_tests else 0
            },
            "overall_status": "passed" if passed_tests == len(performance_tests) else "failed"
        })
        
        return test_result
    
    def run_validation_tests(self) -> Dict[str, Any]:
        """Run validation tests for RL experience functionality"""
        logger.info("Running validation tests...")
        
        test_result = {
            "test_category": "validation_tests",
            "start_time": datetime.now().isoformat(),
            "tests": []
        }
        
        # Validation test categories
        validation_tests = [
            {
                "name": "experience_storage_validation",
                "description": "Validate RL experience storage functionality",
                "test_function": self.validate_experience_storage
            },
            {
                "name": "database_connectivity",
                "description": "Validate production database connectivity",
                "test_function": self.validate_database_connectivity
            },
            {
                "name": "api_endpoints",
                "description": "Validate dashboard API endpoints",
                "test_function": self.validate_api_endpoints
            },
            {
                "name": "monitoring_systems",
                "description": "Validate monitoring and alerting systems",
                "test_function": self.validate_monitoring_systems
            }
        ]
        
        passed_tests = 0
        
        for test in validation_tests:
            logger.info(f"Running {test['description']}...")
            
            try:
                result = test["test_function"]()
                test_result["tests"].append({
                    "name": test["name"],
                    "description": test["description"],
                    "success": result.get("success", False),
                    "details": result,
                    "errors": result.get("errors", [])
                })
                
                if result.get("success", False):
                    passed_tests += 1
                    
            except Exception as e:
                test_result["tests"].append({
                    "name": test["name"],
                    "description": test["description"],
                    "success": False,
                    "error": str(e)
                })
        
        test_result.update({
            "end_time": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(validation_tests),
                "passed_tests": passed_tests,
                "failed_tests": len(validation_tests) - passed_tests,
                "success_rate": passed_tests / len(validation_tests) if validation_tests else 0
            },
            "overall_status": "passed" if passed_tests == len(validation_tests) else "failed"
        })
        
        return test_result
    
    def validate_experience_storage(self) -> Dict[str, Any]:
        """Validate RL experience storage functionality"""
        validation_result = {
            "success": True,
            "checks": [],
            "errors": []
        }
        
        try:
            # Check if migration 004 was applied
            migration_check = self.run_command([
                "uv", "run", "python", "-c",
                "import sys; sys.path.insert(0, 'src'); from scripts.run_production_migrations import ProductionMigrationRunner; import asyncio; runner = ProductionMigrationRunner(); print('Migration check complete')"
            ], "Check RL migration status")
            
            validation_result["checks"].append({
                "name": "migration_004_check",
                "success": migration_check["success"],
                "details": "RL experience schema migration validation"
            })
            
            if not migration_check["success"]:
                validation_result["success"] = False
                validation_result["errors"].append("Migration 004 validation failed")
            
            # Check configuration files
            config_files = [
                "config/config.yaml",
                ".env.production",
                "src/utils/production_database.py"
            ]
            
            for config_file in config_files:
                file_path = self.project_root / config_file
                file_exists = file_path.exists()
                
                validation_result["checks"].append({
                    "name": f"config_file_{config_file.replace('/', '_')}",
                    "success": file_exists,
                    "details": f"Configuration file exists: {config_file}"
                })
                
                if not file_exists:
                    validation_result["success"] = False
                    validation_result["errors"].append(f"Missing configuration file: {config_file}")
            
        except Exception as e:
            validation_result["success"] = False
            validation_result["errors"].append(f"Experience storage validation error: {str(e)}")
        
        return validation_result
    
    def validate_database_connectivity(self) -> Dict[str, Any]:
        """Validate database connectivity"""
        validation_result = {
            "success": True,
            "checks": [],
            "errors": []
        }
        
        try:
            # Test production database connection
            db_test = self.run_command([
                "gcloud", "sql", "instances", "describe", "shyvr-rlte-db-prod",
                "--project", "shvyr-ai-bots",
                "--format", "value(state)"
            ], "Check production database status")
            
            db_running = db_test["success"] and "RUNNABLE" in db_test.get("stdout", "")
            
            validation_result["checks"].append({
                "name": "production_database_status",
                "success": db_running,
                "details": f"Database status: {db_test.get('stdout', 'unknown').strip()}"
            })
            
            if not db_running:
                validation_result["success"] = False
                validation_result["errors"].append("Production database is not running")
            
            # Check backup configuration
            backup_test = self.run_command([
                "gcloud", "sql", "instances", "describe", "shyvr-rlte-db-prod",
                "--project", "shvyr-ai-bots",
                "--format", "json"
            ], "Check backup configuration")
            
            if backup_test["success"]:
                try:
                    import json
                    instance_data = json.loads(backup_test["stdout"])
                    backup_config = instance_data.get("settings", {}).get("backupConfiguration", {})
                    backup_enabled = backup_config.get("enabled", False)
                    
                    validation_result["checks"].append({
                        "name": "backup_configuration",
                        "success": backup_enabled,
                        "details": f"Automated backups: {'enabled' if backup_enabled else 'disabled'}"
                    })
                    
                    if not backup_enabled:
                        validation_result["success"] = False
                        validation_result["errors"].append("Automated backups not configured")
                        
                except json.JSONDecodeError:
                    validation_result["checks"].append({
                        "name": "backup_configuration", 
                        "success": False,
                        "details": "Failed to parse backup configuration"
                    })
            
        except Exception as e:
            validation_result["success"] = False
            validation_result["errors"].append(f"Database connectivity validation error: {str(e)}")
        
        return validation_result
    
    def validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate dashboard API endpoints"""
        validation_result = {
            "success": True,
            "checks": [],
            "errors": []
        }
        
        try:
            # Check if dashboard service files exist
            dashboard_files = [
                "src/dashboard/api.py",
                "src/dashboard/service.py",
                "static/experience-dashboard.js",
                "static/index.html"
            ]
            
            for file_path in dashboard_files:
                file_exists = (self.project_root / file_path).exists()
                
                validation_result["checks"].append({
                    "name": f"dashboard_file_{file_path.replace('/', '_')}",
                    "success": file_exists,
                    "details": f"Dashboard file exists: {file_path}"
                })
                
                if not file_exists:
                    validation_result["success"] = False
                    validation_result["errors"].append(f"Missing dashboard file: {file_path}")
            
            # Check if API endpoints are implemented (simple file content check)
            api_file = self.project_root / "src/dashboard/api.py"
            if api_file.exists():
                api_content = api_file.read_text()
                expected_endpoints = [
                    "/api/v1/experiences/recent",
                    "/api/v1/experiences/stats", 
                    "/api/v1/experiences/performance"
                ]
                
                for endpoint in expected_endpoints:
                    endpoint_exists = endpoint in api_content
                    
                    validation_result["checks"].append({
                        "name": f"api_endpoint_{endpoint.replace('/', '_')}",
                        "success": endpoint_exists,
                        "details": f"API endpoint implemented: {endpoint}"
                    })
                    
                    if not endpoint_exists:
                        validation_result["success"] = False
                        validation_result["errors"].append(f"Missing API endpoint: {endpoint}")
            
        except Exception as e:
            validation_result["success"] = False
            validation_result["errors"].append(f"API endpoints validation error: {str(e)}")
        
        return validation_result
    
    def validate_monitoring_systems(self) -> Dict[str, Any]:
        """Validate monitoring and alerting systems"""
        validation_result = {
            "success": True,
            "checks": [],
            "errors": []
        }
        
        try:
            # Check monitoring configuration files
            monitoring_files = [
                "monitoring/grafana/rl_production_dashboard.json",
                "scripts/setup_basic_monitoring.sh",
                "scripts/manage_production_backups.py"
            ]
            
            for file_path in monitoring_files:
                file_exists = (self.project_root / file_path).exists()
                
                validation_result["checks"].append({
                    "name": f"monitoring_file_{file_path.replace('/', '_')}",
                    "success": file_exists,
                    "details": f"Monitoring file exists: {file_path}"
                })
                
                if not file_exists:
                    validation_result["success"] = False
                    validation_result["errors"].append(f"Missing monitoring file: {file_path}")
            
            # Check log-based metrics exist
            metrics_check = self.run_command([
                "gcloud", "logging", "metrics", "list",
                "--project", "shvyr-ai-bots",
                "--format", "value(name)"
            ], "Check log-based metrics")
            
            if metrics_check["success"]:
                metrics_output = metrics_check.get("stdout", "")
                expected_metrics = ["rl_database_errors", "rl_experience_storage_rate"]
                
                for metric in expected_metrics:
                    metric_exists = metric in metrics_output
                    
                    validation_result["checks"].append({
                        "name": f"log_metric_{metric}",
                        "success": metric_exists,
                        "details": f"Log-based metric exists: {metric}"
                    })
                    
                    if not metric_exists:
                        validation_result["success"] = False
                        validation_result["errors"].append(f"Missing log metric: {metric}")
            
        except Exception as e:
            validation_result["success"] = False
            validation_result["errors"].append(f"Monitoring systems validation error: {str(e)}")
        
        return validation_result
    
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("Starting full integration test suite...")
        
        suite_result = {
            "test_suite": "full_integration_test_suite",
            "start_time": self.start_time.isoformat(),
            "test_categories": []
        }
        
        # Define test sequence
        test_categories = [
            ("unit_tests", self.run_unit_tests),
            ("integration_tests", self.run_integration_tests),
            ("performance_tests", self.run_performance_tests),
            ("validation_tests", self.run_validation_tests)
        ]
        
        passed_categories = 0
        total_categories = len(test_categories)
        
        for category_name, test_function in test_categories:
            logger.info(f"Running {category_name}...")
            
            try:
                category_result = test_function()
                suite_result["test_categories"].append(category_result)
                
                if category_result.get("overall_status") == "passed":
                    passed_categories += 1
                    
            except Exception as e:
                logger.error(f"Test category {category_name} failed: {e}")
                suite_result["test_categories"].append({
                    "test_category": category_name,
                    "overall_status": "failed",
                    "error": str(e)
                })
        
        # Generate suite summary
        end_time = datetime.now()
        suite_result.update({
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - self.start_time).total_seconds(),
            "summary": {
                "total_categories": total_categories,
                "passed_categories": passed_categories,
                "failed_categories": total_categories - passed_categories,
                "success_rate": passed_categories / total_categories if total_categories > 0 else 0,
                "overall_status": "passed" if passed_categories == total_categories else "failed"
            }
        })
        
        return suite_result
    
    def save_test_report(self, results: Dict[str, Any]) -> str:
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integration_test_suite_{timestamp}.json"
        
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test report saved: {report_path}")
        return str(report_path)


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Full Integration Test Suite")
    parser.add_argument("--category", choices=[
        "unit", "integration", "performance", "validation", "all"
    ], default="all", help="Test category to run")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    test_suite = IntegrationTestSuite()
    
    try:
        if args.category == "all":
            results = await test_suite.run_full_test_suite()
            
            # Display results
            print(f"\n🧪 Integration Test Suite Results")
            print(f"=" * 50)
            print(f"Overall Status: {results['summary']['overall_status'].upper()}")
            print(f"Duration: {results['duration_seconds']:.1f} seconds")
            print(f"Categories: {results['summary']['passed_categories']}/{results['summary']['total_categories']} passed")
            print(f"Success Rate: {results['summary']['success_rate']:.1%}")
            
            print(f"\n📋 Category Results:")
            for category in results["test_categories"]:
                status_emoji = "✅" if category.get("overall_status") == "passed" else "❌"
                category_name = category.get("test_category", "unknown")
                
                print(f"  {status_emoji} {category_name}: {category.get('overall_status', 'unknown')}")
                
                if "summary" in category:
                    summary = category["summary"]
                    if "success_rate" in summary:
                        print(f"    Success Rate: {summary['success_rate']:.1%}")
            
            # Save report
            report_path = test_suite.save_test_report(results)
            print(f"\n📄 Detailed report: {report_path}")
            
            # Exit with appropriate code
            sys.exit(0 if results["summary"]["overall_status"] == "passed" else 1)
            
        else:
            # Run individual category
            category_map = {
                "unit": test_suite.run_unit_tests,
                "integration": test_suite.run_integration_tests,
                "performance": test_suite.run_performance_tests,
                "validation": test_suite.run_validation_tests
            }
            
            if args.category in category_map:
                result = category_map[args.category]()
                print(f"\n🧪 {args.category.title()} Test Results:")
                print(f"Status: {result.get('overall_status', 'unknown')}")
                print(json.dumps(result, indent=2, default=str))
                
                sys.exit(0 if result.get("overall_status") == "passed" else 1)
    
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())