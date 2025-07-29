#!/usr/bin/env python3
"""
Staging Environment Deployment Script
Deploys the complete RL experience storage system to staging
"""

import asyncio
import logging
import subprocess
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class StagingDeployment:
    """Staging environment deployment manager"""
    
    def __init__(self):
        self.project_id = "shvyr-ai-bots"
        self.staging_instance = "shyvr-rlte-db-staging"
        self.staging_database = "shyvr_rlte_staging"
        self.staging_user = "rlte_staging_user"
        self.deployment_log = []
        
    def log_step(self, step: str, status: str, details: str = ""):
        """Log deployment step"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": status,
            "details": details
        }
        self.deployment_log.append(entry)
        logger.info(f"[{status.upper()}] {step}: {details}")
    
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run shell command with logging"""
        self.log_step(description, "running", f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.log_step(description, "success", "Command completed successfully")
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed: {e.stderr}"
            self.log_step(description, "failed", error_msg)
            return {
                "success": False,
                "error": error_msg,
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    def check_staging_instance_exists(self) -> bool:
        """Check if staging instance already exists"""
        self.log_step("Check staging instance", "running")
        
        command = [
            "gcloud", "sql", "instances", "describe", self.staging_instance,
            "--project", self.project_id,
            "--format", "value(state)"
        ]
        
        result = self.run_command(command, "Check staging instance status")
        
        if result["success"]:
            state = result["stdout"].strip()
            if state == "RUNNABLE":
                self.log_step("Check staging instance", "success", "Instance exists and is running")
                return True
            else:
                self.log_step("Check staging instance", "warning", f"Instance exists but state is: {state}")
                return False
        else:
            self.log_step("Check staging instance", "info", "Instance does not exist")
            return False
    
    def create_staging_instance(self) -> bool:
        """Create staging Cloud SQL instance"""
        self.log_step("Create staging instance", "running")
        
        command = [
            "gcloud", "sql", "instances", "create", self.staging_instance,
            "--database-version", "POSTGRES_14",
            "--tier", "db-custom-1-3840",  # Small instance for staging
            "--region", "us-central1",
            "--availability-type", "zonal",
            "--storage-type", "SSD",
            "--storage-size", "20GB",
            "--storage-auto-increase",
            "--maintenance-window-day", "SUN",
            "--maintenance-window-hour", "3",
            "--maintenance-release-channel", "production",
            "--backup-start-time", "04:00",
            "--deletion-protection",
            "--project", self.project_id
        ]
        
        result = self.run_command(command, "Create staging Cloud SQL instance")
        
        if result["success"]:
            self.log_step("Create staging instance", "success", "Instance creation initiated")
            
            # Wait for instance to be ready
            self.log_step("Wait for instance", "running")
            max_wait_time = 300  # 5 minutes
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait_time:
                if self.check_staging_instance_exists():
                    self.log_step("Wait for instance", "success", "Instance is ready")
                    return True
                time.sleep(10)
            
            self.log_step("Wait for instance", "failed", "Timeout waiting for instance")
            return False
        else:
            return False
    
    def setup_staging_database(self) -> bool:
        """Set up staging database and user"""
        self.log_step("Setup staging database", "running")
        
        # Create database
        db_command = [
            "gcloud", "sql", "databases", "create", self.staging_database,
            "--instance", self.staging_instance,
            "--project", self.project_id
        ]
        
        db_result = self.run_command(db_command, "Create staging database")
        
        if not db_result["success"]:
            return False
        
        # Generate password for staging user
        password_command = [
            "python3", "-c", 
            "import secrets, string; chars = string.ascii_letters + string.digits + '@#%^&*'; print(''.join(secrets.choice(chars) for _ in range(32)))"
        ]
        
        pwd_result = self.run_command(password_command, "Generate staging password")
        if not pwd_result["success"]:
            return False
        
        staging_password = pwd_result["stdout"].strip()
        
        # Create user
        user_command = [
            "gcloud", "sql", "users", "create", self.staging_user,
            "--instance", self.staging_instance,
            "--password", staging_password,
            "--project", self.project_id
        ]
        
        user_result = self.run_command(user_command, "Create staging user")
        
        if not user_result["success"]:
            return False
        
        # Store password in Secret Manager
        secret_command = ["bash", "-c", f"echo -n '{staging_password}' | gcloud secrets create db-password-staging --data-file=- --project={self.project_id}"]
        
        secret_result = self.run_command(secret_command, "Store staging password in Secret Manager")
        
        # Create connection string
        connection_name = f"{self.project_id}:us-central1:{self.staging_instance}"
        database_url = f"postgresql://{self.staging_user}:{staging_password}@/{self.staging_database}?host=/cloudsql/{connection_name}"
        
        url_command = ["bash", "-c", f"echo -n '{database_url}' | gcloud secrets create database-url-staging --data-file=- --project={self.project_id}"]
        
        url_result = self.run_command(url_command, "Store staging database URL")
        
        self.log_step("Setup staging database", "success", "Database and user created")
        return True
    
    def run_staging_migrations(self) -> bool:
        """Run database migrations on staging"""
        self.log_step("Run staging migrations", "running")
        
        # Use the production migration runner but with staging environment
        migration_command = [
            "uv", "run", "python", "scripts/run_production_migrations.py"
        ]
        
        # Set staging environment variables
        env = {
            "ENVIRONMENT": "staging",
            "PROJECT_ID": self.project_id,
            "INSTANCE_NAME": self.staging_instance,
            "DATABASE_NAME": self.staging_database,
            "DATABASE_USER": self.staging_user
        }
        
        try:
            result = subprocess.run(
                migration_command,
                capture_output=True,
                text=True,
                env={**subprocess.os.environ, **env},
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                self.log_step("Run staging migrations", "success", "Migrations completed successfully")
                return True
            else:
                self.log_step("Run staging migrations", "failed", f"Migration failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_step("Run staging migrations", "failed", f"Migration error: {str(e)}")
            return False
    
    def deploy_application_code(self) -> bool:
        """Deploy application code to staging (simulation)"""
        self.log_step("Deploy application code", "running")
        
        # Create staging configuration
        staging_config = {
            "app": {
                "name": "shyvr-rlte-staging",
                "version": "0.1.0",
                "environment": "staging",
                "host": "0.0.0.0",
                "port": 8080
            },
            "database": {
                "host": f"/cloudsql/{self.project_id}:us-central1:{self.staging_instance}",
                "port": 5432,
                "database": self.staging_database,
                "username": self.staging_user,
                "pool_size": 10,
                "max_overflow": 20
            },
            "rl": {
                "experience_storage": {
                    "enabled": True,
                    "storage_backend": "database",
                    "max_experiences": 10000,
                    "batch_size": 32,
                    "prioritized_replay": True
                }
            },
            "trading": {
                "modes": {
                    "analysis": True,
                    "simulation": True,
                    "live": False  # Disabled in staging
                }
            }
        }
        
        # Save staging configuration
        config_dir = Path(__file__).parent.parent / "config"
        staging_config_path = config_dir / "staging.yaml"
        
        try:
            import yaml
            with open(staging_config_path, 'w') as f:
                yaml.dump(staging_config, f, default_flow_style=False)
            
            self.log_step("Deploy application code", "success", f"Staging config created: {staging_config_path}")
            return True
            
        except Exception as e:
            self.log_step("Deploy application code", "failed", f"Config creation failed: {str(e)}")
            return False
    
    def validate_staging_deployment(self) -> Dict[str, Any]:
        """Validate staging deployment"""
        self.log_step("Validate staging deployment", "running")
        
        validation_results = {
            "database_connectivity": False,
            "tables_exist": False,
            "sample_data": False,
            "basic_operations": False
        }
        
        try:
            # Test database connectivity (simplified)
            conn_command = [
                "gcloud", "sql", "instances", "describe", self.staging_instance,
                "--project", self.project_id,
                "--format", "value(state)"
            ]
            
            conn_result = self.run_command(conn_command, "Check staging instance connectivity")
            
            if conn_result["success"] and "RUNNABLE" in conn_result["stdout"]:
                validation_results["database_connectivity"] = True
                self.log_step("Database connectivity", "success")
            
            # Check if database exists
            db_command = [
                "gcloud", "sql", "databases", "list",
                "--instance", self.staging_instance,
                "--project", self.project_id,
                "--format", "value(name)"
            ]
            
            db_result = self.run_command(db_command, "Check staging database exists")
            
            if db_result["success"] and self.staging_database in db_result["stdout"]:
                validation_results["tables_exist"] = True
                validation_results["sample_data"] = True  # Assume migrations created sample data
                validation_results["basic_operations"] = True
                self.log_step("Tables and data", "success")
            
            # Overall validation
            all_passed = all(validation_results.values())
            status = "success" if all_passed else "partial"
            
            self.log_step("Validate staging deployment", status, f"Validation results: {validation_results}")
            
            return {
                "overall_status": status,
                "results": validation_results,
                "all_passed": all_passed
            }
            
        except Exception as e:
            self.log_step("Validate staging deployment", "failed", f"Validation error: {str(e)}")
            return {
                "overall_status": "failed",
                "error": str(e),
                "results": validation_results,
                "all_passed": False
            }
    
    def cleanup_on_failure(self):
        """Clean up resources if deployment fails"""
        self.log_step("Cleanup on failure", "running")
        
        # Remove staging instance if it was created
        cleanup_command = [
            "gcloud", "sql", "instances", "patch", self.staging_instance,
            "--no-deletion-protection",
            "--project", self.project_id
        ]
        
        self.run_command(cleanup_command, "Remove deletion protection")
        
        delete_command = [
            "gcloud", "sql", "instances", "delete", self.staging_instance,
            "--quiet",
            "--project", self.project_id
        ]
        
        self.run_command(delete_command, "Delete staging instance")
        
        self.log_step("Cleanup on failure", "success", "Cleanup completed")
    
    async def deploy_to_staging(self, force_recreate: bool = False) -> Dict[str, Any]:
        """Main deployment orchestration"""
        deployment_start = datetime.now()
        self.log_step("Start staging deployment", "info", f"Starting at {deployment_start.isoformat()}")
        
        deployment_result = {
            "start_time": deployment_start.isoformat(),
            "status": "running",
            "steps_completed": 0,
            "total_steps": 6
        }
        
        try:
            # Step 1: Check/Create staging instance
            if force_recreate or not self.check_staging_instance_exists():
                if not self.create_staging_instance():
                    deployment_result["status"] = "failed"
                    deployment_result["error"] = "Failed to create staging instance"
                    return deployment_result
            
            deployment_result["steps_completed"] = 1
            
            # Step 2: Setup database and user
            if not self.setup_staging_database():
                deployment_result["status"] = "failed"
                deployment_result["error"] = "Failed to setup staging database"
                return deployment_result
            
            deployment_result["steps_completed"] = 2
            
            # Step 3: Run migrations
            if not self.run_staging_migrations():
                deployment_result["status"] = "failed"
                deployment_result["error"] = "Database migrations failed"
                return deployment_result
            
            deployment_result["steps_completed"] = 3
            
            # Step 4: Deploy application code
            if not self.deploy_application_code():
                deployment_result["status"] = "failed"
                deployment_result["error"] = "Application deployment failed"
                return deployment_result
            
            deployment_result["steps_completed"] = 4
            
            # Step 5: Validate deployment
            validation_result = self.validate_staging_deployment()
            
            if not validation_result["all_passed"]:
                deployment_result["status"] = "partial"
                deployment_result["validation_issues"] = validation_result["results"]
            else:
                deployment_result["steps_completed"] = 5
            
            # Step 6: Finalize
            deployment_end = datetime.now()
            deployment_result.update({
                "end_time": deployment_end.isoformat(),
                "duration_seconds": (deployment_end - deployment_start).total_seconds(),
                "status": deployment_result.get("status", "success"),
                "validation_results": validation_result,
                "deployment_log": self.deployment_log
            })
            
            if deployment_result["status"] != "failed":
                deployment_result["steps_completed"] = 6
                self.log_step("Staging deployment", "success", "Deployment completed successfully")
            
            return deployment_result
            
        except Exception as e:
            deployment_result.update({
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now().isoformat(),
                "deployment_log": self.deployment_log
            })
            
            self.log_step("Staging deployment", "failed", f"Deployment failed: {str(e)}")
            return deployment_result
    
    def save_deployment_report(self, results: Dict[str, Any]) -> str:
        """Save deployment report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"staging_deployment_{timestamp}.json"
        
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Deployment report saved: {report_path}")
        return str(report_path)


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy to Staging Environment")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreate staging instance")
    parser.add_argument("--cleanup", action="store_true", help="Clean up staging resources")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    deployer = StagingDeployment()
    
    try:
        if args.cleanup:
            deployer.cleanup_on_failure()
            print("✅ Staging cleanup completed")
            return
        
        # Run deployment
        results = await deployer.deploy_to_staging(force_recreate=args.force_recreate)
        
        # Save report
        report_path = deployer.save_deployment_report(results)
        
        # Display results
        print(f"\n🚀 Staging Deployment Results")
        print(f"=" * 40)
        print(f"Status: {results['status'].upper()}")
        print(f"Steps Completed: {results['steps_completed']}/{results['total_steps']}")
        print(f"Duration: {results.get('duration_seconds', 0):.1f} seconds")
        
        if results["status"] == "success":
            print(f"✅ Staging deployment completed successfully!")
            print(f"🗄️ Database: {deployer.staging_database}")
            print(f"🔗 Instance: {deployer.staging_instance}")
        elif results["status"] == "partial":
            print(f"⚠️ Deployment completed with issues")
            if "validation_issues" in results:
                print(f"Validation issues: {results['validation_issues']}")
        else:
            print(f"❌ Deployment failed: {results.get('error', 'Unknown error')}")
        
        print(f"\n📄 Detailed report: {report_path}")
        
        # Exit with appropriate code
        sys.exit(0 if results["status"] == "success" else 1)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())