#!/usr/bin/env python3
"""
Unit Tests for RL Experience Database Deployment Automation

Tests the deployment automation script functionality including:
- Environment-specific configuration validation
- Integration with existing infrastructure
- Script argument parsing and validation
- Dry-run functionality

This test suite validates Phase 6.2 deployment automation implementation.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json


class TestDeploymentAutomation(unittest.TestCase):
    """Test suite for deployment automation script"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.deploy_script = self.project_root / "deploy" / "setup_experience_database.sh"
        self.assertTrue(self.deploy_script.exists(), "Deployment script not found")
        
        # Test configurations for each environment
        self.environment_configs = {
            'development': {
                'instance_name': 'shyvr-rlte-db-dev',
                'database_name': 'shyvr_rlte_dev',
                'user_name': 'rlte_dev_user',
                'tier': 'db-f1-micro',
                'storage': '10GB'
            },
            'staging': {
                'instance_name': 'shyvr-rlte-db-staging',
                'database_name': 'shyvr_rlte_staging',
                'user_name': 'rlte_staging_user',
                'tier': 'db-g1-small',
                'storage': '20GB'
            },
            'production': {
                'instance_name': 'shyvr-rlte-db-prod',
                'database_name': 'shyvr_rlte_prod',
                'user_name': 'rlte_prod_user',
                'tier': 'db-n1-standard-1',
                'storage': '100GB'
            }
        }
    
    def test_script_exists_and_executable(self):
        """Test that deployment script exists and is executable"""
        self.assertTrue(self.deploy_script.exists())
        self.assertTrue(os.access(self.deploy_script, os.X_OK))
    
    def test_help_functionality(self):
        """Test that script displays help correctly"""
        result = subprocess.run(
            [str(self.deploy_script), "--help"],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("RL Experience Database Deployment Automation", result.stdout)
        self.assertIn("--environment", result.stdout)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--validate-only", result.stdout)
    
    def test_environment_specific_configurations(self):
        """Test that each environment uses correct configuration"""
        for env_name, expected_config in self.environment_configs.items():
            with self.subTest(environment=env_name):
                result = subprocess.run(
                    [str(self.deploy_script), "--environment", env_name, "--dry-run"],
                    capture_output=True,
                    text=True
                )
                
                # Script should complete successfully in dry-run mode
                # (It may return 1 due to health check failures in dry-run, which is expected)
                self.assertIn("RL Experience Database Deployment Automation", result.stderr)
                self.assertIn(f"Environment: {env_name}", result.stderr)
                self.assertIn(expected_config['instance_name'], result.stderr)
                self.assertIn(expected_config['database_name'], result.stderr)
                self.assertIn(expected_config['user_name'], result.stderr)
                self.assertIn(expected_config['tier'], result.stderr)
                self.assertIn(expected_config['storage'], result.stderr)
    
    def test_dry_run_functionality(self):
        """Test that dry-run mode works correctly"""
        result = subprocess.run(
            [str(self.deploy_script), "--environment", "development", "--dry-run", "--verbose"],
            capture_output=True,
            text=True
        )
        
        # Should show what would be done without executing
        self.assertIn("[DRY RUN]", result.stderr)
        self.assertIn("Would create Cloud SQL instance", result.stderr)
        self.assertIn("Would enable API", result.stderr)
        self.assertIn("Would run database schema migrations", result.stderr)
    
    def test_validate_only_functionality(self):
        """Test validation-only mode"""
        result = subprocess.run(
            [str(self.deploy_script), "--environment", "development", "--validate-only"],
            capture_output=True,
            text=True
        )
        
        # Should attempt validation and report non-existent resources
        self.assertIn("Validating existing RL experience database deployment", result.stderr)
        # Expected to fail since we don't have actual deployments in test
        self.assertNotEqual(result.returncode, 0)
    
    def test_invalid_environment_handling(self):
        """Test handling of invalid environment names"""
        result = subprocess.run(
            [str(self.deploy_script), "--environment", "invalid"],
            capture_output=True,
            text=True
        )
        
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid environment", result.stderr)
    
    def test_custom_configuration_override(self):
        """Test that custom configuration values override defaults"""
        custom_instance = "custom-test-instance"
        custom_tier = "db-custom-1"
        
        result = subprocess.run([
            str(self.deploy_script),
            "--environment", "development",
            "--instance-name", custom_instance,
            "--tier", custom_tier,
            "--dry-run"
        ], capture_output=True, text=True)
        
        self.assertIn(custom_instance, result.stderr)
        self.assertIn(custom_tier, result.stderr)
    
    def test_required_files_exist(self):
        """Test that all required files for deployment exist"""
        required_files = [
            "database/migrations/004_create_rl_experience_schema.sql",
            "database/init_database.py",
            "deploy/setup_monitoring.sh",
            "scripts/manage_experience_lifecycle.py"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            self.assertTrue(
                full_path.exists(), 
                f"Required file {file_path} not found"
            )
    
    def test_integration_with_existing_scripts(self):
        """Test integration with existing deployment infrastructure"""
        # Check if the script references existing deployment scripts
        with open(self.deploy_script, 'r') as f:
            script_content = f.read()
        
        # Should integrate with existing monitoring setup
        self.assertIn("setup_monitoring.sh", script_content)
        
        # Should use existing database initialization
        self.assertIn("database/init_database.py", script_content)
        
        # Should integrate with lifecycle management
        self.assertIn("manage_experience_lifecycle.py", script_content)
    
    def test_environment_variable_support(self):
        """Test that script respects environment variables"""
        test_project_id = "test-project-123"
        test_env = os.environ.copy()
        test_env["PROJECT_ID"] = test_project_id
        
        result = subprocess.run(
            [str(self.deploy_script), "--environment", "development", "--dry-run"],
            capture_output=True,
            text=True,
            env=test_env
        )
        
        self.assertIn(test_project_id, result.stderr)
    
    def test_secret_naming_convention(self):
        """Test that secrets follow correct naming convention"""
        result = subprocess.run(
            [str(self.deploy_script), "--environment", "staging", "--dry-run"],
            capture_output=True,
            text=True
        )
        
        # Should mention environment-specific secret names
        self.assertIn("postgres-password-staging", result.stderr)
        self.assertIn("db-password-staging", result.stderr)
        self.assertIn("database-url-staging", result.stderr)
    
    def test_comprehensive_feature_coverage(self):
        """Test that all required features are mentioned in output"""
        result = subprocess.run(
            [str(self.deploy_script), "--environment", "production", "--dry-run"],
            capture_output=True,
            text=True
        )
        
        required_features = [
            "RL experiences table with optimized indexes",
            "Training sessions tracking", 
            "Performance metrics collection",
            "Automated experience lifecycle management",
            "Environment-specific configuration"
        ]
        
        for feature in required_features:
            self.assertIn(feature, result.stderr, f"Feature '{feature}' not mentioned in output")
    
    def test_monitoring_integration_options(self):
        """Test monitoring setup integration options"""
        # Test skip monitoring option
        result = subprocess.run(
            [str(self.deploy_script), "--environment", "development", "--skip-monitoring", "--dry-run"],
            capture_output=True,
            text=True
        )
        
        self.assertIn("Skipping monitoring setup", result.stderr)
    
    def test_migration_integration_options(self):
        """Test database migration integration options"""
        # Test skip migration option
        result = subprocess.run(
            [str(self.deploy_script), "--environment", "development", "--skip-migration", "--dry-run"],
            capture_output=True,
            text=True
        )
        
        self.assertIn("Skipping database migrations", result.stderr)


class TestDeploymentScriptIntegration(unittest.TestCase):
    """Integration tests for deployment script with existing infrastructure"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.deploy_script = self.project_root / "deploy" / "setup_experience_database.sh"
    
    def test_config_yaml_integration(self):
        """Test integration with main configuration file"""
        config_file = self.project_root / "config" / "config.yaml"
        self.assertTrue(config_file.exists(), "Main config file not found")
        
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        # Should have RL experience storage configuration
        self.assertIn("experience_storage", config_content)
        self.assertIn("database", config_content)
    
    def test_migration_schema_compatibility(self):
        """Test that deployment script uses correct migration schema"""
        migration_file = self.project_root / "database" / "migrations" / "004_create_rl_experience_schema.sql"
        self.assertTrue(migration_file.exists(), "RL experience migration not found")
        
        with open(migration_file, 'r') as f:
            migration_content = f.read()
        
        # Should contain required RL experience tables
        required_tables = [
            "rl_experiences",
            "rl_training_sessions", 
            "rl_performance_metrics"
        ]
        
        for table in required_tables:
            self.assertIn(table, migration_content)
    
    def test_lifecycle_management_integration(self):
        """Test integration with experience lifecycle management"""
        lifecycle_script = self.project_root / "scripts" / "manage_experience_lifecycle.py"
        self.assertTrue(lifecycle_script.exists(), "Lifecycle management script not found")
        
        # Script should be executable
        self.assertTrue(os.access(lifecycle_script, os.X_OK))


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PROJECT_ID', 'test-project')
    os.environ.setdefault('ENVIRONMENT', 'development')
    
    # Run tests
    unittest.main(verbosity=2)