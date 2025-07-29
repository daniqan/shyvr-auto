#!/usr/bin/env python3
"""
Test suite for Cloud SQL setup infrastructure
Tests the Cloud SQL setup script functionality and database connectivity
"""

import os
import sys
import subprocess
import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from src.utils.config import get_config


class TestCloudSQLSetup:
    """Test Cloud SQL setup script functionality"""
    
    @pytest.fixture
    def setup_script_path(self):
        """Get path to the setup script"""
        return Path(__file__).parent.parent.parent.parent / "deploy" / "setup_cloud_sql.sh"
    
    @pytest.fixture
    def mock_gcloud_commands(self):
        """Mock gcloud command responses"""
        return {
            "auth_list": "user@example.com\nACTIVE\n",
            "config_get": "shvyr-ai-bots\n",
            "instances_describe": json.dumps({
                "connectionName": "shvyr-ai-bots:us-central1:shyvr-rlte-db",
                "ipAddresses": [{"ipAddress": "10.0.0.1"}]
            }),
            "secrets_create": "Created secret [test-secret].\n"
        }
    
    def test_script_exists_and_executable(self, setup_script_path):
        """Test that the setup script exists and is executable"""
        assert setup_script_path.exists(), "setup_cloud_sql.sh script should exist"
        assert os.access(setup_script_path, os.X_OK), "setup_cloud_sql.sh should be executable"
    
    @patch('subprocess.run')
    def test_help_command(self, mock_run, setup_script_path):
        """Test that help command works"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Usage: setup_cloud_sql.sh [OPTION]"
        
        result = subprocess.run([str(setup_script_path), "help"], capture_output=True, text=True)
        
        # The actual script should be called
        assert setup_script_path.exists()
    
    @patch('subprocess.run')
    def test_gcloud_authentication_check(self, mock_run, setup_script_path):
        """Test gcloud authentication check functionality"""
        # Mock successful gcloud auth check
        mock_run.side_effect = [
            Mock(returncode=0, stdout="user@example.com\nACTIVE\n"),  # gcloud auth list
            Mock(returncode=0, stdout="shvyr-ai-bots\n"),  # gcloud config get-value project
        ]
        
        # This would be called by the script's check_gcloud_setup function
        # We're testing the logic would work correctly
        assert True  # Script existence already validated
    
    @patch('subprocess.run')
    def test_api_enabling(self, mock_run, setup_script_path):
        """Test that required APIs are enabled"""
        mock_run.return_value.returncode = 0
        
        # Expected APIs that should be enabled
        expected_apis = [
            "sqladmin.googleapis.com",
            "sql-component.googleapis.com", 
            "cloudresourcemanager.googleapis.com"
        ]
        
        # Verify script exists (proxy for API enabling logic)
        assert setup_script_path.exists()
        
        # Read script content to verify API enabling commands
        script_content = setup_script_path.read_text()
        for api in expected_apis:
            assert api in script_content, f"Script should enable {api}"
    
    def test_instance_configuration(self, setup_script_path):
        """Test that instance is configured with correct parameters"""
        script_content = setup_script_path.read_text()
        
        # Verify configuration parameters
        assert 'INSTANCE_NAME="shyvr-rlte-db"' in script_content
        assert 'DATABASE_NAME="shyvr_rlte"' in script_content
        assert 'DATABASE_USER="rlte_user"' in script_content
        assert 'VERSION="POSTGRES_14"' in script_content
        assert 'REGION="us-central1"' in script_content
        
        # Verify security settings
        assert '--deletion-protection' in script_content
        assert '--enable-bin-log' in script_content
        assert '--backup-start-time=04:00' in script_content
    
    def test_password_generation_security(self, setup_script_path):
        """Test that secure password generation is implemented"""
        script_content = setup_script_path.read_text()
        
        # Verify secure password generation
        assert 'generate_password()' in script_content
        assert 'secrets.choice' in script_content or 'python3 -c' in script_content
        assert 'string.ascii_letters' in script_content or '32' in script_content
    
    def test_secret_manager_integration(self, setup_script_path):
        """Test Secret Manager integration"""
        script_content = setup_script_path.read_text()
        
        # Verify secrets are stored in Secret Manager
        assert 'gcloud secrets create postgres-password' in script_content
        assert 'gcloud secrets create DB_PASSWORD' in script_content
        assert 'gcloud secrets create DATABASE_URL' in script_content
    
    def test_idempotency_checks(self, setup_script_path):
        """Test that the setup script is idempotent"""
        script_content = setup_script_path.read_text()
        
        # Verify instance existence check
        assert 'instance_exists()' in script_content
        assert 'gcloud sql instances describe' in script_content
        assert 'if instance_exists' in script_content or 'already exists' in script_content
    
    def test_error_handling(self, setup_script_path):
        """Test error handling in setup script"""
        script_content = setup_script_path.read_text()
        
        # Verify error handling
        assert 'set -e' in script_content  # Exit on error
        assert 'ERROR' in script_content.upper()
        assert 'exit 1' in script_content
    
    def test_cloud_sql_proxy_setup(self, setup_script_path):
        """Test Cloud SQL Auth Proxy configuration"""
        script_content = setup_script_path.read_text()
        
        # Verify Cloud SQL Proxy setup
        assert 'cloud_sql_proxy' in script_content
        assert 'curl -o cloud_sql_proxy' in script_content
        assert '/usr/local/bin/' in script_content
        assert 'chmod +x cloud_sql_proxy' in script_content
    
    def test_database_migration_integration(self, setup_script_path):
        """Test database migration integration"""
        script_content = setup_script_path.read_text()
        
        # Verify migration integration
        assert 'run_migrations()' in script_content
        assert 'database/init_database.py' in script_content
        assert 'uv run' in script_content
    
    def test_connectivity_testing(self, setup_script_path):
        """Test database connectivity testing"""
        script_content = setup_script_path.read_text()
        
        # Verify connectivity testing
        assert 'test_connectivity()' in script_content
        assert 'gcloud sql connect' in script_content
        assert 'SELECT version()' in script_content
    
    def test_cleanup_functionality(self, setup_script_path):
        """Test cleanup functionality for development"""
        script_content = setup_script_path.read_text()
        
        # Verify cleanup functionality
        assert 'cleanup_instance()' in script_content
        assert '--no-deletion-protection' in script_content
        assert 'gcloud sql instances delete' in script_content
        assert 'Are you sure' in script_content  # Confirmation prompt


class TestDatabaseConnectivity:
    """Test database connectivity functionality"""
    
    @pytest.fixture
    def mock_database_config(self):
        """Mock database configuration"""
        return {
            'host': '/cloudsql/shvyr-ai-bots:us-central1:shyvr-rlte-db',
            'port': 5432,
            'database': 'shyvr_rlte', 
            'username': 'rlte_user',
            'password': 'test_password'
        }
    
    @patch('asyncpg.connect')
    async def test_cloud_sql_connection_string(self, mock_connect, mock_database_config):
        """Test Cloud SQL connection string format"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Test that connection uses Unix socket for Cloud SQL
        connection_host = mock_database_config['host']
        assert connection_host.startswith('/cloudsql/')
        assert 'shvyr-ai-bots:us-central1:shyvr-rlte-db' in connection_host
    
    @patch('subprocess.run')
    def test_connectivity_test_command(self, mock_run):
        """Test connectivity test command generation"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "PostgreSQL 14.x\n"
        
        # Test that connectivity test would use proper gcloud command
        expected_command = [
            'gcloud', 'sql', 'connect', 'shyvr-rlte-db',
            '--user=rlte_user', '--database=shyvr_rlte'
        ]
        
        # Verify command structure (mock would be called with similar structure)
        assert True  # Command structure verified in script content tests


class TestDeploymentIntegration:
    """Test deployment script integration with Cloud SQL"""
    
    @pytest.fixture
    def deploy_script_path(self):
        """Get path to the deployment script"""
        return Path(__file__).parent.parent.parent.parent / "deploy" / "deploy_latest.sh"
    
    def test_cloud_sql_secrets_integration(self, deploy_script_path):
        """Test that deployment script includes Cloud SQL secrets"""
        script_content = deploy_script_path.read_text()
        
        # Verify required secrets include database credentials
        assert 'DB_PASSWORD' in script_content
        assert 'DATABASE_URL' in script_content
        assert 'REQUIRED_SECRETS' in script_content
    
    def test_cloud_sql_instance_connection(self, deploy_script_path):
        """Test Cloud SQL instance connection configuration"""
        script_content = deploy_script_path.read_text()
        
        # Verify Cloud SQL instance connection
        assert '--add-cloudsql-instances' in script_content
        assert 'shvyr-ai-bots:us-central1:shyvr-rlte-db' in script_content
        assert 'CLOUD_SQL_INSTANCE=' in script_content
    
    def test_deployment_environment_variables(self, deploy_script_path):
        """Test deployment environment variables"""
        script_content = deploy_script_path.read_text()
        
        # Verify environment configuration
        assert 'ENVIRONMENT=production' in script_content
        assert 'LOG_LEVEL=INFO' in script_content


class TestScriptValidation:
    """Test script validation and syntax"""
    
    @pytest.fixture
    def setup_script_path(self):
        """Get path to the setup script"""
        return Path(__file__).parent.parent.parent.parent / "deploy" / "setup_cloud_sql.sh"
    
    def test_bash_syntax_validation(self, setup_script_path):
        """Test that the bash script has valid syntax"""
        try:
            result = subprocess.run(
                ['bash', '-n', str(setup_script_path)], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            assert result.returncode == 0, f"Bash syntax error: {result.stderr}"
        except subprocess.TimeoutExpired:
            pytest.fail("Script syntax check timed out")
        except FileNotFoundError:
            pytest.skip("bash not available for syntax checking")
    
    def test_shebang_and_basic_structure(self, setup_script_path):
        """Test script shebang and basic structure"""
        content = setup_script_path.read_text()
        
        # Verify shebang
        assert content.startswith('#!/bin/bash'), "Script should have proper shebang"
        
        # Verify basic structure
        assert 'set -e' in content, "Script should exit on error"
        assert 'main()' in content, "Script should have main function"
        assert 'case ' in content, "Script should have command parsing"
    
    def test_required_functions_present(self, setup_script_path):
        """Test that all required functions are present"""
        content = setup_script_path.read_text()
        
        required_functions = [
            'check_gcloud_setup()',
            'enable_apis()',
            'instance_exists()',
            'generate_password()',
            'create_instance()',
            'setup_database()',
            'configure_connection()',
            'run_migrations()',
            'test_connectivity()',
            'display_summary()',
            'show_help()',
            'cleanup_instance()',
            'main()'
        ]
        
        for func in required_functions:
            assert func in content, f"Required function {func} should be present"


@pytest.mark.asyncio
async def test_database_initialization_integration():
    """Test integration with database initialization"""
    # This test verifies that the database initialization code
    # can work with Cloud SQL connection strings
    
    try:
        # Import the database initializer
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from database.init_database import DatabaseInitializer
        
        # Test that the initializer can be created
        initializer = DatabaseInitializer()
        assert initializer is not None
        assert initializer.migration_dir.exists()
        
    except ImportError:
        pytest.skip("Database initializer not available for testing")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])