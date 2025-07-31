"""
Unit tests for GCS setup scripts
Tests the validation logic and script functionality without requiring actual GCS access
"""

import json
import os
import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest

# Import would normally be from deploy.validate_gcs_setup, but we'll test the script directly


class TestLifecyclePolicyConfiguration:
    """Test lifecycle policy JSON configuration"""
    
    def test_lifecycle_policy_structure(self):
        """Test that lifecycle.json has correct structure"""
        lifecycle_path = Path(__file__).parent.parent.parent.parent / "deploy" / "lifecycle.json"
        
        assert lifecycle_path.exists(), "lifecycle.json file should exist"
        
        with open(lifecycle_path, 'r') as f:
            policy = json.load(f)
        
        assert "lifecycle" in policy
        assert "rule" in policy["lifecycle"]
        
        rules = policy["lifecycle"]["rule"]
        assert isinstance(rules, list)
        assert len(rules) > 0
        
    def test_lifecycle_policy_rules(self):
        """Test specific lifecycle policy rules"""
        lifecycle_path = Path(__file__).parent.parent.parent.parent / "deploy" / "lifecycle.json"
        
        with open(lifecycle_path, 'r') as f:
            policy = json.load(f)
        
        rules = policy["lifecycle"]["rule"]
        
        # Check for required rule types
        action_types = [rule["action"]["type"] for rule in rules]
        
        assert "SetStorageClass" in action_types
        assert "Delete" in action_types
        assert "AbortIncompleteMultipartUpload" in action_types
        
        # Check for storage class transitions
        storage_class_rules = [rule for rule in rules if rule["action"]["type"] == "SetStorageClass"]
        storage_classes = [rule["action"]["storageClass"] for rule in storage_class_rules]
        
        assert "NEARLINE" in storage_classes
        assert "COLDLINE" in storage_classes
        assert "ARCHIVE" in storage_classes
        
    def test_lifecycle_policy_conditions(self):
        """Test lifecycle policy conditions are reasonable"""
        lifecycle_path = Path(__file__).parent.parent.parent.parent / "deploy" / "lifecycle.json"
        
        with open(lifecycle_path, 'r') as f:
            policy = json.load(f)
        
        rules = policy["lifecycle"]["rule"]
        
        # Find age-based transitions
        nearline_rule = next((rule for rule in rules 
                             if rule["action"]["type"] == "SetStorageClass" 
                             and rule["action"]["storageClass"] == "NEARLINE"), None)
        
        assert nearline_rule is not None
        assert nearline_rule["condition"]["age"] == 30  # 30 days to NEARLINE
        
        coldline_rule = next((rule for rule in rules 
                             if rule["action"]["type"] == "SetStorageClass" 
                             and rule["action"]["storageClass"] == "COLDLINE"), None)
        
        assert coldline_rule is not None
        assert coldline_rule["condition"]["age"] == 90  # 90 days to COLDLINE
        
        # Check version limit
        version_rule = next((rule for rule in rules 
                           if "numNewerVersions" in rule["condition"]), None)
        
        assert version_rule is not None
        assert version_rule["condition"]["numNewerVersions"] == 10


class TestGCSSetupScript:
    """Test GCS setup script functionality"""
    
    def test_setup_script_exists(self):
        """Test that setup script exists and is executable"""
        script_path = Path(__file__).parent.parent.parent.parent / "deploy" / "setup_gcs_infrastructure.sh"
        
        assert script_path.exists(), "setup_gcs_infrastructure.sh should exist"
        
        # Check if file is executable (on Unix systems)
        if hasattr(os, 'access'):
            assert os.access(script_path, os.X_OK), "setup script should be executable"
    
    def test_setup_script_content(self):
        """Test setup script contains required components"""
        script_path = Path(__file__).parent.parent.parent.parent / "deploy" / "setup_gcs_infrastructure.sh"
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for required functions
        assert "create_buckets()" in content
        assert "enable_versioning()" in content
        assert "apply_lifecycle_policies()" in content
        assert "setup_iam_permissions()" in content
        assert "enable_audit_logging()" in content
        
        # Check for bucket names
        assert "shyvr-models-prod" in content
        assert "shyvr-models-staging" in content
        
        # Check for error handling
        assert "set -euo pipefail" in content


class TestValidationScript:
    """Test GCS validation script functionality"""
    
    def test_validation_script_exists(self):
        """Test that validation script exists and is executable"""
        script_path = Path(__file__).parent.parent.parent.parent / "deploy" / "validate_gcs_setup.py"
        
        assert script_path.exists(), "validate_gcs_setup.py should exist"
        
        # Check if file is executable (on Unix systems)
        if hasattr(os, 'access'):
            assert os.access(script_path, os.X_OK), "validation script should be executable"
    
    def test_validation_script_imports(self):
        """Test validation script has proper imports"""
        script_path = Path(__file__).parent.parent.parent.parent / "deploy" / "validate_gcs_setup.py"
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for required imports
        assert "from google.cloud import storage" in content
        assert "import json" in content
        assert "import sys" in content
        
    def test_validation_script_classes(self):
        """Test validation script contains required classes and methods"""
        script_path = Path(__file__).parent.parent.parent.parent / "deploy" / "validate_gcs_setup.py"
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for main class
        assert "class GCSValidator:" in content
        
        # Check for required methods
        assert "def validate_bucket_existence" in content
        assert "def validate_versioning" in content
        assert "def validate_lifecycle_policy" in content
        assert "def test_read_write_permissions" in content
        assert "def run_validation" in content


class TestDocumentation:
    """Test documentation completeness"""
    
    def test_readme_exists(self):
        """Test that README documentation exists"""
        readme_path = Path(__file__).parent.parent.parent.parent / "deploy" / "README_GCS_SETUP.md"
        
        assert readme_path.exists(), "README_GCS_SETUP.md should exist"
    
    def test_readme_content(self):
        """Test README contains required sections"""
        readme_path = Path(__file__).parent.parent.parent.parent / "deploy" / "README_GCS_SETUP.md"
        
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Check for required sections
        assert "# GCS Infrastructure Setup Guide" in content
        assert "## Prerequisites" in content
        assert "## Quick Start" in content
        assert "## Troubleshooting" in content
        assert "## Security Best Practices" in content
        
        # Check for bucket names
        assert "shyvr-models-prod" in content
        assert "shyvr-models-staging" in content
        
        # Check for script references
        assert "setup_gcs_infrastructure.sh" in content
        assert "validate_gcs_setup.py" in content
        assert "lifecycle.json" in content


class TestFilePermissions:
    """Test file permissions and structure"""
    
    def test_script_permissions(self):
        """Test that scripts have correct permissions"""
        deploy_dir = Path(__file__).parent.parent.parent.parent / "deploy"
        
        setup_script = deploy_dir / "setup_gcs_infrastructure.sh"
        validation_script = deploy_dir / "validate_gcs_setup.py"
        
        # Test files exist
        assert setup_script.exists()
        assert validation_script.exists()
        
        # On Unix-like systems, test executability
        if hasattr(os, 'access'):
            assert os.access(setup_script, os.X_OK), "Setup script should be executable"
            assert os.access(validation_script, os.X_OK), "Validation script should be executable"
    
    def test_json_validity(self):
        """Test that JSON files are valid"""
        lifecycle_path = Path(__file__).parent.parent.parent.parent / "deploy" / "lifecycle.json"
        
        assert lifecycle_path.exists()
        
        # Test JSON is valid
        with open(lifecycle_path, 'r') as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"lifecycle.json is not valid JSON: {e}")


class TestIntegrationReadiness:
    """Test that files are ready for integration"""
    
    def test_environment_variable_usage(self):
        """Test scripts use appropriate environment variables"""
        script_path = Path(__file__).parent.parent.parent.parent / "deploy" / "setup_gcs_infrastructure.sh"
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for environment variable usage
        assert "GCP_PROJECT_ID" in content
        assert "CLOUD_RUN_SA" in content
        assert "GCP_REGION" in content
        
    def test_validation_script_environment(self):
        """Test validation script uses environment variables"""
        script_path = Path(__file__).parent.parent.parent.parent / "deploy" / "validate_gcs_setup.py"
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for environment variable usage
        assert "os.getenv('GCP_PROJECT_ID')" in content
        assert "os.getenv('GCS_PROD_BUCKET'" in content
        assert "os.getenv('GCS_STAGING_BUCKET'" in content
    
    def test_error_handling(self):
        """Test scripts have proper error handling"""
        script_path = Path(__file__).parent.parent.parent.parent / "deploy" / "setup_gcs_infrastructure.sh"
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for error handling
        assert "log_error" in content
        assert "exit 1" in content
        assert "check_prerequisites" in content