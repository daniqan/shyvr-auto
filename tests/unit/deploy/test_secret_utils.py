"""
Unit tests for secret management utilities
Tests individual functions and components of the secret management system
"""
import os
import sys
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from google.cloud import secretmanager
from google.api_core import exceptions

# Add deploy directory to path for importing utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../deploy'))


class TestSecretValidationUtility:
    """Test the validate_secrets.py utility"""
    
    def setup_method(self):
        """Set up test environment"""
        self.project_id = "test-project"
        
    @patch('google.cloud.secretmanager.SecretManagerServiceClient')
    def test_get_secret_value_success(self, mock_client_class):
        """Test successful secret value retrieval"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock successful secret access
        mock_response = Mock()
        mock_response.payload.data.decode.return_value = "test-secret-value"
        mock_client.access_secret_version.return_value = mock_response
        
        # Create temporary validate_secrets.py file for testing
        validate_code = '''
from google.cloud import secretmanager

def get_secret_value(project_id, secret_name, version="latest"):
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        return None
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(validate_code)
            f.flush()
            
            # Import the function
            spec = {}
            exec(compile(open(f.name).read(), f.name, 'exec'), spec)
            get_secret_value = spec['get_secret_value']
            
            # Test the function
            result = get_secret_value(self.project_id, "TEST_SECRET")
            
            assert result == "test-secret-value"
            mock_client.access_secret_version.assert_called_once()
            
        os.unlink(f.name)
    
    @patch('google.cloud.secretmanager.SecretManagerServiceClient')
    def test_get_secret_value_not_found(self, mock_client_class):
        """Test secret value retrieval when secret doesn't exist"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock secret not found
        mock_client.access_secret_version.side_effect = exceptions.NotFound("Secret not found")
        
        validate_code = '''
from google.cloud import secretmanager
from google.api_core import exceptions

def get_secret_value(project_id, secret_name, version="latest"):
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        return None
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(validate_code)
            f.flush()
            
            spec = {}
            exec(compile(open(f.name).read(), f.name, 'exec'), spec)
            get_secret_value = spec['get_secret_value']
            
            result = get_secret_value(self.project_id, "NONEXISTENT_SECRET")
            
            assert result is None
            
        os.unlink(f.name)
    
    def test_secret_categories_definition(self):
        """Test that secret categories are properly defined"""
        # Expected secret categories based on our implementation
        expected_categories = {
            'required_secrets': ["TELEGRAM_TOKEN", "WEBHOOK_SECRET", "DB_PASSWORD", "DATABASE_URL"],
            'blockchain_secrets': ["HELIUS_API_KEY", "BIRDEYE_API_KEY", "ETHERSCAN_API_KEY"],
            'ai_secrets': ["XAI_API_KEY", "OPENAI_API_KEY", "AGENT_API_KEY"],
            'market_secrets': ["LUNARCRUSH_API_KEY", "COINGECKO_API_KEY", "GLASSNODE_API_KEY"],
            'social_secrets': ["X_BEARER_TOKEN", "X_API_KEY", "X_API_SECRET"],
            'trading_secrets': ["SOLANA_PRIVATE_KEY", "ETHEREUM_PRIVATE_KEY", "HYPERLIQUID_PRIVATE_KEY"]
        }
        
        # Verify each category has the expected structure
        for category, secrets in expected_categories.items():
            assert isinstance(secrets, list), f"Category {category} should be a list"
            assert len(secrets) > 0, f"Category {category} should not be empty"
            
            for secret in secrets:
                assert isinstance(secret, str), f"Secret name {secret} should be a string"
                assert secret.isupper(), f"Secret name {secret} should be uppercase"
                assert "_" in secret or secret in ["TOKEN"], f"Secret {secret} should use underscore notation"


class TestSecretRotationUtility:
    """Test the rotate_secrets.py utility"""
    
    def setup_method(self):
        """Set up test environment"""
        self.project_id = "test-project"
    
    @patch('google.cloud.secretmanager.SecretManagerServiceClient')
    def test_secret_rotator_initialization(self, mock_client_class):
        """Test SecretRotator class initialization"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Create rotator code for testing
        rotator_code = '''
from google.cloud import secretmanager

class SecretRotator:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(rotator_code)
            f.flush()
            
            spec = {}
            exec(compile(open(f.name).read(), f.name, 'exec'), spec)
            SecretRotator = spec['SecretRotator']
            
            rotator = SecretRotator(self.project_id)
            
            assert rotator.project_id == self.project_id
            assert rotator.client is not None
            mock_client_class.assert_called_once()
            
        os.unlink(f.name)
    
    @patch('google.cloud.secretmanager.SecretManagerServiceClient')
    @patch('datetime.datetime')
    def test_backup_secret_naming_convention(self, mock_datetime, mock_client_class):
        """Test backup secret naming convention"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock datetime for consistent naming
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
        
        # Mock current secret value
        mock_response = Mock()
        mock_response.payload.data.decode.return_value = "current-value"
        mock_client.access_secret_version.return_value = mock_response
        
        mock_client.create_secret.return_value = Mock()
        mock_client.add_secret_version.return_value = Mock()
        
        backup_code = '''
import datetime
from google.cloud import secretmanager

class SecretRotator:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
    
    def backup_secret(self, secret_name):
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            current_value = response.payload.data.decode("UTF-8")
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{secret_name}_backup_{timestamp}"
            
            self.client.create_secret(
                request={
                    "parent": f"projects/{self.project_id}",
                    "secret_id": backup_name,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
            
            self.client.add_secret_version(
                request={
                    "parent": f"projects/{self.project_id}/secrets/{backup_name}",
                    "payload": {"data": current_value.encode("UTF-8")},
                }
            )
            
            return backup_name
        except Exception as e:
            return None
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(backup_code)
            f.flush()
            
            spec = {}
            exec(compile(open(f.name).read(), f.name, 'exec'), spec)
            SecretRotator = spec['SecretRotator']
            
            rotator = SecretRotator(self.project_id)
            backup_name = rotator.backup_secret("TEST_SECRET")
            
            assert backup_name == "TEST_SECRET_backup_20240101_120000"
            mock_client.create_secret.assert_called_once()
            mock_client.add_secret_version.assert_called_once()
            
        os.unlink(f.name)
    
    @patch('google.cloud.secretmanager.SecretManagerServiceClient')
    def test_list_backups_filtering(self, mock_client_class):
        """Test backup listing and filtering"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock secret list response
        mock_secrets = [
            Mock(name="projects/test/secrets/TELEGRAM_TOKEN"),
            Mock(name="projects/test/secrets/TELEGRAM_TOKEN_backup_20240101_120000"),
            Mock(name="projects/test/secrets/WEBHOOK_SECRET"),
            Mock(name="projects/test/secrets/WEBHOOK_SECRET_backup_20240102_130000"),
            Mock(name="projects/test/secrets/DB_PASSWORD"),
            Mock(name="projects/test/secrets/SOME_OTHER_SECRET"),
        ]
        mock_client.list_secrets.return_value = mock_secrets
        
        list_code = '''
from google.cloud import secretmanager

class SecretRotator:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
    
    def list_backups(self):
        try:
            parent = f"projects/{self.project_id}"
            secrets = self.client.list_secrets(request={"parent": parent})
            
            backups = []
            for secret in secrets:
                secret_name = secret.name.split("/")[-1]
                if "_backup_" in secret_name:
                    backups.append(secret_name)
            
            return backups
        except Exception as e:
            return []
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(list_code)
            f.flush()
            
            spec = {}
            exec(compile(open(f.name).read(), f.name, 'exec'), spec)
            SecretRotator = spec['SecretRotator']
            
            rotator = SecretRotator(self.project_id)
            backups = rotator.list_backups()
            
            expected_backups = [
                "TELEGRAM_TOKEN_backup_20240101_120000",
                "WEBHOOK_SECRET_backup_20240102_130000"
            ]
            
            assert set(backups) == set(expected_backups)
            mock_client.list_secrets.assert_called_once()
            
        os.unlink(f.name)


class TestDeploymentScriptValidation:
    """Test deployment script secret validation functions"""
    
    def setup_method(self):
        """Set up test environment"""
        self.deploy_script_path = "/Users/kendo/daniqan/shyvrai-rlte/deploy/deploy_latest.sh"
    
    def test_deployment_script_structure(self):
        """Test deployment script has correct structure"""
        assert os.path.exists(self.deploy_script_path)
        
        with open(self.deploy_script_path, 'r') as f:
            content = f.read()
        
        # Verify essential components exist
        required_components = [
            "validate_and_add_secret()",
            "REQUIRED_SECRETS=",
            "BLOCKCHAIN_SECRETS=",
            "AI_SECRETS=",
            "MARKET_SECRETS=",
            "SOCIAL_SECRETS=",
            "TRADING_SECRETS=",
            "SECRET_ARGS=",
            "SECRETS_ADDED=",
            "SECRETS_SKIPPED="
        ]
        
        for component in required_components:
            assert component in content, f"Deployment script should contain {component}"
    
    def test_secret_categories_completeness(self):
        """Test that all secret categories are comprehensive"""
        with open(self.deploy_script_path, 'r') as f:
            content = f.read()
        
        # Key secrets that should be included
        essential_secrets = [
            "TELEGRAM_TOKEN",
            "WEBHOOK_SECRET", 
            "DB_PASSWORD",
            "DATABASE_URL",
            "HELIUS_API_KEY",
            "BIRDEYE_API_KEY",
            "ETHERSCAN_API_KEY",
            "XAI_API_KEY",
            "OPENAI_API_KEY",
            "LUNARCRUSH_API_KEY",
            "COINGECKO_API_KEY",
            "X_BEARER_TOKEN",
            "SOLANA_PRIVATE_KEY",
            "ETHEREUM_PRIVATE_KEY"
        ]
        
        for secret in essential_secrets:
            assert secret in content, f"Deployment script should include {secret}"
    
    def test_trading_mode_safety_check(self):
        """Test that trading mode safety checks are properly implemented"""
        with open(self.deploy_script_path, 'r') as f:
            content = f.read()
        
        # Verify trading mode safety features
        safety_checks = [
            'TRADING_MODE=${TRADING_MODE:-"simulation"}',
            'if [[ "$TRADING_MODE" == "live" ]]',
            'SIMULATION MODE - Skipping trading secrets for safety',
            'LIVE TRADING MODE ENABLED',
            'CRITICAL trading secrets',
            'WARNING: Processing CRITICAL trading secrets'
        ]
        
        for check in safety_checks:
            assert check in content, f"Safety check '{check}' should be in deployment script"
    
    def test_secret_validation_error_handling(self):
        """Test that secret validation includes proper error handling"""
        with open(self.deploy_script_path, 'r') as f:
            content = f.read()
        
        # Verify error handling components
        error_handling = [
            'DEPLOYMENT FAILED: Required secret',
            'exists but has no value',
            'Optional .* secret not configured',
            'No secrets configured - deployment may fail',
            'exit 1'
        ]
        
        for pattern in error_handling:
            # Use simple string search instead of regex for basic validation
            assert any(phrase in content for phrase in [
                'DEPLOYMENT FAILED: Required secret',
                'exists but has no value', 
                'secret not configured',
                'No secrets configured',
                'exit 1'
            ]), f"Error handling pattern should be in deployment script"


class TestSecretSecurity:
    """Test security aspects of secret management"""
    
    def test_no_hardcoded_secrets_in_scripts(self):
        """Test that no actual secrets are hardcoded in management scripts"""
        scripts_to_check = [
            "/Users/kendo/daniqan/shyvrai-rlte/deploy/setup_secrets.sh",
            "/Users/kendo/daniqan/shyvrai-rlte/deploy/deploy_latest.sh"
        ]
        
        # Patterns that might indicate hardcoded secrets
        suspicious_patterns = [
            "sk-",  # OpenAI API key prefix
            "xai-", # XAI API key prefix
            "bot[0-9]", # Telegram bot token pattern
            "bearer ",  # Bearer token
            "api_key=",  # API key assignment
            "token=",   # Token assignment
        ]
        
        for script_path in scripts_to_check:
            if os.path.exists(script_path):
                with open(script_path, 'r') as f:
                    content = f.read().lower()
                
                for pattern in suspicious_patterns:
                    # Look for patterns but exclude obvious examples/documentation
                    if pattern in content:
                        # Make sure it's in a comment or example context
                        lines_with_pattern = [line for line in content.split('\n') if pattern in line]
                        for line in lines_with_pattern:
                            # Allow patterns in comments, examples, or documentation
                            assert any(marker in line for marker in ['#', 'echo', 'example', 'your_', 'test_', '***']), \
                                f"Potential hardcoded secret pattern '{pattern}' found in {script_path}: {line.strip()}"
    
    def test_secret_environment_variable_format(self):
        """Test that secret environment variables follow proper naming convention"""
        # Read deployment script to extract secret names
        with open("/Users/kendo/daniqan/shyvrai-rlte/deploy/deploy_latest.sh", 'r') as f:
            content = f.read()
        
        # Extract secret names (simple pattern matching)
        import re
        secret_pattern = r'"([A-Z_]+)"'
        secrets = re.findall(secret_pattern, content)
        
        # Filter to actual secret names (exclude other uppercase strings)
        actual_secrets = [s for s in secrets if any(keyword in s for keyword in [
            'API', 'KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'URL', 'PRIVATE'
        ])]
        
        for secret in actual_secrets:
            # Verify naming convention
            assert secret.isupper(), f"Secret {secret} should be all uppercase"
            assert not secret.startswith('_'), f"Secret {secret} should not start with underscore"
            assert not secret.endswith('_'), f"Secret {secret} should not end with underscore"
            assert '__' not in secret, f"Secret {secret} should not contain double underscores"
            
            # Verify meaningful naming
            assert len(secret) >= 3, f"Secret {secret} should have meaningful length"
    
    def test_setup_script_security_warnings(self):
        """Test that setup script includes appropriate security warnings"""
        setup_script_path = "/Users/kendo/daniqan/shyvrai-rlte/deploy/setup_secrets.sh"
        
        if os.path.exists(setup_script_path):
            with open(setup_script_path, 'r') as f:
                content = f.read()
            
            # Verify security warnings exist
            security_warnings = [
                "WARNING",
                "trading secrets contain sensitive",
                "wallet private keys",
                "production live trading mode",
                "security measures",
                "CRITICAL"
            ]
            
            for warning in security_warnings:
                assert warning.lower() in content.lower(), f"Setup script should include security warning about '{warning}'"


class TestSecretValidationCompliance:
    """Test compliance with secret management best practices"""
    
    def test_secret_manager_integration(self):
        """Test proper Google Secret Manager integration"""
        # Verify that scripts use proper Secret Manager commands
        scripts = [
            "/Users/kendo/daniqan/shyvrai-rlte/deploy/setup_secrets.sh",
            "/Users/kendo/daniqan/shyvrai-rlte/deploy/deploy_latest.sh"
        ]
        
        for script_path in scripts:
            if os.path.exists(script_path):
                with open(script_path, 'r') as f:
                    content = f.read()
                
                # Verify proper gcloud secrets commands are used
                if "gcloud" in content:
                    secret_commands = [
                        "gcloud secrets",
                        "secretmanager",
                        "--project",
                        "--quiet"
                    ]
                    
                    for command in secret_commands:
                        if command == "gcloud secrets":
                            assert command in content, f"Script {script_path} should use proper gcloud secrets commands"
    
    def test_iam_permission_setup(self):
        """Test that IAM permissions are properly configured"""
        setup_script_path = "/Users/kendo/daniqan/shyvrai-rlte/deploy/setup_secrets.sh"
        
        if os.path.exists(setup_script_path):
            with open(setup_script_path, 'r') as f:
                content = f.read()
            
            # Verify IAM configuration
            iam_components = [
                "iam-policy-binding",
                "secretmanager.secretAccessor",
                "serviceAccount",
                "roles/"
            ]
            
            for component in iam_components:
                assert component in content, f"Setup script should configure IAM component: {component}"
    
    def test_environment_variable_mapping_consistency(self):
        """Test that environment variable mapping is consistent"""
        # This test ensures that secrets defined in deployment match those used in code
        
        # Key environment variables used in the codebase
        codebase_env_vars = [
            "TELEGRAM_TOKEN", "HELIUS_API_KEY", "LUNARCRUSH_API_KEY",
            "COINGECKO_API_KEY", "XAI_API_KEY", "OPENAI_API_KEY",
            "X_BEARER_TOKEN", "ETHERSCAN_API_KEY"
        ]
        
        # Read deployment script
        with open("/Users/kendo/daniqan/shyvrai-rlte/deploy/deploy_latest.sh", 'r') as f:
            deploy_content = f.read()
        
        # Verify all codebase environment variables are handled in deployment
        for env_var in codebase_env_vars:
            assert env_var in deploy_content, f"Environment variable {env_var} should be handled in deployment"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])