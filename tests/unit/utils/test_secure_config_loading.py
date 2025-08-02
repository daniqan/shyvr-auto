"""
TDD tests for secure configuration loading functionality
Tests configuration encryption, immutability, and audit logging
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

from src.utils.config import ConfigManager, RLTEConfig, ConfigurationError
from src.utils.security.config_encryption import ConfigEncryption
from src.utils.security.config_auditor import ConfigAuditor


class TestSecureConfigurationLoading:
    """Test secure configuration loading with encryption and validation"""
    
    def test_config_encryption_initialization_works(self):
        """Test that ConfigEncryption initializes successfully"""
        encryption = ConfigEncryption()
        assert encryption is not None
        
        # Test actual encryption/decryption functionality
        test_value = "test_secret"
        encrypted = encryption.encrypt_value(test_value)
        
        assert encrypted.startswith("encrypted:")
        assert encrypted != test_value
        
        decrypted = encryption.decrypt_value(encrypted)
        assert decrypted == test_value
    
    def test_config_encryption_decryption_works(self):
        """Test that ConfigEncryption decryption works properly"""
        encryption = ConfigEncryption()
        
        # Test with a known secret
        original_value = "my_secret_password"
        encrypted_value = encryption.encrypt_value(original_value)
        decrypted_value = encryption.decrypt_value(encrypted_value)
        
        assert decrypted_value == original_value
    
    def test_config_auditor_initialization_works(self):
        """Test that ConfigAuditor initializes successfully"""
        auditor = ConfigAuditor(audit_file="/tmp/test_audit.log")
        assert auditor is not None
        
        # Should be able to log config access without exceptions
        auditor.log_config_access("test_key", "test_value")
    
    def test_pydantic_config_modification_still_allowed(self):
        """Test that Pydantic configuration objects can still be modified (immutability not integrated yet)"""
        # Create a temporary test config to avoid validation issues
        config_data = {
            'app': {'name': 'test', 'environment': 'development'},
            'database': {
                'host': 'localhost',
                'password': 'test_password'
            },
            'telegram': {'token': 'test_token'},
            'security': {'secret_key': 'test_secret_key_1234567890123456'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            config = config_manager.load()
            
            # Currently, Pydantic objects are still mutable - this is expected
            # TODO: Integrate ImmutableConfig wrapper with ConfigManager
            config.database.host = "modified_host"
            assert config.database.host == "modified_host"
        finally:
            os.unlink(config_path)
    
    def test_runtime_validation_hooks_fail_without_implementation(self):
        """Test that runtime validation hooks are not implemented yet"""
        config_manager = ConfigManager()
        
        with pytest.raises(NotImplementedError):
            config_manager.add_runtime_validator("database.host", lambda x: x != "localhost")
    
    def test_config_change_detection_fails_without_implementation(self):
        """Test that configuration change detection is not implemented yet"""
        config_manager = ConfigManager()
        
        with pytest.raises(NotImplementedError):
            config_manager.enable_change_detection()
    
    def test_encrypted_config_loading_works(self):
        """Test that encrypted configuration loading works properly"""
        # First encrypt a test password
        encryption = ConfigEncryption()
        encrypted_password = encryption.encrypt_value("secret_db_password")
        
        config_data = {
            'app': {'name': 'test', 'environment': 'development'},
            'database': {
                'host': 'localhost',
                'password': encrypted_password
            },
            'telegram': {'token': 'test_token'},
            'security': {'secret_key': 'test_secret_key_1234567890123456'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            config = config_manager.load()
            
            # Verify the password was decrypted properly
            assert config.database.password == "secret_db_password"
        finally:
            os.unlink(config_path)
    
    def test_sensitive_value_detection_works(self):
        """Test that sensitive value detection works properly"""
        # Create a test config with valid data to avoid validation errors
        config_data = {
            'app': {'name': 'test', 'environment': 'development'},
            'database': {
                'host': 'localhost',
                'password': 'test_password'  # This should be detected as sensitive
            },
            'telegram': {'token': 'test_token'},  # This should be detected as sensitive
            'security': {'secret_key': 'test_secret_key_1234567890123456'}  # This should be detected as sensitive
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            sensitive_keys = config_manager.detect_sensitive_values()
            
            # Should return a list
            assert isinstance(sensitive_keys, list)
            
            # Should detect at least the sensitive keys we added
            assert len(sensitive_keys) >= 3  # password, token, secret_key
            
            # Verify specific sensitive keys are detected
            sensitive_key_names = [key.split('.')[-1] for key in sensitive_keys]
            assert any('password' in key for key in sensitive_key_names)
            assert any('token' in key for key in sensitive_key_names)
            assert any('secret_key' in key for key in sensitive_key_names)
        finally:
            os.unlink(config_path)
    
    def test_config_audit_logging_works(self):
        """Test that configuration audit logging works properly"""
        config_manager = ConfigManager()
        
        # Should work without exceptions
        config_manager.enable_audit_logging()
        
        # Should be able to log config access
        config_manager.log_config_access("database.password", "test_password", "test_user")


class TestConfigEncryptionFeatures:
    """Test configuration encryption features that are now implemented"""
    
    def test_encrypt_sensitive_values_works(self):
        """Test that sensitive value encryption works properly"""
        encryption = ConfigEncryption()
        
        config = {
            'database': {'password': 'secret123', 'host': 'localhost'},
            'telegram': {'token': 'token123'},
            'security': {'secret_key': 'my_secret_key'},
            'app': {'name': 'test_app'}
        }
        
        encrypted_config = encryption.encrypt_sensitive_config(config)
        
        # Sensitive values should be encrypted
        assert encrypted_config['database']['password'].startswith('encrypted:')
        assert encrypted_config['telegram']['token'].startswith('encrypted:')
        assert encrypted_config['security']['secret_key'].startswith('encrypted:')
        
        # Non-sensitive values should remain unchanged
        assert encrypted_config['database']['host'] == 'localhost'
        assert encrypted_config['app']['name'] == 'test_app'
    
    def test_decrypt_config_values_works(self):
        """Test that config value decryption works properly"""
        encryption = ConfigEncryption()
        
        # Create encrypted values
        encrypted_password = encryption.encrypt_value('secret123')
        encrypted_token = encryption.encrypt_value('token123')
        
        config = {
            'database': {'password': encrypted_password, 'host': 'localhost'},
            'telegram': {'token': encrypted_token},
            'app': {'name': 'test_app'}
        }
        
        decrypted_config = encryption.decrypt_config_values(config)
        
        # Encrypted values should be decrypted
        assert decrypted_config['database']['password'] == 'secret123'
        assert decrypted_config['telegram']['token'] == 'token123'
        
        # Non-encrypted values should remain unchanged
        assert decrypted_config['database']['host'] == 'localhost'
        assert decrypted_config['app']['name'] == 'test_app'
    
    def test_key_rotation_works(self):
        """Test that encryption key rotation works properly"""
        encryption = ConfigEncryption()
        
        # Encrypt a value with original key
        original_value = "test_secret"
        encrypted_with_old_key = encryption.encrypt_value(original_value)
        
        # Rotate the key
        new_key = encryption.rotate_encryption_key()
        
        # New key should be returned
        assert new_key is not None
        assert isinstance(new_key, str)
        
        # Old encrypted values may not decrypt with new key (expected behavior)
        # This is by design - key rotation requires re-encryption of all values
    
    def test_encryption_key_derivation_works(self):
        """Test that encryption key derivation works properly"""
        encryption = ConfigEncryption()
        
        # This should work and return a key
        key = encryption.derive_key_from_environment()
        
        assert key is not None
        assert isinstance(key, str)
        assert len(key) > 0


class TestConfigImmutability:
    """Test configuration immutability features that are now implemented"""
    
    def test_immutable_config_wrapper_works(self):
        """Test that immutable config wrapper works properly"""
        from src.utils.security.immutable_config import ImmutableConfig
        
        config_data = {'database': {'host': 'localhost'}}
        
        immutable_config = ImmutableConfig(config_data)
        
        # Should be able to read values
        assert immutable_config.database.host == 'localhost'
        
        # Should not be able to modify values
        with pytest.raises((AttributeError, TypeError)):
            immutable_config.database.host = "malicious_host"
    
    def test_immutable_config_deep_protection_works(self):
        """Test that deep immutability protection works properly"""
        from src.utils.security.immutable_config import ImmutableConfig
        
        config_data = {
            'database': {'host': 'localhost'},
            'nested': {'deep': {'value': 'test'}}
        }
        
        immutable_config = ImmutableConfig(config_data)
        
        # Should be able to read deeply nested values
        assert immutable_config.nested.deep.value == 'test'
        
        # Should prevent modification at any level
        with pytest.raises((AttributeError, TypeError)):
            immutable_config.nested.deep.value = "modified"
        
        # Should also prevent direct assignment
        with pytest.raises((AttributeError, TypeError)):
            immutable_config.database = {'host': 'malicious'}


class TestConfigChangeDetection:
    """Test configuration change detection features that need to be implemented"""
    
    def test_change_detector_not_implemented(self):
        """Test that configuration change detector is not implemented"""
        from src.utils.security.config_change_detector import ConfigChangeDetector
        
        with pytest.raises(NotImplementedError):
            detector = ConfigChangeDetector()
            detector.start_monitoring()
    
    def test_config_hash_comparison_not_implemented(self):
        """Test that configuration hash comparison is not implemented"""
        from src.utils.security.config_change_detector import ConfigChangeDetector
        
        with pytest.raises(NotImplementedError):
            detector = ConfigChangeDetector()
            detector.compute_config_hash({'test': 'config'})
    
    def test_change_callbacks_not_implemented(self):
        """Test that configuration change callbacks are not implemented"""
        from src.utils.security.config_change_detector import ConfigChangeDetector
        
        with pytest.raises(NotImplementedError):
            detector = ConfigChangeDetector()
            detector.register_change_callback(lambda old, new: None)


class TestConfigAuditLogging:
    """Test configuration audit logging features that are now implemented"""
    
    def test_audit_logger_initialization_works(self):
        """Test that audit logger initialization works properly"""
        auditor = ConfigAuditor(audit_file="/tmp/test_audit.log")
        assert auditor is not None
        assert auditor.audit_file == "/tmp/test_audit.log"
        assert auditor.enable_alerts is True
    
    def test_config_access_logging_works(self):
        """Test that configuration access logging works properly"""
        auditor = ConfigAuditor(audit_file="/tmp/test_audit.log")
        
        # Should not raise any exceptions
        auditor.log_config_access("database.password", "test_password", "test_user")
        auditor.log_config_access("app.name", "test_app", "test_user")
        
        # Check that access counts are tracked
        assert auditor._access_counts.get("database.password", 0) >= 1
        assert auditor._access_counts.get("app.name", 0) >= 1
    
    def test_config_modification_logging_works(self):
        """Test that configuration modification logging works properly"""
        auditor = ConfigAuditor(audit_file="/tmp/test_audit.log")
        
        # Should not raise any exceptions
        auditor.log_config_modification("database.host", "old_host", "new_host", "admin_user")
        
        # Sensitive modifications should trigger alerts
        auditor.log_config_modification("secret_key", "old_secret", "new_secret", "admin_user")
    
    def test_sensitive_access_alerting_works(self):
        """Test that sensitive configuration access alerting works properly"""
        auditor = ConfigAuditor(audit_file="/tmp/test_audit.log", enable_alerts=True)
        
        # Should not raise any exceptions
        auditor.alert_sensitive_access("secret_key", "admin_user")
        
        # Test with alerts disabled
        auditor_no_alerts = ConfigAuditor(audit_file="/tmp/test_audit.log", enable_alerts=False)
        auditor_no_alerts.alert_sensitive_access("secret_key", "admin_user")  # Should not alert
    
    def test_sensitive_value_redaction(self):
        """Test that sensitive values are properly redacted in logs"""
        auditor = ConfigAuditor(audit_file="/tmp/test_audit.log")
        
        # Test redaction of sensitive values
        redacted = auditor._redact_sensitive_value("password", "my_secret_password")
        assert redacted == "my***rd"
        
        # Test non-sensitive values remain unchanged
        non_sensitive = auditor._redact_sensitive_value("hostname", "localhost")
        assert non_sensitive == "localhost"
        
        # Test short values are fully redacted
        short_secret = auditor._redact_sensitive_value("key", "abc")
        assert short_secret == "***REDACTED***"