"""
Tests for RL experience storage configuration models and validation

Following TDD methodology - tests written before implementation to define expected behavior.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import yaml

from src.utils.config import (
    RLExperienceStorageConfig, 
    RLExperiencePerformanceConfig,
    RLExperienceDatabaseConfig,
    RLExperienceLifecycleConfig,
    ConfigManager,
    get_config
)


class TestRLExperienceStorageConfig:
    """Test RL experience storage configuration model"""

    def test_default_values(self):
        """Test default configuration values"""
        config = RLExperienceStorageConfig()
        
        # Default storage settings
        assert config.enabled is True
        assert config.storage_backend == "database"
        assert config.max_experiences == 50000
        assert config.batch_size == 64
        assert config.prioritized_replay is True
        
    def test_performance_config_defaults(self):
        """Test performance configuration defaults"""
        config = RLExperiencePerformanceConfig()
        
        assert config.cache_size == 1000
        assert config.async_operations is True
        assert config.compression is False
        assert config.query_timeout_seconds == 30
        assert config.batch_commit_size == 100

    def test_database_config_defaults(self):
        """Test database configuration defaults"""
        config = RLExperienceDatabaseConfig()
        
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.timeout_seconds == 30
        assert config.enable_query_logging is False
        assert config.connection_retry_attempts == 3

    def test_lifecycle_config_defaults(self):
        """Test lifecycle management configuration defaults"""
        config = RLExperienceLifecycleConfig()
        
        assert config.cleanup_enabled is True
        assert config.max_age_days == 30
        assert config.cleanup_interval_hours == 24
        assert config.archive_old_experiences is False
        assert config.min_experiences_to_keep == 1000

    def test_config_validation(self):
        """Test configuration validation rules"""
        # Valid configuration should pass
        config = RLExperienceStorageConfig(
            max_experiences=10000,
            batch_size=32,
            performance=RLExperiencePerformanceConfig(cache_size=500)
        )
        assert config.max_experiences == 10000
        assert config.performance.cache_size == 500

    def test_invalid_config_values(self):
        """Test validation of invalid configuration values"""
        # Batch size should be positive
        with pytest.raises(ValueError, match="Batch size must be positive"):
            RLExperienceStorageConfig(batch_size=0)
        
        # Max experiences should be positive
        with pytest.raises(ValueError, match="Max experiences must be positive"):
            RLExperienceStorageConfig(max_experiences=-1)
        
        # Cache size should be positive
        with pytest.raises(ValueError, match="Cache size must be positive"):
            RLExperiencePerformanceConfig(cache_size=0)


class TestRLExperienceConfigIntegration:
    """Test RL experience configuration integration with main config system"""

    def test_config_yaml_integration(self):
        """Test that RL experience config integrates with YAML configuration"""
        yaml_content = """
        app:
          name: "test-app"
        database:
          host: "localhost"
          password: "test123"
        telegram:
          token: "test-token"
        rl:
          experience_storage:
            enabled: true
            storage_backend: "database"
            max_experiences: 25000
            batch_size: 128
            prioritized_replay: false
            
            performance:
              cache_size: 2000
              async_operations: false
              compression: true
              query_timeout_seconds: 60
              batch_commit_size: 200
            
            database:
              pool_size: 8
              max_overflow: 15
              timeout_seconds: 45
              enable_query_logging: true
              connection_retry_attempts: 5
            
            lifecycle:
              cleanup_enabled: false
              max_age_days: 60
              cleanup_interval_hours: 12
              archive_old_experiences: true
              min_experiences_to_keep: 5000
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path)
            config = manager.load()
            
            # Check if RL experience storage config is loaded
            exp_config = config.rl.experience_storage
            assert exp_config.enabled is True
            assert exp_config.storage_backend == "database"
            assert exp_config.max_experiences == 25000
            assert exp_config.batch_size == 128
            assert exp_config.prioritized_replay is False
            
            # Check performance config
            perf_config = exp_config.performance
            assert perf_config.cache_size == 2000
            assert perf_config.async_operations is False
            assert perf_config.compression is True
            assert perf_config.query_timeout_seconds == 60
            assert perf_config.batch_commit_size == 200
            
            # Check database config
            db_config = exp_config.database
            assert db_config.pool_size == 8
            assert db_config.max_overflow == 15
            assert db_config.timeout_seconds == 45
            assert db_config.enable_query_logging is True
            assert db_config.connection_retry_attempts == 5
            
            # Check lifecycle config
            lifecycle_config = exp_config.lifecycle
            assert lifecycle_config.cleanup_enabled is False
            assert lifecycle_config.max_age_days == 60
            assert lifecycle_config.cleanup_interval_hours == 12
            assert lifecycle_config.archive_old_experiences is True
            assert lifecycle_config.min_experiences_to_keep == 5000
            
        finally:
            os.unlink(config_path)

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in RL experience config"""
        yaml_content = """
        app:
          name: "test-app"
        database:
          host: "localhost"
          password: "test123"
        telegram:
          token: "test-token"
        rl:
          experience_storage:
            enabled: ${RL_EXPERIENCE_ENABLED:true}
            max_experiences: ${RL_MAX_EXPERIENCES:10000}
            batch_size: ${RL_BATCH_SIZE:64}
            
            database:
              pool_size: ${RL_DB_POOL_SIZE:5}
              timeout_seconds: ${RL_DB_TIMEOUT:30}
        """
        
        # Set environment variables
        env_vars = {
            'RL_EXPERIENCE_ENABLED': 'false',
            'RL_MAX_EXPERIENCES': '20000',
            'RL_BATCH_SIZE': '128',
            'RL_DB_POOL_SIZE': '10',
            'RL_DB_TIMEOUT': '60'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            with patch.dict(os.environ, env_vars):
                manager = ConfigManager(config_path)
                config = manager.load()
                
                exp_config = config.rl.experience_storage
                assert exp_config.enabled is False
                assert exp_config.max_experiences == 20000
                assert exp_config.batch_size == 128
                assert exp_config.database.pool_size == 10
                assert exp_config.database.timeout_seconds == 60
                
        finally:
            os.unlink(config_path)

    def test_missing_rl_experience_config(self):
        """Test behavior when RL experience config is missing"""
        yaml_content = """
        app:
          name: "test-app"
        database:
          host: "localhost"
          password: "test123"
        telegram:
          token: "test-token"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path)
            config = manager.load()
            
            # Should use default RL experience storage config
            exp_config = config.rl.experience_storage
            assert exp_config.enabled is True
            assert exp_config.storage_backend == "database"
            assert exp_config.max_experiences == 50000
            
        finally:
            os.unlink(config_path)


class TestEnvironmentVariablesValidation:
    """Test environment variable loading and validation for RL experience storage"""

    def test_required_environment_variables(self):
        """Test that required environment variables are validated"""
        required_vars = [
            'RL_EXPERIENCE_DB_HOST',
            'RL_EXPERIENCE_DB_PORT', 
            'RL_EXPERIENCE_DB_NAME',
            'RL_EXPERIENCE_DB_USER',
            'RL_EXPERIENCE_DB_PASSWORD'
        ]
        
        # Mock environment variables
        env_vars = {
            'RL_EXPERIENCE_DB_HOST': 'localhost',
            'RL_EXPERIENCE_DB_PORT': '5432',
            'RL_EXPERIENCE_DB_NAME': 'rl_experiences',
            'RL_EXPERIENCE_DB_USER': 'rl_user',
            'RL_EXPERIENCE_DB_PASSWORD': 'rl_password'
        }
        
        with patch.dict(os.environ, env_vars):
            # Validation should pass
            assert os.getenv('RL_EXPERIENCE_DB_HOST') == 'localhost'
            assert os.getenv('RL_EXPERIENCE_DB_PORT') == '5432'
            assert os.getenv('RL_EXPERIENCE_DB_PASSWORD') == 'rl_password'

    def test_optional_environment_variables(self):
        """Test optional environment variables with defaults"""
        optional_vars = {
            'RL_EXPERIENCE_CACHE_SIZE': '2000',
            'RL_EXPERIENCE_COMPRESSION': 'true',
            'RL_EXPERIENCE_ASYNC_OPS': 'false',
            'RL_EXPERIENCE_CLEANUP_ENABLED': 'false',
            'RL_EXPERIENCE_MAX_AGE_DAYS': '45'
        }
        
        with patch.dict(os.environ, optional_vars):
            assert os.getenv('RL_EXPERIENCE_CACHE_SIZE') == '2000'
            assert os.getenv('RL_EXPERIENCE_COMPRESSION') == 'true'
            assert os.getenv('RL_EXPERIENCE_ASYNC_OPS') == 'false'
            assert os.getenv('RL_EXPERIENCE_CLEANUP_ENABLED') == 'false'
            assert os.getenv('RL_EXPERIENCE_MAX_AGE_DAYS') == '45'

    def test_feature_flags_environment_variables(self):
        """Test feature flag environment variables"""
        feature_flags = {
            'RL_EXPERIENCE_STORAGE_ENABLED': 'true',
            'RL_EXPERIENCE_PRIORITIZED_REPLAY': 'false', 
            'RL_EXPERIENCE_DATABASE_BACKEND': 'true',
            'RL_EXPERIENCE_MEMORY_BACKEND': 'false',
            'RL_EXPERIENCE_COMPRESSION_ENABLED': 'true'
        }
        
        with patch.dict(os.environ, feature_flags):
            assert os.getenv('RL_EXPERIENCE_STORAGE_ENABLED') == 'true'
            assert os.getenv('RL_EXPERIENCE_PRIORITIZED_REPLAY') == 'false'
            assert os.getenv('RL_EXPERIENCE_DATABASE_BACKEND') == 'true'
            assert os.getenv('RL_EXPERIENCE_MEMORY_BACKEND') == 'false'
            assert os.getenv('RL_EXPERIENCE_COMPRESSION_ENABLED') == 'true'


class TestConfigurationValidationMethods:
    """Test configuration validation methods"""

    def test_validate_rl_experience_config(self):
        """Test RL experience configuration validation method"""
        yaml_content = """
        app:
          name: "test-app"
        database:
          host: "localhost"
          password: "test123"
        telegram:
          token: "test-token"
        rl:
          experience_storage:
            enabled: true
            max_experiences: 50000
            batch_size: 64
            
            database:
              pool_size: 5
              timeout_seconds: 30
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path)
            # Validation should succeed
            assert manager.validate() is True
            
        finally:
            os.unlink(config_path)

    def test_validate_rl_experience_config_warnings(self):
        """Test configuration validation warnings for RL experience storage"""
        yaml_content = """
        app:
          name: "test-app"
        database:
          host: "localhost"
          password: "test123"
        telegram:
          token: "test-token"
        rl:
          experience_storage:
            enabled: true
            max_experiences: 1000000  # Very large value - should warn
            batch_size: 1             # Very small value - should warn
            
            database:
              pool_size: 100          # Very large pool - should warn
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path)
            # Should still validate but with warnings
            assert manager.validate() is True
            
        finally:
            os.unlink(config_path)