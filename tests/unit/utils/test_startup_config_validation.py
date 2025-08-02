"""
Test startup configuration validation - TDD Phase 3.1
Tests to ensure configuration validates properly at application startup
"""

import os
import pytest
from unittest.mock import patch

from src.utils.config import ConfigManager, ConfigurationError
from src.utils.base import ConfigurationError as BaseConfigurationError


class TestStartupConfigValidation:
    """Test startup configuration validation"""
    
    def test_production_startup_validation_passes_with_valid_config(self):
        """Test that valid production configuration passes startup validation"""
        valid_config = {
            'app': {'environment': 'production'},
            'database': {
                'host': '${DB_HOST}',
                'password': '${DB_PASSWORD}',
                'pool_size': 10
            },
            'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
            'telegram': {'token': '${TELEGRAM_TOKEN}'},
            'apis': {
                'helius': {'api_key': '${HELIUS_API_KEY}', 'base_url': 'https://api.helius.xyz'},
                'etherscan': {'api_key': '${ETHERSCAN_API_KEY}', 'base_url': 'https://api.etherscan.io'},
                'birdeye': {'api_key': '${BIRDEYE_API_KEY}', 'base_url': 'https://public-api.birdeye.so'}
            },
            'trading': {
                'risk_management': {
                    'max_position_size_pct': 2.0,
                    'max_daily_loss_pct': 5.0
                }
            }
        }
        
        env_vars = {
            'SECRET_KEY': 'production-secret-key-that-is-at-least-32-characters-long',
            'DB_HOST': 'prod-db.example.com',
            'DB_PASSWORD': 'prod_password',
            'TELEGRAM_TOKEN': 'prod_telegram_token',
            'HELIUS_API_KEY': 'test_helius_key',
            'ETHERSCAN_API_KEY': 'test_etherscan_key',
            'BIRDEYE_API_KEY': 'test_birdeye_key'
        }
        
        with patch.dict(os.environ, env_vars), \
             patch('src.utils.config.yaml.safe_load', return_value=valid_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            config = config_manager.load()
            
            # Should load successfully
            assert config is not None
            assert config.app.environment == 'production'
            assert config.database.host == 'prod-db.example.com'
            assert len(config.security.secret_key) >= 32
    
    def test_development_startup_validation_is_relaxed(self):
        """Test that development environment has relaxed validation"""
        dev_config = {
            'app': {'environment': 'development', 'debug': True},
            'database': {
                'host': 'localhost',  # OK in development
                'password': 'dev_password'
            },
            'security': {
                'secret_key': 'dev-key-change-in-prod',  # OK in development
                'jwt_algorithm': 'HS256'
            },
            'telegram': {'token': 'dev_token'},
            'trading': {
                'modes': {'simulation': True, 'live': False},
                'risk_management': {
                    'max_position_size_pct': 10.0  # Higher values OK in dev
                }
            }
        }
        
        with patch('src.utils.config.yaml.safe_load', return_value=dev_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            config = config_manager.load()
            
            # Development config should load successfully with relaxed validation
            assert config is not None
            assert config.app.environment == 'development'
            assert config.database.host == 'localhost'  # OK in dev
    
    def test_configuration_basic_validation_passes(self):
        """Test that basic configuration validation passes"""
        basic_config = {
            'app': {'environment': 'development'},
            'database': {'host': 'localhost', 'password': 'test_password'},
            'security': {'secret_key': 'test-secret-key-that-is-long-enough-to-pass-validation', 'jwt_algorithm': 'HS256'},
            'telegram': {'token': 'test_token'}
        }
        
        with patch('src.utils.config.yaml.safe_load', return_value=basic_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            config = config_manager.load()
            
            # Basic validation should pass
            validation_result = config_manager.validate()
            assert validation_result is True
    
    def test_health_checks_can_be_called(self):
        """Test that health checks can be called without error"""
        basic_config = {
            'app': {'environment': 'development'},
            'database': {'host': 'localhost', 'password': 'test_password'},
            'security': {'secret_key': 'test-secret-key-that-is-long-enough-to-pass-validation', 'jwt_algorithm': 'HS256'},
            'telegram': {'token': 'test_token'}
        }
        
        with patch('src.utils.config.yaml.safe_load', return_value=basic_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            config = config_manager.load()
            
            # Health checks should be callable (may not pass actual connectivity tests)
            try:
                health_result = config_manager.validate_health_checks()
                # Should not raise an exception, result depends on actual connectivity
                assert isinstance(health_result, bool)
            except (ConnectionError, ConfigurationError):
                # Expected for actual connectivity tests in test environment
                pass
    
    def test_api_health_checks_can_be_called(self):
        """Test that API health checks can be called without error"""
        config_with_apis = {
            'app': {'environment': 'development'},
            'database': {'host': 'localhost', 'password': 'test_password'},
            'security': {'secret_key': 'test-secret-key-that-is-long-enough-to-pass-validation', 'jwt_algorithm': 'HS256'},
            'telegram': {'token': 'test_token'},
            'apis': {
                'helius': {'api_key': 'test_key', 'base_url': 'https://api.helius.xyz'}
            }
        }
        
        with patch('src.utils.config.yaml.safe_load', return_value=config_with_apis), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            config = config_manager.load()
            
            # API health checks should be callable
            try:
                api_health_result = config_manager.validate_api_health()
                assert isinstance(api_health_result, bool)
            except (ConnectionError, ConfigurationError):
                # Expected for actual connectivity tests in test environment
                pass