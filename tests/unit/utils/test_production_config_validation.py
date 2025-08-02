"""
Test production configuration validation - TDD Phase 3.1
Tests to ensure production configuration validation and environment-specific separation
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from src.utils.config import ConfigManager, ConfigurationError
from src.utils.base import ConfigurationError as BaseConfigurationError


class TestProductionConfigValidation:
    """Test production configuration validation following TDD methodology"""
    
    def test_production_config_file_should_exist(self):
        """Test that production config file exists - THIS SHOULD FAIL INITIALLY"""
        # Check for production-specific config file
        prod_config_path = Path('/Users/kendo/daniqan/shyvrai-rlte/config/config.production.yaml')
        
        # This should fail initially - no production config file exists
        assert prod_config_path.exists(), f"Production config file missing at {prod_config_path}"
    
    def test_required_production_environment_variables_validation(self):
        """Test validation of required production environment variables - THIS SHOULD FAIL INITIALLY"""
        required_prod_vars = [
            'DB_HOST', 'DB_PASSWORD', 'SECRET_KEY', 'TELEGRAM_TOKEN',
            'HELIUS_API_KEY', 'ETHERSCAN_API_KEY', 'BIRDEYE_API_KEY'
        ]
        
        # Test with missing environment variables
        for missing_var in required_prod_vars:
            env_dict = {var: f'test_{var.lower()}' for var in required_prod_vars}
            env_dict.pop(missing_var)  # Remove one required variable
            
            test_config = {
                'app': {'name': 'test', 'environment': 'production'},
                'database': {
                    'host': '${DB_HOST}',
                    'password': '${DB_PASSWORD}'
                },
                'security': {
                    'secret_key': '${SECRET_KEY}',
                    'jwt_algorithm': 'HS256'
                },
                'telegram': {
                    'token': '${TELEGRAM_TOKEN}'
                },
                'apis': {
                    'helius': {'api_key': '${HELIUS_API_KEY}'},
                    'etherscan': {'api_key': '${ETHERSCAN_API_KEY}'},
                    'birdeye': {'api_key': '${BIRDEYE_API_KEY}'}
                }
            }
            
            with patch.dict(os.environ, env_dict, clear=True), \
                 patch('src.utils.config.yaml.safe_load', return_value=test_config), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open'):
                
                config_manager = ConfigManager()
                
                # Should fail when required production environment variable is missing
                with pytest.raises((ConfigurationError, BaseConfigurationError, ValueError, KeyError),
                                 match=f"{missing_var}|environment.*variable|required.*variable"):
                    config = config_manager.load()
    
    def test_production_config_schema_validation(self):
        """Test production configuration schema validation - THIS SHOULD FAIL INITIALLY"""
        # Invalid production configurations that should fail validation
        invalid_configs = [
            # Missing database config
            {
                'app': {'environment': 'production'},
                'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
                'telegram': {'token': '${TELEGRAM_TOKEN}'}
            },
            # Invalid risk management values
            {
                'app': {'environment': 'production'},
                'database': {'host': '${DB_HOST}', 'password': '${DB_PASSWORD}'},
                'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
                'telegram': {'token': '${TELEGRAM_TOKEN}'},
                'trading': {
                    'risk_management': {
                        'max_position_size_pct': 50.0,  # Too high for production
                        'max_daily_loss_pct': -5.0  # Invalid negative value
                    }
                }
            },
            # Missing API keys for production
            {
                'app': {'environment': 'production'},
                'database': {'host': '${DB_HOST}', 'password': '${DB_PASSWORD}'},
                'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
                'telegram': {'token': '${TELEGRAM_TOKEN}'},
                'apis': {}  # Empty APIs in production
            }
        ]
        
        env_vars = {
            'SECRET_KEY': 'production-secret-key-12345678901234567890',
            'DB_HOST': 'prod-db.example.com',
            'DB_PASSWORD': 'prod_password',
            'TELEGRAM_TOKEN': 'prod_telegram_token',
            'HELIUS_API_KEY': 'test_helius_key',
            'ETHERSCAN_API_KEY': 'test_etherscan_key',
            'BIRDEYE_API_KEY': 'test_birdeye_key'
        }
        
        for i, invalid_config in enumerate(invalid_configs):
            with patch.dict(os.environ, env_vars), \
                 patch('src.utils.config.yaml.safe_load', return_value=invalid_config), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open'):
                
                config_manager = ConfigManager()
                
                # Should fail schema validation for production
                with pytest.raises((ConfigurationError, BaseConfigurationError, ValueError),
                                 match="production.*validation|schema.*validation|required.*field"):
                    config = config_manager.load()
    
    def test_production_localhost_references_should_fail(self):
        """Test that localhost references fail in production - THIS SHOULD FAIL INITIALLY"""
        localhost_configs = [
            # Database host as localhost
            {
                'app': {'environment': 'production'},
                'database': {'host': 'localhost', 'password': '${DB_PASSWORD}'},
                'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
                'telegram': {'token': '${TELEGRAM_TOKEN}'},
                'apis': {
                    'helius': {'api_key': '${HELIUS_API_KEY}', 'base_url': 'https://api.helius.xyz'},
                    'etherscan': {'api_key': '${ETHERSCAN_API_KEY}', 'base_url': 'https://api.etherscan.io'},
                    'birdeye': {'api_key': '${BIRDEYE_API_KEY}', 'base_url': 'https://public-api.birdeye.so'}
                }
            },
            # API base URLs with localhost
            {
                'app': {'environment': 'production'},
                'database': {'host': '${DB_HOST}', 'password': '${DB_PASSWORD}'},
                'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
                'telegram': {'token': '${TELEGRAM_TOKEN}'},
                'apis': {
                    'helius': {'api_key': '${HELIUS_API_KEY}', 'base_url': 'https://api.helius.xyz'},
                    'etherscan': {'api_key': '${ETHERSCAN_API_KEY}', 'base_url': 'https://api.etherscan.io'},
                    'birdeye': {'api_key': '${BIRDEYE_API_KEY}', 'base_url': 'https://public-api.birdeye.so'},
                    'custom': {'base_url': 'http://localhost:8080'}
                }
            }
        ]
        
        env_vars = {
            'SECRET_KEY': 'production-secret-key-12345678901234567890',
            'DB_HOST': 'prod-db.example.com',
            'DB_PASSWORD': 'prod_password',
            'TELEGRAM_TOKEN': 'prod_telegram_token',
            'HELIUS_API_KEY': 'test_helius_key',
            'ETHERSCAN_API_KEY': 'test_etherscan_key',
            'BIRDEYE_API_KEY': 'test_birdeye_key'
        }
        
        for localhost_config in localhost_configs:
            with patch.dict(os.environ, env_vars), \
                 patch('src.utils.config.yaml.safe_load', return_value=localhost_config), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open'):
                
                config_manager = ConfigManager()
                
                # Should fail when localhost references found in production
                with pytest.raises((ConfigurationError, BaseConfigurationError, ValueError),
                                 match="localhost.*production|localhost.*not.*allowed|production.*localhost"):
                    config = config_manager.load()
    
    def test_production_startup_health_checks_should_fail(self):
        """Test production startup health checks - THIS SHOULD FAIL INITIALLY"""
        # Test config that should pass basic validation but fail health checks
        test_config = {
            'app': {'environment': 'production'},
            'database': {
                'host': '${DB_HOST}',
                'password': '${DB_PASSWORD}',
                'pool_size': 0  # Invalid pool size
            },
            'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
            'telegram': {'token': '${TELEGRAM_TOKEN}'},
            'trading': {
                'risk_management': {
                    'max_position_size_pct': 0.0,  # Invalid - no trading possible
                    'max_daily_loss_pct': 0.0      # Invalid - no risk management
                }
            }
        }
        
        env_vars = {
            'SECRET_KEY': 'production-secret-key-12345678901234567890',
            'DB_HOST': 'prod-db.example.com',
            'DB_PASSWORD': 'prod_password',
            'TELEGRAM_TOKEN': 'prod_telegram_token',
            'HELIUS_API_KEY': 'test_helius_key',
            'ETHERSCAN_API_KEY': 'test_etherscan_key',
            'BIRDEYE_API_KEY': 'test_birdeye_key'
        }
        
        with patch.dict(os.environ, env_vars), \
             patch('src.utils.config.yaml.safe_load', return_value=test_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            
            # Should fail health checks even if basic validation passes
            with pytest.raises((ConfigurationError, BaseConfigurationError, ValueError),
                             match="health.*check|startup.*validation|configuration.*invalid"):
                config = config_manager.load()


class TestEnvironmentSpecificConfigSeparation:
    """Test environment-specific configuration separation"""
    
    def test_development_config_allows_mocks_and_defaults(self):
        """Test development config allows mock values - THIS SHOULD FAIL INITIALLY"""
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
                'modes': {'simulation': True, 'live': False},  # Safe for dev
                'risk_management': {
                    'max_position_size_pct': 100.0  # High values OK in dev
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
            # This test will initially fail because environment-specific validation doesn't exist
    
    def test_staging_config_requires_production_like_validation(self):
        """Test staging config requires production-like validation - THIS SHOULD FAIL INITIALLY"""
        staging_config = {
            'app': {'environment': 'staging'},
            'database': {
                'host': 'localhost',  # Should fail in staging
                'password': 'staging_password'
            },
            'security': {
                'secret_key': 'dev-key-change-in-prod',  # Should fail in staging
                'jwt_algorithm': 'HS256'
            },
            'telegram': {'token': 'staging_token'}
        }
        
        with patch('src.utils.config.yaml.safe_load', return_value=staging_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            
            # Staging should have similar validation to production
            with pytest.raises((ConfigurationError, BaseConfigurationError, ValueError),
                             match="staging.*validation|localhost.*staging|default.*secret"):
                config = config_manager.load()
    
    def test_environment_specific_config_file_loading(self):
        """Test loading environment-specific config files - THIS SHOULD FAIL INITIALLY"""
        # Test that ConfigManager can load environment-specific files
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            config_manager = ConfigManager()
            
            # Should attempt to load config.production.yaml when ENVIRONMENT=production
            # This will fail initially because this functionality doesn't exist
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.side_effect = lambda path: str(path).endswith('config.production.yaml')
                
                with patch('builtins.open') as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = """
app:
  environment: production
database:
  host: ${DB_HOST}
  password: ${DB_PASSWORD}
security:
  secret_key: ${SECRET_KEY}
  jwt_algorithm: HS256
telegram:
  token: ${TELEGRAM_TOKEN}
"""
                    
                    # This should succeed when environment-specific loading is implemented
                    # Will fail initially because ConfigManager doesn't support this
                    try:
                        config = config_manager.load()
                        assert config.app.environment == 'production'
                    except Exception as e:
                        pytest.fail(f"Environment-specific config loading not implemented: {e}")


class TestConfigurationHealthChecks:
    """Test configuration health checks and monitoring integration"""
    
    def test_database_connection_health_check_should_fail(self):
        """Test database connection health check - THIS SHOULD FAIL INITIALLY"""
        test_config = {
            'app': {'environment': 'production'},
            'database': {
                'host': '${DB_HOST}',
                'password': '${DB_PASSWORD}',
                'pool_size': 10
            },
            'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
            'telegram': {'token': '${TELEGRAM_TOKEN}'}
        }
        
        env_vars = {
            'SECRET_KEY': 'production-secret-key-12345678901234567890',
            'DB_HOST': 'unreachable-db.example.com',  # Unreachable host
            'DB_PASSWORD': 'prod_password',
            'TELEGRAM_TOKEN': 'prod_telegram_token'
        }
        
        with patch.dict(os.environ, env_vars), \
             patch('src.utils.config.yaml.safe_load', return_value=test_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            config = config_manager.load()
            
            # Health check should fail for unreachable database
            # This will initially fail because health checks don't exist
            with pytest.raises((ConfigurationError, BaseConfigurationError, ConnectionError),
                             match="database.*connection|health.*check|connection.*failed"):
                # This should trigger health check validation
                health_check_result = config_manager.validate_health_checks()
                if not health_check_result:
                    raise ConfigurationError("Database connection health check failed")
    
    def test_api_endpoints_health_check_should_fail(self):
        """Test API endpoints health check - THIS SHOULD FAIL INITIALLY"""
        test_config = {
            'app': {'environment': 'production'},
            'database': {'host': '${DB_HOST}', 'password': '${DB_PASSWORD}'},
            'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
            'telegram': {'token': '${TELEGRAM_TOKEN}'},
            'apis': {
                'helius': {
                    'api_key': '${HELIUS_API_KEY}',
                    'base_url': 'https://unreachable-api.example.com'  # Unreachable
                }
            }
        }
        
        env_vars = {
            'SECRET_KEY': 'production-secret-key-12345678901234567890',
            'DB_HOST': 'prod-db.example.com',
            'DB_PASSWORD': 'prod_password',
            'TELEGRAM_TOKEN': 'prod_telegram_token',
            'HELIUS_API_KEY': 'test_api_key'
        }
        
        with patch.dict(os.environ, env_vars), \
             patch('src.utils.config.yaml.safe_load', return_value=test_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            config = config_manager.load()
            
            # API health check should fail for unreachable endpoints
            # This will initially fail because API health checks don't exist
            with pytest.raises((ConfigurationError, BaseConfigurationError, ConnectionError),
                             match="api.*health|endpoint.*unreachable|api.*validation"):
                # This should trigger API health check validation
                api_health_result = config_manager.validate_api_health()
                if not api_health_result:
                    raise ConfigurationError("API endpoints health check failed")
    
    def test_complete_startup_validation_suite_should_fail(self):
        """Test complete startup validation suite - THIS SHOULD FAIL INITIALLY"""
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
                'etherscan': {'api_key': '${ETHERSCAN_API_KEY}', 'base_url': 'https://api.etherscan.io'}
            },
            'trading': {
                'risk_management': {
                    'max_position_size_pct': 2.0,
                    'max_daily_loss_pct': 5.0
                }
            }
        }
        
        env_vars = {
            'SECRET_KEY': 'production-secret-key-12345678901234567890',
            'DB_HOST': 'prod-db.example.com',
            'DB_PASSWORD': 'prod_password',
            'TELEGRAM_TOKEN': 'prod_telegram_token',
            'HELIUS_API_KEY': 'test_helius_key',
            'ETHERSCAN_API_KEY': 'test_etherscan_key'
        }
        
        with patch.dict(os.environ, env_vars), \
             patch('src.utils.config.yaml.safe_load', return_value=valid_config), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open'):
            
            config_manager = ConfigManager()
            config = config_manager.load()
            
            # Complete validation suite should run all checks
            # This will initially fail because comprehensive validation doesn't exist
            try:
                validation_result = config_manager.validate_production_readiness()
                assert validation_result is True, "Production readiness validation should pass"
            except AttributeError:
                pytest.fail("Production readiness validation method not implemented")


class TestConfigurationSchemaValidation:
    """Test configuration schema validation and type checking"""
    
    def test_configuration_type_validation_should_fail(self):
        """Test configuration type validation - THIS SHOULD FAIL INITIALLY"""
        invalid_type_configs = [
            # String where integer expected
            {
                'app': {'environment': 'production'},
                'database': {
                    'host': '${DB_HOST}',
                    'password': '${DB_PASSWORD}',
                    'port': 'not_a_number',  # Invalid type
                    'pool_size': 10
                }
            },
            # Boolean where string expected
            {
                'app': {'environment': True},  # Invalid type for environment
                'database': {'host': '${DB_HOST}', 'password': '${DB_PASSWORD}'}
            },
            # Invalid nested structure
            {
                'app': {'environment': 'production'},
                'database': {'host': '${DB_HOST}', 'password': '${DB_PASSWORD}'},
                'trading': {
                    'risk_management': 'invalid_structure'  # Should be dict, not string
                }
            }
        ]
        
        env_vars = {
            'DB_HOST': 'prod-db.example.com',
            'DB_PASSWORD': 'prod_password'
        }
        
        for invalid_config in invalid_type_configs:
            with patch.dict(os.environ, env_vars), \
                 patch('src.utils.config.yaml.safe_load', return_value=invalid_config), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open'):
                
                config_manager = ConfigManager()
                
                # Should fail type validation
                with pytest.raises((ConfigurationError, BaseConfigurationError, ValueError, TypeError),
                                 match="type.*validation|invalid.*type|schema.*validation"):
                    config = config_manager.load()
    
    def test_configuration_constraint_validation_should_fail(self):
        """Test configuration constraint validation - THIS SHOULD FAIL INITIALLY"""
        constraint_violation_configs = [
            # Risk management constraints
            {
                'app': {'environment': 'production'},
                'database': {'host': '${DB_HOST}', 'password': '${DB_PASSWORD}'},
                'security': {'secret_key': '${SECRET_KEY}', 'jwt_algorithm': 'HS256'},
                'telegram': {'token': '${TELEGRAM_TOKEN}'},
                'trading': {
                    'risk_management': {
                        'max_position_size_pct': 150.0,  # > 100% is invalid
                        'max_daily_loss_pct': -10.0,     # Negative is invalid
                        'stop_loss_pct': 200.0           # > 100% is invalid
                    }
                }
            },
            # Database constraints
            {
                'app': {'environment': 'production'},
                'database': {
                    'host': '${DB_HOST}',
                    'password': '${DB_PASSWORD}',
                    'pool_size': -1,      # Negative is invalid
                    'max_overflow': -5    # Negative is invalid
                }
            }
        ]
        
        env_vars = {
            'SECRET_KEY': 'production-secret-key-12345678901234567890',
            'DB_HOST': 'prod-db.example.com',
            'DB_PASSWORD': 'prod_password',
            'TELEGRAM_TOKEN': 'prod_telegram_token',
            'HELIUS_API_KEY': 'test_helius_key',
            'ETHERSCAN_API_KEY': 'test_etherscan_key',
            'BIRDEYE_API_KEY': 'test_birdeye_key'
        }
        
        for constraint_config in constraint_violation_configs:
            with patch.dict(os.environ, env_vars), \
                 patch('src.utils.config.yaml.safe_load', return_value=constraint_config), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open'):
                
                config_manager = ConfigManager()
                
                # Should fail constraint validation
                with pytest.raises((ConfigurationError, BaseConfigurationError, ValueError),
                                 match="constraint.*violation|invalid.*value|validation.*failed"):
                    config = config_manager.load()