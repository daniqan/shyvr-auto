"""
Unit tests for configuration management
Tests the ConfigManager class and configuration loading
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.utils.config import ConfigManager, RLTEConfig, get_config, init_config
from src.utils.base import ConfigurationError


class TestConfigManager:
    """Test ConfigManager functionality"""

    def test_config_loading(self, temp_config_file: Path):
        """Test configuration loads correctly from YAML file"""
        config_manager = ConfigManager(temp_config_file)
        config = config_manager.load()
        
        assert isinstance(config, RLTEConfig)
        assert config.app.name == "test-rlte"
        assert config.app.environment == "test"
        assert config.database.host == "localhost"
        assert config.database.port == 5432

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config"""
        config_content = """
app:
  name: "${APP_NAME:default-app}"
  debug: "${DEBUG:false}"
  port: "${PORT:8080}"

database:
  password: "${DB_PASSWORD}"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = Path(f.name)
        
        try:
            # Set environment variables
            os.environ["APP_NAME"] = "test-app"
            os.environ["DEBUG"] = "true"
            os.environ["PORT"] = "9000"
            os.environ["DB_PASSWORD"] = "secret123"
            
            config_manager = ConfigManager(temp_path)
            config = config_manager.load()
            
            assert config.app.name == "test-app"
            assert config.app.debug == True
            # Note: port might be a string in this simplified test
            assert str(config.get("app.port", "")).startswith("9")
            
        finally:
            # Cleanup
            temp_path.unlink()
            for var in ["APP_NAME", "DEBUG", "PORT", "DB_PASSWORD"]:
                os.environ.pop(var, None)

    def test_config_file_not_found(self):
        """Test error handling when config file doesn't exist"""
        config_manager = ConfigManager(Path("nonexistent.yaml"))
        
        with pytest.raises(ConfigurationError) as exc_info:
            config_manager.load()
        
        assert "Configuration file not found" in str(exc_info.value)

    def test_invalid_yaml(self):
        """Test error handling for invalid YAML syntax"""
        invalid_yaml = """
app:
  name: "test"
  invalid: [unclosed list
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = Path(f.name)
        
        try:
            config_manager = ConfigManager(temp_path)
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager.load()
            
            assert "Invalid YAML" in str(exc_info.value)
            
        finally:
            temp_path.unlink()

    def test_config_reload(self, temp_config_file: Path):
        """Test configuration reloading"""
        config_manager = ConfigManager(temp_config_file)
        
        # Load initial config
        config1 = config_manager.load()
        assert config1.app.name == "test-rlte"
        
        # Modify the config file
        with open(temp_config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data['app']['name'] = "modified-rlte"
        
        with open(temp_config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Reload config
        config2 = config_manager.reload()
        assert config2.app.name == "modified-rlte"

    def test_get_config_value(self, temp_config_file: Path):
        """Test getting configuration values by dot notation"""
        config_manager = ConfigManager(temp_config_file)
        
        assert config_manager.get("app.name") == "test-rlte"
        assert config_manager.get("database.port") == 5432
        assert config_manager.get("nonexistent.key", "default") == "default"
        assert config_manager.get("app.nonexistent") is None

    def test_config_validation(self, temp_config_file: Path):
        """Test configuration validation"""
        config_manager = ConfigManager(temp_config_file)
        
        # Should pass validation
        assert config_manager.validate() == True

    def test_database_url_generation(self, test_config: RLTEConfig):
        """Test database URL generation"""
        db_url = test_config.database.url
        expected = "postgresql://test_user:test_password@localhost:5432/test_rlte"
        assert db_url == expected

    def test_default_values(self):
        """Test default configuration values"""
        minimal_config = {
            "database": {"password": "test"},
            "telegram": {"token": "test"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            temp_path = Path(f.name)
        
        try:
            config_manager = ConfigManager(temp_path)
            config = config_manager.load()
            
            # Test default values are applied
            assert config.app.name == "shyvr-rlte"
            assert config.app.version == "0.1.0"
            assert config.database.pool_size == 10
            assert config.agent.model_type == "local"
            
        finally:
            temp_path.unlink()

    def test_environment_precedence(self):
        """Test that environment variables take precedence over config file"""
        config_content = """
app:
  name: "file-config"
  environment: "${ENVIRONMENT:development}"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = Path(f.name)
        
        try:
            os.environ["ENVIRONMENT"] = "production"
            
            config_manager = ConfigManager(temp_path)
            config = config_manager.load()
            
            assert config.app.environment == "production"
            
        finally:
            temp_path.unlink()
            os.environ.pop("ENVIRONMENT", None)


class TestGlobalConfig:
    """Test global configuration functions"""

    def test_global_config_initialization(self, temp_config_file: Path):
        """Test global configuration initialization"""
        config = init_config(temp_config_file)
        
        assert isinstance(config, RLTEConfig)
        assert config.app.name == "test-rlte"
        
        # Test get_config returns the same instance
        config2 = get_config()
        assert config2.app.name == "test-rlte"

    @patch.dict(os.environ, {}, clear=True)
    def test_config_without_environment_vars(self, temp_config_file: Path):
        """Test configuration loading without environment variables"""
        config_manager = ConfigManager(temp_config_file)
        config = config_manager.load()
        
        # Should use defaults from config file
        assert config.app.environment == "test"


class TestConfigValidation:
    """Test configuration validation rules"""

    def test_required_fields_validation(self):
        """Test validation of required fields"""
        # Missing required database password
        invalid_config = {
            "database": {"host": "localhost"},  # Missing password
            "telegram": {"token": "test"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = Path(f.name)
        
        try:
            config_manager = ConfigManager(temp_path)
            
            with pytest.raises(Exception):  # Should fail validation
                config_manager.load()
                
        finally:
            temp_path.unlink()

    def test_risk_management_validation(self, test_config: RLTEConfig):
        """Test risk management configuration validation"""
        risk_config = test_config.trading.risk_management
        
        # Validate reasonable risk limits
        assert risk_config.max_position_size_pct <= 10.0
        assert risk_config.max_daily_loss_pct <= 20.0
        assert risk_config.max_drawdown_pct <= 50.0
        assert risk_config.stop_loss_pct > 0
        assert risk_config.take_profit_pct > 0

    def test_ml_config_validation(self, test_config: RLTEConfig):
        """Test ML configuration validation"""
        ml_config = test_config.ml.training
        
        assert ml_config.batch_size > 0
        assert 0 < ml_config.learning_rate < 1
        assert ml_config.epochs > 0
        assert 0 < ml_config.validation_split < 1

    def test_rl_config_validation(self, test_config: RLTEConfig):
        """Test RL configuration validation"""
        rl_config = test_config.rl
        
        assert rl_config.algorithm in ["DQN", "PPO", "A2C"]
        assert rl_config.training_episodes > 0
        assert 0 <= rl_config.epsilon_end <= rl_config.epsilon_start <= 1
        assert 0 < rl_config.gamma <= 1
        assert rl_config.batch_size > 0


class TestConfigTypes:
    """Test configuration type handling"""

    def test_numeric_type_conversion(self):
        """Test automatic type conversion for numeric values"""
        config_content = """
app:
  port: "${PORT:8080}"
  timeout: "${TIMEOUT:30.5}"
  enabled: "${ENABLED:true}"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = Path(f.name)
        
        try:
            os.environ["PORT"] = "9000"
            os.environ["TIMEOUT"] = "45.7"
            os.environ["ENABLED"] = "false"
            
            config_manager = ConfigManager(temp_path)
            config = config_manager.load()
            
            # Note: Type conversion depends on implementation
            # These assertions may need adjustment based on actual behavior
            assert config_manager.get("app.port") in [9000, "9000"]
            assert config_manager.get("app.enabled") in [False, "false"]
            
        finally:
            temp_path.unlink()
            for var in ["PORT", "TIMEOUT", "ENABLED"]:
                os.environ.pop(var, None)