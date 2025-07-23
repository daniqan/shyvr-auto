"""
Configuration management for the RLTE system
Handles loading and validation of configuration from YAML files and environment variables
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, validator, Field

from .base import ConfigurationError

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "shyvr_rlte"
    username: str = "rlte_user"
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    @property
    def url(self) -> str:
        """Generate database URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None


class APIConfig(BaseModel):
    """Generic API configuration"""
    api_key: Optional[str] = None
    base_url: str
    rate_limit: int = 60
    timeout: int = 30


class TelegramConfig(BaseModel):
    """Telegram bot configuration"""
    token: str
    webhook_secret: Optional[str] = None
    admin_users: list[int] = field(default_factory=list)
    rate_limit: int = 30


class AgentConfig(BaseModel):
    """Agent framework configuration"""
    model_type: str = "local"
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.1
    max_active_rules: int = 10
    rule_expiry_hours: int = 24


class RiskManagementConfig(BaseModel):
    """Risk management configuration"""
    max_position_size_pct: float = 1.0
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 15.0
    stop_loss_pct: float = 8.0
    take_profit_pct: float = 40.0
    max_open_positions: int = 5
    min_trade_amount_usd: float = 10.0


class TradingConfig(BaseModel):
    """Trading configuration"""
    modes: Dict[str, bool] = {
        "analysis": True,
        "simulation": True,
        "live": False
    }
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)


class MLConfig(BaseModel):
    """Machine learning configuration"""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10


class RLConfig(BaseModel):
    """Reinforcement learning configuration"""
    algorithm: str = "DQN"
    training_episodes: int = 10000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    learning_rate: float = 0.0001
    gamma: float = 0.99
    batch_size: int = 64
    memory_size: int = 50000
    target_update_freq: int = 1000


class AppConfig(BaseModel):
    """Application configuration"""
    name: str = "shyvr-rlte"
    version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    debug: bool = False


class RLTEConfig(BaseModel):
    """Main RLTE configuration"""
    app: AppConfig = field(default_factory=AppConfig)
    database: DatabaseConfig
    redis: RedisConfig = field(default_factory=RedisConfig)
    telegram: TelegramConfig
    agent: AgentConfig = field(default_factory=AgentConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    apis: Dict[str, APIConfig] = field(default_factory=dict)
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "allow"  # Allow extra fields for flexibility


class ConfigManager:
    """Configuration manager with environment variable substitution"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[RLTEConfig] = None
        
    def _find_config_file(self) -> Path:
        """Find configuration file in standard locations"""
        possible_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path("/app/config/config.yaml"),
            Path.home() / ".config" / "shyvr-rlte" / "config.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # Default to config/config.yaml
        return Path("config/config.yaml")
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._expand_env_vars(obj)
        else:
            return obj
    
    def _expand_env_vars(self, value: str) -> Any:
        """Expand environment variables in string values"""
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            env_value = os.environ.get(var_name, default_value)
            
            # Try to convert to appropriate type
            if env_value.lower() in ('true', 'false'):
                return env_value.lower() == 'true'
            elif env_value.isdigit():
                return int(env_value)
            elif self._is_float(env_value):
                return float(env_value)
            else:
                return env_value
        
        result = re.sub(pattern, replace_var, value)
        
        # If the entire string was a variable, return the converted value
        if result != value and isinstance(result, (bool, int, float)):
            return result
        
        return result
    
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def load(self) -> RLTEConfig:
        """Load configuration from file with environment variable substitution"""
        if self._config is not None:
            return self._config
            
        try:
            if not self.config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            # Substitute environment variables
            processed_config = self._substitute_env_vars(raw_config)
            
            # Parse with Pydantic
            self._config = RLTEConfig(**processed_config)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return self._config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def reload(self) -> RLTEConfig:
        """Reload configuration from file"""
        self._config = None
        return self.load()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        config = self.load()
        keys = key.split('.')
        
        try:
            value = config
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            config = self.load()
            
            # Check required fields
            if not config.database.password:
                logger.warning("Database password not configured")
                
            if not config.telegram.token:
                logger.warning("Telegram token not configured")
            
            # Validate environment
            if config.app.environment not in ['development', 'staging', 'production']:
                logger.warning(f"Unknown environment: {config.app.environment}")
            
            # Validate risk management
            if config.trading.risk_management.max_position_size_pct > 10.0:
                logger.warning("Maximum position size is very high (>10%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> RLTEConfig:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.load()


def reload_config() -> RLTEConfig:
    """Reload global configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.reload()


def init_config(config_path: Optional[Union[str, Path]] = None) -> RLTEConfig:
    """Initialize configuration with custom path"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager.load()


# Convenience functions for common config access
def get_database_url() -> str:
    """Get database connection URL"""
    return get_config().database.url


def get_api_config(api_name: str) -> Optional[APIConfig]:
    """Get API configuration by name"""
    return get_config().apis.get(api_name)


def is_production() -> bool:
    """Check if running in production environment"""
    return get_config().app.environment == "production"


def is_development() -> bool:
    """Check if running in development environment"""
    return get_config().app.environment == "development"