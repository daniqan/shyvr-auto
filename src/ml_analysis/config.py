"""
Configuration management for market data APIs
Handles API keys, endpoints, and settings for crypto market data sources
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import structlog
from pathlib import Path
import yaml


logger = structlog.get_logger()


@dataclass
class APIConfig:
    """Configuration for a specific API"""
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    timeout: int = 30  # seconds
    cache_ttl: int = 300  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    enabled: bool = True
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class MarketDataConfig:
    """Configuration for all market data sources"""
    
    # API Configurations
    fear_greed: APIConfig = field(default_factory=lambda: APIConfig(
        base_url="https://api.alternative.me/fng/",
        rate_limit=30,  # Conservative rate limit
        cache_ttl=600,  # 10 minutes cache for sentiment data
    ))
    
    defi_llama: APIConfig = field(default_factory=lambda: APIConfig(
        base_url="https://api.llama.fi",
        rate_limit=120,  # DeFi Llama allows more requests
        cache_ttl=300,  # 5 minutes cache for DeFi data
    ))
    
    coingecko: APIConfig = field(default_factory=lambda: APIConfig(
        base_url="https://api.coingecko.com/api/v3",
        rate_limit=10,   # Free tier limit
        cache_ttl=300,   # 5 minutes cache
        headers={"Content-Type": "application/json"}
    ))
    
    coingecko_pro: APIConfig = field(default_factory=lambda: APIConfig(
        base_url="https://pro-api.coingecko.com/api/v3",
        rate_limit=500,  # Pro tier limit
        cache_ttl=300,
        headers={"Content-Type": "application/json"}
    ))
    
    # Placeholder configs for future integrations
    glassnode: APIConfig = field(default_factory=lambda: APIConfig(
        base_url="https://api.glassnode.com",
        rate_limit=600,
        cache_ttl=600,
        enabled=False  # Disabled by default until implemented
    ))
    
    messari: APIConfig = field(default_factory=lambda: APIConfig(
        base_url="https://data.messari.io/api/v1",
        rate_limit=120,
        cache_ttl=300,
        enabled=False
    ))
    
    lunarcrush: APIConfig = field(default_factory=lambda: APIConfig(
        base_url="https://api.lunarcrush.com/v2",
        rate_limit=100,
        cache_ttl=600,
        enabled=False
    ))
    
    # General settings
    default_cache_ttl: int = 300
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    enable_fallback: bool = True
    fallback_timeout: int = 5
    health_check_interval: int = 300  # seconds
    
    def get_api_config(self, api_name: str) -> Optional[APIConfig]:
        """Get configuration for a specific API"""
        return getattr(self, api_name, None)
    
    def is_api_enabled(self, api_name: str) -> bool:
        """Check if an API is enabled"""
        config = self.get_api_config(api_name)
        return config is not None and config.enabled
    
    def get_enabled_apis(self) -> Dict[str, APIConfig]:
        """Get all enabled API configurations"""
        enabled = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if isinstance(attr, APIConfig) and attr.enabled:
                    enabled[attr_name] = attr
        return enabled


class MarketDataConfigManager:
    """Manages market data configuration from multiple sources"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = structlog.get_logger().bind(component="MarketDataConfigManager")
        self.config_file = config_file
        self._config: Optional[MarketDataConfig] = None
    
    def load_config(self) -> MarketDataConfig:
        """Load configuration from multiple sources with priority"""
        if self._config is not None:
            return self._config
        
        # Start with default configuration
        config = MarketDataConfig()
        
        # Load from config file if provided
        if self.config_file and Path(self.config_file).exists():
            try:
                config = self._load_from_file(self.config_file, config)
                self.logger.info("Configuration loaded from file", file=self.config_file)
            except Exception as e:
                self.logger.warning("Failed to load config file", file=self.config_file, error=str(e))
        
        # Override with environment variables
        config = self._load_from_env(config)
        
        # Validate configuration
        config = self._validate_config(config)
        
        self._config = config
        return config
    
    def _load_from_file(self, config_file: str, base_config: MarketDataConfig) -> MarketDataConfig:
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            file_config = yaml.safe_load(f)
        
        market_data_config = file_config.get('market_data', {})
        
        # Update API configurations
        for api_name, api_config in market_data_config.get('apis', {}).items():
            if hasattr(base_config, api_name):
                current_config = getattr(base_config, api_name)
                
                # Update API config fields
                for key, value in api_config.items():
                    if hasattr(current_config, key):
                        setattr(current_config, key, value)
        
        # Update general settings
        general_settings = market_data_config.get('settings', {})
        for key, value in general_settings.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
        
        return base_config
    
    def _load_from_env(self, config: MarketDataConfig) -> MarketDataConfig:
        """Load configuration from environment variables"""
        
        # API keys from environment
        env_mappings = {
            'COINGECKO_API_KEY': ('coingecko', 'api_key'),
            'COINGECKO_PRO_API_KEY': ('coingecko_pro', 'api_key'),
            'GLASSNODE_API_KEY': ('glassnode', 'api_key'),
            'MESSARI_API_KEY': ('messari', 'api_key'),
            'LUNARCRUSH_API_KEY': ('lunarcrush', 'api_key'),
        }
        
        for env_var, (api_name, field_name) in env_mappings.items():
            value = os.getenv(env_var)
            if value and hasattr(config, api_name):
                api_config = getattr(config, api_name)
                setattr(api_config, field_name, value)
                
                # Enable API if key is provided
                if field_name == 'api_key' and value:
                    api_config.enabled = True
                    self.logger.info("API key loaded from environment", api=api_name)
        
        # General settings from environment
        env_settings = {
            'MARKET_DATA_CACHE_TTL': ('default_cache_ttl', int),
            'MARKET_DATA_TIMEOUT': ('request_timeout', int),
            'MARKET_DATA_MAX_CONCURRENT': ('max_concurrent_requests', int),
            'MARKET_DATA_ENABLE_FALLBACK': ('enable_fallback', lambda x: x.lower() == 'true'),
        }
        
        for env_var, (field_name, converter) in env_settings.items():
            value = os.getenv(env_var)
            if value:
                try:
                    converted_value = converter(value)
                    setattr(config, field_name, converted_value)
                    self.logger.debug("Setting loaded from environment", 
                                    setting=field_name, 
                                    value=converted_value)
                except (ValueError, TypeError) as e:
                    self.logger.warning("Invalid environment variable value", 
                                      env_var=env_var, 
                                      value=value, 
                                      error=str(e))
        
        # Special handling for CoinGecko Pro
        if config.coingecko_pro.api_key:
            # Use Pro version if API key is available
            config.coingecko.enabled = False
            config.coingecko_pro.enabled = True
            # Set Pro API key header
            config.coingecko_pro.headers["X-CG-Pro-API-Key"] = config.coingecko_pro.api_key
        
        return config
    
    def _validate_config(self, config: MarketDataConfig) -> MarketDataConfig:
        """Validate and sanitize configuration"""
        
        # Ensure at least one API is enabled
        enabled_apis = config.get_enabled_apis()
        if not enabled_apis:
            self.logger.warning("No APIs enabled, enabling Fear & Greed Index as fallback")
            config.fear_greed.enabled = True
        
        # Validate rate limits
        for api_name, api_config in enabled_apis.items():
            if api_config.rate_limit <= 0:
                self.logger.warning("Invalid rate limit, setting to default", 
                                  api=api_name, 
                                  rate_limit=api_config.rate_limit)
                api_config.rate_limit = 60
            
            if api_config.cache_ttl < 0:
                api_config.cache_ttl = config.default_cache_ttl
            
            if api_config.timeout <= 0:
                api_config.timeout = config.request_timeout
        
        # Validate general settings
        if config.default_cache_ttl <= 0:
            config.default_cache_ttl = 300
        
        if config.max_concurrent_requests <= 0:
            config.max_concurrent_requests = 10
        
        if config.request_timeout <= 0:
            config.request_timeout = 30
        
        return config
    
    def get_config(self) -> MarketDataConfig:
        """Get the current configuration"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> MarketDataConfig:
        """Reload configuration from sources"""
        self._config = None
        return self.load_config()
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """Save current configuration to file"""
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        file_path = config_file or self.config_file
        if not file_path:
            raise ValueError("No config file specified")
        
        # Convert config to dict for YAML serialization
        config_dict = self._config_to_dict(self._config)
        
        with open(file_path, 'w') as f:
            yaml.dump({'market_data': config_dict}, f, default_flow_style=False, indent=2)
        
        self.logger.info("Configuration saved to file", file=file_path)
    
    def _config_to_dict(self, config: MarketDataConfig) -> Dict[str, Any]:
        """Convert MarketDataConfig to dictionary for serialization"""
        config_dict = {
            'apis': {},
            'settings': {
                'default_cache_ttl': config.default_cache_ttl,
                'max_concurrent_requests': config.max_concurrent_requests,
                'request_timeout': config.request_timeout,
                'enable_fallback': config.enable_fallback,
                'fallback_timeout': config.fallback_timeout,
                'health_check_interval': config.health_check_interval,
            }
        }
        
        # Add API configurations
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                attr = getattr(config, attr_name)
                if isinstance(attr, APIConfig):
                    config_dict['apis'][attr_name] = {
                        'base_url': attr.base_url,
                        'rate_limit': attr.rate_limit,
                        'timeout': attr.timeout,
                        'cache_ttl': attr.cache_ttl,
                        'max_retries': attr.max_retries,
                        'retry_delay': attr.retry_delay,
                        'enabled': attr.enabled,
                        'headers': attr.headers,
                        # Don't save API keys to file for security
                    }
        
        return config_dict
    
    def get_api_summary(self) -> Dict[str, Any]:
        """Get a summary of API configurations for logging/debugging"""
        if self._config is None:
            self.load_config()
        
        enabled_apis = self._config.get_enabled_apis()
        
        return {
            'enabled_apis': list(enabled_apis.keys()),
            'total_apis': len([attr for attr in dir(self._config) 
                             if isinstance(getattr(self._config, attr), APIConfig)]),
            'config_file': self.config_file,
            'default_cache_ttl': self._config.default_cache_ttl,
            'fallback_enabled': self._config.enable_fallback,
        }


# Global config manager instance
_config_manager: Optional[MarketDataConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> MarketDataConfigManager:
    """Get or create the global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = MarketDataConfigManager(config_file)
    return _config_manager


def get_market_data_config() -> MarketDataConfig:
    """Get the current market data configuration"""
    return get_config_manager().get_config()


def reload_market_data_config() -> MarketDataConfig:
    """Reload market data configuration from sources"""
    return get_config_manager().reload_config()