"""
Configuration management for the RLTE system
Handles loading and validation of configuration from YAML files and environment variables
"""

import logging
import os
import re
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from .base import ConfigurationError
from .security.config_encryption import ConfigEncryption
from .security.config_auditor import ConfigAuditor

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration"""

    host: str = "localhost"
    port: int = 5432
    database: str = "shyvr_rlte_prod"
    username: str = "rlte_user"
    password: str = ""  # Default empty, will be loaded from env or Secret Manager
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    def __init__(self, **data):
        """Initialize DatabaseConfig with password from Secret Manager"""
        # Only populate from Secret Manager if password not already provided
        if not data.get('password'):
            from .system_secrets import get_system_secrets
            
            system_secrets = get_system_secrets()
            db_config = system_secrets.get_database_config()
            
            # Use values from Secret Manager if not already set
            if db_config.get('password'):
                data['password'] = db_config['password']
            if not data.get('host') and db_config.get('host'):
                data['host'] = db_config['host']
            if not data.get('port') and db_config.get('port'):
                data['port'] = int(db_config['port'])
            if not data.get('username') and db_config.get('username'):
                data['username'] = db_config['username']
            if not data.get('database') and db_config.get('database'):
                data['database'] = db_config['database']
            
            # If still no password, check environment for development
            if not data.get('password'):
                import os
                if os.environ.get('ENVIRONMENT') in ['production', 'staging']:
                    raise ValueError("Database password is required in production/staging")
                else:
                    # Development default
                    data['password'] = 'DeFi2024_Secure'
                
        super().__init__(**data)

    @property
    def url(self) -> str:
        """Generate database URL"""
        return (
            f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        )




class APIConfig(BaseModel):
    """Generic API configuration"""

    api_key: str | None = None
    base_url: str
    rate_limit: int = 60
    timeout: int = 30


class TelegramConfig(BaseModel):
    """Telegram bot configuration"""

    token: str
    webhook_secret: str | None = None
    admin_users: list[int] = field(default_factory=list)
    rate_limit: int = 30


class AgentConfig(BaseModel):
    """Agent framework configuration"""

    model_type: str = "local"
    model_name: str = "gpt-4o-mini"
    api_key: str | None = None
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

    modes: dict[str, bool] = {"analysis": True, "simulation": True, "live": False}
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)


class MLConfig(BaseModel):
    """Machine learning configuration"""

    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10


class RLExperiencePerformanceConfig(BaseModel):
    """RL experience storage performance configuration"""
    
    cache_size: int = 1000
    async_operations: bool = True
    compression: bool = False
    query_timeout_seconds: int = 30
    batch_commit_size: int = 100
    
    @model_validator(mode='after')
    def validate_performance_config(self):
        if self.cache_size <= 0:
            raise ValueError("Cache size must be positive")
        return self


class RLExperienceDatabaseConfig(BaseModel):
    """RL experience database configuration"""
    
    pool_size: int = 5
    max_overflow: int = 10
    timeout_seconds: int = 30
    enable_query_logging: bool = False
    connection_retry_attempts: int = 3


class RLExperienceLifecycleConfig(BaseModel):
    """RL experience lifecycle management configuration"""
    
    cleanup_enabled: bool = True
    max_age_days: int = 30
    cleanup_interval_hours: int = 24
    archive_old_experiences: bool = False
    min_experiences_to_keep: int = 1000


class RLExperienceStorageConfig(BaseModel):
    """RL experience storage configuration"""
    
    enabled: bool = True
    storage_backend: str = "database"
    max_experiences: int = 50000
    batch_size: int = 64
    prioritized_replay: bool = True
    
    performance: RLExperiencePerformanceConfig = field(default_factory=RLExperiencePerformanceConfig)
    database: RLExperienceDatabaseConfig = field(default_factory=RLExperienceDatabaseConfig)
    lifecycle: RLExperienceLifecycleConfig = field(default_factory=RLExperienceLifecycleConfig)
    
    @model_validator(mode='after')
    def validate_storage_config(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_experiences <= 0:
            raise ValueError("Max experiences must be positive")
        return self


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
    
    experience_storage: RLExperienceStorageConfig = field(default_factory=RLExperienceStorageConfig)


class ModelPreservationConfig(BaseModel):
    """Model preservation configuration"""
    
    enabled: bool = True
    gcs_bucket: str | None = None
    backup_interval_hours: int = 6
    max_versions_per_model: int = 10
    enable_compression: bool = True
    mode_isolation: bool = True
    
    @model_validator(mode='after')
    def validate_preservation_config(self):
        """Validate model preservation configuration"""
        if self.enabled and not self.gcs_bucket:
            raise ValueError("GCS bucket is required when model preservation is enabled")
        
        if self.backup_interval_hours <= 0:
            raise ValueError("Backup interval must be positive")
            
        if self.max_versions_per_model <= 0:
            raise ValueError("Max versions per model must be positive")
            
        return self


class MonitoringAlertsConfig(BaseModel):
    """Monitoring alerts configuration"""
    
    enabled: bool = True
    webhook_url: str | None = None
    critical_threshold: float = 0.95
    warning_threshold: float = 0.8


class MonitoringGrafanaConfig(BaseModel):
    """Grafana monitoring configuration"""
    
    enabled: bool = True
    port: int = 3000
    admin_password: str | None = None


class MonitoringPrometheusConfig(BaseModel):
    """Prometheus monitoring configuration"""
    
    enabled: bool = True
    port: int = 9090
    retention_days: int = 30


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    
    enabled: bool = True
    metrics_port: int = 9090
    health_check_port: int = 8081
    log_level: str = "INFO"
    alerts: MonitoringAlertsConfig = field(default_factory=MonitoringAlertsConfig)
    grafana: MonitoringGrafanaConfig = field(default_factory=MonitoringGrafanaConfig)
    prometheus: MonitoringPrometheusConfig = field(default_factory=MonitoringPrometheusConfig)


class FeaturesConfig(BaseModel):
    """Feature flags configuration"""
    
    live_trading: bool = False
    paper_trading: bool = True
    ml_predictions: bool = True
    rl_decisions: bool = True
    risk_management: bool = True
    position_monitoring: bool = True
    automated_stops: bool = True
    emergency_shutdown: bool = True


class SecurityConfig(BaseModel):
    """Security configuration"""

    secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    api_key_length: int = 32
    rate_limit_storage: str = "memory"

    def __init__(self, **data):
        """Initialize SecurityConfig with secret_key from Secret Manager if needed"""
        # Only populate from Secret Manager if secret_key not already provided or is empty
        if not data.get('secret_key') or data.get('secret_key') == '' or data.get('secret_key') == 'NOT_SET':
            from .system_secrets import get_system_secrets

            try:
                system_secrets = get_system_secrets()
                secret_key = system_secrets.secret_key

                if secret_key:
                    data['secret_key'] = secret_key
                else:
                    # Fallback for development
                    import os
                    if os.environ.get('ENVIRONMENT') in ['production', 'staging']:
                        raise ValueError("SECRET_KEY is required in production/staging but not found in Secret Manager")
                    else:
                        # Development fallback
                        data['secret_key'] = 'dev_secret_key_for_development_16chars'
            except Exception as e:
                # Fallback for development if Secret Manager unavailable
                import os
                if os.environ.get('ENVIRONMENT') in ['production', 'staging']:
                    raise ValueError(f"Failed to retrieve SECRET_KEY from Secret Manager in production: {e}")
                else:
                    data['secret_key'] = 'dev_secret_key_for_development_16chars'

        super().__init__(**data)

    @model_validator(mode='after')
    def validate_security_config(self):
        """Validate security configuration"""
        # Check for default secrets in production
        if hasattr(self, '_environment') and self._environment == 'production':
            if 'dev-key' in self.secret_key.lower() or self.secret_key == 'dev-key-change-in-prod':
                raise ValueError("Default secret key detected in production configuration")
        
        # Get environment context
        environment = getattr(self, '_environment', 'development')
        
        # Validate secret key length (minimum 32 characters for production/staging, 16 for development)
        min_length = 32 if environment in ['production', 'staging'] else 16
        if len(self.secret_key) < min_length:
            if environment in ['production', 'staging']:
                raise ValueError("Secret key must be at least 32 characters long in production/staging")
            elif len(self.secret_key) < 16:
                raise ValueError("Secret key must be at least 16 characters long")
        
        # Validate JWT algorithm
        allowed_algorithms = ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512', 'ES256', 'ES384', 'ES512']
        if self.jwt_algorithm not in allowed_algorithms:
            raise ValueError(f"JWT algorithm '{self.jwt_algorithm}' is not secure. Use one of: {allowed_algorithms}")
        
        # Deprecated/weak algorithms
        weak_algorithms = ['none', 'HS1', 'RS1']
        if self.jwt_algorithm in weak_algorithms:
            raise ValueError(f"JWT algorithm '{self.jwt_algorithm}' is deprecated and insecure")
        
        return self


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
    telegram: TelegramConfig
    security: SecurityConfig
    agent: AgentConfig = field(default_factory=AgentConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    apis: dict[str, APIConfig] = field(default_factory=dict)
    model_preservation: ModelPreservationConfig | None = None
    monitoring: MonitoringConfig | None = None
    features: FeaturesConfig | None = None

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow"  # Allow extra fields for flexibility
    )
    
    @model_validator(mode='after')
    def validate_production_security(self):
        """Validate production security requirements"""
        # Set environment context for security validation
        if self.security:
            self.security._environment = self.app.environment
            
        if self.app.environment in ['production', 'staging']:
            # Security config is required in production/staging
            if not self.security:
                raise ValueError(f"Security configuration is required in {self.app.environment} environment")
            
            # Re-validate security config with environment context
            self.security.validate_security_config()
        
        return self


class ConfigManager:
    """Configuration manager with environment variable substitution"""

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            self.config_path = self._find_config_file()
        else:
            self.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        self._config: RLTEConfig | None = None
        self._required_prod_vars = [
            'DB_HOST', 'DB_PASSWORD', 'SECRET_KEY', 'TELEGRAM_TOKEN',
            'HELIUS_API_KEY', 'ETHERSCAN_API_KEY', 'BIRDEYE_API_KEY'
        ]

    def _find_config_file(self) -> Path:
        """Find configuration file in standard locations with environment-specific support"""
        environment = os.environ.get('ENVIRONMENT', 'development')
        
        # Environment-specific config file paths
        env_specific_paths = [
            Path(f"config/config.{environment}.yaml"),
            Path(f"config.{environment}.yaml"),
            Path(f"/app/config/config.{environment}.yaml"),
        ]
        
        # Check for environment-specific config first
        for path in env_specific_paths:
            if path.exists():
                logger.info(f"Using environment-specific config: {path}")
                return path
        
        # Fallback to standard config file locations
        standard_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path("/app/config/config.yaml"),
            Path.home() / ".config" / "shyvr-rlte" / "config.yaml",
        ]

        for path in standard_paths:
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
            # Check for encrypted values and decrypt them
            if obj.startswith('encrypted:'):
                if not hasattr(self, '_encryptor'):
                    self._encryptor = ConfigEncryption()
                return self._encryptor.decrypt_value(obj)
            return self._expand_env_vars(obj)
        else:
            return obj

    def _expand_env_vars(self, value: str) -> Any:
        """Expand environment variables in string values"""
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            env_value = os.environ.get(var_name, default_value)
            # Always return string for regex substitution
            return env_value

        result = re.sub(pattern, replace_var, value)

        # If the entire string was a single variable, try to convert to appropriate type
        if re.match(r"^\$\{[^}:]+(?::[^}]*)?\}$", value):
            # This was a single variable reference, try type conversion
            if result.lower() in ("true", "false"):
                return result.lower() == "true"
            elif result.isdigit():
                return int(result)
            elif self._is_float(result):
                return float(result)

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

            with open(self.config_path) as f:
                raw_config = yaml.safe_load(f)

            # Substitute environment variables
            processed_config = self._substitute_env_vars(raw_config)
            
            # Get environment for validation
            environment = processed_config.get('app', {}).get('environment', 'development')
            
            # Validate production requirements before parsing
            if environment == 'production':
                self._validate_production_requirements(processed_config)
            elif environment == 'staging':
                self._validate_staging_requirements(processed_config)

            # Parse with Pydantic
            self._config = RLTEConfig(**processed_config)
            
            # Post-load validation
            if environment in ['production', 'staging']:
                self._validate_production_constraints(self._config)
                self._validate_no_localhost_references(self._config)

            logger.info(f"Configuration loaded from {self.config_path} (environment: {environment})")
            return self._config

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e

    def reload(self) -> RLTEConfig:
        """Reload configuration from file"""
        self._config = None
        return self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        config = self.load()
        keys = key.split(".")

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

    def _validate_production_requirements(self, config: dict) -> None:
        """Validate production-specific requirements"""
        logger.info("Validating production configuration requirements")
        
        # Check required environment variables
        missing_vars = []
        for var in self._required_prod_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ConfigurationError(
                f"Required production environment variables missing: {', '.join(missing_vars)}"
            )
        
        # Validate critical configuration sections exist
        required_sections = ['app', 'database', 'security', 'telegram']
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Required configuration section missing: {section}")
        
        # Validate APIs section has required APIs
        if 'apis' not in config or not config['apis']:
            raise ConfigurationError("APIs configuration is required in production")
        
        required_apis = ['helius', 'etherscan', 'birdeye']
        apis = config.get('apis', {})
        missing_apis = [api for api in required_apis if api not in apis]
        if missing_apis:
            raise ConfigurationError(
                f"Required API configurations missing: {', '.join(missing_apis)}"
            )
    
    def _validate_staging_requirements(self, config: dict) -> None:
        """Validate staging-specific requirements (similar to production)"""
        logger.info("Validating staging configuration requirements")
        
        # Staging uses similar validation to production
        self._validate_production_requirements(config)
    
    def _validate_production_constraints(self, config: RLTEConfig) -> None:
        """Validate production configuration constraints"""
        logger.info("Validating production configuration constraints")
        
        # Validate database configuration
        if config.database.pool_size <= 0:
            raise ConfigurationError("Database pool size must be positive")
        
        if config.database.max_overflow < 0:
            raise ConfigurationError("Database max overflow cannot be negative")
        
        # Validate risk management constraints
        risk_config = config.trading.risk_management
        
        if risk_config.max_position_size_pct <= 0 or risk_config.max_position_size_pct > 100:
            raise ConfigurationError(
                "Max position size must be between 0 and 100 percent"
            )
        
        if risk_config.max_daily_loss_pct <= 0 or risk_config.max_daily_loss_pct > 100:
            raise ConfigurationError(
                "Max daily loss must be between 0 and 100 percent"
            )
        
        if risk_config.max_open_positions <= 0:
            raise ConfigurationError("Max open positions must be positive")
        
        if risk_config.min_trade_amount_usd <= 0:
            raise ConfigurationError("Minimum trade amount must be positive")
    
    def _validate_no_localhost_references(self, config: RLTEConfig) -> None:
        """Validate no localhost references in production/staging"""
        logger.info("Validating no localhost references in production configuration")
        
        # Check database host
        if 'localhost' in config.database.host.lower():
            raise ConfigurationError(
                "Localhost database host not allowed in production environment"
            )
        
        # Check API base URLs
        for api_name, api_config in config.apis.items():
            if hasattr(api_config, 'base_url') and 'localhost' in api_config.base_url.lower():
                raise ConfigurationError(
                    f"Localhost reference in {api_name} API base URL not allowed in production"
                )
    
    def validate_health_checks(self) -> bool:
        """Validate configuration health checks"""
        logger.info("Running configuration health checks")
        
        try:
            config = self.load()
            
            # Database connection health check would go here
            # For now, just validate configuration is loadable
            if not config:
                raise ConfigurationError("Configuration failed to load")
            
            # TODO: Add actual database connection test
            # TODO: Add API endpoint connectivity tests
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise ConfigurationError(f"Configuration health check failed: {e}")
    
    def validate_api_health(self) -> bool:
        """Validate API endpoint health"""
        logger.info("Running API health checks")
        
        try:
            config = self.load()
            
            # API health checks would go here
            # For now, just validate APIs are configured
            if not config.apis:
                raise ConfigurationError("No APIs configured")
            
            # TODO: Add actual API connectivity tests
            
            return True
            
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            raise ConfigurationError(f"API health check failed: {e}")
    
    def validate_production_readiness(self) -> bool:
        """Validate complete production readiness"""
        logger.info("Running complete production readiness validation")
        
        try:
            # Load and validate configuration
            config = self.load()
            
            # Run all validation checks
            basic_validation = self.validate()
            health_checks = self.validate_health_checks()
            api_health = self.validate_api_health()
            
            if not all([basic_validation, health_checks, api_health]):
                raise ConfigurationError("Production readiness validation failed")
            
            logger.info("Production readiness validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Production readiness validation failed: {e}")
            raise ConfigurationError(f"Production readiness validation failed: {e}")
    
    def add_runtime_validator(self, key: str, validator: Any) -> None:
        """Add runtime validation hook for configuration key"""
        raise NotImplementedError("Runtime validation hooks not implemented yet")
    
    def enable_change_detection(self) -> None:
        """Enable configuration change detection"""
        raise NotImplementedError("Configuration change detection not implemented yet")
    
    def detect_sensitive_values(self) -> list[str]:
        """Detect sensitive values in configuration"""
        config = self.load()
        config_dict = self._config_to_dict(config)
        
        if not hasattr(self, '_encryptor'):
            self._encryptor = ConfigEncryption()
        
        sensitive_keys = self._encryptor.get_sensitive_keys_in_config(config_dict)
        
        if sensitive_keys:
            logger.warning(f"Found {len(sensitive_keys)} sensitive configuration keys: {sensitive_keys}")
        else:
            logger.info("No sensitive configuration keys detected")
        
        return sensitive_keys
    
    def _config_to_dict(self, config: 'RLTEConfig') -> dict:
        """Convert RLTEConfig to dictionary for processing"""
        return config.model_dump()
    
    def enable_audit_logging(self) -> None:
        """Enable configuration audit logging"""
        if not hasattr(self, '_auditor'):
            self._auditor = ConfigAuditor()
            logger.info("Configuration audit logging enabled")
        else:
            logger.info("Configuration audit logging already enabled")
    
    def log_config_access(self, key: str, value: Any = None, user: str = "system") -> None:
        """Log configuration access if auditing is enabled"""
        if hasattr(self, '_auditor'):
            self._auditor.log_config_access(key, value, user)
    
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
            if config.app.environment not in ["development", "staging", "production"]:
                logger.warning(f"Unknown environment: {config.app.environment}")

            # Validate risk management
            if config.trading.risk_management.max_position_size_pct > 10.0:
                logger.warning("Maximum position size is very high (>10%)")

            # Validate security configuration
            self._validate_security_config(config)
            
            # Validate RL experience storage
            self._validate_rl_experience_config(config)
            
            # Validate model preservation
            self._validate_model_preservation_config(config)

            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def _validate_rl_experience_config(self, config: 'RLTEConfig') -> None:
        """Validate RL experience storage configuration"""
        exp_config = config.rl.experience_storage
        
        # Warn about potentially problematic values
        if exp_config.max_experiences > 100000:
            logger.warning("Very large max_experiences value (>100k) may impact performance")
        
        if exp_config.batch_size == 1:
            logger.warning("Batch size of 1 may be inefficient for database operations")
            
        if exp_config.database.pool_size > 50:
            logger.warning("Very large database pool size (>50) may be excessive")
            
        if not exp_config.enabled:
            logger.info("RL experience storage is disabled")
            
        if exp_config.storage_backend not in ["database", "memory"]:
            logger.warning(f"Unknown storage backend: {exp_config.storage_backend}")

    def _validate_model_preservation_config(self, config: 'RLTEConfig') -> None:
        """Validate model preservation configuration"""
        if config.model_preservation is None:
            logger.info("Model preservation is not configured")
            return
            
        preservation_config = config.model_preservation
        
        if preservation_config.enabled:
            if not preservation_config.gcs_bucket:
                logger.error("GCS bucket is required when model preservation is enabled")
            else:
                logger.info(f"Model preservation enabled with bucket: {preservation_config.gcs_bucket}")
                
            if preservation_config.backup_interval_hours < 1:
                logger.warning("Very frequent backup interval (<1 hour) may impact performance")
            elif preservation_config.backup_interval_hours > 24:
                logger.warning("Very infrequent backup interval (>24 hours) may increase data loss risk")
                
            if preservation_config.max_versions_per_model > 50:
                logger.warning("Very high max versions (>50) may increase storage costs")
            elif preservation_config.max_versions_per_model < 3:
                logger.warning("Very low max versions (<3) may limit rollback options")
        else:
            logger.info("Model preservation is disabled")

    def _validate_security_config(self, config: 'RLTEConfig') -> None:
        """Validate security configuration"""
        if not hasattr(config, 'security') or config.security is None:
            if config.app.environment == 'production':
                logger.error("Security configuration is required in production environment")
                raise ConfigurationError("Security configuration is missing in production")
            else:
                logger.warning("Security configuration is not configured")
                return
        
        security_config = config.security
        
        # Check for default secrets
        if 'dev-key' in security_config.secret_key.lower():
            if config.app.environment == 'production':
                logger.error("Default secret key detected in production")
                raise ConfigurationError("Default secret key 'dev-key-change-in-prod' is not allowed in production")
            else:
                logger.warning("Using default secret key in development environment")
        
        # Validate secret key strength
        if len(security_config.secret_key) < 32:
            logger.error(f"Secret key is too short ({len(security_config.secret_key)} chars). Minimum 32 characters required.")
            raise ConfigurationError("Secret key must be at least 32 characters long")
        
        # Check if secret comes from environment variable
        if security_config.secret_key == 'dev-key-change-in-prod':
            logger.error("Secret key is using default value")
            if config.app.environment == 'production':
                raise ConfigurationError("Default secret key is not allowed in production")
        
        # Validate JWT configuration
        if security_config.jwt_algorithm not in ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512']:
            logger.error(f"Insecure JWT algorithm: {security_config.jwt_algorithm}")
            raise ConfigurationError(f"JWT algorithm '{security_config.jwt_algorithm}' is not secure")
        
        if security_config.jwt_expiry_hours > 24:
            logger.warning("JWT expiry time is very long (>24 hours)")
        elif security_config.jwt_expiry_hours < 1:
            logger.warning("JWT expiry time is very short (<1 hour)")
        
        logger.info("Security configuration validated successfully")


# Global configuration instance
_config_manager: ConfigManager | None = None


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


def init_config(config_path: str | Path | None = None) -> RLTEConfig:
    """Initialize configuration with custom path"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager.load()


# Convenience functions for common config access
def get_database_url() -> str:
    """Get database connection URL"""
    return get_config().database.url


def get_api_config(api_name: str) -> APIConfig | None:
    """Get API configuration by name"""
    return get_config().apis.get(api_name)


def is_production() -> bool:
    """Check if running in production environment"""
    return get_config().app.environment == "production"


def is_development() -> bool:
    """Check if running in development environment"""
    return get_config().app.environment == "development"
