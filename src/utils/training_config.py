"""
Training Configuration Management

This module provides functionality to load and validate training configuration
with environment variable substitution and type conversion.

Key features:
- Load training_config.yaml with environment variable substitution
- Type conversion and validation
- Environment-specific configuration overrides
- Integration with existing configuration patterns
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration data class"""
    corpus: Dict[str, Any]
    models: Dict[str, Any]
    training: Dict[str, Any]
    ensemble: Optional[Dict[str, Any]] = None
    model_persistence: Optional[Dict[str, Any]] = None
    feature_engineering: Optional[Dict[str, Any]] = None
    pipeline: Optional[Dict[str, Any]] = None
    performance_tracking: Optional[Dict[str, Any]] = None
    environments: Optional[Dict[str, Any]] = None


class TrainingConfigLoader:
    """
    Loads and validates training configuration with environment variable support
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize training config loader
        
        Args:
            config_path: Path to training_config.yaml file
        """
        if config_path is None:
            # Default to config/training_config.yaml
            self.config_path = Path(__file__).parent.parent.parent / "config" / "training_config.yaml"
        else:
            self.config_path = Path(config_path)
        
        self._config = None
        self._environment = os.getenv('ENVIRONMENT', 'development').lower()
    
    def load(self, force_reload: bool = False) -> TrainingConfig:
        """
        Load training configuration with environment variable substitution
        
        Args:
            force_reload: Force reload configuration from file
            
        Returns:
            TrainingConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValueError: If configuration is invalid
        """
        if self._config is None or force_reload:
            self._config = self._load_config()
        
        return self._config
    
    def _load_config(self) -> TrainingConfig:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Training config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            # Substitute environment variables
            resolved_config = self._resolve_environment_variables(raw_config)
            
            # Apply environment-specific overrides
            final_config = self._apply_environment_overrides(resolved_config)
            
            # Validate configuration
            self._validate_config(final_config)
            
            # Convert to dataclass
            return TrainingConfig(
                corpus=final_config['corpus'],
                models=final_config['models'],
                training=final_config['training'],
                ensemble=final_config.get('ensemble'),
                model_persistence=final_config.get('model_persistence'),
                feature_engineering=final_config.get('feature_engineering'),
                pipeline=final_config.get('pipeline'),
                performance_tracking=final_config.get('performance_tracking'),
                environments=final_config.get('environments')
            )
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in training config: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load training configuration: {e}") from e
    
    def _resolve_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve environment variables in configuration
        
        Supports patterns like:
        - ${VAR_NAME}
        - ${VAR_NAME:default_value}
        """
        if isinstance(config, dict):
            return {key: self._resolve_environment_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._resolve_environment_variables(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_vars(config)
        else:
            return config
    
    def _substitute_env_vars(self, value: str) -> Union[str, int, float, bool]:
        """
        Substitute environment variables in string value with type conversion
        """
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ''
            return os.getenv(var_name, default_value)
        
        # Replace environment variables
        resolved = re.sub(pattern, replace_var, value)
        
        # If the entire string was a single environment variable, try type conversion
        if resolved != value or (resolved.startswith('${') and resolved.endswith('}')):
            return self._convert_type(resolved)
        else:
            return resolved
    
    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        """
        Convert string value to appropriate Python type
        """
        if not isinstance(value, str):
            return value
        
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value and not value.startswith('0x'):
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            if '.' in value:
                return float(value)
        except ValueError:
            pass
        
        # Return as string if no conversion possible
        return value
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment-specific configuration overrides
        """
        if 'environments' not in config:
            return config
        
        env_overrides = config['environments'].get(self._environment, {})
        if not env_overrides:
            logger.info(f"No environment overrides found for '{self._environment}'")
            return config
        
        logger.info(f"Applying environment overrides for '{self._environment}'")
        
        # Deep merge environment overrides
        merged_config = self._deep_merge(config.copy(), env_overrides)
        
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate training configuration structure and values
        """
        # Required top-level sections
        required_sections = ['corpus', 'models', 'training']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate corpus configuration
        self._validate_corpus_config(config['corpus'])
        
        # Validate models configuration
        self._validate_models_config(config['models'])
        
        # Validate training configuration
        self._validate_training_config(config['training'])
        
        logger.info("Training configuration validation passed")
    
    def _validate_corpus_config(self, corpus_config: Dict[str, Any]) -> None:
        """Validate corpus configuration section"""
        required_fields = ['bucket', 'prefix', 'version', 'cache_ttl_hours']
        for field in required_fields:
            if field not in corpus_config:
                raise ValueError(f"Missing required corpus field: {field}")
        
        # Validate cache_ttl_hours
        cache_ttl = corpus_config['cache_ttl_hours']
        if not isinstance(cache_ttl, int) or cache_ttl <= 0:
            raise ValueError(f"cache_ttl_hours must be positive integer, got: {cache_ttl}")
        
        # Validate granularities
        if 'granularities' in corpus_config:
            valid_granularities = {'daily', 'hourly', 'four_hour', '15min', 'minute'}
            for gran in corpus_config['granularities']:
                if gran not in valid_granularities:
                    logger.warning(f"Unknown granularity: {gran}")
    
    def _validate_models_config(self, models_config: Dict[str, Any]) -> None:
        """Validate models configuration section"""
        supported_models = {'lstm', 'transformer', 'itransformer', 'patchtst', 'timesmixer', 'dqn'}
        
        for model_name, model_config in models_config.items():
            if model_name not in supported_models:
                logger.warning(f"Unknown model type: {model_name}")
            
            # Validate common fields
            if not isinstance(model_config, dict):
                raise ValueError(f"Model config for {model_name} must be a dictionary")
            
            # Check for required training parameters
            if 'batch_size' in model_config:
                batch_size = model_config['batch_size']
                if not isinstance(batch_size, int) or batch_size <= 0:
                    raise ValueError(f"{model_name}.batch_size must be positive integer")
            
            if 'epochs' in model_config:
                epochs = model_config['epochs']
                if not isinstance(epochs, int) or epochs <= 0:
                    raise ValueError(f"{model_name}.epochs must be positive integer")
            
            if 'learning_rate' in model_config:
                lr = model_config['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0:
                    raise ValueError(f"{model_name}.learning_rate must be positive number")
    
    def _validate_training_config(self, training_config: Dict[str, Any]) -> None:
        """Validate training configuration section"""
        # Validate split ratios
        if 'split_ratios' in training_config:
            ratios = training_config['split_ratios']
            if not isinstance(ratios, list) or len(ratios) != 3:
                raise ValueError("split_ratios must be a list of 3 values")
            
            for ratio in ratios:
                if not isinstance(ratio, (int, float)) or not (0 < ratio < 1):
                    raise ValueError("Each split ratio must be between 0 and 1")
            
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError(f"Split ratios must sum to 1.0, got: {sum(ratios)}")
        
        # Validate early stopping patience
        if 'early_stopping_patience' in training_config:
            patience = training_config['early_stopping_patience']
            if not isinstance(patience, int) or patience <= 0:
                raise ValueError("early_stopping_patience must be positive integer")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific model
        
        Args:
            model_name: Name of the model (lstm, transformer, etc.)
            
        Returns:
            Model configuration dictionary
            
        Raises:
            ValueError: If model not found in configuration
        """
        config = self.load()
        if model_name not in config.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return config.models[model_name]
    
    def get_corpus_config(self) -> Dict[str, Any]:
        """Get corpus loading configuration"""
        config = self.load()
        return config.corpus
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get general training configuration"""
        config = self.load()
        return config.training
    
    def get_environment(self) -> str:
        """Get current environment"""
        return self._environment


# Global training config loader instance
_training_config_loader = None


def get_training_config() -> TrainingConfig:
    """
    Get global training configuration instance
    
    Returns:
        TrainingConfig instance
    """
    global _training_config_loader
    if _training_config_loader is None:
        _training_config_loader = TrainingConfigLoader()
    return _training_config_loader.load()


def reload_training_config() -> TrainingConfig:
    """
    Reload training configuration from file
    
    Returns:
        TrainingConfig instance
    """
    global _training_config_loader
    if _training_config_loader is None:
        _training_config_loader = TrainingConfigLoader()
    return _training_config_loader.load(force_reload=True)


def init_training_config(config_path: Optional[Union[str, Path]] = None) -> TrainingConfig:
    """
    Initialize training configuration with custom path
    
    Args:
        config_path: Path to training configuration file
        
    Returns:
        TrainingConfig instance
    """
    global _training_config_loader
    _training_config_loader = TrainingConfigLoader(config_path)
    return _training_config_loader.load()


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for specific model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
    """
    config = get_training_config()
    if model_name not in config.models:
        raise ValueError(f"Model '{model_name}' not found in configuration")
    return config.models[model_name]


def get_corpus_config() -> Dict[str, Any]:
    """
    Get corpus loading configuration
    
    Returns:
        Corpus configuration dictionary
    """
    config = get_training_config()
    return config.corpus


def get_training_settings() -> Dict[str, Any]:
    """
    Get general training settings
    
    Returns:
        Training configuration dictionary
    """
    config = get_training_config()
    return config.training