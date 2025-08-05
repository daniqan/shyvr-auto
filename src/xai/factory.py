"""
Factory pattern for creating XAI explainers.

This module provides a factory class for creating different types of explainers
based on configuration, with support for dynamic registration and validation.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Type
import logging
from .base import BaseExplainer, ModelAgnosticExplainer, GradientBasedExplainer

logger = logging.getLogger(__name__)


class ExplainerFactory:
    """
    Factory class for creating XAI explainers.
    
    Supports dynamic registration of explainer types and provides
    unified interface for creating explainers with validation.
    """
    
    def __init__(self, cache_explainers: bool = False):
        """
        Initialize explainer factory.
        
        Args:
            cache_explainers: Whether to cache created explainers
        """
        self.cache_explainers = cache_explainers
        self._explainer_cache: Dict[str, BaseExplainer] = {}
        self._explainer_registry: Dict[str, Type[BaseExplainer]] = {}
        self._explainer_info: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in explainers
        self._register_builtin_explainers()
    
    def _register_builtin_explainers(self) -> None:
        """Register built-in explainer types."""
        # For now, register placeholder info for expected explainer types
        # These will be replaced with actual implementations
        
        self._explainer_info['lime'] = {
            'name': 'LIME (Local Interpretable Model-agnostic Explanations)',
            'description': 'Explains individual predictions by approximating the model locally with an interpretable model',
            'supported_model_types': ['sklearn', 'torch', 'lightgbm', 'custom'],
            'default_config': {
                'num_samples': 1000,
                'kernel_width': 0.75,
                'distance_metric': 'euclidean',
                'feature_selection': 'auto'
            }
        }
        
        self._explainer_info['permutation'] = {
            'name': 'Permutation Feature Importance',
            'description': 'Measures feature importance by evaluating prediction change when feature values are permuted',
            'supported_model_types': ['sklearn', 'torch', 'lightgbm', 'custom'],
            'default_config': {
                'num_permutations': 100,
                'scoring_metric': 'accuracy',
                'random_state': 42
            }
        }
        
        self._explainer_info['gradient'] = {
            'name': 'Gradient-based Attribution',
            'description': 'Computes feature attribution using gradients of model output with respect to input features',
            'supported_model_types': ['torch', 'tensorflow'],
            'default_config': {
                'baseline': 'zero',
                'steps': 50,
                'method': 'integrated_gradients'
            }
        }
        
        self._explainer_info['attention'] = {
            'name': 'Attention Weight Analysis',
            'description': 'Explains predictions by analyzing attention weights from transformer models',
            'supported_model_types': ['iTransformer', 'PatchTST', 'TimesMixer', 'TimesFM', 'TransformerPredictor'],
            'default_config': {
                'extract_attention': True,
                'attention_heads': 'all',
                'attention_layers': 'last',
                'attention_rollout': True,
                'normalize_attention': True,
                'performance_mode': True
            }
        }
        
        self._explainer_info['temporal_attention'] = {
            'name': 'Temporal Attention Analysis',
            'description': 'Analyzes temporal patterns in attention weights for time-series forecasting',
            'supported_model_types': ['iTransformer', 'PatchTST', 'TimesMixer', 'TimesFM', 'TransformerPredictor'],
            'default_config': {
                'analyze_recency_bias': True,
                'detect_periodic_patterns': True,
                'identify_regime_transitions': True,
                'time_window_analysis': True,
                'seasonal_attention_patterns': True,
                'min_period': 2,
                'max_period': 168,
                'regime_threshold': 0.3
            }
        }
        
        self._explainer_info['cross_attention'] = {
            'name': 'Cross-Asset Attention Analysis',
            'description': 'Analyzes cross-attention patterns between different assets for multivariate trading',
            'supported_model_types': ['iTransformer', 'PatchTST', 'TimesMixer', 'TimesFM', 'TransformerPredictor'],
            'default_config': {
                'analyze_asset_correlations': True,
                'detect_lead_lag_relationships': True,
                'identify_arbitrage_patterns': True,
                'cross_asset_momentum': True,
                'asset_list': [],
                'correlation_threshold': 0.3,
                'lead_lag_max_steps': 5,
                'arbitrage_threshold': 0.5,
                'real_time_mode': False,
                'latency_optimization': False
            }
        }
        
        # Import explainer implementations
        try:
            from .explainers.lime_explainer import LimeExplainer
            self._explainer_registry['lime'] = LimeExplainer
        except ImportError:
            # LIME explainer not yet implemented
            pass
        
        try:
            from .explainers.permutation_explainer import PermutationExplainer
            self._explainer_registry['permutation'] = PermutationExplainer
        except ImportError:
            # Permutation explainer not yet implemented
            pass
        
        try:
            from .explainers.gradient_explainer import GradientExplainer
            self._explainer_registry['gradient'] = GradientExplainer
        except ImportError:
            # Gradient explainer not yet implemented
            pass
        
        # Import transformer-specific explainers
        try:
            from .transformers.attention_explainer import AttentionExplainer
            self._explainer_registry['attention'] = AttentionExplainer
        except ImportError:
            # Attention explainer not yet implemented
            pass
        
        try:
            from .transformers.temporal_attention_analyzer import TemporalAttentionAnalyzer
            self._explainer_registry['temporal_attention'] = TemporalAttentionAnalyzer
        except ImportError:
            # Temporal attention analyzer not yet implemented
            pass
        
        try:
            from .transformers.cross_attention_analyzer import CrossAttentionAnalyzer
            self._explainer_registry['cross_attention'] = CrossAttentionAnalyzer
        except ImportError:
            # Cross attention analyzer not yet implemented
            pass
    
    @property
    def supported_explainers(self) -> List[str]:
        """Get list of supported explainer types."""
        return list(self._explainer_registry.keys())
    
    def is_supported(self, explainer_type: str) -> bool:
        """
        Check if explainer type is supported.
        
        Args:
            explainer_type: Type of explainer to check
            
        Returns:
            True if supported, False otherwise
        """
        return explainer_type in self._explainer_registry
    
    def create_explainer(
        self,
        explainer_type: str,
        model: Any,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> BaseExplainer:
        """
        Create an explainer instance.
        
        Args:
            explainer_type: Type of explainer to create
            model: The ML/RL model to explain
            feature_names: List of feature names
            config: Optional configuration for the explainer
            
        Returns:
            Configured explainer instance
            
        Raises:
            ValueError: If explainer type is not supported
            TypeError: If model or feature_names are invalid
        """
        if not self.is_supported(explainer_type):
            raise ValueError(f"Unsupported explainer type: {explainer_type}. "
                           f"Supported types: {self.supported_explainers}")
        
        # Validate inputs
        if model is None:
            raise ValueError("Model cannot be None")
        
        if not isinstance(feature_names, list) or len(feature_names) == 0:
            raise TypeError("feature_names must be a non-empty list")
        
        # Check cache if enabled
        cache_key = f"{explainer_type}_{id(model)}_{hash(tuple(feature_names))}"
        if self.cache_explainers and cache_key in self._explainer_cache:
            return self._explainer_cache[cache_key]
        
        # Get explainer class
        explainer_class = self._explainer_registry[explainer_type]
        
        # Merge default config with provided config
        default_config = self.get_default_config(explainer_type)
        final_config = {**default_config, **(config or {})}
        
        # Validate configuration
        is_valid, errors = self.validate_config(explainer_type, final_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration for {explainer_type}: {errors}")
        
        # Create explainer instance
        try:
            explainer = explainer_class(
                model=model,
                feature_names=feature_names,
                config=final_config
            )
            
            # Cache if enabled
            if self.cache_explainers:
                self._explainer_cache[cache_key] = explainer
            
            logger.info(f"Created {explainer_type} explainer with {len(feature_names)} features")
            return explainer
            
        except Exception as e:
            logger.error(f"Failed to create {explainer_type} explainer: {str(e)}")
            raise
    
    def register_explainer(
        self,
        explainer_type: str,
        explainer_class: Type[BaseExplainer],
        override: bool = False
    ) -> None:
        """
        Register a custom explainer type.
        
        Args:
            explainer_type: Name for the explainer type
            explainer_class: Explainer class to register
            override: Whether to override existing registration
            
        Raises:
            ValueError: If explainer type already registered and override=False
        """
        if explainer_type in self._explainer_registry and not override:
            raise ValueError(f"Explainer type '{explainer_type}' already registered. "
                           "Use override=True to replace existing registration.")
        
        if not issubclass(explainer_class, BaseExplainer):
            raise TypeError("explainer_class must be a subclass of BaseExplainer")
        
        self._explainer_registry[explainer_type] = explainer_class
        
        # Set default info if not provided
        if explainer_type not in self._explainer_info:
            self._explainer_info[explainer_type] = {
                'name': explainer_type,
                'description': f'Custom {explainer_type} explainer',
                'supported_model_types': ['custom'],
                'default_config': {}
            }
        
        logger.info(f"Registered {explainer_type} explainer class: {explainer_class.__name__}")
    
    def list_supported_explainers(self) -> List[str]:
        """
        Get list of all supported explainer types.
        
        Returns:
            List of explainer type names
        """
        return list(self._explainer_registry.keys())
    
    def get_explainer_info(self, explainer_type: str) -> Dict[str, Any]:
        """
        Get information about a specific explainer type.
        
        Args:
            explainer_type: Type of explainer
            
        Returns:
            Dictionary containing explainer information
            
        Raises:
            ValueError: If explainer type is not supported
        """
        if not self.is_supported(explainer_type):
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
        
        return self._explainer_info[explainer_type].copy()
    
    def get_default_config(self, explainer_type: str) -> Dict[str, Any]:
        """
        Get default configuration for an explainer type.
        
        Args:
            explainer_type: Type of explainer
            
        Returns:
            Dictionary of default configuration values
            
        Raises:
            ValueError: If explainer type is not supported
        """
        if not self.is_supported(explainer_type):
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
        
        return self._explainer_info[explainer_type]['default_config'].copy()
    
    def validate_config(
        self,
        explainer_type: str,
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate configuration for an explainer type.
        
        Args:
            explainer_type: Type of explainer
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not self.is_supported(explainer_type):
            return False, [f"Unsupported explainer type: {explainer_type}"]
        
        errors = []
        
        # Basic validation based on explainer type
        if explainer_type == 'lime':
            if 'num_samples' in config:
                if not isinstance(config['num_samples'], int) or config['num_samples'] <= 0:
                    errors.append("num_samples must be a positive integer")
            
            if 'kernel_width' in config:
                if not isinstance(config['kernel_width'], (int, float)) or config['kernel_width'] <= 0:
                    errors.append("kernel_width must be a positive number")
        
        elif explainer_type == 'permutation':
            if 'num_permutations' in config:
                if not isinstance(config['num_permutations'], int) or config['num_permutations'] <= 0:
                    errors.append("num_permutations must be a positive integer")
        
        elif explainer_type == 'gradient':
            if 'steps' in config:
                if not isinstance(config['steps'], int) or config['steps'] <= 0:
                    errors.append("steps must be a positive integer")
        
        elif explainer_type == 'attention':
            if 'attention_heads' in config:
                if config['attention_heads'] not in ['all', 'first', 'last'] and not isinstance(config['attention_heads'], int):
                    errors.append("attention_heads must be 'all', 'first', 'last', or an integer")
            
            if 'attention_layers' in config:
                if config['attention_layers'] not in ['all', 'first', 'last'] and not isinstance(config['attention_layers'], int):
                    errors.append("attention_layers must be 'all', 'first', 'last', or an integer")
        
        elif explainer_type == 'temporal_attention':
            if 'min_period' in config:
                if not isinstance(config['min_period'], int) or config['min_period'] < 2:
                    errors.append("min_period must be an integer >= 2")
            
            if 'max_period' in config:
                if not isinstance(config['max_period'], int) or config['max_period'] <= 0:
                    errors.append("max_period must be a positive integer")
            
            if 'regime_threshold' in config:
                if not isinstance(config['regime_threshold'], (int, float)) or not (0 < config['regime_threshold'] < 1):
                    errors.append("regime_threshold must be a number between 0 and 1")
        
        elif explainer_type == 'cross_attention':
            if 'correlation_threshold' in config:
                if not isinstance(config['correlation_threshold'], (int, float)) or not (0 <= config['correlation_threshold'] <= 1):
                    errors.append("correlation_threshold must be a number between 0 and 1")
            
            if 'lead_lag_max_steps' in config:
                if not isinstance(config['lead_lag_max_steps'], int) or config['lead_lag_max_steps'] <= 0:
                    errors.append("lead_lag_max_steps must be a positive integer")
            
            if 'arbitrage_threshold' in config:
                if not isinstance(config['arbitrage_threshold'], (int, float)) or not (0 <= config['arbitrage_threshold'] <= 1):
                    errors.append("arbitrage_threshold must be a number between 0 and 1")
        
        return len(errors) == 0, errors
    
    def clear_cache(self) -> None:
        """Clear the explainer cache."""
        self._explainer_cache.clear()
        logger.info("Cleared explainer cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return {
            'cache_enabled': self.cache_explainers,
            'cached_explainers': len(self._explainer_cache),
            'registered_types': len(self._explainer_registry),
            'cache_keys': list(self._explainer_cache.keys()) if self.cache_explainers else []
        }