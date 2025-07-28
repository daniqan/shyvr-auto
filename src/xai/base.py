"""
Base abstract interfaces for XAI explainers.

This module defines the abstract base classes that all explainers must implement,
ensuring consistent interface across different explanation methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from .data_models import ExplanationData


class BaseExplainer(ABC):
    """
    Abstract base class for all explainers.
    
    All explainer implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the explainer.
        
        Args:
            model: The ML/RL model to explain
            feature_names: List of feature names
            config: Optional configuration dictionary
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config or {}
        
        # Validate inputs during initialization
        is_valid, error_msg = self.validate_input(model, feature_names)
        if not is_valid:
            raise ValueError(f"Invalid explainer input: {error_msg}")
    
    @abstractmethod
    def explain(
        self,
        instances: Union[np.ndarray, List[List[float]]],
        **kwargs
    ) -> List[ExplanationData]:
        """
        Explain multiple instances.
        
        Args:
            instances: Input instances to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            List of ExplanationData objects, one per instance
        """
        pass
    
    @abstractmethod
    def explain_instance(
        self,
        instance: Union[np.ndarray, List[float]],
        **kwargs
    ) -> ExplanationData:
        """
        Explain a single instance.
        
        Args:
            instance: Single input instance to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            ExplanationData object containing the explanation
        """
        pass
    
    @abstractmethod
    def get_feature_importance(
        self,
        instance: Union[np.ndarray, List[float]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Get feature importance scores for an instance.
        
        Args:
            instance: Input instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    @abstractmethod
    def validate_input(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that the model and feature names are compatible with this explainer.
        
        Args:
            model: The model to validate
            feature_names: List of feature names to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def get_explanation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this explainer and its configuration.
        
        Returns:
            Dictionary containing explainer metadata
        """
        pass
    
    def get_supported_model_types(self) -> List[str]:
        """
        Get list of supported model types for this explainer.
        
        Returns:
            List of supported model type names
        """
        return ['sklearn', 'torch', 'lightgbm', 'custom']
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for this explainer.
        
        Returns:
            Dictionary of default configuration values
        """
        return {}
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update explainer configuration.
        
        Args:
            new_config: New configuration values to merge
        """
        self.config.update(new_config)


class ModelAgnosticExplainer(BaseExplainer):
    """
    Base class for model-agnostic explainers (LIME, SHAP, Permutation, etc.).
    
    These explainers work by analyzing model predictions rather than
    model internals, making them compatible with any model type.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model, feature_names, config)
        
        # Ensure model has a predict method
        if not hasattr(model, 'predict') and not callable(getattr(model, 'predict', None)):
            if not hasattr(model, '__call__'):
                raise ValueError("Model must have a 'predict' method or be callable")
    
    def _predict_fn(self, instances: np.ndarray) -> np.ndarray:
        """
        Wrapper function to get predictions from the model.
        
        Args:
            instances: Input instances
            
        Returns:
            Model predictions
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(instances)
        elif callable(self.model):
            return self.model(instances)
        else:
            raise ValueError("Cannot get predictions from model")
    
    def validate_input(
        self,
        model: Any, 
        feature_names: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate model and feature names for model-agnostic explainers.
        
        Args:
            model: The model to validate
            feature_names: List of feature names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if model has predict method or is callable
        if not hasattr(model, 'predict') and not callable(model):
            return False, "Model must have a 'predict' method or be callable"
        
        # Check feature names
        if not isinstance(feature_names, list):
            return False, "feature_names must be a list"
        
        if len(feature_names) == 0:
            return False, "feature_names cannot be empty"
        
        if not all(isinstance(name, str) for name in feature_names):
            return False, "All feature names must be strings"
        
        return True, None


class GradientBasedExplainer(BaseExplainer):  
    """
    Base class for gradient-based explainers (GradCAM, Integrated Gradients, etc.).
    
    These explainers work by analyzing gradients of the model with respect
    to inputs, requiring models that support gradient computation.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str], 
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model, feature_names, config)
        
        # Ensure model supports gradient computation
        self._validate_gradient_support()
    
    def _validate_gradient_support(self) -> None:
        """
        Validate that the model supports gradient computation.
        
        Raises:
            ValueError: If model doesn't support gradients
        """
        # Check for PyTorch models
        if hasattr(self.model, 'parameters'):
            return
        
        # Check for TensorFlow/Keras models  
        if hasattr(self.model, 'trainable_weights'):
            return
            
        # Could add more framework checks here
        
        raise ValueError("Model must support gradient computation for gradient-based explanations")
    
    def validate_input(
        self,
        model: Any,
        feature_names: List[str] 
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate model and feature names for gradient-based explainers.
        
        Args:
            model: The model to validate
            feature_names: List of feature names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check feature names first
        if not isinstance(feature_names, list):
            return False, "feature_names must be a list"
        
        if len(feature_names) == 0:
            return False, "feature_names cannot be empty"
        
        if not all(isinstance(name, str) for name in feature_names):
            return False, "All feature names must be strings"
        
        # Check gradient support
        if not (hasattr(model, 'parameters') or hasattr(model, 'trainable_weights')):
            return False, "Model must support gradient computation"
        
        return True, None
    
    @abstractmethod
    def compute_gradients(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute gradients of model output with respect to input.
        
        Args:
            instance: Input instance
            target_class: Optional target class for classification models
            
        Returns:
            Gradient array with same shape as input
        """
        pass