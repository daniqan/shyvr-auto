"""
Gradient-based Attribution Explainer

This module implements gradient-based explanation methods including
vanilla gradients, integrated gradients, and gradient * input attribution
for models that support gradient computation.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import logging

from ..base import GradientBasedExplainer
from ..data_models import ExplanationData

logger = logging.getLogger(__name__)


class GradientExplainer(GradientBasedExplainer):
    """
    Gradient-based explainer implementation.
    
    This explainer computes feature attribution using gradients of the model
    output with respect to input features. Supports multiple attribution methods
    including vanilla gradients, integrated gradients, and gradient * input.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Gradient explainer.
        
        Args:
            model: The ML/RL model to explain (must support gradients)
            feature_names: List of feature names
            config: Optional configuration dictionary
        """
        super().__init__(model, feature_names, config)
        
        # Set default configuration values
        self._set_default_config()
        
        # Initialize framework-specific components
        self.framework = self._detect_framework()
        self._setup_framework_components()
        
        logger.info(f"Initialized Gradient explainer with {len(feature_names)} features "
                   f"for {self.framework} model")
    
    def _set_default_config(self) -> None:
        """Set default configuration values."""
        defaults = {
            'baseline': 'zero',
            'steps': 50,
            'method': 'integrated_gradients',
            'multiply_by_inputs': True,
            'noise_tunnel': False,
            'noise_level': 0.1,
            'noise_samples': 10
        }
        
        # Update with user config, keeping defaults for missing keys
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _detect_framework(self) -> str:
        """Detect the deep learning framework being used."""
        # Check for PyTorch
        if hasattr(self.model, 'parameters') and hasattr(self.model, 'forward'):
            return 'pytorch'
        
        # Check for TensorFlow/Keras
        if hasattr(self.model, 'trainable_weights'):
            return 'tensorflow'
        
        # Check if it's a wrapped model with specific attributes
        if hasattr(self.model, '__class__'):
            class_name = self.model.__class__.__name__.lower()
            if 'torch' in class_name or 'pytorch' in class_name:
                return 'pytorch'
            elif 'tensorflow' in class_name or 'keras' in class_name:
                return 'tensorflow'
        
        # Default to PyTorch-like interface
        logger.warning("Could not detect framework, assuming PyTorch-like interface")
        return 'pytorch'
    
    def _setup_framework_components(self) -> None:
        """Setup framework-specific components for gradient computation."""
        if self.framework == 'pytorch':
            self._setup_pytorch_components()
        elif self.framework == 'tensorflow':
            self._setup_tensorflow_components()
        else:
            logger.warning(f"Unsupported framework: {self.framework}")
    
    def _setup_pytorch_components(self) -> None:
        """Setup PyTorch-specific components."""
        try:
            import torch
            self.torch = torch
            self.model.eval()  # Set to evaluation mode
        except ImportError:
            logger.error("PyTorch not available but PyTorch model detected")
            raise ImportError("PyTorch is required for PyTorch models")
    
    def _setup_tensorflow_components(self) -> None:
        """Setup TensorFlow-specific components."""
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            logger.error("TensorFlow not available but TensorFlow model detected")
            raise ImportError("TensorFlow is required for TensorFlow models")
    
    def explain(
        self,
        instances: Union[np.ndarray, List[List[float]]],
        **kwargs
    ) -> List[ExplanationData]:
        """
        Explain multiple instances using gradient attribution.
        
        Args:
            instances: Input instances to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            List of ExplanationData objects, one per instance
        """
        if isinstance(instances, list):
            instances = np.array(instances)
        
        explanations = []
        
        for i, instance in enumerate(instances):
            try:
                explanation = self.explain_instance(instance, **kwargs)
                explanations.append(explanation)
                logger.debug(f"Generated gradient explanation for instance {i}")
            except Exception as e:
                logger.error(f"Failed to explain instance {i}: {str(e)}")
                # Create a fallback explanation with zero importance
                fallback_importance = {name: 0.0 for name in self.feature_names}
                explanation = ExplanationData(
                    feature_importance=fallback_importance,
                    explanation_type='gradient',
                    instance_data=instance.tolist(),
                    model_prediction=0.0,
                    confidence_score=0.0,
                    explanation_metadata={'error': str(e), 'fallback': True}
                )
                explanations.append(explanation)
        
        return explanations
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, List[float]],
        **kwargs
    ) -> ExplanationData:
        """
        Explain a single instance using gradient attribution.
        
        Args:
            instance: Single input instance to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            ExplanationData object containing the explanation
        """
        if isinstance(instance, list):
            instance = np.array(instance)
        
        # Validate instance dimensions
        if len(instance) != len(self.feature_names):
            raise ValueError(f"Instance has {len(instance)} features, "
                           f"expected {len(self.feature_names)}")
        
        # Compute gradients based on selected method
        method = self.config['method']
        if method == 'vanilla_gradients':
            attributions = self._vanilla_gradients(instance, **kwargs)
        elif method == 'integrated_gradients':
            attributions = self._integrated_gradients(instance, **kwargs)
        elif method == 'gradient_x_input':
            attributions = self._gradient_times_input(instance, **kwargs)
        else:
            logger.warning(f"Unknown method '{method}', using integrated_gradients")
            attributions = self._integrated_gradients(instance, **kwargs)
        
        # Get model prediction for the instance
        original_prediction = self._get_model_prediction(instance)
        
        # Create feature importance dictionary
        feature_importance = {
            name: float(attribution) 
            for name, attribution in zip(self.feature_names, attributions)
        }
        
        # Calculate confidence based on attribution distribution
        confidence_score = self._calculate_confidence(attributions)
        
        # Create explanation metadata
        explanation_metadata = {
            'method': method,
            'baseline': self.config['baseline'],
            'steps': self.config.get('steps', 50),
            'framework': self.framework,
            'multiply_by_inputs': self.config['multiply_by_inputs'],
            'noise_tunnel': self.config['noise_tunnel']
        }
        
        return ExplanationData(
            feature_importance=feature_importance,
            explanation_type='gradient',
            instance_data=instance.tolist(),
            model_prediction=float(original_prediction),
            confidence_score=confidence_score,
            explanation_metadata=explanation_metadata,
            feature_names=self.feature_names
        )
    
    def get_feature_importance(
        self,
        instance: Union[np.ndarray, List[float]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Get feature importance scores for an instance using gradients.
        
        Args:
            instance: Input instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        explanation = self.explain_instance(instance, **kwargs)
        return explanation.feature_importance
    
    def validate_input(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that model and feature names are compatible with gradient attribution.
        
        Args:
            model: The model to validate
            feature_names: List of feature names to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Use parent validation first
        is_valid, error_msg = super().validate_input(model, feature_names)
        if not is_valid:
            return is_valid, error_msg
        
        # Gradient-specific validation
        if len(feature_names) < 1:
            return False, "Gradient attribution requires at least 1 feature"
        
        return True, None
    
    def get_explanation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this gradient explainer and its configuration.
        
        Returns:
            Dictionary containing explainer metadata
        """
        return {
            'explainer_type': 'gradient',
            'name': 'Gradient-based Attribution',
            'description': 'Computes feature attribution using gradients of model output with respect to input features',
            'supported_model_types': ['torch', 'tensorflow'],
            'configuration': self.config.copy(),
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names.copy(),
            'framework': self.framework,
            'supported_methods': ['vanilla_gradients', 'integrated_gradients', 'gradient_x_input']
        }
    
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
        if self.framework == 'pytorch':
            return self._compute_pytorch_gradients(instance, target_class)
        elif self.framework == 'tensorflow':
            return self._compute_tensorflow_gradients(instance, target_class)
        else:
            raise NotImplementedError(f"Gradient computation not implemented for {self.framework}")
    
    def _vanilla_gradients(self, instance: np.ndarray, **kwargs) -> np.ndarray:
        """Compute vanilla gradients."""
        gradients = self.compute_gradients(instance, kwargs.get('target_class'))
        
        if self.config['multiply_by_inputs']:
            gradients = gradients * instance
        
        return gradients
    
    def _integrated_gradients(self, instance: np.ndarray, **kwargs) -> np.ndarray:
        """Compute integrated gradients."""
        steps = self.config['steps']
        baseline = self._get_baseline(instance)
        
        # Create path from baseline to instance
        alphas = np.linspace(0, 1, steps)
        attributions = np.zeros_like(instance)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (instance - baseline)
            gradients = self.compute_gradients(interpolated, kwargs.get('target_class'))
            attributions += gradients
        
        # Average and scale by path
        attributions = (attributions / steps) * (instance - baseline)
        
        return attributions
    
    def _gradient_times_input(self, instance: np.ndarray, **kwargs) -> np.ndarray:
        """Compute gradient * input attribution."""
        gradients = self.compute_gradients(instance, kwargs.get('target_class'))
        return gradients * instance
    
    def _get_baseline(self, instance: np.ndarray) -> np.ndarray:
        """Get baseline for integrated gradients."""
        baseline_type = self.config['baseline']
        
        if baseline_type == 'zero':
            return np.zeros_like(instance)
        elif baseline_type == 'mean':
            # This would require a dataset to compute mean
            # For now, use zero baseline
            logger.warning("Mean baseline not implemented, using zero baseline")
            return np.zeros_like(instance)
        elif baseline_type == 'random':
            np.random.seed(42)
            return np.random.normal(0, 0.1, instance.shape)
        else:
            return np.zeros_like(instance)
    
    def _get_model_prediction(self, instance: np.ndarray) -> float:
        """Get model prediction for instance."""
        if self.framework == 'pytorch':
            return self._get_pytorch_prediction(instance)
        elif self.framework == 'tensorflow':
            return self._get_tensorflow_prediction(instance)
        else:
            # Fallback to generic prediction
            return 0.0
    
    def _compute_pytorch_gradients(self, instance: np.ndarray, target_class: Optional[int] = None) -> np.ndarray:
        """Compute gradients using PyTorch."""
        try:
            # Convert to tensor and enable gradients
            input_tensor = self.torch.tensor(instance, dtype=self.torch.float32, requires_grad=True)
            
            # Forward pass
            output = self.model(input_tensor.unsqueeze(0))
            
            # Select target for gradient computation
            if target_class is not None and output.dim() > 1:
                target = output[0, target_class]
            else:
                target = output[0] if output.dim() > 0 else output
            
            # Backward pass
            target.backward()
            
            # Get gradients
            gradients = input_tensor.grad.detach().numpy()
            
            return gradients
            
        except Exception as e:
            logger.error(f"Failed to compute PyTorch gradients: {str(e)}")
            return np.zeros_like(instance)
    
    def _compute_tensorflow_gradients(self, instance: np.ndarray, target_class: Optional[int] = None) -> np.ndarray:
        """Compute gradients using TensorFlow."""
        try:
            # Convert to tensor
            input_tensor = self.tf.constant(instance.reshape(1, -1), dtype=self.tf.float32)
            
            # Compute gradients
            with self.tf.GradientTape() as tape:
                tape.watch(input_tensor)
                output = self.model(input_tensor)
                
                # Select target for gradient computation
                if target_class is not None and len(output.shape) > 1:
                    target = output[0, target_class]
                else:
                    target = output[0] if len(output.shape) > 0 else output
            
            # Get gradients
            gradients = tape.gradient(target, input_tensor)
            
            return gradients.numpy().flatten()
            
        except Exception as e:
            logger.error(f"Failed to compute TensorFlow gradients: {str(e)}")
            return np.zeros_like(instance)
    
    def _get_pytorch_prediction(self, instance: np.ndarray) -> float:
        """Get PyTorch model prediction."""
        try:
            input_tensor = self.torch.tensor(instance, dtype=self.torch.float32)
            with self.torch.no_grad():
                output = self.model(input_tensor.unsqueeze(0))
            return float(output[0])
        except Exception:
            return 0.0
    
    def _get_tensorflow_prediction(self, instance: np.ndarray) -> float:
        """Get TensorFlow model prediction."""
        try:
            input_tensor = self.tf.constant(instance.reshape(1, -1), dtype=self.tf.float32)
            output = self.model(input_tensor)
            return float(output[0])
        except Exception:
            return 0.0
    
    def _calculate_confidence(self, attributions: np.ndarray) -> float:
        """
        Calculate confidence score based on attribution distribution.
        
        Args:
            attributions: Array of feature attributions
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Calculate signal-to-noise ratio as proxy for confidence
            attribution_magnitude = np.mean(np.abs(attributions))
            attribution_variance = np.var(attributions)
            
            if attribution_variance == 0:
                return 1.0 if attribution_magnitude > 0 else 0.5
            
            # Signal to noise ratio
            snr = attribution_magnitude / np.sqrt(attribution_variance)
            
            # Convert to confidence score
            confidence = 1.0 / (1.0 + np.exp(-snr))
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {str(e)}")
            return 0.5  # Default moderate confidence
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for gradient explainer.
        
        Returns:
            Dictionary of default configuration values
        """
        return {
            'baseline': 'zero',
            'steps': 50,
            'method': 'integrated_gradients',
            'multiply_by_inputs': True,
            'noise_tunnel': False,
            'noise_level': 0.1,
            'noise_samples': 10
        }