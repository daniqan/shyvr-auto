"""
LIME (Local Interpretable Model-agnostic Explanations) Explainer

This module implements the LIME explanation method for generating
local explanations of model predictions by approximating models locally
with interpretable models.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances
import logging

from ..base import ModelAgnosticExplainer
from ..data_models import ExplanationData

logger = logging.getLogger(__name__)


class LimeExplainer(ModelAgnosticExplainer):
    """
    LIME explainer implementation.
    
    LIME explains predictions by learning an interpretable model locally
    around the prediction. It generates samples around the instance of
    interest and trains a local linear model on these samples.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: The ML/RL model to explain
            feature_names: List of feature names
            config: Optional configuration dictionary
        """
        super().__init__(model, feature_names, config)
        
        # Set default configuration values
        self._set_default_config()
        
        # Initialize components
        self.local_model = Ridge(alpha=self.config.get('ridge_alpha', 1.0))
        
        logger.info(f"Initialized LIME explainer with {len(feature_names)} features")
    
    def _set_default_config(self) -> None:
        """Set default configuration values."""
        defaults = {
            'num_samples': 1000,
            'kernel_width': 0.75,
            'distance_metric': 'euclidean',
            'feature_selection': 'auto',
            'ridge_alpha': 1.0,
            'random_state': 42
        }
        
        # Update with user config, keeping defaults for missing keys
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def explain(
        self,
        instances: Union[np.ndarray, List[List[float]]],
        **kwargs
    ) -> List[ExplanationData]:
        """
        Explain multiple instances using LIME.
        
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
                logger.debug(f"Generated LIME explanation for instance {i}")
            except Exception as e:
                logger.error(f"Failed to explain instance {i}: {str(e)}")
                # Create a fallback explanation with zero importance
                fallback_importance = {name: 0.0 for name in self.feature_names}
                explanation = ExplanationData(
                    feature_importance=fallback_importance,
                    explanation_type='lime',
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
        Explain a single instance using LIME.
        
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
        
        # Get model prediction for original instance
        original_prediction = self._predict_fn(instance.reshape(1, -1))[0]
        
        # Generate neighborhood samples
        samples, weights = self._generate_neighborhood(instance)
        
        # Get predictions for neighborhood samples
        sample_predictions = self._predict_fn(samples)
        
        # Train local interpretable model
        local_importance = self._train_local_model(
            samples, sample_predictions, weights, instance
        )
        
        # Create feature importance dictionary
        feature_importance = {
            name: float(importance) 
            for name, importance in zip(self.feature_names, local_importance)
        }
        
        # Calculate confidence based on local model fit quality
        confidence_score = self._calculate_confidence(
            samples, sample_predictions, weights, local_importance, instance
        )
        
        # Create explanation metadata
        explanation_metadata = {
            'num_samples_generated': len(samples),
            'kernel_width': self.config['kernel_width'],
            'distance_metric': self.config['distance_metric'],
            'ridge_alpha': self.config['ridge_alpha'],
            'local_model_score': getattr(self.local_model, 'score_', None)
        }
        
        return ExplanationData(
            feature_importance=feature_importance,
            explanation_type='lime',
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
        Get feature importance scores for an instance using LIME.
        
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
        Validate that model and feature names are compatible with LIME.
        
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
        
        # LIME-specific validation
        if len(feature_names) < 2:
            return False, "LIME requires at least 2 features"
        
        return True, None
    
    def get_explanation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this LIME explainer and its configuration.
        
        Returns:
            Dictionary containing explainer metadata
        """
        return {
            'explainer_type': 'lime',
            'name': 'LIME (Local Interpretable Model-agnostic Explanations)',
            'description': 'Explains predictions by approximating the model locally with an interpretable model',
            'supported_model_types': self.get_supported_model_types(),
            'configuration': self.config.copy(),
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names.copy(),
            'local_model_type': str(type(self.local_model).__name__)
        }
    
    def _generate_neighborhood(self, instance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate neighborhood samples around the instance.
        
        Args:
            instance: Original instance to generate samples around
            
        Returns:
            Tuple of (samples, weights)
        """
        num_samples = self.config['num_samples']
        np.random.seed(self.config.get('random_state', 42))
        
        # Calculate feature statistics for sampling
        feature_means = instance
        feature_stds = np.abs(instance) * 0.1  # 10% of feature values as std
        feature_stds = np.maximum(feature_stds, 0.01)  # Minimum std to avoid zero
        
        # Generate samples using normal distribution around instance
        samples = np.random.normal(
            loc=feature_means,
            scale=feature_stds,
            size=(num_samples, len(instance))
        )
        
        # Add the original instance as first sample
        samples[0] = instance
        
        # Calculate weights based on distance to original instance
        weights = self._calculate_sample_weights(samples, instance)
        
        return samples, weights
    
    def _calculate_sample_weights(self, samples: np.ndarray, instance: np.ndarray) -> np.ndarray:
        """
        Calculate weights for neighborhood samples based on distance.
        
        Args:
            samples: Generated samples
            instance: Original instance
            
        Returns:
            Array of weights for each sample
        """
        # Calculate distances
        distances = euclidean_distances(samples, instance.reshape(1, -1)).flatten()
        
        # Apply kernel function to convert distances to weights
        kernel_width = self.config['kernel_width']
        weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))
        
        # Ensure original instance has maximum weight
        weights[0] = 1.0
        
        return weights
    
    def _train_local_model(
        self,
        samples: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray,
        instance: np.ndarray
    ) -> np.ndarray:
        """
        Train local interpretable model on neighborhood samples.
        
        Args:
            samples: Neighborhood samples
            predictions: Model predictions for samples
            weights: Sample weights
            instance: Original instance
            
        Returns:
            Local feature importance coefficients
        """
        # Train Ridge regression with sample weights
        self.local_model.fit(samples, predictions, sample_weight=weights)
        
        # Return coefficients as feature importance
        return self.local_model.coef_
    
    def _calculate_confidence(
        self,
        samples: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray,
        local_importance: np.ndarray,
        instance: np.ndarray
    ) -> float:
        """
        Calculate confidence score for the explanation.
        
        Args:
            samples: Neighborhood samples
            predictions: Model predictions for samples
            weights: Sample weights
            local_importance: Local feature importance
            instance: Original instance
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Calculate R² score of local model
            local_predictions = self.local_model.predict(samples)
            
            # Weighted R² calculation
            ss_res = np.sum(weights * (predictions - local_predictions) ** 2)
            ss_tot = np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2)
            
            if ss_tot == 0:
                r_squared = 1.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, r_squared))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {str(e)}")
            return 0.5  # Default moderate confidence
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for LIME explainer.
        
        Returns:
            Dictionary of default configuration values
        """
        return {
            'num_samples': 1000,
            'kernel_width': 0.75,
            'distance_metric': 'euclidean',
            'feature_selection': 'auto',
            'ridge_alpha': 1.0,
            'random_state': 42
        }