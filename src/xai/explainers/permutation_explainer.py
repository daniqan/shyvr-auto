"""
Permutation Feature Importance Explainer

This module implements permutation-based feature importance explanation
by measuring the change in model performance when feature values are
randomly permuted.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import logging

from ..base import ModelAgnosticExplainer
from ..data_models import ExplanationData

logger = logging.getLogger(__name__)


class PermutationExplainer(ModelAgnosticExplainer):
    """
    Permutation feature importance explainer implementation.
    
    This explainer measures feature importance by evaluating how much
    the model performance decreases when each feature is randomly permuted,
    breaking the relationship between the feature and the target.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Permutation explainer.
        
        Args:
            model: The ML/RL model to explain
            feature_names: List of feature names
            config: Optional configuration dictionary
        """
        super().__init__(model, feature_names, config)
        
        # Set default configuration values
        self._set_default_config()
        
        # Initialize scoring function
        self.scoring_function = self._get_scoring_function()
        
        logger.info(f"Initialized Permutation explainer with {len(feature_names)} features")
    
    def _set_default_config(self) -> None:
        """Set default configuration values."""
        defaults = {
            'num_permutations': 100,
            'scoring_metric': 'accuracy',
            'random_state': 42,
            'return_baseline': True,
            'normalize_importance': True
        }
        
        # Update with user config, keeping defaults for missing keys
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _get_scoring_function(self) -> Callable:
        """Get scoring function based on configuration."""
        metric = self.config['scoring_metric'].lower()
        
        if metric == 'accuracy':
            return self._accuracy_score
        elif metric == 'mse' or metric == 'mean_squared_error':
            return self._mse_score
        elif metric == 'r2' or metric == 'r_squared':
            return self._r2_score
        else:
            logger.warning(f"Unknown scoring metric '{metric}', using accuracy")
            return self._accuracy_score
    
    def _accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score (higher is better)."""
        try:
            # Handle regression case by checking if predictions are continuous
            if len(np.unique(y_pred)) > 10:  # Likely regression
                # Use R² as proxy for accuracy in regression
                return r2_score(y_true, y_pred)
            else:
                # Classification case
                return accuracy_score(y_true, y_pred)
        except Exception:
            # Fallback to MSE-based score
            mse = mean_squared_error(y_true, y_pred)
            return 1.0 / (1.0 + mse)  # Convert to higher-is-better
    
    def _mse_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate negative MSE score (higher is better)."""
        return -mean_squared_error(y_true, y_pred)
    
    def _r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score (higher is better)."""
        return r2_score(y_true, y_pred)
    
    def explain(
        self,
        instances: Union[np.ndarray, List[List[float]]],
        **kwargs
    ) -> List[ExplanationData]:
        """
        Explain multiple instances using permutation importance.
        
        Args:
            instances: Input instances to explain
            **kwargs: Additional explanation parameters
            
        Returns:
            List of ExplanationData objects, one per instance
        """
        if isinstance(instances, list):
            instances = np.array(instances)
        
        # For permutation importance, we typically need true labels
        # If not provided, we'll use the model's predictions as proxy
        y_true = kwargs.get('y_true', None)
        if y_true is None:
            logger.warning("No true labels provided, using model predictions as baseline")
            y_true = self._predict_fn(instances)
        
        explanations = []
        
        for i, instance in enumerate(instances):
            try:
                explanation = self.explain_instance(
                    instance, 
                    y_true=y_true[i] if hasattr(y_true, '__len__') else y_true,
                    X_background=instances,
                    **kwargs
                )
                explanations.append(explanation)
                logger.debug(f"Generated permutation explanation for instance {i}")
            except Exception as e:
                logger.error(f"Failed to explain instance {i}: {str(e)}")
                # Create a fallback explanation with zero importance
                fallback_importance = {name: 0.0 for name in self.feature_names}
                explanation = ExplanationData(
                    feature_importance=fallback_importance,
                    explanation_type='permutation',
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
        Explain a single instance using permutation importance.
        
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
        
        # Get background dataset for permutation
        X_background = kwargs.get('X_background', None)
        if X_background is None:
            # Create synthetic background by adding noise to the instance
            logger.warning("No background dataset provided, creating synthetic background")
            X_background = self._create_synthetic_background(instance)
        
        # Get true labels
        y_true = kwargs.get('y_true', None)
        if y_true is None:
            y_true = self._predict_fn(X_background)
        
        # Ensure y_true is array-like
        if not hasattr(y_true, '__len__'):
            y_true = np.full(len(X_background), y_true)
        
        # Get model prediction for original instance
        original_prediction = self._predict_fn(instance.reshape(1, -1))[0]
        
        # Calculate baseline performance
        baseline_score = self.scoring_function(y_true, self._predict_fn(X_background))
        
        # Calculate permutation importance for each feature
        feature_importance = self._calculate_permutation_importance(
            X_background, y_true, baseline_score
        )
        
        # Create feature importance dictionary
        importance_dict = {
            name: float(importance) 
            for name, importance in zip(self.feature_names, feature_importance)
        }
        
        # Calculate confidence based on importance variance
        confidence_score = self._calculate_confidence(feature_importance)
        
        # Create explanation metadata
        explanation_metadata = {
            'num_permutations': self.config['num_permutations'],
            'scoring_metric': self.config['scoring_metric'],
            'baseline_score': baseline_score,
            'background_size': len(X_background),
            'normalized': self.config['normalize_importance']
        }
        
        return ExplanationData(
            feature_importance=importance_dict,
            explanation_type='permutation',
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
        Get feature importance scores for an instance using permutation.
        
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
        Validate that model and feature names are compatible with permutation importance.
        
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
        
        # Permutation-specific validation
        if len(feature_names) < 1:
            return False, "Permutation importance requires at least 1 feature"
        
        return True, None
    
    def get_explanation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this permutation explainer and its configuration.
        
        Returns:
            Dictionary containing explainer metadata
        """
        return {
            'explainer_type': 'permutation',
            'name': 'Permutation Feature Importance',
            'description': 'Measures feature importance by evaluating prediction change when feature values are permuted',
            'supported_model_types': self.get_supported_model_types(),
            'configuration': self.config.copy(),
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names.copy(),
            'scoring_function': self.config['scoring_metric']
        }
    
    def _create_synthetic_background(
        self, 
        instance: np.ndarray, 
        size: int = 100
    ) -> np.ndarray:
        """
        Create synthetic background dataset by adding noise to instance.
        
        Args:
            instance: Original instance
            size: Number of background samples to create
            
        Returns:
            Synthetic background dataset
        """
        np.random.seed(self.config.get('random_state', 42))
        
        # Calculate noise scale based on instance values
        noise_scale = np.abs(instance) * 0.2  # 20% of feature values
        noise_scale = np.maximum(noise_scale, 0.1)  # Minimum noise
        
        # Generate samples by adding noise
        background = np.random.normal(
            loc=instance,
            scale=noise_scale,
            size=(size, len(instance))
        )
        
        return background
    
    def _calculate_permutation_importance(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        baseline_score: float
    ) -> np.ndarray:
        """
        Calculate permutation importance for all features.
        
        Args:
            X: Background dataset
            y_true: True labels
            baseline_score: Baseline model performance
            
        Returns:
            Array of importance scores for each feature
        """
        num_features = X.shape[1]
        num_permutations = self.config['num_permutations']
        importance_scores = np.zeros(num_features)
        
        np.random.seed(self.config.get('random_state', 42))
        
        for feature_idx in range(num_features):
            feature_scores = []
            
            for _ in range(num_permutations):
                # Create copy of data
                X_permuted = X.copy()
                
                # Permute the feature column
                permutation_idx = np.random.permutation(len(X))
                X_permuted[:, feature_idx] = X[permutation_idx, feature_idx]
                
                # Get predictions and calculate score
                y_pred_permuted = self._predict_fn(X_permuted)
                permuted_score = self.scoring_function(y_true, y_pred_permuted)
                
                # Importance is the decrease in performance
                feature_importance = baseline_score - permuted_score
                feature_scores.append(feature_importance)
            
            # Average importance across permutations
            importance_scores[feature_idx] = np.mean(feature_scores)
        
        # Normalize if requested
        if self.config['normalize_importance']:
            importance_sum = np.sum(np.abs(importance_scores))
            if importance_sum > 0:
                importance_scores = importance_scores / importance_sum
        
        return importance_scores
    
    def _calculate_confidence(self, feature_importance: np.ndarray) -> float:
        """
        Calculate confidence score based on importance distribution.
        
        Args:
            feature_importance: Array of feature importance scores
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Calculate coefficient of variation as proxy for confidence
            # Lower variation = higher confidence
            mean_importance = np.mean(np.abs(feature_importance))
            
            if mean_importance == 0:
                return 0.5  # Moderate confidence when all features have zero importance
            
            std_importance = np.std(feature_importance)
            cv = std_importance / mean_importance
            
            # Convert to confidence score (lower CV = higher confidence)
            confidence = 1.0 / (1.0 + cv)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {str(e)}")
            return 0.5  # Default moderate confidence
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for permutation explainer.
        
        Returns:
            Dictionary of default configuration values
        """
        return {
            'num_permutations': 100,
            'scoring_metric': 'accuracy',
            'random_state': 42,
            'return_baseline': True,
            'normalize_importance': True
        }