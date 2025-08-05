"""
Data models for XAI explanation structures.

This module defines the data structures used to represent explanations
from different explainer types, including serialization and validation.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone
import json
import numpy as np


class ExplanationData:
    """
    Data structure to hold explanation results from XAI methods.
    
    This class standardizes explanation data across different explainer types
    and provides utilities for serialization, validation, and analysis.
    """
    
    VALID_EXPLANATION_TYPES = {
        'lime', 'shap', 'permutation', 'grad_cam', 'gradient', 'custom', 'merged',
        'attention', 'temporal_attention', 'cross_attention'
    }
    
    def __init__(
        self,
        feature_importance: Dict[str, float],
        explanation_type: str,
        instance_data: Union[List[float], np.ndarray],
        model_prediction: Union[float, List[float], np.ndarray],
        confidence_score: Optional[float] = None,
        explanation_metadata: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        prediction_probability: Optional[Union[List[float], np.ndarray]] = None,
        timestamp: Optional[str] = None
    ):
        """
        Initialize explanation data.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            explanation_type: Type of explanation method used
            instance_data: The input instance that was explained
            model_prediction: The model's prediction for this instance
            confidence_score: Optional confidence score (0-1)
            explanation_metadata: Optional metadata about the explanation method
            feature_names: Optional ordered list of feature names
            prediction_probability: Optional prediction probabilities for multi-class
            timestamp: Optional timestamp, auto-generated if not provided
        """
        # Validate and set required fields
        self.feature_importance = self._validate_feature_importance(feature_importance)
        self.explanation_type = self._validate_explanation_type(explanation_type)
        self.instance_data = self._validate_instance_data(instance_data)
        self.model_prediction = self._validate_model_prediction(model_prediction)
        
        # Set optional fields
        self.confidence_score = self._validate_confidence_score(confidence_score)
        self.explanation_metadata = explanation_metadata or {}
        self.feature_names = feature_names or list(self.feature_importance.keys())
        self.prediction_probability = prediction_probability
        
        # Auto-generate timestamp if not provided
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    def _validate_feature_importance(self, feature_importance: Dict[str, float]) -> Dict[str, float]:
        """Validate feature importance dictionary."""
        if not isinstance(feature_importance, dict):
            raise TypeError("feature_importance must be a dictionary")
        
        if len(feature_importance) == 0:
            raise ValueError("feature_importance cannot be empty")
        
        for name, importance in feature_importance.items():
            if not isinstance(name, str):
                raise TypeError("Feature names must be strings")
            if not isinstance(importance, (int, float, np.number)):
                raise TypeError(f"Feature importance for '{name}' must be numeric")
        
        return feature_importance
    
    def _validate_explanation_type(self, explanation_type: str) -> str:
        """Validate explanation type."""
        if not isinstance(explanation_type, str):
            raise TypeError("explanation_type must be a string")
        
        if explanation_type not in self.VALID_EXPLANATION_TYPES:
            raise ValueError(f"Invalid explanation_type '{explanation_type}'. "
                           f"Must be one of: {self.VALID_EXPLANATION_TYPES}")
        
        return explanation_type
    
    def _validate_instance_data(self, instance_data: Union[List[float], np.ndarray]) -> List[float]:
        """Validate instance data."""
        if isinstance(instance_data, np.ndarray):
            instance_data = instance_data.tolist()
        
        if not isinstance(instance_data, list):
            raise TypeError("instance_data must be a list or numpy array")
        
        if not all(isinstance(x, (int, float, np.number)) for x in instance_data):
            raise TypeError("All instance_data values must be numeric")
        
        return instance_data
    
    def _validate_model_prediction(
        self,
        model_prediction: Union[float, List[float], np.ndarray]
    ) -> Union[float, List[float]]:
        """Validate model prediction."""
        if isinstance(model_prediction, np.ndarray):
            model_prediction = model_prediction.tolist()
        
        if isinstance(model_prediction, list):
            if not all(isinstance(x, (int, float, np.number)) for x in model_prediction):
                raise TypeError("All model_prediction values must be numeric")
            return model_prediction
        
        if not isinstance(model_prediction, (int, float, np.number)):
            raise TypeError("model_prediction must be numeric or list of numeric values")
        
        return float(model_prediction)
    
    def _validate_confidence_score(self, confidence_score: Optional[float]) -> Optional[float]:
        """Validate confidence score."""
        if confidence_score is None:
            return None
        
        if not isinstance(confidence_score, (int, float, np.number)):
            raise TypeError("confidence_score must be numeric")
        
        if not (0 <= confidence_score <= 1):
            raise ValueError("confidence_score must be between 0 and 1")
        
        return float(confidence_score)
    
    def is_valid(self) -> bool:
        """
        Check if this explanation data is valid.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            self._validate_feature_importance(self.feature_importance)
            self._validate_explanation_type(self.explanation_type)
            self._validate_instance_data(self.instance_data)
            self._validate_model_prediction(self.model_prediction)
            self._validate_confidence_score(self.confidence_score)
            return True
        except (ValueError, TypeError):
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert explanation data to dictionary.
        
        Returns:
            Dictionary representation of explanation data
        """
        result = {
            'feature_importance': self.feature_importance,
            'explanation_type': self.explanation_type,
            'instance_data': self.instance_data,
            'model_prediction': self.model_prediction,
            'timestamp': self.timestamp
        }
        
        # Add optional fields if present
        if self.confidence_score is not None:
            result['confidence_score'] = self.confidence_score
        
        if self.explanation_metadata:
            result['explanation_metadata'] = self.explanation_metadata
        
        if self.feature_names:
            result['feature_names'] = self.feature_names
        
        if self.prediction_probability is not None:
            result['prediction_probability'] = self.prediction_probability
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExplanationData':
        """
        Create ExplanationData from dictionary.
        
        Args:
            data: Dictionary containing explanation data
            
        Returns:
            ExplanationData instance
        """
        return cls(
            feature_importance=data['feature_importance'],
            explanation_type=data['explanation_type'],
            instance_data=data['instance_data'],
            model_prediction=data['model_prediction'],
            confidence_score=data.get('confidence_score'),
            explanation_metadata=data.get('explanation_metadata'),
            feature_names=data.get('feature_names'),
            prediction_probability=data.get('prediction_probability'),
            timestamp=data.get('timestamp')
        )
    
    def to_json(self) -> str:
        """
        Convert explanation data to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExplanationData':
        """
        Create ExplanationData from JSON string.
        
        Args:
            json_str: JSON string containing explanation data
            
        Returns:
            ExplanationData instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_top_features(
        self,
        n: int = 5,
        include_negative: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get top contributing features by importance.
        
        Args:
            n: Number of top features to return
            include_negative: Whether to include negative importance features
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        features = list(self.feature_importance.items())
        
        if include_negative:
            # Sort by absolute importance
            features.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            # Filter positive and sort by importance
            features = [(name, imp) for name, imp in features if imp > 0]
            features.sort(key=lambda x: x[1], reverse=True)
        
        return features[:n]
    
    def get_feature_impact(self, feature_name: str) -> float:
        """
        Get impact score for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature importance score, 0.0 if feature not found
        """
        return self.feature_importance.get(feature_name, 0.0)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the explanation.
        
        Returns:
            Dictionary containing summary information
        """
        positive_features = [(name, imp) for name, imp in self.feature_importance.items() if imp > 0]
        negative_features = [(name, imp) for name, imp in self.feature_importance.items() if imp < 0]
        
        return {
            'explanation_type': self.explanation_type,
            'model_prediction': self.model_prediction,
            'confidence_score': self.confidence_score,
            'total_features': len(self.feature_importance),
            'positive_features_count': len(positive_features),
            'negative_features_count': len(negative_features),
            'top_positive_features': sorted(positive_features, key=lambda x: x[1], reverse=True)[:3],
            'top_negative_features': sorted(negative_features, key=lambda x: x[1])[:3],
            'timestamp': self.timestamp
        }
    
    @classmethod
    def merge_explanations(cls, explanations: List['ExplanationData']) -> 'ExplanationData':
        """
        Merge multiple explanations into a single averaged explanation.
        
        Args:
            explanations: List of ExplanationData objects to merge
            
        Returns:
            Merged ExplanationData object
        """
        if not explanations:
            raise ValueError("Cannot merge empty list of explanations")
        
        # Collect all unique features
        all_features = set()
        for exp in explanations:
            all_features.update(exp.feature_importance.keys())
        
        # Average feature importance across explanations
        merged_importance = {}
        for feature in all_features:
            scores = [exp.feature_importance.get(feature, 0.0) for exp in explanations]
            merged_importance[feature] = sum(scores) / len(scores)
        
        # Use first explanation as template for other fields
        template = explanations[0]
        
        # Average model predictions if they're numeric
        predictions = [exp.model_prediction for exp in explanations]
        if all(isinstance(p, (int, float)) for p in predictions):
            avg_prediction = sum(predictions) / len(predictions)
        else:
            avg_prediction = template.model_prediction
        
        # Average confidence scores if available
        confidence_scores = [exp.confidence_score for exp in explanations if exp.confidence_score is not None]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
        
        return cls(
            feature_importance=merged_importance,
            explanation_type='merged',
            instance_data=template.instance_data,
            model_prediction=avg_prediction,
            confidence_score=avg_confidence,
            explanation_metadata={
                'merged_from': [exp.explanation_type for exp in explanations],
                'num_explanations': len(explanations)
            }
        )