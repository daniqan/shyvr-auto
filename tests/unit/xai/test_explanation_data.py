"""
Tests for XAI Explanation Data Structure

Following TDD principles, these tests define the expected behavior
of the explanation data structure before implementation.
"""

import pytest
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock
import numpy as np
import pandas as pd
import json
from datetime import datetime


class TestExplanationDataStructure:
    """Test suite for explanation data structure following TDD principles."""
    
    def test_explanation_data_import(self):
        """Test that ExplanationData can be imported."""
        # This test will fail until we implement ExplanationData
        from src.xai.explanation_data import ExplanationData
        assert ExplanationData is not None
    
    def test_explanation_data_initialization(self):
        """Test ExplanationData initialization with required fields."""
        from src.xai.explanation_data import ExplanationData
        
        # Required fields
        feature_importance = {'feature_1': 0.5, 'feature_2': -0.3}
        explanation_type = 'lime'
        instance_data = [1, 2, 3]
        model_prediction = 0.8
        
        explanation = ExplanationData(
            feature_importance=feature_importance,
            explanation_type=explanation_type,
            instance_data=instance_data,
            model_prediction=model_prediction
        )
        
        assert explanation.feature_importance == feature_importance
        assert explanation.explanation_type == explanation_type
        assert explanation.instance_data == instance_data
        assert explanation.model_prediction == model_prediction
    
    def test_explanation_data_optional_fields(self):
        """Test ExplanationData with optional fields."""
        from src.xai.explanation_data import ExplanationData
        
        # Optional fields
        confidence_score = 0.9
        explanation_metadata = {'method': 'lime', 'samples': 1000}
        feature_names = ['feature_1', 'feature_2']
        prediction_probability = [0.2, 0.8]
        
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8,
            confidence_score=confidence_score,
            explanation_metadata=explanation_metadata,
            feature_names=feature_names,
            prediction_probability=prediction_probability
        )
        
        assert explanation.confidence_score == confidence_score
        assert explanation.explanation_metadata == explanation_metadata
        assert explanation.feature_names == feature_names
        assert explanation.prediction_probability == prediction_probability
    
    def test_explanation_data_timestamp_auto_generation(self):
        """Test that timestamp is automatically generated if not provided."""
        from src.xai.explanation_data import ExplanationData
        
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8
        )
        
        assert hasattr(explanation, 'timestamp')
        assert explanation.timestamp is not None
        assert isinstance(explanation.timestamp, str)
        
        # Should be valid ISO timestamp
        datetime.fromisoformat(explanation.timestamp.replace('Z', '+00:00'))
    
    def test_explanation_data_custom_timestamp(self):
        """Test that custom timestamp can be provided."""
        from src.xai.explanation_data import ExplanationData
        
        custom_timestamp = '2024-01-01T12:00:00Z'
        
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8,
            timestamp=custom_timestamp
        )
        
        assert explanation.timestamp == custom_timestamp


class TestExplanationDataValidation:
    """Test validation logic for explanation data."""
    
    def test_feature_importance_validation(self):
        """Test validation of feature importance data."""
        from src.xai.explanation_data import ExplanationData
        
        # Valid feature importance
        valid_importance = {'feature_1': 0.5, 'feature_2': -0.3}
        explanation = ExplanationData(
            feature_importance=valid_importance,
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8
        )
        assert explanation.is_valid()
        
        # Invalid feature importance - not a dict
        with pytest.raises((ValueError, TypeError)):
            ExplanationData(
                feature_importance=[0.5, -0.3],  # Should be dict
                explanation_type='lime',
                instance_data=[1, 2],
                model_prediction=0.8
            )
        
        # Invalid feature importance - non-numeric values
        with pytest.raises((ValueError, TypeError)):
            ExplanationData(
                feature_importance={'feature_1': 'invalid'},
                explanation_type='lime',
                instance_data=[1, 2],
                model_prediction=0.8
            )
    
    def test_explanation_type_validation(self):
        """Test validation of explanation type."""
        from src.xai.explanation_data import ExplanationData
        
        # Valid explanation types
        valid_types = ['lime', 'shap', 'permutation', 'grad_cam', 'custom']
        
        for exp_type in valid_types:
            explanation = ExplanationData(
                feature_importance={'feature_1': 0.5},
                explanation_type=exp_type,
                instance_data=[1, 2],
                model_prediction=0.8
            )
            assert explanation.is_valid()
        
        # Invalid explanation type
        with pytest.raises(ValueError):
            ExplanationData(
                feature_importance={'feature_1': 0.5},
                explanation_type='invalid_type',
                instance_data=[1, 2],
                model_prediction=0.8
            )
    
    def test_model_prediction_validation(self):
        """Test validation of model prediction."""
        from src.xai.explanation_data import ExplanationData
        
        # Valid numeric prediction
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8
        )
        assert explanation.is_valid()
        
        # Valid array prediction (for multi-class)
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=[0.2, 0.8]
        )
        assert explanation.is_valid()
        
        # Invalid prediction type
        with pytest.raises((ValueError, TypeError)):
            ExplanationData(
                feature_importance={'feature_1': 0.5},
                explanation_type='lime',
                instance_data=[1, 2],
                model_prediction='invalid'
            )
    
    def test_confidence_score_validation(self):
        """Test validation of confidence score."""
        from src.xai.explanation_data import ExplanationData
        
        # Valid confidence score (0-1)
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8,
            confidence_score=0.9
        )
        assert explanation.is_valid()
        
        # Invalid confidence score - out of range
        with pytest.raises(ValueError):
            ExplanationData(
                feature_importance={'feature_1': 0.5},
                explanation_type='lime',
                instance_data=[1, 2],
                model_prediction=0.8,
                confidence_score=1.5
            )
        
        with pytest.raises(ValueError):
            ExplanationData(
                feature_importance={'feature_1': 0.5},
                explanation_type='lime',
                instance_data=[1, 2],
                model_prediction=0.8,
                confidence_score=-0.1
            )


class TestExplanationDataSerialization:
    """Test serialization and deserialization of explanation data."""
    
    def test_to_dict_method(self):
        """Test conversion to dictionary."""
        from src.xai.explanation_data import ExplanationData
        
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5, 'feature_2': -0.3},
            explanation_type='lime',
            instance_data=[1, 2, 3],
            model_prediction=0.8,
            confidence_score=0.9
        )
        
        result_dict = explanation.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'feature_importance' in result_dict
        assert 'explanation_type' in result_dict
        assert 'instance_data' in result_dict
        assert 'model_prediction' in result_dict
        assert 'confidence_score' in result_dict
        assert 'timestamp' in result_dict
        
        # Values should match
        assert result_dict['feature_importance'] == explanation.feature_importance
        assert result_dict['explanation_type'] == explanation.explanation_type
    
    def test_from_dict_method(self):
        """Test creation from dictionary."""
        from src.xai.explanation_data import ExplanationData
        
        data_dict = {
            'feature_importance': {'feature_1': 0.5, 'feature_2': -0.3},
            'explanation_type': 'lime',
            'instance_data': [1, 2, 3],
            'model_prediction': 0.8,
            'confidence_score': 0.9,
            'timestamp': '2024-01-01T12:00:00Z'
        }
        
        explanation = ExplanationData.from_dict(data_dict)
        
        assert explanation.feature_importance == data_dict['feature_importance']
        assert explanation.explanation_type == data_dict['explanation_type']
        assert explanation.instance_data == data_dict['instance_data']
        assert explanation.model_prediction == data_dict['model_prediction']
        assert explanation.confidence_score == data_dict['confidence_score']
        assert explanation.timestamp == data_dict['timestamp']
    
    def test_to_json_method(self):
        """Test JSON serialization."""
        from src.xai.explanation_data import ExplanationData
        
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8
        )
        
        json_str = explanation.to_json()
        
        assert isinstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert 'feature_importance' in parsed
    
    def test_from_json_method(self):
        """Test JSON deserialization."""
        from src.xai.explanation_data import ExplanationData
        
        json_str = json.dumps({
            'feature_importance': {'feature_1': 0.5},
            'explanation_type': 'lime',
            'instance_data': [1, 2],
            'model_prediction': 0.8,
            'timestamp': '2024-01-01T12:00:00Z'
        })
        
        explanation = ExplanationData.from_json(json_str)
        
        assert explanation.feature_importance == {'feature_1': 0.5}
        assert explanation.explanation_type == 'lime'
        assert explanation.instance_data == [1, 2]
        assert explanation.model_prediction == 0.8


class TestExplanationDataUtilityMethods:
    """Test utility methods of explanation data."""
    
    def test_get_top_features_method(self):
        """Test getting top contributing features."""
        from src.xai.explanation_data import ExplanationData
        
        explanation = ExplanationData(
            feature_importance={
                'feature_1': 0.8,
                'feature_2': -0.6,
                'feature_3': 0.4,
                'feature_4': -0.2,
                'feature_5': 0.1
            },
            explanation_type='lime',
            instance_data=[1, 2, 3, 4, 5],
            model_prediction=0.8
        )
        
        # Get top 3 positive features
        top_positive = explanation.get_top_features(n=3, include_negative=False)
        assert len(top_positive) == 3
        assert top_positive[0][0] == 'feature_1'  # (name, importance)
        assert top_positive[0][1] == 0.8
        
        # Get top 3 features including negative
        top_all = explanation.get_top_features(n=3, include_negative=True)
        assert len(top_all) == 3
        # Should be sorted by absolute importance
        assert abs(top_all[0][1]) >= abs(top_all[1][1])
    
    def test_get_feature_impact_method(self):
        """Test getting impact of specific feature."""
        from src.xai.explanation_data import ExplanationData
        
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5, 'feature_2': -0.3},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8
        )
        
        # Get impact of existing feature
        impact = explanation.get_feature_impact('feature_1')
        assert impact == 0.5
        
        # Get impact of non-existing feature
        impact = explanation.get_feature_impact('feature_nonexistent')
        assert impact == 0.0  # Should return 0 for missing features
    
    def test_summary_method(self):
        """Test explanation summary generation."""
        from src.xai.explanation_data import ExplanationData
        
        explanation = ExplanationData(
            feature_importance={'feature_1': 0.5, 'feature_2': -0.3},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8,
            confidence_score=0.9
        )
        
        summary = explanation.get_summary()
        
        assert isinstance(summary, dict)
        assert 'explanation_type' in summary
        assert 'model_prediction' in summary
        assert 'confidence_score' in summary
        assert 'top_positive_features' in summary
        assert 'top_negative_features' in summary
        assert 'total_features' in summary
    
    def test_merge_explanations_method(self):
        """Test merging multiple explanations."""
        from src.xai.explanation_data import ExplanationData
        
        explanation1 = ExplanationData(
            feature_importance={'feature_1': 0.5, 'feature_2': -0.3},
            explanation_type='lime',
            instance_data=[1, 2],
            model_prediction=0.8
        )
        
        explanation2 = ExplanationData(
            feature_importance={'feature_1': 0.6, 'feature_3': 0.4},
            explanation_type='shap',
            instance_data=[1, 2, 3],
            model_prediction=0.7
        )
        
        merged = ExplanationData.merge_explanations([explanation1, explanation2])
        
        assert isinstance(merged, ExplanationData)
        assert merged.explanation_type == 'merged'
        
        # Should average feature importance for common features
        assert 'feature_1' in merged.feature_importance
        assert merged.feature_importance['feature_1'] == (0.5 + 0.6) / 2
        
        # Should include unique features
        assert 'feature_2' in merged.feature_importance
        assert 'feature_3' in merged.feature_importance