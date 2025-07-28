"""
Tests for XAI Base Explainer Interface

Following TDD principles, these tests define the expected behavior
of the base explainer interface before implementation.
"""

import pytest
from abc import ABC
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd


class TestBaseExplainerInterface:
    """Test suite for base explainer interface following TDD principles."""
    
    def test_base_explainer_is_abstract_class(self):
        """Test that BaseExplainer is an abstract base class."""
        # This test will fail until we implement BaseExplainer
        with pytest.raises(ImportError):
            from src.xai.base import BaseExplainer
    
    def test_base_explainer_has_required_abstract_methods(self):
        """Test that BaseExplainer defines required abstract methods."""
        # This test defines what methods must be implemented
        from src.xai.base import BaseExplainer
        
        # Should have abstract methods
        required_methods = [
            'explain',
            'explain_instance', 
            'get_feature_importance',
            'validate_input',
            'get_explanation_metadata'
        ]
        
        for method_name in required_methods:
            assert hasattr(BaseExplainer, method_name)
            # Should be abstract
            assert getattr(getattr(BaseExplainer, method_name), '__isabstractmethod__', False)
    
    def test_base_explainer_initialization(self):
        """Test BaseExplainer initialization parameters."""
        from src.xai.base import BaseExplainer
        
        # Mock concrete implementation for testing
        class MockExplainer(BaseExplainer):
            def explain(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): pass  
            def get_feature_importance(self, *args, **kwargs): pass
            def validate_input(self, *args, **kwargs): pass
            def get_explanation_metadata(self, *args, **kwargs): pass
        
        # Should accept model and feature names
        model = Mock()
        feature_names = ['feature_1', 'feature_2', 'feature_3']
        
        explainer = MockExplainer(model=model, feature_names=feature_names)
        
        assert explainer.model == model
        assert explainer.feature_names == feature_names
        assert hasattr(explainer, 'config')
        assert explainer.config is not None
    
    def test_base_explainer_validate_input_contract(self):
        """Test the contract for validate_input method."""
        from src.xai.base import BaseExplainer
        
        class MockExplainer(BaseExplainer):
            def explain(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): pass
            def get_feature_importance(self, *args, **kwargs): pass
            def get_explanation_metadata(self, *args, **kwargs): pass
            
            def validate_input(self, data):
                # Should return bool and optional error message
                if not isinstance(data, (np.ndarray, pd.DataFrame, list)):
                    return False, "Invalid data type"
                return True, None
        
        explainer = MockExplainer(model=Mock(), feature_names=[])
        
        # Valid input
        valid, msg = explainer.validate_input(np.array([1, 2, 3]))
        assert valid is True
        assert msg is None
        
        # Invalid input
        valid, msg = explainer.validate_input("invalid")
        assert valid is False
        assert isinstance(msg, str)
    
    def test_base_explainer_explain_method_contract(self):
        """Test the contract for explain method."""
        from src.xai.base import BaseExplainer
        from src.xai.explanation_data import ExplanationData
        
        class MockExplainer(BaseExplainer):
            def explain_instance(self, *args, **kwargs): pass
            def get_feature_importance(self, *args, **kwargs): pass
            def validate_input(self, *args, **kwargs): return True, None
            def get_explanation_metadata(self, *args, **kwargs): pass
            
            def explain(self, data, **kwargs):
                # Should return ExplanationData
                return ExplanationData(
                    feature_importance={'feature_1': 0.5},
                    explanation_type='mock',
                    instance_data=data,
                    model_prediction=0.8
                )
        
        explainer = MockExplainer(model=Mock(), feature_names=['feature_1'])
        result = explainer.explain([1, 2, 3])
        
        assert isinstance(result, ExplanationData)
        assert hasattr(result, 'feature_importance')
        assert hasattr(result, 'explanation_type')
    
    def test_base_explainer_feature_importance_format(self):
        """Test expected format for feature importance results."""
        from src.xai.base import BaseExplainer
        
        class MockExplainer(BaseExplainer):
            def explain(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): pass
            def validate_input(self, *args, **kwargs): return True, None
            def get_explanation_metadata(self, *args, **kwargs): pass
            
            def get_feature_importance(self, data, **kwargs):
                # Should return dict with feature names as keys
                return {
                    'feature_1': 0.7,
                    'feature_2': 0.3,
                    'feature_3': -0.1
                }
        
        explainer = MockExplainer(
            model=Mock(), 
            feature_names=['feature_1', 'feature_2', 'feature_3']
        )
        
        importance = explainer.get_feature_importance([1, 2, 3])
        
        assert isinstance(importance, dict)
        assert all(isinstance(k, str) for k in importance.keys())
        assert all(isinstance(v, (int, float)) for v in importance.values())
        assert len(importance) == len(explainer.feature_names)
    
    def test_base_explainer_metadata_structure(self):
        """Test expected structure for explanation metadata."""
        from src.xai.base import BaseExplainer
        
        class MockExplainer(BaseExplainer):
            def explain(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): pass
            def get_feature_importance(self, *args, **kwargs): pass
            def validate_input(self, *args, **kwargs): return True, None
            
            def get_explanation_metadata(self, **kwargs):
                return {
                    'explainer_type': 'mock',
                    'model_type': 'test_model',
                    'explanation_method': 'mock_method',
                    'confidence_threshold': 0.8,
                    'timestamp': '2024-01-01T00:00:00Z'
                }
        
        explainer = MockExplainer(model=Mock(), feature_names=[])
        metadata = explainer.get_explanation_metadata()
        
        assert isinstance(metadata, dict)
        required_keys = ['explainer_type', 'model_type', 'explanation_method']
        for key in required_keys:
            assert key in metadata
        
        # Optional metadata
        optional_keys = ['confidence_threshold', 'timestamp', 'parameters']
        # At least one optional key should be present
        assert any(key in metadata for key in optional_keys)


class TestBaseExplainerErrorHandling:
    """Test error handling in base explainer."""
    
    def test_invalid_model_raises_error(self):
        """Test that invalid model raises appropriate error."""
        from src.xai.base import BaseExplainer
        
        with pytest.raises((ValueError, TypeError)):
            class MockExplainer(BaseExplainer):
                def explain(self, *args, **kwargs): pass
                def explain_instance(self, *args, **kwargs): pass
                def get_feature_importance(self, *args, **kwargs): pass
                def validate_input(self, *args, **kwargs): pass
                def get_explanation_metadata(self, *args, **kwargs): pass
            
            MockExplainer(model=None, feature_names=['feature_1'])
    
    def test_invalid_feature_names_raises_error(self):
        """Test that invalid feature names raise appropriate error."""
        from src.xai.base import BaseExplainer
        
        with pytest.raises((ValueError, TypeError)):
            class MockExplainer(BaseExplainer):
                def explain(self, *args, **kwargs): pass
                def explain_instance(self, *args, **kwargs): pass
                def get_feature_importance(self, *args, **kwargs): pass
                def validate_input(self, *args, **kwargs): pass
                def get_explanation_metadata(self, *args, **kwargs): pass
            
            MockExplainer(model=Mock(), feature_names=None)
    
    def test_empty_feature_names_raises_error(self):
        """Test that empty feature names raise appropriate error."""
        from src.xai.base import BaseExplainer
        
        with pytest.raises(ValueError):
            class MockExplainer(BaseExplainer):
                def explain(self, *args, **kwargs): pass
                def explain_instance(self, *args, **kwargs): pass
                def get_feature_importance(self, *args, **kwargs): pass
                def validate_input(self, *args, **kwargs): pass
                def get_explanation_metadata(self, *args, **kwargs): pass
            
            MockExplainer(model=Mock(), feature_names=[])


class TestBaseExplainerConfigurationHandling:
    """Test configuration handling in base explainer."""
    
    def test_default_config_initialization(self):
        """Test that default configuration is properly initialized."""
        from src.xai.base import BaseExplainer
        
        class MockExplainer(BaseExplainer):
            def explain(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): pass
            def get_feature_importance(self, *args, **kwargs): pass
            def validate_input(self, *args, **kwargs): pass
            def get_explanation_metadata(self, *args, **kwargs): pass
        
        explainer = MockExplainer(model=Mock(), feature_names=['feature_1'])
        
        assert hasattr(explainer, 'config')
        assert isinstance(explainer.config, dict)
        
        # Should have default values
        expected_defaults = ['explanation_method', 'max_features', 'confidence_threshold']
        for key in expected_defaults:
            assert key in explainer.config
    
    def test_custom_config_override(self):
        """Test that custom configuration properly overrides defaults."""
        from src.xai.base import BaseExplainer
        
        class MockExplainer(BaseExplainer):
            def explain(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): pass
            def get_feature_importance(self, *args, **kwargs): pass
            def validate_input(self, *args, **kwargs): pass
            def get_explanation_metadata(self, *args, **kwargs): pass
        
        custom_config = {
            'confidence_threshold': 0.9,
            'max_features': 15,
            'custom_param': 'test_value'
        }
        
        explainer = MockExplainer(
            model=Mock(), 
            feature_names=['feature_1'],
            config=custom_config
        )
        
        assert explainer.config['confidence_threshold'] == 0.9
        assert explainer.config['max_features'] == 15
        assert explainer.config['custom_param'] == 'test_value'