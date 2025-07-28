"""
Tests for XAI Explainer Factory

Following TDD principles, these tests define the expected behavior
of the explainer factory before implementation.
"""

import pytest
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch
import numpy as np


class TestExplainerFactory:
    """Test suite for explainer factory following TDD principles."""
    
    def test_explainer_factory_import(self):
        """Test that ExplainerFactory can be imported."""
        # This test will fail until we implement ExplainerFactory
        from src.xai.explainer_factory import ExplainerFactory
        assert ExplainerFactory is not None
    
    def test_explainer_factory_initialization(self):
        """Test ExplainerFactory initialization."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        assert factory is not None
        assert hasattr(factory, 'supported_explainers')
        assert isinstance(factory.supported_explainers, (list, dict))
    
    def test_supported_explainer_types(self):
        """Test that factory supports expected explainer types."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        # Should support these basic explainer types
        expected_types = ['lime', 'permutation', 'gradient']
        
        for explainer_type in expected_types:
            assert factory.is_supported(explainer_type)
    
    def test_create_explainer_method(self):
        """Test creating explainers through factory."""
        from src.xai.explainer_factory import ExplainerFactory
        from src.xai.base import BaseExplainer
        
        factory = ExplainerFactory()
        model = Mock()
        feature_names = ['feature_1', 'feature_2', 'feature_3']
        
        # Create LIME explainer
        lime_explainer = factory.create_explainer(
            explainer_type='lime',
            model=model,
            feature_names=feature_names
        )
        
        assert lime_explainer is not None
        assert isinstance(lime_explainer, BaseExplainer)
        assert lime_explainer.model == model
        assert lime_explainer.feature_names == feature_names
    
    def test_create_explainer_with_config(self):
        """Test creating explainer with custom configuration."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        model = Mock()
        feature_names = ['feature_1', 'feature_2']
        
        config = {
            'num_samples': 5000,
            'kernel_width': 0.75,
            'distance_metric': 'euclidean'
        }
        
        explainer = factory.create_explainer(
            explainer_type='lime',
            model=model,
            feature_names=feature_names,
            config=config
        )
        
        assert explainer is not None
        # Should have passed config to explainer
        assert hasattr(explainer, 'config')
        assert explainer.config['num_samples'] == 5000
        assert explainer.config['kernel_width'] == 0.75
    
    def test_unsupported_explainer_type_raises_error(self):
        """Test that unsupported explainer type raises appropriate error."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        with pytest.raises(ValueError, match="Unsupported explainer type"):
            factory.create_explainer(
                explainer_type='unsupported_type',
                model=Mock(),
                feature_names=['feature_1']
            )
    
    def test_invalid_model_raises_error(self):
        """Test that invalid model raises appropriate error."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        with pytest.raises((ValueError, TypeError)):
            factory.create_explainer(
                explainer_type='lime',
                model=None,
                feature_names=['feature_1']
            )
    
    def test_invalid_feature_names_raises_error(self):
        """Test that invalid feature names raise appropriate error."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        with pytest.raises((ValueError, TypeError)):
            factory.create_explainer(
                explainer_type='lime',
                model=Mock(),
                feature_names=None
            )


class TestExplainerFactoryRegistration:
    """Test dynamic registration of explainer types."""
    
    def test_register_custom_explainer(self):
        """Test registering custom explainer type."""
        from src.xai.explainer_factory import ExplainerFactory
        from src.xai.base import BaseExplainer
        
        # Create mock custom explainer class
        class CustomExplainer(BaseExplainer):
            def explain(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): pass
            def get_feature_importance(self, *args, **kwargs): pass
            def validate_input(self, *args, **kwargs): return True, None
            def get_explanation_metadata(self, *args, **kwargs): pass
        
        factory = ExplainerFactory()
        
        # Register custom explainer
        factory.register_explainer('custom', CustomExplainer)
        
        assert factory.is_supported('custom')
        
        # Should be able to create custom explainer
        explainer = factory.create_explainer(
            explainer_type='custom',
            model=Mock(),
            feature_names=['feature_1']
        )
        
        assert isinstance(explainer, CustomExplainer)
    
    def test_override_existing_explainer(self):
        """Test overriding existing explainer type."""
        from src.xai.explainer_factory import ExplainerFactory
        from src.xai.base import BaseExplainer
        
        class CustomLimeExplainer(BaseExplainer):
            def explain(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): pass
            def get_feature_importance(self, *args, **kwargs): pass
            def validate_input(self, *args, **kwargs): return True, None
            def get_explanation_metadata(self, *args, **kwargs): pass
        
        factory = ExplainerFactory()
        
        # Override LIME explainer
        factory.register_explainer('lime', CustomLimeExplainer, override=True)
        
        explainer = factory.create_explainer(
            explainer_type='lime',
            model=Mock(),
            feature_names=['feature_1']
        )
        
        assert isinstance(explainer, CustomLimeExplainer)
    
    def test_register_without_override_raises_error(self):
        """Test that registering existing type without override raises error."""
        from src.xai.explainer_factory import ExplainerFactory
        from src.xai.base import BaseExplainer
        
        class CustomExplainer(BaseExplainer):
            def explain(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): pass
            def get_feature_importance(self, *args, **kwargs): pass
            def validate_input(self, *args, **kwargs): return True, None
            def get_explanation_metadata(self, *args, **kwargs): pass
        
        factory = ExplainerFactory()
        
        # Try to register over existing without override
        with pytest.raises(ValueError, match="already registered"):
            factory.register_explainer('lime', CustomExplainer, override=False)


class TestExplainerFactoryModelTypeValidation:
    """Test model type validation in factory."""
    
    def test_supported_model_types(self):
        """Test that factory validates supported model types."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        # Should support common ML model types
        supported_models = [
            Mock(__class__=Mock(__name__='sklearn.ensemble.RandomForestClassifier')),
            Mock(__class__=Mock(__name__='sklearn.linear_model.LogisticRegression')),
            Mock(__class__=Mock(__name__='torch.nn.Module')),
            Mock(__class__=Mock(__name__='lightgbm.LGBMClassifier')),
        ]
        
        for model in supported_models:
            # Should not raise error
            explainer = factory.create_explainer(
                explainer_type='lime',
                model=model,
                feature_names=['feature_1']
            )
            assert explainer is not None
    
    def test_model_validation_for_explainer_type(self):
        """Test that different explainer types validate models differently."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        # Some explainers might have specific model requirements
        sklearn_model = Mock(__class__=Mock(__name__='sklearn.ensemble.RandomForestClassifier'))
        torch_model = Mock(__class__=Mock(__name__='torch.nn.Module'))
        
        # Both should work with LIME
        lime_sklearn = factory.create_explainer(
            explainer_type='lime',
            model=sklearn_model,
            feature_names=['feature_1']
        )
        assert lime_sklearn is not None
        
        lime_torch = factory.create_explainer(
            explainer_type='lime',
            model=torch_model,
            feature_names=['feature_1']
        )
        assert lime_torch is not None


class TestExplainerFactoryUtilityMethods:
    """Test utility methods of explainer factory."""
    
    def test_list_supported_explainers(self):
        """Test listing all supported explainer types."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        supported = factory.list_supported_explainers()
        
        assert isinstance(supported, list)
        assert len(supported) > 0
        assert 'lime' in supported
        assert 'permutation' in supported
    
    def test_get_explainer_info(self):
        """Test getting information about specific explainer type."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        lime_info = factory.get_explainer_info('lime')
        
        assert isinstance(lime_info, dict)
        assert 'name' in lime_info
        assert 'description' in lime_info
        assert 'supported_model_types' in lime_info
        assert 'default_config' in lime_info
    
    def test_get_default_config(self):
        """Test getting default configuration for explainer type."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        lime_config = factory.get_default_config('lime')
        
        assert isinstance(lime_config, dict)
        assert 'num_samples' in lime_config
        assert 'kernel_width' in lime_config
        
        # Config values should be reasonable defaults
        assert isinstance(lime_config['num_samples'], int)
        assert lime_config['num_samples'] > 0
        assert isinstance(lime_config['kernel_width'], (int, float))
        assert lime_config['kernel_width'] > 0
    
    def test_validate_config(self):
        """Test configuration validation for explainer types."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        
        # Valid config
        valid_config = {
            'num_samples': 1000,
            'kernel_width': 0.75
        }
        
        is_valid, errors = factory.validate_config('lime', valid_config)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid config
        invalid_config = {
            'num_samples': -100,  # Invalid negative value
            'kernel_width': 'invalid'  # Invalid type
        }
        
        is_valid, errors = factory.validate_config('lime', invalid_config)
        assert is_valid is False
        assert len(errors) > 0
        assert any('num_samples' in error for error in errors)
        assert any('kernel_width' in error for error in errors)


class TestExplainerFactoryIntegration:
    """Test integration scenarios for explainer factory."""
    
    def test_create_multiple_explainers(self):
        """Test creating multiple explainers for same model."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory()
        model = Mock()
        feature_names = ['feature_1', 'feature_2']
        
        # Create different types of explainers
        lime_explainer = factory.create_explainer(
            explainer_type='lime',
            model=model,
            feature_names=feature_names
        )
        
        permutation_explainer = factory.create_explainer(
            explainer_type='permutation',
            model=model,
            feature_names=feature_names
        )
        
        assert lime_explainer is not None
        assert permutation_explainer is not None
        assert type(lime_explainer) != type(permutation_explainer)
        
        # Both should work with same model
        assert lime_explainer.model == model
        assert permutation_explainer.model == model
    
    def test_factory_caching_behavior(self):
        """Test that factory can cache created explainers if needed."""
        from src.xai.explainer_factory import ExplainerFactory
        
        factory = ExplainerFactory(cache_explainers=True)
        model = Mock()
        feature_names = ['feature_1']
        
        # Create explainer twice with same parameters
        explainer1 = factory.create_explainer(
            explainer_type='lime',
            model=model,
            feature_names=feature_names
        )
        
        explainer2 = factory.create_explainer(
            explainer_type='lime',
            model=model,
            feature_names=feature_names
        )
        
        # Should return same instance if caching is enabled
        if factory.cache_explainers:
            assert explainer1 is explainer2
        else:
            assert explainer1 is not explainer2