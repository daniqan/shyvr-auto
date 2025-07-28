"""
XAI (Explainable AI) module for trading model explanations.

This module provides explainable AI capabilities for understanding
ML and RL trading model decisions, including feature importance,
prediction explanations, and visualization tools.
"""

from .base import BaseExplainer, ModelAgnosticExplainer, GradientBasedExplainer
from .data_models import ExplanationData
from .factory import ExplainerFactory

__all__ = [
    'BaseExplainer',
    'ModelAgnosticExplainer', 
    'GradientBasedExplainer',
    'ExplanationData', 
    'ExplainerFactory'
]

__version__ = '0.1.0'