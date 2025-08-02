"""
XAI (Explainable AI) module for trading model explanations.

This module provides explainable AI capabilities for understanding
ML and RL trading model decisions, including feature importance,
prediction explanations, and visualization tools.

Phase 5.5 Enhancement: Enhanced audit capabilities for regulatory compliance
including comprehensive audit trails, feature importance tracking, and
compliance reporting for financial trading regulations.
"""

from .base import BaseExplainer, ModelAgnosticExplainer, GradientBasedExplainer
from .data_models import ExplanationData
from .factory import ExplainerFactory
from .enhanced_audit import (
    ModelAuditTracker,
    FeatureImportanceTracker,
    DecisionExplanationGenerator,
    ComplianceReportGenerator,
    EnhancedExplanationAuditor
)

__all__ = [
    'BaseExplainer',
    'ModelAgnosticExplainer', 
    'GradientBasedExplainer',
    'ExplanationData', 
    'ExplainerFactory',
    # Phase 5.5 Enhanced Audit Capabilities
    'ModelAuditTracker',
    'FeatureImportanceTracker',
    'DecisionExplanationGenerator',
    'ComplianceReportGenerator',
    'EnhancedExplanationAuditor'
]

__version__ = '0.2.0'  # Updated for Phase 5.5 enhancements