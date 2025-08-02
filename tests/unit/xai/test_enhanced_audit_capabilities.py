"""
Test suite for enhanced XAI audit capabilities.

This module tests the Phase 5.5 enhanced model explainability features
for regulatory compliance, including audit trails, feature importance tracking,
and decision explanation generation.

Following TDD methodology - these tests are written first and should fail initially.
"""

import pytest
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from unittest.mock import Mock
import json

from src.xai.enhanced_audit import (
    ModelAuditTracker,
    FeatureImportanceTracker,
    DecisionExplanationGenerator,
    ComplianceReportGenerator,
    EnhancedExplanationAuditor
)
from src.xai.data_models import ExplanationData


class TestModelAuditTracker:
    """Test Model Audit Tracker for regulatory compliance."""
    
    def test_audit_tracker_initialization(self):
        """Test proper initialization of audit tracker."""
        tracker = ModelAuditTracker(
            audit_storage_path="/tmp/audit",
            retention_days=90,
            compliance_level="strict"
        )
        
        assert tracker.audit_storage_path == "/tmp/audit"
        assert tracker.retention_days == 90
        assert tracker.compliance_level == "strict"
        assert tracker.is_enabled is True
        
    def test_record_model_decision_audit(self):
        """Test recording model decision for audit trail."""
        tracker = ModelAuditTracker()
        
        decision_data = {
            "model_id": "lstm_v1.2.3",
            "decision_type": "buy",
            "symbol": "BTC-USD",
            "confidence": 0.85,
            "features": {"price": 50000, "volume": 1000000},
            "prediction": 0.75,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        audit_id = tracker.record_decision(decision_data)
        
        assert audit_id is not None
        assert isinstance(audit_id, str)
        assert len(audit_id) > 0
        
        # Verify decision can be retrieved
        retrieved = tracker.get_decision(audit_id)
        assert retrieved is not None
        assert retrieved["model_id"] == "lstm_v1.2.3"
        assert retrieved["decision_type"] == "buy"
        
    def test_audit_trail_query_by_time_range(self):
        """Test querying audit trail by time range."""
        tracker = ModelAuditTracker()
        
        # Record multiple decisions
        base_time = datetime.utcnow()
        
        for i in range(5):
            decision_time = base_time + timedelta(minutes=i)
            decision_data = {
                "model_id": f"model_v{i}",
                "decision_type": "hold",
                "symbol": "ETH-USD",
                "timestamp": decision_time.isoformat()
            }
            tracker.record_decision(decision_data)
        
        # Query decisions in time range
        start_time = base_time + timedelta(minutes=1)
        end_time = base_time + timedelta(minutes=3)
        
        decisions = tracker.query_decisions_by_time_range(start_time, end_time)
        
        assert len(decisions) == 3  # Should get decisions at minutes 1, 2, 3
        
    def test_audit_trail_query_by_model(self):
        """Test querying audit trail by model ID."""
        tracker = ModelAuditTracker()
        
        # Record decisions for different models
        for model_id in ["lstm_v1", "dqn_v2", "lstm_v1"]:
            decision_data = {
                "model_id": model_id,
                "decision_type": "sell",
                "symbol": "SOL-USD"
            }
            tracker.record_decision(decision_data)
        
        # Query decisions for specific model
        lstm_decisions = tracker.query_decisions_by_model("lstm_v1")
        
        assert len(lstm_decisions) == 2
        assert all(d["model_id"] == "lstm_v1" for d in lstm_decisions)
        
    def test_compliance_report_generation(self):
        """Test generating compliance reports for regulatory audit."""
        tracker = ModelAuditTracker()
        
        # Record sample decisions
        for i in range(10):
            decision_data = {
                "model_id": "production_model",
                "decision_type": "buy" if i % 2 == 0 else "sell",
                "symbol": "BTC-USD",
                "confidence": 0.7 + (i * 0.02),
                "risk_score": 0.3 + (i * 0.01)
            }
            tracker.record_decision(decision_data)
        
        # Generate compliance report
        report = tracker.generate_compliance_report(
            start_date=datetime.utcnow() - timedelta(days=1),
            end_date=datetime.utcnow()
        )
        
        assert "total_decisions" in report
        assert "model_usage_summary" in report
        assert "risk_distribution" in report
        assert "compliance_metrics" in report
        assert report["total_decisions"] == 10
        
    def test_audit_data_retention_policy(self):
        """Test automatic cleanup of old audit data."""
        tracker = ModelAuditTracker(retention_days=7)
        
        # Record old decision (should be cleaned up)
        old_time = datetime.utcnow() - timedelta(days=10)
        old_decision = {
            "model_id": "old_model",
            "timestamp": old_time.isoformat()
        }
        tracker.record_decision(old_decision)
        
        # Record recent decision (should be kept)
        recent_decision = {
            "model_id": "recent_model",
            "timestamp": datetime.utcnow().isoformat()
        }
        tracker.record_decision(recent_decision)
        
        # Apply retention policy
        cleaned_count = tracker.apply_retention_policy()
        
        assert cleaned_count == 1
        
        # Verify only recent decision remains
        all_decisions = tracker.get_all_decisions()
        assert len(all_decisions) == 1
        assert all_decisions[0]["model_id"] == "recent_model"


class TestFeatureImportanceTracker:
    """Test Feature Importance Tracking for analysis and compliance."""
    
    def test_feature_importance_tracker_initialization(self):
        """Test proper initialization of feature importance tracker."""
        tracker = FeatureImportanceTracker(
            window_size=100,
            tracking_enabled=True
        )
        
        assert tracker.window_size == 100
        assert tracker.tracking_enabled is True
        assert len(tracker.importance_history) == 0
        
    def test_record_feature_importance(self):
        """Test recording feature importance for a decision."""
        tracker = FeatureImportanceTracker()
        
        importance_data = {
            "price": 0.35,
            "volume": 0.25,
            "rsi": 0.20,
            "macd": 0.15,
            "bollinger": 0.05
        }
        
        record_id = tracker.record_importance(
            model_id="lstm_v1",
            decision_id="dec_123",
            feature_importance=importance_data,
            symbol="BTC-USD"
        )
        
        assert record_id is not None
        assert isinstance(record_id, str)
        
        # Verify importance was recorded
        history = tracker.get_importance_history("lstm_v1")
        assert len(history) == 1
        assert history[0]["feature_importance"] == importance_data
        
    def test_aggregate_feature_importance_over_time(self):
        """Test aggregating feature importance over time windows."""
        tracker = FeatureImportanceTracker(window_size=50)
        
        # Record multiple importance measurements
        for i in range(20):
            importance_data = {
                "feature_a": 0.4 + (i * 0.01),
                "feature_b": 0.3 - (i * 0.005),
                "feature_c": 0.3
            }
            tracker.record_importance(
                model_id="test_model",
                decision_id=f"dec_{i}",
                feature_importance=importance_data
            )
        
        # Get aggregated importance
        aggregated = tracker.get_aggregated_importance(
            model_id="test_model",
            window_size=10
        )
        
        assert "feature_a" in aggregated
        assert "feature_b" in aggregated
        assert "feature_c" in aggregated
        assert abs(aggregated["feature_c"] - 0.3) < 0.01  # Should be stable
        
    def test_feature_importance_trend_analysis(self):
        """Test trend analysis for feature importance changes."""
        tracker = FeatureImportanceTracker()
        
        # Record importance data with clear trend
        for i in range(30):
            importance_data = {
                "trending_up": 0.1 + (i * 0.02),  # Increasing trend
                "trending_down": 0.9 - (i * 0.02),  # Decreasing trend
                "stable": 0.5  # Stable feature
            }
            tracker.record_importance(
                model_id="trend_model",
                decision_id=f"trend_dec_{i}",
                feature_importance=importance_data
            )
        
        # Analyze trends
        trends = tracker.analyze_importance_trends(
            model_id="trend_model",
            lookback_window=20
        )
        
        assert "trending_up" in trends
        assert "trending_down" in trends
        assert "stable" in trends
        
        assert trends["trending_up"]["trend"] == "increasing"
        assert trends["trending_down"]["trend"] == "decreasing"
        assert trends["stable"]["trend"] == "stable"
        
    def test_feature_importance_anomaly_detection(self):
        """Test anomaly detection in feature importance patterns."""
        tracker = FeatureImportanceTracker()
        
        # Record normal importance patterns
        for i in range(50):
            normal_importance = {
                "price": 0.4 + np.random.normal(0, 0.05),
                "volume": 0.3 + np.random.normal(0, 0.03),
                "technical": 0.3 + np.random.normal(0, 0.02)
            }
            tracker.record_importance(
                model_id="anomaly_model",
                decision_id=f"normal_{i}",
                feature_importance=normal_importance
            )
        
        # Record anomalous importance pattern
        anomalous_importance = {
            "price": 0.9,  # Unusual spike
            "volume": 0.05,  # Unusual drop
            "technical": 0.05
        }
        anomaly_id = tracker.record_importance(
            model_id="anomaly_model",
            decision_id="anomaly_1",
            feature_importance=anomalous_importance
        )
        
        # Detect anomalies
        anomalies = tracker.detect_importance_anomalies(
            model_id="anomaly_model",
            threshold=2.0  # 2 standard deviations
        )
        
        assert len(anomalies) > 0
        assert any(a["decision_id"] == "anomaly_1" for a in anomalies)


class TestDecisionExplanationGenerator:
    """Test Decision Explanation Generator for human-readable explanations."""
    
    def test_explanation_generator_initialization(self):
        """Test proper initialization of explanation generator."""
        generator = DecisionExplanationGenerator(
            explanation_templates_path="/tmp/templates",
            language="en",
            detail_level="detailed"
        )
        
        assert generator.language == "en"
        assert generator.detail_level == "detailed"
        assert generator.templates_loaded is True
        
    def test_generate_trading_decision_explanation(self):
        """Test generating human-readable trading decision explanations."""
        generator = DecisionExplanationGenerator()
        
        decision_context = {
            "decision_type": "buy",
            "symbol": "BTC-USD",
            "confidence": 0.85,
            "price": 45000,
            "prediction": 0.75
        }
        
        feature_importance = {
            "price_momentum": 0.35,
            "volume_indicator": 0.25,
            "rsi": 0.20,
            "macd": 0.15,
            "support_resistance": 0.05
        }
        
        explanation = generator.generate_explanation(
            decision_context=decision_context,
            feature_importance=feature_importance,
            model_type="lstm"
        )
        
        assert isinstance(explanation, dict)
        assert "summary" in explanation
        assert "detailed_reasoning" in explanation
        assert "confidence_explanation" in explanation
        assert "risk_factors" in explanation
        assert "key_features" in explanation
        
        # Verify explanation content
        assert "BTC-USD" in explanation["summary"]
        assert "buy" in explanation["summary"].lower()
        assert "85%" in explanation["confidence_explanation"]
        
    def test_generate_risk_warning_explanations(self):
        """Test generating risk warning explanations for high-risk decisions."""
        generator = DecisionExplanationGenerator()
        
        high_risk_context = {
            "decision_type": "buy",
            "symbol": "VOLATILE-COIN",
            "confidence": 0.55,  # Lower confidence
            "risk_score": 0.85,  # High risk
            "volatility": 0.45
        }
        
        explanation = generator.generate_explanation(
            decision_context=high_risk_context,
            include_warnings=True
        )
        
        assert "risk_warnings" in explanation
        assert len(explanation["risk_warnings"]) > 0
        assert any("high risk" in w.lower() for w in explanation["risk_warnings"])
        assert any("volatility" in w.lower() for w in explanation["risk_warnings"])
        
    def test_explanation_templates_customization(self):
        """Test customization of explanation templates."""
        generator = DecisionExplanationGenerator()
        
        custom_template = {
            "buy_summary": "Model recommends BUYING {symbol} with {confidence}% confidence based on {top_feature}",
            "risk_high": "⚠️ HIGH RISK: This decision has elevated risk factors"
        }
        
        generator.update_templates(custom_template)
        
        decision_context = {
            "decision_type": "buy",
            "symbol": "ETH-USD",
            "confidence": 0.78
        }
        
        feature_importance = {"momentum": 0.6, "volume": 0.4}
        
        explanation = generator.generate_explanation(
            decision_context=decision_context,
            feature_importance=feature_importance
        )
        
        assert "ETH-USD" in explanation["summary"]
        assert "78%" in explanation["summary"]
        assert "momentum" in explanation["summary"]
        
    def test_multilingual_explanation_support(self):
        """Test support for multiple languages in explanations."""
        # English generator
        en_generator = DecisionExplanationGenerator(language="en")
        
        # Spanish generator (if supported)
        es_generator = DecisionExplanationGenerator(language="es")
        
        decision_context = {
            "decision_type": "sell",
            "symbol": "BTC-USD",
            "confidence": 0.92
        }
        
        en_explanation = en_generator.generate_explanation(decision_context)
        es_explanation = es_generator.generate_explanation(decision_context)
        
        assert en_explanation["summary"] != es_explanation["summary"]
        assert "sell" in en_explanation["summary"].lower()
        assert ("vender" in es_explanation["summary"].lower() or 
                "sell" in es_explanation["summary"].lower())  # Fallback to English


class TestComplianceReportGenerator:
    """Test Compliance Report Generator for regulatory documentation."""
    
    def test_compliance_report_generator_initialization(self):
        """Test proper initialization of compliance report generator."""
        generator = ComplianceReportGenerator(
            report_format="json",
            compliance_standards=["MiFID II", "GDPR"],
            output_directory="/tmp/compliance"
        )
        
        assert generator.report_format == "json"
        assert "MiFID II" in generator.compliance_standards
        assert generator.output_directory == "/tmp/compliance"
        
    def test_generate_model_explainability_report(self):
        """Test generating comprehensive model explainability reports."""
        generator = ComplianceReportGenerator()
        
        # Sample audit data
        audit_data = {
            "total_decisions": 1000,
            "models_used": ["lstm_v1.2.3", "dqn_v2.1.0"],
            "decision_distribution": {"buy": 400, "sell": 350, "hold": 250},
            "average_confidence": 0.78,
            "explanation_coverage": 0.95
        }
        
        feature_analysis = {
            "most_important_features": ["price", "volume", "rsi"],
            "feature_stability": {"price": 0.95, "volume": 0.88, "rsi": 0.92},
            "anomalies_detected": 5
        }
        
        report = generator.generate_explainability_report(
            audit_data=audit_data,
            feature_analysis=feature_analysis,
            time_period="2025-08-01 to 2025-08-02"
        )
        
        assert "executive_summary" in report
        assert "model_usage_analysis" in report
        assert "feature_importance_analysis" in report
        assert "explainability_metrics" in report
        assert "compliance_assessment" in report
        assert "recommendations" in report
        
        # Verify content
        assert report["explainability_metrics"]["explanation_coverage"] == 0.95
        assert len(report["model_usage_analysis"]["models_used"]) == 2
        
    def test_regulatory_compliance_validation(self):
        """Test validation against specific regulatory requirements."""
        generator = ComplianceReportGenerator(
            compliance_standards=["MiFID II", "SEC Rule 3a-4"]
        )
        
        model_data = {
            "model_id": "production_lstm",
            "explanation_coverage": 0.98,
            "audit_trail_completeness": 1.0,
            "human_readable_explanations": True,
            "risk_disclosure_present": True
        }
        
        compliance_result = generator.validate_compliance(model_data)
        
        assert "MiFID II" in compliance_result
        assert "SEC Rule 3a-4" in compliance_result
        assert compliance_result["MiFID II"]["compliant"] is True
        assert compliance_result["SEC Rule 3a-4"]["compliant"] is True
        assert "audit_trail" in compliance_result["MiFID II"]["requirements_met"]
        
    def test_export_compliance_documentation(self):
        """Test exporting compliance documentation in different formats."""
        generator = ComplianceReportGenerator()
        
        report_data = {
            "report_id": "COMP_2025_08_02_001",
            "generated_at": datetime.utcnow().isoformat(),
            "compliance_status": "COMPLIANT",
            "findings": ["All requirements met", "No critical issues"],
            "model_performance": {"accuracy": 0.85, "explainability": 0.92}
        }
        
        # Test JSON export
        json_path = generator.export_report(report_data, format="json")
        assert json_path.endswith(".json")
        
        # Test PDF export (if available)
        try:
            pdf_path = generator.export_report(report_data, format="pdf")
            assert pdf_path.endswith(".pdf")
        except ImportError:
            # PDF export dependencies not available
            pass
        
        # Test HTML export
        html_path = generator.export_report(report_data, format="html")
        assert html_path.endswith(".html")


class TestEnhancedExplanationAuditor:
    """Test Enhanced Explanation Auditor for comprehensive audit capabilities."""
    
    def test_enhanced_auditor_initialization(self):
        """Test proper initialization of enhanced explanation auditor."""
        auditor = EnhancedExplanationAuditor(
            audit_tracker=Mock(),
            importance_tracker=Mock(),
            explanation_generator=Mock(),
            compliance_generator=Mock()
        )
        
        assert auditor.audit_tracker is not None
        assert auditor.importance_tracker is not None
        assert auditor.explanation_generator is not None
        assert auditor.compliance_generator is not None
        assert auditor.is_enabled is True
        
    def test_comprehensive_decision_audit(self):
        """Test comprehensive auditing of trading decisions."""
        # Mock components
        audit_tracker = Mock()
        importance_tracker = Mock()
        explanation_generator = Mock()
        compliance_generator = Mock()
        
        auditor = EnhancedExplanationAuditor(
            audit_tracker=audit_tracker,
            importance_tracker=importance_tracker,
            explanation_generator=explanation_generator,
            compliance_generator=compliance_generator
        )
        
        # Mock return values
        audit_tracker.record_decision.return_value = "audit_123"
        importance_tracker.record_importance.return_value = "importance_456"
        explanation_generator.generate_explanation.return_value = {
            "summary": "Buy recommendation based on technical indicators"
        }
        
        # Test comprehensive audit
        decision_data = {
            "model_id": "lstm_v1",
            "decision_type": "buy",
            "symbol": "BTC-USD",
            "confidence": 0.88,
            "features": {"price": 45000, "volume": 1000000}
        }
        
        feature_importance = {
            "price": 0.4,
            "volume": 0.3,
            "rsi": 0.3
        }
        
        audit_result = auditor.audit_decision(
            decision_data=decision_data,
            feature_importance=feature_importance,
            generate_explanation=True
        )
        
        assert "audit_id" in audit_result
        assert "importance_id" in audit_result
        assert "explanation" in audit_result
        assert "compliance_status" in audit_result
        
        # Verify components were called
        audit_tracker.record_decision.assert_called_once()
        importance_tracker.record_importance.assert_called_once()
        explanation_generator.generate_explanation.assert_called_once()
        
    def test_batch_audit_processing(self):
        """Test batch processing of multiple decisions for audit."""
        auditor = EnhancedExplanationAuditor(
            audit_tracker=Mock(),
            importance_tracker=Mock(),
            explanation_generator=Mock(),
            compliance_generator=Mock()
        )
        
        # Sample batch of decisions
        decisions_batch = [
            {
                "decision_data": {"model_id": "lstm_v1", "decision_type": "buy"},
                "feature_importance": {"price": 0.5, "volume": 0.5}
            },
            {
                "decision_data": {"model_id": "dqn_v2", "decision_type": "sell"},
                "feature_importance": {"momentum": 0.6, "rsi": 0.4}
            }
        ]
        
        # Process batch
        batch_results = auditor.audit_decisions_batch(decisions_batch)
        
        assert len(batch_results) == 2
        assert all("audit_id" in result for result in batch_results)
        
    def test_audit_performance_monitoring(self):
        """Test monitoring of audit system performance."""
        auditor = EnhancedExplanationAuditor(
            audit_tracker=Mock(),
            importance_tracker=Mock(),
            explanation_generator=Mock(),
            compliance_generator=Mock()
        )
        
        # Simulate audit operations
        for i in range(10):
            decision_data = {"model_id": f"model_{i}", "decision_type": "hold"}
            auditor.audit_decision(decision_data, {})
        
        # Get performance metrics
        performance = auditor.get_audit_performance_metrics()
        
        assert "total_audits_processed" in performance
        assert "average_audit_latency" in performance
        assert "audit_success_rate" in performance
        assert "compliance_coverage" in performance
        
        assert performance["total_audits_processed"] == 10
        assert performance["audit_success_rate"] >= 0.0
        
    def test_regulatory_compliance_integration(self):
        """Test integration with regulatory compliance requirements."""
        compliance_generator = Mock()
        compliance_generator.validate_compliance.return_value = {
            "MiFID II": {"compliant": True, "score": 0.95},
            "GDPR": {"compliant": True, "score": 0.98}
        }
        
        auditor = EnhancedExplanationAuditor(
            audit_tracker=Mock(),
            importance_tracker=Mock(),
            explanation_generator=Mock(),
            compliance_generator=compliance_generator
        )
        
        # Test compliance validation
        model_data = {
            "model_id": "production_model",
            "explanation_coverage": 0.95,
            "audit_completeness": 1.0
        }
        
        compliance_result = auditor.validate_model_compliance(model_data)
        
        assert "MiFID II" in compliance_result
        assert "GDPR" in compliance_result
        assert compliance_result["MiFID II"]["compliant"] is True
        
        # Verify compliance generator was called
        compliance_generator.validate_compliance.assert_called_once_with(model_data)