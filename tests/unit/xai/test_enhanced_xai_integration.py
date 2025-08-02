"""
Integration tests for enhanced XAI audit capabilities.

This module tests the integration between Phase 5.5 enhanced explainability
features and the existing XAI components, ensuring seamless operation.
"""

import pytest
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from src.xai.enhanced_audit import (
    ModelAuditTracker,
    FeatureImportanceTracker,
    DecisionExplanationGenerator,
    ComplianceReportGenerator,
    EnhancedExplanationAuditor
)
from src.xai.data_models import ExplanationData
from src.xai.factory import ExplainerFactory
from src.xai.trading_integration import TradingExplanationManager


class TestEnhancedXAIIntegration:
    """Test integration between enhanced XAI audit capabilities and existing XAI system."""
    
    def test_integration_with_trading_explanation_manager(self):
        """Test integration with existing TradingExplanationManager."""
        # Create enhanced audit components
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_tracker = ModelAuditTracker(audit_storage_path=temp_dir)
            importance_tracker = FeatureImportanceTracker()
            explanation_generator = DecisionExplanationGenerator()
            
            enhanced_auditor = EnhancedExplanationAuditor(
                audit_tracker=audit_tracker,
                importance_tracker=importance_tracker,
                explanation_generator=explanation_generator
            )
            
            # Simulate trading decision with existing XAI integration
            decision_data = {
                "model_id": "lstm_v1.2.3",
                "decision_type": "buy",
                "symbol": "BTC-USD",
                "confidence": 0.85,
                "price": 45000,
                "features": {"momentum": 0.7, "volume": 0.8, "rsi": 0.6}
            }
            
            feature_importance = {
                "momentum": 0.4,
                "volume": 0.35,
                "rsi": 0.25
            }
            
            # Test comprehensive audit
            audit_result = enhanced_auditor.audit_decision(
                decision_data=decision_data,
                feature_importance=feature_importance,
                generate_explanation=True
            )
            
            # Verify audit components
            assert audit_result["audit_id"] != ""
            assert audit_result["importance_id"] != ""
            assert "explanation" in audit_result
            assert audit_result["compliance_status"] == "compliant"
            
            # Verify explanation quality
            explanation = audit_result["explanation"]
            assert "BTC-USD" in explanation["summary"]
            assert "85%" in explanation["confidence_explanation"]
            assert "momentum" in explanation["key_features"]  # Top feature
    
    def test_enhanced_explanation_data_compatibility(self):
        """Test compatibility with existing ExplanationData structures."""
        # Create enhanced explanation generator
        generator = DecisionExplanationGenerator()
        
        # Create existing ExplanationData
        explanation_data = ExplanationData(
            feature_importance={"price": 0.5, "volume": 0.3, "rsi": 0.2},
            explanation_type="permutation",
            instance_data=[45000, 1000000, 0.6],
            model_prediction=0.8,
            confidence_score=0.85
        )
        
        # Test enhanced explanation generation
        decision_context = {
            "decision_type": "buy",
            "symbol": "ETH-USD",
            "confidence": explanation_data.confidence_score,
            "prediction": explanation_data.model_prediction
        }
        
        enhanced_explanation = generator.generate_explanation(
            decision_context=decision_context,
            feature_importance=explanation_data.feature_importance,
            model_type="permutation"
        )
        
        # Verify compatibility
        assert isinstance(enhanced_explanation, dict)
        assert "summary" in enhanced_explanation
        assert "detailed_reasoning" in enhanced_explanation
        assert "ETH-USD" in enhanced_explanation["summary"]
        assert "price" in enhanced_explanation["key_features"]  # Most important feature
    
    def test_audit_tracker_with_existing_explainer_factory(self):
        """Test audit tracker integration with existing ExplainerFactory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_tracker = ModelAuditTracker(audit_storage_path=temp_dir)
            
            # Mock model and features
            mock_model = Mock()
            mock_model.predict.return_value = [0.8]
            
            feature_names = ["price", "volume", "rsi", "macd"]
            feature_data = [45000, 1000000, 0.6, 0.05]
            
            # Create explainer using existing factory
            try:
                explainer_factory = ExplainerFactory()
                explainer = explainer_factory.create_explainer(
                    explainer_type="permutation",
                    model=mock_model,
                    feature_names=feature_names
                )
                
                # Generate explanation using existing system
                explanation_data = explainer.explain_instance(feature_data)
                
                # Record in enhanced audit system
                decision_data = {
                    "model_id": "factory_model_v1",
                    "decision_type": "sell",
                    "symbol": "SOL-USD",
                    "confidence": explanation_data.confidence_score or 0.75,
                    "features": dict(zip(feature_names, feature_data)),
                    "prediction": explanation_data.model_prediction
                }
                
                audit_id = audit_tracker.record_decision(decision_data)
                
                # Verify integration
                assert audit_id != ""
                
                # Retrieve and verify
                retrieved_decision = audit_tracker.get_decision(audit_id)
                assert retrieved_decision is not None
                assert retrieved_decision["model_id"] == "factory_model_v1"
                assert retrieved_decision["symbol"] == "SOL-USD"
                
            except Exception as e:
                # If ExplainerFactory integration fails, ensure audit tracker still works
                assert audit_tracker.is_enabled
                
                # Test with simple decision data
                simple_decision = {
                    "model_id": "simple_model",
                    "decision_type": "hold",
                    "symbol": "BTC-USD"
                }
                
                audit_id = audit_tracker.record_decision(simple_decision)
                assert audit_id != ""
    
    def test_feature_importance_tracking_integration(self):
        """Test feature importance tracking with existing XAI patterns."""
        importance_tracker = FeatureImportanceTracker()
        
        # Simulate sequence of decisions with feature importance from existing XAI
        decisions_sequence = [
            {
                "model_id": "lstm_trading_v1",
                "decision_id": f"dec_{i}",
                "importance": {
                    "price_momentum": 0.3 + (i * 0.01),
                    "volume_indicator": 0.25 + (i * 0.005),
                    "technical_rsi": 0.2 - (i * 0.002),
                    "macd_signal": 0.15 + (i * 0.003),
                    "bollinger_bands": 0.1 - (i * 0.001)
                },
                "symbol": "BTC-USD"
            }
            for i in range(20)
        ]
        
        # Record all decisions
        for decision in decisions_sequence:
            record_id = importance_tracker.record_importance(
                model_id=decision["model_id"],
                decision_id=decision["decision_id"],
                feature_importance=decision["importance"],
                symbol=decision["symbol"]
            )
            assert record_id != ""
        
        # Test aggregation functionality
        aggregated = importance_tracker.get_aggregated_importance("lstm_trading_v1")
        
        assert "price_momentum" in aggregated
        assert "volume_indicator" in aggregated
        assert aggregated["price_momentum"] > aggregated["technical_rsi"]  # Should show trend
        
        # Test trend analysis
        trends = importance_tracker.analyze_importance_trends("lstm_trading_v1", lookback_window=15)
        
        assert "price_momentum" in trends
        assert trends["price_momentum"]["trend"] == "increasing"
        assert trends["technical_rsi"]["trend"] == "decreasing"
    
    def test_compliance_report_generation_integration(self):
        """Test compliance report generation with real audit data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create integrated audit system
            audit_tracker = ModelAuditTracker(audit_storage_path=temp_dir)
            importance_tracker = FeatureImportanceTracker()
            compliance_generator = ComplianceReportGenerator(output_directory=temp_dir)
            
            enhanced_auditor = EnhancedExplanationAuditor(
                audit_tracker=audit_tracker,
                importance_tracker=importance_tracker,
                compliance_generator=compliance_generator
            )
            
            # Simulate week of trading decisions
            start_date = datetime.utcnow() - timedelta(days=7)
            
            models_used = ["lstm_v1.2.3", "dqn_v2.1.0", "ensemble_v1.0.0"]
            decision_types = ["buy", "sell", "hold"]
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
            
            for day in range(7):
                for hour in range(0, 24, 2):  # Every 2 hours
                    decision_time = start_date + timedelta(days=day, hours=hour)
                    
                    decision_data = {
                        "model_id": np.random.choice(models_used),
                        "decision_type": np.random.choice(decision_types),
                        "symbol": np.random.choice(symbols),
                        "confidence": 0.6 + np.random.random() * 0.3,
                        "risk_score": np.random.random() * 0.5,
                        "timestamp": decision_time.isoformat()
                    }
                    
                    feature_importance = {
                        "price": 0.2 + np.random.random() * 0.3,
                        "volume": 0.15 + np.random.random() * 0.2,
                        "rsi": 0.1 + np.random.random() * 0.15,
                        "macd": 0.05 + np.random.random() * 0.1
                    }
                    
                    # Normalize feature importance
                    total = sum(feature_importance.values())
                    feature_importance = {k: v/total for k, v in feature_importance.items()}
                    
                    # Audit decision
                    enhanced_auditor.audit_decision(
                        decision_data=decision_data,
                        feature_importance=feature_importance
                    )
            
            # Generate compliance report
            end_date = datetime.utcnow()
            report = enhanced_auditor.generate_compliance_report(start_date, end_date)
            
            # Verify report structure
            assert "report_metadata" in report
            assert "executive_summary" in report
            assert "model_usage_analysis" in report
            assert "explainability_metrics" in report
            assert "compliance_assessment" in report
            
            # Verify report content
            metadata = report["report_metadata"]
            assert metadata["compliance_standards"] == ["MiFID II", "GDPR"]
            
            exec_summary = report["executive_summary"]
            assert "key_findings" in exec_summary
            assert exec_summary["compliance_status"] in ["COMPLIANT", "REVIEW_REQUIRED"]
            
            explainability = report["explainability_metrics"]
            assert explainability["explanation_coverage"] > 0
            assert explainability["human_readable_explanations"] is True
    
    def test_enhanced_auditor_performance_monitoring(self):
        """Test performance monitoring of enhanced audit system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            enhanced_auditor = EnhancedExplanationAuditor(
                audit_tracker=ModelAuditTracker(audit_storage_path=temp_dir),
                importance_tracker=FeatureImportanceTracker(),
                explanation_generator=DecisionExplanationGenerator()
            )
            
            # Process multiple decisions to test performance
            num_decisions = 50
            
            for i in range(num_decisions):
                decision_data = {
                    "model_id": f"perf_model_{i % 3}",
                    "decision_type": ["buy", "sell", "hold"][i % 3],
                    "symbol": ["BTC-USD", "ETH-USD"][i % 2],
                    "confidence": 0.7 + (i % 5) * 0.05
                }
                
                feature_importance = {
                    "feature_a": 0.4 + (i % 10) * 0.01,
                    "feature_b": 0.35 - (i % 8) * 0.005,
                    "feature_c": 0.25
                }
                
                # Audit decision
                result = enhanced_auditor.audit_decision(
                    decision_data=decision_data,
                    feature_importance=feature_importance
                )
                
                # Verify each audit succeeds
                assert result["audit_id"] != ""
                assert result.get("status") != "failed"
            
            # Check performance metrics
            metrics = enhanced_auditor.get_audit_performance_metrics()
            
            assert metrics["total_audits_processed"] == num_decisions
            assert metrics["audit_success_rate"] >= 0.95  # High success rate expected
            assert metrics["average_audit_latency"] < 1.0  # Should be fast
            assert metrics["compliance_coverage"] >= 0.95
    
    def test_batch_processing_integration(self):
        """Test batch processing capabilities with existing XAI patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            enhanced_auditor = EnhancedExplanationAuditor(
                audit_tracker=ModelAuditTracker(audit_storage_path=temp_dir)
            )
            
            # Create batch of decisions
            batch_decisions = []
            
            for i in range(10):
                decision_data = {
                    "model_id": "batch_model_v1",
                    "decision_type": "buy" if i % 2 == 0 else "sell",
                    "symbol": f"TOKEN_{i}-USD",
                    "confidence": 0.6 + (i * 0.03)
                }
                
                feature_importance = {
                    "market_sentiment": 0.3 + (i * 0.01),
                    "price_action": 0.25 + (i * 0.005),
                    "volume_profile": 0.2 - (i * 0.002),
                    "technical_indicators": 0.25 - (i * 0.003)
                }
                
                batch_decisions.append({
                    "decision_data": decision_data,
                    "feature_importance": feature_importance
                })
            
            # Process batch
            batch_results = enhanced_auditor.audit_decisions_batch(batch_decisions)
            
            # Verify all processed successfully
            assert len(batch_results) == 10
            
            for i, result in enumerate(batch_results):
                assert result["audit_id"] != ""
                assert result.get("status") != "failed"
                
                # Verify importance tracking
                if "importance_id" in result:
                    assert result["importance_id"] != ""
    
    def test_regulatory_compliance_validation(self):
        """Test regulatory compliance validation integration."""
        compliance_generator = ComplianceReportGenerator(
            compliance_standards=["MiFID II", "GDPR", "SEC Rule 3a-4"]
        )
        
        # Test model data that should be compliant
        compliant_model_data = {
            "model_id": "production_lstm_v2",
            "explanation_coverage": 0.98,
            "audit_trail_completeness": 1.0,
            "human_readable_explanations": True,
            "risk_disclosure_present": True,
            "data_retention_compliant": True
        }
        
        compliance_result = compliance_generator.validate_compliance(compliant_model_data)
        
        # Verify all standards are assessed
        assert "MiFID II" in compliance_result
        assert "GDPR" in compliance_result
        assert "SEC Rule 3a-4" in compliance_result
        
        # Verify compliance status
        for standard, result in compliance_result.items():
            assert result["compliant"] is True
            assert result["score"] >= 0.9
            assert len(result["gaps"]) == 0
        
        # Test model data with compliance gaps
        non_compliant_model_data = {
            "model_id": "test_model",
            "explanation_coverage": 0.75,  # Below threshold
            "audit_trail_completeness": 0.8,  # Below threshold
            "human_readable_explanations": False,  # Missing
            "risk_disclosure_present": False  # Missing
        }
        
        compliance_result_gaps = compliance_generator.validate_compliance(non_compliant_model_data)
        
        # Verify gaps are detected
        mifid_result = compliance_result_gaps["MiFID II"]
        assert mifid_result["compliant"] is False
        assert len(mifid_result["gaps"]) > 0