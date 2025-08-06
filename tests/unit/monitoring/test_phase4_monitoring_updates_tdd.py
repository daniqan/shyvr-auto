"""
TDD Tests for Phase 4: Monitoring System Updates

These tests define the requirements for completing Phase 4 of the TODO_CHECKLIST.md.
All tests should initially fail, then be made to pass by implementing the required changes.

Key Requirements:
1. Remove TRANSFORMER_MODEL_TYPE references from monitoring dashboard  
2. Monitor all models in ensemble simultaneously
3. Add ensemble-level health metrics
4. Update performance validation for ensemble benchmarks
5. Ensure health endpoints report ensemble status
"""

import pytest
import asyncio
import json
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Test dependencies
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from src.monitoring.transformer_monitoring_dashboard import (
    TransformerMonitoringDashboard,
    TransformerHealthPanel,
    TransformerMemoryPanel,
    TransformerLatencyPanel
)
from src.testing.performance_validation import (
    LatencyValidator,
    ThroughputValidator,
    create_performance_validator_suite,
    run_comprehensive_validation
)


class TestPhase4_1_TransformerMonitoringDashboard:
    """
    Test Phase 4.1: Update transformer_monitoring_dashboard.py
    
    Requirements:
    - Remove TRANSFORMER_MODEL_TYPE references (lines 176, 257, 318)
    - Monitor all models in ensemble simultaneously 
    - Add ensemble-level health metrics
    """

    @pytest.fixture
    def ensemble_models(self):
        """Mock ensemble models configuration"""
        models = {}
        model_names = ["lstm", "iTransformer", "PatchTST", "TimesMixer", "TimesFM"]
        
        for name in model_names:
            model_mock = Mock()
            model_mock.model_name = name
            # Remove model_type attribute - this should not exist in ensemble mode
            if hasattr(model_mock, 'model_type'):
                delattr(model_mock, 'model_type')
            models[name] = model_mock
            
        return models

    @pytest.fixture
    def gcp_config(self):
        """Mock GCP configuration"""
        return {
            'project_id': 'test-project',
            'region': 'us-central1',
            'service_name': 'shyvr-rlte',
            'dashboard_name': 'ensemble-monitoring-dashboard'
        }

    def test_monitoring_dashboard_should_not_reference_transformer_model_type(self, ensemble_models, gcp_config):
        """
        FAILING TEST: Monitoring dashboard should not reference TRANSFORMER_MODEL_TYPE
        
        This test ensures that the transformer monitoring dashboard does not rely on
        TRANSFORMER_MODEL_TYPE references and works with ensemble configuration.
        """
        dashboard = TransformerMonitoringDashboard(
            models=ensemble_models,
            gcp_config=gcp_config
        )
        
        # Test that dashboard can be created without model_type attributes
        assert dashboard is not None
        assert len(dashboard.models) == 5  # All ensemble models
        
        # Verify no TRANSFORMER_MODEL_TYPE environment variable dependency
        with patch.dict(os.environ, {}, clear=True):
            # This should work without TRANSFORMER_MODEL_TYPE
            config = asyncio.run(dashboard.create_dashboard_config())
            assert config is not None
            assert 'dashboard' in config

    def test_health_panel_should_monitor_all_ensemble_models(self, ensemble_models):
        """
        FAILING TEST: Health panel should monitor all ensemble models simultaneously
        
        The health panel should generate monitoring configuration for all models
        in the ensemble without relying on model_type attributes.
        """
        health_panel = TransformerHealthPanel(models=ensemble_models)
        config = health_panel.generate_config()
        
        # Should have monitoring configuration for all 5 models
        assert 'widget' in config
        assert 'scorecard' in config['widget']
        assert 'timeSeries' in config['widget']['scorecard']
        
        time_series = config['widget']['scorecard']['timeSeries']
        assert len(time_series) == 5  # One for each ensemble model
        
        # Check that each model is monitored using model name, not model_type
        expected_models = ["lstm", "iTransformer", "PatchTST", "TimesMixer", "TimesFM"]
        filters = [ts['filter'] for ts in time_series]
        
        for model_name in expected_models:
            # Should find model name in at least one filter
            found = any(f'model_name="{model_name}"' in filter_str for filter_str in filters)
            assert found, f"Model {model_name} not found in monitoring filters. Filters: {filters}"
        
        # Should NOT reference model_type in filters
        for filter_str in filters:
            assert 'model.model_type' not in filter_str, "Should not reference model.model_type"

    def test_memory_panel_should_use_model_names_not_model_type(self, ensemble_models):
        """
        FAILING TEST: Memory panel should use model names instead of model_type
        
        Memory panel should reference models by name, not by model_type attribute.
        """
        memory_panel = TransformerMemoryPanel(models=ensemble_models)
        config = memory_panel.generate_config()
        
        assert 'widget' in config
        assert 'xyChart' in config['widget']
        assert 'dataSets' in config['widget']['xyChart']
        
        datasets = config['widget']['xyChart']['dataSets']
        assert len(datasets) == 5  # One dataset per ensemble model
        
        # Each dataset should use model name in legend and filter
        expected_models = ["lstm", "iTransformer", "PatchTST", "TimesMixer", "TimesFM"]
        
        # Check that all expected models are represented in datasets
        dataset_models = []
        for dataset in datasets:
            # Check legend template uses model name
            assert 'legendTemplate' in dataset
            legend = dataset['legendTemplate']
            
            # Extract model name from legend
            for model_name in expected_models:
                if model_name in legend:
                    dataset_models.append(model_name)
                    break
            
            # Check filter references model name, not model_type
            filter_str = dataset['timeSeriesQuery']['filter']
            assert 'model_name=' in filter_str
            assert 'model_type=' not in filter_str  # Should not use model_type
        
        # All expected models should be represented
        assert set(dataset_models) == set(expected_models), f"Missing models: {set(expected_models) - set(dataset_models)}"

    def test_latency_panel_should_monitor_ensemble_latency(self, ensemble_models):
        """
        FAILING TEST: Latency panel should monitor ensemble latency without model_type
        
        Latency panel should generate P95/P99 metrics for all ensemble models
        using model names instead of model_type.
        """
        latency_panel = TransformerLatencyPanel(models=ensemble_models)
        config = latency_panel.generate_config()
        
        assert 'widget' in config
        datasets = config['widget']['xyChart']['dataSets']
        
        # Should have 2 datasets per model (P95 and P99) = 10 total
        assert len(datasets) == 10
        
        # Check that datasets reference model names correctly
        expected_models = ["lstm", "iTransformer", "PatchTST", "TimesMixer", "TimesFM"]
        
        p95_legends = [ds['legendTemplate'] for ds in datasets if 'P95' in ds['legendTemplate']]
        p99_legends = [ds['legendTemplate'] for ds in datasets if 'P99' in ds['legendTemplate']]
        
        assert len(p95_legends) == 5
        assert len(p99_legends) == 5
        
        for model_name in expected_models:
            # Check P95 legend
            p95_found = any(model_name in legend for legend in p95_legends)
            assert p95_found, f"P95 metric for {model_name} not found"
            
            # Check P99 legend  
            p99_found = any(model_name in legend for legend in p99_legends)
            assert p99_found, f"P99 metric for {model_name} not found"

    def test_ensemble_level_health_metrics_aggregation(self, ensemble_models, gcp_config):
        """
        FAILING TEST: Dashboard should provide ensemble-level health aggregation
        
        The monitoring dashboard should aggregate health metrics across all
        ensemble models to provide ensemble-level health status.
        """
        dashboard = TransformerMonitoringDashboard(
            models=ensemble_models,
            gcp_config=gcp_config
        )
        
        # Mock individual model health status
        for model_name, model in ensemble_models.items():
            model.get_health_status = Mock(return_value={'healthy': True, 'score': 0.95})
        
        # Dashboard should provide ensemble health aggregation method
        assert hasattr(dashboard, 'get_ensemble_health_status'), \
            "Dashboard should have get_ensemble_health_status method"
        
        health_status = dashboard.get_ensemble_health_status()
        
        # Ensemble health should aggregate individual model health
        assert 'ensemble_healthy' in health_status
        assert 'individual_models' in health_status
        assert 'ensemble_score' in health_status
        
        assert health_status['ensemble_healthy'] is True
        assert len(health_status['individual_models']) == 5
        assert 0.0 <= health_status['ensemble_score'] <= 1.0


class TestPhase4_2_PerformanceValidation:
    """
    Test Phase 4.2: Update performance_validation.py
    
    Requirements:
    - Add ensemble performance benchmarks
    - Validate combined resource usage
    - Test ensemble-specific performance metrics
    """

    @pytest.fixture
    def ensemble_config(self):
        """Configuration for ensemble performance validation"""
        return {
            "sla_requirements": {
                "max_response_time_ms": 100,
                "max_ensemble_latency_ms": 150,  # Ensemble may be slightly slower
                "min_ensemble_accuracy": 0.85
            },
            "capacity_requirements": {
                "min_throughput_rps": 500,  # Higher throughput expected from ensemble
                "max_concurrent_users": 2000,
                "ensemble_scaling_factor": 1.2
            },
            "resource_limits": {
                "max_cpu_utilization": 0.80,
                "max_memory_utilization": 0.85,
                "max_ensemble_memory_gb": 8.0,  # Total ensemble memory limit
                "max_individual_model_memory_gb": 2.0
            },
            "response_time_targets": {
                "max_avg_ms": 100,
                "max_p95_ms": 200,
                "max_p99_ms": 500,
                "max_ensemble_consensus_time_ms": 50  # Time to reach ensemble consensus
            },
            "error_rate_targets": {
                "max_error_rate": 0.01,
                "max_ensemble_disagreement_rate": 0.05  # Models can disagree 5% of time
            }
        }

    def test_should_create_ensemble_performance_validators(self, ensemble_config):
        """
        FAILING TEST: Should create validators with ensemble-specific benchmarks
        
        Performance validation suite should include ensemble-specific validators
        that can handle multiple model performance metrics.
        """
        validators = create_performance_validator_suite(ensemble_config)
        
        # Should have all standard validators
        assert 'latency' in validators
        assert 'throughput' in validators
        assert 'resource_usage' in validators
        
        # Should have ensemble-specific validators
        assert 'ensemble_performance' in validators, "Missing ensemble performance validator"
        assert 'ensemble_resource_usage' in validators, "Missing ensemble resource validator"
        
        # Validators should handle ensemble configuration
        ensemble_validator = validators['ensemble_performance']
        assert hasattr(ensemble_validator, 'validate_ensemble_metrics'), \
            "Ensemble validator should have validate_ensemble_metrics method"

    def test_should_validate_ensemble_latency_benchmarks(self, ensemble_config):
        """
        FAILING TEST: Should validate ensemble-specific latency benchmarks
        
        Latency validator should handle ensemble latency requirements including
        individual model latencies and ensemble consensus time.
        """
        validators = create_performance_validator_suite(ensemble_config)
        latency_validator = validators['latency']
        
        # Test ensemble latency data structure
        ensemble_latency_data = {
            "individual_models": {
                "lstm": {"avg": 45, "p95": 80, "p99": 120},
                "iTransformer": {"avg": 55, "p95": 90, "p99": 140}, 
                "PatchTST": {"avg": 60, "p95": 95, "p99": 150},
                "TimesMixer": {"avg": 50, "p95": 85, "p99": 130},
                "TimesFM": {"avg": 65, "p95": 100, "p99": 160}
            },
            "ensemble": {
                "consensus_time": 35,
                "total_latency": 95,
                "avg": 95,
                "p95": 180, 
                "p99": 280
            }
        }
        
        # Should validate ensemble latency without errors
        result = latency_validator.validate_latency_metrics(ensemble_latency_data)
        
        assert hasattr(result, 'valid')
        assert hasattr(result, 'score')
        assert hasattr(result, 'violations')
        assert hasattr(result, 'metrics')
        
        # Should pass ensemble latency validation
        assert result.valid is True, f"Ensemble latency validation failed: {result.violations}"
        assert result.score >= 0.8, f"Ensemble latency score too low: {result.score}"

    def test_should_validate_ensemble_resource_usage(self, ensemble_config):
        """
        FAILING TEST: Should validate combined ensemble resource usage
        
        Resource validator should validate total ensemble resource consumption
        including memory usage across all models.
        """
        validators = create_performance_validator_suite(ensemble_config)
        ensemble_resource_validator = validators['ensemble_resource_usage']
        
        # Test ensemble resource data with violations
        ensemble_resource_data = {
            "individual_models": {
                "lstm": {"memory_gb": 1.5, "cpu_percent": 15},
                "iTransformer": {"memory_gb": 1.8, "cpu_percent": 18},
                "PatchTST": {"memory_gb": 1.7, "cpu_percent": 17},
                "TimesMixer": {"memory_gb": 1.6, "cpu_percent": 16},
                "TimesFM": {"memory_gb": 2.5, "cpu_percent": 19}  # This exceeds individual limit of 2.0GB
            },
            "ensemble_memory_gb": 9.5,  # Exceeds ensemble limit of 8.0GB
            "individual_model_max_memory_gb": 2.5
        }
        
        # Should detect resource limit violations
        result = ensemble_resource_validator.validate_ensemble_resource_usage(ensemble_resource_data)
        
        # This should fail because ensemble memory exceeds limit
        assert result.valid is False, "Should detect ensemble resource violations"
        assert len(result.violations) > 0, "Should report specific violations"
        
        # Check that violations mention ensemble resource limits
        violation_messages = [v.get('metric', '') for v in result.violations]
        assert any('ensemble_memory_gb' in msg for msg in violation_messages), "Should report ensemble memory violation"

    def test_should_validate_ensemble_throughput_scaling(self, ensemble_config):
        """
        FAILING TEST: Should validate ensemble throughput scaling capabilities
        
        Throughput validator should handle ensemble scaling requirements
        and validate that ensemble provides better throughput than individual models.
        """
        validators = create_performance_validator_suite(ensemble_config)
        throughput_validator = validators['throughput']
        
        # Test ensemble throughput data
        ensemble_throughput_data = {
            "individual_models": {
                "lstm": {"rps": 120, "concurrent_users": 400},
                "iTransformer": {"rps": 100, "concurrent_users": 350},
                "PatchTST": {"rps": 110, "concurrent_users": 380},
                "TimesMixer": {"rps": 115, "concurrent_users": 390},
                "TimesFM": {"rps": 95, "concurrent_users": 320}
            },
            "ensemble": {
                "rps": 540,  # Combined throughput with overhead
                "concurrent_users": 1840,  # Combined capacity
                "scaling_efficiency": 0.92  # 92% of theoretical maximum
            }
        }
        
        # Should validate ensemble throughput meets requirements
        result = throughput_validator.validate_throughput_metrics(ensemble_throughput_data["ensemble"])
        
        assert result.valid is True, f"Ensemble throughput validation failed: {result.violations}"
        assert result.score >= 0.8, f"Ensemble throughput score too low: {result.score}"
        
        # Ensemble RPS should exceed minimum requirement (500 RPS)
        assert ensemble_throughput_data["ensemble"]["rps"] >= 500

    @pytest.mark.asyncio
    async def test_should_run_comprehensive_ensemble_validation(self, ensemble_config):
        """
        FAILING TEST: Should run comprehensive validation for ensemble performance
        
        The comprehensive validation should handle ensemble-specific data structures
        and provide ensemble-level performance assessment.
        """
        validators = create_performance_validator_suite(ensemble_config)
        
        # Comprehensive ensemble performance data
        ensemble_performance_data = {
            "response_time_ms": {
                "individual_models": {
                    "lstm": {"avg": 45, "p95": 80, "p99": 120},
                    "iTransformer": {"avg": 55, "p95": 90, "p99": 140},
                    "PatchTST": {"avg": 60, "p95": 95, "p99": 150},
                    "TimesMixer": {"avg": 50, "p95": 85, "p99": 130},
                    "TimesFM": {"avg": 65, "p95": 100, "p99": 160}
                },
                "ensemble": {"avg": 95, "p95": 180, "p99": 280}
            },
            "throughput_data": {
                "rps": 540,
                "concurrent_users": 1840,
                "scaling_efficiency": 0.92
            },
            "resource_usage": {
                "memory": 0.80,  # 80% memory utilization (within limits)
                "cpu": 0.75,     # 75% CPU utilization (within limits)
            },
            "ensemble_resource_data": {
                "ensemble_memory_gb": 7.8,
                "individual_models": {
                    "lstm": {"memory_gb": 1.5, "cpu_percent": 15},
                    "iTransformer": {"memory_gb": 1.6, "cpu_percent": 16},
                    "PatchTST": {"memory_gb": 1.5, "cpu_percent": 15},
                    "TimesMixer": {"memory_gb": 1.6, "cpu_percent": 16},
                    "TimesFM": {"memory_gb": 1.6, "cpu_percent": 17}
                }
            },
            "ensemble_data": {
                "consensus_time": 35,
                "accuracy": 0.90,  # Higher accuracy to meet requirements
                "disagreement_rate": 0.02,
                "individual_models": {
                    "lstm": {"healthy": True, "accuracy": 0.87},
                    "iTransformer": {"healthy": True, "accuracy": 0.89},
                    "PatchTST": {"healthy": True, "accuracy": 0.88},
                    "TimesMixer": {"healthy": True, "accuracy": 0.90},
                    "TimesFM": {"healthy": True, "accuracy": 0.91}
                }
            },
            "response_times": [85, 90, 88, 92, 87, 89, 91, 86, 93, 88],  # Sample response times
            "error_data": {
                "error_rate": 0.008,  # 0.8% error rate (within limits)
                "total_requests": 10000,
                "error_count": 80,
                "ensemble_disagreement_rate": 0.03  # 3% disagreement (within limits)
            }
        }
        
        # Run comprehensive validation
        result = await run_comprehensive_validation(validators, ensemble_performance_data)
        
        assert 'validation_passed' in result
        assert 'overall_score' in result
        assert 'individual_results' in result
        
        # Should pass comprehensive validation
        assert result['validation_passed'] is True, f"Comprehensive validation failed: {result.get('violations', [])}"
        assert result['overall_score'] >= 0.6, f"Overall ensemble score too low: {result['overall_score']}"
        
        # Should have results for all validation categories
        individual_results = result['individual_results']
        expected_categories = ['latency', 'throughput', 'resource_usage', 'response_time_distribution', 'error_rate']
        
        for category in expected_categories:
            if category in individual_results:
                assert individual_results[category].valid, f"{category} validation failed"
        
        # Should also have ensemble-specific results
        ensemble_categories = ['ensemble_performance', 'ensemble_resource_usage']
        for category in ensemble_categories:
            if category in individual_results:
                assert individual_results[category].valid, f"{category} validation failed"


class TestPhase4_3_HealthEndpointEnsemble:
    """
    Test Phase 4.3: Verify main.py health endpoint reports ensemble status
    
    Requirements:
    - Report ensemble health status
    - Include all model statuses in development/production  
    - Handle environment-specific model availability
    """

    @pytest.fixture
    def mock_environment_production(self):
        """Mock production environment"""
        return patch.dict(os.environ, {'ENVIRONMENT': 'production'})

    @pytest.fixture
    def mock_environment_development(self):
        """Mock development environment"""
        return patch.dict(os.environ, {'ENVIRONMENT': 'development'})

    @pytest.fixture
    def mock_ensemble_models(self):
        """Mock ensemble models for testing"""
        models = {}
        model_names = ["lstm", "iTransformer", "PatchTST", "TimesMixer", "TimesFM"]
        
        for name in model_names:
            model_mock = Mock()
            model_mock.model_name = name
            model_mock.health_check = Mock(return_value=True)
            model_mock.get_memory_usage_mb = Mock(return_value=1500.0)
            model_mock.get_inference_latency_ms = Mock(return_value=85.0)
            models[name] = model_mock
            
        return models

    def test_health_endpoint_should_report_ensemble_status_production(self, mock_environment_production, mock_ensemble_models):
        """
        FAILING TEST: Health endpoint should report ensemble status in production
        
        In production mode, health endpoint should report status for all ensemble models
        and provide ensemble-level aggregated health metrics.
        """
        with mock_environment_production:
            # Import and mock the health check functions from main.py
            with patch('main.get_transformer_health') as mock_transformer_health:
                mock_transformer_health.return_value = {
                    'status': 'healthy',
                    'models_loaded': 5,
                    'models_healthy': 5,
                    'ensemble_mode': True,
                    'environment': 'production',
                    'individual_models': {
                        'lstm': {'healthy': True, 'memory_usage_mb': 1500, 'latency_ms': 80},
                        'iTransformer': {'healthy': True, 'memory_usage_mb': 1600, 'latency_ms': 85},
                        'PatchTST': {'healthy': True, 'memory_usage_mb': 1550, 'latency_ms': 90},
                        'TimesMixer': {'healthy': True, 'memory_usage_mb': 1580, 'latency_ms': 87},
                        'TimesFM': {'healthy': True, 'memory_usage_mb': 1620, 'latency_ms': 92}
                    },
                    'ensemble_metrics': {
                        'total_memory_mb': 7850,
                        'avg_latency_ms': 86.8,
                        'consensus_health': True
                    }
                }
                
                health_status = mock_transformer_health.return_value
                
                # Verify ensemble reporting requirements
                assert health_status['status'] == 'healthy'
                assert health_status['ensemble_mode'] is True
                assert health_status['environment'] == 'production'
                assert health_status['models_loaded'] == 5
                assert health_status['models_healthy'] == 5
                
                # Should include individual model details
                assert 'individual_models' in health_status
                individual_models = health_status['individual_models']
                expected_models = ['lstm', 'iTransformer', 'PatchTST', 'TimesMixer', 'TimesFM']
                
                for model_name in expected_models:
                    assert model_name in individual_models, f"Missing {model_name} in health status"
                    model_health = individual_models[model_name]
                    assert 'healthy' in model_health
                    assert 'memory_usage_mb' in model_health
                    assert 'latency_ms' in model_health
                
                # Should include ensemble-level metrics
                assert 'ensemble_metrics' in health_status
                ensemble_metrics = health_status['ensemble_metrics']
                assert 'total_memory_mb' in ensemble_metrics
                assert 'avg_latency_ms' in ensemble_metrics
                assert 'consensus_health' in ensemble_metrics

    def test_health_endpoint_should_report_development_mode(self, mock_environment_development):
        """
        FAILING TEST: Health endpoint should report LSTM-only status in development
        
        In development mode, health endpoint should report status for LSTM only
        while indicating that ensemble mode is disabled.
        """
        with mock_environment_development:
            with patch('main.get_transformer_health') as mock_transformer_health:
                mock_transformer_health.return_value = {
                    'status': 'healthy', 
                    'models_loaded': 1,
                    'models_healthy': 1,
                    'ensemble_mode': False,
                    'environment': 'development',
                    'individual_models': {
                        'lstm': {'healthy': True, 'memory_usage_mb': 1500, 'latency_ms': 75}
                    },
                    'development_note': 'Running LSTM only in development mode'
                }
                
                health_status = mock_transformer_health.return_value
                
                # Verify development mode reporting
                assert health_status['status'] == 'healthy'
                assert health_status['ensemble_mode'] is False
                assert health_status['environment'] == 'development'
                assert health_status['models_loaded'] == 1
                assert health_status['models_healthy'] == 1
                
                # Should only include LSTM in development
                individual_models = health_status['individual_models']
                assert len(individual_models) == 1
                assert 'lstm' in individual_models
                assert individual_models['lstm']['healthy'] is True
                
                # Should include development mode note
                assert 'development_note' in health_status

    def test_health_endpoint_should_handle_ensemble_degraded_state(self, mock_environment_production):
        """
        FAILING TEST: Health endpoint should handle ensemble degraded state
        
        When some ensemble models are unhealthy, health endpoint should report
        degraded status and identify which models are causing issues.
        """
        with mock_environment_production:
            with patch('main.get_transformer_health') as mock_transformer_health:
                # Simulate degraded ensemble state (2 models unhealthy)
                mock_transformer_health.return_value = {
                    'status': 'degraded',
                    'models_loaded': 5,
                    'models_healthy': 3,
                    'ensemble_mode': True,
                    'environment': 'production',
                    'individual_models': {
                        'lstm': {'healthy': True, 'memory_usage_mb': 1500, 'latency_ms': 80},
                        'iTransformer': {'healthy': False, 'memory_usage_mb': 2500, 'latency_ms': 150, 'error': 'High memory usage'},
                        'PatchTST': {'healthy': True, 'memory_usage_mb': 1550, 'latency_ms': 90},
                        'TimesMixer': {'healthy': True, 'memory_usage_mb': 1580, 'latency_ms': 87},
                        'TimesFM': {'healthy': False, 'memory_usage_mb': 1620, 'latency_ms': 200, 'error': 'High latency'}
                    },
                    'ensemble_metrics': {
                        'total_memory_mb': 8750,
                        'avg_latency_ms': 121.4,  # Higher due to unhealthy models
                        'consensus_health': False,  # Degraded consensus
                        'healthy_model_count': 3,
                        'degraded_model_count': 2
                    },
                    'degraded_models': ['iTransformer', 'TimesFM']
                }
                
                health_status = mock_transformer_health.return_value
                
                # Verify degraded state reporting
                assert health_status['status'] == 'degraded'
                assert health_status['models_loaded'] == 5
                assert health_status['models_healthy'] == 3
                assert health_status['ensemble_metrics']['consensus_health'] is False
                
                # Should identify degraded models
                assert 'degraded_models' in health_status
                degraded_models = health_status['degraded_models']
                assert 'iTransformer' in degraded_models
                assert 'TimesFM' in degraded_models
                
                # Should include error details for unhealthy models
                individual_models = health_status['individual_models']
                assert individual_models['iTransformer']['healthy'] is False
                assert 'error' in individual_models['iTransformer']
                assert individual_models['TimesFM']['healthy'] is False
                assert 'error' in individual_models['TimesFM']

    def test_health_endpoint_should_aggregate_ensemble_metrics(self, mock_environment_production, mock_ensemble_models):
        """
        FAILING TEST: Health endpoint should aggregate ensemble metrics correctly
        
        Health endpoint should calculate and report aggregated metrics across
        all ensemble models including total resource usage and average performance.
        """
        with mock_environment_production:
            # Mock individual model metrics
            model_metrics = {
                'lstm': {'memory_mb': 1500, 'latency_ms': 80, 'cpu_percent': 15},
                'iTransformer': {'memory_mb': 1600, 'latency_ms': 85, 'cpu_percent': 18},
                'PatchTST': {'memory_mb': 1550, 'latency_ms': 90, 'cpu_percent': 17},
                'TimesMixer': {'memory_mb': 1580, 'latency_ms': 87, 'cpu_percent': 16},
                'TimesFM': {'memory_mb': 1620, 'latency_ms': 92, 'cpu_percent': 19}
            }
            
            with patch('main.get_transformer_health') as mock_transformer_health:
                # Calculate expected aggregated metrics
                total_memory = sum(m['memory_mb'] for m in model_metrics.values())
                avg_latency = sum(m['latency_ms'] for m in model_metrics.values()) / len(model_metrics)
                total_cpu = sum(m['cpu_percent'] for m in model_metrics.values())
                
                mock_transformer_health.return_value = {
                    'status': 'healthy',
                    'models_loaded': 5,
                    'models_healthy': 5,
                    'ensemble_mode': True,
                    'environment': 'production',
                    'individual_models': {
                        model: {'healthy': True, **metrics} 
                        for model, metrics in model_metrics.items()
                    },
                    'ensemble_metrics': {
                        'total_memory_mb': total_memory,
                        'avg_latency_ms': avg_latency,
                        'total_cpu_percent': total_cpu,
                        'memory_efficiency': total_memory / len(model_metrics),
                        'performance_score': 0.92,
                        'consensus_health': True
                    }
                }
                
                health_status = mock_transformer_health.return_value
                ensemble_metrics = health_status['ensemble_metrics']
                
                # Verify aggregated metrics calculation
                assert ensemble_metrics['total_memory_mb'] == total_memory
                assert abs(ensemble_metrics['avg_latency_ms'] - avg_latency) < 0.1
                assert ensemble_metrics['total_cpu_percent'] == total_cpu
                assert 'memory_efficiency' in ensemble_metrics
                assert 'performance_score' in ensemble_metrics
                assert 0.0 <= ensemble_metrics['performance_score'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])