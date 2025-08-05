#!/usr/bin/env python3

"""
Transformer Monitoring Dashboard Tests

Tests for comprehensive transformer monitoring dashboards and alerting rules 
for GCP Cloud Monitoring following TDD methodology.

This test suite creates failing tests FIRST before any implementation.
All tests are designed to fail initially as the components don't exist yet.

Key Requirements:
1. Transformer-specific monitoring dashboards
   - Model health and status overview
   - Memory usage per model type  
   - Inference latency trends
   - Cache hit rates
   - Model loading times

2. Alerting rules for:
   - High memory usage (>90% of 8Gi)
   - Slow inference latency (>1000ms)
   - Model loading failures
   - Low cache hit rates (<50%)

3. GCP Cloud Monitoring integration
4. Update existing monitoring patterns
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# These imports will FAIL initially - that's expected for TDD
try:
    from src.monitoring.transformer_monitoring_dashboard import (
        TransformerMonitoringDashboard,
        TransformerAlertingRules,
        TransformerMetricsCollector,
        TransformerHealthPanel,
        TransformerMemoryPanel, 
        TransformerLatencyPanel,
        TransformerCachePanel,
        TransformerLoadingPanel
    )
    from src.monitoring.gcp_transformer_monitoring import (
        GCPTransformerMonitoring,
        TransformerCloudMonitoringClient,
        TransformerAlertPolicyManager
    )
except ImportError:
    # Expected to fail initially - components don't exist yet
    pass


class TestTransformerMonitoringDashboard:
    """Test transformer monitoring dashboard functionality."""
    
    @pytest.fixture
    def mock_models(self):
        """Mock transformer models for testing."""
        return {
            'itransformer': Mock(
                model_type='ITRANSFORMER',
                memory_usage_mb=1500.0,
                last_inference_latency_ms=85.0,
                cache_hit_rate=0.75,
                loading_time_ms=2500.0,
                is_healthy=True
            ),
            'patchtst': Mock(
                model_type='PATCHTST', 
                memory_usage_mb=1200.0,
                last_inference_latency_ms=120.0,
                cache_hit_rate=0.82,
                loading_time_ms=1800.0,
                is_healthy=True
            ),
            'timesmixer': Mock(
                model_type='TIMESMIXER',
                memory_usage_mb=1800.0,
                last_inference_latency_ms=95.0,
                cache_hit_rate=0.45,  # Below 50% threshold
                loading_time_ms=3200.0,
                is_healthy=False
            ),
            'timesfm': Mock(
                model_type='TIMESFM',
                memory_usage_mb=7500.0,  # High memory usage (>90% of 8Gi)
                last_inference_latency_ms=1200.0,  # High latency (>1000ms)
                cache_hit_rate=0.68,
                loading_time_ms=4500.0,
                is_healthy=False
            )
        }
    
    @pytest.fixture
    def mock_gcp_config(self):
        """Mock GCP configuration for testing."""
        return {
            'project_id': 'shvyr-ai-bots',
            'region': 'us-central1',
            'service_name': 'shyvr-rlte',
            'dashboard_name': 'transformer-monitoring-dashboard'
        }
    
    def test_transformer_monitoring_dashboard_initialization(self, mock_models, mock_gcp_config):
        """Test TransformerMonitoringDashboard initialization."""
        # This test will FAIL initially - TransformerMonitoringDashboard doesn't exist
        dashboard = TransformerMonitoringDashboard(
            models=mock_models,
            gcp_config=mock_gcp_config
        )
        
        assert dashboard.models == mock_models
        assert dashboard.project_id == mock_gcp_config['project_id']
        assert dashboard.region == mock_gcp_config['region']
        assert dashboard.service_name == mock_gcp_config['service_name']
        assert dashboard.dashboard_name == mock_gcp_config['dashboard_name']
        
        # Verify components initialized
        assert hasattr(dashboard, 'health_panel')
        assert hasattr(dashboard, 'memory_panel')
        assert hasattr(dashboard, 'latency_panel')
        assert hasattr(dashboard, 'cache_panel')
        assert hasattr(dashboard, 'loading_panel')
        assert hasattr(dashboard, 'alerting_rules')
    
    @pytest.mark.asyncio
    async def test_create_dashboard_config(self, mock_models, mock_gcp_config):
        """Test dashboard configuration creation."""
        dashboard = TransformerMonitoringDashboard(
            models=mock_models,
            gcp_config=mock_gcp_config
        )
        
        config = await dashboard.create_dashboard_config()
        
        # Verify dashboard structure
        assert 'dashboard' in config
        assert 'displayName' in config['dashboard']
        assert config['dashboard']['displayName'] == 'Transformer Model Monitoring'
        
        # Verify required panels exist
        panels = config['dashboard']['mosaicLayout']['tiles']
        panel_titles = [panel['widget']['title'] for panel in panels]
        
        expected_panels = [
            'Model Health Overview',
            'Memory Usage by Model Type',
            'Inference Latency Trends',
            'Cache Hit Rates',
            'Model Loading Times'
        ]
        
        for expected_panel in expected_panels:
            assert expected_panel in panel_titles
    
    def test_transformer_health_panel_config(self, mock_models):
        """Test transformer health panel configuration."""
        panel = TransformerHealthPanel(models=mock_models)
        
        config = panel.generate_config()
        
        # Verify panel structure
        assert config['widget']['title'] == 'Model Health Overview'
        assert 'scorecard' in config['widget']
        
        # Verify metrics configuration
        metrics = config['widget']['scorecard']['timeSeries']
        assert len(metrics) >= len(mock_models)
        
        # Verify health status metrics
        for metric in metrics:
            assert 'filter' in metric
            assert 'custom.googleapis.com/transformer/health' in metric['filter']
    
    def test_transformer_memory_panel_config(self, mock_models):
        """Test transformer memory usage panel configuration."""
        panel = TransformerMemoryPanel(models=mock_models)
        
        config = panel.generate_config()
        
        # Verify panel structure
        assert config['widget']['title'] == 'Memory Usage by Model Type'
        assert 'xyChart' in config['widget']
        
        # Verify memory metrics
        datasets = config['widget']['xyChart']['dataSets']
        assert len(datasets) >= len(mock_models)
        
        # Verify memory usage metric filter
        for dataset in datasets:
            assert 'custom.googleapis.com/transformer/memory_usage' in dataset['timeSeriesQuery']['filter']
    
    def test_transformer_latency_panel_config(self, mock_models):
        """Test transformer inference latency panel configuration.""" 
        panel = TransformerLatencyPanel(models=mock_models)
        
        config = panel.generate_config()
        
        # Verify panel structure
        assert config['widget']['title'] == 'Inference Latency Trends'
        assert 'xyChart' in config['widget']
        
        # Verify latency metrics
        datasets = config['widget']['xyChart']['dataSets']
        assert len(datasets) >= len(mock_models)
        
        # Verify latency metric filter and percentiles
        for dataset in datasets:
            assert 'custom.googleapis.com/transformer/inference_latency' in dataset['timeSeriesQuery']['filter']
            
        # Verify percentile configurations (p50, p95, p99)
        percentile_datasets = [d for d in datasets if 'percentile' in d.get('legendTemplate', '')]
        assert len(percentile_datasets) >= 3  # p50, p95, p99
    
    def test_transformer_cache_panel_config(self, mock_models):
        """Test transformer cache hit rate panel configuration."""
        panel = TransformerCachePanel(models=mock_models)
        
        config = panel.generate_config()
        
        # Verify panel structure  
        assert config['widget']['title'] == 'Cache Hit Rates'
        assert 'xyChart' in config['widget']
        
        # Verify cache metrics
        datasets = config['widget']['xyChart']['dataSets']
        assert len(datasets) >= len(mock_models)
        
        # Verify cache hit rate metric
        for dataset in datasets:
            assert 'custom.googleapis.com/transformer/cache_hit_rate' in dataset['timeSeriesQuery']['filter']
    
    def test_transformer_loading_panel_config(self, mock_models):
        """Test transformer model loading time panel configuration."""
        panel = TransformerLoadingPanel(models=mock_models)
        
        config = panel.generate_config()
        
        # Verify panel structure
        assert config['widget']['title'] == 'Model Loading Times' 
        assert 'xyChart' in config['widget']
        
        # Verify loading time metrics
        datasets = config['widget']['xyChart']['dataSets']
        assert len(datasets) >= len(mock_models)
        
        # Verify loading time metric
        for dataset in datasets:
            assert 'custom.googleapis.com/transformer/loading_time' in dataset['timeSeriesQuery']['filter']


class TestTransformerAlertingRules:
    """Test transformer alerting rules functionality."""
    
    @pytest.fixture
    def mock_alert_config(self):
        """Mock alerting configuration."""
        return {
            'project_id': 'shvyr-ai-bots',
            'notification_channels': ['email-channel-1', 'slack-channel-1'],
            'thresholds': {
                'memory_usage_percent': 90.0,
                'inference_latency_ms': 1000.0,
                'cache_hit_rate_percent': 50.0,
                'loading_time_ms': 5000.0
            }
        }
    
    def test_transformer_alerting_rules_initialization(self, mock_alert_config):
        """Test TransformerAlertingRules initialization."""
        # This test will FAIL initially - TransformerAlertingRules doesn't exist
        alerting = TransformerAlertingRules(config=mock_alert_config)
        
        assert alerting.project_id == mock_alert_config['project_id']
        assert alerting.notification_channels == mock_alert_config['notification_channels']
        assert alerting.thresholds == mock_alert_config['thresholds']
    
    def test_high_memory_usage_alert_rule(self, mock_alert_config):
        """Test high memory usage alert rule creation."""
        alerting = TransformerAlertingRules(config=mock_alert_config)
        
        alert_rule = alerting.create_high_memory_usage_alert()
        
        # Verify alert rule structure
        assert alert_rule['displayName'] == 'Transformer High Memory Usage'
        assert alert_rule['combiner'] == 'OR'
        
        # Verify condition
        condition = alert_rule['conditions'][0]
        assert condition['displayName'] == 'Memory usage > 90% of 8Gi'
        assert condition['conditionThreshold']['comparison'] == 'COMPARISON_GREATER_THAN'
        assert condition['conditionThreshold']['thresholdValue'] == 7372.8  # 90% of 8Gi in MB
        
        # Verify filter
        assert 'custom.googleapis.com/transformer/memory_usage' in condition['conditionThreshold']['filter']
        
        # Verify duration and aggregation
        assert condition['conditionThreshold']['duration'] == '300s'  # 5 minutes
        assert len(condition['conditionThreshold']['aggregations']) > 0
    
    def test_slow_inference_latency_alert_rule(self, mock_alert_config):
        """Test slow inference latency alert rule creation."""
        alerting = TransformerAlertingRules(config=mock_alert_config)
        
        alert_rule = alerting.create_slow_inference_latency_alert()
        
        # Verify alert rule structure
        assert alert_rule['displayName'] == 'Transformer Slow Inference Latency'
        assert alert_rule['combiner'] == 'OR'
        
        # Verify condition
        condition = alert_rule['conditions'][0]
        assert condition['displayName'] == 'Inference latency > 1000ms'
        assert condition['conditionThreshold']['comparison'] == 'COMPARISON_GREATER_THAN'
        assert condition['conditionThreshold']['thresholdValue'] == 1000.0
        
        # Verify filter includes percentile (p95 or p99)
        assert 'custom.googleapis.com/transformer/inference_latency' in condition['conditionThreshold']['filter']
        
        # Verify duration  
        assert condition['conditionThreshold']['duration'] == '180s'  # 3 minutes
    
    def test_model_loading_failure_alert_rule(self, mock_alert_config):
        """Test model loading failure alert rule creation."""
        alerting = TransformerAlertingRules(config=mock_alert_config)
        
        alert_rule = alerting.create_model_loading_failure_alert()
        
        # Verify alert rule structure
        assert alert_rule['displayName'] == 'Transformer Model Loading Failures'
        assert alert_rule['combiner'] == 'OR'
        
        # Verify condition
        condition = alert_rule['conditions'][0]
        assert condition['displayName'] == 'Model loading failures detected'
        assert condition['conditionThreshold']['comparison'] == 'COMPARISON_GREATER_THAN'
        assert condition['conditionThreshold']['thresholdValue'] == 0.0  # Any failures
        
        # Verify filter for loading failures
        assert 'custom.googleapis.com/transformer/loading_failures' in condition['conditionThreshold']['filter']
        
        # Verify duration
        assert condition['conditionThreshold']['duration'] == '60s'  # 1 minute
    
    def test_low_cache_hit_rate_alert_rule(self, mock_alert_config):
        """Test low cache hit rate alert rule creation."""
        alerting = TransformerAlertingRules(config=mock_alert_config)
        
        alert_rule = alerting.create_low_cache_hit_rate_alert()
        
        # Verify alert rule structure
        assert alert_rule['displayName'] == 'Transformer Low Cache Hit Rate'
        assert alert_rule['combiner'] == 'OR'
        
        # Verify condition
        condition = alert_rule['conditions'][0]
        assert condition['displayName'] == 'Cache hit rate < 50%'
        assert condition['conditionThreshold']['comparison'] == 'COMPARISON_LESS_THAN'
        assert condition['conditionThreshold']['thresholdValue'] == 0.5  # 50%
        
        # Verify filter for cache hit rate
        assert 'custom.googleapis.com/transformer/cache_hit_rate' in condition['conditionThreshold']['filter']
        
        # Verify duration
        assert condition['conditionThreshold']['duration'] == '600s'  # 10 minutes
    
    @pytest.mark.asyncio
    async def test_create_all_alert_rules(self, mock_alert_config):
        """Test creation of all transformer alert rules."""
        alerting = TransformerAlertingRules(config=mock_alert_config)
        
        alert_rules = await alerting.create_all_alert_rules()
        
        # Verify all required alert rules are created
        expected_alerts = [
            'Transformer High Memory Usage',
            'Transformer Slow Inference Latency', 
            'Transformer Model Loading Failures',
            'Transformer Low Cache Hit Rate'
        ]
        
        alert_names = [rule['displayName'] for rule in alert_rules]
        
        for expected_alert in expected_alerts:
            assert expected_alert in alert_names
        
        # Verify each alert rule has proper structure
        for rule in alert_rules:
            assert 'displayName' in rule
            assert 'conditions' in rule
            assert 'combiner' in rule
            assert 'notificationChannels' in rule
            assert rule['notificationChannels'] == mock_alert_config['notification_channels']


class TestGCPTransformerMonitoring:
    """Test GCP Cloud Monitoring integration for transformers."""
    
    @pytest.fixture
    def mock_gcp_client(self):
        """Mock GCP monitoring client."""
        client = Mock()
        client.project_name = 'projects/shvyr-ai-bots'
        client.create_metric_descriptor = Mock()
        client.create_alert_policy = Mock()
        client.create_dashboard = Mock()
        return client
    
    def test_gcp_transformer_monitoring_initialization(self, mock_gcp_client):
        """Test GCP transformer monitoring initialization."""
        # This test will FAIL initially - GCPTransformerMonitoring doesn't exist
        monitoring = GCPTransformerMonitoring(
            project_id='shvyr-ai-bots',
            region='us-central1'
        )
        
        assert monitoring.project_id == 'shvyr-ai-bots'
        assert monitoring.region == 'us-central1'
        assert hasattr(monitoring, 'monitoring_client')
        assert hasattr(monitoring, 'alert_client')
        assert hasattr(monitoring, 'dashboard_client')
    
    @pytest.mark.asyncio
    async def test_create_transformer_custom_metrics(self, mock_gcp_client):
        """Test creation of transformer custom metrics in GCP."""
        monitoring = GCPTransformerMonitoring(
            project_id='shvyr-ai-bots',
            region='us-central1'
        )
        monitoring.monitoring_client = mock_gcp_client
        
        metrics = await monitoring.create_transformer_custom_metrics()
        
        # Verify required custom metrics are created
        expected_metrics = [
            'custom.googleapis.com/transformer/memory_usage',
            'custom.googleapis.com/transformer/inference_latency',
            'custom.googleapis.com/transformer/cache_hit_rate',
            'custom.googleapis.com/transformer/loading_time',
            'custom.googleapis.com/transformer/health_status',
            'custom.googleapis.com/transformer/loading_failures'
        ]
        
        metric_types = [metric['type'] for metric in metrics]
        
        for expected_metric in expected_metrics:
            assert expected_metric in metric_types
    
    @pytest.mark.asyncio
    async def test_deploy_dashboard_to_gcp(self, mock_gcp_client, mock_models, mock_gcp_config):
        """Test deploying transformer dashboard to GCP Cloud Monitoring."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_gcp_config['project_id'],
            region=mock_gcp_config['region']
        )
        monitoring.dashboard_client = mock_gcp_client
        
        dashboard = TransformerMonitoringDashboard(
            models=mock_models,
            gcp_config=mock_gcp_config
        )
        
        dashboard_config = await dashboard.create_dashboard_config()
        result = await monitoring.deploy_dashboard(dashboard_config)
        
        # Verify dashboard deployment
        assert result['status'] == 'success'
        assert 'dashboard_id' in result
        mock_gcp_client.create_dashboard.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deploy_alert_policies_to_gcp(self, mock_gcp_client, mock_alert_config):
        """Test deploying transformer alert policies to GCP Cloud Monitoring."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_alert_config['project_id'],
            region='us-central1'
        )
        monitoring.alert_client = mock_gcp_client
        
        alerting = TransformerAlertingRules(config=mock_alert_config)
        alert_rules = await alerting.create_all_alert_rules()
        
        result = await monitoring.deploy_alert_policies(alert_rules)
        
        # Verify alert policy deployment
        assert result['status'] == 'success'
        assert result['deployed_policies'] == len(alert_rules)
        assert mock_gcp_client.create_alert_policy.call_count == len(alert_rules)


class TestTransformerMetricsCollection:
    """Test transformer metrics collection functionality."""
    
    @pytest.fixture
    def mock_transformer_models(self):
        """Mock transformer models with realistic metrics."""
        models = {}
        model_types = ['itransformer', 'patchtst', 'timesmixer', 'timesfm']
        
        for i, model_type in enumerate(model_types):
            models[model_type] = Mock()
            models[model_type].model_type = model_type.upper()
            models[model_type].get_memory_usage.return_value = 1000.0 + (i * 500)
            models[model_type].get_inference_latency.return_value = 50.0 + (i * 25)
            models[model_type].get_cache_hit_rate.return_value = 0.7 + (i * 0.05)
            models[model_type].get_loading_time.return_value = 2000.0 + (i * 500)
            models[model_type].is_healthy.return_value = True
            
        return models
    
    def test_transformer_metrics_collector_initialization(self, mock_transformer_models):
        """Test TransformerMetricsCollector initialization."""
        # This test will FAIL initially - TransformerMetricsCollector doesn't exist
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=30
        )
        
        assert collector.models == mock_transformer_models
        assert collector.collection_interval_seconds == 30
        assert hasattr(collector, 'metrics_history')
        assert hasattr(collector, 'is_collecting')
    
    @pytest.mark.asyncio
    async def test_collect_memory_metrics(self, mock_transformer_models):
        """Test memory usage metrics collection."""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=30
        )
        
        memory_metrics = await collector.collect_memory_metrics()
        
        # Verify memory metrics structure
        assert len(memory_metrics) == len(mock_transformer_models)
        
        for model_name, metrics in memory_metrics.items():
            assert 'memory_usage_mb' in metrics
            assert 'memory_usage_percent' in metrics  # Of 8Gi total
            assert 'timestamp' in metrics
            assert metrics['memory_usage_mb'] > 0
            assert 0 <= metrics['memory_usage_percent'] <= 100
    
    @pytest.mark.asyncio
    async def test_collect_latency_metrics(self, mock_transformer_models):
        """Test inference latency metrics collection."""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=30
        )
        
        latency_metrics = await collector.collect_latency_metrics()
        
        # Verify latency metrics structure
        assert len(latency_metrics) == len(mock_transformer_models)
        
        for model_name, metrics in latency_metrics.items():
            assert 'inference_latency_ms' in metrics
            assert 'latency_percentiles' in metrics
            assert 'timestamp' in metrics
            assert metrics['inference_latency_ms'] > 0
            
            # Verify percentiles
            percentiles = metrics['latency_percentiles']
            assert 'p50' in percentiles
            assert 'p95' in percentiles
            assert 'p99' in percentiles
    
    @pytest.mark.asyncio
    async def test_collect_cache_metrics(self, mock_transformer_models):
        """Test cache hit rate metrics collection."""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=30
        )
        
        cache_metrics = await collector.collect_cache_metrics()
        
        # Verify cache metrics structure
        assert len(cache_metrics) == len(mock_transformer_models)
        
        for model_name, metrics in cache_metrics.items():
            assert 'cache_hit_rate' in metrics
            assert 'cache_miss_rate' in metrics
            assert 'total_requests' in metrics
            assert 'timestamp' in metrics
            assert 0 <= metrics['cache_hit_rate'] <= 1
            assert metrics['cache_hit_rate'] + metrics['cache_miss_rate'] == 1
    
    @pytest.mark.asyncio
    async def test_collect_loading_metrics(self, mock_transformer_models):
        """Test model loading time metrics collection."""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=30
        )
        
        loading_metrics = await collector.collect_loading_metrics()
        
        # Verify loading metrics structure
        assert len(loading_metrics) == len(mock_transformer_models)
        
        for model_name, metrics in loading_metrics.items():
            assert 'loading_time_ms' in metrics
            assert 'loading_success' in metrics
            assert 'loading_failures' in metrics
            assert 'timestamp' in metrics
            assert metrics['loading_time_ms'] > 0
            assert isinstance(metrics['loading_success'], bool)
            assert isinstance(metrics['loading_failures'], int)
    
    @pytest.mark.asyncio
    async def test_collect_all_transformer_metrics(self, mock_transformer_models):
        """Test collection of all transformer metrics."""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=30
        )
        
        all_metrics = await collector.collect_all_metrics()
        
        # Verify comprehensive metrics collection
        assert 'memory_metrics' in all_metrics
        assert 'latency_metrics' in all_metrics
        assert 'cache_metrics' in all_metrics
        assert 'loading_metrics' in all_metrics
        assert 'health_metrics' in all_metrics
        assert 'timestamp' in all_metrics
        
        # Verify each metric category has data for all models
        for metric_category in ['memory_metrics', 'latency_metrics', 'cache_metrics', 'loading_metrics']:
            metrics = all_metrics[metric_category]
            assert len(metrics) == len(mock_transformer_models)
    
    @pytest.mark.asyncio
    async def test_start_continuous_collection(self, mock_transformer_models):
        """Test starting continuous metrics collection."""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=1  # Fast for testing
        )
        
        # Start collection
        await collector.start_collection()
        assert collector.is_collecting is True
        
        # Wait for some metrics to be collected
        await asyncio.sleep(2.5)
        
        # Stop collection
        await collector.stop_collection()
        assert collector.is_collecting is False
        
        # Verify metrics were collected
        assert len(collector.metrics_history) > 0
        
        # Verify metrics structure
        latest_metrics = collector.metrics_history[-1]
        assert 'memory_metrics' in latest_metrics
        assert 'latency_metrics' in latest_metrics
        assert 'cache_metrics' in latest_metrics
        assert 'loading_metrics' in latest_metrics


class TestTransformerMonitoringIntegration:
    """Test integration with existing monitoring systems."""
    
    @pytest.mark.asyncio
    async def test_integration_with_existing_monitoring(self):
        """Test integration with existing production monitoring dashboard."""
        # Test will FAIL initially - integration doesn't exist
        from src.monitoring.production_monitoring_dashboard import ProductionMonitoringDashboard
        
        # Create transformer monitoring
        transformer_monitoring = TransformerMonitoringDashboard(
            models={},
            gcp_config={'project_id': 'test', 'region': 'us-central1', 'service_name': 'test', 'dashboard_name': 'test'}
        )
        
        # Create existing production monitoring
        production_monitoring = ProductionMonitoringDashboard(
            project_id='test',
            region='us-central1'
        )
        
        # Integrate transformer metrics into production dashboard
        integrated_dashboard = await transformer_monitoring.integrate_with_production_monitoring(
            production_monitoring
        )
        
        # Verify integration
        assert integrated_dashboard is not None
        assert hasattr(integrated_dashboard, 'transformer_metrics')
        assert hasattr(integrated_dashboard, 'transformer_alerts')
    
    @pytest.mark.asyncio  
    async def test_update_existing_monitoring_scripts(self):
        """Test updating existing monitoring scripts with transformer metrics."""
        # Test will FAIL initially - script updates don't exist
        from src.monitoring.transformer_monitoring_dashboard import update_monitoring_scripts
        
        script_updates = await update_monitoring_scripts()
        
        # Verify script updates
        assert 'setup_production_monitoring.py' in script_updates
        assert 'production_monitoring_dashboard.py' in script_updates
        
        # Verify transformer metrics added to existing scripts
        for script_name, updates in script_updates.items():
            assert 'transformer_metrics' in updates
            assert 'transformer_alerts' in updates


class TestTransformerMonitoringPerformance:
    """Test performance requirements for transformer monitoring."""
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self, mock_transformer_models):
        """Test metrics collection performance requirements."""
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=30
        )
        
        # Measure collection time
        start_time = datetime.now()
        metrics = await collector.collect_all_metrics()
        end_time = datetime.now()
        
        collection_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance requirements
        assert collection_time_ms < 1000  # <1 second collection time
        assert metrics is not None
        assert len(metrics) > 0
    
    @pytest.mark.asyncio
    async def test_dashboard_creation_performance(self, mock_models, mock_gcp_config):
        """Test dashboard creation performance."""
        dashboard = TransformerMonitoringDashboard(
            models=mock_models,
            gcp_config=mock_gcp_config
        )
        
        # Measure dashboard creation time
        start_time = datetime.now()
        config = await dashboard.create_dashboard_config()
        end_time = datetime.now()
        
        creation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance requirements
        assert creation_time_ms < 2000  # <2 seconds creation time
        assert config is not None
        assert 'dashboard' in config
    
    @pytest.mark.asyncio
    async def test_alert_rule_creation_performance(self, mock_alert_config):
        """Test alert rule creation performance."""
        alerting = TransformerAlertingRules(config=mock_alert_config)
        
        # Measure alert rule creation time
        start_time = datetime.now()
        alert_rules = await alerting.create_all_alert_rules()
        end_time = datetime.now()
        
        creation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance requirements
        assert creation_time_ms < 1500  # <1.5 seconds creation time
        assert alert_rules is not None
        assert len(alert_rules) >= 4  # All required alert rules


class TestTransformerMonitoringErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_handle_missing_models(self):
        """Test handling when transformer models are missing."""
        dashboard = TransformerMonitoringDashboard(
            models={},  # Empty models
            gcp_config={'project_id': 'test', 'region': 'us-central1', 'service_name': 'test', 'dashboard_name': 'test'}
        )
        
        config = await dashboard.create_dashboard_config()
        
        # Should handle gracefully with empty models
        assert config is not None
        assert 'dashboard' in config
        
        # Should have default panels even with no models
        panels = config['dashboard']['mosaicLayout']['tiles']
        assert len(panels) > 0
    
    @pytest.mark.asyncio
    async def test_handle_gcp_api_errors(self, mock_models, mock_gcp_config):
        """Test handling GCP API errors."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_gcp_config['project_id'],
            region=mock_gcp_config['region']
        )
        
        # Mock GCP API error
        monitoring.monitoring_client = Mock()
        monitoring.monitoring_client.create_metric_descriptor.side_effect = Exception("GCP API Error")
        
        # Should handle API errors gracefully
        result = await monitoring.create_transformer_custom_metrics()
        
        assert result is not None
        assert 'error' in result
        assert 'status' in result
        assert result['status'] == 'partial_failure'
    
    @pytest.mark.asyncio
    async def test_handle_model_metric_errors(self, mock_transformer_models):
        """Test handling when model metrics are unavailable."""
        # Mock model to raise errors
        mock_transformer_models['itransformer'].get_memory_usage.side_effect = Exception("Memory unavailable")
        mock_transformer_models['patchtst'].get_inference_latency.side_effect = Exception("Latency unavailable")
        
        collector = TransformerMetricsCollector(
            models=mock_transformer_models,
            collection_interval_seconds=30
        )
        
        # Should handle model errors gracefully
        metrics = await collector.collect_all_metrics()
        
        assert metrics is not None
        assert 'error_count' in metrics
        assert metrics['error_count'] > 0
        
        # Should still collect available metrics
        assert 'memory_metrics' in metrics
        assert 'latency_metrics' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])