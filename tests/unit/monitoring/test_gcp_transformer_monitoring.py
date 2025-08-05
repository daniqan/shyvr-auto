#!/usr/bin/env python3

"""
GCP Transformer Monitoring Tests

Tests for GCP Cloud Monitoring integration with transformer models following TDD methodology.

This test suite creates failing tests FIRST before any implementation.
All tests are designed to fail initially as the components don't exist yet.

Key Requirements:
1. GCP Cloud Monitoring dashboard creation
2. Custom metrics for transformer models
3. Alert policy deployment
4. Integration with existing GCP monitoring infrastructure
5. Cloud Run transformer monitoring
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# These imports will FAIL initially - that's expected for TDD
try:
    from src.monitoring.gcp_transformer_monitoring import (
        GCPTransformerMonitoring,
        TransformerCloudMonitoringClient,
        TransformerAlertPolicyManager,
        TransformerDashboardManager,
        TransformerCustomMetrics
    )
    from deploy.monitoring.setup_transformer_monitoring import (
        TransformerMonitoringSetup,
        setup_transformer_monitoring,
        create_transformer_dashboards,
        deploy_transformer_alerts
    )
except ImportError:
    # Expected to fail initially - components don't exist yet
    pass


class TestGCPTransformerMonitoring:
    """Test GCP Cloud Monitoring integration for transformer models."""
    
    @pytest.fixture
    def mock_gcp_config(self):
        """Mock GCP configuration."""
        return {
            'project_id': 'shvyr-ai-bots',
            'region': 'us-central1',
            'service_name': 'shyvr-rlte',
            'instance_name': 'shyvr-rlte-prod'
        }
    
    @pytest.fixture
    def mock_transformer_models(self):
        """Mock transformer model configurations."""
        return {
            'itransformer': {
                'model_type': 'ITRANSFORMER',
                'memory_limit_mb': 2048,
                'latency_threshold_ms': 100,
                'cache_enabled': True
            },
            'patchtst': {
                'model_type': 'PATCHTST',
                'memory_limit_mb': 1536,
                'latency_threshold_ms': 150,
                'cache_enabled': True
            },
            'timesmixer': {
                'model_type': 'TIMESMIXER',
                'memory_limit_mb': 2560,
                'latency_threshold_ms': 120,
                'cache_enabled': True
            },
            'timesfm': {
                'model_type': 'TIMESFM',
                'memory_limit_mb': 7680,  # 7.5Gi for large model
                'latency_threshold_ms': 500,
                'cache_enabled': True
            }
        }
    
    def test_gcp_transformer_monitoring_initialization(self, mock_gcp_config):
        """Test GCP transformer monitoring initialization."""
        # This test will FAIL initially - GCPTransformerMonitoring doesn't exist
        monitoring = GCPTransformerMonitoring(
            project_id=mock_gcp_config['project_id'],
            region=mock_gcp_config['region'],
            service_name=mock_gcp_config['service_name']
        )
        
        assert monitoring.project_id == mock_gcp_config['project_id']
        assert monitoring.region == mock_gcp_config['region']
        assert monitoring.service_name == mock_gcp_config['service_name']
        
        # Verify GCP clients initialized
        assert hasattr(monitoring, 'monitoring_client')
        assert hasattr(monitoring, 'alert_client')
        assert hasattr(monitoring, 'dashboard_client')
        assert hasattr(monitoring, 'logging_client')
    
    @pytest.mark.asyncio
    async def test_create_transformer_custom_metrics(self, mock_gcp_config, mock_transformer_models):
        """Test creation of transformer-specific custom metrics."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_gcp_config['project_id'],
            region=mock_gcp_config['region']
        )
        
        custom_metrics = await monitoring.create_transformer_custom_metrics(mock_transformer_models)
        
        # Verify required custom metrics created
        expected_metrics = [
            'custom.googleapis.com/transformer/memory_usage_mb',
            'custom.googleapis.com/transformer/memory_usage_percent',
            'custom.googleapis.com/transformer/inference_latency_ms', 
            'custom.googleapis.com/transformer/inference_latency_p95',
            'custom.googleapis.com/transformer/inference_latency_p99',
            'custom.googleapis.com/transformer/cache_hit_rate',
            'custom.googleapis.com/transformer/cache_miss_rate',
            'custom.googleapis.com/transformer/model_loading_time_ms',
            'custom.googleapis.com/transformer/model_loading_failures',
            'custom.googleapis.com/transformer/model_health_status',
            'custom.googleapis.com/transformer/attention_entropy',
            'custom.googleapis.com/transformer/gradient_norm',
            'custom.googleapis.com/transformer/active_models_count'
        ]
        
        created_metric_types = [metric['type'] for metric in custom_metrics]
        
        for expected_metric in expected_metrics:
            assert expected_metric in created_metric_types
        
        # Verify metric descriptors have proper configuration
        for metric in custom_metrics:
            assert 'type' in metric
            assert 'metricKind' in metric
            assert 'valueType' in metric
            assert 'displayName' in metric
            assert 'description' in metric
            assert 'labels' in metric
            
            # Verify labels include model_type
            label_keys = [label['key'] for label in metric['labels']]
            assert 'model_type' in label_keys
    
    @pytest.mark.asyncio
    async def test_create_transformer_dashboard(self, mock_gcp_config, mock_transformer_models):
        """Test creation of comprehensive transformer monitoring dashboard."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_gcp_config['project_id'],
            region=mock_gcp_config['region']
        )
        
        dashboard_config = await monitoring.create_transformer_dashboard(mock_transformer_models)
        
        # Verify dashboard structure
        assert 'dashboard' in dashboard_config
        dashboard = dashboard_config['dashboard']
        
        assert 'displayName' in dashboard
        assert dashboard['displayName'] == 'Transformer Model Monitoring Dashboard'
        
        # Verify mosaic layout with tiles
        assert 'mosaicLayout' in dashboard
        assert 'tiles' in dashboard['mosaicLayout']
        
        tiles = dashboard['mosaicLayout']['tiles']
        assert len(tiles) >= 8  # At least 8 panels
        
        # Verify required panels exist
        panel_titles = []
        for tile in tiles:
            if 'widget' in tile and 'title' in tile['widget']:
                panel_titles.append(tile['widget']['title'])
        
        expected_panels = [
            'Transformer Model Health Overview',
            'Memory Usage by Model Type',
            'Inference Latency Trends (P95/P99)',
            'Cache Hit Rates by Model',
            'Model Loading Performance',
            'Attention Entropy Monitoring',
            'Gradient Flow Analysis',
            'Resource Utilization Summary'
        ]
        
        for expected_panel in expected_panels:
            assert expected_panel in panel_titles
    
    def test_transformer_memory_usage_panel(self, mock_gcp_config, mock_transformer_models):
        """Test memory usage panel configuration."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_gcp_config['project_id'],
            region=mock_gcp_config['region']
        )
        
        panel_config = monitoring.create_memory_usage_panel(mock_transformer_models)
        
        # Verify panel structure
        assert 'widget' in panel_config
        widget = panel_config['widget']
        
        assert widget['title'] == 'Memory Usage by Model Type'
        assert 'xyChart' in widget
        
        # Verify chart configuration
        chart = widget['xyChart']
        assert 'dataSets' in chart
        assert len(chart['dataSets']) >= len(mock_transformer_models)
        
        # Verify Y-axis configuration for memory
        assert 'yAxis' in chart
        assert chart['yAxis']['label'] == 'Memory Usage (MB)'
        assert chart['yAxis']['scale'] == 'LINEAR'
        
        # Verify datasets have proper filters
        for dataset in chart['dataSets']:
            filter_str = dataset['timeSeriesQuery']['filter']
            assert 'custom.googleapis.com/transformer/memory_usage_mb' in filter_str
            assert 'model_type' in filter_str
    
    def test_transformer_latency_panel(self, mock_gcp_config, mock_transformer_models):
        """Test inference latency panel configuration."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_gcp_config['project_id'],
            region=mock_gcp_config['region']
        )
        
        panel_config = monitoring.create_latency_panel(mock_transformer_models)
        
        # Verify panel structure
        assert 'widget' in panel_config
        widget = panel_config['widget']
        
        assert widget['title'] == 'Inference Latency Trends (P95/P99)'
        assert 'xyChart' in widget
        
        # Verify chart configuration
        chart = widget['xyChart']
        assert 'dataSets' in chart
        
        # Should have datasets for both P95 and P99 percentiles per model
        expected_datasets = len(mock_transformer_models) * 2  # P95 and P99
        assert len(chart['dataSets']) >= expected_datasets
        
        # Verify Y-axis configuration for latency
        assert 'yAxis' in chart
        assert chart['yAxis']['label'] == 'Latency (ms)'
        
        # Verify percentile datasets
        p95_datasets = [d for d in chart['dataSets'] if 'p95' in d.get('legendTemplate', '').lower()]
        p99_datasets = [d for d in chart['dataSets'] if 'p99' in d.get('legendTemplate', '').lower()]
        
        assert len(p95_datasets) >= len(mock_transformer_models)
        assert len(p99_datasets) >= len(mock_transformer_models)
    
    def test_transformer_cache_panel(self, mock_gcp_config, mock_transformer_models):
        """Test cache hit rate panel configuration."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_gcp_config['project_id'],
            region=mock_gcp_config['region']
        )
        
        panel_config = monitoring.create_cache_panel(mock_transformer_models)
        
        # Verify panel structure
        assert 'widget' in panel_config
        widget = panel_config['widget']
        
        assert widget['title'] == 'Cache Hit Rates by Model'
        assert 'xyChart' in widget
        
        # Verify chart configuration
        chart = widget['xyChart']
        assert 'dataSets' in chart
        assert len(chart['dataSets']) >= len(mock_transformer_models)
        
        # Verify Y-axis configuration for percentage
        assert 'yAxis' in chart
        assert chart['yAxis']['label'] == 'Cache Hit Rate (%)'
        assert chart['yAxis']['scale'] == 'LINEAR'
        
        # Verify datasets have cache hit rate filters
        for dataset in chart['dataSets']:
            filter_str = dataset['timeSeriesQuery']['filter']
            assert 'custom.googleapis.com/transformer/cache_hit_rate' in filter_str
    
    def test_transformer_health_panel(self, mock_gcp_config, mock_transformer_models):
        """Test model health overview panel configuration."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_gcp_config['project_id'],
            region=mock_gcp_config['region']
        )
        
        panel_config = monitoring.create_health_panel(mock_transformer_models)
        
        # Verify panel structure
        assert 'widget' in panel_config
        widget = panel_config['widget']
        
        assert widget['title'] == 'Transformer Model Health Overview'
        assert 'scorecard' in widget
        
        # Verify scorecard configuration
        scorecard = widget['scorecard']
        assert 'timeSeries' in scorecard
        assert len(scorecard['timeSeries']) >= len(mock_transformer_models)
        
        # Verify gauge configuration
        assert 'gaugeView' in scorecard
        gauge = scorecard['gaugeView']
        assert gauge['lowerBound'] == 0.0
        assert gauge['upperBound'] == 1.0
        
        # Verify health status filters
        for timeseries in scorecard['timeSeries']:
            filter_str = timeseries['filter']
            assert 'custom.googleapis.com/transformer/model_health_status' in filter_str


class TestTransformerAlertPolicies:
    """Test transformer-specific alert policy creation."""
    
    @pytest.fixture
    def mock_alert_config(self):
        """Mock alert configuration."""
        return {
            'project_id': 'shvyr-ai-bots',
            'notification_channels': [
                'projects/shvyr-ai-bots/notificationChannels/email-channel-1',
                'projects/shvyr-ai-bots/notificationChannels/slack-channel-1'
            ],
            'thresholds': {
                'memory_usage_percent': 90.0,  # 90% of 8Gi
                'memory_usage_mb': 7372.8,     # 90% of 8192MB
                'inference_latency_ms': 1000.0,
                'cache_hit_rate': 0.5,         # 50%
                'loading_time_ms': 5000.0,     # 5 seconds
                'loading_failures': 3          # 3 failures in window
            }
        }
    
    @pytest.mark.asyncio
    async def test_create_high_memory_alert_policy(self, mock_alert_config):
        """Test high memory usage alert policy creation."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_alert_config['project_id'],
            region='us-central1'
        )
        
        alert_policy = await monitoring.create_high_memory_alert_policy(mock_alert_config)
        
        # Verify alert policy structure
        assert 'displayName' in alert_policy
        assert alert_policy['displayName'] == 'Transformer High Memory Usage Alert'
        
        assert 'conditions' in alert_policy
        assert len(alert_policy['conditions']) >= 1
        
        # Verify condition configuration
        condition = alert_policy['conditions'][0]
        assert condition['displayName'] == 'Memory usage exceeds 90% of 8Gi'
        
        # Verify threshold condition
        threshold = condition['conditionThreshold']
        assert threshold['comparison'] == 'COMPARISON_GREATER_THAN'
        assert threshold['thresholdValue'] == mock_alert_config['thresholds']['memory_usage_mb']
        
        # Verify filter includes transformer memory metric
        assert 'custom.googleapis.com/transformer/memory_usage_mb' in threshold['filter']
        
        # Verify duration and aggregation
        assert 'duration' in threshold
        assert threshold['duration'] == '300s'  # 5 minutes
        assert 'aggregations' in threshold
        
        # Verify notification channels
        assert 'notificationChannels' in alert_policy
        assert alert_policy['notificationChannels'] == mock_alert_config['notification_channels']
    
    @pytest.mark.asyncio
    async def test_create_slow_inference_alert_policy(self, mock_alert_config):
        """Test slow inference latency alert policy creation."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_alert_config['project_id'],
            region='us-central1'
        )
        
        alert_policy = await monitoring.create_slow_inference_alert_policy(mock_alert_config)
        
        # Verify alert policy structure
        assert alert_policy['displayName'] == 'Transformer Slow Inference Latency Alert'
        
        # Verify condition configuration
        condition = alert_policy['conditions'][0]
        assert condition['displayName'] == 'Inference latency exceeds 1000ms'
        
        # Verify threshold for P99 latency
        threshold = condition['conditionThreshold']
        assert threshold['comparison'] == 'COMPARISON_GREATER_THAN'
        assert threshold['thresholdValue'] == mock_alert_config['thresholds']['inference_latency_ms']
        
        # Verify filter includes P99 latency metric
        assert 'custom.googleapis.com/transformer/inference_latency_p99' in threshold['filter']
        
        # Verify duration for latency alerts (shorter than memory)
        assert threshold['duration'] == '180s'  # 3 minutes
    
    @pytest.mark.asyncio
    async def test_create_model_loading_failure_alert_policy(self, mock_alert_config):
        """Test model loading failure alert policy creation."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_alert_config['project_id'],
            region='us-central1'
        )
        
        alert_policy = await monitoring.create_model_loading_failure_alert_policy(mock_alert_config)
        
        # Verify alert policy structure
        assert alert_policy['displayName'] == 'Transformer Model Loading Failures Alert'
        
        # Verify condition configuration
        condition = alert_policy['conditions'][0]
        assert condition['displayName'] == 'Model loading failures detected'
        
        # Verify threshold for loading failures
        threshold = condition['conditionThreshold']
        assert threshold['comparison'] == 'COMPARISON_GREATER_THAN'
        assert threshold['thresholdValue'] == mock_alert_config['thresholds']['loading_failures']
        
        # Verify filter includes loading failures metric
        assert 'custom.googleapis.com/transformer/model_loading_failures' in threshold['filter']
        
        # Verify short duration for critical failures
        assert threshold['duration'] == '60s'  # 1 minute
    
    @pytest.mark.asyncio
    async def test_create_low_cache_hit_rate_alert_policy(self, mock_alert_config):
        """Test low cache hit rate alert policy creation."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_alert_config['project_id'],
            region='us-central1'
        )
        
        alert_policy = await monitoring.create_low_cache_hit_rate_alert_policy(mock_alert_config)
        
        # Verify alert policy structure
        assert alert_policy['displayName'] == 'Transformer Low Cache Hit Rate Alert'
        
        # Verify condition configuration
        condition = alert_policy['conditions'][0]
        assert condition['displayName'] == 'Cache hit rate below 50%'
        
        # Verify threshold for cache hit rate
        threshold = condition['conditionThreshold']
        assert threshold['comparison'] == 'COMPARISON_LESS_THAN'
        assert threshold['thresholdValue'] == mock_alert_config['thresholds']['cache_hit_rate']
        
        # Verify filter includes cache hit rate metric
        assert 'custom.googleapis.com/transformer/cache_hit_rate' in threshold['filter']
        
        # Verify longer duration for cache performance issues
        assert threshold['duration'] == '600s'  # 10 minutes
    
    @pytest.mark.asyncio
    async def test_deploy_all_transformer_alert_policies(self, mock_alert_config):
        """Test deployment of all transformer alert policies."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_alert_config['project_id'],
            region='us-central1'
        )
        
        # Mock GCP alert client
        monitoring.alert_client = Mock()
        monitoring.alert_client.create_alert_policy = Mock(return_value=Mock())
        
        deployment_result = await monitoring.deploy_all_alert_policies(mock_alert_config)
        
        # Verify deployment result
        assert 'status' in deployment_result
        assert deployment_result['status'] == 'success'
        assert 'deployed_policies' in deployment_result
        assert deployment_result['deployed_policies'] >= 4  # At least 4 policies
        
        # Verify all required alert policies were created
        expected_policies = [
            'Transformer High Memory Usage Alert',
            'Transformer Slow Inference Latency Alert',
            'Transformer Model Loading Failures Alert', 
            'Transformer Low Cache Hit Rate Alert'
        ]
        
        deployed_policy_names = deployment_result['policy_names']
        for expected_policy in expected_policies:
            assert expected_policy in deployed_policy_names


class TestTransformerMonitoringSetup:
    """Test transformer monitoring setup and deployment scripts."""
    
    @pytest.fixture
    def mock_setup_config(self):
        """Mock setup configuration."""
        return {
            'project_id': 'shvyr-ai-bots',
            'region': 'us-central1',
            'service_name': 'shyvr-rlte',
            'monitoring_config': {
                'dashboards_enabled': True,
                'alerts_enabled': True,
                'custom_metrics_enabled': True
            },
            'transformer_models': {
                'itransformer': {'enabled': True},
                'patchtst': {'enabled': True},
                'timesmixer': {'enabled': True},
                'timesfm': {'enabled': True}
            }
        }
    
    @pytest.mark.asyncio
    async def test_transformer_monitoring_setup_initialization(self, mock_setup_config):
        """Test TransformerMonitoringSetup initialization."""
        # This test will FAIL initially - TransformerMonitoringSetup doesn't exist
        setup = TransformerMonitoringSetup(config=mock_setup_config)
        
        assert setup.project_id == mock_setup_config['project_id']
        assert setup.region == mock_setup_config['region']
        assert setup.service_name == mock_setup_config['service_name']
        assert setup.transformer_models == mock_setup_config['transformer_models']
        
        # Verify components initialized
        assert hasattr(setup, 'gcp_monitoring')
        assert hasattr(setup, 'dashboard_manager')
        assert hasattr(setup, 'alert_manager')
        assert hasattr(setup, 'metrics_manager')
    
    @pytest.mark.asyncio
    async def test_setup_transformer_monitoring_complete(self, mock_setup_config):
        """Test complete transformer monitoring setup."""
        # This test will FAIL initially - setup_transformer_monitoring doesn't exist
        result = await setup_transformer_monitoring(mock_setup_config)
        
        # Verify setup result
        assert 'status' in result
        assert result['status'] == 'success'
        
        # Verify all components were set up
        assert 'custom_metrics' in result
        assert 'dashboards' in result
        assert 'alert_policies' in result
        
        # Verify metrics were created
        assert result['custom_metrics']['created'] >= 10  # At least 10 custom metrics
        
        # Verify dashboards were created
        assert result['dashboards']['created'] >= 1
        assert 'Transformer Model Monitoring Dashboard' in result['dashboards']['names']
        
        # Verify alert policies were created
        assert result['alert_policies']['created'] >= 4
    
    @pytest.mark.asyncio
    async def test_create_transformer_dashboards(self, mock_setup_config):
        """Test transformer dashboard creation function."""
        # This test will FAIL initially - create_transformer_dashboards doesn't exist
        dashboards_result = await create_transformer_dashboards(mock_setup_config)
        
        # Verify dashboards creation
        assert 'status' in dashboards_result
        assert dashboards_result['status'] == 'success'
        assert 'dashboards' in dashboards_result
        assert len(dashboards_result['dashboards']) >= 1
        
        # Verify main dashboard created
        main_dashboard = dashboards_result['dashboards'][0]
        assert main_dashboard['name'] == 'Transformer Model Monitoring Dashboard'
        assert 'dashboard_id' in main_dashboard
        assert 'panels_count' in main_dashboard
        assert main_dashboard['panels_count'] >= 8
    
    @pytest.mark.asyncio
    async def test_deploy_transformer_alerts(self, mock_setup_config):
        """Test transformer alert deployment function."""
        # This test will FAIL initially - deploy_transformer_alerts doesn't exist
        alerts_result = await deploy_transformer_alerts(mock_setup_config)
        
        # Verify alerts deployment
        assert 'status' in alerts_result
        assert alerts_result['status'] == 'success'
        assert 'alert_policies' in alerts_result
        assert len(alerts_result['alert_policies']) >= 4
        
        # Verify required alert policies
        policy_names = [policy['name'] for policy in alerts_result['alert_policies']]
        expected_policies = [
            'Transformer High Memory Usage Alert',
            'Transformer Slow Inference Latency Alert',
            'Transformer Model Loading Failures Alert',
            'Transformer Low Cache Hit Rate Alert'
        ]
        
        for expected_policy in expected_policies:
            assert expected_policy in policy_names


class TestTransformerCloudRunMonitoring:
    """Test Cloud Run specific transformer monitoring."""
    
    @pytest.fixture
    def mock_cloud_run_config(self):
        """Mock Cloud Run configuration."""
        return {
            'project_id': 'shvyr-ai-bots',
            'region': 'us-central1',
            'service_name': 'shyvr-rlte',
            'revision_name': 'shyvr-rlte-00001-abc',
            'resource_limits': {
                'memory': '8Gi',
                'cpu': '6',
                'timeout': '4200s',
                'concurrency': 15
            }
        }
    
    @pytest.mark.asyncio
    async def test_cloud_run_transformer_metrics_integration(self, mock_cloud_run_config):
        """Test integration with Cloud Run metrics for transformers."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_cloud_run_config['project_id'],
            region=mock_cloud_run_config['region']
        )
        
        cloud_run_metrics = await monitoring.get_cloud_run_transformer_metrics(mock_cloud_run_config)
        
        # Verify Cloud Run metrics integration
        assert 'container_memory_utilization' in cloud_run_metrics
        assert 'container_cpu_utilization' in cloud_run_metrics
        assert 'request_count' in cloud_run_metrics
        assert 'request_latencies' in cloud_run_metrics
        assert 'billable_instance_time' in cloud_run_metrics
        
        # Verify transformer-specific metrics
        assert 'transformer_memory_usage' in cloud_run_metrics
        assert 'transformer_request_latency' in cloud_run_metrics
        assert 'transformer_model_loading_time' in cloud_run_metrics
    
    @pytest.mark.asyncio
    async def test_cloud_run_transformer_dashboard_panel(self, mock_cloud_run_config):
        """Test Cloud Run transformer dashboard panel creation."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_cloud_run_config['project_id'],
            region=mock_cloud_run_config['region']
        )
        
        panel_config = await monitoring.create_cloud_run_transformer_panel(mock_cloud_run_config)
        
        # Verify panel structure
        assert 'widget' in panel_config
        widget = panel_config['widget']
        
        assert widget['title'] == 'Cloud Run Transformer Resource Usage'
        assert 'xyChart' in widget
        
        # Verify datasets for memory and CPU
        chart = widget['xyChart']
        assert 'dataSets' in chart
        assert len(chart['dataSets']) >= 4  # Memory %, CPU %, Request count, Latency
        
        # Verify Cloud Run filters
        for dataset in chart['dataSets']:
            filter_str = dataset['timeSeriesQuery']['filter']
            assert 'resource.type="cloud_run_revision"' in filter_str
            assert f'resource.labels.service_name="{mock_cloud_run_config["service_name"]}"' in filter_str
    
    @pytest.mark.asyncio
    async def test_cloud_run_transformer_alerts(self, mock_cloud_run_config):
        """Test Cloud Run transformer-specific alerts."""
        monitoring = GCPTransformerMonitoring(
            project_id=mock_cloud_run_config['project_id'],
            region=mock_cloud_run_config['region']
        )
        
        alert_config = {
            'notification_channels': ['test-channel'],
            'thresholds': {
                'memory_utilization': 0.9,     # 90%
                'cpu_utilization': 0.8,        # 80%
                'instance_count': 10,           # Max instances
                'request_latency_p99': 5000.0   # 5 seconds
            }
        }
        
        cloud_run_alerts = await monitoring.create_cloud_run_transformer_alerts(
            mock_cloud_run_config, alert_config
        )
        
        # Verify Cloud Run alert policies
        assert len(cloud_run_alerts) >= 4
        
        alert_names = [alert['displayName'] for alert in cloud_run_alerts]
        expected_alerts = [
            'Cloud Run Transformer High Memory Usage',
            'Cloud Run Transformer High CPU Usage',
            'Cloud Run Transformer High Instance Count',
            'Cloud Run Transformer High Request Latency'
        ]
        
        for expected_alert in expected_alerts:
            assert expected_alert in alert_names


class TestTransformerMonitoringIntegration:
    """Test integration with existing monitoring systems."""
    
    @pytest.mark.asyncio
    async def test_integrate_with_existing_production_monitoring(self):
        """Test integration with existing production monitoring dashboard."""
        # Import existing production monitoring
        from src.monitoring.production_monitoring_dashboard import ProductionMonitoringDashboard
        
        # Create transformer monitoring
        gcp_monitoring = GCPTransformerMonitoring(
            project_id='shvyr-ai-bots',
            region='us-central1'
        )
        
        # Create existing production monitoring
        production_monitoring = ProductionMonitoringDashboard(
            project_id='shvyr-ai-bots',
            region='us-central1'
        )
        
        # Test integration
        integration_result = await gcp_monitoring.integrate_with_production_monitoring(
            production_monitoring
        )
        
        # Verify integration
        assert integration_result['status'] == 'success'
        assert 'enhanced_dashboard' in integration_result
        assert 'added_transformer_panels' in integration_result
        assert integration_result['added_transformer_panels'] >= 5
    
    @pytest.mark.asyncio
    async def test_update_existing_monitoring_scripts(self):
        """Test updating existing monitoring scripts with transformer metrics."""
        gcp_monitoring = GCPTransformerMonitoring(
            project_id='shvyr-ai-bots',
            region='us-central1'
        )
        
        script_updates = await gcp_monitoring.update_existing_monitoring_scripts()
        
        # Verify script updates
        expected_scripts = [
            'scripts/setup_production_monitoring.py',
            'monitoring/production_monitoring_dashboard.py',
            'scripts/setup_gcp_monitoring.py'
        ]
        
        for script in expected_scripts:
            assert script in script_updates
            assert 'transformer_metrics' in script_updates[script]
            assert 'transformer_alerts' in script_updates[script]


class TestTransformerMonitoringPerformance:
    """Test performance requirements for transformer monitoring."""
    
    @pytest.mark.asyncio
    async def test_dashboard_creation_performance(self):
        """Test dashboard creation performance requirements."""
        monitoring = GCPTransformerMonitoring(
            project_id='test-project',
            region='us-central1'
        )
        
        models = {'test_model': {'model_type': 'ITRANSFORMER'}}
        
        start_time = datetime.now()
        dashboard_config = await monitoring.create_transformer_dashboard(models)
        end_time = datetime.now()
        
        creation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance requirements
        assert creation_time_ms < 2000  # <2 seconds
        assert dashboard_config is not None
    
    @pytest.mark.asyncio
    async def test_metrics_creation_performance(self):
        """Test custom metrics creation performance."""
        monitoring = GCPTransformerMonitoring(
            project_id='test-project',
            region='us-central1'
        )
        
        models = {f'model_{i}': {'model_type': 'ITRANSFORMER'} for i in range(4)}
        
        start_time = datetime.now()
        metrics = await monitoring.create_transformer_custom_metrics(models)
        end_time = datetime.now()
        
        creation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance requirements
        assert creation_time_ms < 5000  # <5 seconds for all metrics
        assert len(metrics) >= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])