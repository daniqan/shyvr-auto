"""
Tests for Google Cloud Monitoring integration.

This module tests the Cloud Monitoring integration including:
- Metric descriptor creation
- Time series data writing
- Alert policy management
- Health checks and validation
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal

from src.monitoring.cloud_monitoring import (
    CloudMonitoringClient,
    CloudMonitoringCollector,
    CloudMetricDescriptor,
    AlertPolicyCondition,
    NotificationChannel,
    cloud_monitoring_context
)
from src.monitoring.base import MetricsRegistry


@pytest.fixture
def mock_gcp_clients():
    """Mock GCP client dependencies."""
    with patch('src.monitoring.cloud_monitoring.monitoring_v3') as mock_monitoring:
        mock_client = Mock()
        mock_alert_client = Mock()
        mock_notification_client = Mock()
        mock_dashboard_client = Mock()
        
        mock_monitoring.MetricServiceClient.return_value = mock_client
        mock_monitoring.AlertPolicyServiceClient.return_value = mock_alert_client
        mock_monitoring.NotificationChannelServiceClient.return_value = mock_notification_client
        
        with patch('src.monitoring.cloud_monitoring.dashboard_v1') as mock_dashboard:
            mock_dashboard.DashboardsServiceClient.return_value = mock_dashboard_client
            
            yield {
                'metric_client': mock_client,
                'alert_client': mock_alert_client,
                'notification_client': mock_notification_client,
                'dashboard_client': mock_dashboard_client
            }


@pytest.fixture
def cloud_monitoring_client(mock_gcp_clients):
    """Create CloudMonitoringClient instance with mocked dependencies."""
    return CloudMonitoringClient("test-project-id")


@pytest.fixture
def metrics_registry():
    """Create MetricsRegistry instance."""
    return MetricsRegistry()


@pytest.fixture
def cloud_collector(cloud_monitoring_client, metrics_registry):
    """Create CloudMonitoringCollector instance."""
    with patch.object(CloudMonitoringCollector, '__init__', lambda x, registry, project_id: None):
        collector = CloudMonitoringCollector.__new__(CloudMonitoringCollector)
        collector.registry = metrics_registry
        collector.project_id = "test-project-id"
        collector.cloud_client = cloud_monitoring_client
        collector._metrics_buffer = []
        collector._last_flush_time = 0
        collector._flush_interval = 60
        collector._init_prometheus_metrics()
        return collector


class TestCloudMetricDescriptor:
    """Test CloudMetricDescriptor dataclass."""
    
    def test_metric_descriptor_creation(self):
        """Test creating metric descriptor with all fields."""
        descriptor = CloudMetricDescriptor(
            name="test_metric",
            display_name="Test Metric",
            description="A test metric",
            metric_kind="GAUGE",
            value_type="DOUBLE",
            unit="USD",
            labels=[{"key": "test", "description": "Test label"}]
        )
        
        assert descriptor.name == "test_metric"
        assert descriptor.display_name == "Test Metric"
        assert descriptor.description == "A test metric"
        assert descriptor.metric_kind == "GAUGE"
        assert descriptor.value_type == "DOUBLE"
        assert descriptor.unit == "USD"
        assert len(descriptor.labels) == 1
        assert descriptor.labels[0]["key"] == "test"
    
    def test_metric_descriptor_defaults(self):
        """Test metric descriptor with default values."""
        descriptor = CloudMetricDescriptor(
            name="simple_metric",
            display_name="Simple Metric",
            description="A simple metric",
            metric_kind="COUNTER",
            value_type="INT64"
        )
        
        assert descriptor.unit == ""
        assert descriptor.labels == []


class TestAlertPolicyCondition:
    """Test AlertPolicyCondition dataclass."""
    
    def test_alert_condition_creation(self):
        """Test creating alert policy condition."""
        condition = AlertPolicyCondition(
            filter='metric.type="custom.googleapis.com/test"',
            comparison="COMPARISON_GT",
            threshold_value=0.8,
            duration="300s"
        )
        
        assert condition.filter == 'metric.type="custom.googleapis.com/test"'
        assert condition.comparison == "COMPARISON_GT"
        assert condition.threshold_value == 0.8
        assert condition.duration == "300s"
    
    def test_alert_condition_defaults(self):
        """Test alert condition with default values."""
        condition = AlertPolicyCondition(
            filter="test_filter",
            comparison="COMPARISON_LT",
            threshold_value=10.0
        )
        
        assert condition.duration == "300s"
        assert condition.aggregation_alignment_period == "60s"
        assert condition.aggregation_per_series_aligner == "ALIGN_RATE"


class TestNotificationChannel:
    """Test NotificationChannel dataclass."""
    
    def test_notification_channel_creation(self):
        """Test creating notification channel."""
        channel = NotificationChannel(
            type="email",
            display_name="Test Email",
            labels={"email_address": "test@example.com"},
            enabled=True
        )
        
        assert channel.type == "email"
        assert channel.display_name == "Test Email"
        assert channel.labels["email_address"] == "test@example.com"
        assert channel.enabled is True
    
    def test_notification_channel_defaults(self):
        """Test notification channel with default values."""
        channel = NotificationChannel(
            type="slack",
            display_name="Test Slack",
            labels={"url": "https://hooks.slack.com/test"}
        )
        
        assert channel.enabled is True


class TestCloudMonitoringClient:
    """Test CloudMonitoringClient class."""
    
    def test_client_initialization(self, mock_gcp_clients):
        """Test client initialization."""
        client = CloudMonitoringClient("test-project")
        
        assert client.project_id == "test-project"
        assert client.project_name == "projects/test-project"
        assert client.client is not None
        assert client.alert_client is not None
    
    @pytest.mark.asyncio
    async def test_create_metric_descriptor_success(self, cloud_monitoring_client, mock_gcp_clients):
        """Test successful metric descriptor creation."""
        descriptor = CloudMetricDescriptor(
            name="test_metric",
            display_name="Test Metric",
            description="Test description",
            metric_kind="GAUGE",
            value_type="DOUBLE"
        )
        
        # Mock successful creation
        mock_descriptor = Mock()
        mock_descriptor.type = "custom.googleapis.com/shyvr_rlte/test_metric"
        mock_gcp_clients['metric_client'].create_metric_descriptor.return_value = mock_descriptor
        
        result = await cloud_monitoring_client.create_metric_descriptor(descriptor)
        
        assert result is True
        mock_gcp_clients['metric_client'].create_metric_descriptor.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_metric_descriptor_already_exists(self, cloud_monitoring_client, mock_gcp_clients):
        """Test metric descriptor creation when already exists."""
        descriptor = CloudMetricDescriptor(
            name="existing_metric",
            display_name="Existing Metric",
            description="Already exists",
            metric_kind="COUNTER",
            value_type="INT64"
        )
        
        # Mock AlreadyExists exception
        from google.api_core import exceptions as gcp_exceptions
        mock_gcp_clients['metric_client'].create_metric_descriptor.side_effect = gcp_exceptions.AlreadyExists("Already exists")
        
        result = await cloud_monitoring_client.create_metric_descriptor(descriptor)
        
        assert result is True  # Should return True for already existing
    
    @pytest.mark.asyncio
    async def test_create_metric_descriptor_failure(self, cloud_monitoring_client, mock_gcp_clients):
        """Test metric descriptor creation failure."""
        descriptor = CloudMetricDescriptor(
            name="failing_metric",
            display_name="Failing Metric",
            description="Will fail",
            metric_kind="GAUGE",
            value_type="DOUBLE"
        )
        
        # Mock general exception
        mock_gcp_clients['metric_client'].create_metric_descriptor.side_effect = Exception("Creation failed")
        
        result = await cloud_monitoring_client.create_metric_descriptor(descriptor)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_write_time_series_success(self, cloud_monitoring_client, mock_gcp_clients):
        """Test successful time series writing."""
        mock_gcp_clients['metric_client'].create_time_series.return_value = None
        
        result = await cloud_monitoring_client.write_time_series(
            metric_type="test_metric",
            value=42.0,
            labels={"test": "value"}
        )
        
        assert result is True
        mock_gcp_clients['metric_client'].create_time_series.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_write_time_series_failure(self, cloud_monitoring_client, mock_gcp_clients):
        """Test time series writing failure."""
        mock_gcp_clients['metric_client'].create_time_series.side_effect = Exception("Write failed")
        
        result = await cloud_monitoring_client.write_time_series(
            metric_type="failing_metric",
            value=1.0
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_alert_policy_success(self, cloud_monitoring_client, mock_gcp_clients):
        """Test successful alert policy creation."""
        condition = AlertPolicyCondition(
            filter="test_filter",
            comparison="COMPARISON_GT",
            threshold_value=0.5
        )
        
        mock_policy = Mock()
        mock_policy.name = "projects/test-project/alertPolicies/123"
        mock_gcp_clients['alert_client'].create_alert_policy.return_value = mock_policy
        
        result = await cloud_monitoring_client.create_alert_policy(
            display_name="Test Alert",
            conditions=[condition],
            notification_channels=["test-channel"]
        )
        
        assert result == "projects/test-project/alertPolicies/123"
        mock_gcp_clients['alert_client'].create_alert_policy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_alert_policy_failure(self, cloud_monitoring_client, mock_gcp_clients):
        """Test alert policy creation failure."""
        condition = AlertPolicyCondition(
            filter="test_filter",
            comparison="COMPARISON_GT",
            threshold_value=0.5
        )
        
        mock_gcp_clients['alert_client'].create_alert_policy.side_effect = Exception("Policy creation failed")
        
        result = await cloud_monitoring_client.create_alert_policy(
            display_name="Failing Alert",
            conditions=[condition],
            notification_channels=[]
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_alert_policies(self, cloud_monitoring_client, mock_gcp_clients):
        """Test listing alert policies."""
        mock_policy1 = Mock()
        mock_policy1.name = "policy1"
        mock_policy1.display_name = "Test Policy 1"
        mock_policy1.enabled = True
        mock_policy1.conditions = [Mock()]
        mock_policy1.notification_channels = ["channel1"]
        
        mock_policy2 = Mock()
        mock_policy2.name = "policy2"
        mock_policy2.display_name = "Test Policy 2"
        mock_policy2.enabled = False
        mock_policy2.conditions = [Mock(), Mock()]
        mock_policy2.notification_channels = ["channel2", "channel3"]
        
        mock_gcp_clients['alert_client'].list_alert_policies.return_value = [mock_policy1, mock_policy2]
        
        policies = await cloud_monitoring_client.list_alert_policies()
        
        assert len(policies) == 2
        assert policies[0]["name"] == "policy1"
        assert policies[0]["display_name"] == "Test Policy 1"
        assert policies[0]["enabled"] is True
        assert policies[0]["conditions_count"] == 1
        assert policies[1]["conditions_count"] == 2


class TestCloudMonitoringCollector:
    """Test CloudMonitoringCollector class."""
    
    def test_collector_initialization(self, cloud_collector):
        """Test collector initialization."""
        assert cloud_collector.project_id == "test-project-id"
        assert cloud_collector._metrics_buffer == []
        assert hasattr(cloud_collector, 'cloud_metrics_sent')
        assert hasattr(cloud_collector, 'cloud_alerts_created')
    
    @pytest.mark.asyncio
    async def test_record_trading_metric(self, cloud_collector):
        """Test recording trading metrics."""
        with patch.object(cloud_collector, '_flush_metrics', AsyncMock()) as mock_flush:
            await cloud_collector.record_trading_metric(
                "pnl",
                Decimal("100.50"),
                {"currency": "USD"}
            )
            
            assert len(cloud_collector._metrics_buffer) == 1
            metric = cloud_collector._metrics_buffer[0]
            assert metric["metric_type"] == "trading/pnl"
            assert metric["value"] == 100.50
            assert metric["labels"]["currency"] == "USD"
            assert "timestamp" in metric
    
    @pytest.mark.asyncio
    async def test_record_safety_metric(self, cloud_collector):
        """Test recording safety metrics."""
        await cloud_collector.record_safety_metric(
            "risk_level",
            0.75,
            {"risk_type": "overall"}
        )
        
        assert len(cloud_collector._metrics_buffer) == 1
        metric = cloud_collector._metrics_buffer[0]
        assert metric["metric_type"] == "safety/risk_level"
        assert metric["value"] == 0.75
        assert metric["labels"]["risk_type"] == "overall"
    
    @pytest.mark.asyncio
    async def test_record_system_metric(self, cloud_collector):
        """Test recording system metrics."""
        await cloud_collector.record_system_metric(
            "uptime",
            1,
            {"component": "trading_system"}
        )
        
        assert len(cloud_collector._metrics_buffer) == 1
        metric = cloud_collector._metrics_buffer[0]
        assert metric["metric_type"] == "system/uptime"
        assert metric["value"] == 1
        assert metric["labels"]["component"] == "trading_system"
    
    @pytest.mark.asyncio
    async def test_flush_metrics_success(self, cloud_collector):
        """Test successful metrics flushing."""
        # Add test metrics to buffer
        cloud_collector._metrics_buffer = [
            {
                "metric_type": "trading/test",
                "value": 42.0,
                "labels": {"test": "value"},
                "timestamp": datetime.utcnow()
            },
            {
                "metric_type": "safety/test",
                "value": 0.5,
                "labels": {},
                "timestamp": datetime.utcnow()
            }
        ]
        
        with patch.object(cloud_collector.cloud_client, 'write_time_series', AsyncMock(return_value=True)):
            await cloud_collector._flush_metrics()
        
        assert len(cloud_collector._metrics_buffer) == 0
        assert cloud_collector._last_flush_time > 0
    
    @pytest.mark.asyncio
    async def test_flush_metrics_partial_failure(self, cloud_collector):
        """Test metrics flushing with partial failures."""
        cloud_collector._metrics_buffer = [
            {
                "metric_type": "trading/success",
                "value": 1.0,
                "labels": {},
                "timestamp": datetime.utcnow()
            },
            {
                "metric_type": "trading/failure",
                "value": 2.0,
                "labels": {},
                "timestamp": datetime.utcnow()
            }
        ]
        
        async def mock_write_time_series(metric_type, value, labels, timestamp):
            return "success" in metric_type
        
        with patch.object(cloud_collector.cloud_client, 'write_time_series', side_effect=mock_write_time_series):
            await cloud_collector._flush_metrics()
        
        assert len(cloud_collector._metrics_buffer) == 0
    
    @pytest.mark.asyncio
    async def test_auto_flush_on_buffer_size(self, cloud_collector):
        """Test automatic flushing when buffer reaches size limit."""
        with patch.object(cloud_collector, '_flush_metrics', AsyncMock()) as mock_flush:
            # Fill buffer to trigger flush
            for i in range(100):
                await cloud_collector.record_trading_metric(f"metric_{i}", i, {})
            
            # The 100th metric should trigger a flush
            mock_flush.assert_called()
    
    @pytest.mark.asyncio
    async def test_auto_flush_on_time_interval(self, cloud_collector):
        """Test automatic flushing based on time interval."""
        # Set last flush time to trigger time-based flush
        cloud_collector._last_flush_time = 0
        cloud_collector._flush_interval = 1  # 1 second for testing
        
        with patch.object(cloud_collector, '_flush_metrics', AsyncMock()) as mock_flush:
            await cloud_collector.record_trading_metric("test", 1.0, {})
            mock_flush.assert_called()
    
    @pytest.mark.asyncio
    async def test_health_check(self, cloud_collector):
        """Test health check functionality."""
        with patch.object(cloud_collector.cloud_client, 'list_alert_policies', AsyncMock(return_value=[])):
            health_status = await cloud_collector.health_check()
            
            assert health_status["status"] == "healthy"
            assert health_status["cloud_monitoring_connected"] is True
            assert health_status["project_id"] == "test-project-id"
            assert "metrics_buffer_size" in health_status
    
    @pytest.mark.asyncio
    async def test_health_check_connection_failure(self, cloud_collector):
        """Test health check with connection failure."""
        with patch.object(cloud_collector.cloud_client, 'list_alert_policies', AsyncMock(side_effect=Exception("Connection failed"))):
            health_status = await cloud_collector.health_check()
            
            assert health_status["status"] == "unhealthy"
            assert health_status["cloud_monitoring_connected"] is False
            assert len(health_status["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_setup_standard_alert_policies(self, cloud_collector):
        """Test setting up standard alert policies."""
        with patch.object(cloud_collector.cloud_client, 'create_notification_channel', AsyncMock(return_value="channel-123")):
            with patch.object(cloud_collector.cloud_client, 'create_alert_policy', AsyncMock(return_value="policy-456")):
                policies = await cloud_collector.setup_standard_alert_policies()
                
                assert len(policies) > 0
                assert "system_down" in policies or "high_risk" in policies
    
    def test_get_metric_definitions(self, cloud_collector):
        """Test getting metric definitions."""
        definitions = cloud_collector.get_metric_definitions()
        
        assert isinstance(definitions, dict)
        assert "cloud_monitoring_metrics_sent_total" in definitions
        assert "cloud_monitoring_alerts_created_total" in definitions
        assert "cloud_monitoring_errors_total" in definitions


class TestCloudMonitoringContext:
    """Test cloud monitoring context manager."""
    
    @pytest.mark.asyncio
    async def test_context_manager_success(self, metrics_registry):
        """Test successful context manager usage."""
        with patch('src.monitoring.cloud_monitoring.CloudMonitoringCollector') as mock_collector_class:
            mock_collector = Mock()
            mock_collector._flush_metrics = AsyncMock()
            mock_collector_class.return_value = mock_collector
            
            async with cloud_monitoring_context("test-project", metrics_registry) as collector:
                assert collector == mock_collector
            
            # Verify cleanup was called
            mock_collector._flush_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_exception(self, metrics_registry):
        """Test context manager with exception handling."""
        with patch('src.monitoring.cloud_monitoring.CloudMonitoringCollector') as mock_collector_class:
            mock_collector = Mock()
            mock_collector._flush_metrics = AsyncMock()
            mock_collector_class.return_value = mock_collector
            
            with pytest.raises(ValueError):
                async with cloud_monitoring_context("test-project", metrics_registry):
                    raise ValueError("Test exception")
            
            # Verify cleanup was still called
            mock_collector._flush_metrics.assert_called_once()


class TestIntegration:
    """Integration tests for cloud monitoring."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_metric_flow(self, mock_gcp_clients):
        """Test complete metric flow from recording to GCP."""
        # Setup
        registry = MetricsRegistry()
        client = CloudMonitoringClient("test-project")
        
        # Mock successful operations
        mock_gcp_clients['metric_client'].create_time_series.return_value = None
        
        # Test metric descriptor creation
        descriptor = CloudMetricDescriptor(
            name="integration_test",
            display_name="Integration Test Metric",
            description="Test metric for integration",
            metric_kind="GAUGE",
            value_type="DOUBLE"
        )
        
        mock_descriptor = Mock()
        mock_descriptor.type = "custom.googleapis.com/shyvr_rlte/integration_test"
        mock_gcp_clients['metric_client'].create_metric_descriptor.return_value = mock_descriptor
        
        descriptor_result = await client.create_metric_descriptor(descriptor)
        assert descriptor_result is True
        
        # Test time series writing
        write_result = await client.write_time_series(
            metric_type="integration_test",
            value=123.45,
            labels={"test": "integration"}
        )
        assert write_result is True
        
        # Verify calls were made
        mock_gcp_clients['metric_client'].create_metric_descriptor.assert_called_once()
        mock_gcp_clients['metric_client'].create_time_series.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_alert_policy_creation_flow(self, mock_gcp_clients):
        """Test complete alert policy creation flow."""
        client = CloudMonitoringClient("test-project")
        
        # Mock notification channel creation
        mock_channel = Mock()
        mock_channel.name = "projects/test-project/notificationChannels/123"
        mock_gcp_clients['notification_client'].create_notification_channel.return_value = mock_channel
        
        # Mock alert policy creation
        mock_policy = Mock()
        mock_policy.name = "projects/test-project/alertPolicies/456"
        mock_gcp_clients['alert_client'].create_alert_policy.return_value = mock_policy
        
        # Create notification channel
        channel = NotificationChannel(
            type="email",
            display_name="Test Alerts",
            labels={"email_address": "test@example.com"}
        )
        
        channel_result = await client.create_notification_channel(channel)
        assert channel_result == "projects/test-project/notificationChannels/123"
        
        # Create alert policy
        condition = AlertPolicyCondition(
            filter='metric.type="custom.googleapis.com/shyvr_rlte/test"',
            comparison="COMPARISON_GT",
            threshold_value=0.8
        )
        
        policy_result = await client.create_alert_policy(
            display_name="Integration Test Alert",
            conditions=[condition],
            notification_channels=[channel_result]
        )
        
        assert policy_result == "projects/test-project/alertPolicies/456"
        
        # Verify calls
        mock_gcp_clients['notification_client'].create_notification_channel.assert_called_once()
        mock_gcp_clients['alert_client'].create_alert_policy.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])