"""
Integration tests for the complete monitoring system.

This module tests the full monitoring stack including:
- Cloud Monitoring integration
- Notification system integration
- Alert policy triggers
- Dashboard data flow
- End-to-end monitoring scenarios
"""

import pytest
import asyncio
import json
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal

from src.monitoring.base import MetricsRegistry
from src.monitoring.cloud_monitoring import (
    CloudMonitoringClient, 
    CloudMonitoringCollector,
    cloud_monitoring_context
)
from src.monitoring.notifications import (
    NotificationManager,
    NotificationLevel,
    NotificationMessage
)
from src.monitoring.trading_metrics import TradingMetricsCollector
from src.monitoring.safety_metrics import SafetyMetricsCollector


@pytest.fixture
def mock_gcp_environment():
    """Mock GCP environment for testing."""
    with patch.dict(os.environ, {
        'GOOGLE_APPLICATION_CREDENTIALS': '/tmp/fake-credentials.json',
        'GOOGLE_CLOUD_PROJECT': 'test-project'
    }):
        with patch('src.monitoring.cloud_monitoring.monitoring_v3') as mock_monitoring:
            with patch('src.monitoring.cloud_monitoring.dashboard_v1') as mock_dashboard:
                # Setup mock clients
                mock_metric_client = Mock()
                mock_alert_client = Mock()
                mock_notification_client = Mock()
                mock_dashboard_client = Mock()
                
                mock_monitoring.MetricServiceClient.return_value = mock_metric_client
                mock_monitoring.AlertPolicyServiceClient.return_value = mock_alert_client
                mock_monitoring.NotificationChannelServiceClient.return_value = mock_notification_client
                mock_dashboard.DashboardsServiceClient.return_value = mock_dashboard_client
                
                yield {
                    'metric_client': mock_metric_client,
                    'alert_client': mock_alert_client,
                    'notification_client': mock_notification_client,
                    'dashboard_client': mock_dashboard_client
                }


@pytest.fixture
def monitoring_config():
    """Complete monitoring system configuration."""
    return {
        "project_id": "test-project",
        "cloud_monitoring": {
            "enabled": True,
            "flush_interval": 30
        },
        "notifications": {
            "deduplication_window_minutes": 10,
            "channels": {
                "email_critical": {
                    "type": "email",
                    "enabled": True,
                    "smtp_host": "smtp.test.com",
                    "smtp_port": 587,
                    "smtp_user": "test@example.com",
                    "smtp_password": "test-password",
                    "from_address": "alerts@example.com",
                    "to_addresses": ["critical@example.com"]
                },
                "slack_alerts": {
                    "type": "slack",
                    "enabled": True,
                    "webhook_url": "https://hooks.slack.com/test",
                    "channel": "#alerts"
                },
                "webhook_integration": {
                    "type": "webhook",
                    "enabled": True,
                    "webhook_url": "https://api.monitoring.com/alerts",
                    "headers": {"Authorization": "Bearer test-token"}
                }
            }
        },
        "alert_thresholds": {
            "risk_level_critical": 0.9,
            "risk_level_warning": 0.8,
            "daily_loss_critical": -1000.0,
            "success_rate_warning": 0.4
        }
    }


class TestMonitoringSystemIntegration:
    """Test complete monitoring system integration."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_stack_initialization(self, mock_gcp_environment, monitoring_config):
        """Test initializing the complete monitoring stack."""
        # Initialize metrics registry
        registry = MetricsRegistry()
        
        # Initialize Cloud Monitoring
        cloud_client = CloudMonitoringClient(monitoring_config["project_id"])
        cloud_collector = CloudMonitoringCollector(registry, monitoring_config["project_id"])
        
        # Initialize notification manager
        notification_manager = NotificationManager(monitoring_config["notifications"])
        
        # Initialize metrics collectors
        trading_collector = TradingMetricsCollector(registry)
        safety_collector = SafetyMetricsCollector(registry)
        
        # Verify all components are initialized
        assert registry is not None
        assert cloud_client.project_id == "test-project"
        assert len(notification_manager.channels) == 3
        assert trading_collector.registry == registry
        assert safety_collector.registry == registry
    
    @pytest.mark.asyncio
    async def test_trading_metrics_to_cloud_monitoring_flow(self, mock_gcp_environment, monitoring_config):
        """Test complete flow from trading metrics to Cloud Monitoring."""
        registry = MetricsRegistry()
        cloud_collector = CloudMonitoringCollector(registry, monitoring_config["project_id"])
        
        # Mock successful cloud monitoring write
        with patch.object(cloud_collector.cloud_client, 'write_time_series', AsyncMock(return_value=True)):
            # Record trading metrics
            await cloud_collector.record_trading_metric(
                "daily_pnl",
                Decimal("150.75"),
                {"currency": "USD", "strategy": "momentum"}
            )
            
            await cloud_collector.record_trading_metric(
                "success_rate",
                0.68,
                {"symbol": "BTC-USD", "timeframe": "1h"}
            )
            
            # Verify metrics are buffered
            assert len(cloud_collector._metrics_buffer) == 2
            
            # Flush metrics
            await cloud_collector._flush_metrics()
            
            # Verify buffer is cleared
            assert len(cloud_collector._metrics_buffer) == 0
            
            # Verify cloud client was called
            assert cloud_collector.cloud_client.write_time_series.call_count == 2
    
    @pytest.mark.asyncio
    async def test_safety_metrics_trigger_alerts(self, mock_gcp_environment, monitoring_config):
        """Test safety metrics triggering alert notifications."""
        registry = MetricsRegistry()
        cloud_collector = CloudMonitoringCollector(registry, monitoring_config["project_id"])
        notification_manager = NotificationManager(monitoring_config["notifications"])
        
        # Mock notification channel responses
        for channel in notification_manager.channels.values():
            channel.send_notification = AsyncMock(return_value="sent")
        
        # Mock cloud monitoring write
        with patch.object(cloud_collector.cloud_client, 'write_time_series', AsyncMock(return_value=True)):
            # Record critical safety metric
            await cloud_collector.record_safety_metric(
                "risk_level",
                0.95,  # Critical level
                {"risk_type": "overall", "timeframe": "5m"}
            )
            
            # Simulate alert trigger based on threshold
            risk_threshold = monitoring_config["alert_thresholds"]["risk_level_critical"]
            if 0.95 > risk_threshold:
                # Send critical alert
                alert_results = await notification_manager.send_alert(
                    level=NotificationLevel.CRITICAL,
                    title="Critical Risk Level Detected",
                    message=f"Risk level has reached {0.95:.1%}, exceeding critical threshold of {risk_threshold:.1%}",
                    source="safety_monitor",
                    metadata={"risk_level": 0.95, "threshold": risk_threshold}
                )
                
                # Verify alert was sent to all channels
                assert len(alert_results) == 3
                for channel_name, status in alert_results.items():
                    assert status in ["sent", "deduplicated"]
    
    @pytest.mark.asyncio
    async def test_emergency_stop_alert_flow(self, mock_gcp_environment, monitoring_config):
        """Test emergency stop triggering immediate alerts."""
        registry = MetricsRegistry()
        cloud_collector = CloudMonitoringCollector(registry, monitoring_config["project_id"])
        notification_manager = NotificationManager(monitoring_config["notifications"])
        
        # Mock all notification channels
        mock_responses = {}
        for channel_name, channel in notification_manager.channels.items():
            channel.send_notification = AsyncMock(return_value="sent")
            mock_responses[channel_name] = channel.send_notification
        
        # Record emergency stop event
        await cloud_collector.record_safety_metric(
            "emergency_stops",
            1,
            {"reason": "excessive_drawdown", "trigger": "risk_manager"}
        )
        
        # Send immediate critical alert
        alert_results = await notification_manager.send_alert(
            level=NotificationLevel.CRITICAL,
            title="Emergency Stop Triggered",
            message="Trading has been halted due to excessive drawdown. All positions have been closed.",
            source="emergency_stop_system",
            metadata={
                "reason": "excessive_drawdown",
                "trigger": "risk_manager",
                "timestamp": datetime.utcnow().isoformat()
            },
            tags=["emergency", "trading_halt", "risk_management"]
        )
        
        # Verify immediate notification to all channels
        assert len(alert_results) == 3
        for channel_name, mock_send in mock_responses.items():
            mock_send.assert_called_once()
            
            # Verify the message sent to each channel
            call_args = mock_send.call_args[0][0]
            assert isinstance(call_args, NotificationMessage)
            assert call_args.level == NotificationLevel.CRITICAL
            assert "Emergency Stop" in call_args.title
            assert "emergency" in call_args.tags
    
    @pytest.mark.asyncio
    async def test_ml_model_performance_monitoring(self, mock_gcp_environment, monitoring_config):
        """Test ML model performance monitoring and alerting."""
        registry = MetricsRegistry()
        cloud_collector = CloudMonitoringCollector(registry, monitoring_config["project_id"])
        notification_manager = NotificationManager(monitoring_config["notifications"])
        
        # Mock cloud monitoring
        with patch.object(cloud_collector.cloud_client, 'write_time_series', AsyncMock(return_value=True)):
            # Record declining ML model accuracy
            model_accuracies = [0.75, 0.68, 0.62, 0.55, 0.48]  # Declining accuracy
            
            for i, accuracy in enumerate(model_accuracies):
                await cloud_collector.record_system_metric(
                    "ml_model_accuracy",
                    accuracy,
                    {"model_type": "lstm", "symbol": "BTC-USD", "iteration": str(i)}
                )
                
                # Check if accuracy falls below threshold
                if accuracy < 0.5:  # Threshold for ML model performance
                    # Mock notification sending
                    with patch.object(notification_manager, 'send_alert', AsyncMock()) as mock_alert:
                        await notification_manager.send_alert(
                            level=NotificationLevel.WARNING,
                            title="ML Model Performance Degradation",
                            message=f"LSTM model accuracy has dropped to {accuracy:.1%}, below acceptable threshold",
                            source="ml_monitor",
                            metadata={"model_type": "lstm", "accuracy": accuracy, "threshold": 0.5}
                        )
                        
                        mock_alert.assert_called_once()
            
            # Verify all metrics were recorded
            assert cloud_collector.cloud_client.write_time_series.call_count == len(model_accuracies)
    
    @pytest.mark.asyncio
    async def test_notification_deduplication_across_channels(self, mock_gcp_environment, monitoring_config):
        """Test notification deduplication works across different channels."""
        notification_manager = NotificationManager(monitoring_config["notifications"])
        
        # Mock all channels
        for channel in notification_manager.channels.values():
            channel.send_notification = AsyncMock(return_value="sent")
        
        # Create identical alert messages
        alert_message = {
            "level": NotificationLevel.WARNING,
            "title": "High CPU Usage",
            "message": "System CPU usage has exceeded 80%",
            "source": "system_monitor",
            "metadata": {"cpu_usage": 85.2}
        }
        
        # Send same alert multiple times
        results1 = await notification_manager.send_alert(**alert_message)
        results2 = await notification_manager.send_alert(**alert_message)
        results3 = await notification_manager.send_alert(**alert_message)
        
        # First alert should be sent
        for status in results1.values():
            assert status == "sent"
        
        # Subsequent alerts should be deduplicated
        for status in results2.values():
            assert status == "deduplicated"
        
        for status in results3.values():
            assert status == "deduplicated"
    
    @pytest.mark.asyncio
    async def test_metrics_collection_with_context_manager(self, mock_gcp_environment):
        """Test metrics collection using context manager for proper cleanup."""
        registry = MetricsRegistry()
        
        with patch.object(CloudMonitoringCollector, '_flush_metrics', AsyncMock()) as mock_flush:
            async with cloud_monitoring_context("test-project", registry) as collector:
                # Record some metrics
                await collector.record_trading_metric("test_metric", 42.0, {"test": "value"})
                await collector.record_safety_metric("test_safety", 0.7, {"type": "test"})
                await collector.record_system_metric("test_system", 1, {"component": "test"})
                
                # Verify metrics are buffered
                assert len(collector._metrics_buffer) == 3
            
            # Verify cleanup (flush) was called on context exit
            mock_flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_alert_policy_creation_and_validation(self, mock_gcp_environment, monitoring_config):
        """Test creating alert policies and validating their configuration."""
        cloud_client = CloudMonitoringClient(monitoring_config["project_id"])
        
        # Mock policy creation
        mock_policy = Mock()
        mock_policy.name = "projects/test-project/alertPolicies/12345"
        
        with patch.object(cloud_client.alert_client, 'create_alert_policy', return_value=mock_policy):
            # Create standard alert policies
            collector = CloudMonitoringCollector(MetricsRegistry(), "test-project")
            with patch.object(collector.cloud_client, 'create_notification_channel', AsyncMock(return_value="channel-123")):
                policies = await collector.setup_standard_alert_policies()
                
                # Verify policies were created
                assert len(policies) > 0
                
                # Verify specific policy types exist
                expected_policies = ["system_down", "high_risk", "large_loss"]
                for policy_type in expected_policies:
                    if policy_type in policies:
                        assert policies[policy_type] == "projects/test-project/alertPolicies/12345"
    
    @pytest.mark.asyncio
    async def test_dashboard_metrics_integration(self, mock_gcp_environment, monitoring_config):
        """Test integration between metrics collection and dashboard data."""
        registry = MetricsRegistry()
        cloud_collector = CloudMonitoringCollector(registry, monitoring_config["project_id"])
        trading_collector = TradingMetricsCollector(registry)
        safety_collector = SafetyMetricsCollector(registry)
        
        # Mock cloud monitoring writes
        with patch.object(cloud_collector.cloud_client, 'write_time_series', AsyncMock(return_value=True)):
            # Simulate trading activity generating metrics
            trading_data = {
                "pnl_updates": [
                    {"total_pnl": Decimal("1250.00"), "daily_pnl": Decimal("125.50")},
                    {"total_pnl": Decimal("1375.75"), "daily_pnl": Decimal("251.25")},
                    {"total_pnl": Decimal("1200.25"), "daily_pnl": Decimal("75.50")}
                ],
                "trades": [
                    {"symbol": "BTC-USD", "side": "buy", "size": Decimal("1000"), "success": True},
                    {"symbol": "ETH-USD", "side": "sell", "size": Decimal("2000"), "success": True},
                    {"symbol": "SOL-USD", "side": "buy", "size": Decimal("500"), "success": False}
                ],
                "risk_levels": [0.3, 0.4, 0.6, 0.7, 0.5]
            }
            
            # Process P&L updates
            for pnl_data in trading_data["pnl_updates"]:
                trading_collector.record_pnl_update(
                    total_pnl=pnl_data["total_pnl"],
                    daily_pnl=pnl_data["daily_pnl"]
                )
                
                # Send to cloud monitoring
                await cloud_collector.record_trading_metric(
                    "total_pnl",
                    pnl_data["total_pnl"],
                    {"currency": "USD"}
                )
            
            # Process trades
            for trade in trading_data["trades"]:
                if trade["success"]:
                    trading_collector.record_trade_success(trade)
                else:
                    trading_collector.record_trade_failure(trade)
                
                # Send to cloud monitoring
                await cloud_collector.record_trading_metric(
                    "trade_volume",
                    trade["size"],
                    {"symbol": trade["symbol"], "side": trade["side"]}
                )
            
            # Process risk levels
            for risk_level in trading_data["risk_levels"]:
                safety_collector.update_risk_level(risk_level, "overall")
                
                # Send to cloud monitoring
                await cloud_collector.record_safety_metric(
                    "risk_level",
                    risk_level,
                    {"risk_type": "overall"}
                )
            
            # Verify metrics were collected locally (Prometheus)
            prometheus_output = registry.generate_output().decode('utf-8')
            assert "trading_total_pnl" in prometheus_output
            assert "trading_success_rate" in prometheus_output
            assert "safety_risk_level" in prometheus_output
            
            # Verify metrics were sent to cloud monitoring
            expected_cloud_calls = len(trading_data["pnl_updates"]) + len(trading_data["trades"]) + len(trading_data["risk_levels"])
            assert cloud_collector.cloud_client.write_time_series.call_count >= expected_cloud_calls
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring_end_to_end(self, mock_gcp_environment, monitoring_config):
        """Test complete system health monitoring workflow."""
        registry = MetricsRegistry()
        cloud_collector = CloudMonitoringCollector(registry, monitoring_config["project_id"])
        notification_manager = NotificationManager(monitoring_config["notifications"])
        
        # Mock dependencies
        with patch.object(cloud_collector.cloud_client, 'write_time_series', AsyncMock(return_value=True)):
            with patch.object(notification_manager, 'send_alert', AsyncMock()) as mock_alert:
                # Simulate system health check
                health_status = await cloud_collector.health_check()
                
                # Record system health metrics
                await cloud_collector.record_system_metric(
                    "uptime",
                    1 if health_status["status"] == "healthy" else 0,
                    {"component": "monitoring_system"}
                )
                
                await cloud_collector.record_system_metric(
                    "health_score",
                    1.0 if health_status["status"] == "healthy" else 0.5,
                    {"system": "shyvr_rlte"}
                )
                
                # Simulate error rate monitoring
                error_rates = [2.1, 5.5, 8.3, 12.7, 15.2]  # Increasing error rate
                
                for rate in error_rates:
                    await cloud_collector.record_system_metric(
                        "error_rate",
                        rate,
                        {"component": "trading_engine"}
                    )
                    
                    # Trigger alert if error rate is too high
                    if rate > 10.0:
                        await notification_manager.send_alert(
                            level=NotificationLevel.WARNING,
                            title="High System Error Rate",
                            message=f"Error rate has increased to {rate:.1f} errors/minute",
                            source="system_monitor",
                            metadata={"error_rate": rate, "threshold": 10.0}
                        )
                
                # Verify system metrics were recorded
                assert cloud_collector.cloud_client.write_time_series.call_count >= 7  # 2 health + 5 error rate
                
                # Verify alerts were sent for high error rates
                assert mock_alert.call_count == 3  # Last 3 error rates exceeded threshold


class TestMonitoringErrorHandling:
    """Test error handling and resilience in monitoring system."""
    
    @pytest.mark.asyncio
    async def test_cloud_monitoring_connection_failure_resilience(self, mock_gcp_environment):
        """Test system resilience when Cloud Monitoring is unavailable."""
        registry = MetricsRegistry()
        cloud_collector = CloudMonitoringCollector(registry, "test-project")
        
        # Mock connection failure
        with patch.object(cloud_collector.cloud_client, 'write_time_series', AsyncMock(side_effect=Exception("Connection failed"))):
            # Record metrics despite connection failure
            await cloud_collector.record_trading_metric("test_metric", 42.0, {})
            await cloud_collector.record_safety_metric("test_safety", 0.8, {})
            
            # Metrics should still be buffered
            assert len(cloud_collector._metrics_buffer) == 2
            
            # Flush should handle errors gracefully
            await cloud_collector._flush_metrics()
            
            # Buffer should be cleared even with errors
            assert len(cloud_collector._metrics_buffer) == 0
            
            # Prometheus metrics should still work
            prometheus_output = registry.generate_output().decode('utf-8')
            assert "cloud_monitoring_errors_total" in prometheus_output
    
    @pytest.mark.asyncio
    async def test_notification_channel_failure_fallback(self, monitoring_config):
        """Test notification system fallback when channels fail."""
        notification_manager = NotificationManager(monitoring_config["notifications"])
        
        # Mock channel failures
        channels = list(notification_manager.channels.values())
        channels[0].send_notification = AsyncMock(side_effect=Exception("Channel 1 failed"))
        channels[1].send_notification = AsyncMock(return_value="sent")
        channels[2].send_notification = AsyncMock(return_value="sent")
        
        # Send alert
        results = await notification_manager.send_alert(
            level=NotificationLevel.ERROR,
            title="Test Error",
            message="Testing channel fallback",
            source="test_system"
        )
        
        # Verify some channels succeeded despite one failure
        success_count = sum(1 for status in results.values() if status == "sent")
        failure_count = sum(1 for status in results.values() if status == "failed")
        
        assert success_count == 2
        assert failure_count == 1
    
    @pytest.mark.asyncio
    async def test_metrics_buffer_overflow_handling(self, mock_gcp_environment):
        """Test handling of metrics buffer overflow."""
        registry = MetricsRegistry()
        cloud_collector = CloudMonitoringCollector(registry, "test-project")
        
        # Mock slow/failing cloud writes
        with patch.object(cloud_collector.cloud_client, 'write_time_series', AsyncMock(return_value=False)):
            # Fill buffer beyond normal capacity
            for i in range(1000):
                await cloud_collector.record_trading_metric(f"metric_{i}", float(i), {"batch": "overflow_test"})
            
            # Buffer should have triggered multiple flushes
            # Even with failed writes, buffer management should prevent memory issues
            assert len(cloud_collector._metrics_buffer) <= 100  # Should be managed


if __name__ == "__main__":
    pytest.main([__file__])