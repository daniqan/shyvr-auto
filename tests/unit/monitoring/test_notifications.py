"""
Tests for the notification system.

This module tests notification channels including:
- Slack notifications with rich formatting
- Webhook notifications
- Email notifications
- Notification manager and deduplication
"""

import pytest
import asyncio
import json
import time
import hashlib
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

import httpx

from src.monitoring.notifications import (
    NotificationLevel,
    NotificationStatus,
    NotificationMessage,
    NotificationChannel,
    SlackNotificationChannel,
    WebhookNotificationChannel,
    EmailNotificationChannel,
    NotificationManager
)


@pytest.fixture
def sample_notification_message():
    """Create a sample notification message for testing."""
    return NotificationMessage(
        id="test-123",
        level=NotificationLevel.WARNING,
        title="Test Alert",
        message="This is a test alert message",
        source="test_system",
        timestamp=datetime.utcnow(),
        metadata={"test_key": "test_value", "severity": "medium"},
        tags=["test", "monitoring"]
    )


@pytest.fixture
def slack_config():
    """Slack channel configuration for testing."""
    return {
        "type": "slack",
        "enabled": True,
        "webhook_url": "https://hooks.slack.com/services/TEST/WEBHOOK/URL",
        "channel": "#test-alerts",
        "username": "Test Bot",
        "icon_emoji": ":test:",
        "rate_limit": {
            "critical": {"max_per_hour": 10},
            "warning": {"max_per_hour": 20}
        }
    }


@pytest.fixture
def webhook_config():
    """Webhook channel configuration for testing."""
    return {
        "type": "webhook",
        "enabled": True,
        "webhook_url": "https://api.example.com/alerts",
        "headers": {"Content-Type": "application/json", "X-API-Key": "test-key"},
        "auth": {"type": "bearer", "token": "test-token"},
        "format_template": "json"
    }


@pytest.fixture
def email_config():
    """Email channel configuration for testing."""
    return {
        "type": "email",
        "enabled": True,
        "smtp_host": "smtp.test.com",
        "smtp_port": 587,
        "smtp_user": "test@example.com",
        "smtp_password": "test-password",
        "from_address": "alerts@example.com",
        "to_addresses": ["recipient1@example.com", "recipient2@example.com"],
        "use_tls": True
    }


class TestNotificationMessage:
    """Test NotificationMessage class."""
    
    def test_message_creation(self):
        """Test creating a notification message."""
        timestamp = datetime.utcnow()
        message = NotificationMessage(
            id="msg-123",
            level=NotificationLevel.ERROR,
            title="Error Alert",
            message="Something went wrong",
            source="error_handler",
            timestamp=timestamp,
            metadata={"error_code": 500},
            tags=["error", "critical"]
        )
        
        assert message.id == "msg-123"
        assert message.level == NotificationLevel.ERROR
        assert message.title == "Error Alert"
        assert message.message == "Something went wrong"
        assert message.source == "error_handler"
        assert message.timestamp == timestamp
        assert message.metadata["error_code"] == 500
        assert "error" in message.tags
    
    def test_message_defaults(self):
        """Test message creation with default values."""
        message = NotificationMessage(
            id="simple-msg",
            level=NotificationLevel.INFO,
            title="Simple Message",
            message="Simple content",
            source="test",
            timestamp=datetime.utcnow()
        )
        
        assert message.metadata == {}
        assert message.tags == []
    
    def test_to_dict(self):
        """Test converting message to dictionary."""
        timestamp = datetime.utcnow()
        message = NotificationMessage(
            id="dict-test",
            level=NotificationLevel.CRITICAL,
            title="Dict Test",
            message="Test message",
            source="dict_test",
            timestamp=timestamp,
            metadata={"key": "value"},
            tags=["test"]
        )
        
        message_dict = message.to_dict()
        
        assert message_dict["id"] == "dict-test"
        assert message_dict["level"] == "critical"
        assert message_dict["title"] == "Dict Test"
        assert message_dict["timestamp"] == timestamp.isoformat()
        assert message_dict["metadata"]["key"] == "value"
        assert message_dict["tags"] == ["test"]
    
    def test_get_hash(self):
        """Test message hash generation for deduplication."""
        message1 = NotificationMessage(
            id="hash-1",
            level=NotificationLevel.INFO,
            title="Same Title",
            message="Same Message",
            source="same_source",
            timestamp=datetime.utcnow()
        )
        
        message2 = NotificationMessage(
            id="hash-2",  # Different ID
            level=NotificationLevel.WARNING,  # Different level
            title="Same Title",
            message="Same Message",
            source="same_source",
            timestamp=datetime.utcnow() + timedelta(minutes=5)  # Different timestamp
        )
        
        # Hash should be the same as it's based on title, message, and source
        assert message1.get_hash() == message2.get_hash()
        
        # Different message should have different hash
        message3 = NotificationMessage(
            id="hash-3",
            level=NotificationLevel.INFO,
            title="Different Title",
            message="Same Message",
            source="same_source",
            timestamp=datetime.utcnow()
        )
        
        assert message1.get_hash() != message3.get_hash()


class TestSlackNotificationChannel:
    """Test Slack notification channel."""
    
    def test_slack_channel_initialization(self, slack_config):
        """Test Slack channel initialization."""
        channel = SlackNotificationChannel("test-slack", slack_config)
        
        assert channel.name == "test-slack"
        assert channel.enabled is True
        assert channel.webhook_url == slack_config["webhook_url"]
        assert channel.channel == "#test-alerts"
        assert channel.username == "Test Bot"
        assert channel.icon_emoji == ":test:"
    
    def test_slack_channel_missing_webhook_url(self):
        """Test Slack channel initialization without webhook URL."""
        config = {"type": "slack", "enabled": True}
        
        with pytest.raises(ValueError, match="Slack webhook URL is required"):
            SlackNotificationChannel("test-slack", config)
    
    @pytest.mark.asyncio
    async def test_send_notification_success(self, slack_config, sample_notification_message):
        """Test successful Slack notification sending."""
        channel = SlackNotificationChannel("test-slack", slack_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            status = await channel.send_notification(sample_notification_message)
            
            assert status == NotificationStatus.SENT
            mock_client.post.assert_called_once()
            
            # Verify the call was made with correct parameters
            call_args = mock_client.post.call_args
            assert call_args[0][0] == slack_config["webhook_url"]
            assert "json" in call_args[1]
    
    @pytest.mark.asyncio
    async def test_send_notification_disabled_channel(self, slack_config, sample_notification_message):
        """Test sending notification to disabled channel."""
        slack_config["enabled"] = False
        channel = SlackNotificationChannel("test-slack", slack_config)
        
        status = await channel.send_notification(sample_notification_message)
        
        assert status == NotificationStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_send_notification_rate_limited(self, slack_config, sample_notification_message):
        """Test rate limiting functionality."""
        channel = SlackNotificationChannel("test-slack", slack_config)
        
        # Simulate multiple recent messages to trigger rate limiting
        for _ in range(25):  # Exceed the warning limit of 20
            channel.record_sent(sample_notification_message)
        
        status = await channel.send_notification(sample_notification_message)
        
        assert status == NotificationStatus.RATE_LIMITED
    
    @pytest.mark.asyncio
    async def test_send_notification_http_error(self, slack_config, sample_notification_message):
        """Test handling of HTTP errors."""
        channel = SlackNotificationChannel("test-slack", slack_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request"
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            status = await channel.send_notification(sample_notification_message)
            
            assert status == NotificationStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_send_notification_network_error(self, slack_config, sample_notification_message):
        """Test handling of network errors."""
        channel = SlackNotificationChannel("test-slack", slack_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.RequestError("Network error")
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            status = await channel.send_notification(sample_notification_message)
            
            assert status == NotificationStatus.FAILED
    
    def test_format_slack_message(self, slack_config, sample_notification_message):
        """Test Slack message formatting."""
        channel = SlackNotificationChannel("test-slack", slack_config)
        slack_message = channel._format_slack_message(sample_notification_message)
        
        assert slack_message["channel"] == "#test-alerts"
        assert slack_message["username"] == "Test Bot"
        assert slack_message["icon_emoji"] == ":test:"
        assert len(slack_message["attachments"]) == 1
        
        attachment = slack_message["attachments"][0]
        assert attachment["color"] == "#ff9900"  # Warning color
        assert "Test Alert" in attachment["title"]
        assert attachment["text"] == "This is a test alert message"
        assert len(attachment["fields"]) >= 2  # Source and Time fields
    
    def test_format_slack_message_critical_level(self, slack_config):
        """Test Slack message formatting for critical level."""
        critical_message = NotificationMessage(
            id="critical-test",
            level=NotificationLevel.CRITICAL,
            title="Critical Alert",
            message="System is down",
            source="system_monitor",
            timestamp=datetime.utcnow()
        )
        
        channel = SlackNotificationChannel("test-slack", slack_config)
        slack_message = channel._format_slack_message(critical_message)
        
        attachment = slack_message["attachments"][0]
        assert attachment["color"] == "#8B0000"  # Critical color (dark red)
        assert ":rotating_light:" in attachment["title"]


class TestWebhookNotificationChannel:
    """Test webhook notification channel."""
    
    def test_webhook_channel_initialization(self, webhook_config):
        """Test webhook channel initialization."""
        channel = WebhookNotificationChannel("test-webhook", webhook_config)
        
        assert channel.name == "test-webhook"
        assert channel.enabled is True
        assert channel.webhook_url == webhook_config["webhook_url"]
        assert channel.headers == webhook_config["headers"]
        assert channel.auth == webhook_config["auth"]
    
    def test_webhook_channel_missing_url(self):
        """Test webhook channel initialization without URL."""
        config = {"type": "webhook", "enabled": True}
        
        with pytest.raises(ValueError, match="Webhook URL is required"):
            WebhookNotificationChannel("test-webhook", config)
    
    @pytest.mark.asyncio
    async def test_send_notification_json_format(self, webhook_config, sample_notification_message):
        """Test sending webhook notification with JSON format."""
        channel = WebhookNotificationChannel("test-webhook", webhook_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            status = await channel.send_notification(sample_notification_message)
            
            assert status == NotificationStatus.SENT
            mock_client.post.assert_called_once()
            
            # Verify JSON payload was sent
            call_args = mock_client.post.call_args
            assert "json" in call_args[1]
            assert "headers" in call_args[1]
            
            # Check authorization header was added
            headers = call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-token"
    
    @pytest.mark.asyncio
    async def test_send_notification_form_format(self, webhook_config, sample_notification_message):
        """Test sending webhook notification with form format."""
        webhook_config["format_template"] = "form"
        channel = WebhookNotificationChannel("test-webhook", webhook_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 201
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            status = await channel.send_notification(sample_notification_message)
            
            assert status == NotificationStatus.SENT
            
            # Verify form data was sent
            call_args = mock_client.post.call_args
            assert "data" in call_args[1]
            assert "json" not in call_args[1]
    
    def test_format_webhook_payload_json(self, webhook_config, sample_notification_message):
        """Test webhook payload formatting for JSON."""
        channel = WebhookNotificationChannel("test-webhook", webhook_config)
        payload = channel._format_webhook_payload(sample_notification_message)
        
        assert isinstance(payload, dict)
        assert "notification" in payload
        assert "alert" in payload
        assert payload["alert"]["level"] == "warning"
        assert payload["alert"]["title"] == "Test Alert"
        assert payload["alert"]["source"] == "test_system"
    
    def test_format_webhook_payload_form(self, webhook_config, sample_notification_message):
        """Test webhook payload formatting for form data."""
        webhook_config["format_template"] = "form"
        channel = WebhookNotificationChannel("test-webhook", webhook_config)
        payload = channel._format_webhook_payload(sample_notification_message)
        
        assert isinstance(payload, dict)
        assert payload["level"] == "warning"
        assert payload["title"] == "Test Alert"
        assert payload["source"] == "test_system"
        assert "metadata" in payload
        assert "tags" in payload


class TestEmailNotificationChannel:
    """Test email notification channel."""
    
    def test_email_channel_initialization(self, email_config):
        """Test email channel initialization."""
        channel = EmailNotificationChannel("test-email", email_config)
        
        assert channel.name == "test-email"
        assert channel.enabled is True
        assert channel.smtp_host == "smtp.test.com"
        assert channel.smtp_port == 587
        assert channel.smtp_user == "test@example.com"
        assert channel.from_address == "alerts@example.com"
        assert len(channel.to_addresses) == 2
    
    def test_email_channel_missing_credentials(self):
        """Test email channel initialization with missing credentials."""
        config = {
            "type": "email",
            "enabled": True,
            "smtp_host": "smtp.test.com"
        }
        
        with pytest.raises(ValueError, match="SMTP credentials and from_address are required"):
            EmailNotificationChannel("test-email", config)
    
    def test_email_channel_missing_recipients(self):
        """Test email channel initialization with missing recipients."""
        config = {
            "type": "email",
            "enabled": True,
            "smtp_host": "smtp.test.com",
            "smtp_user": "test@example.com",
            "smtp_password": "password",
            "from_address": "sender@example.com",
            "to_addresses": []
        }
        
        with pytest.raises(ValueError, match="At least one recipient email address is required"):
            EmailNotificationChannel("test-email", config)
    
    @pytest.mark.asyncio
    async def test_send_notification_success(self, email_config, sample_notification_message):
        """Test successful email sending."""
        channel = EmailNotificationChannel("test-email", email_config)
        
        with patch.object(channel, '_send_smtp_email') as mock_send:
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_executor = AsyncMock()
                mock_loop.return_value.run_in_executor = mock_executor
                
                status = await channel.send_notification(sample_notification_message)
                
                assert status == NotificationStatus.SENT
                mock_executor.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_notification_smtp_error(self, email_config, sample_notification_message):
        """Test handling of SMTP errors."""
        channel = EmailNotificationChannel("test-email", email_config)
        
        with patch.object(channel, '_send_smtp_email', side_effect=Exception("SMTP error")):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_executor = AsyncMock(side_effect=Exception("SMTP error"))
                mock_loop.return_value.run_in_executor = mock_executor
                
                status = await channel.send_notification(sample_notification_message)
                
                assert status == NotificationStatus.FAILED
    
    def test_format_email_message(self, email_config, sample_notification_message):
        """Test email message formatting."""
        channel = EmailNotificationChannel("test-email", email_config)
        email_message = channel._format_email_message(sample_notification_message)
        
        assert email_message["Subject"] == "[WARNING] Test Alert"
        assert email_message["From"] == "alerts@example.com"
        assert "recipient1@example.com" in email_message["To"]
        assert "recipient2@example.com" in email_message["To"]
        
        # Check that both plain text and HTML parts exist
        parts = email_message.get_payload()
        assert len(parts) == 2
        assert parts[0].get_content_type() == "text/plain"
        assert parts[1].get_content_type() == "text/html"


class TestNotificationManager:
    """Test notification manager."""
    
    def test_manager_initialization(self):
        """Test notification manager initialization."""
        config = {
            "deduplication_window_minutes": 10,
            "channels": {
                "test_slack": {
                    "type": "slack",
                    "enabled": True,
                    "webhook_url": "https://hooks.slack.com/test"
                },
                "test_webhook": {
                    "type": "webhook",
                    "enabled": True,
                    "webhook_url": "https://api.test.com/webhook"
                }
            }
        }
        
        manager = NotificationManager(config)
        
        assert manager.deduplication_window == 10
        assert len(manager.channels) == 2
        assert "test_slack" in manager.channels
        assert "test_webhook" in manager.channels
    
    def test_manager_initialization_invalid_channel(self):
        """Test manager handles invalid channel types gracefully."""
        config = {
            "channels": {
                "invalid_channel": {
                    "type": "invalid_type",
                    "enabled": True
                },
                "valid_channel": {
                    "type": "slack",
                    "enabled": True,
                    "webhook_url": "https://hooks.slack.com/test"
                }
            }
        }
        
        manager = NotificationManager(config)
        
        # Should only have the valid channel
        assert len(manager.channels) == 1
        assert "valid_channel" in manager.channels
        assert "invalid_channel" not in manager.channels
    
    def test_should_deduplicate(self, sample_notification_message):
        """Test deduplication logic."""
        config = {"deduplication_window_minutes": 15, "channels": {}}
        manager = NotificationManager(config)
        
        # First message should not be deduplicated
        assert manager.should_deduplicate(sample_notification_message) is False
        
        # Identical message within window should be deduplicated
        assert manager.should_deduplicate(sample_notification_message) is True
        
        # Different message should not be deduplicated
        different_message = NotificationMessage(
            id="different-123",
            level=NotificationLevel.ERROR,
            title="Different Alert",
            message="Different message",
            source="different_system",
            timestamp=datetime.utcnow()
        )
        
        assert manager.should_deduplicate(different_message) is False
    
    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        """Test successful notification sending."""
        config = {
            "channels": {
                "test_channel": {
                    "type": "slack",
                    "enabled": True,
                    "webhook_url": "https://hooks.slack.com/test"
                }
            }
        }
        
        manager = NotificationManager(config)
        
        # Mock the channel send method
        mock_channel = Mock()
        mock_channel.send_notification = AsyncMock(return_value=NotificationStatus.SENT)
        manager.channels["test_channel"] = mock_channel
        
        message = NotificationMessage(
            id="test-send",
            level=NotificationLevel.INFO,
            title="Test Send",
            message="Testing send",
            source="test",
            timestamp=datetime.utcnow()
        )
        
        results = await manager.send_notification(message)
        
        assert results["test_channel"] == NotificationStatus.SENT
        mock_channel.send_notification.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_send_notification_deduplicated(self, sample_notification_message):
        """Test notification deduplication."""
        config = {"deduplication_window_minutes": 15, "channels": {"test": {}}}
        manager = NotificationManager(config)
        
        # Add a channel
        mock_channel = Mock()
        manager.channels["test"] = mock_channel
        
        # First send
        await manager.send_notification(sample_notification_message)
        
        # Second send should be deduplicated
        results = await manager.send_notification(sample_notification_message)
        
        assert results["test"] == NotificationStatus.DEDUPLICATED
    
    @pytest.mark.asyncio
    async def test_send_alert_convenience_method(self):
        """Test the send_alert convenience method."""
        config = {"channels": {}}
        manager = NotificationManager(config)
        
        with patch.object(manager, 'send_notification', AsyncMock()) as mock_send:
            await manager.send_alert(
                level=NotificationLevel.ERROR,
                title="Test Error",
                message="Error occurred",
                source="error_handler",
                metadata={"code": 500},
                tags=["error"]
            )
            
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]  # First argument (message)
            assert isinstance(call_args, NotificationMessage)
            assert call_args.level == NotificationLevel.ERROR
            assert call_args.title == "Test Error"
            assert call_args.source == "error_handler"
    
    def test_get_channel_status(self):
        """Test getting channel status."""
        config = {
            "channels": {
                "test_channel": {
                    "type": "slack",
                    "enabled": True,
                    "webhook_url": "https://hooks.slack.com/test"
                }
            }
        }
        
        manager = NotificationManager(config)
        status = manager.get_channel_status()
        
        assert "test_channel" in status
        channel_status = status["test_channel"]
        assert channel_status["name"] == "test_channel"
        assert channel_status["enabled"] is True
        assert "recent_sends" in channel_status
    
    @pytest.mark.asyncio
    async def test_test_channels(self):
        """Test the test channels functionality."""
        config = {"channels": {}}
        manager = NotificationManager(config)
        
        # Mock a channel
        mock_channel = Mock()
        mock_channel.send_notification = AsyncMock(return_value=NotificationStatus.SENT)
        manager.channels["test"] = mock_channel
        
        results = await manager.test_channels()
        
        assert results["test"] == NotificationStatus.SENT
        mock_channel.send_notification.assert_called_once()
        
        # Verify the test message properties
        call_args = mock_channel.send_notification.call_args[0][0]
        assert call_args.title == "Shyvr RLTE Monitoring Test"
        assert call_args.level == NotificationLevel.INFO
        assert "test" in call_args.tags


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_configuration(self):
        """Test rate limiting configuration."""
        config = {
            "type": "slack",
            "enabled": True,
            "webhook_url": "https://hooks.slack.com/test",
            "rate_limit": {
                "critical": {"max_per_hour": 5},
                "warning": {"max_per_hour": 10},
                "info": {"max_per_hour": 20}
            }
        }
        
        channel = SlackNotificationChannel("test", config)
        
        # Test rate limiting for different levels
        critical_msg = NotificationMessage(
            id="crit", level=NotificationLevel.CRITICAL, title="Critical",
            message="Critical message", source="test", timestamp=datetime.utcnow()
        )
        
        # Should not be rate limited initially
        assert channel.should_rate_limit(critical_msg) is False
        
        # Record multiple sends to trigger rate limiting
        for _ in range(6):
            channel.record_sent(critical_msg)
        
        # Should now be rate limited
        assert channel.should_rate_limit(critical_msg) is True
    
    def test_rate_limit_time_window(self):
        """Test rate limiting time window behavior."""
        config = {
            "type": "slack",
            "enabled": True,
            "webhook_url": "https://hooks.slack.com/test",
            "rate_limit": {
                "info": {"max_per_hour": 3}
            }
        }
        
        channel = SlackNotificationChannel("test", config)
        
        info_msg = NotificationMessage(
            id="info", level=NotificationLevel.INFO, title="Info",
            message="Info message", source="test", timestamp=datetime.utcnow()
        )
        
        # Record sends with timestamps outside the window
        old_time = datetime.utcnow() - timedelta(hours=2)
        channel.last_sent_times["info_recent"] = [old_time, old_time, old_time, old_time]
        
        # Should not be rate limited as old timestamps are cleaned up
        assert channel.should_rate_limit(info_msg) is False


if __name__ == "__main__":
    pytest.main([__file__])