"""
Tests for the alerting service functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
from typing import Dict, Any

from src.monitoring.alerting import AlertingService, AlertLevel, Alert


class TestAlertLevel:
    """Tests for the AlertLevel enum."""
    
    def test_alert_levels_have_correct_values(self):
        """Test that alert levels have the expected values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"
    
    def test_alert_levels_ordering(self):
        """Test that alert levels can be compared."""
        assert AlertLevel.INFO < AlertLevel.WARNING
        assert AlertLevel.WARNING < AlertLevel.ERROR
        assert AlertLevel.ERROR < AlertLevel.CRITICAL


class TestAlert:
    """Tests for the Alert class."""
    
    def test_alert_creation(self):
        """Test creating an Alert instance."""
        alert = Alert(
            level=AlertLevel.ERROR,
            message="Test error message",
            source="test_source",
            metadata={"test_key": "test_value"}
        )
        
        assert alert.level == AlertLevel.ERROR
        assert alert.message == "Test error message"
        assert alert.source == "test_source"
        assert alert.metadata == {"test_key": "test_value"}
        assert alert.timestamp is not None
    
    def test_alert_to_dict(self):
        """Test converting Alert to dictionary."""
        alert = Alert(
            level=AlertLevel.WARNING,
            message="Warning message",
            source="test"
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["level"] == "warning"
        assert alert_dict["message"] == "Warning message"
        assert alert_dict["source"] == "test"
        assert "timestamp" in alert_dict
        assert "metadata" in alert_dict
    
    def test_alert_with_empty_metadata(self):
        """Test creating Alert with no metadata."""
        alert = Alert(
            level=AlertLevel.INFO,
            message="Info message",
            source="test"
        )
        
        assert alert.metadata == {}
    
    def test_alert_string_representation(self):
        """Test string representation of Alert."""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            message="Critical alert",
            source="trading"
        )
        
        alert_str = str(alert)
        assert "CRITICAL" in alert_str
        assert "Critical alert" in alert_str
        assert "trading" in alert_str


class TestAlertingService:
    """Tests for the AlertingService class."""
    
    @pytest.fixture
    def mock_telegram_config(self):
        """Mock Telegram configuration."""
        return {
            "bot_token": "test_token",
            "chat_id": "test_chat_id",
            "enabled": True
        }
    
    def test_init_creates_service(self, mock_telegram_config):
        """Test that AlertingService initializes correctly."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        assert service.telegram_config == mock_telegram_config
        assert service.telegram_enabled is True
        assert service.alert_queue is not None
        assert service._telegram_bot is None  # Not initialized until needed
    
    def test_init_with_disabled_telegram(self):
        """Test initialization with disabled Telegram."""
        config = {"enabled": False}
        service = AlertingService(telegram_config=config)
        
        assert service.telegram_enabled is False
    
    @pytest.mark.asyncio
    async def test_send_alert_adds_to_queue(self, mock_telegram_config):
        """Test that send_alert adds alert to queue."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        alert = Alert(
            level=AlertLevel.ERROR,
            message="Test alert",
            source="test"
        )
        
        # Mock the queue processing
        with patch.object(service, '_process_alert_queue') as mock_process:
            await service.send_alert(alert)
            
            # Alert should be in queue
            assert service.alert_queue.qsize() == 1
            
            # Process queue should be called
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_alert_by_parameters(self, mock_telegram_config):
        """Test sending alert by individual parameters."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        with patch.object(service, '_process_alert_queue') as mock_process:
            await service.send_alert(
                AlertLevel.WARNING,
                message="Parameter alert",
                source="param_test",
                metadata={"key": "value"}
            )
            
            assert service.alert_queue.qsize() == 1
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_critical_alert(self, mock_telegram_config):
        """Test sending critical alert."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        with patch.object(service, 'send_alert') as mock_send:
            await service.send_critical_alert("Critical message", "critical_source")
            
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            # First positional argument is the AlertLevel
            assert call_args[0][0] == AlertLevel.CRITICAL
            # Keyword arguments
            assert call_args[1]['message'] == "Critical message"
            assert call_args[1]['source'] == "critical_source"
    
    @pytest.mark.asyncio
    async def test_send_error_alert(self, mock_telegram_config):
        """Test sending error alert."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        with patch.object(service, 'send_alert') as mock_send:
            await service.send_error_alert("Error message", "error_source")
            
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == AlertLevel.ERROR
    
    @pytest.mark.asyncio
    async def test_send_warning_alert(self, mock_telegram_config):
        """Test sending warning alert."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        with patch.object(service, 'send_alert') as mock_send:
            await service.send_warning_alert("Warning message", "warning_source")
            
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == AlertLevel.WARNING
    
    @pytest.mark.asyncio
    async def test_process_alert_queue_sends_telegram(self, mock_telegram_config):
        """Test that processing alert queue sends Telegram messages."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        # Add alert to queue
        alert = Alert(AlertLevel.ERROR, "Test message", "test")
        await service.alert_queue.put(alert)
        
        with patch.object(service, '_send_telegram_alert') as mock_telegram:
            await service._process_alert_queue()
            
            mock_telegram.assert_called_once_with(alert)
    
    @pytest.mark.asyncio
    async def test_process_alert_queue_handles_exceptions(self, mock_telegram_config):
        """Test that alert queue processing handles exceptions gracefully."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        alert = Alert(AlertLevel.ERROR, "Test message", "test")
        await service.alert_queue.put(alert)
        
        with patch.object(service, '_send_telegram_alert', side_effect=Exception("Test error")):
            # Should not raise exception
            await service._process_alert_queue()
    
    @pytest.mark.asyncio
    async def test_send_telegram_alert_formats_message(self, mock_telegram_config):
        """Test that Telegram alert formatting works correctly."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        alert = Alert(
            level=AlertLevel.CRITICAL,
            message="Emergency stop triggered",
            source="risk_manager",
            metadata={"risk_level": 0.95, "positions_closed": 3}
        )
        
        with patch('telegram.Bot') as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot_class.return_value = mock_bot
            
            await service._send_telegram_alert(alert)
            
            # Check that bot was created and message was sent
            mock_bot_class.assert_called_once_with(token="test_token")
            mock_bot.send_message.assert_called_once()
            
            # Check message content
            call_args = mock_bot.send_message.call_args
            assert call_args[1]['chat_id'] == "test_chat_id"
            message_text = call_args[1]['text']
            assert "🚨 <b>CRITICAL</b>" in message_text
            assert "Emergency stop triggered" in message_text
            assert "risk_manager" in message_text
    
    @pytest.mark.asyncio
    async def test_send_telegram_alert_disabled(self):
        """Test that Telegram alerts are not sent when disabled."""
        config = {"enabled": False}
        service = AlertingService(telegram_config=config)
        
        alert = Alert(AlertLevel.ERROR, "Test message", "test")
        
        with patch('telegram.Bot') as mock_bot_class:
            await service._send_telegram_alert(alert)
            
            # Bot should not be created when disabled
            mock_bot_class.assert_not_called()
    
    def test_format_telegram_message_critical(self, mock_telegram_config):
        """Test formatting of critical Telegram messages."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        alert = Alert(
            level=AlertLevel.CRITICAL,
            message="System failure",
            source="system",
            metadata={"component": "trading_engine"}
        )
        
        message = service._format_telegram_message(alert)
        
        assert "🚨 CRITICAL" in message
        assert "System failure" in message
        assert "system" in message
        assert "component: trading_engine" in message
    
    def test_format_telegram_message_error(self, mock_telegram_config):
        """Test formatting of error Telegram messages."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        alert = Alert(
            level=AlertLevel.ERROR,
            message="API error",
            source="api"
        )
        
        message = service._format_telegram_message(alert)
        
        assert "❌ ERROR" in message
        assert "API error" in message
    
    def test_format_telegram_message_warning(self, mock_telegram_config):
        """Test formatting of warning Telegram messages."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        alert = Alert(
            level=AlertLevel.WARNING,
            message="High risk detected",
            source="risk_monitor"
        )
        
        message = service._format_telegram_message(alert)
        
        assert "⚠️ WARNING" in message
        assert "High risk detected" in message
    
    def test_format_telegram_message_info(self, mock_telegram_config):
        """Test formatting of info Telegram messages."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        alert = Alert(
            level=AlertLevel.INFO,
            message="System started",
            source="startup"
        )
        
        message = service._format_telegram_message(alert)
        
        assert "ℹ️ INFO" in message
        assert "System started" in message
    
    def test_get_alert_emoji(self, mock_telegram_config):
        """Test that correct emojis are returned for alert levels."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        assert service._get_alert_emoji(AlertLevel.CRITICAL) == "🚨"
        assert service._get_alert_emoji(AlertLevel.ERROR) == "❌"
        assert service._get_alert_emoji(AlertLevel.WARNING) == "⚠️"
        assert service._get_alert_emoji(AlertLevel.INFO) == "ℹ️"
    
    @pytest.mark.asyncio
    async def test_start_stops_service(self, mock_telegram_config):
        """Test starting and stopping the alerting service."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        # Test start
        await service.start()
        assert service._running is True
        
        # Test stop
        await service.stop()
        assert service._running is False
    
    def test_is_healthy_checks_service_status(self, mock_telegram_config):
        """Test health check functionality."""
        service = AlertingService(telegram_config=mock_telegram_config)
        
        # Should be healthy by default
        assert service.is_healthy() is True
        
        # Test with large queue (should still be healthy)
        for i in range(10):
            alert = Alert(AlertLevel.INFO, f"Message {i}", "test")
            service.alert_queue.put_nowait(alert)
        
        assert service.is_healthy() is True