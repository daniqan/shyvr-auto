"""
Advanced notification system for Shyvr RLTE monitoring.

This module provides comprehensive notification capabilities including:
- Slack integration with rich formatting
- Webhook notifications for custom integrations
- Email notifications with templates
- SMS notifications (optional)
- Notification rate limiting and deduplication
"""

import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

import httpx
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationStatus(Enum):
    """Notification delivery status."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    DEDUPLICATED = "deduplicated"


@dataclass
class NotificationMessage:
    """Represents a notification message."""
    
    id: str
    level: NotificationLevel
    title: str
    message: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        return data
    
    def get_hash(self) -> str:
        """Generate hash for deduplication."""
        content = f"{self.title}:{self.message}:{self.source}"
        return hashlib.md5(content.encode()).hexdigest()


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.rate_limit = config.get('rate_limit', {})
        self.last_sent_times: Dict[str, datetime] = {}
        
    @abstractmethod
    async def send_notification(self, message: NotificationMessage) -> NotificationStatus:
        """Send a notification message."""
        pass
    
    def should_rate_limit(self, message: NotificationMessage) -> bool:
        """Check if message should be rate limited."""
        if not self.rate_limit:
            return False
        
        level_limits = self.rate_limit.get(message.level.value, {})
        if not level_limits:
            return False
        
        max_per_hour = level_limits.get('max_per_hour')
        if not max_per_hour:
            return False
        
        # Check recent message count for this level
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_key = f"{message.level.value}_recent"
        
        if recent_key not in self.last_sent_times:
            self.last_sent_times[recent_key] = []
        
        # Clean old timestamps
        recent_times = [t for t in self.last_sent_times[recent_key] if t > cutoff_time]
        self.last_sent_times[recent_key] = recent_times
        
        return len(recent_times) >= max_per_hour
    
    def record_sent(self, message: NotificationMessage):
        """Record that a message was sent."""
        recent_key = f"{message.level.value}_recent"
        if recent_key not in self.last_sent_times:
            self.last_sent_times[recent_key] = []
        
        self.last_sent_times[recent_key].append(datetime.utcnow())


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel with rich formatting."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'Shyvr RLTE Bot')
        self.icon_emoji = config.get('icon_emoji', ':robot_face:')
        
        if not self.webhook_url:
            raise ValueError("Slack webhook URL is required")
    
    async def send_notification(self, message: NotificationMessage) -> NotificationStatus:
        """Send notification to Slack."""
        if not self.enabled:
            return NotificationStatus.FAILED
        
        if self.should_rate_limit(message):
            logger.warning(f"Rate limiting Slack notification: {message.title}")
            return NotificationStatus.RATE_LIMITED
        
        try:
            slack_message = self._format_slack_message(message)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=slack_message,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    self.record_sent(message)
                    logger.info(f"Sent Slack notification: {message.title}")
                    return NotificationStatus.SENT
                else:
                    logger.error(f"Slack notification failed: {response.status_code} - {response.text}")
                    return NotificationStatus.FAILED
                    
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return NotificationStatus.FAILED
    
    def _format_slack_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Format message for Slack."""
        # Color coding based on level
        color_map = {
            NotificationLevel.INFO: "#36a64f",      # Green
            NotificationLevel.WARNING: "#ff9900",   # Orange
            NotificationLevel.ERROR: "#ff0000",     # Red
            NotificationLevel.CRITICAL: "#8B0000"   # Dark Red
        }
        
        # Emoji mapping
        emoji_map = {
            NotificationLevel.INFO: ":information_source:",
            NotificationLevel.WARNING: ":warning:",
            NotificationLevel.ERROR: ":x:",
            NotificationLevel.CRITICAL: ":rotating_light:"
        }
        
        # Build fields
        fields = [
            {
                "title": "Source",
                "value": message.source,
                "short": True
            },
            {
                "title": "Time",
                "value": message.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "short": True
            }
        ]
        
        # Add metadata fields
        if message.metadata:
            for key, value in message.metadata.items():
                if len(fields) < 8:  # Slack limit
                    fields.append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
        
        # Build attachment
        attachment = {
            "color": color_map.get(message.level, "#cccccc"),
            "title": f"{emoji_map.get(message.level, '')} {message.title}",
            "text": message.message,
            "fields": fields,
            "footer": "Shyvr RLTE Monitoring",
            "footer_icon": "https://example.com/icon.png",  # Replace with actual icon
            "ts": int(message.timestamp.timestamp()),
            "mrkdwn_in": ["text", "pretext"]
        }
        
        # Add tags as a field if present
        if message.tags:
            attachment["fields"].append({
                "title": "Tags",
                "value": ", ".join(message.tags),
                "short": False
            })
        
        return {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [attachment]
        }


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {})
        self.auth = config.get('auth', {})
        self.format_template = config.get('format_template', 'json')
        
        if not self.webhook_url:
            raise ValueError("Webhook URL is required")
    
    async def send_notification(self, message: NotificationMessage) -> NotificationStatus:
        """Send notification via webhook."""
        if not self.enabled:
            return NotificationStatus.FAILED
        
        if self.should_rate_limit(message):
            logger.warning(f"Rate limiting webhook notification: {message.title}")
            return NotificationStatus.RATE_LIMITED
        
        try:
            payload = self._format_webhook_payload(message)
            headers = self.headers.copy()
            
            # Add authentication if configured
            if self.auth.get('type') == 'bearer':
                headers['Authorization'] = f"Bearer {self.auth['token']}"
            elif self.auth.get('type') == 'api_key':
                headers[self.auth['header']] = self.auth['key']
            
            async with httpx.AsyncClient() as client:
                if self.format_template == 'json':
                    headers['Content-Type'] = 'application/json'
                    response = await client.post(
                        self.webhook_url,
                        json=payload,
                        headers=headers,
                        timeout=10.0
                    )
                else:
                    headers['Content-Type'] = 'application/x-www-form-urlencoded'
                    response = await client.post(
                        self.webhook_url,
                        data=payload,
                        headers=headers,
                        timeout=10.0
                    )
                
                if 200 <= response.status_code < 300:
                    self.record_sent(message)
                    logger.info(f"Sent webhook notification: {message.title}")
                    return NotificationStatus.SENT
                else:
                    logger.error(f"Webhook notification failed: {response.status_code} - {response.text}")
                    return NotificationStatus.FAILED
                    
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return NotificationStatus.FAILED
    
    def _format_webhook_payload(self, message: NotificationMessage) -> Union[Dict[str, Any], str]:
        """Format message for webhook."""
        if self.format_template == 'json':
            return {
                "notification": message.to_dict(),
                "alert": {
                    "level": message.level.value,
                    "title": message.title,
                    "description": message.message,
                    "source": message.source,
                    "timestamp": message.timestamp.isoformat(),
                    "metadata": message.metadata,
                    "tags": message.tags
                }
            }
        elif self.format_template == 'form':
            return {
                'level': message.level.value,
                'title': message.title,
                'message': message.message,
                'source': message.source,
                'timestamp': message.timestamp.isoformat(),
                'metadata': json.dumps(message.metadata),
                'tags': ','.join(message.tags)
            }
        else:
            # Custom template (simplified)
            return message.to_dict()


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel with templates."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.smtp_host = config.get('smtp_host', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.smtp_user = config.get('smtp_user')
        self.smtp_password = config.get('smtp_password')
        self.from_address = config.get('from_address')
        self.to_addresses = config.get('to_addresses', [])
        self.use_tls = config.get('use_tls', True)
        
        if not all([self.smtp_user, self.smtp_password, self.from_address]):
            raise ValueError("SMTP credentials and from_address are required")
        
        if not self.to_addresses:
            raise ValueError("At least one recipient email address is required")
    
    async def send_notification(self, message: NotificationMessage) -> NotificationStatus:
        """Send notification via email."""
        if not self.enabled:
            return NotificationStatus.FAILED
        
        if self.should_rate_limit(message):
            logger.warning(f"Rate limiting email notification: {message.title}")
            return NotificationStatus.RATE_LIMITED
        
        try:
            email_message = self._format_email_message(message)
            
            # Use asyncio to run SMTP in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_smtp_email, email_message)
            
            self.record_sent(message)
            logger.info(f"Sent email notification: {message.title}")
            return NotificationStatus.SENT
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return NotificationStatus.FAILED
    
    def _format_email_message(self, message: NotificationMessage) -> MIMEMultipart:
        """Format message for email."""
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{message.level.value.upper()}] {message.title}"
        msg['From'] = self.from_address
        msg['To'] = ', '.join(self.to_addresses)
        
        # Plain text version
        text_content = f"""
Shyvr RLTE Alert

Level: {message.level.value.upper()}
Title: {message.title}
Source: {message.source}
Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Message:
{message.message}

Metadata:
{json.dumps(message.metadata, indent=2) if message.metadata else 'None'}

Tags: {', '.join(message.tags) if message.tags else 'None'}

---
Shyvr RLTE Monitoring System
        """.strip()
        
        # HTML version
        html_content = f"""
        <html>
        <body>
        <h2 style="color: {'red' if message.level in [NotificationLevel.ERROR, NotificationLevel.CRITICAL] else 'orange' if message.level == NotificationLevel.WARNING else 'blue'};">
            Shyvr RLTE Alert
        </h2>
        
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td><strong>Level:</strong></td><td style="color: {'red' if message.level in [NotificationLevel.ERROR, NotificationLevel.CRITICAL] else 'orange' if message.level == NotificationLevel.WARNING else 'green'};">{message.level.value.upper()}</td></tr>
            <tr><td><strong>Title:</strong></td><td>{message.title}</td></tr>
            <tr><td><strong>Source:</strong></td><td>{message.source}</td></tr>
            <tr><td><strong>Time:</strong></td><td>{message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td></tr>
        </table>
        
        <h3>Message:</h3>
        <p>{message.message}</p>
        
        <h3>Additional Information:</h3>
        <pre>{json.dumps(message.metadata, indent=2) if message.metadata else 'None'}</pre>
        
        <p><strong>Tags:</strong> {', '.join(message.tags) if message.tags else 'None'}</p>
        
        <hr>
        <p><em>Shyvr RLTE Monitoring System</em></p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        return msg
    
    def _send_smtp_email(self, message: MIMEMultipart):
        """Send email via SMTP (synchronous)."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(message)


class NotificationManager:
    """
    Central notification manager that handles multiple channels and deduplication.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.channels: Dict[str, NotificationChannel] = {}
        self.deduplication_window = config.get('deduplication_window_minutes', 15)
        self.recent_messages: Dict[str, datetime] = {}
        
        self._initialize_channels()
    
    def _initialize_channels(self):
        """Initialize notification channels from configuration."""
        channels_config = self.config.get('channels', {})
        
        for channel_name, channel_config in channels_config.items():
            channel_type = channel_config.get('type')
            
            try:
                if channel_type == 'slack':
                    channel = SlackNotificationChannel(channel_name, channel_config)
                elif channel_type == 'webhook':
                    channel = WebhookNotificationChannel(channel_name, channel_config)
                elif channel_type == 'email':
                    channel = EmailNotificationChannel(channel_name, channel_config)
                else:
                    logger.warning(f"Unknown channel type: {channel_type}")
                    continue
                
                self.channels[channel_name] = channel
                logger.info(f"Initialized notification channel: {channel_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize channel {channel_name}: {e}")
    
    def should_deduplicate(self, message: NotificationMessage) -> bool:
        """Check if message should be deduplicated."""
        message_hash = message.get_hash()
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.deduplication_window)
        
        if message_hash in self.recent_messages:
            if self.recent_messages[message_hash] > cutoff_time:
                return True
        
        self.recent_messages[message_hash] = datetime.utcnow()
        
        # Clean old entries
        self.recent_messages = {
            h: t for h, t in self.recent_messages.items() 
            if t > cutoff_time
        }
        
        return False
    
    async def send_notification(
        self, 
        message: NotificationMessage, 
        channels: Optional[List[str]] = None
    ) -> Dict[str, NotificationStatus]:
        """
        Send notification through specified channels.
        
        Args:
            message: Notification message to send
            channels: List of channel names (None = all enabled channels)
            
        Returns:
            Dictionary mapping channel names to delivery status
        """
        # Check for deduplication
        if self.should_deduplicate(message):
            logger.info(f"Deduplicated notification: {message.title}")
            return {channel: NotificationStatus.DEDUPLICATED for channel in (channels or self.channels.keys())}
        
        target_channels = channels or list(self.channels.keys())
        results = {}
        
        for channel_name in target_channels:
            if channel_name not in self.channels:
                logger.warning(f"Unknown notification channel: {channel_name}")
                results[channel_name] = NotificationStatus.FAILED
                continue
            
            channel = self.channels[channel_name]
            
            try:
                status = await channel.send_notification(message)
                results[channel_name] = status
                
            except Exception as e:
                logger.error(f"Error sending notification to {channel_name}: {e}")
                results[channel_name] = NotificationStatus.FAILED
        
        return results
    
    async def send_alert(
        self,
        level: NotificationLevel,
        title: str,
        message: str,
        source: str,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, NotificationStatus]:
        """
        Convenience method to send an alert notification.
        
        Args:
            level: Alert severity level
            title: Alert title
            message: Alert message
            source: Source component that generated the alert
            metadata: Additional metadata
            tags: List of tags
            channels: Target channels (None = all)
            
        Returns:
            Dictionary mapping channel names to delivery status
        """
        notification = NotificationMessage(
            id=f"{int(time.time())}-{hashlib.md5(title.encode()).hexdigest()[:8]}",
            level=level,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            tags=tags or []
        )
        
        return await self.send_notification(notification, channels)
    
    def get_channel_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all notification channels."""
        status = {}
        
        for name, channel in self.channels.items():
            status[name] = {
                "name": name,
                "type": type(channel).__name__,
                "enabled": channel.enabled,
                "recent_sends": len([
                    t for times in channel.last_sent_times.values() 
                    for t in (times if isinstance(times, list) else [times])
                    if t > datetime.utcnow() - timedelta(hours=1)
                ])
            }
        
        return status
    
    async def test_channels(self) -> Dict[str, NotificationStatus]:
        """Send test notifications to all channels."""
        test_message = NotificationMessage(
            id=f"test-{int(time.time())}",
            level=NotificationLevel.INFO,
            title="Shyvr RLTE Monitoring Test",
            message="This is a test notification to verify the monitoring system is working correctly.",
            source="notification_manager",
            timestamp=datetime.utcnow(),
            metadata={"test": True},
            tags=["test", "monitoring"]
        )
        
        return await self.send_notification(test_message)