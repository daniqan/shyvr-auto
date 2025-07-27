"""
Alerting service for sending notifications about critical events.

This module provides alerting capabilities including:
- Telegram bot integration for real-time notifications
- Alert level management (info, warning, error, critical)
- Queue-based alert processing
- Formatted messages for different alert types
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    def __lt__(self, other):
        """Allow comparison of alert levels."""
        if not isinstance(other, AlertLevel):
            return NotImplemented
        
        order = {
            AlertLevel.INFO: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.ERROR: 3,
            AlertLevel.CRITICAL: 4
        }
        return order[self] < order[other]


@dataclass
class Alert:
    """Represents an alert to be sent."""
    level: AlertLevel
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary representation."""
        return {
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation of the alert."""
        return f"[{self.level.value.upper()}] {self.source}: {self.message}"


class AlertingService:
    """
    Service for sending alerts via various channels.
    
    Currently supports Telegram bot notifications with plans for
    additional channels like email, Slack, etc.
    """
    
    def __init__(self, telegram_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the alerting service.
        
        Args:
            telegram_config: Configuration for Telegram bot
        """
        self.telegram_config = telegram_config or {}
        self.telegram_enabled = self.telegram_config.get("enabled", False)
        
        # Initialize alert queue for async processing
        self.alert_queue: asyncio.Queue = asyncio.Queue()
        
        # Service state
        self._running = False
        self._telegram_bot = None
        
        logger.info(
            "AlertingService initialized",
            telegram_enabled=self.telegram_enabled
        )
    
    async def send_alert(
        self, 
        alert_or_level: Union[Alert, AlertLevel],
        message: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send an alert.
        
        Args:
            alert_or_level: Either an Alert object or AlertLevel
            message: Alert message (if alert_or_level is AlertLevel)
            source: Alert source (if alert_or_level is AlertLevel)
            metadata: Additional metadata (if alert_or_level is AlertLevel)
        """
        if isinstance(alert_or_level, Alert):
            alert = alert_or_level
        else:
            # Create alert from parameters
            if message is None or source is None:
                raise ValueError("message and source are required when passing AlertLevel")
            
            alert = Alert(
                level=alert_or_level,
                message=message,
                source=source,
                metadata=metadata or {}
            )
        
        # Add to queue for processing
        await self.alert_queue.put(alert)
        
        # Process the queue
        await self._process_alert_queue()
    
    async def send_critical_alert(
        self, 
        message: str, 
        source: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a critical alert."""
        await self.send_alert(
            AlertLevel.CRITICAL,
            message=message,
            source=source,
            metadata=metadata
        )
    
    async def send_error_alert(
        self, 
        message: str, 
        source: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send an error alert."""
        await self.send_alert(
            AlertLevel.ERROR,
            message=message,
            source=source,
            metadata=metadata
        )
    
    async def send_warning_alert(
        self, 
        message: str, 
        source: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a warning alert."""
        await self.send_alert(
            AlertLevel.WARNING,
            message=message,
            source=source,
            metadata=metadata
        )
    
    async def send_info_alert(
        self, 
        message: str, 
        source: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send an info alert."""
        await self.send_alert(
            AlertLevel.INFO,
            message=message,
            source=source,
            metadata=metadata
        )
    
    async def _process_alert_queue(self) -> None:
        """Process alerts in the queue."""
        while not self.alert_queue.empty():
            try:
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # Send via all enabled channels
                if self.telegram_enabled:
                    await self._send_telegram_alert(alert)
                
                # Mark task as done
                self.alert_queue.task_done()
                
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error("Error processing alert", error=str(e))
                continue
    
    async def _send_telegram_alert(self, alert: Alert) -> None:
        """
        Send alert via Telegram bot.
        
        Args:
            alert: Alert to send
        """
        if not self.telegram_enabled:
            return
        
        try:
            # Import telegram here to avoid import errors if not installed
            from telegram import Bot
            
            # Initialize bot if needed
            if self._telegram_bot is None:
                bot_token = self.telegram_config.get("bot_token")
                if not bot_token:
                    logger.warning("Telegram bot token not configured")
                    return
                
                self._telegram_bot = Bot(token=bot_token)
            
            # Format message
            message_text = self._format_telegram_message(alert)
            
            # Send message
            chat_id = self.telegram_config.get("chat_id")
            if chat_id:
                await self._telegram_bot.send_message(
                    chat_id=chat_id,
                    text=message_text,
                    parse_mode="HTML"
                )
                
                logger.info(
                    "Telegram alert sent",
                    level=alert.level.value,
                    source=alert.source
                )
            else:
                logger.warning("Telegram chat_id not configured")
                
        except ImportError:
            logger.warning("python-telegram-bot not installed, skipping Telegram alert")
        except Exception as e:
            logger.error("Failed to send Telegram alert", error=str(e))
    
    def _format_telegram_message(self, alert: Alert) -> str:
        """
        Format alert message for Telegram.
        
        Args:
            alert: Alert to format
            
        Returns:
            Formatted message string
        """
        emoji = self._get_alert_emoji(alert.level)
        level_name = alert.level.value.upper()
        
        # Base message
        message = f"{emoji} <b>{level_name}</b>\n\n"
        message += f"<b>Source:</b> {alert.source}\n"
        message += f"<b>Message:</b> {alert.message}\n"
        message += f"<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        
        # Add metadata if present
        if alert.metadata:
            message += "\n<b>Details:</b>\n"
            for key, value in alert.metadata.items():
                message += f"• {key}: {value}\n"
        
        return message
    
    def _get_alert_emoji(self, level: AlertLevel) -> str:
        """
        Get emoji for alert level.
        
        Args:
            level: Alert level
            
        Returns:
            Emoji string
        """
        emoji_map = {
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.ERROR: "❌",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.INFO: "ℹ️"
        }
        return emoji_map.get(level, "📢")
    
    async def start(self) -> None:
        """Start the alerting service."""
        self._running = True
        logger.info("AlertingService started")
    
    async def stop(self) -> None:
        """Stop the alerting service."""
        self._running = False
        
        # Wait for queue to be processed
        if not self.alert_queue.empty():
            await self.alert_queue.join()
        
        logger.info("AlertingService stopped")
    
    def is_healthy(self) -> bool:
        """
        Check if the alerting service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        # Service is healthy if it's not severely backed up
        queue_size = self.alert_queue.qsize()
        max_queue_size = 100  # Arbitrary threshold
        
        return queue_size < max_queue_size
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information for the alerting service.
        
        Returns:
            Dictionary with service status information
        """
        return {
            "running": self._running,
            "telegram_enabled": self.telegram_enabled,
            "queue_size": self.alert_queue.qsize(),
            "healthy": self.is_healthy()
        }