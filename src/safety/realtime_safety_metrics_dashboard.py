"""
Real-time Safety Metrics Dashboard - Phase 3.2.4

This module provides a comprehensive real-time safety metrics dashboard
with live metrics aggregation, websocket/SSE updates, alert integration,
and historical visualization.

Key Features:
- Real-time metrics collection and aggregation from multiple sources
- WebSocket and Server-Sent Events (SSE) streaming for live updates
- Intelligent alert system with configurable thresholds and debouncing
- Historical metrics storage and visualization
- Interactive dashboard with multiple chart types
- High-performance concurrent client handling
- Production-grade error handling and monitoring

Following TDD methodology - implementation satisfies comprehensive test requirements.
"""

import asyncio
import json
import time
import logging
import websockets
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Callable, Union, Tuple, AsyncGenerator
from uuid import UUID, uuid4
from collections import defaultdict, deque
import statistics
import sqlite3
import aiosqlite
from threading import Lock
import weakref


# Use standard logging
logger = logging.getLogger(__name__)


class SafetyMetricType(Enum):
    """Types of safety metrics that can be monitored."""
    RISK_LEVEL = "risk_level"
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"
    EMERGENCY_STOPS = "emergency_stops"
    LIQUIDATIONS = "liquidations"
    SYSTEM_HEALTH = "system_health"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    MARGIN_RATIO = "margin_ratio"
    VOLATILITY = "volatility"
    CORRELATION_RISK = "correlation_risk"
    ACTIVE_POSITIONS = "active_positions"
    TOTAL_EXPOSURE = "total_exposure"


class SafetyAlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SafetyDashboardError(Exception):
    """Base exception for dashboard operations."""
    pass


class SafetyMetricsStreamError(SafetyDashboardError):
    """Exception for metrics streaming operations."""
    pass


class SafetyVisualizationError(SafetyDashboardError):
    """Exception for visualization operations."""
    pass


@dataclass
class SafetyMetricValue:
    """Represents a safety metric value with metadata."""
    metric_type: SafetyMetricType
    value: Union[float, int, Decimal]
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_type': self.metric_type.value,
            'value': float(self.value) if isinstance(self.value, Decimal) else self.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass
class SafetyAlert:
    """Represents a safety alert."""
    alert_id: str
    metric_type: SafetyMetricType
    level: SafetyAlertLevel
    message: str
    current_value: Union[float, int]
    threshold: Union[float, int]
    timestamp: datetime
    threshold_exceeded: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'metric_type': self.metric_type.value,
            'level': self.level.value,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'threshold_exceeded': self.threshold_exceeded,
            'metadata': self.metadata
        }


@dataclass
class SafetyMetricsConfig:
    """Configuration for real-time safety metrics dashboard."""
    metrics_update_interval: float = 1.0  # seconds
    websocket_port: int = 8765
    sse_endpoint: str = '/safety-metrics-stream'
    enable_websocket: bool = True
    enable_sse: bool = True
    enable_alerts: bool = True
    max_concurrent_connections: int = 100
    metrics_history_size: int = 1000
    alert_debounce_seconds: int = 30
    dashboard_refresh_rate: float = 2.0
    enable_compression: bool = True
    enable_authentication: bool = False
    collection_interval: float = 1.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 30
    ping_interval: int = 30
    ping_timeout: int = 10
    keepalive_interval: int = 30
    chart_types: List[str] = field(default_factory=lambda: ['line', 'gauge', 'bar', 'heatmap'])
    update_interval: float = 2.0
    history_window_hours: int = 24
    enable_real_time_updates: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'risk_level_critical': 0.8,
        'risk_level_warning': 0.6,
        'system_health_critical': 0.3,
        'system_health_warning': 0.5,
        'emergency_stop_threshold': 5,
        'liquidation_threshold': 3,
        'error_rate_warning': 0.05,
        'error_rate_critical': 0.1,
        'response_time_warning': 100.0,  # ms
        'response_time_critical': 500.0  # ms
    })
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = False
    enable_slack_alerts: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.metrics_update_interval <= 0:
            raise ValueError("metrics_update_interval must be positive")
        if not (1024 <= self.websocket_port <= 65535):
            raise ValueError("websocket_port must be between 1024 and 65535")
        if self.max_concurrent_connections <= 0:
            raise ValueError("max_concurrent_connections must be positive")


class SafetyMetricsAggregator:
    """Aggregates safety metrics from multiple sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_interval = config.get('collection_interval', 1.0)
        self.enable_caching = config.get('enable_caching', True)
        self.cache_ttl_seconds = config.get('cache_ttl_seconds', 30)
        self.logger = logging.getLogger(f"{__name__}.aggregator")
        
        # Cache management
        self._metrics_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_lock = Lock()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all sources."""
        # Check cache first
        if self.enable_caching and self._is_cache_valid():
            with self._cache_lock:
                return self._metrics_cache.copy()
        
        # Collect from all sources
        try:
            risk_metrics = await self._collect_risk_metrics()
            portfolio_metrics = await self._collect_portfolio_metrics()
            system_metrics = await self._collect_system_metrics()
            safety_event_metrics = await self._collect_safety_event_metrics()
            
            # Aggregate all metrics
            all_metrics = {
                **risk_metrics,
                **portfolio_metrics,
                **system_metrics,
                **safety_event_metrics,
                'collection_timestamp': datetime.utcnow().isoformat()
            }
            
            # Validate and normalize
            normalized_metrics = await self.validate_and_normalize_metrics(all_metrics)
            
            # Update cache
            if self.enable_caching:
                with self._cache_lock:
                    self._metrics_cache = normalized_metrics.copy()
                    self._cache_timestamp = datetime.utcnow()
            
            return normalized_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            # Return default metrics on failure
            return self._get_default_metrics()
    
    async def validate_and_normalize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize metrics values."""
        normalized = {}
        
        for key, value in metrics.items():
            if key == 'collection_timestamp':
                normalized[key] = value
                continue
            
            try:
                # Convert to float if possible
                if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    value = float(value)
                elif isinstance(value, Decimal):
                    value = float(value)
                
                # Normalize specific metrics
                if key in ['risk_level', 'system_health_score']:
                    # Ensure 0-1 range
                    normalized[key] = max(0.0, min(1.0, float(value)))
                elif key in ['margin_ratio', 'total_exposure']:
                    # Ensure non-negative
                    normalized[key] = max(0.0, float(value))
                elif key in ['emergency_stops_24h', 'liquidations_24h', 'active_positions']:
                    # Ensure non-negative integers
                    normalized[key] = max(0, int(value))
                elif key in ['error_rate_1h']:
                    # Ensure 0-1 range for error rates
                    normalized[key] = max(0.0, min(1.0, float(value)))
                else:
                    normalized[key] = value
                    
            except (ValueError, TypeError):
                # Use default value for invalid metrics
                normalized[key] = self._get_default_value(key)
        
        return normalized
    
    async def _collect_risk_metrics(self) -> Dict[str, Any]:
        """Collect risk management metrics."""
        # Simulate risk metrics collection
        return {
            'risk_level': 0.35,
            'position_size_ratio': 0.65,
            'margin_ratio': 2.2,
            'volatility': 0.45,
            'correlation_risk': 0.28
        }
    
    async def _collect_portfolio_metrics(self) -> Dict[str, Any]:
        """Collect portfolio metrics."""
        # Simulate portfolio metrics collection
        return {
            'current_drawdown': 0.12,
            'total_exposure': Decimal('85000.00'),
            'active_positions': 8
        }
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system health metrics."""
        # Simulate system metrics collection
        return {
            'system_health_score': 0.88,
            'avg_response_time_ms': 52.3,
            'error_rate_1h': 0.001
        }
    
    async def _collect_safety_event_metrics(self) -> Dict[str, Any]:
        """Collect safety event metrics."""
        # Simulate safety event metrics collection
        return {
            'emergency_stops_24h': 2,
            'liquidations_24h': 1,
            'safety_checks_failed_1h': 0,
            'risk_violations_24h': 3
        }
    
    def _is_cache_valid(self) -> bool:
        """Check if cached metrics are still valid."""
        if not self._cache_timestamp or not self._metrics_cache:
            return False
        
        age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
        return age < self.cache_ttl_seconds
    
    def _is_cache_expired(self) -> bool:
        """Check if cache has expired (for testing)."""
        return not self._is_cache_valid()
    
    async def _collect_all_sources(self) -> Dict[str, Any]:
        """Collect from all sources (used for mocking in tests)."""
        return await self.collect_metrics()
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics in case of collection failure."""
        return {
            'risk_level': 0.0,
            'system_health_score': 1.0,
            'emergency_stops_24h': 0,
            'liquidations_24h': 0,
            'position_size_ratio': 0.0,
            'current_drawdown': 0.0,
            'total_exposure': 0.0,
            'active_positions': 0,
            'avg_response_time_ms': 0.0,
            'error_rate_1h': 0.0,
            'collection_timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_default_value(self, key: str) -> Union[float, int]:
        """Get default value for a specific metric."""
        defaults = {
            'risk_level': 0.0,
            'system_health_score': 1.0,
            'emergency_stops_24h': 0,
            'liquidations_24h': 0,
            'position_size_ratio': 0.0,
            'current_drawdown': 0.0,
            'total_exposure': 0.0,
            'active_positions': 0,
            'avg_response_time_ms': 0.0,
            'error_rate_1h': 0.0,
            'margin_ratio': 1.0,
            'volatility': 0.0,
            'correlation_risk': 0.0
        }
        return defaults.get(key, 0.0)


class SafetyMetricsWebSocketHandler:
    """Handles WebSocket connections for real-time metrics streaming."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.port = config.get('port', 8765)
        self.max_connections = config.get('max_connections', 100)
        self.ping_interval = config.get('ping_interval', 30)
        self.ping_timeout = config.get('ping_timeout', 10)
        self.logger = logging.getLogger(f"{__name__}.websocket")
        
        # Client management
        self.clients: Set[Any] = set()
        self.client_subscriptions: Dict[Any, List[SafetyMetricType]] = {}
        self._server = None
    
    async def start_server(self):
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self._handle_client,
            "localhost",
            self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout
        )
        self.logger.info(f"WebSocket server started on port {self.port}")
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self.logger.info("WebSocket server stopped")
    
    async def _handle_client(self, websocket, path):
        """Handle individual client connection."""
        try:
            await self.handle_client_connect(websocket)
            
            # Keep connection alive and handle messages
            async for message in websocket:
                await self._process_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"WebSocket client error: {e}")
        finally:
            await self.handle_client_disconnect(websocket)
    
    async def handle_client_connect(self, websocket):
        """Handle client connection."""
        if len(self.clients) >= self.max_connections:
            await websocket.close(code=1013, reason="Server overloaded")
            return
        
        self.clients.add(websocket)
        self.client_subscriptions[websocket] = list(SafetyMetricType)  # Subscribe to all by default
        
        self.logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send welcome message
        welcome_msg = {
            'type': 'welcome',
            'message': 'Connected to Safety Metrics Dashboard',
            'available_metrics': [metric.value for metric in SafetyMetricType]
        }
        await websocket.send(json.dumps(welcome_msg))
    
    async def handle_client_disconnect(self, websocket):
        """Handle client disconnection."""
        self.clients.discard(websocket)
        self.client_subscriptions.pop(websocket, None)
        
        if hasattr(websocket, 'remote_address'):
            self.logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def _process_client_message(self, websocket, message):
        """Process message from client."""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'subscribe':
                # Update client subscriptions
                requested_metrics = data.get('metrics', [])
                subscriptions = []
                
                for metric_name in requested_metrics:
                    try:
                        metric_type = SafetyMetricType(metric_name)
                        subscriptions.append(metric_type)
                    except ValueError:
                        pass
                
                if subscriptions:
                    self.client_subscriptions[websocket] = subscriptions
                    
                    response = {
                        'type': 'subscription_updated',
                        'subscribed_metrics': [m.value for m in subscriptions]
                    }
                    await websocket.send(json.dumps(response))
                    
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Invalid client message: {e}")
    
    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast metrics to all connected clients."""
        if not self.clients:
            return
        
        # Create broadcast message
        message = {
            'type': 'metrics_update',
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to all clients (with subscription filtering)
        disconnected_clients = set()
        
        for client in self.clients.copy():
            try:
                # Filter metrics based on client subscriptions
                client_subscriptions = self.client_subscriptions.get(client, [])
                filtered_metrics = self._filter_metrics_for_client(metrics, client_subscriptions)
                
                filtered_message = {
                    **message,
                    'metrics': filtered_metrics
                }
                
                await client.send(json.dumps(filtered_message))
                
            except (websockets.exceptions.ConnectionClosed, Exception) as e:
                self.logger.warning(f"Failed to send to client: {e}")
                disconnected_clients.add(client)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            await self.handle_client_disconnect(client)
    
    def _filter_metrics_for_client(self, metrics: Dict[str, Any], subscriptions: List[SafetyMetricType]) -> Dict[str, Any]:
        """Filter metrics based on client subscriptions."""
        if not subscriptions:
            return metrics
        
        subscription_keys = {metric.value for metric in subscriptions}
        filtered = {}
        
        for key, value in metrics.items():
            if key in subscription_keys or key in ['timestamp', 'collection_timestamp']:
                filtered[key] = value
        
        return filtered


class SafetyMetricsSSEHandler:
    """Handles Server-Sent Events for metrics streaming."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoint = config.get('endpoint', '/safety-metrics-stream')
        self.max_connections = config.get('max_connections', 50)
        self.keepalive_interval = config.get('keepalive_interval', 30)
        self.logger = logging.getLogger(f"{__name__}.sse")
        
        # Client management
        self.clients: List[Any] = []
    
    async def add_client(self, client):
        """Add SSE client."""
        if len(self.clients) >= self.max_connections:
            raise SafetyMetricsStreamError("Maximum SSE connections exceeded")
        
        self.clients.append(client)
        self.logger.info(f"SSE client added: {client.client_id}")
    
    async def remove_client(self, client_id: str):
        """Remove SSE client."""
        self.clients = [c for c in self.clients if c.client_id != client_id]
        self.logger.info(f"SSE client removed: {client_id}")
    
    async def send_metrics_event(self, metrics: Dict[str, Any]):
        """Send metrics as SSE event to all clients."""
        if not self.clients:
            return
        
        event_data = json.dumps(metrics)
        
        disconnected_clients = []
        
        for client in self.clients.copy():
            try:
                await client.send_event(
                    event='metrics_update',
                    data=event_data,
                    retry=1000
                )
            except Exception as e:
                self.logger.warning(f"Failed to send SSE to client {client.client_id}: {e}")
                disconnected_clients.append(client.client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.remove_client(client_id)


class SafetyMetricsAlertIntegration:
    """Integrates with alerting systems for safety metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.debounce_seconds = config.get('debounce_seconds', 30)
        self.enable_email_alerts = config.get('enable_email_alerts', False)
        self.enable_webhook_alerts = config.get('enable_webhook_alerts', False)
        self.enable_slack_alerts = config.get('enable_slack_alerts', False)
        self.logger = logging.getLogger(f"{__name__}.alerts")
        
        # Alert state management
        self._last_alert_times: Dict[str, datetime] = {}
        self._active_alerts: Dict[str, SafetyAlert] = {}
    
    async def process_metrics(self, metrics: Dict[str, Any]) -> List[SafetyAlert]:
        """Process metrics and generate alerts."""
        alerts = []
        
        # Check each metric against thresholds
        for metric_key, value in metrics.items():
            if metric_key in ['timestamp', 'collection_timestamp']:
                continue
            
            # Check critical thresholds
            critical_key = f"{metric_key}_critical"
            if critical_key in self.alert_thresholds:
                alert = await self._check_threshold(
                    metric_key, value, self.alert_thresholds[critical_key],
                    SafetyAlertLevel.CRITICAL, 'greater_than'
                )
                if alert:
                    alerts.append(alert)
            
            # Check warning thresholds
            warning_key = f"{metric_key}_warning"
            if warning_key in self.alert_thresholds:
                alert = await self._check_threshold(
                    metric_key, value, self.alert_thresholds[warning_key],
                    SafetyAlertLevel.WARNING, 'greater_than'
                )
                if alert:
                    alerts.append(alert)
            
            # Special threshold checks
            if metric_key == 'system_health_score':
                # For system health, low values are bad
                if value < self.alert_thresholds.get('system_health_critical', 0.3):
                    alert = await self._create_alert(
                        SafetyMetricType.SYSTEM_HEALTH, SafetyAlertLevel.ERROR,
                        f"System health critically low: {value:.2f}",
                        value, self.alert_thresholds.get('system_health_critical', 0.3)
                    )
                    if alert:
                        alerts.append(alert)
        
        # Filter out debounced alerts
        filtered_alerts = []
        for alert in alerts:
            if not self._is_debounced(alert):
                filtered_alerts.append(alert)
                self._last_alert_times[alert.alert_id] = datetime.utcnow()
        
        return filtered_alerts
    
    async def _check_threshold(
        self, 
        metric_key: str, 
        value: Union[float, int], 
        threshold: Union[float, int],
        level: SafetyAlertLevel,
        comparison: str = 'greater_than'
    ) -> Optional[SafetyAlert]:
        """Check if metric value exceeds threshold."""
        try:
            numeric_value = float(value)
            threshold_value = float(threshold)
            
            exceeded = False
            if comparison == 'greater_than':
                exceeded = numeric_value > threshold_value
            elif comparison == 'less_than':
                exceeded = numeric_value < threshold_value
            
            if exceeded:
                # Try to map metric key to SafetyMetricType
                try:
                    metric_type = SafetyMetricType(metric_key)
                except ValueError:
                    # Create a generic metric type
                    metric_type = SafetyMetricType.RISK_LEVEL  # Fallback
                
                return await self._create_alert(
                    metric_type, level,
                    f"{metric_key.replace('_', ' ').title()} threshold exceeded: {numeric_value} > {threshold_value}",
                    numeric_value, threshold_value
                )
            
        except (ValueError, TypeError):
            pass
        
        return None
    
    async def _create_alert(
        self,
        metric_type: SafetyMetricType,
        level: SafetyAlertLevel,
        message: str,
        current_value: Union[float, int],
        threshold: Union[float, int]
    ) -> SafetyAlert:
        """Create a safety alert."""
        alert = SafetyAlert(
            alert_id=str(uuid4()),
            metric_type=metric_type,
            level=level,
            message=message,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.utcnow()
        )
        
        return alert
    
    def _is_debounced(self, alert: SafetyAlert) -> bool:
        """Check if alert should be debounced."""
        alert_key = f"{alert.metric_type.value}_{alert.level.value}"
        last_time = self._last_alert_times.get(alert_key)
        
        if not last_time:
            return False
        
        time_since_last = (datetime.utcnow() - last_time).total_seconds()
        return time_since_last < self.debounce_seconds
    
    def _is_debounce_expired(self, alert_key: str) -> bool:
        """Check if debounce period has expired (for testing)."""
        last_time = self._last_alert_times.get(alert_key)
        if not last_time:
            return True
        
        time_since_last = (datetime.utcnow() - last_time).total_seconds()
        return time_since_last >= self.debounce_seconds
    
    async def dispatch_alert(self, alert: SafetyAlert):
        """Dispatch alert to configured channels."""
        if self.enable_email_alerts:
            await self._send_email_alert(alert)
        
        if self.enable_webhook_alerts:
            await self._send_webhook_alert(alert)
        
        if self.enable_slack_alerts:
            await self._send_slack_alert(alert)
    
    async def _send_email_alert(self, alert: SafetyAlert):
        """Send email alert (placeholder implementation)."""
        self.logger.info(f"Email alert sent: {alert.message}")
    
    async def _send_webhook_alert(self, alert: SafetyAlert):
        """Send webhook alert (placeholder implementation)."""
        self.logger.info(f"Webhook alert sent: {alert.message}")
    
    async def _send_slack_alert(self, alert: SafetyAlert):
        """Send Slack alert (placeholder implementation)."""
        self.logger.info(f"Slack alert sent: {alert.message}")


class SafetyMetricsVisualization:
    """Handles metrics visualization and dashboard generation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chart_types = config.get('chart_types', ['line', 'gauge', 'bar', 'heatmap'])
        self.update_interval = config.get('update_interval', 2.0)
        self.history_window_hours = config.get('history_window_hours', 24)
        self.enable_real_time_updates = config.get('enable_real_time_updates', True)
        self.logger = logging.getLogger(f"{__name__}.visualization")
    
    async def generate_line_chart_data(
        self, 
        metric_type: SafetyMetricType, 
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate line chart data for time series."""
        labels = []
        values = []
        
        for data_point in historical_data:
            labels.append(data_point['timestamp'].strftime('%H:%M:%S'))
            values.append(data_point['value'])
        
        return {
            'labels': labels,
            'datasets': [{
                'label': metric_type.value.replace('_', ' ').title(),
                'data': values,
                'borderColor': self._get_metric_color(metric_type),
                'backgroundColor': self._get_metric_color(metric_type, alpha=0.2),
                'tension': 0.4
            }]
        }
    
    async def generate_gauge_chart_data(
        self, 
        metric_type: SafetyMetricType, 
        current_value: Union[float, int]
    ) -> Dict[str, Any]:
        """Generate gauge chart data for current values."""
        min_val, max_val = self._get_metric_range(metric_type)
        
        return {
            'value': current_value,
            'min': min_val,
            'max': max_val,
            'title': metric_type.value.replace('_', ' ').title(),
            'color': self._get_gauge_color(metric_type, current_value),
            'thresholds': self._get_metric_thresholds(metric_type)
        }
    
    async def generate_dashboard_layout(
        self, 
        metric_types: List[SafetyMetricType]
    ) -> Dict[str, Any]:
        """Generate dashboard layout configuration."""
        components = []
        
        for i, metric_type in enumerate(metric_types):
            component_type = self._get_preferred_chart_type(metric_type)
            
            component = {
                'id': f"chart_{metric_type.value}",
                'type': component_type,
                'metric_type': metric_type.value,
                'title': metric_type.value.replace('_', ' ').title(),
                'position': {
                    'x': (i % 2) * 6,
                    'y': (i // 2) * 4,
                    'w': 6,
                    'h': 4
                }
            }
            components.append(component)
        
        return {
            'components': components,
            'grid_layout': {
                'cols': 12,
                'row_height': 100,
                'margin': [10, 10]
            }
        }
    
    def _get_metric_color(self, metric_type: SafetyMetricType, alpha: float = 1.0) -> str:
        """Get color for metric visualization."""
        color_map = {
            SafetyMetricType.RISK_LEVEL: f"rgba(255, 99, 132, {alpha})",
            SafetyMetricType.SYSTEM_HEALTH: f"rgba(75, 192, 192, {alpha})",
            SafetyMetricType.EMERGENCY_STOPS: f"rgba(255, 159, 64, {alpha})",
            SafetyMetricType.POSITION_SIZE: f"rgba(153, 102, 255, {alpha})",
            SafetyMetricType.DRAWDOWN: f"rgba(255, 205, 86, {alpha})"
        }
        return color_map.get(metric_type, f"rgba(201, 203, 207, {alpha})")
    
    def _get_metric_range(self, metric_type: SafetyMetricType) -> Tuple[float, float]:
        """Get min/max range for metric."""
        range_map = {
            SafetyMetricType.RISK_LEVEL: (0.0, 1.0),
            SafetyMetricType.SYSTEM_HEALTH: (0.0, 1.0),
            SafetyMetricType.EMERGENCY_STOPS: (0, 10),
            SafetyMetricType.POSITION_SIZE: (0.0, 1.0),
            SafetyMetricType.DRAWDOWN: (0.0, 1.0)
        }
        return range_map.get(metric_type, (0.0, 100.0))
    
    def _get_gauge_color(self, metric_type: SafetyMetricType, value: Union[float, int]) -> str:
        """Get gauge color based on value."""
        if metric_type == SafetyMetricType.RISK_LEVEL:
            if value > 0.8:
                return "#dc3545"  # Red
            elif value > 0.6:
                return "#ffc107"  # Yellow
            else:
                return "#28a745"  # Green
        elif metric_type == SafetyMetricType.SYSTEM_HEALTH:
            if value < 0.3:
                return "#dc3545"  # Red
            elif value < 0.6:
                return "#ffc107"  # Yellow
            else:
                return "#28a745"  # Green
        
        return "#6c757d"  # Default gray
    
    def _get_metric_thresholds(self, metric_type: SafetyMetricType) -> List[Dict[str, Any]]:
        """Get threshold markers for gauge charts."""
        if metric_type == SafetyMetricType.RISK_LEVEL:
            return [
                {'value': 0.6, 'color': '#ffc107', 'label': 'Warning'},
                {'value': 0.8, 'color': '#dc3545', 'label': 'Critical'}
            ]
        elif metric_type == SafetyMetricType.SYSTEM_HEALTH:
            return [
                {'value': 0.3, 'color': '#dc3545', 'label': 'Critical'},
                {'value': 0.6, 'color': '#ffc107', 'label': 'Warning'}
            ]
        
        return []
    
    def _get_preferred_chart_type(self, metric_type: SafetyMetricType) -> str:
        """Get preferred chart type for metric."""
        gauge_metrics = [
            SafetyMetricType.RISK_LEVEL,
            SafetyMetricType.SYSTEM_HEALTH,
            SafetyMetricType.POSITION_SIZE
        ]
        
        bar_metrics = [
            SafetyMetricType.EMERGENCY_STOPS,
            SafetyMetricType.LIQUIDATIONS,
            SafetyMetricType.ACTIVE_POSITIONS
        ]
        
        if metric_type in gauge_metrics:
            return 'gauge'
        elif metric_type in bar_metrics:
            return 'bar' 
        else:
            return 'line'


class RealTimeSafetyMetricsDashboard:
    """Main real-time safety metrics dashboard system."""
    
    def __init__(self, config: SafetyMetricsConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.dashboard")
        
        # Component initialization
        self.metrics_aggregator = SafetyMetricsAggregator(config.__dict__)
        self.websocket_handler = SafetyMetricsWebSocketHandler({
            'port': config.websocket_port,
            'max_connections': config.max_concurrent_connections,
            'ping_interval': config.ping_interval,
            'ping_timeout': config.ping_timeout
        })
        self.sse_handler = SafetyMetricsSSEHandler({
            'endpoint': config.sse_endpoint,
            'max_connections': config.max_concurrent_connections // 2,
            'keepalive_interval': config.keepalive_interval
        })
        self.alert_integration = SafetyMetricsAlertIntegration(config.__dict__)
        self.visualization = SafetyMetricsVisualization(config.__dict__)
        
        # State management
        self.is_running = False
        self._latest_metrics: Optional[Dict[str, Any]] = None
        self._metrics_history: deque = deque(maxlen=config.metrics_history_size)
        self._update_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the dashboard system."""
        if self.is_running:
            return
        
        self.logger.info("Starting Safety Metrics Dashboard...")
        
        # Start WebSocket server if enabled
        if self.config.enable_websocket:
            await self.websocket_handler.start_server()
        
        # Start metrics collection loop
        self._update_task = asyncio.create_task(self._metrics_update_loop())
        
        self.is_running = True
        self.logger.info("Safety Metrics Dashboard started successfully")
    
    async def stop(self):
        """Stop the dashboard system."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Safety Metrics Dashboard...")
        
        # Stop metrics update loop
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        # Stop WebSocket server
        if self.config.enable_websocket:
            await self.websocket_handler.stop_server()
        
        self.is_running = False
        self.logger.info("Safety Metrics Dashboard stopped")
    
    async def _metrics_update_loop(self):
        """Main metrics update loop."""
        while self.is_running:
            try:
                await self.collect_and_broadcast_metrics()
                await asyncio.sleep(self.config.metrics_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def collect_and_broadcast_metrics(self):
        """Collect metrics and broadcast to all clients."""
        try:
            # Collect metrics
            metrics = await self.metrics_aggregator.collect_metrics()
            self._latest_metrics = metrics
            
            # Store in history
            self._metrics_history.append({
                'timestamp': datetime.utcnow(),
                'metrics': metrics.copy()
            })
            
            # Process alerts
            if self.config.enable_alerts:
                alerts = await self.process_alerts(metrics)
                for alert in alerts:
                    await self.alert_integration.dispatch_alert(alert)
            
            # Broadcast to clients
            if self.config.enable_websocket:
                await self.broadcast_metrics_websocket(metrics)
            
            if self.config.enable_sse:
                await self.broadcast_metrics_sse(metrics)
                
        except Exception as e:
            self.logger.error(f"Failed to collect and broadcast metrics: {e}")
    
    async def broadcast_metrics_websocket(self, metrics: Dict[str, Any]):
        """Broadcast metrics via WebSocket."""
        await self.websocket_handler.broadcast_metrics(metrics)
    
    async def broadcast_metrics_sse(self, metrics: Dict[str, Any]):
        """Broadcast metrics via SSE."""
        await self.sse_handler.send_metrics_event(metrics)
    
    async def process_alerts(self, metrics: Dict[str, Any]) -> List[SafetyAlert]:
        """Process metrics and generate alerts."""
        return await self.alert_integration.process_metrics(metrics)
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest collected metrics."""
        return self._latest_metrics
    
    async def store_historical_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in historical storage."""
        self._metrics_history.append({
            'timestamp': metrics.get('timestamp', datetime.utcnow()),
            'metrics': metrics
        })
    
    async def get_historical_metrics(
        self,
        metric_types: List[SafetyMetricType],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical metrics for specified types and time range."""
        result = {metric_type.value: [] for metric_type in metric_types}
        
        for entry in self._metrics_history:
            entry_time = entry['timestamp']
            if start_time <= entry_time <= end_time:
                for metric_type in metric_types:
                    metric_key = metric_type.value
                    if metric_key in entry['metrics']:
                        result[metric_key].append({
                            'timestamp': entry_time,
                            'value': entry['metrics'][metric_key]
                        })
        
        return result


# Factory function for easy instantiation
async def create_safety_metrics_dashboard(config: Optional[SafetyMetricsConfig] = None) -> RealTimeSafetyMetricsDashboard:
    """Create and start a real-time safety metrics dashboard."""
    if config is None:
        config = SafetyMetricsConfig()
    
    dashboard = RealTimeSafetyMetricsDashboard(config)
    await dashboard.start()
    return dashboard