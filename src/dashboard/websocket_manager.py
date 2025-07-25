"""
WebSocket Manager for Real-time Dashboard Updates
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import structlog

from .base import DashboardData, dashboard_cache, DashboardError

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        # Subscriptions by topic
        self.subscriptions: Dict[str, Set[str]] = {}
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket, connection_id: str, 
                      user_id: Optional[str] = None) -> None:
        """Accept and register a new WebSocket connection"""
        try:
            await websocket.accept()
            
            async with self._lock:
                self.active_connections[connection_id] = websocket
                self.connection_metadata[connection_id] = {
                    "user_id": user_id,
                    "connected_at": datetime.utcnow(),
                    "last_ping": datetime.utcnow(),
                    "subscriptions": set()
                }
            
            logger.info(
                "WebSocket connection established",
                connection_id=connection_id,
                user_id=user_id,
                total_connections=len(self.active_connections)
            )
            
        except Exception as e:
            logger.error(
                "Failed to establish WebSocket connection",
                connection_id=connection_id,
                error=str(e)
            )
            raise DashboardError(f"Failed to connect: {e}")
    
    async def disconnect(self, connection_id: str) -> None:
        """Remove a WebSocket connection"""
        async with self._lock:
            if connection_id in self.active_connections:
                # Remove from all subscriptions
                metadata = self.connection_metadata.get(connection_id, {})
                subscriptions = metadata.get("subscriptions", set())
                
                for topic in subscriptions:
                    if topic in self.subscriptions:
                        self.subscriptions[topic].discard(connection_id)
                        if not self.subscriptions[topic]:
                            del self.subscriptions[topic]
                
                # Remove connection
                del self.active_connections[connection_id]
                del self.connection_metadata[connection_id]
                
                logger.info(
                    "WebSocket connection closed",
                    connection_id=connection_id,
                    remaining_connections=len(self.active_connections)
                )
    
    async def subscribe(self, connection_id: str, topic: str) -> None:
        """Subscribe a connection to a topic"""
        async with self._lock:
            if connection_id not in self.active_connections:
                raise DashboardError(f"Connection {connection_id} not found")
            
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            
            self.subscriptions[topic].add(connection_id)
            self.connection_metadata[connection_id]["subscriptions"].add(topic)
            
            logger.debug(
                "Connection subscribed to topic",
                connection_id=connection_id,
                topic=topic
            )
    
    async def unsubscribe(self, connection_id: str, topic: str) -> None:
        """Unsubscribe a connection from a topic"""
        async with self._lock:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(connection_id)
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
            
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscriptions"].discard(topic)
            
            logger.debug(
                "Connection unsubscribed from topic",
                connection_id=connection_id,
                topic=topic
            )
    
    async def send_personal_message(self, message: Dict[str, Any], 
                                    connection_id: str) -> None:
        """Send a message to a specific connection"""
        if connection_id not in self.active_connections:
            logger.warning("Connection not found", connection_id=connection_id)
            return
        
        websocket = self.active_connections[connection_id]
        try:
            await websocket.send_text(json.dumps(message))
            logger.debug("Personal message sent", connection_id=connection_id)
        
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
        except Exception as e:
            logger.error(
                "Failed to send personal message",
                connection_id=connection_id,
                error=str(e)
            )
            await self.disconnect(connection_id)
    
    async def broadcast_to_topic(self, message: Dict[str, Any], topic: str) -> None:
        """Broadcast a message to all connections subscribed to a topic"""
        if topic not in self.subscriptions:
            logger.debug("No subscribers for topic", topic=topic)
            return
        
        subscribers = list(self.subscriptions[topic])  # Copy to avoid race conditions
        message_json = json.dumps(message)
        
        # Track failed connections to remove them
        failed_connections = []
        
        for connection_id in subscribers:
            if connection_id not in self.active_connections:
                failed_connections.append(connection_id)
                continue
            
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(message_json)
                
            except WebSocketDisconnect:
                failed_connections.append(connection_id)
            except Exception as e:
                logger.error(
                    "Failed to send broadcast message",
                    connection_id=connection_id,
                    topic=topic,
                    error=str(e)
                )
                failed_connections.append(connection_id)
        
        # Remove failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
        
        if subscribers:
            logger.debug(
                "Message broadcast to topic",
                topic=topic,
                subscribers=len(subscribers) - len(failed_connections),
                failed=len(failed_connections)
            )
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all active connections"""
        if not self.active_connections:
            return
        
        connection_ids = list(self.active_connections.keys())
        message_json = json.dumps(message)
        failed_connections = []
        
        for connection_id in connection_ids:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(message_json)
                
            except WebSocketDisconnect:
                failed_connections.append(connection_id)
            except Exception as e:
                logger.error(
                    "Failed to send broadcast message",
                    connection_id=connection_id,
                    error=str(e)
                )
                failed_connections.append(connection_id)
        
        # Remove failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
        
        logger.debug(
            "Message broadcast to all connections",
            total_connections=len(connection_ids),
            successful=len(connection_ids) - len(failed_connections),
            failed=len(failed_connections)
        )
    
    async def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
    
    async def get_topic_subscribers(self, topic: str) -> int:
        """Get the number of subscribers for a topic"""
        return len(self.subscriptions.get(topic, set()))
    
    async def ping_connections(self) -> None:
        """Send ping to all connections to keep them alive"""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast_to_all(ping_message)


class WebSocketManager:
    """Manages WebSocket connections and real-time updates"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self._update_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Update intervals (seconds)
        self.dashboard_update_interval = 2.0  # 2 seconds for dashboard data
        self.ping_interval = 30.0  # 30 seconds for connection keepalive
        
    async def start(self) -> None:
        """Start the WebSocket manager"""
        if self._running:
            return
        
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())
        
        logger.info("WebSocket manager started")
    
    async def stop(self) -> None:
        """Stop the WebSocket manager"""
        if not self._running:
            return
        
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        logger.info("WebSocket manager stopped")
    
    async def connect(self, websocket: WebSocket, connection_id: str,
                      user_id: Optional[str] = None) -> None:
        """Connect a new WebSocket client"""
        await self.connection_manager.connect(websocket, connection_id, user_id)
    
    async def disconnect(self, connection_id: str) -> None:
        """Disconnect a WebSocket client"""
        await self.connection_manager.disconnect(connection_id)
    
    async def handle_message(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message"""
        try:
            message_type = message.get("type")
            
            if message_type == "subscribe":
                topic = message.get("topic")
                if topic:
                    await self.connection_manager.subscribe(connection_id, topic)
                    await self.connection_manager.send_personal_message({
                        "type": "subscription_confirmed",
                        "topic": topic,
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
            
            elif message_type == "unsubscribe":
                topic = message.get("topic")
                if topic:
                    await self.connection_manager.unsubscribe(connection_id, topic)
                    await self.connection_manager.send_personal_message({
                        "type": "unsubscription_confirmed",
                        "topic": topic,
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
            
            elif message_type == "pong":
                # Update last ping time
                if connection_id in self.connection_manager.connection_metadata:
                    self.connection_manager.connection_metadata[connection_id]["last_ping"] = datetime.utcnow()
            
            else:
                logger.warning(
                    "Unknown message type received",
                    connection_id=connection_id,
                    message_type=message_type
                )
        
        except Exception as e:
            logger.error(
                "Error handling WebSocket message",
                connection_id=connection_id,
                error=str(e)
            )
    
    async def send_dashboard_update(self, data: DashboardData) -> None:
        """Send dashboard data update to subscribers"""
        message = {
            "type": "dashboard_update",
            "data": self._serialize_dashboard_data(data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.broadcast_to_topic(message, "dashboard")
    
    async def send_system_alert(self, alert_type: str, message: str, 
                                severity: str = "info") -> None:
        """Send system alert to all connections"""
        alert_message = {
            "type": "system_alert",
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.broadcast_to_all(alert_message)
    
    async def send_trading_update(self, update_type: str, data: Dict[str, Any]) -> None:
        """Send trading-specific updates"""
        message = {
            "type": "trading_update",
            "update_type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.broadcast_to_topic(message, "trading")
    
    def _serialize_dashboard_data(self, data: DashboardData) -> Dict[str, Any]:
        """Serialize dashboard data for JSON transmission"""
        try:
            return {
                "system_metrics": {
                    "status": data.system_metrics.status.value,
                    "uptime_seconds": data.system_metrics.uptime_seconds,
                    "cpu_usage_pct": data.system_metrics.cpu_usage_pct,
                    "memory_usage_mb": data.system_metrics.memory_usage_mb,
                    "memory_usage_pct": data.system_metrics.memory_usage_pct,
                    "active_connections": data.system_metrics.active_connections,
                    "requests_per_minute": data.system_metrics.requests_per_minute,
                    "error_rate_pct": data.system_metrics.error_rate_pct,
                    "response_time_ms": data.system_metrics.response_time_ms,
                    "last_updated": data.system_metrics.last_updated.isoformat(),
                    "component_statuses": {
                        "database": data.system_metrics.database_status.value,
                        "redis": data.system_metrics.redis_status.value,
                        "ml_models": data.system_metrics.ml_models_status.value,
                        "rl_agent": data.system_metrics.rl_agent_status.value,
                        "dex_connections": data.system_metrics.dex_connections_status.value
                    },
                    "performance": {
                        "total_requests": data.system_metrics.total_requests,
                        "total_errors": data.system_metrics.total_errors,
                        "cache_hit_rate_pct": data.system_metrics.cache_hit_rate_pct
                    }
                },
                "portfolio_status": {
                    "total_value_usd": str(data.portfolio_status.total_value_usd),
                    "available_balance_usd": str(data.portfolio_status.available_balance_usd),
                    "unrealized_pnl_usd": str(data.portfolio_status.unrealized_pnl_usd),
                    "realized_pnl_usd": str(data.portfolio_status.realized_pnl_usd),
                    "daily_pnl_usd": str(data.portfolio_status.daily_pnl_usd),
                    "daily_pnl_pct": str(data.portfolio_status.daily_pnl_pct),
                    "total_return_pct": str(data.portfolio_status.total_return_pct),
                    "position_count": data.portfolio_status.position_count,
                    "chain_balances": {k: str(v) for k, v in data.portfolio_status.chain_balances.items()},
                    "last_updated": data.portfolio_status.last_updated.isoformat()
                },
                "trading_status": {
                    "mode": data.trading_status.mode.value,
                    "is_trading_active": data.trading_status.is_trading_active,
                    "trades_today": data.trading_status.trades_today,
                    "volume_today_usd": str(data.trading_status.volume_today_usd),
                    "emergency_stop_active": data.trading_status.emergency_stop_active,
                    "win_rate_pct": data.trading_status.win_rate_pct,
                    "tokens_analyzed_today": data.trading_status.tokens_analyzed_today,
                    "high_confidence_signals": data.trading_status.high_confidence_signals,
                    "last_updated": data.trading_status.last_updated.isoformat()
                },
                "ml_rl_status": {
                    "ml_rl_integration_active": data.ml_rl_status.ml_rl_integration_active,
                    "ml_rl_decision_latency_ms": data.ml_rl_status.ml_rl_decision_latency_ms,
                    "ml_training_active": data.ml_rl_status.ml_training_active,
                    "rl_training_active": data.ml_rl_status.rl_training_active,
                    "ml_prediction_accuracy_pct": data.ml_rl_status.ml_prediction_accuracy_pct,
                    "rl_action_success_rate_pct": data.ml_rl_status.rl_action_success_rate_pct,
                    "last_updated": data.ml_rl_status.last_updated.isoformat()
                },
                "alerts": {
                    "active_alerts": data.active_alerts,
                    "warnings_count": data.warnings_count,
                    "errors_count": data.errors_count
                },
                "timestamp": data.timestamp.isoformat()
            }
        except Exception as e:
            logger.error("Error serializing dashboard data", error=str(e))
            raise DashboardError(f"Serialization error: {e}")
    
    async def _update_loop(self) -> None:
        """Background task to send periodic dashboard updates"""
        while self._running:
            try:
                # Get current dashboard data
                data = await dashboard_cache.get_data()
                
                # Send update to subscribers
                await self.send_dashboard_update(data)
                
                # Wait for next update
                await asyncio.sleep(self.dashboard_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in dashboard update loop", error=str(e))
                await asyncio.sleep(self.dashboard_update_interval)
    
    async def _ping_loop(self) -> None:
        """Background task to ping connections for keepalive"""
        while self._running:
            try:
                await self.connection_manager.ping_connections()
                await asyncio.sleep(self.ping_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in ping loop", error=str(e))
                await asyncio.sleep(self.ping_interval)


# Global WebSocket manager instance
websocket_manager = WebSocketManager()