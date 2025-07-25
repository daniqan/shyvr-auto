"""
Tests for dashboard WebSocket manager
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.dashboard.websocket_manager import (
    ConnectionManager, WebSocketManager, websocket_manager
)
from src.dashboard.base import DashboardData, DashboardError


class MockWebSocket:
    """Mock WebSocket for testing"""
    
    def __init__(self):
        self.state = "OPEN"
        self.sent_messages = []
        self.closed = False
    
    async def accept(self):
        """Mock accept method"""
        pass
    
    async def send_text(self, message):
        """Mock send_text method"""
        if self.closed:
            raise ConnectionError("WebSocket closed")
        self.sent_messages.append(message)
    
    async def close(self):
        """Mock close method"""
        self.closed = True
        self.state = "CLOSED"


class TestConnectionManager:
    """Test suite for ConnectionManager"""
    
    @pytest.fixture
    def connection_manager(self):
        """Connection manager instance for testing"""
        return ConnectionManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket for testing"""
        return MockWebSocket()
    
    @pytest.mark.asyncio
    async def test_connection_manager_initialization(self, connection_manager):
        """Test ConnectionManager initialization"""
        assert len(connection_manager.active_connections) == 0
        assert len(connection_manager.subscriptions) == 0
        assert len(connection_manager.connection_metadata) == 0
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, connection_manager, mock_websocket):
        """Test WebSocket connection"""
        connection_id = "test-connection-1"
        user_id = "user-123"
        
        await connection_manager.connect(mock_websocket, connection_id, user_id)
        
        # Should add connection
        assert connection_id in connection_manager.active_connections
        assert connection_manager.active_connections[connection_id] == mock_websocket
        
        # Should add metadata
        assert connection_id in connection_manager.connection_metadata
        metadata = connection_manager.connection_metadata[connection_id]
        assert metadata["user_id"] == user_id
        assert isinstance(metadata["connected_at"], datetime)
        assert isinstance(metadata["last_ping"], datetime)
        assert isinstance(metadata["subscriptions"], set)
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, connection_manager, mock_websocket):
        """Test WebSocket disconnection"""
        connection_id = "test-connection-1"
        
        # Connect first
        await connection_manager.connect(mock_websocket, connection_id)
        assert connection_id in connection_manager.active_connections
        
        # Disconnect
        await connection_manager.disconnect(connection_id)
        assert connection_id not in connection_manager.active_connections
        assert connection_id not in connection_manager.connection_metadata
    
    @pytest.mark.asyncio
    async def test_subscribe_to_topic(self, connection_manager, mock_websocket):
        """Test subscribing to topics"""
        connection_id = "test-connection-1"
        topic = "dashboard"
        
        # Connect first
        await connection_manager.connect(mock_websocket, connection_id)
        
        # Subscribe to topic
        await connection_manager.subscribe(connection_id, topic)
        
        # Should add to subscriptions
        assert topic in connection_manager.subscriptions
        assert connection_id in connection_manager.subscriptions[topic]
        
        # Should update metadata
        metadata = connection_manager.connection_metadata[connection_id]
        assert topic in metadata["subscriptions"]
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_topic(self, connection_manager, mock_websocket):
        """Test unsubscribing from topics"""
        connection_id = "test-connection-1"
        topic = "dashboard"
        
        # Connect and subscribe first
        await connection_manager.connect(mock_websocket, connection_id)
        await connection_manager.subscribe(connection_id, topic)
        
        # Unsubscribe
        await connection_manager.unsubscribe(connection_id, topic)
        
        # Should remove from subscriptions
        assert topic not in connection_manager.subscriptions or \
               connection_id not in connection_manager.subscriptions[topic]
        
        # Should update metadata
        metadata = connection_manager.connection_metadata[connection_id]
        assert topic not in metadata["subscriptions"]
    
    @pytest.mark.asyncio
    async def test_send_personal_message(self, connection_manager, mock_websocket):
        """Test sending personal message"""
        connection_id = "test-connection-1"
        message = {"type": "test", "data": "hello"}
        
        # Connect first
        await connection_manager.connect(mock_websocket, connection_id)
        
        # Send message
        await connection_manager.send_personal_message(message, connection_id)
        
        # Should send message
        assert len(mock_websocket.sent_messages) == 1
        sent_message = json.loads(mock_websocket.sent_messages[0])
        assert sent_message["type"] == "test"
        assert sent_message["data"] == "hello"
    
    @pytest.mark.asyncio
    async def test_send_personal_message_to_nonexistent_connection(self, connection_manager):
        """Test sending message to non-existent connection"""
        connection_id = "nonexistent"
        message = {"type": "test"}
        
        # Should not raise error, just log warning
        await connection_manager.send_personal_message(message, connection_id)
    
    @pytest.mark.asyncio
    async def test_broadcast_to_topic(self, connection_manager):
        """Test broadcasting to topic subscribers"""
        # Setup multiple connections
        connection1 = MockWebSocket()
        connection2 = MockWebSocket()
        connection3 = MockWebSocket()
        
        await connection_manager.connect(connection1, "conn1")
        await connection_manager.connect(connection2, "conn2")
        await connection_manager.connect(connection3, "conn3")
        
        # Subscribe connections to different topics
        await connection_manager.subscribe("conn1", "dashboard")
        await connection_manager.subscribe("conn2", "dashboard")
        await connection_manager.subscribe("conn3", "trading")
        
        # Broadcast to dashboard topic
        message = {"type": "dashboard_update", "data": "test"}
        await connection_manager.broadcast_to_topic(message, "dashboard")
        
        # Should send to dashboard subscribers only
        assert len(connection1.sent_messages) == 1
        assert len(connection2.sent_messages) == 1
        assert len(connection3.sent_messages) == 0
        
        # Check message content
        sent_message = json.loads(connection1.sent_messages[0])
        assert sent_message["type"] == "dashboard_update"
    
    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, connection_manager):
        """Test broadcasting to all connections"""
        # Setup multiple connections
        connection1 = MockWebSocket()
        connection2 = MockWebSocket()
        
        await connection_manager.connect(connection1, "conn1")
        await connection_manager.connect(connection2, "conn2")
        
        # Broadcast to all
        message = {"type": "system_alert", "message": "test"}
        await connection_manager.broadcast_to_all(message)
        
        # Should send to all connections
        assert len(connection1.sent_messages) == 1
        assert len(connection2.sent_messages) == 1
        
        # Check message content
        sent_message = json.loads(connection1.sent_messages[0])
        assert sent_message["type"] == "system_alert"
    
    @pytest.mark.asyncio
    async def test_ping_connections(self, connection_manager):
        """Test pinging all connections"""
        # Setup connections
        connection1 = MockWebSocket()
        connection2 = MockWebSocket()
        
        await connection_manager.connect(connection1, "conn1")
        await connection_manager.connect(connection2, "conn2")
        
        # Ping all connections
        await connection_manager.ping_connections()
        
        # Should send ping to all connections
        assert len(connection1.sent_messages) == 1
        assert len(connection2.sent_messages) == 1
        
        # Check ping message
        ping_message = json.loads(connection1.sent_messages[0])
        assert ping_message["type"] == "ping"
        assert "timestamp" in ping_message
    
    @pytest.mark.asyncio
    async def test_get_connection_count(self, connection_manager):
        """Test getting connection count"""
        assert await connection_manager.get_connection_count() == 0
        
        # Add connections
        await connection_manager.connect(MockWebSocket(), "conn1")
        await connection_manager.connect(MockWebSocket(), "conn2")
        
        assert await connection_manager.get_connection_count() == 2
    
    @pytest.mark.asyncio
    async def test_get_topic_subscribers(self, connection_manager):
        """Test getting topic subscriber count"""
        topic = "dashboard"
        assert await connection_manager.get_topic_subscribers(topic) == 0
        
        # Add subscribers
        await connection_manager.connect(MockWebSocket(), "conn1")
        await connection_manager.connect(MockWebSocket(), "conn2")
        await connection_manager.subscribe("conn1", topic)
        await connection_manager.subscribe("conn2", topic)
        
        assert await connection_manager.get_topic_subscribers(topic) == 2


class TestWebSocketManager:
    """Test suite for WebSocketManager"""
    
    @pytest.fixture
    def websocket_manager(self):
        """WebSocket manager instance for testing"""
        return WebSocketManager()
    
    @pytest.mark.asyncio
    async def test_websocket_manager_initialization(self, websocket_manager):
        """Test WebSocketManager initialization"""
        assert websocket_manager.connection_manager is not None
        assert websocket_manager._running is False
        assert websocket_manager._update_task is None
        assert websocket_manager._ping_task is None
        assert websocket_manager.dashboard_update_interval == 2.0
        assert websocket_manager.ping_interval == 30.0
    
    @pytest.mark.asyncio
    async def test_start_websocket_manager(self, websocket_manager):
        """Test starting WebSocket manager"""
        await websocket_manager.start()
        
        assert websocket_manager._running is True
        assert websocket_manager._update_task is not None
        assert websocket_manager._ping_task is not None
        
        # Cleanup
        await websocket_manager.stop()
    
    @pytest.mark.asyncio
    async def test_stop_websocket_manager(self, websocket_manager):
        """Test stopping WebSocket manager"""
        # Start first
        await websocket_manager.start()
        assert websocket_manager._running is True
        
        # Stop
        await websocket_manager.stop()
        assert websocket_manager._running is False
    
    @pytest.mark.asyncio
    async def test_connect_websocket_client(self, websocket_manager):
        """Test connecting WebSocket client"""
        mock_websocket = MockWebSocket()
        connection_id = "test-connection"
        user_id = "user-123"
        
        await websocket_manager.connect(mock_websocket, connection_id, user_id)
        
        # Should delegate to connection manager
        assert connection_id in websocket_manager.connection_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket_client(self, websocket_manager):
        """Test disconnecting WebSocket client"""
        mock_websocket = MockWebSocket()
        connection_id = "test-connection"
        
        # Connect first
        await websocket_manager.connect(mock_websocket, connection_id)
        assert connection_id in websocket_manager.connection_manager.active_connections
        
        # Disconnect
        await websocket_manager.disconnect(connection_id)
        assert connection_id not in websocket_manager.connection_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, websocket_manager):
        """Test handling subscribe message"""
        mock_websocket = MockWebSocket()
        connection_id = "test-connection"
        
        # Connect first
        await websocket_manager.connect(mock_websocket, connection_id)
        
        # Handle subscribe message
        message = {"type": "subscribe", "topic": "dashboard"}
        await websocket_manager.handle_message(connection_id, message)
        
        # Should subscribe to topic
        assert "dashboard" in websocket_manager.connection_manager.subscriptions
        assert connection_id in websocket_manager.connection_manager.subscriptions["dashboard"]
        
        # Should send confirmation
        assert len(mock_websocket.sent_messages) == 1
        confirmation = json.loads(mock_websocket.sent_messages[0])
        assert confirmation["type"] == "subscription_confirmed"
        assert confirmation["topic"] == "dashboard"
    
    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self, websocket_manager):
        """Test handling unsubscribe message"""
        mock_websocket = MockWebSocket()
        connection_id = "test-connection"
        
        # Connect and subscribe first
        await websocket_manager.connect(mock_websocket, connection_id)
        await websocket_manager.connection_manager.subscribe(connection_id, "dashboard")
        
        # Handle unsubscribe message
        message = {"type": "unsubscribe", "topic": "dashboard"}
        await websocket_manager.handle_message(connection_id, message)
        
        # Should unsubscribe from topic
        if "dashboard" in websocket_manager.connection_manager.subscriptions:
            assert connection_id not in websocket_manager.connection_manager.subscriptions["dashboard"]
        
        # Should send confirmation
        assert len(mock_websocket.sent_messages) == 1
        confirmation = json.loads(mock_websocket.sent_messages[0])
        assert confirmation["type"] == "unsubscription_confirmed"
    
    @pytest.mark.asyncio
    async def test_handle_pong_message(self, websocket_manager):
        """Test handling pong message"""
        mock_websocket = MockWebSocket()
        connection_id = "test-connection"
        
        # Connect first
        await websocket_manager.connect(mock_websocket, connection_id)
        
        # Handle pong message
        message = {"type": "pong"}
        await websocket_manager.handle_message(connection_id, message)
        
        # Should update last ping time
        metadata = websocket_manager.connection_manager.connection_metadata[connection_id]
        assert isinstance(metadata["last_ping"], datetime)
    
    @pytest.mark.asyncio
    async def test_handle_unknown_message(self, websocket_manager):
        """Test handling unknown message type"""
        mock_websocket = MockWebSocket()
        connection_id = "test-connection"
        
        # Connect first
        await websocket_manager.connect(mock_websocket, connection_id)
        
        # Handle unknown message - should not raise error
        message = {"type": "unknown_type", "data": "test"}
        await websocket_manager.handle_message(connection_id, message)
        
        # Should not send any response
        assert len(mock_websocket.sent_messages) == 0
    
    @pytest.mark.asyncio
    async def test_send_dashboard_update(self, websocket_manager):
        """Test sending dashboard update"""
        mock_websocket = MockWebSocket()
        connection_id = "test-connection"
        
        # Connect and subscribe to dashboard topic
        await websocket_manager.connect(mock_websocket, connection_id)
        await websocket_manager.connection_manager.subscribe(connection_id, "dashboard")
        
        # Send dashboard update
        dashboard_data = DashboardData.create_default()
        dashboard_data.active_alerts = 5
        
        await websocket_manager.send_dashboard_update(dashboard_data)
        
        # Should send update to subscriber
        assert len(mock_websocket.sent_messages) == 1
        update_message = json.loads(mock_websocket.sent_messages[0])
        assert update_message["type"] == "dashboard_update"
        assert "data" in update_message
        assert "timestamp" in update_message
    
    @pytest.mark.asyncio
    async def test_send_system_alert(self, websocket_manager):
        """Test sending system alert"""
        mock_websocket = MockWebSocket()
        connection_id = "test-connection"
        
        # Connect
        await websocket_manager.connect(mock_websocket, connection_id)
        
        # Send system alert
        await websocket_manager.send_system_alert("error", "Test error message", "high")
        
        # Should send alert to all connections
        assert len(mock_websocket.sent_messages) == 1
        alert_message = json.loads(mock_websocket.sent_messages[0])
        assert alert_message["type"] == "system_alert"
        assert alert_message["alert_type"] == "error"
        assert alert_message["message"] == "Test error message"
        assert alert_message["severity"] == "high"
    
    @pytest.mark.asyncio
    async def test_send_trading_update(self, websocket_manager):
        """Test sending trading update"""
        mock_websocket = MockWebSocket()
        connection_id = "test-connection"
        
        # Connect and subscribe to trading topic
        await websocket_manager.connect(mock_websocket, connection_id)
        await websocket_manager.connection_manager.subscribe(connection_id, "trading")
        
        # Send trading update
        update_data = {"symbol": "BTC/USDC", "action": "buy", "amount": 1000}
        await websocket_manager.send_trading_update("trade_executed", update_data)
        
        # Should send update to trading subscribers
        assert len(mock_websocket.sent_messages) == 1
        trading_message = json.loads(mock_websocket.sent_messages[0])
        assert trading_message["type"] == "trading_update"
        assert trading_message["update_type"] == "trade_executed"
        assert trading_message["data"]["symbol"] == "BTC/USDC"
    
    @pytest.mark.asyncio
    async def test_serialize_dashboard_data(self, websocket_manager):
        """Test dashboard data serialization"""
        # Create test data
        dashboard_data = DashboardData.create_default()
        dashboard_data.active_alerts = 3
        dashboard_data.portfolio_status.total_value_usd = 50000
        
        # Serialize
        serialized = websocket_manager._serialize_dashboard_data(dashboard_data)
        
        # Should contain all expected fields
        assert "system_metrics" in serialized
        assert "portfolio_status" in serialized
        assert "trading_status" in serialized
        assert "ml_rl_status" in serialized
        assert "alerts" in serialized
        assert "timestamp" in serialized
        
        # Check specific values
        assert serialized["alerts"]["active_alerts"] == 3
        assert serialized["portfolio_status"]["total_value_usd"] == "50000"
    
    @pytest.mark.asyncio
    async def test_serialize_dashboard_data_error(self, websocket_manager):
        """Test dashboard data serialization error handling"""
        # Create invalid data that would cause serialization error
        invalid_data = MagicMock()
        invalid_data.system_metrics = MagicMock()
        invalid_data.system_metrics.status = MagicMock()
        invalid_data.system_metrics.status.value = MagicMock()
        invalid_data.system_metrics.status.value.side_effect = Exception("Serialization error")
        
        # Should raise DashboardError
        with pytest.raises(DashboardError):
            websocket_manager._serialize_dashboard_data(invalid_data)


class TestWebSocketManagerIntegration:
    """Integration tests for WebSocket manager"""
    
    @pytest.mark.asyncio
    async def test_full_websocket_flow(self):
        """Test complete WebSocket communication flow"""
        manager = WebSocketManager()
        
        # Start manager
        await manager.start()
        
        try:
            # Connect multiple clients
            client1 = MockWebSocket()
            client2 = MockWebSocket()
            
            await manager.connect(client1, "client1", "user1")
            await manager.connect(client2, "client2", "user2")
            
            # Subscribe to different topics
            await manager.handle_message("client1", {"type": "subscribe", "topic": "dashboard"})
            await manager.handle_message("client2", {"type": "subscribe", "topic": "trading"})
            
            # Send updates
            dashboard_data = DashboardData.create_default()
            await manager.send_dashboard_update(dashboard_data)
            
            trading_data = {"symbol": "ETH/USDC", "action": "sell"}
            await manager.send_trading_update("trade_executed", trading_data)
            
            # Send system alert
            await manager.send_system_alert("warning", "System warning", "medium")
            
            # Check messages received
            # Client1: subscription confirmation + dashboard update + system alert
            assert len(client1.sent_messages) == 3
            
            # Client2: subscription confirmation + trading update + system alert
            assert len(client2.sent_messages) == 3
            
            # Verify message types
            client1_messages = [json.loads(msg) for msg in client1.sent_messages]
            client2_messages = [json.loads(msg) for msg in client2.sent_messages]
            
            # Client1 should receive dashboard update
            dashboard_update = next(msg for msg in client1_messages if msg["type"] == "dashboard_update")
            assert dashboard_update is not None
            
            # Client2 should receive trading update
            trading_update = next(msg for msg in client2_messages if msg["type"] == "trading_update")
            assert trading_update["update_type"] == "trade_executed"
            
            # Both should receive system alert
            client1_alert = next(msg for msg in client1_messages if msg["type"] == "system_alert")
            client2_alert = next(msg for msg in client2_messages if msg["type"] == "system_alert")
            assert client1_alert["message"] == "System warning"
            assert client2_alert["message"] == "System warning"
            
        finally:
            # Cleanup
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self):
        """Test WebSocket error handling"""
        manager = WebSocketManager()
        
        # Test with faulty websocket
        faulty_websocket = MockWebSocket()
        faulty_websocket.closed = True  # Simulate closed connection
        
        await manager.connect(faulty_websocket, "faulty")
        
        # Subscribe should work
        await manager.handle_message("faulty", {"type": "subscribe", "topic": "dashboard"})
        
        # Sending updates should handle the error gracefully
        dashboard_data = DashboardData.create_default()
        await manager.send_dashboard_update(dashboard_data)
        
        # Connection should be cleaned up
        assert "faulty" not in manager.connection_manager.active_connections


# Test global websocket_manager instance
class TestGlobalWebSocketManager:
    """Test global websocket_manager instance"""
    
    def test_global_instance_exists(self):
        """Test that global websocket_manager instance exists"""
        assert websocket_manager is not None
        assert isinstance(websocket_manager, WebSocketManager)
    
    @pytest.mark.asyncio
    async def test_global_instance_functionality(self):
        """Test that global instance works correctly"""
        # Should be able to get connection count
        count = await websocket_manager.connection_manager.get_connection_count()
        assert isinstance(count, int)
        assert count >= 0