"""
Tests for Emergency Response Automation System - Phase 3.2.4

This module tests comprehensive emergency response automation including:
- Automated emergency procedures
- Multi-channel notifications (email, SMS, Slack)
- Position liquidation automation
- Graceful shutdown sequences
- Recovery procedures

Following TDD methodology - tests drive implementation.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4, UUID
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from src.safety.emergency_response_automation import (
    EmergencyResponseSystem,
    EmergencyEvent,
    EmergencyEventType,
    EmergencyEventSeverity,
    EmergencyResponse,
    EmergencyResponseType,
    EmergencyResponseStatus,
    NotificationChannel,
    NotificationMessage,
    PositionLiquidationManager,
    EmergencyShutdownManager,
    NotificationManager,
    EmergencyResponseConfig,
    EmergencyResponseError,
    EmergencyResponsePlan,
    EmergencyResponseExecution,
    EmergencyContactManager,
    EmergencyEscalationManager
)


class TestEmergencyResponseSystem:
    """Test suite for emergency response system."""

    @pytest.fixture
    def mock_portfolio(self):
        """Mock portfolio for testing."""
        portfolio = Mock()
        portfolio.get_all_positions.return_value = {
            'BTC-USD': {'size': Decimal('1.5'), 'side': 'long'},
            'ETH-USD': {'size': Decimal('10.0'), 'side': 'short'}
        }
        portfolio.get_total_balance.return_value = Decimal('100000.00')
        portfolio.close_position = AsyncMock(return_value=True)
        portfolio.close_all_positions = AsyncMock(return_value=True)
        return portfolio

    @pytest.fixture
    def mock_integrations(self):
        """Mock external integrations."""
        return {
            'safety_state_manager': Mock(),
            'circuit_breaker': Mock(),
            'emergency_controller': Mock(),
            'monitoring_system': Mock(),
            'trading_client': Mock()
        }

    @pytest.fixture
    def emergency_config(self):
        """Create emergency response configuration."""
        return EmergencyResponseConfig(
            enable_automated_response=True,
            enable_position_liquidation=True,
            enable_emergency_shutdown=True,
            enable_notifications=True,
            notification_channels=['email', 'sms', 'slack', 'webhook'],
            escalation_enabled=True,
            escalation_timeout_minutes=15,
            position_liquidation_timeout=30,
            emergency_shutdown_timeout=60,
            max_concurrent_responses=5,
            response_retry_attempts=3,
            notification_retry_attempts=2,
            contact_groups=['primary', 'secondary', 'escalation'],
            enable_audit_logging=True
        )

    @pytest.fixture
    def emergency_system(self, emergency_config, mock_portfolio, mock_integrations):
        """Create emergency response system for testing."""
        return EmergencyResponseSystem(
            config=emergency_config,
            portfolio=mock_portfolio,
            integrations=mock_integrations
        )

    @pytest.mark.asyncio
    async def test_system_initialization(self, emergency_system):
        """Test emergency response system initialization."""
        await emergency_system.initialize()
        
        assert emergency_system.is_initialized
        assert emergency_system.notification_manager is not None
        assert emergency_system.liquidation_manager is not None
        assert emergency_system.shutdown_manager is not None
        assert emergency_system.contact_manager is not None
        assert emergency_system.escalation_manager is not None

    @pytest.mark.asyncio
    async def test_handle_critical_emergency_event(self, emergency_system):
        """Test handling of critical emergency event."""
        await emergency_system.initialize()
        
        # Create critical emergency event
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.SYSTEM_FAILURE,
            severity=EmergencyEventSeverity.CRITICAL,
            title="Critical System Failure",
            description="Database connection lost",
            affected_components=['database', 'trading_engine'],
            estimated_impact="High trading risk",
            requires_immediate_action=True
        )
        
        # Mock successful response execution
        emergency_system.liquidation_manager.execute_emergency_liquidation = AsyncMock(
            return_value={'success': True, 'liquidated_positions': 2}
        )
        emergency_system.notification_manager.send_emergency_notifications = AsyncMock(
            return_value={'sent': 4, 'failed': 0}
        )
        
        response = await emergency_system.handle_emergency_event(emergency_event)
        
        assert response.event_id == emergency_event.event_id
        assert response.response_type == EmergencyResponseType.FULL_RESPONSE
        assert response.status == EmergencyResponseStatus.COMPLETED
        assert response.liquidation_executed is True
        assert response.notifications_sent > 0

    @pytest.mark.asyncio
    async def test_handle_market_anomaly_event(self, emergency_system):
        """Test handling of market anomaly event."""
        await emergency_system.initialize()
        
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.MARKET_ANOMALY,
            severity=EmergencyEventSeverity.HIGH,
            title="Unusual Market Activity",
            description="Extreme volatility detected",
            market_data={'volatility': 0.8, 'price_change': -0.15},
            requires_position_review=True
        )
        
        # Mock position analysis
        emergency_system.liquidation_manager.analyze_position_risk = AsyncMock(
            return_value={'high_risk_positions': ['BTC-USD'], 'liquidation_recommended': True}
        )
        emergency_system.liquidation_manager.execute_selective_liquidation = AsyncMock(
            return_value={'success': True, 'liquidated_positions': 1}
        )
        
        response = await emergency_system.handle_emergency_event(emergency_event)
        
        assert response.response_type == EmergencyResponseType.SELECTIVE_LIQUIDATION
        assert response.status == EmergencyResponseStatus.COMPLETED
        assert 'position_analysis' in response.execution_details

    @pytest.mark.asyncio
    async def test_handle_security_breach_event(self, emergency_system):
        """Test handling of security breach event."""
        await emergency_system.initialize()
        
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.SECURITY_BREACH,
            severity=EmergencyEventSeverity.CRITICAL,
            title="Security Breach Detected",
            description="Unauthorized access attempt",
            security_details={'source_ip': '192.168.1.100', 'attack_type': 'brute_force'},
            requires_immediate_shutdown=True
        )
        
        # Mock emergency shutdown
        emergency_system.shutdown_manager.execute_emergency_shutdown = AsyncMock(
            return_value={'success': True, 'shutdown_time': 5.2}
        )
        
        response = await emergency_system.handle_emergency_event(emergency_event)
        
        assert response.response_type == EmergencyResponseType.EMERGENCY_SHUTDOWN
        assert response.status == EmergencyResponseStatus.COMPLETED
        assert response.emergency_shutdown_executed is True

    @pytest.mark.asyncio
    async def test_escalation_on_response_failure(self, emergency_system):
        """Test escalation when emergency response fails."""
        await emergency_system.initialize()
        
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.SYSTEM_FAILURE,
            severity=EmergencyEventSeverity.HIGH,
            title="System Component Failure",
            description="Trading engine unresponsive"
        )
        
        # Mock response failure
        emergency_system.liquidation_manager.execute_emergency_liquidation = AsyncMock(
            side_effect=Exception("Liquidation failed")
        )
        emergency_system.escalation_manager.escalate_emergency = AsyncMock(
            return_value={'escalated': True, 'escalation_level': 2}
        )
        
        response = await emergency_system.handle_emergency_event(emergency_event)
        
        assert response.status == EmergencyResponseStatus.FAILED
        assert response.escalation_triggered is True
        assert 'error' in response.execution_details

    @pytest.mark.asyncio
    async def test_multiple_concurrent_emergencies(self, emergency_system):
        """Test handling multiple concurrent emergency events."""
        await emergency_system.initialize()
        
        # Create multiple emergency events
        events = []
        for i in range(3):
            event = EmergencyEvent(
                event_id=uuid4(),
                event_type=EmergencyEventType.MARKET_ANOMALY,
                severity=EmergencyEventSeverity.MEDIUM,
                title=f"Market Event {i}",
                description=f"Market anomaly {i}"
            )
            events.append(event)
        
        # Mock successful responses
        emergency_system.liquidation_manager.analyze_position_risk = AsyncMock(
            return_value={'high_risk_positions': [], 'liquidation_recommended': False}
        )
        emergency_system.notification_manager.send_emergency_notifications = AsyncMock(
            return_value={'sent': 2, 'failed': 0}
        )
        
        # Handle events concurrently
        tasks = [emergency_system.handle_emergency_event(event) for event in events]
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 3
        assert all(response.status in [EmergencyResponseStatus.COMPLETED, EmergencyResponseStatus.PARTIAL] 
                  for response in responses)

    @pytest.mark.asyncio
    async def test_emergency_response_timeout(self, emergency_system):
        """Test emergency response timeout handling."""
        await emergency_system.initialize()
        
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.SYSTEM_FAILURE,
            severity=EmergencyEventSeverity.CRITICAL,
            title="Slow Response Test"
        )
        
        # Mock slow response
        async def slow_liquidation():
            await asyncio.sleep(2)  # Longer than timeout
            return {'success': True}
        
        emergency_system.liquidation_manager.execute_emergency_liquidation = slow_liquidation
        emergency_system.config.position_liquidation_timeout = 1  # 1 second timeout
        
        response = await emergency_system.handle_emergency_event(emergency_event)
        
        assert response.status == EmergencyResponseStatus.TIMEOUT
        assert 'timeout' in response.execution_details.get('error', '').lower()

    @pytest.mark.asyncio
    async def test_get_emergency_status(self, emergency_system):
        """Test retrieving emergency response system status."""
        await emergency_system.initialize()
        
        status = await emergency_system.get_emergency_status()
        
        assert 'system_ready' in status
        assert 'active_emergencies' in status
        assert 'recent_responses' in status
        assert 'system_health' in status
        assert status['system_ready'] is True


class TestPositionLiquidationManager:
    """Test suite for position liquidation manager."""

    @pytest.fixture
    def mock_portfolio(self):
        """Mock portfolio for testing."""
        portfolio = Mock()
        portfolio.get_all_positions.return_value = {
            'BTC-USD': {'size': Decimal('1.0'), 'side': 'long', 'unrealized_pnl': Decimal('-500')},
            'ETH-USD': {'size': Decimal('5.0'), 'side': 'short', 'unrealized_pnl': Decimal('200')}
        }
        portfolio.close_position = AsyncMock(return_value=True)
        portfolio.close_all_positions = AsyncMock(return_value=True)
        portfolio.get_position_risk = Mock(return_value={'risk_score': 0.7})
        return portfolio

    @pytest.fixture
    def liquidation_manager(self, mock_portfolio):
        """Create position liquidation manager for testing."""
        return PositionLiquidationManager(portfolio=mock_portfolio)

    @pytest.mark.asyncio
    async def test_analyze_position_risk(self, liquidation_manager):
        """Test position risk analysis."""
        analysis = await liquidation_manager.analyze_position_risk()
        
        assert 'high_risk_positions' in analysis
        assert 'liquidation_recommended' in analysis
        assert 'total_exposure' in analysis
        assert isinstance(analysis['high_risk_positions'], list)

    @pytest.mark.asyncio
    async def test_execute_emergency_liquidation(self, liquidation_manager):
        """Test emergency liquidation execution."""
        result = await liquidation_manager.execute_emergency_liquidation()
        
        assert result['success'] is True
        assert 'liquidated_positions' in result
        assert 'execution_time' in result
        assert result['liquidated_positions'] >= 0

    @pytest.mark.asyncio
    async def test_execute_selective_liquidation(self, liquidation_manager):
        """Test selective position liquidation."""
        high_risk_positions = ['BTC-USD']
        
        result = await liquidation_manager.execute_selective_liquidation(high_risk_positions)
        
        assert result['success'] is True
        assert 'liquidated_positions' in result
        assert result['liquidated_positions'] == len(high_risk_positions)

    @pytest.mark.asyncio
    async def test_liquidation_with_portfolio_error(self, liquidation_manager):
        """Test liquidation handling when portfolio operations fail."""
        liquidation_manager.portfolio.close_all_positions = AsyncMock(
            side_effect=Exception("Portfolio error")
        )
        
        result = await liquidation_manager.execute_emergency_liquidation()
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Portfolio error' in result['error']


class TestNotificationManager:
    """Test suite for notification manager."""

    @pytest.fixture
    def notification_config(self):
        """Create notification configuration."""
        return {
            'channels': ['email', 'sms', 'slack'],
            'email_settings': {'smtp_server': 'smtp.test.com'},
            'sms_settings': {'provider': 'twilio'},
            'slack_settings': {'webhook_url': 'https://hooks.slack.com/test'}
        }

    @pytest.fixture
    def notification_manager(self, notification_config):
        """Create notification manager for testing."""
        return NotificationManager(config=notification_config)

    @pytest.mark.asyncio
    async def test_send_emergency_notifications(self, notification_manager):
        """Test sending emergency notifications."""
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.SYSTEM_FAILURE,
            severity=EmergencyEventSeverity.CRITICAL,
            title="Test Emergency",
            description="Test emergency notification"
        )
        
        # Mock notification senders
        notification_manager._send_email_notification = AsyncMock(return_value=True)
        notification_manager._send_sms_notification = AsyncMock(return_value=True)
        notification_manager._send_slack_notification = AsyncMock(return_value=True)
        
        result = await notification_manager.send_emergency_notifications(emergency_event)
        
        assert result['sent'] >= 0
        assert result['failed'] >= 0
        assert result['sent'] + result['failed'] > 0

    @pytest.mark.asyncio
    async def test_notification_retry_on_failure(self, notification_manager):
        """Test notification retry mechanism."""
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.MARKET_ANOMALY,
            severity=EmergencyEventSeverity.HIGH,
            title="Test Retry",
            description="Test notification retry"
        )
        
        # Mock notification failure then success
        call_count = 0
        async def flaky_send():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network error")
            return True
        
        notification_manager._send_email_notification = flaky_send
        notification_manager._send_sms_notification = AsyncMock(return_value=True)
        
        result = await notification_manager.send_emergency_notifications(emergency_event)
        
        assert call_count >= 2  # Original attempt + retry
        assert result['sent'] > 0

    @pytest.mark.asyncio
    async def test_format_emergency_message(self, notification_manager):
        """Test emergency message formatting."""
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.SECURITY_BREACH,
            severity=EmergencyEventSeverity.CRITICAL,
            title="Security Alert",
            description="Unauthorized access detected",
            security_details={'source_ip': '192.168.1.100'}
        )
        
        message = notification_manager.format_emergency_message(emergency_event)
        
        assert message.subject is not None
        assert message.body is not None
        assert emergency_event.title in message.subject
        assert emergency_event.description in message.body
        assert str(emergency_event.severity.value).upper() in message.body


class TestEmergencyShutdownManager:
    """Test suite for emergency shutdown manager."""

    @pytest.fixture
    def mock_integrations(self):
        """Mock system integrations."""
        return {
            'trading_client': Mock(),
            'database': Mock(),
            'monitoring_system': Mock(),
            'safety_manager': Mock()
        }

    @pytest.fixture
    def shutdown_manager(self, mock_integrations):
        """Create emergency shutdown manager for testing."""
        return EmergencyShutdownManager(integrations=mock_integrations)

    @pytest.mark.asyncio
    async def test_execute_emergency_shutdown(self, shutdown_manager):
        """Test emergency shutdown execution."""
        # Mock successful shutdown of all components
        for component in shutdown_manager.integrations.values():
            component.shutdown = AsyncMock(return_value=True)
        
        result = await shutdown_manager.execute_emergency_shutdown()
        
        assert result['success'] is True
        assert 'shutdown_time' in result
        assert 'components_shutdown' in result
        assert result['components_shutdown'] > 0

    @pytest.mark.asyncio
    async def test_graceful_shutdown_sequence(self, shutdown_manager):
        """Test graceful shutdown sequence."""
        # Mock graceful shutdown methods
        shutdown_manager._graceful_shutdown_trading = AsyncMock(return_value=True)
        shutdown_manager._graceful_shutdown_database = AsyncMock(return_value=True)
        shutdown_manager._graceful_shutdown_monitoring = AsyncMock(return_value=True)
        
        result = await shutdown_manager.execute_graceful_shutdown()
        
        assert result['success'] is True
        assert result['shutdown_type'] == 'graceful'

    @pytest.mark.asyncio
    async def test_forced_shutdown_on_graceful_failure(self, shutdown_manager):
        """Test forced shutdown when graceful shutdown fails."""
        # Mock graceful shutdown failure
        shutdown_manager._graceful_shutdown_trading = AsyncMock(
            side_effect=Exception("Graceful shutdown failed")
        )
        shutdown_manager._force_shutdown_all = AsyncMock(return_value=True)
        
        result = await shutdown_manager.execute_emergency_shutdown()
        
        assert result['success'] is True
        assert result.get('shutdown_type') == 'forced'


class TestEmergencyContactManager:
    """Test suite for emergency contact manager."""

    @pytest.fixture
    def contact_config(self):
        """Create contact configuration."""
        return {
            'contact_groups': {
                'primary': [
                    {'name': 'John Doe', 'email': 'john@test.com', 'phone': '+1234567890'},
                    {'name': 'Jane Smith', 'email': 'jane@test.com', 'phone': '+1234567891'}
                ],
                'secondary': [
                    {'name': 'Bob Wilson', 'email': 'bob@test.com', 'phone': '+1234567892'}
                ]
            }
        }

    @pytest.fixture
    def contact_manager(self, contact_config):
        """Create emergency contact manager for testing."""
        return EmergencyContactManager(config=contact_config)

    def test_get_contacts_for_severity(self, contact_manager):
        """Test getting contacts based on emergency severity."""
        # Critical emergencies should get primary contacts
        critical_contacts = contact_manager.get_contacts_for_severity(
            EmergencyEventSeverity.CRITICAL
        )
        assert len(critical_contacts) >= 2
        
        # Medium emergencies should get fewer contacts
        medium_contacts = contact_manager.get_contacts_for_severity(
            EmergencyEventSeverity.MEDIUM
        )
        assert len(medium_contacts) >= 1

    def test_get_escalation_contacts(self, contact_manager):
        """Test getting escalation contacts."""
        escalation_contacts = contact_manager.get_escalation_contacts()
        assert isinstance(escalation_contacts, list)
        # Should include contacts from multiple groups for escalation


class TestEmergencyEscalationManager:
    """Test suite for emergency escalation manager."""

    @pytest.fixture
    def escalation_manager(self):
        """Create emergency escalation manager for testing."""
        return EmergencyEscalationManager()

    @pytest.mark.asyncio
    async def test_escalate_emergency(self, escalation_manager):
        """Test emergency escalation."""
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.SYSTEM_FAILURE,
            severity=EmergencyEventSeverity.HIGH,
            title="System Failure",
            description="Critical component failure"
        )
        
        result = await escalation_manager.escalate_emergency(emergency_event)
        
        assert result['escalated'] is True
        assert 'escalation_level' in result
        assert result['escalation_level'] > 0

    @pytest.mark.asyncio
    async def test_escalation_timeout_handling(self, escalation_manager):
        """Test escalation timeout handling."""
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.MARKET_ANOMALY,
            severity=EmergencyEventSeverity.MEDIUM,
            title="Market Event"
        )
        
        # Simulate escalation timeout
        escalation_manager.escalation_timeout = 0.1  # Very short timeout
        
        result = await escalation_manager.escalate_emergency(emergency_event)
        
        assert 'timeout_triggered' in result
        assert result['escalation_level'] >= 1


class TestEmergencyResponseIntegration:
    """Integration tests for complete emergency response workflow."""

    @pytest.fixture
    def full_system_setup(self, emergency_config, mock_portfolio, mock_integrations):
        """Set up complete emergency response system for integration testing."""
        system = EmergencyResponseSystem(
            config=emergency_config,
            portfolio=mock_portfolio,
            integrations=mock_integrations
        )
        return system

    @pytest.mark.asyncio
    async def test_complete_emergency_workflow(self, full_system_setup):
        """Test complete emergency response workflow from event to resolution."""
        system = full_system_setup
        await system.initialize()
        
        # Create emergency event
        emergency_event = EmergencyEvent(
            event_id=uuid4(),
            event_type=EmergencyEventType.SYSTEM_FAILURE,
            severity=EmergencyEventSeverity.CRITICAL,
            title="Complete Workflow Test",
            description="Test complete emergency response workflow",
            requires_immediate_action=True
        )
        
        # Mock all subsystems to succeed
        system.liquidation_manager.execute_emergency_liquidation = AsyncMock(
            return_value={'success': True, 'liquidated_positions': 2}
        )
        system.notification_manager.send_emergency_notifications = AsyncMock(
            return_value={'sent': 4, 'failed': 0}
        )
        system.shutdown_manager.execute_emergency_shutdown = AsyncMock(
            return_value={'success': True, 'shutdown_time': 3.5}
        )
        
        # Execute emergency response
        response = await system.handle_emergency_event(emergency_event)
        
        # Verify complete workflow
        assert response.status == EmergencyResponseStatus.COMPLETED
        assert response.liquidation_executed is True
        assert response.notifications_sent > 0
        assert response.emergency_shutdown_executed is True
        assert response.response_time_ms > 0

    @pytest.mark.asyncio
    async def test_emergency_response_audit_trail(self, full_system_setup):
        """Test emergency response audit trail generation."""
        system = full_system_setup
        await system.initialize()
        
        # Execute multiple emergency responses
        for i in range(3):
            event = EmergencyEvent(
                event_id=uuid4(),
                event_type=EmergencyEventType.MARKET_ANOMALY,
                severity=EmergencyEventSeverity.MEDIUM,
                title=f"Audit Test {i}"
            )
            
            system.liquidation_manager.analyze_position_risk = AsyncMock(
                return_value={'liquidation_recommended': False}
            )
            system.notification_manager.send_emergency_notifications = AsyncMock(
                return_value={'sent': 2, 'failed': 0}
            )
            
            await system.handle_emergency_event(event)
        
        # Get audit trail
        audit_trail = await system.get_emergency_audit_trail(limit=10)
        
        assert len(audit_trail) >= 3
        assert all('event_id' in entry for entry in audit_trail)
        assert all('response_details' in entry for entry in audit_trail)
        assert all('timestamp' in entry for entry in audit_trail)

    @pytest.mark.asyncio
    async def test_system_resilience_under_load(self, full_system_setup):
        """Test emergency response system resilience under high load."""
        system = full_system_setup
        await system.initialize()
        
        # Create many concurrent emergency events
        events = []
        for i in range(10):
            event = EmergencyEvent(
                event_id=uuid4(),
                event_type=EmergencyEventType.MARKET_ANOMALY,
                severity=EmergencyEventSeverity.LOW,
                title=f"Load Test {i}"
            )
            events.append(event)
        
        # Mock lightweight responses
        system.notification_manager.send_emergency_notifications = AsyncMock(
            return_value={'sent': 1, 'failed': 0}
        )
        
        # Process all events concurrently
        tasks = [system.handle_emergency_event(event) for event in events[:5]]  # Respect max_concurrent_responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify system handled load gracefully
        successful_responses = [r for r in responses if isinstance(r, EmergencyResponse)]
        assert len(successful_responses) > 0
        
        # Should not have crashed or raised unhandled exceptions
        exceptions = [r for r in responses if isinstance(r, Exception)]
        assert len(exceptions) == 0  # No unhandled exceptions