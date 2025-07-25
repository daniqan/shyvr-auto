"""
Tests for dashboard service
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.dashboard.service import DashboardService, dashboard_service
from src.dashboard.base import (
    DashboardData, SystemMetrics, TradingStatus, PortfolioStatus, MLRLStatus,
    SystemStatus, TradingMode, DashboardError
)


class TestDashboardService:
    """Test suite for DashboardService"""
    
    @pytest.fixture
    def dashboard_service_instance(self):
        """Dashboard service instance for testing"""
        return DashboardService()
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        config = MagicMock()
        config.app.environment = "test"
        return config
    
    def test_dashboard_service_initialization(self, dashboard_service_instance):
        """Test DashboardService initialization"""
        service = dashboard_service_instance
        
        assert service._running is False
        assert service._update_task is None
        assert service._start_time > 0
        assert service.update_interval == 5.0
        assert service._request_count == 0
        assert service._error_count == 0
        assert isinstance(service._request_times, list)
        
        # Components should be None initially
        assert service.mode_manager is None
        assert service.health_monitor is None
        assert service.portfolio_manager is None
        assert service.model_manager is None
        assert service.rl_agent is None
        assert service.ml_rl_bridge is None
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.get_config')
    @patch('src.dashboard.service.websocket_manager')
    async def test_start_dashboard_service(self, mock_websocket_manager, mock_get_config, dashboard_service_instance):
        """Test starting dashboard service"""
        mock_get_config.return_value = MagicMock()
        mock_websocket_manager.start = AsyncMock()
        
        service = dashboard_service_instance
        
        await service.start()
        
        assert service._running is True
        assert service._update_task is not None
        mock_websocket_manager.start.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.websocket_manager')
    async def test_stop_dashboard_service(self, mock_websocket_manager, dashboard_service_instance):
        """Test stopping dashboard service"""
        mock_websocket_manager.stop = AsyncMock()
        
        service = dashboard_service_instance
        
        # Start first
        await service.start()
        assert service._running is True
        
        # Stop
        await service.stop()
        assert service._running is False
        mock_websocket_manager.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_components(self, dashboard_service_instance):
        """Test component initialization"""
        service = dashboard_service_instance
        
        # Should not raise error even with mock components
        await service._initialize_components()
        
        # Some components should be initialized (even if mocked)
        assert service.risk_manager is None  # Will be None due to mocking, but method should complete
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.psutil')
    async def test_get_system_metrics(self, mock_psutil, dashboard_service_instance):
        """Test getting system metrics"""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 25.5
        mock_memory = MagicMock()
        mock_memory.used = 512 * 1024 * 1024  # 512 MB in bytes
        mock_memory.percent = 45.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        service = dashboard_service_instance
        service._request_count = 100
        service._error_count = 2
        service._request_times = [0.1, 0.2, 0.15, 0.3, 0.25]
        
        metrics = await service._get_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.status == SystemStatus.HEALTHY
        assert metrics.cpu_usage_pct == 25.5
        assert metrics.memory_usage_mb == 512.0
        assert metrics.memory_usage_pct == 45.0
        assert metrics.total_requests == 100
        assert metrics.total_errors == 2
        assert metrics.uptime_seconds > 0
    
    @pytest.mark.asyncio
    async def test_get_portfolio_status(self, dashboard_service_instance):
        """Test getting portfolio status"""
        service = dashboard_service_instance
        
        portfolio = await service._get_portfolio_status()
        
        assert isinstance(portfolio, PortfolioStatus)
        # Should return mock data since no real portfolio manager
        assert portfolio.total_value_usd == Decimal("10000.00")
        assert portfolio.position_count == 3
        assert len(portfolio.chain_balances) == 3
    
    @pytest.mark.asyncio
    async def test_get_trading_status(self, dashboard_service_instance):
        """Test getting trading status"""
        service = dashboard_service_instance
        
        trading = await service._get_trading_status()
        
        assert isinstance(trading, TradingStatus)
        assert trading.mode == TradingMode.ANALYSIS
        assert trading.trades_today == 8
        assert trading.risk_limits_active is True
        assert trading.emergency_stop_active is False
    
    @pytest.mark.asyncio
    async def test_get_ml_rl_status(self, dashboard_service_instance):
        """Test getting ML/RL status"""
        service = dashboard_service_instance
        
        mlrl = await service._get_ml_rl_status()
        
        assert isinstance(mlrl, MLRLStatus)
        assert len(mlrl.ml_models) == 1
        assert len(mlrl.rl_agents) == 1
        assert mlrl.ml_rl_integration_active is True
        assert mlrl.continuous_learning_active is True
        assert mlrl.ml_models[0].name == "LSTM Price Predictor"
        assert mlrl.rl_agents[0].algorithm == "DQN"
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.psutil')
    async def test_get_dashboard_data(self, mock_psutil, dashboard_service_instance):
        """Test getting complete dashboard data"""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 30.0
        mock_memory = MagicMock()
        mock_memory.used = 1024 * 1024 * 1024  # 1 GB
        mock_memory.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        service = dashboard_service_instance
        
        data = await service.get_dashboard_data()
        
        assert isinstance(data, DashboardData)
        assert isinstance(data.system_metrics, SystemMetrics)
        assert isinstance(data.portfolio_status, PortfolioStatus)
        assert isinstance(data.trading_status, TradingStatus)
        assert isinstance(data.ml_rl_status, MLRLStatus)
        assert isinstance(data.timestamp, datetime)
        
        # Should have realistic values
        assert data.system_metrics.cpu_usage_pct == 30.0
        assert data.system_metrics.memory_usage_pct == 60.0
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.psutil')
    async def test_get_dashboard_data_error_handling(self, mock_psutil, dashboard_service_instance):
        """Test dashboard data error handling"""
        # Make psutil raise an error
        mock_psutil.cpu_percent.side_effect = Exception("System error")
        
        service = dashboard_service_instance
        
        # Should return default data on error
        data = await service.get_dashboard_data()
        
        assert isinstance(data, DashboardData)
        # Should be default data
        assert data.active_alerts == 0
        assert data.warnings_count == 0
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.websocket_manager')
    async def test_switch_trading_mode(self, mock_websocket_manager, dashboard_service_instance):
        """Test switching trading mode"""
        mock_websocket_manager.send_system_alert = AsyncMock()
        
        service = dashboard_service_instance
        user_id = "test-user"
        
        # Test valid mode switch
        result = await service.switch_trading_mode("simulation", user_id)
        
        assert result is True
        mock_websocket_manager.send_system_alert.assert_called_once()
        
        # Check alert parameters
        call_args = mock_websocket_manager.send_system_alert.call_args
        assert call_args[0][0] == "mode_change"
        assert "simulation" in call_args[0][1]
        assert user_id in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_switch_trading_mode_invalid(self, dashboard_service_instance):
        """Test switching to invalid trading mode"""
        service = dashboard_service_instance
        
        # Should return False for invalid mode (but not raise error)
        result = await service.switch_trading_mode("invalid_mode", "user")
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.websocket_manager')
    async def test_emergency_stop(self, mock_websocket_manager, dashboard_service_instance):
        """Test emergency stop activation"""
        mock_websocket_manager.send_system_alert = AsyncMock()
        
        service = dashboard_service_instance
        user_id = "test-user"
        
        result = await service.emergency_stop(user_id)
        
        assert result is True
        mock_websocket_manager.send_system_alert.assert_called_once()
        
        # Check alert parameters
        call_args = mock_websocket_manager.send_system_alert.call_args
        assert call_args[0][0] == "emergency_stop"
        assert "Emergency stop activated" in call_args[0][1]
        assert call_args[0][2] == "critical"
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.websocket_manager')
    async def test_update_risk_limits(self, mock_websocket_manager, dashboard_service_instance):
        """Test updating risk limits"""
        mock_websocket_manager.send_system_alert = AsyncMock()
        
        service = dashboard_service_instance
        user_id = "test-user"
        limits = {
            "max_position_size_pct": 0.15,
            "max_daily_loss_pct": 0.03,
            "stop_loss_pct": 0.1
        }
        
        result = await service.update_risk_limits(limits, user_id)
        
        assert result is True
        mock_websocket_manager.send_system_alert.assert_called_once()
        
        # Check alert parameters
        call_args = mock_websocket_manager.send_system_alert.call_args
        assert call_args[0][0] == "risk_limits_updated"
        assert user_id in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_get_trading_history(self, dashboard_service_instance):
        """Test getting trading history"""
        service = dashboard_service_instance
        
        # Should return empty list for now (mock implementation)
        history = await service.get_trading_history(limit=50)
        
        assert isinstance(history, list)
        assert len(history) == 0  # Mock returns empty list
    
    @pytest.mark.asyncio
    async def test_get_system_logs(self, dashboard_service_instance):
        """Test getting system logs"""
        service = dashboard_service_instance
        
        logs = await service.get_system_logs(limit=10)
        
        assert isinstance(logs, list)
        assert len(logs) == 5  # Mock returns 5 logs
        assert "System started successfully" in logs[0]
        assert "Dashboard service started" in logs[-1]
    
    def test_record_request_success(self, dashboard_service_instance):
        """Test recording successful request"""
        service = dashboard_service_instance
        initial_count = service._request_count
        initial_error_count = service._error_count
        
        service.record_request(0.150, error=False)
        
        assert service._request_count == initial_count + 1
        assert service._error_count == initial_error_count
        assert 0.150 in service._request_times
    
    def test_record_request_error(self, dashboard_service_instance):
        """Test recording request with error"""
        service = dashboard_service_instance
        initial_count = service._request_count
        initial_error_count = service._error_count
        
        service.record_request(0.500, error=True)
        
        assert service._request_count == initial_count + 1
        assert service._error_count == initial_error_count + 1
        assert 0.500 in service._request_times
    
    def test_record_request_times_limit(self, dashboard_service_instance):
        """Test request times list doesn't grow too large"""
        service = dashboard_service_instance
        
        # Add many request times
        for i in range(1200):
            service.record_request(0.1 + i * 0.001)
        
        # Should be limited to 1000 entries
        assert len(service._request_times) == 1000
        # Should keep the most recent entries
        assert service._request_times[-1] > service._request_times[0]
    
    @pytest.mark.asyncio
    async def test_update_loop_error_handling(self, dashboard_service_instance):
        """Test update loop error handling"""
        service = dashboard_service_instance
        
        # Mock get_dashboard_data to raise error
        original_get_data = service.get_dashboard_data
        
        async def mock_get_data_with_error():
            raise Exception("Test error")
        
        service.get_dashboard_data = mock_get_data_with_error
        
        # Start service briefly
        service._running = True
        
        try:
            # Should handle error gracefully in update loop
            await asyncio.wait_for(service._update_loop(), timeout=0.1)
        except asyncio.TimeoutError:
            # Expected - loop runs indefinitely
            pass
        finally:
            service._running = False
            service.get_dashboard_data = original_get_data


class TestDashboardServiceIntegration:
    """Integration tests for DashboardService"""
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.websocket_manager')
    @patch('src.dashboard.service.psutil')
    async def test_full_service_lifecycle(self, mock_psutil, mock_websocket_manager):
        """Test complete service lifecycle"""
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 25.0
        mock_memory = MagicMock()
        mock_memory.used = 512 * 1024 * 1024
        mock_memory.percent = 40.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_websocket_manager.start = AsyncMock()
        mock_websocket_manager.stop = AsyncMock()
        mock_websocket_manager.send_system_alert = AsyncMock()
        
        service = DashboardService()
        
        try:
            # Start service
            await service.start()
            assert service._running is True
            
            # Get dashboard data
            data = await service.get_dashboard_data()
            assert isinstance(data, DashboardData)
            assert data.system_metrics.cpu_usage_pct == 25.0
            
            # Perform operations
            await service.switch_trading_mode("simulation", "user1")
            await service.emergency_stop("user1")
            
            # Record some requests
            service.record_request(0.1)
            service.record_request(0.2, error=True)
            
            # Get updated data
            updated_data = await service.get_dashboard_data()
            assert updated_data.system_metrics.total_requests == 2
            assert updated_data.system_metrics.total_errors == 1
            
        finally:
            # Stop service
            await service.stop()
            assert service._running is False
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent dashboard operations"""
        service = DashboardService()
        
        # Concurrent data requests
        tasks = []
        for i in range(10):
            task = asyncio.create_task(service.get_dashboard_data())
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 10
        for result in results:
            assert isinstance(result, DashboardData)
    
    @pytest.mark.asyncio
    @patch('src.dashboard.service.websocket_manager')
    async def test_error_recovery(self, mock_websocket_manager):
        """Test service error recovery"""
        mock_websocket_manager.send_system_alert = AsyncMock()
        
        service = DashboardService()
        
        # Test operations that might fail
        results = []
        
        # These should not raise errors
        results.append(await service.switch_trading_mode("invalid", "user"))
        results.append(await service.update_risk_limits({}, "user"))
        results.append(await service.get_trading_history())
        results.append(await service.get_system_logs())
        
        # Service should still be functional
        data = await service.get_dashboard_data()
        assert isinstance(data, DashboardData)


class TestGlobalDashboardService:
    """Test global dashboard_service instance"""
    
    def test_global_instance_exists(self):
        """Test that global dashboard_service instance exists"""
        assert dashboard_service is not None
        assert isinstance(dashboard_service, DashboardService)
    
    @pytest.mark.asyncio
    async def test_global_instance_functionality(self):
        """Test that global instance works correctly"""
        # Should be able to get dashboard data
        data = await dashboard_service.get_dashboard_data()
        assert isinstance(data, DashboardData)
    
    def test_global_instance_request_recording(self):
        """Test request recording on global instance"""
        initial_count = dashboard_service._request_count
        
        dashboard_service.record_request(0.123)
        
        assert dashboard_service._request_count == initial_count + 1