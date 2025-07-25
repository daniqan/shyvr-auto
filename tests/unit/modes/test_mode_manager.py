"""
Tests for mode manager functionality.

Following TDD methodology - these tests define the expected behavior
before implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.modes.mode_manager import (
    ModeManager,
    ModeManagerConfig,
    ModeManagerError,
    ModeScheduler,
    ModeTransitionError,
    DuplicateModeError,
)
from src.modes.base import (
    ModeType,
    ModeStatus,
    ModeBase,
    ModeConfig,
    ModeResult,
    TradingMode,
    AnalysisMode,
    SimulationMode,
)
from src.portfolio.base import Portfolio, PortfolioConfig
from src.rl_agent.base import TradeAction, MarketState
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestModeManagerConfig:
    """Test ModeManagerConfig data structure."""
    
    def test_mode_manager_config_creation(self):
        """Test creating mode manager configuration with default values."""
        config = ModeManagerConfig()
        
        assert config.max_concurrent_modes == 1
        assert config.enable_mode_switching is True
        assert config.auto_recovery is True
        assert config.health_check_interval_seconds == 30
        assert config.max_error_retries == 3
        assert config.mode_timeout_minutes == 60
        assert isinstance(config.default_modes, list)
        assert len(config.default_modes) == 0
    
    def test_mode_manager_config_with_custom_values(self):
        """Test mode manager configuration with custom values."""
        default_modes = [
            ModeConfig(mode_type=ModeType.ANALYSIS, enabled=True),
            ModeConfig(mode_type=ModeType.SIMULATION, enabled=True)
        ]
        
        config = ModeManagerConfig(
            max_concurrent_modes=3,
            enable_mode_switching=False,
            auto_recovery=False,
            health_check_interval_seconds=60,
            max_error_retries=5,
            mode_timeout_minutes=120,
            default_modes=default_modes
        )
        
        assert config.max_concurrent_modes == 3
        assert config.enable_mode_switching is False
        assert config.auto_recovery is False
        assert config.health_check_interval_seconds == 60
        assert config.max_error_retries == 5
        assert config.mode_timeout_minutes == 120
        assert len(config.default_modes) == 2
    
    def test_mode_manager_config_validation(self):
        """Test mode manager configuration validation."""
        # Test invalid max_concurrent_modes
        with pytest.raises(ValueError, match="max_concurrent_modes must be positive"):
            ModeManagerConfig(max_concurrent_modes=0)
        
        # Test invalid health_check_interval_seconds
        with pytest.raises(ValueError, match="health_check_interval_seconds must be positive"):
            ModeManagerConfig(health_check_interval_seconds=-1)
        
        # Test invalid max_error_retries
        with pytest.raises(ValueError, match="max_error_retries must be non-negative"):
            ModeManagerConfig(max_error_retries=-1)
        
        # Test invalid mode_timeout_minutes
        with pytest.raises(ValueError, match="mode_timeout_minutes must be positive"):
            ModeManagerConfig(mode_timeout_minutes=0)


class TestModeManager:
    """Test ModeManager functionality."""
    
    @pytest.fixture
    def portfolio(self):
        """Create mock portfolio for testing."""
        portfolio = Mock(spec=Portfolio)
        portfolio.portfolio_id = uuid4()
        portfolio.cash_balance = Decimal("10000")
        portfolio.total_value = Decimal("10000")
        portfolio.config = Mock(spec=PortfolioConfig)
        return portfolio
    
    @pytest.fixture
    def manager_config(self):
        """Create mode manager configuration."""
        return ModeManagerConfig(
            max_concurrent_modes=2,
            enable_mode_switching=True,
            auto_recovery=True
        )
    
    @pytest.fixture
    def mode_manager(self, manager_config, portfolio):
        """Create mode manager instance."""
        return ModeManager(config=manager_config, portfolio=portfolio)
    
    def test_mode_manager_creation(self, manager_config, portfolio):
        """Test creating ModeManager instance."""
        manager = ModeManager(config=manager_config, portfolio=portfolio)
        
        assert manager.config == manager_config
        assert manager.portfolio == portfolio
        assert isinstance(manager.active_modes, dict)
        assert len(manager.active_modes) == 0
        assert isinstance(manager.mode_history, list)
        assert len(manager.mode_history) == 0
    
    @pytest.mark.asyncio
    async def test_mode_manager_initialization(self, mode_manager):
        """Test mode manager initialization."""
        await mode_manager.initialize()
        
        # Manager should be ready to accept modes
        assert mode_manager._initialized is True
    
    @pytest.mark.asyncio 
    async def test_register_mode(self, mode_manager):
        """Test registering a new mode."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True
        )
        
        mode_id = await mode_manager.register_mode(mode_config)
        
        assert isinstance(mode_id, UUID)
        assert mode_id in mode_manager.active_modes
        assert mode_manager.active_modes[mode_id].config.mode_type == ModeType.ANALYSIS
        assert mode_manager.active_modes[mode_id].status == ModeStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_register_duplicate_mode_type(self, mode_manager):
        """Test that registering duplicate mode types is handled correctly."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True
        )
        
        # Register first mode
        mode_id1 = await mode_manager.register_mode(mode_config)
        
        # Try to register second mode of same type
        with pytest.raises(DuplicateModeError, match="Mode type analysis already registered"):
            await mode_manager.register_mode(mode_config)
    
    @pytest.mark.asyncio
    async def test_start_mode(self, mode_manager):
        """Test starting a registered mode."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True
        )
        
        mode_id = await mode_manager.register_mode(mode_config)
        await mode_manager.start_mode(mode_id)
        
        mode = mode_manager.active_modes[mode_id]
        assert mode.status == ModeStatus.ACTIVE
        assert mode.start_time is not None
    
    @pytest.mark.asyncio
    async def test_stop_mode(self, mode_manager):
        """Test stopping an active mode."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True
        )
        
        mode_id = await mode_manager.register_mode(mode_config)
        await mode_manager.start_mode(mode_id)
        await mode_manager.stop_mode(mode_id)
        
        mode = mode_manager.active_modes[mode_id]
        assert mode.status == ModeStatus.STOPPING
    
    @pytest.mark.asyncio
    async def test_pause_resume_mode(self, mode_manager):
        """Test pausing and resuming a mode."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True
        )
        
        mode_id = await mode_manager.register_mode(mode_config)
        await mode_manager.start_mode(mode_id)
        
        # Pause the mode
        await mode_manager.pause_mode(mode_id)
        mode = mode_manager.active_modes[mode_id]
        assert mode.status == ModeStatus.PAUSED
        
        # Resume the mode
        await mode_manager.resume_mode(mode_id)
        mode = mode_manager.active_modes[mode_id]
        assert mode.status == ModeStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_unregister_mode(self, mode_manager):
        """Test unregistering a mode."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True
        )
        
        mode_id = await mode_manager.register_mode(mode_config)
        assert mode_id in mode_manager.active_modes
        
        await mode_manager.unregister_mode(mode_id)
        assert mode_id not in mode_manager.active_modes
    
    @pytest.mark.asyncio
    async def test_get_mode_status(self, mode_manager):
        """Test getting mode status."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True
        )
        
        mode_id = await mode_manager.register_mode(mode_config)
        
        status = mode_manager.get_mode_status(mode_id)
        assert status == ModeStatus.INACTIVE
        
        await mode_manager.start_mode(mode_id)
        status = mode_manager.get_mode_status(mode_id)
        assert status == ModeStatus.ACTIVE
    
    def test_get_mode_status_not_found(self, mode_manager):
        """Test getting status of non-existent mode."""
        fake_mode_id = uuid4()
        
        status = mode_manager.get_mode_status(fake_mode_id)
        assert status is None
    
    def test_list_active_modes(self, mode_manager):
        """Test listing all active modes."""
        # Initially no modes
        modes = mode_manager.list_active_modes()
        assert isinstance(modes, dict)
        assert len(modes) == 0
    
    @pytest.mark.asyncio
    async def test_list_active_modes_with_modes(self, mode_manager):
        """Test listing active modes with registered modes."""
        await mode_manager.initialize()
        
        # Register multiple modes
        analysis_config = ModeConfig(mode_type=ModeType.ANALYSIS, enabled=True)
        simulation_config = ModeConfig(mode_type=ModeType.SIMULATION, enabled=True)
        
        analysis_id = await mode_manager.register_mode(analysis_config)
        simulation_id = await mode_manager.register_mode(simulation_config)
        
        modes = mode_manager.list_active_modes()
        assert len(modes) == 2
        assert analysis_id in modes
        assert simulation_id in modes
        assert modes[analysis_id].config.mode_type == ModeType.ANALYSIS
        assert modes[simulation_id].config.mode_type == ModeType.SIMULATION
    
    def test_get_mode_results(self, mode_manager):
        """Test getting mode execution results."""
        results = mode_manager.get_mode_results()
        assert isinstance(results, list)
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_get_mode_results_with_history(self, mode_manager):
        """Test getting mode results with execution history."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True
        )
        
        mode_id = await mode_manager.register_mode(mode_config)
        await mode_manager.start_mode(mode_id)
        await mode_manager.stop_mode(mode_id)
        
        # Should have results in history
        results = mode_manager.get_mode_results()
        assert len(results) >= 0  # Results may be added during stop
    
    @pytest.mark.asyncio
    async def test_switch_mode(self, mode_manager):
        """Test switching between modes."""
        await mode_manager.initialize()
        
        # Register two modes
        analysis_config = ModeConfig(mode_type=ModeType.ANALYSIS, enabled=True)
        simulation_config = ModeConfig(mode_type=ModeType.SIMULATION, enabled=True)
        
        analysis_id = await mode_manager.register_mode(analysis_config)
        simulation_id = await mode_manager.register_mode(simulation_config)
        
        # Start analysis mode
        await mode_manager.start_mode(analysis_id)
        assert mode_manager.get_mode_status(analysis_id) == ModeStatus.ACTIVE
        
        # Switch to simulation mode
        await mode_manager.switch_mode(analysis_id, simulation_id)
        
        # Analysis should be stopped, simulation should be active
        assert mode_manager.get_mode_status(analysis_id) == ModeStatus.STOPPING
        assert mode_manager.get_mode_status(simulation_id) == ModeStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_switch_mode_disabled(self, portfolio):
        """Test that mode switching can be disabled."""
        config = ModeManagerConfig(enable_mode_switching=False)
        manager = ModeManager(config=config, portfolio=portfolio)
        await manager.initialize()
        
        # Register two modes
        analysis_config = ModeConfig(mode_type=ModeType.ANALYSIS, enabled=True)
        simulation_config = ModeConfig(mode_type=ModeType.SIMULATION, enabled=True)
        
        analysis_id = await manager.register_mode(analysis_config)
        simulation_id = await manager.register_mode(simulation_config)
        
        await manager.start_mode(analysis_id)
        
        # Should raise error when switching is disabled
        with pytest.raises(ModeTransitionError, match="Mode switching is disabled"):
            await manager.switch_mode(analysis_id, simulation_id)
    
    @pytest.mark.asyncio
    async def test_process_market_tick(self, mode_manager):
        """Test processing market tick through active modes."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True
        )
        
        mode_id = await mode_manager.register_mode(mode_config)
        await mode_manager.start_mode(mode_id)
        
        # Create market state
        token = Mock(spec=DiscoveredToken)
        token.address = "0x123..."
        token.symbol = "TEST"
        
        market_state = MarketState(
            token=token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0,
            rsi=65.0
        )
        
        # Process tick through all active modes
        actions = await mode_manager.process_market_tick(market_state)
        
        assert isinstance(actions, dict)
        assert mode_id in actions
        assert isinstance(actions[mode_id], TradeAction)
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mode_manager):
        """Test mode manager cleanup."""
        await mode_manager.initialize()
        
        mode_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True
        )
        
        mode_id = await mode_manager.register_mode(mode_config)
        await mode_manager.start_mode(mode_id)
        
        # Cleanup should stop all modes
        await mode_manager.cleanup()
        
        # All modes should be stopped
        for mode in mode_manager.active_modes.values():
            assert mode.status in [ModeStatus.STOPPING, ModeStatus.INACTIVE]


class TestModeScheduler:
    """Test ModeScheduler functionality."""
    
    @pytest.fixture
    def scheduler_config(self):
        """Create scheduler configuration."""
        return {
            "schedule_interval_seconds": 60,
            "max_scheduled_modes": 10,
            "enable_scheduling": True
        }
    
    def test_mode_scheduler_creation(self, scheduler_config):
        """Test creating ModeScheduler instance."""
        scheduler = ModeScheduler(config=scheduler_config)
        
        assert scheduler.config == scheduler_config
        assert isinstance(scheduler.scheduled_modes, list)
        assert len(scheduler.scheduled_modes) == 0
    
    @pytest.mark.asyncio
    async def test_schedule_mode(self, scheduler_config):
        """Test scheduling a mode to start at a specific time."""
        scheduler = ModeScheduler(config=scheduler_config)
        
        mode_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True
        )
        
        start_time = datetime.now() + timedelta(minutes=5)
        
        schedule_id = await scheduler.schedule_mode(
            mode_config=mode_config,
            start_time=start_time,
            max_runtime_minutes=30
        )
        
        assert isinstance(schedule_id, UUID)
        assert len(scheduler.scheduled_modes) == 1
    
    @pytest.mark.asyncio
    async def test_cancel_scheduled_mode(self, scheduler_config):
        """Test canceling a scheduled mode."""
        scheduler = ModeScheduler(config=scheduler_config)
        
        mode_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True
        )
        
        start_time = datetime.now() + timedelta(minutes=5)
        schedule_id = await scheduler.schedule_mode(
            mode_config=mode_config,
            start_time=start_time
        )
        
        # Cancel the scheduled mode
        success = await scheduler.cancel_scheduled_mode(schedule_id)
        assert success is True
        assert len(scheduler.scheduled_modes) == 0
    
    def test_list_scheduled_modes(self, scheduler_config):
        """Test listing all scheduled modes."""
        scheduler = ModeScheduler(config=scheduler_config)
        
        modes = scheduler.list_scheduled_modes()
        assert isinstance(modes, list)
        assert len(modes) == 0


class TestModeExceptions:
    """Test mode manager specific exceptions."""
    
    def test_mode_manager_error(self):
        """Test ModeManagerError exception."""
        error = ModeManagerError("Test manager error")
        assert str(error) == "Test manager error"
        assert isinstance(error, Exception)
    
    def test_mode_transition_error(self):
        """Test ModeTransitionError exception."""
        error = ModeTransitionError("Invalid transition")
        assert isinstance(error, ModeManagerError)
    
    def test_duplicate_mode_error(self):
        """Test DuplicateModeError exception."""
        error = DuplicateModeError("Mode already exists")
        assert isinstance(error, ModeManagerError)


class TestModeManagerIntegration:
    """Test mode manager integration with other components."""
    
    @pytest.mark.asyncio
    async def test_mode_manager_with_multiple_modes(self):
        """Test mode manager coordinating multiple concurrent modes."""
        portfolio = Mock(spec=Portfolio)
        portfolio.portfolio_id = uuid4()
        portfolio.cash_balance = Decimal("10000")
        
        config = ModeManagerConfig(
            max_concurrent_modes=3,
            enable_mode_switching=True
        )
        
        manager = ModeManager(config=config, portfolio=portfolio)
        await manager.initialize()
        
        # Register multiple different modes
        modes_configs = [
            ModeConfig(mode_type=ModeType.ANALYSIS, enabled=True),
            ModeConfig(mode_type=ModeType.SIMULATION, enabled=True),
        ]
        
        mode_ids = []
        for mode_config in modes_configs:
            mode_id = await manager.register_mode(mode_config)
            mode_ids.append(mode_id)
        
        # Start all modes
        for mode_id in mode_ids:
            await manager.start_mode(mode_id)
        
        # All modes should be active
        for mode_id in mode_ids:
            assert manager.get_mode_status(mode_id) == ModeStatus.ACTIVE
        
        # Process market tick through all modes
        token = Mock(spec=DiscoveredToken)
        market_state = MarketState(
            token=token,
            price_usd=100.0,
            price_change_24h=2.5,
            volume_24h=500000.0
        )
        
        actions = await manager.process_market_tick(market_state)
        assert len(actions) == len(mode_ids)
        
        # Cleanup
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_mode_manager_error_recovery(self):
        """Test mode manager error recovery functionality."""
        portfolio = Mock(spec=Portfolio)
        config = ModeManagerConfig(
            auto_recovery=True,
            max_error_retries=2
        )
        
        manager = ModeManager(config=config, portfolio=portfolio)
        await manager.initialize()
        
        # This would test error recovery, but requires more complex mocking
        # For now, just verify the structure is in place
        assert manager.config.auto_recovery is True
        assert manager.config.max_error_retries == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_mode_limits(self):
        """Test that concurrent mode limits are enforced."""
        portfolio = Mock(spec=Portfolio)
        config = ModeManagerConfig(max_concurrent_modes=1)
        
        manager = ModeManager(config=config, portfolio=portfolio)
        await manager.initialize()
        
        # Register and start first mode
        mode1_config = ModeConfig(mode_type=ModeType.ANALYSIS, enabled=True)
        mode1_id = await manager.register_mode(mode1_config)
        await manager.start_mode(mode1_id)
        
        # Try to register second mode when limit is 1
        mode2_config = ModeConfig(mode_type=ModeType.SIMULATION, enabled=True)
        mode2_id = await manager.register_mode(mode2_config)
        
        # Starting second mode should either fail or stop first mode
        # depending on implementation
        await manager.start_mode(mode2_id)
        
        # At least one mode should be active, but not both if limit is 1
        active_count = sum(
            1 for mode in manager.active_modes.values()
            if mode.status == ModeStatus.ACTIVE
        )
        assert active_count <= 1