"""
Tests for mode base data structures and enums.

Following TDD methodology - these tests define the expected behavior
before implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional
from uuid import UUID, uuid4
from unittest.mock import Mock, AsyncMock, patch

from src.modes.base import (
    ModeType,
    ModeStatus,
    ModeBase,
    ModeConfig,
    ModeResult,
    TradingMode,
    AnalysisMode,
    SimulationMode,
    ModeError,
    ModeNotFoundError,
    ModeConfigError,
    InvalidModeTransitionError,
)
from src.portfolio.base import Portfolio, PortfolioConfig
from src.rl_agent.base import TradeAction, MarketState
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestModeType:
    """Test ModeType enum."""
    
    def test_mode_type_values(self):
        """Test all mode type values are defined correctly."""
        assert ModeType.ANALYSIS.value == "analysis"
        assert ModeType.SIMULATION.value == "simulation"
        assert ModeType.LIVE_TRADING.value == "live_trading"
        assert ModeType.BACKTESTING.value == "backtesting"
        assert ModeType.PAPER_TRADING.value == "paper_trading"
    
    def test_mode_type_enum_completeness(self):
        """Test that all expected mode types exist."""
        expected_modes = {
            "analysis", "simulation", "live_trading", 
            "backtesting", "paper_trading"
        }
        actual_modes = {mode.value for mode in ModeType}
        assert actual_modes == expected_modes


class TestModeStatus:
    """Test ModeStatus enum."""
    
    def test_mode_status_values(self):
        """Test all mode status values are defined correctly."""
        assert ModeStatus.INACTIVE.value == "inactive"
        assert ModeStatus.ACTIVE.value == "active"
        assert ModeStatus.PAUSED.value == "paused"
        assert ModeStatus.ERROR.value == "error"
        assert ModeStatus.STOPPING.value == "stopping"
    
    def test_mode_status_transitions(self):
        """Test valid status transitions."""
        # These are the expected valid transitions
        valid_transitions = {
            ModeStatus.INACTIVE: [ModeStatus.ACTIVE],
            ModeStatus.ACTIVE: [ModeStatus.PAUSED, ModeStatus.STOPPING, ModeStatus.ERROR],
            ModeStatus.PAUSED: [ModeStatus.ACTIVE, ModeStatus.STOPPING],
            ModeStatus.ERROR: [ModeStatus.INACTIVE, ModeStatus.ACTIVE],
            ModeStatus.STOPPING: [ModeStatus.INACTIVE],
        }
        
        # This test defines expected behavior
        for status, valid_next in valid_transitions.items():
            assert isinstance(status, ModeStatus)
            for next_status in valid_next:
                assert isinstance(next_status, ModeStatus)


class TestModeConfig:
    """Test ModeConfig data structure."""
    
    def test_mode_config_creation(self):
        """Test creating mode configuration with default values."""
        config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True
        )
        
        assert config.mode_type == ModeType.ANALYSIS
        assert config.enabled is True
        assert config.auto_start is False
        assert config.max_runtime_minutes is None
        assert config.stop_on_error is True
        assert config.log_level == "INFO"
        assert isinstance(config.parameters, dict)
        assert len(config.parameters) == 0
    
    def test_mode_config_with_parameters(self):
        """Test mode configuration with custom parameters."""
        params = {
            "max_positions": 5,
            "risk_tolerance": 0.02,
            "rebalance_frequency": "hourly"
        }
        
        config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            auto_start=True,
            max_runtime_minutes=120,
            parameters=params
        )
        
        assert config.mode_type == ModeType.SIMULATION
        assert config.auto_start is True
        assert config.max_runtime_minutes == 120
        assert config.parameters == params
    
    def test_mode_config_validation(self):
        """Test mode configuration validation."""
        # Test invalid max_runtime_minutes
        with pytest.raises(ValueError, match="max_runtime_minutes must be positive"):
            ModeConfig(
                mode_type=ModeType.ANALYSIS,
                enabled=True,
                max_runtime_minutes=-1
            )
        
        # Test invalid log_level
        with pytest.raises(ValueError, match="Invalid log_level"):
            ModeConfig(
                mode_type=ModeType.ANALYSIS,
                enabled=True,
                log_level="INVALID"
            )


class TestModeResult:
    """Test ModeResult data structure."""
    
    def test_mode_result_creation(self):
        """Test creating mode result."""
        mode_id = uuid4()
        start_time = datetime.now()
        
        result = ModeResult(
            mode_id=mode_id,
            mode_type=ModeType.ANALYSIS,
            status=ModeStatus.ACTIVE,
            start_time=start_time
        )
        
        assert result.mode_id == mode_id
        assert result.mode_type == ModeType.ANALYSIS
        assert result.status == ModeStatus.ACTIVE
        assert result.start_time == start_time
        assert result.end_time is None
        assert result.error_message is None
        assert isinstance(result.metrics, dict)
        assert isinstance(result.metadata, dict)
    
    def test_mode_result_duration(self):
        """Test mode result duration calculation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=30)
        
        result = ModeResult(
            mode_id=uuid4(),
            mode_type=ModeType.SIMULATION,
            status=ModeStatus.INACTIVE,
            start_time=start_time,
            end_time=end_time
        )
        
        duration = result.duration
        assert duration is not None
        assert duration.total_seconds() == 1800  # 30 minutes
    
    def test_mode_result_duration_none_when_running(self):
        """Test that duration is None when mode is still running."""
        result = ModeResult(
            mode_id=uuid4(),
            mode_type=ModeType.LIVE_TRADING,
            status=ModeStatus.ACTIVE,
            start_time=datetime.now()
        )
        
        assert result.duration is None


class TestModeBase:
    """Test ModeBase abstract class."""
    
    def test_mode_base_is_abstract(self):
        """Test that ModeBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModeBase()
    
    def test_mode_base_subclass_requirements(self):
        """Test that subclasses must implement abstract methods."""
        
        class IncompleteModeImpl(ModeBase):
            pass
        
        with pytest.raises(TypeError):
            IncompleteModeImpl()
    
    def test_mode_base_complete_implementation(self):
        """Test a complete ModeBase implementation."""
        
        class CompleteModeImpl(ModeBase):
            async def initialize(self) -> None:
                pass
            
            async def start(self) -> None:
                pass
            
            async def stop(self) -> None:
                pass
            
            async def pause(self) -> None:
                pass
            
            async def resume(self) -> None:
                pass
            
            async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
                return TradeAction.HOLD
            
            async def cleanup(self) -> None:
                pass
        
        # Should be able to create instance
        mode = CompleteModeImpl(
            mode_id=uuid4(),
            config=ModeConfig(mode_type=ModeType.ANALYSIS, enabled=True),
            portfolio=Mock()
        )
        
        assert isinstance(mode, ModeBase)
        assert mode.status == ModeStatus.INACTIVE
        assert mode.config.mode_type == ModeType.ANALYSIS


class TestTradingMode:
    """Test TradingMode concrete implementation."""
    
    @pytest.fixture
    def portfolio(self):
        """Create mock portfolio for testing."""
        portfolio = Mock(spec=Portfolio)
        portfolio.portfolio_id = uuid4()
        portfolio.cash_balance = Decimal("10000")
        portfolio.total_value = Decimal("10000")
        return portfolio
    
    @pytest.fixture
    def config(self):
        """Create trading mode configuration."""
        return ModeConfig(
            mode_type=ModeType.LIVE_TRADING,
            enabled=True,
            parameters={
                "max_position_size": 1000,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.15
            }
        )
    
    def test_trading_mode_creation(self, config, portfolio):
        """Test creating TradingMode instance."""
        mode = TradingMode(
            mode_id=uuid4(),
            config=config,
            portfolio=portfolio
        )
        
        assert mode.config.mode_type == ModeType.LIVE_TRADING
        assert mode.portfolio == portfolio
        assert mode.status == ModeStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_trading_mode_lifecycle(self, config, portfolio):
        """Test complete trading mode lifecycle."""
        mode = TradingMode(
            mode_id=uuid4(),
            config=config,
            portfolio=portfolio
        )
        
        # Test initialization
        await mode.initialize()
        assert mode.status == ModeStatus.INACTIVE
        
        # Test start
        await mode.start()
        assert mode.status == ModeStatus.ACTIVE
        
        # Test pause
        await mode.pause()
        assert mode.status == ModeStatus.PAUSED
        
        # Test resume
        await mode.resume()
        assert mode.status == ModeStatus.ACTIVE
        
        # Test stop
        await mode.stop()
        assert mode.status == ModeStatus.STOPPING
        
        # Test cleanup
        await mode.cleanup()


class TestAnalysisMode:
    """Test AnalysisMode concrete implementation."""
    
    @pytest.fixture
    def config(self):
        """Create analysis mode configuration."""
        return ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                "analysis_depth": "comprehensive",
                "include_sentiment": True,
                "timeframe": "1h"
            }
        )
    
    def test_analysis_mode_creation(self, config):
        """Test creating AnalysisMode instance."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=config,
            portfolio=Mock()
        )
        
        assert mode.config.mode_type == ModeType.ANALYSIS
        assert mode.status == ModeStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_analysis_mode_process_tick(self, config):
        """Test analysis mode processing market tick."""
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=config,
            portfolio=Mock()
        )
        
        # Mock market state
        market_state = Mock(spec=MarketState)
        market_state.token = Mock(spec=DiscoveredToken)
        market_state.price_usd = 100.0
        market_state.price_change_24h = 5.0
        
        # Analysis mode should not return trade actions
        action = await mode.process_tick(market_state)
        assert action is None or action == TradeAction.HOLD


class TestSimulationMode:
    """Test SimulationMode concrete implementation."""
    
    @pytest.fixture
    def config(self):
        """Create simulation mode configuration."""
        return ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "simulation_duration_hours": 24,
                "initial_balance": 10000,
                "enable_fees": True,
                "slippage_bps": 10
            }
        )
    
    def test_simulation_mode_creation(self, config):
        """Test creating SimulationMode instance."""
        mode = SimulationMode(
            mode_id=uuid4(),
            config=config,
            portfolio=Mock()
        )
        
        assert mode.config.mode_type == ModeType.SIMULATION
        assert mode.status == ModeStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_simulation_mode_process_tick(self, config):
        """Test simulation mode processing market tick."""
        mode = SimulationMode(
            mode_id=uuid4(),
            config=config,
            portfolio=Mock()
        )
        
        # Initialize and start the mode
        await mode.initialize()
        await mode.start()
        
        # Mock market state
        market_state = Mock(spec=MarketState)
        market_state.token = Mock(spec=DiscoveredToken)
        market_state.price_usd = 100.0
        market_state.rsi = 30.0  # Oversold
        
        # Simulation mode should be able to return trade actions
        action = await mode.process_tick(market_state)
        assert action is not None
        assert isinstance(action, TradeAction)


class TestModeExceptions:
    """Test mode-specific exceptions."""
    
    def test_mode_error(self):
        """Test ModeError exception."""
        error = ModeError("Test mode error")
        assert str(error) == "Test mode error"
        assert isinstance(error, Exception)
    
    def test_mode_not_found_error(self):
        """Test ModeNotFoundError exception."""
        mode_id = uuid4()
        error = ModeNotFoundError(f"Mode {mode_id} not found")
        assert isinstance(error, ModeError)
    
    def test_mode_config_error(self):
        """Test ModeConfigError exception."""
        error = ModeConfigError("Invalid configuration")
        assert isinstance(error, ModeError)
    
    def test_invalid_mode_transition_error(self):
        """Test InvalidModeTransitionError exception."""
        error = InvalidModeTransitionError("Cannot transition from ACTIVE to INACTIVE")
        assert isinstance(error, ModeError)


class TestModeIntegration:
    """Test mode integration with other system components."""
    
    @pytest.mark.asyncio
    async def test_mode_with_real_portfolio_structure(self):
        """Test mode with realistic portfolio structure."""
        # Create realistic portfolio config
        portfolio_config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC"
        )
        
        # Mock portfolio with proper structure
        portfolio = Mock(spec=Portfolio)
        portfolio.config = portfolio_config
        portfolio.cash_balance = Decimal("10000")
        portfolio.total_value = Decimal("10000")
        portfolio.open_positions = {}
        
        # Create mode
        mode_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True
        )
        
        mode = SimulationMode(
            mode_id=uuid4(),
            config=mode_config,
            portfolio=portfolio
        )
        
        # Test that mode can access portfolio properties
        assert mode.portfolio.cash_balance == Decimal("10000")
        assert mode.portfolio.config.base_currency == "USDC"
    
    @pytest.mark.asyncio
    async def test_mode_market_state_integration(self):
        """Test mode integration with MarketState."""
        config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True
        )
        
        mode = AnalysisMode(
            mode_id=uuid4(),
            config=config,
            portfolio=Mock()
        )
        
        # Create realistic market state
        token = Mock(spec=DiscoveredToken)
        token.address = "0x123..."
        token.symbol = "TEST"
        token.chain = Chain.ETHEREUM
        
        market_state = MarketState(
            token=token,
            price_usd=100.0,
            price_change_24h=-2.5,
            volume_24h=1000000.0,
            rsi=65.0,
            macd=1.2
        )
        
        # Mode should be able to process the market state
        result = await mode.process_tick(market_state)
        assert result is None or isinstance(result, TradeAction)