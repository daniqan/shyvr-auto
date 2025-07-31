"""
Test preservation integration for Trading Modes

Tests for Phase 1.5: Integration with Existing System
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from datetime import datetime

from src.modes.base import (
    ModeBase, TradingMode, AnalysisMode, SimulationMode,
    ModeType, ModeStatus, ModeConfig, Portfolio
)
from src.model_preservation.manager import PreservationManager
from src.model_preservation.base import PreservationError, PreservationPriority


@pytest.fixture
def mock_preservation_manager():
    """Create a mock preservation manager"""
    manager = Mock(spec=PreservationManager)
    manager.save_model = AsyncMock(return_value="mode_model_123")
    manager.load_model = AsyncMock(return_value=(b"model_data", {"model_id": "mode_model_123"}))
    manager.migrate_model = AsyncMock(return_value="migrated_model_123")
    manager.emergency_backup = AsyncMock(return_value=["backup_1", "backup_2"])
    return manager


@pytest.fixture
def mock_portfolio():
    """Create a mock portfolio"""
    portfolio = Mock(spec=Portfolio)
    portfolio.get_balance = Mock(return_value=10000)
    portfolio.get_positions = Mock(return_value={})
    return portfolio


@pytest.fixture
def mode_config():
    """Create test mode configuration"""
    return ModeConfig(
        mode_type=ModeType.ANALYSIS,
        enabled=True,
        auto_start=False,
        parameters={
            'preservation': {
                'enabled': True,
                'auto_backup_on_change': True,
                'priority': 'high'
            }
        }
    )


class TestModeBasePreservation:
    """Test base mode preservation integration"""
    
    @pytest.mark.asyncio
    async def test_mode_change_triggers_backup(self, mock_preservation_manager, mock_portfolio, mode_config):
        """Test that changing modes triggers model backup"""
        # Create mode with preservation
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mode._preservation_manager = mock_preservation_manager
        mode._ml_models = {'lstm': Mock(), 'dqn': Mock()}
        
        # Start mode
        await mode.start()
        
        # Change to different mode (simulate by stopping)
        await mode.stop()
        
        # Verify backup was triggered
        assert mock_preservation_manager.save_model.call_count >= len(mode._ml_models)
        
        # Check that models were saved with mode context
        for call in mock_preservation_manager.save_model.call_args_list:
            assert call.kwargs.get('mode') == 'analysis'
            assert 'mode_change' in call.kwargs.get('tags', [])
    
    @pytest.mark.asyncio
    async def test_mode_specific_model_isolation(self, mock_preservation_manager, mock_portfolio):
        """Test that models are isolated by mode"""
        # Create different modes
        analysis_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={'preservation': {'enabled': True}}
        )
        simulation_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={'preservation': {'enabled': True}}
        )
        
        analysis_mode = AnalysisMode(uuid4(), analysis_config, mock_portfolio)
        simulation_mode = SimulationMode(uuid4(), simulation_config, mock_portfolio)
        
        # Set preservation managers
        analysis_mode._preservation_manager = mock_preservation_manager
        simulation_mode._preservation_manager = mock_preservation_manager
        
        # Load models for each mode
        analysis_mode._model_type = 'lstm'
        simulation_mode._model_type = 'lstm'
        
        # Mock load model calls
        await analysis_mode._load_preserved_models()
        await simulation_mode._load_preserved_models()
        
        # Verify models were loaded with correct mode context
        calls = mock_preservation_manager.load_model.call_args_list
        assert any(call.kwargs.get('mode') == 'analysis' for call in calls)
        assert any(call.kwargs.get('mode') == 'simulation' for call in calls)
    
    @pytest.mark.asyncio
    async def test_emergency_backup_on_error(self, mock_preservation_manager, mock_portfolio, mode_config):
        """Test that emergency backup is triggered on mode error"""
        # Create mode
        mode = TradingMode(uuid4(), mode_config, mock_portfolio)
        mode._preservation_manager = mock_preservation_manager
        mode.status = ModeStatus.ACTIVE
        
        # Simulate error
        mode._set_status(ModeStatus.ERROR, "Critical error occurred")
        
        # Trigger error handling
        await mode._handle_error("Critical error")
        
        # Verify emergency backup was called
        mock_preservation_manager.emergency_backup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_migration_between_modes(self, mock_preservation_manager, mock_portfolio):
        """Test model migration when switching between modes"""
        # Create analysis mode
        analysis_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={'preservation': {'enabled': True}}
        )
        analysis_mode = AnalysisMode(uuid4(), analysis_config, mock_portfolio)
        analysis_mode._preservation_manager = mock_preservation_manager
        
        # Simulate model trained in analysis mode
        model_type = 'lstm'
        version = 'v1.0.0'
        
        # Migrate to simulation mode
        await analysis_mode._migrate_model_to_mode(
            model_type=model_type,
            version=version,
            from_mode='analysis',
            to_mode='simulation'
        )
        
        # Verify migration was called
        mock_preservation_manager.migrate_model.assert_called_once_with(
            model_type=model_type,
            version=version,
            from_mode='analysis',
            to_mode='simulation'
        )
    
    @pytest.mark.asyncio
    async def test_mode_preserves_state_on_pause(self, mock_preservation_manager, mock_portfolio, mode_config):
        """Test that mode state is preserved when paused"""
        # Create mode
        mode = SimulationMode(uuid4(), mode_config, mock_portfolio)
        mode._preservation_manager = mock_preservation_manager
        mode.status = ModeStatus.ACTIVE
        
        # Set some state
        mode.virtual_balance = 15000
        mode._simulation_metrics = {'trades': 10, 'profit': 500}
        
        # Pause mode
        await mode.pause()
        
        # Verify state was preserved
        call_args = mock_preservation_manager.save_model.call_args
        assert call_args is not None
        metadata = call_args.kwargs.get('metadata', {})
        assert 'mode_state' in metadata
        assert metadata['mode_state']['virtual_balance'] == 15000
    
    @pytest.mark.asyncio
    async def test_mode_restores_state_on_resume(self, mock_preservation_manager, mock_portfolio, mode_config):
        """Test that mode state is restored when resumed"""
        # Create mode
        mode = SimulationMode(uuid4(), mode_config, mock_portfolio)
        mode._preservation_manager = mock_preservation_manager
        mode.status = ModeStatus.PAUSED
        
        # Mock preserved state
        preserved_state = {
            'virtual_balance': 20000,
            'simulation_metrics': {'trades': 20, 'profit': 1000}
        }
        import pickle
        preserved_data = pickle.dumps(preserved_state)
        mock_preservation_manager.load_model.return_value = (
            preserved_data,
            {'metadata': {'mode_state': preserved_state}}
        )
        
        # Resume mode
        await mode.resume()
        
        # Verify state was restored
        assert mode.virtual_balance == 20000
    
    @pytest.mark.asyncio
    async def test_preservation_disabled_in_config(self, mock_portfolio):
        """Test that preservation is not used when disabled"""
        # Create mode with preservation disabled
        config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={'preservation': {'enabled': False}}
        )
        
        mode = AnalysisMode(uuid4(), config, mock_portfolio)
        
        # Verify preservation manager is not created
        assert not hasattr(mode, '_preservation_manager') or mode._preservation_manager is None
    
    @pytest.mark.asyncio
    async def test_mode_specific_preservation_priority(self, mock_preservation_manager, mock_portfolio):
        """Test that different modes use appropriate preservation priorities"""
        # Live trading should have highest priority
        live_config = ModeConfig(
            mode_type=ModeType.LIVE_TRADING,
            enabled=True,
            parameters={'preservation': {'enabled': True}}
        )
        live_mode = TradingMode(uuid4(), live_config, mock_portfolio)
        live_mode._preservation_manager = mock_preservation_manager
        
        await live_mode._preserve_current_state()
        
        # Verify critical priority for live trading
        call_args = mock_preservation_manager.save_model.call_args
        assert call_args.kwargs.get('priority') == PreservationPriority.CRITICAL
    
    @pytest.mark.asyncio
    async def test_backtesting_mode_preservation(self, mock_preservation_manager, mock_portfolio):
        """Test preservation for backtesting mode results"""
        # Create backtesting mode
        config = ModeConfig(
            mode_type=ModeType.BACKTESTING,
            enabled=True,
            parameters={
                'preservation': {'enabled': True},
                'backtest': {
                    'start_date': '2024-01-01',
                    'end_date': '2024-12-31'
                }
            }
        )
        
        mode = AnalysisMode(uuid4(), config, mock_portfolio)
        mode._preservation_manager = mock_preservation_manager
        
        # Simulate backtest results
        backtest_results = {
            'total_return': 0.25,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.15,
            'trades': 100
        }
        
        # Save backtest results
        await mode._preserve_backtest_results(backtest_results)
        
        # Verify preservation with backtest metadata
        call_args = mock_preservation_manager.save_model.call_args
        metadata = call_args.kwargs.get('metadata', {})
        assert 'backtest_results' in metadata
        assert metadata['backtest_results']['total_return'] == 0.25
    
    @pytest.mark.asyncio
    async def test_mode_cleanup_preserves_final_state(self, mock_preservation_manager, mock_portfolio, mode_config):
        """Test that cleanup preserves final mode state"""
        # Create mode
        mode = TradingMode(uuid4(), mode_config, mock_portfolio)
        mode._preservation_manager = mock_preservation_manager
        mode.status = ModeStatus.ACTIVE
        mode.start_time = datetime.now()
        
        # Add some metrics
        mode._record_metric('total_trades', 50)
        mode._record_metric('profit_loss', 2500.0)
        
        # Cleanup
        await mode.cleanup()
        
        # Verify final state was preserved
        assert mock_preservation_manager.save_model.called
        call_args = mock_preservation_manager.save_model.call_args
        assert 'final_state' in call_args.kwargs.get('tags', [])