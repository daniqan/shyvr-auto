"""
Test mode base preservation functionality

Tests for Phase 1.5: Mode preservation base integration functionality
Tests the core preservation hooks that need to be added to ModeBase and concrete mode classes.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from datetime import datetime
import pickle

from src.modes.base import ModeBase, TradingMode, AnalysisMode, SimulationMode, ModeType, ModeStatus, ModeConfig
from src.portfolio.base import Portfolio
from src.model_preservation.manager import PreservationManager
from src.model_preservation.base import PreservationPriority, ModelState, PreservationError
import asyncio
import json
import tempfile
import os


@pytest.fixture
def mock_portfolio():
    """Create a mock portfolio"""
    portfolio = Mock(spec=Portfolio)
    portfolio.get_balance = Mock(return_value=10000)
    portfolio.get_positions = Mock(return_value={})
    return portfolio


@pytest.fixture
def mock_preservation_manager():
    """Create a mock PreservationManager"""
    manager = Mock(spec=PreservationManager)
    manager.save_model = AsyncMock(return_value="model_123")
    manager.load_model = AsyncMock(return_value=(b"model_data", {"model_id": "model_123"}))
    manager.migrate_model = AsyncMock(return_value="migrated_model_123")
    manager.emergency_backup = AsyncMock(return_value=["backup_1", "backup_2"])
    manager.cleanup_old_versions = AsyncMock(return_value=3)
    return manager


@pytest.fixture
def mode_config():
    """Create test mode configuration with preservation enabled"""
    return ModeConfig(
        mode_type=ModeType.ANALYSIS,
        enabled=True,
        auto_start=False,
        parameters={
            'preservation': {
                'enabled': True,
                'auto_backup_on_change': True,
                'priority': 'high',
                'mode_isolation': True
            }
        }
    )


@pytest.fixture
def trading_mode_config():
    """Create trading mode configuration with critical preservation"""
    return ModeConfig(
        mode_type=ModeType.LIVE_TRADING,
        enabled=True,
        auto_start=False,
        parameters={
            'preservation': {
                'enabled': True,
                'auto_backup_on_change': True,
                'priority': 'critical',
                'mode_isolation': True
            }
        }
    )


class TestModeBasePreservationHooks:
    """Test preservation hooks in ModeBase class"""
    
    @pytest.mark.asyncio
    async def test_backup_models_on_mode_change_method_exists(self, mock_portfolio, mode_config):
        """Test that _backup_models_on_mode_change method exists and is callable"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        
        # Should have the method
        assert hasattr(mode, '_backup_models_on_mode_change')
        assert callable(getattr(mode, '_backup_models_on_mode_change'))
        
        # Should be async
        result = mode._backup_models_on_mode_change()
        assert hasattr(result, '__await__')
    
    @pytest.mark.asyncio
    async def test_initialize_preservation_manager_method_exists(self, mock_portfolio, mode_config):
        """Test that _initialize_preservation_manager method exists and is callable"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        
        # Should have the method
        assert hasattr(mode, '_initialize_preservation_manager')
        assert callable(getattr(mode, '_initialize_preservation_manager'))
        
        # Should be async
        result = mode._initialize_preservation_manager()
        assert hasattr(result, '__await__')
    
    @pytest.mark.asyncio
    async def test_load_mode_specific_models_method_exists(self, mock_portfolio, mode_config):
        """Test that _load_mode_specific_models method exists and is callable"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        
        # Should have the method
        assert hasattr(mode, '_load_mode_specific_models')
        assert callable(getattr(mode, '_load_mode_specific_models'))
        
        # Should be async
        result = mode._load_mode_specific_models()
        assert hasattr(result, '__await__')
    
    @pytest.mark.asyncio
    async def test_backup_models_on_mode_change_implementation(self, mock_portfolio, mode_config, mock_preservation_manager):
        """Test _backup_models_on_mode_change implementation details"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mode.preservation_manager = mock_preservation_manager
        mode._ml_models = {
            'lstm': Mock(),
            'dqn': Mock(),
            'transformer': Mock()
        }
        
        # Should backup all models when called
        backup_results = await mode._backup_models_on_mode_change()
        
        # Should return list of backup model IDs
        assert isinstance(backup_results, list)
        assert len(backup_results) == 3  # One for each model
        
        # Should call save_model for each ML model
        assert mode.preservation_manager.save_model.call_count == 3
        
        # Verify save calls include mode context and change tags
        for call in mode.preservation_manager.save_model.call_args_list:
            assert call.kwargs.get('mode') == 'analysis'
            assert 'mode_change' in call.kwargs.get('tags', [])
            # Note: Priority is determined by mode type and config, not hard-coded
    
    @pytest.mark.asyncio
    async def test_initialize_preservation_manager_implementation(self, mock_portfolio, mode_config):
        """Test _initialize_preservation_manager implementation"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        
        # Should initialize when preservation is enabled in config
        await mode._initialize_preservation_manager()
        
        # Should have preservation manager instance
        assert hasattr(mode, 'preservation_manager')
        assert mode.preservation_manager is not None
        
        # Should configure manager with mode-specific settings
        assert mode.preservation_manager.config.mode_isolation is True
    
    @pytest.mark.asyncio
    async def test_load_mode_specific_models_implementation(self, mock_portfolio, mode_config, mock_preservation_manager):
        """Test _load_mode_specific_models implementation"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mode.preservation_manager = mock_preservation_manager
        
        # Mock model types that should be loaded for analysis mode
        mode._required_model_types = ['lstm', 'dqn']
        
        loaded_models = await mode._load_mode_specific_models()
        
        # Should return dictionary of loaded models
        assert isinstance(loaded_models, dict)
        assert len(loaded_models) == 2
        
        # Should call load_model with mode isolation
        assert mode.preservation_manager.load_model.call_count == 2
        for call in mode.preservation_manager.load_model.call_args_list:
            assert call.kwargs.get('mode') == 'analysis'
    
    @pytest.mark.asyncio
    async def test_preservation_disabled_in_config(self, mock_portfolio):
        """Test that preservation methods handle disabled configuration"""
        config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={'preservation': {'enabled': False}}
        )
        
        mode = AnalysisMode(uuid4(), config, mock_portfolio)
        
        # Methods should exist but do nothing when preservation is disabled
        await mode._initialize_preservation_manager()
        assert not hasattr(mode, 'preservation_manager') or mode.preservation_manager is None
        
        backup_results = await mode._backup_models_on_mode_change()
        assert backup_results == []
        
        loaded_models = await mode._load_mode_specific_models()
        assert loaded_models == {}


class TestModeLifecyclePreservationHooks:
    """Test preservation hooks integrated into mode lifecycle methods"""
    
    @pytest.mark.asyncio
    async def test_start_method_calls_load_mode_specific_models(self, mock_portfolio, mode_config):
        """Test that start() method calls _load_mode_specific_models"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Mock the preservation methods
        mode._load_mode_specific_models = AsyncMock(return_value={'lstm': Mock()})
        
        await mode.start()
        
        # Should call load models during start
        mode._load_mode_specific_models.assert_called_once()
        
        # Should store loaded models
        assert hasattr(mode, '_ml_models')
        assert 'lstm' in mode._ml_models
    
    @pytest.mark.asyncio
    async def test_stop_method_calls_backup_models_on_mode_change(self, mock_portfolio, mode_config):
        """Test that stop() method calls _backup_models_on_mode_change"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode.status = ModeStatus.ACTIVE
        
        # Mock the preservation methods
        mode._backup_models_on_mode_change = AsyncMock(return_value=['backup_1', 'backup_2'])
        
        await mode.stop()
        
        # Should call backup during stop
        mode._backup_models_on_mode_change.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pause_method_preserves_mode_state(self, mock_portfolio, mode_config):
        """Test that pause() method preserves current mode state"""
        mode = SimulationMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode.status = ModeStatus.ACTIVE
        mode.virtual_balance = 15000
        mode._simulation_metrics = {'trades': 10, 'profit': 500}
        
        await mode.pause()
        
        # Should save current state during pause
        mode.preservation_manager.save_model.assert_called()
        
        # Should include mode state in metadata
        call_args = mode.preservation_manager.save_model.call_args
        metadata = call_args.kwargs.get('metadata', {})
        assert 'mode_state' in metadata
        assert metadata['mode_state']['virtual_balance'] == 15000
        assert 'pause' in call_args.kwargs.get('tags', [])
    
    @pytest.mark.asyncio
    async def test_resume_method_restores_mode_state(self, mock_portfolio, mode_config):
        """Test that resume() method restores preserved mode state"""
        mode = SimulationMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode.status = ModeStatus.PAUSED
        
        # Mock preserved state
        preserved_state = {
            'virtual_balance': 20000,
            'simulation_metrics': {'trades': 20, 'profit': 1000}
        }
        mode_data = pickle.dumps(preserved_state)
        mode.preservation_manager.load_model.return_value = (
            mode_data,
            {'metadata': {'mode_state': preserved_state}}
        )
        
        await mode.resume()
        
        # Should restore state from preservation
        assert mode.virtual_balance == 20000
        assert mode._simulation_metrics == {'trades': 20, 'profit': 1000}
        
        # Should load with correct tags
        call_args = mode.preservation_manager.load_model.call_args
        assert 'resume' in call_args.kwargs.get('tags', [])


class TestModeSpecificModelIsolation:
    """Test mode-specific model isolation functionality"""
    
    @pytest.mark.asyncio
    async def test_different_modes_use_isolated_models(self, mock_portfolio):
        """Test that different modes maintain isolated model storage"""
        # Create different mode configurations
        analysis_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={'preservation': {'enabled': True, 'mode_isolation': True}}
        )
        simulation_config = ModeConfig(
            mode_type=ModeType.SIMULATION,  
            enabled=True,
            parameters={'preservation': {'enabled': True, 'mode_isolation': True}}
        )
        
        analysis_mode = AnalysisMode(uuid4(), analysis_config, mock_portfolio)
        simulation_mode = SimulationMode(uuid4(), simulation_config, mock_portfolio)
        
        # Set preservation managers
        analysis_mode.preservation_manager = mock_preservation_manager()
        simulation_mode.preservation_manager = mock_preservation_manager()
        
        # Load models for each mode
        await analysis_mode._load_mode_specific_models()
        await simulation_mode._load_mode_specific_models()
        
        # Verify models were loaded with correct mode context
        analysis_calls = analysis_mode.preservation_manager.load_model.call_args_list
        simulation_calls = simulation_mode.preservation_manager.load_model.call_args_list
        
        # Analysis mode should load with 'analysis' mode context
        for call in analysis_calls:
            assert call.kwargs.get('mode') == 'analysis'
        
        # Simulation mode should load with 'simulation' mode context  
        for call in simulation_calls:
            assert call.kwargs.get('mode') == 'simulation'
    
    @pytest.mark.asyncio
    async def test_model_migration_between_modes(self, mock_portfolio, mode_config):
        """Test migration of models between different modes"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Should have method to migrate models between modes
        assert hasattr(mode, '_migrate_model_to_mode')
        assert callable(getattr(mode, '_migrate_model_to_mode'))
        
        # Test migration from analysis to simulation
        migrated_id = await mode._migrate_model_to_mode(
            model_type='lstm',
            version='v1.0.0',
            from_mode='analysis',
            to_mode='simulation'
        )
        
        # Should call preservation manager migrate_model
        mode.preservation_manager.migrate_model.assert_called_once_with(
            model_type='lstm',
            version='v1.0.0',  
            from_mode='analysis',
            to_mode='simulation'
        )
        
        assert migrated_id == 'migrated_model_123'
    
    @pytest.mark.asyncio
    async def test_mode_specific_preservation_priorities(self, mock_portfolio):
        """Test that different modes use appropriate preservation priorities"""
        # Live trading should use critical priority
        live_config = ModeConfig(
            mode_type=ModeType.LIVE_TRADING,
            enabled=True,
            parameters={'preservation': {'enabled': True, 'priority': 'critical'}}
        )
        
        # Analysis should use normal priority
        analysis_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={'preservation': {'enabled': True, 'priority': 'normal'}}
        )
        
        live_mode = TradingMode(uuid4(), live_config, mock_portfolio)
        analysis_mode = AnalysisMode(uuid4(), analysis_config, mock_portfolio)
        
        live_mode.preservation_manager = mock_preservation_manager()
        analysis_mode.preservation_manager = mock_preservation_manager()
        
        # Test preservation priority for live trading
        live_mode._ml_models = {'dqn': Mock()}
        await live_mode._backup_models_on_mode_change()
        
        live_call_args = live_mode.preservation_manager.save_model.call_args
        assert live_call_args.kwargs.get('priority') == PreservationPriority.CRITICAL
        
        # Test preservation priority for analysis
        analysis_mode._ml_models = {'lstm': Mock()}
        await analysis_mode._backup_models_on_mode_change()
        
        analysis_call_args = analysis_mode.preservation_manager.save_model.call_args
        assert analysis_call_args.kwargs.get('priority') == PreservationPriority.NORMAL


class TestModePreservationErrorHandling:
    """Test error handling in mode preservation functionality"""
    
    @pytest.mark.asyncio
    async def test_preservation_error_during_backup_doesnt_crash_mode(self, mock_portfolio, mode_config):
        """Test that preservation errors during backup don't crash the mode"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode.preservation_manager.save_model.side_effect = PreservationError("Storage failed")
        mode._ml_models = {'lstm': Mock()}
        
        # Should not raise exception, but return empty list
        backup_results = await mode._backup_models_on_mode_change()
        assert backup_results == []
        
        # Mode should still be functional
        assert mode.status != ModeStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_preservation_error_during_load_uses_fallback(self, mock_portfolio, mode_config):
        """Test that preservation errors during load attempt fallback mechanisms"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # First call fails, second succeeds (fallback)
        mode.preservation_manager.load_model.side_effect = [
            PreservationError("Model not found"),
            (b"fallback_data", {"model_id": "fallback_model"})
        ]
        
        loaded_models = await mode._load_mode_specific_models()
        
        # Should attempt fallback and succeed
        assert mode.preservation_manager.load_model.call_count == 2
        # First call should use primary version, second should use fallback=True
        fallback_call = mode.preservation_manager.load_model.call_args_list[1]
        assert fallback_call.kwargs.get('fallback') is True
    
    @pytest.mark.asyncio
    async def test_emergency_backup_on_critical_error(self, mock_portfolio, trading_mode_config):
        """Test that critical errors trigger emergency backup"""
        mode = TradingMode(uuid4(), trading_mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode.status = ModeStatus.ACTIVE
        mode._ml_models = {'dqn': Mock()}
        
        # Should have method to handle errors with emergency backup
        assert hasattr(mode, '_handle_error')
        assert callable(getattr(mode, '_handle_error'))
        
        # Simulate critical error
        await mode._handle_error("Critical system failure")
        
        # Should trigger emergency backup
        mode.preservation_manager.emergency_backup.assert_called_once()
        
        # Should set error status
        assert mode.status == ModeStatus.ERROR


class TestModePreservationConfiguration:
    """Test preservation configuration handling in modes"""
    
    @pytest.mark.asyncio
    async def test_preservation_config_validation(self, mock_portfolio):
        """Test that mode validates preservation configuration properly"""
        # Invalid preservation config should be handled gracefully
        invalid_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                'preservation': {
                    'enabled': True,
                    'invalid_setting': 'should_be_ignored'
                }
            }
        )
        
        mode = AnalysisMode(uuid4(), invalid_config, mock_portfolio)
        
        # Should initialize without error despite invalid settings
        await mode._initialize_preservation_manager()
        
        # Should have valid preservation manager if enabled
        assert mode.preservation_manager is not None
    
    @pytest.mark.asyncio
    async def test_preservation_settings_override_defaults(self, mock_portfolio):
        """Test that mode-specific preservation settings override defaults"""
        custom_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                'preservation': {
                    'enabled': True,
                    'priority': 'high',
                    'mode_isolation': False,  # Override default
                    'auto_backup_on_change': False  # Override default
                }
            }
        )
        
        mode = SimulationMode(uuid4(), custom_config, mock_portfolio)
        await mode._initialize_preservation_manager()
        
        # Should respect custom settings
        assert mode.preservation_manager.config.mode_isolation is False
        
        # Test backup behavior respects auto_backup setting
        mode._ml_models = {'lstm': Mock()}
        await mode.stop()
        
        # Should not auto-backup on stop when disabled
        assert not mode.preservation_manager.save_model.called


class TestModePreservationMetrics:
    """Test preservation metrics and monitoring integration"""
    
    @pytest.mark.asyncio
    async def test_preservation_metrics_recorded(self, mock_portfolio, mode_config):
        """Test that preservation operations record appropriate metrics"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._ml_models = {'lstm': Mock(), 'dqn': Mock()}
        
        # Should record metrics during backup
        await mode._backup_models_on_mode_change()
        
        # Should have recorded preservation metrics
        assert 'preservation_backup_count' in mode.metrics
        assert mode.metrics['preservation_backup_count'] == 2
        assert 'preservation_last_backup' in mode.metrics
    
    @pytest.mark.asyncio
    async def test_preservation_performance_tracking(self, mock_portfolio, mode_config):
        """Test that preservation operations track performance metrics"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Add delay to simulate preservation time
        async def slow_save(*args, **kwargs):
            import asyncio
            await asyncio.sleep(0.1)
            return "model_123"
        
        mode.preservation_manager.save_model = slow_save
        mode._ml_models = {'lstm': Mock()}
        
        start_time = datetime.now()
        await mode._backup_models_on_mode_change()
        end_time = datetime.now()
        
        # Should track preservation duration
        assert 'preservation_backup_duration_ms' in mode.metrics
        duration_ms = mode.metrics['preservation_backup_duration_ms']
        assert duration_ms >= 100  # At least 100ms due to sleep
        assert duration_ms <= (end_time - start_time).total_seconds() * 1000 + 50  # Some tolerance


class TestModeLifecycleIntegrationHooks:
    """Test integration between preservation hooks and mode lifecycle transitions"""
    
    @pytest.mark.asyncio
    async def test_start_calls_preservation_hooks_in_correct_order(self, mock_portfolio, mode_config):
        """Test that start() calls preservation hooks in the correct order"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        
        # Track method call order
        call_order = []
        
        async def track_initialize(*args, **kwargs):
            call_order.append('initialize_preservation_manager')
            mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        async def track_load(*args, **kwargs):
            call_order.append('load_mode_specific_models')
            return {'lstm': Mock()}
        
        mode._initialize_preservation_manager = track_initialize
        mode._load_mode_specific_models = track_load
        
        await mode.start()
        
        # Should initialize preservation manager before loading models
        assert call_order == ['initialize_preservation_manager', 'load_mode_specific_models']
        assert mode.status == ModeStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_stop_calls_preservation_hooks_before_cleanup(self, mock_portfolio, mode_config):
        """Test that stop() calls preservation hooks before mode cleanup"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode.status = ModeStatus.ACTIVE
        mode._ml_models = {'lstm': Mock()}
        
        # Track cleanup order
        cleanup_order = []
        
        async def track_backup(*args, **kwargs):
            cleanup_order.append('backup_models')
            return ['backup_1']
        
        def track_cleanup(*args, **kwargs):
            cleanup_order.append('cleanup_resources')
        
        mode._backup_models_on_mode_change = track_backup
        mode._cleanup_resources = track_cleanup
        
        await mode.stop()
        
        # Should backup before cleanup
        assert cleanup_order == ['backup_models', 'cleanup_resources']
        assert mode.status == ModeStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_pause_preserves_complete_mode_state(self, mock_portfolio, mode_config):
        """Test that pause() preserves complete mode state including ML models"""
        mode = SimulationMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode.status = ModeStatus.ACTIVE
        mode.virtual_balance = 25000
        mode._simulation_metrics = {'trades': 15, 'profit': 750}
        mode._ml_models = {'dqn': Mock(), 'lstm': Mock()}
        
        await mode.pause()
        
        # Should save both mode state and ML models
        assert mode.preservation_manager.save_model.call_count >= 3  # state + 2 models
        
        # Verify state preservation call
        state_calls = [call for call in mode.preservation_manager.save_model.call_args_list 
                      if 'mode_state' in call.kwargs.get('metadata', {})]
        assert len(state_calls) == 1
        
        state_metadata = state_calls[0].kwargs['metadata']['mode_state']
        assert state_metadata['virtual_balance'] == 25000
        assert state_metadata['simulation_metrics'] == {'trades': 15, 'profit': 750}
    
    @pytest.mark.asyncio
    async def test_resume_restores_models_and_state_atomically(self, mock_portfolio, mode_config):
        """Test that resume() restores both models and state in atomic operation"""
        mode = SimulationMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode.status = ModeStatus.PAUSED
        
        # Mock preserved state and models
        preserved_state = {
            'virtual_balance': 30000,
            'simulation_metrics': {'trades': 25, 'profit': 1500}
        }
        
        def mock_load_model(*args, **kwargs):
            if 'mode_state' in kwargs.get('model_type', ''):
                state_data = pickle.dumps(preserved_state)
                return (state_data, {'metadata': {'mode_state': preserved_state}})
            else:
                return (b'model_data', {'model_id': 'test_model'})
        
        mode.preservation_manager.load_model = AsyncMock(side_effect=mock_load_model)
        
        await mode.resume()
        
        # Should restore state
        assert mode.virtual_balance == 30000
        assert mode._simulation_metrics == {'trades': 25, 'profit': 1500}
        assert mode.status == ModeStatus.ACTIVE
        
        # Should load models with proper context
        assert mode.preservation_manager.load_model.call_count >= 1


class TestModePreservationConfigurationValidation:
    """Test comprehensive configuration validation and error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_preservation_priority_handled_gracefully(self, mock_portfolio):
        """Test handling of invalid preservation priority configuration"""
        invalid_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                'preservation': {
                    'enabled': True,
                    'priority': 'invalid_priority',  # Invalid priority
                    'mode_isolation': True
                }
            }
        )
        
        mode = AnalysisMode(uuid4(), invalid_config, mock_portfolio)
        
        # Should initialize with default priority when invalid
        await mode._initialize_preservation_manager()
        
        assert mode.preservation_manager is not None
        # Should default to NORMAL priority for invalid values
        mode._ml_models = {'lstm': Mock()}
        await mode._backup_models_on_mode_change()
        
        call_args = mode.preservation_manager.save_model.call_args
        assert call_args.kwargs.get('priority') == PreservationPriority.NORMAL
    
    @pytest.mark.asyncio
    async def test_missing_preservation_config_uses_defaults(self, mock_portfolio):
        """Test that missing preservation config uses sensible defaults"""
        minimal_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={}  # No preservation config
        )
        
        mode = SimulationMode(uuid4(), minimal_config, mock_portfolio)
        
        # Should initialize with defaults when config missing
        await mode._initialize_preservation_manager()
        
        # Preservation should be disabled by default
        assert not hasattr(mode, 'preservation_manager') or mode.preservation_manager is None
    
    @pytest.mark.asyncio
    async def test_configuration_inheritance_and_overrides(self, mock_portfolio):
        """Test that child modes can override parent preservation settings"""
        base_config = ModeConfig(
            mode_type=ModeType.LIVE_TRADING,
            enabled=True,
            parameters={
                'preservation': {
                    'enabled': True,
                    'priority': 'normal',
                    'mode_isolation': False
                }
            }
        )
        
        # Trading mode should override with critical priority
        mode = TradingMode(uuid4(), base_config, mock_portfolio)
        await mode._initialize_preservation_manager()
        
        # Should have method to get effective preservation config
        assert hasattr(mode, '_get_effective_preservation_config')
        effective_config = mode._get_effective_preservation_config()
        
        # Trading mode should enforce critical priority and isolation
        assert effective_config['priority'] == 'critical'
        assert effective_config['mode_isolation'] is True


class TestAsynchronousOperationHandling:
    """Test handling of asynchronous preservation operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_preservation_operations_handled(self, mock_portfolio, mode_config):
        """Test that concurrent preservation operations are handled safely"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._ml_models = {'lstm': Mock(), 'dqn': Mock(), 'transformer': Mock()}
        
        # Add delay to simulate concurrent operations
        async def slow_save(*args, **kwargs):
            await asyncio.sleep(0.05)
            return f"model_{kwargs.get('model_type', 'unknown')}"
        
        mode.preservation_manager.save_model = slow_save
        
        # Start multiple preservation operations concurrently
        tasks = [
            mode._backup_models_on_mode_change(),
            mode._backup_models_on_mode_change(),
            mode._backup_models_on_mode_change()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_preservation_operation_timeout_handling(self, mock_portfolio, mode_config):
        """Test handling of preservation operations that timeout"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._ml_models = {'lstm': Mock()}
        
        # Mock a hanging preservation operation
        async def hanging_save(*args, **kwargs):
            await asyncio.sleep(10)  # Long delay
            return "model_123"
        
        mode.preservation_manager.save_model = hanging_save
        
        # Should have timeout mechanism in backup operation
        start_time = asyncio.get_event_loop().time()
        result = await mode._backup_models_on_mode_change()
        end_time = asyncio.get_event_loop().time()
        
        # Should timeout and return gracefully within reasonable time
        assert (end_time - start_time) < 5.0  # Should timeout before 5 seconds
        assert result == []  # Should return empty list on timeout
    
    @pytest.mark.asyncio
    async def test_async_model_loading_with_fallback(self, mock_portfolio, mode_config):
        """Test asynchronous model loading with fallback mechanisms"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Mock primary load failure, fallback success
        load_attempts = []
        
        async def track_load_attempts(*args, **kwargs):
            load_attempts.append(kwargs)
            if len(load_attempts) == 1:
                raise PreservationError("Primary storage unavailable")
            else:
                return (b"fallback_model_data", {"model_id": "fallback_model"})
        
        mode.preservation_manager.load_model = track_load_attempts
        mode._required_model_types = ['lstm']
        
        loaded_models = await mode._load_mode_specific_models()
        
        # Should have attempted primary then fallback
        assert len(load_attempts) == 2
        assert not load_attempts[0].get('fallback', False)  # Primary attempt
        assert load_attempts[1].get('fallback', True)  # Fallback attempt
        
        # Should successfully load from fallback
        assert 'lstm' in loaded_models


class TestModelMigrationFunctionality:
    """Test comprehensive model migration between modes"""
    
    @pytest.mark.asyncio
    async def test_model_migration_with_validation(self, mock_portfolio, mode_config):
        """Test model migration includes validation and compatibility checks"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Should validate migration compatibility
        assert hasattr(mode, '_validate_migration_compatibility')
        
        # Test valid migration
        is_compatible = await mode._validate_migration_compatibility(
            model_type='lstm',
            from_mode='analysis',
            to_mode='simulation'
        )
        assert is_compatible is True
        
        # Test invalid migration (e.g., live trading to analysis)
        is_compatible = await mode._validate_migration_compatibility(
            model_type='dqn',
            from_mode='live_trading',
            to_mode='analysis'
        )
        assert is_compatible is False
    
    @pytest.mark.asyncio
    async def test_bulk_model_migration(self, mock_portfolio, mode_config):
        """Test migration of multiple models between modes"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Should support bulk migration
        assert hasattr(mode, '_migrate_models_bulk')
        
        models_to_migrate = [
            {'model_type': 'lstm', 'version': 'v1.0.0'},
            {'model_type': 'dqn', 'version': 'v2.1.0'},
            {'model_type': 'transformer', 'version': 'v1.5.0'}
        ]
        
        migration_results = await mode._migrate_models_bulk(
            models=models_to_migrate,
            from_mode='analysis',
            to_mode='simulation'
        )
        
        # Should return results for all models
        assert len(migration_results) == 3
        assert all('migrated_id' in result for result in migration_results)
        
        # Should call migrate_model for each model
        assert mode.preservation_manager.migrate_model.call_count == 3
    
    @pytest.mark.asyncio
    async def test_migration_rollback_on_failure(self, mock_portfolio, mode_config):
        """Test that failed migrations can be rolled back"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Mock partial migration failure
        migration_calls = []
        
        async def track_migration(*args, **kwargs):
            migration_calls.append(kwargs)
            if len(migration_calls) == 2:  # Fail on second migration
                raise PreservationError("Migration failed")
            return f"migrated_{kwargs['model_type']}"
        
        mode.preservation_manager.migrate_model = track_migration
        
        models_to_migrate = [
            {'model_type': 'lstm', 'version': 'v1.0.0'},
            {'model_type': 'dqn', 'version': 'v2.1.0'},
            {'model_type': 'transformer', 'version': 'v1.5.0'}
        ]
        
        # Should handle migration failure and rollback
        with pytest.raises(PreservationError):
            await mode._migrate_models_bulk(
                models=models_to_migrate,
                from_mode='analysis',
                to_mode='simulation',
                rollback_on_failure=True
            )
        
        # Should have attempted rollback (cleanup of successful migrations)
        # This would involve calling rollback methods on preservation manager


class TestDatabaseIntegrationPreservation:
    """Test database integration with preservation manager"""
    
    @pytest.mark.asyncio
    async def test_database_metadata_operations(self, mock_portfolio, mode_config):
        """Test that preservation operations properly update database metadata"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._ml_models = {'lstm': Mock()}
        
        # Mock database operations in preservation manager
        mode.preservation_manager.update_model_metadata = AsyncMock()
        
        await mode._backup_models_on_mode_change()
        
        # Should update database metadata for preserved models
        mode.preservation_manager.update_model_metadata.assert_called()
        
        call_args = mode.preservation_manager.update_model_metadata.call_args
        metadata = call_args.kwargs
        assert 'mode' in metadata
        assert 'preservation_timestamp' in metadata
        assert 'model_version' in metadata
    
    @pytest.mark.asyncio
    async def test_database_transaction_integrity(self, mock_portfolio, mode_config):
        """Test that preservation operations maintain database transaction integrity"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._ml_models = {'lstm': Mock(), 'dqn': Mock()}
        
        # Mock database transaction failure
        mode.preservation_manager.save_model.side_effect = [
            "model_1",  # First save succeeds
            Exception("Transaction failed")  # Second save fails
        ]
        
        # Should handle database transaction failures gracefully
        backup_results = await mode._backup_models_on_mode_change()
        
        # Should return partial results or handle rollback
        assert isinstance(backup_results, list)
        # Implementation should ensure database consistency
    
    @pytest.mark.asyncio
    async def test_database_connection_recovery(self, mock_portfolio, mode_config):
        """Test recovery from database connection failures"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Mock database connection failure then recovery
        connection_attempts = []
        
        async def mock_save_with_recovery(*args, **kwargs):
            connection_attempts.append(len(connection_attempts) + 1)
            if len(connection_attempts) == 1:
                raise Exception("Connection lost")
            return "model_recovered"
        
        mode.preservation_manager.save_model = mock_save_with_recovery
        mode._ml_models = {'lstm': Mock()}
        
        # Should retry database operations on connection failure
        backup_results = await mode._backup_models_on_mode_change()
        
        # Should succeed after retry
        assert len(connection_attempts) >= 2  # Initial attempt + retry
        assert backup_results == ["model_recovered"]


class TestRealWorldErrorScenarios:
    """Test handling of real-world error scenarios"""
    
    @pytest.mark.asyncio
    async def test_network_failure_during_remote_backup(self, mock_portfolio, mode_config):
        """Test handling of network failures during remote backup operations"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._ml_models = {'lstm': Mock()}
        
        # Mock network failure
        mode.preservation_manager.save_model.side_effect = [
            ConnectionError("Network unreachable"),
            "local_backup_123"  # Local fallback succeeds
        ]
        
        backup_results = await mode._backup_models_on_mode_change()
        
        # Should fallback to local storage on network failure
        assert backup_results == ["local_backup_123"]
        assert mode.preservation_manager.save_model.call_count == 2
    
    @pytest.mark.asyncio
    async def test_storage_corruption_recovery(self, mock_portfolio, mode_config):
        """Test recovery from storage corruption scenarios"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Mock storage corruption on load
        mode.preservation_manager.load_model.side_effect = [
            PreservationError("Corrupted model data"),
            (b"recovered_data", {'model_id': 'recovered_model'})
        ]
        
        loaded_models = await mode._load_mode_specific_models()
        
        # Should attempt recovery from corruption
        assert mode.preservation_manager.load_model.call_count == 2
        # Should use data integrity verification in real implementation
    
    @pytest.mark.asyncio
    async def test_disk_space_exhaustion_handling(self, mock_portfolio, mode_config):
        """Test handling of disk space exhaustion during model preservation"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._ml_models = {'lstm': Mock(), 'dqn': Mock(), 'transformer': Mock()}
        
        # Mock disk space exhaustion
        mode.preservation_manager.save_model.side_effect = OSError("No space left on device")
        mode.preservation_manager.cleanup_old_versions = AsyncMock(return_value=5)
        
        # Should trigger cleanup and retry on disk space issues
        backup_results = await mode._backup_models_on_mode_change()
        
        # Should attempt cleanup when storage fails
        mode.preservation_manager.cleanup_old_versions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_pressure_during_large_model_operations(self, mock_portfolio, mode_config):
        """Test handling of memory pressure during large model operations"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Create mock large model that could cause memory issues
        large_model = Mock()
        large_model.state_dict = Mock(return_value={'large_tensor': b'x' * 1000000})  # 1MB mock
        mode._ml_models = {'large_model': large_model}
        
        # Should handle memory pressure gracefully
        backup_results = await mode._backup_models_on_mode_change()
        
        # Should implement streaming or chunked operations for large models
        assert isinstance(backup_results, list)
        # Real implementation should use memory-efficient serialization


class TestPerformanceAndTimingConstraints:
    """Test performance requirements and timing constraints"""
    
    @pytest.mark.asyncio
    async def test_backup_operation_performance_requirements(self, mock_portfolio, mode_config):
        """Test that backup operations meet performance requirements"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._ml_models = {f'model_{i}': Mock() for i in range(10)}  # Multiple models
        
        start_time = asyncio.get_event_loop().time()
        await mode._backup_models_on_mode_change()
        end_time = asyncio.get_event_loop().time()
        
        duration = end_time - start_time
        
        # Should complete backup operations within reasonable time
        assert duration < 2.0  # Should complete within 2 seconds for 10 models
        
        # Should record performance metrics
        assert 'preservation_backup_duration_ms' in mode.metrics
        assert mode.metrics['preservation_backup_duration_ms'] < 2000
    
    @pytest.mark.asyncio
    async def test_model_loading_performance_constraints(self, mock_portfolio, mode_config):
        """Test that model loading meets performance constraints"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._required_model_types = ['lstm', 'dqn', 'transformer']
        
        # Add realistic delay to simulate loading time
        async def realistic_load(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms per model
            return (b'model_data', {'model_id': 'test_model'})
        
        mode.preservation_manager.load_model = realistic_load
        
        start_time = asyncio.get_event_loop().time()
        loaded_models = await mode._load_mode_specific_models()
        end_time = asyncio.get_event_loop().time()
        
        # Should use parallel loading for better performance
        duration = end_time - start_time
        assert duration < 0.5  # Should be much faster than sequential (0.3s)
        
        # Should load all required models
        assert len(loaded_models) == 3
    
    @pytest.mark.asyncio
    async def test_mode_transition_timing_requirements(self, mock_portfolio, mode_config):
        """Test that mode transitions meet timing requirements"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Test start transition timing
        start_time = asyncio.get_event_loop().time()
        await mode.start()
        start_duration = asyncio.get_event_loop().time() - start_time
        
        # Mode start should be fast
        assert start_duration < 1.0  # Less than 1 second
        
        # Test stop transition timing
        mode._ml_models = {'lstm': Mock(), 'dqn': Mock()}
        stop_start = asyncio.get_event_loop().time()
        await mode.stop()
        stop_duration = asyncio.get_event_loop().time() - stop_start
        
        # Mode stop should complete quickly
        assert stop_duration < 2.0  # Less than 2 seconds including backup


class TestModeIsolationBoundaries:
    """Test mode isolation boundaries and security"""
    
    @pytest.mark.asyncio
    async def test_mode_isolation_prevents_cross_mode_access(self, mock_portfolio):
        """Test that mode isolation prevents unauthorized cross-mode access"""
        # Create two modes with different configurations
        analysis_config = ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={'preservation': {'enabled': True, 'mode_isolation': True}}
        )
        
        trading_config = ModeConfig(
            mode_type=ModeType.LIVE_TRADING,
            enabled=True,
            parameters={'preservation': {'enabled': True, 'mode_isolation': True}}
        )
        
        analysis_mode = AnalysisMode(uuid4(), analysis_config, mock_portfolio)
        trading_mode = TradingMode(uuid4(), trading_config, mock_portfolio)
        
        analysis_mode.preservation_manager = mock_preservation_manager()
        trading_mode.preservation_manager = mock_preservation_manager()
        
        # Analysis mode should not be able to access trading mode models
        with pytest.raises(PreservationError, match="Access denied"):
            await analysis_mode._load_mode_specific_models(mode_filter='live_trading')
        
        # Trading mode should not access analysis mode models
        with pytest.raises(PreservationError, match="Access denied"):
            await trading_mode._load_mode_specific_models(mode_filter='analysis')
    
    @pytest.mark.asyncio
    async def test_mode_isolation_model_namespacing(self, mock_portfolio, mode_config):
        """Test that mode isolation properly namespaces models"""
        mode = AnalysisMode(uuid4(), mode_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        mode._ml_models = {'lstm': Mock()}
        
        # Should include mode namespace in model storage
        await mode._backup_models_on_mode_change()
        
        call_args = mode.preservation_manager.save_model.call_args
        
        # Should include mode-specific namespace or prefix
        assert call_args.kwargs.get('mode') == 'analysis'
        assert 'namespace' in call_args.kwargs or 'mode' in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_isolation_boundary_enforcement(self, mock_portfolio):
        """Test enforcement of isolation boundaries between modes"""
        # Create mode with strict isolation
        strict_config = ModeConfig(
            mode_type=ModeType.LIVE_TRADING,
            enabled=True,
            parameters={
                'preservation': {
                    'enabled': True,
                    'mode_isolation': True,
                    'strict_isolation': True  # Stricter isolation
                }
            }
        )
        
        mode = TradingMode(uuid4(), strict_config, mock_portfolio)
        mock_manager = mock_preservation_manager()
        mode.preservation_manager = mock_manager
        
        # Should have isolation validation method
        assert hasattr(mode, '_validate_isolation_boundary')
        
        # Test boundary validation
        is_allowed = await mode._validate_isolation_boundary(
            requested_mode='analysis',
            operation='load_model'
        )
        assert is_allowed is False
        
        # Same mode should be allowed
        is_allowed = await mode._validate_isolation_boundary(
            requested_mode='live_trading',
            operation='load_model'
        )
        assert is_allowed is True