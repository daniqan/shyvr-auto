"""
Test preservation integration for ModelManager

Tests for Phase 1.5: Integration with Existing System
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from src.ml_analysis.model_manager import ModelManager
from src.ml_analysis.base import ModelType
from src.model_preservation.manager import PreservationManager, PreservationConfig
from src.model_preservation.base import PreservationError


@pytest.fixture
def mock_preservation_manager():
    """Create a mock preservation manager"""
    manager = Mock(spec=PreservationManager)
    manager.save_model = AsyncMock(return_value="model_123")
    
    # Mock preserved state data
    import pickle
    preserved_state = {'test': 'state', 'layer1': 'weights'}
    model_data = pickle.dumps(preserved_state)
    
    manager.load_model = AsyncMock(return_value=(
        model_data, 
        {
            "model_id": "model_123",
            "version": "v1.0.0",
            "metadata": {"performance": {"accuracy": 0.85}}
        }
    ))
    return manager


@pytest.fixture
def model_manager_with_preservation(mock_preservation_manager):
    """Create a model manager with preservation integration"""
    config = {
        'model_dir': 'models',
        'preservation': {
            'enabled': True,
            'gcs_bucket': 'test-bucket',
            'backup_interval_hours': 6,
            'max_versions_per_model': 10
        }
    }
    
    # Mock all dependencies
    with patch('src.ml_analysis.model_manager.FeatureEngineer'), \
         patch('src.ml_analysis.model_manager.LSTMPricePredictor'), \
         patch('src.ml_analysis.model_manager.structlog.get_logger') as mock_logger, \
         patch('src.ml_analysis.model_manager.Path'):
        
        mock_logger.return_value.bind.return_value = Mock()
        
        manager = ModelManager(config)
        
        # Set up internal state
        manager._models = {}
        manager._model_weights = {}
        manager._model_performance = {}
        manager._current_mode = 'analysis'
        
        # Inject mock preservation manager
        manager._preservation_manager = mock_preservation_manager
        
        # Mock the model directory
        manager.model_dir = Mock()
        manager.model_dir.__truediv__ = Mock(return_value="models/lstm_model.pt")
        
        return manager


class TestModelManagerPreservationIntegration:
    """Test ModelManager preservation integration"""
    
    @pytest.mark.asyncio
    async def test_save_model_with_preservation(self, model_manager_with_preservation, mock_preservation_manager):
        """Test that save_model triggers preservation"""
        # Setup
        model_type = ModelType.LSTM
        
        # Create a mock model with state_dict
        mock_model = Mock()
        mock_model.save_model = Mock(return_value=True)
        mock_model.model = Mock()
        mock_model.model.state_dict = Mock(return_value={'test': 'state'})
        
        model_manager_with_preservation._models[model_type] = mock_model
        model_manager_with_preservation._model_performance[model_type] = {'accuracy': 0.85}
        
        # Execute
        await model_manager_with_preservation._save_model(model_type)
        
        # Verify model is saved locally first
        mock_model.save_model.assert_called_once()
        
        # Verify preservation is triggered
        model_manager_with_preservation._preservation_manager.save_model.assert_called_once()
        
        # Check preservation call arguments
        call_args = model_manager_with_preservation._preservation_manager.save_model.call_args
        assert call_args.kwargs['model_type'] == 'lstm'
        assert 'model_data' in call_args.kwargs
        assert 'mode' in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_save_model_preservation_fallback(self, model_manager_with_preservation, mock_preservation_manager):
        """Test that local save succeeds even if preservation fails"""
        # Setup
        model_type = ModelType.LSTM
        model_manager_with_preservation._models[model_type] = Mock()
        model_manager_with_preservation._models[model_type].save_model = Mock(return_value=True)
        
        # Make preservation fail
        mock_preservation_manager.save_model.side_effect = PreservationError("Storage failed")
        
        # Execute - should not raise
        await model_manager_with_preservation._save_model(model_type)
        
        # Verify model is still saved locally
        model_manager_with_preservation._models[model_type].save_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_model_with_preservation_fallback(self, model_manager_with_preservation, mock_preservation_manager):
        """Test that load_model falls back to preserved version when local load fails"""
        # Setup
        model_type = ModelType.LSTM
        filepath = Path('models/lstm_model.pt')
        
        # Mock model that doesn't support loading
        mock_model = Mock()
        mock_model.load_model = Mock(return_value=False)  # Local load fails
        mock_model.model = Mock()
        mock_model.model.load_state_dict = Mock()
        mock_model.is_trained = False
        model_manager_with_preservation._models[model_type] = mock_model
        
        # Mock the model dir path
        mock_path = Mock()
        mock_path.exists.return_value = True
        model_manager_with_preservation.model_dir.__truediv__.return_value = mock_path
        
        # Execute
        results = await model_manager_with_preservation.load_models([model_type])
        
        # Verify preservation fallback was attempted
        model_manager_with_preservation._preservation_manager.load_model.assert_called_once_with(
            model_type='lstm',
            version=None,
            mode='analysis',
            fallback=True
        )
        
        # Verify the model was restored from preservation
        assert results[model_type] is True
        assert mock_model.model.load_state_dict.called
    
    @pytest.mark.asyncio
    async def test_load_model_no_preservation_when_local_succeeds(self, model_manager_with_preservation, mock_preservation_manager):
        """Test that preservation is not used when local load succeeds"""
        # Setup
        model_type = ModelType.LSTM
        filepath = Path('models/lstm_model.pt')
        
        # Mock successful local load
        mock_model = Mock()
        mock_model.load_model = Mock(return_value=True)
        model_manager_with_preservation._models[model_type] = mock_model
        
        # Mock the model dir path
        mock_path = Mock()
        mock_path.exists.return_value = True
        model_manager_with_preservation.model_dir.__truediv__.return_value = mock_path
        
        # Execute
        results = await model_manager_with_preservation.load_models([model_type])
        
        # Verify preservation was not called
        model_manager_with_preservation._preservation_manager.load_model.assert_not_called()
        
        # Verify result
        assert results[model_type] is True
    
    @pytest.mark.asyncio
    async def test_preservation_disabled(self):
        """Test that preservation is not used when disabled in config"""
        # Setup config with preservation disabled
        config = {
            'model_dir': 'models',
            'preservation': {
                'enabled': False
            }
        }
        
        with patch('src.ml_analysis.model_manager.FeatureEngineer'), \
             patch('src.ml_analysis.model_manager.LSTMPricePredictor'), \
             patch('src.ml_analysis.model_manager.structlog.get_logger') as mock_logger, \
             patch('src.ml_analysis.model_manager.Path'):
            
            mock_logger.return_value.bind.return_value = Mock()
            manager = ModelManager(config)
            
            # Verify preservation manager is not created
            assert manager._preservation_manager is None
    
    @pytest.mark.asyncio
    async def test_save_model_preserves_model_data(self, model_manager_with_preservation, mock_preservation_manager):
        """Test that actual model data is preserved correctly"""
        # Setup
        model_type = ModelType.LSTM
        
        # Mock model with state_dict
        mock_model = Mock()
        mock_model.save_model = Mock(return_value=True)
        mock_state_dict = {'layer1': 'weights', 'layer2': 'biases'}
        
        # Mock the model's internal network
        mock_model.model = Mock()
        mock_model.model.state_dict = Mock(return_value=mock_state_dict)
        
        model_manager_with_preservation._models[model_type] = mock_model
        
        # Execute
        await model_manager_with_preservation._save_model(model_type)
        
        # Verify preservation was called with serialized model data
        call_args = mock_preservation_manager.save_model.call_args
        assert 'model_data' in call_args.kwargs
        assert isinstance(call_args.kwargs['model_data'], bytes)
    
    @pytest.mark.asyncio
    async def test_load_model_restores_model_state(self, model_manager_with_preservation, mock_preservation_manager):
        """Test that model state is properly restored from preservation"""
        # Setup
        model_type = ModelType.LSTM
        
        # Mock preserved model data
        preserved_state = {'layer1': 'preserved_weights'}
        import pickle
        preserved_data = pickle.dumps(preserved_state)
        mock_preservation_manager.load_model.return_value = (preserved_data, {"model_id": "model_123"})
        
        # Mock model
        mock_model = Mock()
        mock_model.load_model = Mock(return_value=False)  # Local load fails
        mock_model.model = Mock()
        mock_model.model.load_state_dict = Mock()
        
        model_manager_with_preservation._models[model_type] = mock_model
        
        # Mock the model dir path
        mock_path = Mock()
        mock_path.exists.return_value = True
        model_manager_with_preservation.model_dir.__truediv__.return_value = mock_path
        
        # Execute
        results = await model_manager_with_preservation.load_models([model_type])
        
        # Verify model state was restored
        assert mock_model.model.load_state_dict.called
        assert results[model_type] is True
    
    @pytest.mark.asyncio
    async def test_preservation_with_mode_context(self, model_manager_with_preservation, mock_preservation_manager):
        """Test that preservation includes current mode context"""
        # Setup
        model_type = ModelType.LSTM
        
        # Create a mock model with state_dict
        mock_model = Mock()
        mock_model.save_model = Mock(return_value=True)
        mock_model.model = Mock()
        mock_model.model.state_dict = Mock(return_value={'test': 'state'})
        
        model_manager_with_preservation._models[model_type] = mock_model
        
        # Set current mode
        model_manager_with_preservation._current_mode = 'simulation'
        
        # Execute
        await model_manager_with_preservation._save_model(model_type)
        
        # Verify mode was passed to preservation
        call_args = model_manager_with_preservation._preservation_manager.save_model.call_args
        assert call_args.kwargs.get('mode') == 'simulation'
    
    @pytest.mark.asyncio
    async def test_preservation_with_performance_metadata(self, model_manager_with_preservation, mock_preservation_manager):
        """Test that preservation includes model performance metadata"""
        # Setup
        model_type = ModelType.LSTM
        
        # Create a mock model with state_dict
        mock_model = Mock()
        mock_model.save_model = Mock(return_value=True)
        mock_model.model = Mock()
        mock_model.model.state_dict = Mock(return_value={'test': 'state'})
        
        model_manager_with_preservation._models[model_type] = mock_model
        
        # Set performance metrics
        model_manager_with_preservation._model_performance[model_type] = {
            'accuracy': 0.85,
            'predictions_made': 100,
            'last_updated': datetime.now().timestamp()
        }
        
        # Execute
        await model_manager_with_preservation._save_model(model_type)
        
        # Verify metadata was included
        call_args = model_manager_with_preservation._preservation_manager.save_model.call_args
        metadata = call_args.kwargs.get('metadata', {})
        assert 'performance' in metadata
        assert metadata['performance']['accuracy'] == 0.85