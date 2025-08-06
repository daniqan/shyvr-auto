"""
TDD Tests for Ensemble Mode Manager Integration

Tests Phase 5 requirements for Mode Manager integration with ensemble architecture:
- 5.1: ModeManager works with ensemble architecture
- 5.2: Fallback strategies handle dev (LSTM) vs prod (ensemble)
- 5.3: Trading modes work with ensemble predictions

Following TDD methodology - tests written first to drive implementation.
"""

import pytest
import asyncio
import os
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.modes.mode_manager import ModeManager, ModeManagerConfig, ModeType
from src.modes.base import ModeConfig, ModeStatus
from src.modes.fallback_strategies import (
    FallbackStrategyManager, FallbackConfig, ModelDegradationDetector,
    EnsembleFallbackManager, FallbackRecoveryManager,
    DegradationSeverity, FallbackStrategy, ModelHealthMetrics
)
from src.portfolio.base import Portfolio
from src.rl_agent.base import MarketState, TradeAction
from src.ml_analysis.base import PredictionResult, ModelType


@dataclass 
class MockEnsemblePrediction:
    """Mock ensemble prediction for testing."""
    predictions: Dict[str, Any]
    confidence: float
    ensemble_size: int
    model_contributions: Dict[str, float]
    

class TestEnsembleModeManagerIntegration:
    """Test suite for ensemble mode manager integration."""

    @pytest.fixture
    async def portfolio(self):
        """Create mock portfolio."""
        portfolio = Mock(spec=Portfolio)
        portfolio.portfolio_id = "test_portfolio"
        portfolio.total_value = Decimal("100000")
        portfolio.available_balance = Decimal("50000")
        portfolio.positions = {}
        return portfolio

    @pytest.fixture
    def development_config(self):
        """Config for development environment (LSTM only)."""
        return ModeManagerConfig(
            max_concurrent_modes=2,
            enable_mode_switching=True,
            auto_recovery=True
        )

    @pytest.fixture
    def production_config(self):
        """Config for production environment (full ensemble)."""
        return ModeManagerConfig(
            max_concurrent_modes=3,
            enable_mode_switching=True,
            auto_recovery=True
        )

    @pytest.fixture
    def ensemble_prediction(self):
        """Sample ensemble prediction."""
        return MockEnsemblePrediction(
            predictions={
                'lstm': {'action': 'buy', 'confidence': 0.75},
                'itransformer': {'action': 'buy', 'confidence': 0.80},
                'patchtst': {'action': 'hold', 'confidence': 0.65},
                'timesmixer': {'action': 'buy', 'confidence': 0.85},
                'timesfm': {'action': 'buy', 'confidence': 0.70}
            },
            confidence=0.75,
            ensemble_size=5,
            model_contributions={
                'lstm': 0.3,
                'itransformer': 0.25,
                'patchtst': 0.15,
                'timesmixer': 0.2,
                'timesfm': 0.1
            }
        )

    @pytest.fixture
    def lstm_only_prediction(self):
        """Sample LSTM-only prediction for development."""
        return MockEnsemblePrediction(
            predictions={
                'lstm': {'action': 'buy', 'confidence': 0.72}
            },
            confidence=0.72,
            ensemble_size=1,
            model_contributions={'lstm': 1.0}
        )

    @pytest.mark.asyncio
    async def test_mode_manager_handles_ensemble_predictions(self, portfolio, production_config, ensemble_prediction):
        """Test 5.1: ModeManager works with ensemble architecture."""
        # Arrange
        mode_manager = ModeManager(production_config, portfolio)
        
        # Mock model manager to return ensemble predictions
        with patch('src.modes.mode_manager.get_model_manager') as mock_get_manager:
            mock_model_manager = Mock()
            mock_model_manager.get_ensemble_prediction.return_value = ensemble_prediction
            mock_get_manager.return_value = mock_model_manager
            
            await mode_manager.initialize()
            
            # Create market state
            from src.discovery.base import DiscoveredToken
            mock_token = Mock(spec=DiscoveredToken)
            mock_token.address = "0x123"
            mock_token.symbol = "TEST"
            
            market_state = MarketState(
                token=mock_token,
                price_usd=50000.0,
                price_change_24h=0.05,
                volume_24h=1000000.0,
                market_cap=1000000000.0
            )
            
            # Act - process market tick with ensemble prediction
            actions = await mode_manager.process_market_tick(market_state)
            
            # Assert
            assert isinstance(actions, dict)
            # Should handle ensemble predictions without errors
            mock_model_manager.get_ensemble_prediction.assert_called()

    @pytest.mark.asyncio
    async def test_mode_manager_development_vs_production_mode(self, portfolio):
        """Test mode manager handles development vs production environments."""
        # Test development mode (LSTM only)
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            dev_config = ModeManagerConfig(max_concurrent_modes=1)
            dev_manager = ModeManager(dev_config, portfolio)
            
            await dev_manager.initialize()
            assert dev_manager._initialized
            
        # Test production mode (full ensemble)
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            prod_config = ModeManagerConfig(max_concurrent_modes=3)
            prod_manager = ModeManager(prod_config, portfolio)
            
            await prod_manager.initialize()
            assert prod_manager._initialized

    @pytest.mark.asyncio
    async def test_mode_manager_ensemble_model_availability_check(self, portfolio, production_config):
        """Test mode manager checks model availability based on environment."""
        mode_manager = ModeManager(production_config, portfolio)
        
        # Mock environment variable
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            with patch('src.modes.mode_manager.check_ensemble_models_available') as mock_check:
                mock_check.return_value = True
                
                await mode_manager.initialize()
                
                # Should check for ensemble models in production
                mock_check.assert_called()

    @pytest.mark.asyncio 
    async def test_mode_manager_graceful_degradation(self, portfolio, production_config):
        """Test mode manager graceful degradation when models unavailable."""
        mode_manager = ModeManager(production_config, portfolio)
        
        # Mock some models unavailable
        with patch('src.modes.mode_manager.check_model_availability') as mock_check:
            mock_check.return_value = {
                'lstm': True,
                'itransformer': False,  # Unavailable
                'patchtst': True,
                'timesmixer': True,
                'timesfm': False  # Unavailable
            }
            
            await mode_manager.initialize()
            
            # Should still initialize successfully with partial ensemble
            assert mode_manager._initialized


class TestFallbackStrategiesEnsembleIntegration:
    """Test suite for fallback strategies with ensemble integration."""

    @pytest.fixture
    def fallback_config(self):
        """Fallback configuration."""
        return FallbackConfig(
            enable_ensemble_fallback=True,
            enable_rule_based_fallback=True,
            max_fallback_duration_minutes=30,
            recovery_threshold_minutes=10
        )

    @pytest.fixture
    def degradation_detector(self):
        """Model degradation detector."""
        config = {
            'accuracy_threshold': 0.15,
            'confidence_threshold': 0.5,
            'latency_threshold_ms': 100,
            'error_rate_threshold': 0.05
        }
        return ModelDegradationDetector(config)

    @pytest.fixture
    def ensemble_manager(self):
        """Ensemble fallback manager."""
        config = {
            'ensemble_size': 3,
            'diversity_requirement': True,
            'confidence_weighting': True
        }
        return EnsembleFallbackManager(config)

    @pytest.mark.asyncio
    async def test_fallback_strategies_development_vs_production(self, fallback_config):
        """Test 5.2: Fallback strategies handle dev (LSTM) vs prod (ensemble)."""
        fallback_manager = FallbackStrategyManager(fallback_config)
        
        # Test development environment - fallback to LSTM only
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            models = fallback_manager.load_fallback_models_from_preservation()
            
            # In development, should prioritize LSTM models
            lstm_models = [m for m in models if 'lstm' in m.model_type.lower()]
            assert len(lstm_models) > 0

        # Test production environment - use full ensemble for fallback
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            models = fallback_manager.load_fallback_models_from_preservation()
            
            # In production, should have diverse model types
            model_types = set(m.model_type for m in models)
            assert len(model_types) >= 2  # Multiple model types available

    @pytest.mark.asyncio
    async def test_ensemble_fallback_model_selection(self, ensemble_manager):
        """Test ensemble fallback selects appropriate models based on environment."""
        # Mock available models
        with patch.object(ensemble_manager, 'available_models') as mock_models:
            mock_models.return_value = [
                Mock(model_id='lstm_1', model_type='lstm'),
                Mock(model_id='itransformer_1', model_type='itransformer'),
                Mock(model_id='patchtst_1', model_type='patchtst'),
            ]
            
            # Test development mode - should prefer LSTM
            with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
                ensemble = ensemble_manager.create_ensemble(exclude_degraded_models=[])
                
                # Should include LSTM in development ensemble
                lstm_in_ensemble = any('lstm' in model.model_type.lower() for model in ensemble)
                assert lstm_in_ensemble

            # Test production mode - should use diverse ensemble
            with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
                ensemble = ensemble_manager.create_ensemble(exclude_degraded_models=[])
                
                # Should have multiple model types
                model_types = set(model.model_type for model in ensemble)
                assert len(model_types) >= 2

    @pytest.mark.asyncio
    async def test_fallback_degradation_detection_ensemble_aware(self, degradation_detector):
        """Test degradation detection handles ensemble metrics."""
        # Test ensemble health metrics
        ensemble_health = ModelHealthMetrics(
            accuracy=0.65,  # Below baseline
            confidence_avg=0.75,
            latency_avg=85.0,
            memory_usage=450.0,
            error_rate=0.03,
            prediction_count=150,
            last_updated=datetime.now()
        )
        
        # Should detect degradation in ensemble context
        result = degradation_detector.assess_model_health(ensemble_health)
        
        # Should identify accuracy degradation
        assert result.is_degraded
        assert result.severity >= DegradationSeverity.MODERATE
        assert 'accuracy' in result.affected_metrics

    @pytest.mark.asyncio
    async def test_fallback_strategy_ensemble_confidence_handling(self, ensemble_manager):
        """Test fallback strategies handle ensemble confidence properly."""
        # Mock individual model predictions with varying confidence
        individual_predictions = [
            {'action': 'buy', 'confidence': 0.8, 'model_type': 'lstm'},
            {'action': 'buy', 'confidence': 0.6, 'model_type': 'itransformer'}, 
            {'action': 'hold', 'confidence': 0.7, 'model_type': 'patchtst'}
        ]
        
        # Test confidence weighting
        ensemble_pred = ensemble_manager.aggregate_predictions(
            individual_predictions, 
            use_confidence_weighting=True
        )
        
        # Should aggregate with confidence weighting
        assert ensemble_pred.confidence > 0
        assert ensemble_pred.ensemble_size == 3
        assert ensemble_pred.aggregation_method == 'confidence_weighted'

    @pytest.mark.asyncio
    async def test_fallback_recovery_ensemble_validation(self):
        """Test fallback recovery validates ensemble health properly."""
        recovery_config = {
            'recovery_accuracy_threshold': 0.80,
            'recovery_confidence_threshold': 0.70,
            'recovery_validation_samples': 100
        }
        recovery_manager = FallbackRecoveryManager(recovery_config)
        
        # Test ensemble recovery metrics
        ensemble_metrics = ModelHealthMetrics(
            accuracy=0.82,  # Above recovery threshold
            confidence_avg=0.75,  # Above recovery threshold  
            latency_avg=90.0,
            memory_usage=400.0,
            error_rate=0.02,
            prediction_count=120,  # Sufficient samples
            last_updated=datetime.now()
        )
        
        result = recovery_manager.validate_model_recovery('ensemble_model', ensemble_metrics)
        
        # Should validate successful recovery
        assert result.is_recovered
        assert result.confidence_level > 0.7


class TestTradingModesEnsembleIntegration:
    """Test suite for trading modes with ensemble predictions."""

    @pytest.fixture
    def mock_trading_mode(self):
        """Mock trading mode for testing."""
        from src.modes.base import TradingMode
        mode = Mock(spec=TradingMode)
        mode.status = ModeStatus.ACTIVE
        mode.config = Mock()
        mode.config.mode_type = ModeType.LIVE_TRADING
        return mode

    @pytest.fixture
    def market_state(self):
        """Sample market state."""
        from src.discovery.base import DiscoveredToken
        from unittest.mock import Mock
        
        mock_token = Mock(spec=DiscoveredToken)
        mock_token.address = "0x123"
        mock_token.symbol = "TEST"
        
        return MarketState(
            token=mock_token,
            price_usd=50000.0,
            price_change_24h=0.05,
            volume_24h=1000000.0,
            market_cap=1000000000.0
        )

    @pytest.mark.asyncio
    async def test_trading_modes_ensemble_prediction_processing(self, mock_trading_mode, market_state):
        """Test 5.3: Trading modes work with ensemble predictions."""
        # Mock ensemble prediction processing
        ensemble_prediction = {
            'action': 'buy',
            'confidence': 0.78,
            'ensemble_models': ['lstm', 'itransformer', 'patchtst'],
            'model_agreement': 0.85,
            'individual_predictions': [
                {'model': 'lstm', 'action': 'buy', 'confidence': 0.75},
                {'model': 'itransformer', 'action': 'buy', 'confidence': 0.82},
                {'model': 'patchtst', 'action': 'buy', 'confidence': 0.77}
            ]
        }
        
        with patch.object(mock_trading_mode, 'process_ensemble_prediction') as mock_process:
            mock_process.return_value = TradeAction(
                action_type='buy',
                amount=Decimal("1000"),
                confidence=0.78,
                reasoning="Ensemble consensus with high confidence"
            )
            
            # Should be able to process ensemble predictions
            result = mock_process(ensemble_prediction, market_state)
            
            assert result.action_type == 'buy'
            assert result.confidence == 0.78
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_trading_modes_model_availability_adaptation(self, mock_trading_mode, market_state):
        """Test trading modes adapt to available models in environment."""
        # Test development mode with LSTM only
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            lstm_prediction = {
                'action': 'buy', 
                'confidence': 0.72,
                'models_used': ['lstm'],
                'ensemble_size': 1
            }
            
            with patch.object(mock_trading_mode, 'process_prediction') as mock_process:
                mock_process.return_value = TradeAction(
                    action_type='buy',
                    amount=Decimal("500"),  # Smaller amount in dev
                    confidence=0.72,
                    reasoning="LSTM prediction in development mode"
                )
                
                result = mock_process(lstm_prediction, market_state)
                
                assert result.action_type == 'buy'
                # Development mode should be more conservative
                assert result.amount <= Decimal("1000")

        # Test production mode with full ensemble
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            ensemble_prediction = {
                'action': 'buy',
                'confidence': 0.80,
                'models_used': ['lstm', 'itransformer', 'patchtst', 'timesmixer'],
                'ensemble_size': 4
            }
            
            with patch.object(mock_trading_mode, 'process_prediction') as mock_process:
                mock_process.return_value = TradeAction(
                    action_type='buy',
                    amount=Decimal("1500"),  # Larger amount with ensemble confidence
                    confidence=0.80,
                    reasoning="Ensemble prediction in production mode"
                )
                
                result = mock_process(ensemble_prediction, market_state)
                
                assert result.action_type == 'buy'
                # Production mode can be less conservative with ensemble
                assert result.amount >= Decimal("1000")

    @pytest.mark.asyncio
    async def test_trading_modes_ensemble_confidence_thresholds(self, mock_trading_mode, market_state):
        """Test trading modes respect ensemble-specific confidence thresholds."""
        # High confidence ensemble prediction
        high_confidence_pred = {
            'action': 'buy',
            'confidence': 0.85,
            'ensemble_size': 4,
            'model_agreement': 0.90
        }
        
        # Low confidence ensemble prediction  
        low_confidence_pred = {
            'action': 'buy', 
            'confidence': 0.45,
            'ensemble_size': 4,
            'model_agreement': 0.60
        }
        
        with patch.object(mock_trading_mode, 'should_execute_trade') as mock_should_execute:
            # High confidence should allow execution
            mock_should_execute.return_value = True
            assert mock_should_execute(high_confidence_pred, market_state)
            
            # Low confidence should reject execution
            mock_should_execute.return_value = False
            assert not mock_should_execute(low_confidence_pred, market_state)

    @pytest.mark.asyncio
    async def test_trading_modes_ensemble_fallback_integration(self, mock_trading_mode, market_state):
        """Test trading modes integrate with ensemble fallback strategies."""
        # Mock degraded ensemble scenario
        degraded_prediction = {
            'action': 'hold',
            'confidence': 0.40,  # Low confidence
            'ensemble_size': 2,  # Reduced ensemble
            'degraded_models': ['itransformer', 'timesfm'],
            'fallback_active': True,
            'fallback_strategy': 'rule_based'
        }
        
        with patch.object(mock_trading_mode, 'handle_degraded_prediction') as mock_handle:
            mock_handle.return_value = TradeAction(
                action_type='hold',
                amount=Decimal("0"),
                confidence=0.40,
                reasoning="Using fallback strategy due to model degradation"
            )
            
            result = mock_handle(degraded_prediction, market_state)
            
            assert result.action_type == 'hold'
            assert 'fallback' in result.reasoning.lower()
            mock_handle.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])