"""
Tests for ML-RL Safety Bridge Integration - Phase 3.2.4

Comprehensive test suite for ML-RL safety bridge system including:
- ML prediction validation with confidence thresholds
- RL action validation with safety constraints
- Cross-system consistency validation
- Model confidence threshold enforcement
- Fallback strategy integration
- Safety metrics tracking

Following TDD methodology with no mocks in production code.
"""

import pytest
import asyncio
import os
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from typing import Dict, List, Any, Optional

# Set up test environment before imports
os.environ["RLTE_ENVIRONMENT"] = "test"
os.environ["RLTE_CONFIG_FILE"] = "config/config.test.yaml"
os.environ["SECRET_KEY"] = "test_secret_key_1234567890_secure"

from src.safety.ml_rl_safety_bridge import (
    MLRLSafetyBridge,
    MLPredictionValidator,
    RLActionValidator,
    CrossSystemValidator,
    ValidationContext,
    MLPredictionValidation,
    RLActionValidation,
    CrossSystemSafetyCheck,
    SafetyConstraints,
    SafetyValidationResult,
    ValidationResult,
    SafetyViolationType,
    ModelSafetyStatus,
    ModelSafetyMetrics,
    SafetyBridgeConfig
)
from src.ml_analysis.base import PredictionResult, ModelType
from src.rl_agent.base import TradeAction, MarketState
from src.safety.unified_safety_state_manager import UnifiedSafetyStateManager
from src.safety.emergency_stop_controller import EmergencyStopController
from src.modes.fallback_strategies import FallbackSystemIntegration


@pytest.fixture
def safety_constraints():
    """Create test safety constraints."""
    return SafetyConstraints(
        min_confidence_threshold=0.7,
        max_position_size=Decimal('10000.0'),
        max_risk_per_trade=0.05,
        max_daily_loss=0.10,
        max_drawdown=0.15,
        min_prediction_samples=50,
        max_prediction_uncertainty=0.3,
        prediction_bounds=(-1.0, 1.0),
        max_action_magnitude=1.0,
        action_bounds={
            'buy': (0.0, 1.0),
            'sell': (0.0, 1.0),
            'hold': (0.0, 1.0)
        },
        volatility_threshold=0.05,
        correlation_limit=0.8,
        concentration_limit=0.3,
        min_accuracy=0.7,
        max_latency_ms=100.0,
        max_error_rate=0.05
    )


@pytest.fixture
def validation_context():
    """Create test validation context."""
    return ValidationContext(
        model_id="test_model_123",
        model_type=ModelType.LSTM,
        timestamp=datetime.utcnow(),
        market_conditions={
            'volatility': 0.03,
            'trend': 'bullish',
            'volume_ratio': 1.2
        },
        current_positions={
            'BTC': Decimal('2.0'),
            'ETH': Decimal('10.0')
        },
        portfolio_value=Decimal('100000.0'),
        recent_performance={
            'accuracy': 0.75,
            'precision': 0.72,
            'recall': 0.78
        }
    )


@pytest.fixture
def sample_ml_prediction():
    """Create sample ML prediction for testing."""
    return PredictionResult(
        prediction_id=str(uuid4()),
        model_name="test_predictor",
        model_version="v1.2.0",
        prediction_type="price_direction",
        predicted_value=0.75,  # Bullish signal
        confidence_score=0.85,
        features={
            'price_momentum': 0.15,
            'volume_ratio': 1.3,
            'volatility': 0.25
        },
        timestamp=datetime.utcnow(),
        data_quality_score=0.9
    )


@pytest.fixture
def mock_safety_manager():
    """Create mock safety state manager."""
    manager = MagicMock()
    manager.is_initialized = True
    manager.get_current_state = AsyncMock()
    manager.get_current_state.return_value = MagicMock(
        emergency_stop_active=False,
        risk_level=0.3,
        circuit_breakers_active=0
    )
    return manager


@pytest.fixture
def mock_emergency_stop():
    """Create mock emergency stop controller."""
    controller = MagicMock()
    controller.is_emergency_active = AsyncMock(return_value=False)
    controller.get_emergency_status = AsyncMock(return_value={
        'active': False,
        'reason': None,
        'triggered_at': None
    })
    return controller


@pytest.fixture
def mock_fallback_integration():
    """Create mock fallback system integration."""
    integration = MagicMock()
    integration.is_degraded_mode_active = AsyncMock(return_value=False)
    integration.get_active_fallbacks = AsyncMock(return_value=[])
    integration.activate_fallback = AsyncMock(return_value=True)
    return integration


class TestSafetyConstraints:
    """Test safety constraints configuration."""
    
    def test_safety_constraints_creation(self, safety_constraints):
        """Test safety constraints creation."""
        assert safety_constraints.min_confidence_threshold == 0.7
        assert safety_constraints.max_position_size == Decimal('10000.0')
        assert safety_constraints.max_risk_per_trade == 0.05
        assert 'buy' in safety_constraints.action_bounds
        assert safety_constraints.volatility_threshold == 0.05
    
    def test_safety_constraints_defaults(self):
        """Test safety constraints with default values."""
        constraints = SafetyConstraints()
        
        assert constraints.min_confidence_threshold == 0.6
        assert constraints.max_position_size == Decimal("10000.0")
        assert constraints.max_risk_per_trade == 0.02
        assert len(constraints.action_bounds) == 3
        assert 'buy' in constraints.action_bounds
        assert 'sell' in constraints.action_bounds
        assert 'hold' in constraints.action_bounds


class TestValidationContext:
    """Test validation context creation and handling."""
    
    def test_validation_context_creation(self, validation_context):
        """Test validation context creation."""
        assert validation_context.model_id == "test_model_123"
        assert validation_context.model_type == ModelType.LSTM
        assert validation_context.timestamp is not None
        assert isinstance(validation_context.market_conditions, dict)
        assert isinstance(validation_context.current_positions, dict)
        assert validation_context.portfolio_value == Decimal('100000.0')
    
    def test_validation_context_defaults(self):
        """Test validation context with minimal required fields."""
        context = ValidationContext(
            model_id="test_model",
            model_type=ModelType.LINEAR_REGRESSION,
            timestamp=datetime.utcnow(),
            market_conditions={}
        )
        
        assert context.model_id == "test_model"
        assert isinstance(context.current_positions, dict)
        assert context.portfolio_value == Decimal("0.0")
        assert isinstance(context.recent_performance, dict)


class TestMLPredictionValidator:
    """Test ML prediction validation."""
    
    @pytest.fixture
    def ml_validator(self, safety_constraints):
        """Create ML prediction validator for testing."""
        return MLPredictionValidator(safety_constraints)
    
    @pytest.mark.asyncio
    async def test_validate_prediction_success(self, ml_validator, sample_ml_prediction, validation_context):
        """Test successful ML prediction validation."""
        validation = await ml_validator.validate_prediction(sample_ml_prediction, validation_context)
        
        assert isinstance(validation, MLPredictionValidation)
        assert validation.original_prediction == sample_ml_prediction
        assert isinstance(validation.validation_result, SafetyValidationResult)
        assert validation.validation_result.result == ValidationResult.APPROVED
        assert validation.validation_result.confidence_score == 0.85
        assert len(validation.validation_result.violation_types) == 0
    
    @pytest.mark.asyncio
    async def test_validate_prediction_low_confidence(self, ml_validator, sample_ml_prediction, validation_context):
        """Test ML prediction validation with low confidence."""
        # Create prediction with low confidence
        low_confidence_prediction = PredictionResult(
            prediction_id=str(uuid4()),
            model_name="test_predictor",
            model_version="v1.0.0",
            prediction_type="price_direction", 
            predicted_value=0.3,
            confidence_score=0.5,  # Below threshold
            features={'momentum': 0.1},
            timestamp=datetime.utcnow(),
            data_quality_score=0.6
        )
        
        validation = await ml_validator.validate_prediction(low_confidence_prediction, validation_context)
        
        assert validation.validation_result.result in [ValidationResult.REJECTED, ValidationResult.CONDITIONAL]
        assert SafetyViolationType.CONFIDENCE_TOO_LOW in validation.validation_result.violation_types
        assert len(validation.validation_result.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_validate_prediction_out_of_bounds(self, ml_validator, sample_ml_prediction, validation_context):
        """Test ML prediction validation with out-of-bounds values."""
        # Create prediction with out-of-bounds value
        oob_prediction = PredictionResult(
            prediction_id=str(uuid4()),
            model_name="test_predictor",
            model_version="v1.0.0",
            prediction_type="price_direction",
            predicted_value=2.5,  # Outside (-1.0, 1.0) bounds
            confidence_score=0.8,
            features={'momentum': 0.2},
            timestamp=datetime.utcnow(),
            data_quality_score=0.8
        )
        
        validation = await ml_validator.validate_prediction(oob_prediction, validation_context)
        
        assert validation.validation_result.result in [ValidationResult.REJECTED, ValidationResult.CONDITIONAL]
        assert SafetyViolationType.PREDICTION_OUT_OF_BOUNDS in validation.validation_result.violation_types


class TestRLActionValidator:
    """Test RL action validation."""
    
    @pytest.fixture
    def rl_validator(self, safety_constraints):
        """Create RL action validator for testing."""
        return RLActionValidator(safety_constraints)
    
    @pytest.mark.asyncio
    async def test_validate_action_success(self, rl_validator, validation_context):
        """Test successful RL action validation."""
        action = TradeAction.BUY
        
        validation = await rl_validator.validate_action(action, validation_context)
        
        assert isinstance(validation, RLActionValidation)
        assert validation.original_action == action
        assert isinstance(validation.validation_result, SafetyValidationResult)
        assert validation.validation_result.result == ValidationResult.APPROVED
        assert len(validation.validation_result.violation_types) == 0
    
    @pytest.mark.asyncio
    async def test_validate_action_with_constraints(self, rl_validator, validation_context):
        """Test RL action validation with various constraints."""
        # Test different actions
        actions = [TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD]
        
        for action in actions:
            validation = await rl_validator.validate_action(action, validation_context)
            assert isinstance(validation, RLActionValidation)
            assert validation.original_action == action
    
    @pytest.mark.asyncio
    async def test_validate_action_risk_limits(self, rl_validator, validation_context):
        """Test RL action validation with risk limit considerations."""
        # Create context with high risk positions
        high_risk_context = ValidationContext(
            model_id="test_model",
            model_type=ModelType.TRANSFORMER,
            timestamp=datetime.utcnow(),
            market_conditions={'volatility': 0.15},  # High volatility
            current_positions={'BTC': Decimal('9500.0')},  # Near position limit
            portfolio_value=Decimal('100000.0')
        )
        
        validation = await rl_validator.validate_action(TradeAction.BUY, high_risk_context)
        
        # Should have risk considerations
        assert validation.validation_result.risk_assessment is not None
        assert len(validation.validation_result.risk_assessment) > 0


class TestCrossSystemValidator:
    """Test cross-system validation."""
    
    @pytest.fixture
    def cross_validator(self, safety_constraints):
        """Create cross-system validator for testing."""
        return CrossSystemValidator(safety_constraints)
    
    @pytest.mark.asyncio
    async def test_validate_consistency_success(self, cross_validator, sample_ml_prediction, validation_context):
        """Test successful cross-system consistency validation."""
        # Create consistent ML prediction and RL action
        ml_prediction = sample_ml_prediction  # Bullish (0.75)
        rl_action = TradeAction.BUY  # Consistent with bullish prediction
        
        consistency_check = await cross_validator.validate_consistency(
            ml_prediction, rl_action, validation_context
        )
        
        assert isinstance(consistency_check, CrossSystemSafetyCheck)
        assert consistency_check.ml_prediction == ml_prediction
        assert consistency_check.rl_action == rl_action
        assert consistency_check.is_consistent is True
        assert consistency_check.consistency_score > 0.5
    
    @pytest.mark.asyncio
    async def test_validate_consistency_conflict(self, cross_validator, validation_context):
        """Test cross-system consistency validation with conflicts."""
        # Create conflicting ML prediction and RL action
        bearish_prediction = PredictionResult(
            prediction_id=str(uuid4()),
            model_name="test_predictor",
            model_version="v1.0.0",
            prediction_type="price_direction",
            predicted_value=-0.6,  # Bearish signal
            confidence_score=0.8,
            features={'momentum': -0.2},
            timestamp=datetime.utcnow(),
            data_quality_score=0.8
        )
        
        bullish_action = TradeAction.BUY  # Conflicts with bearish prediction
        
        consistency_check = await cross_validator.validate_consistency(
            bearish_prediction, bullish_action, validation_context
        )
        
        assert consistency_check.is_consistent is False
        assert consistency_check.consistency_score < 0.5
        assert len(consistency_check.safety_concerns) > 0


class TestMLRLSafetyBridge:
    """Test ML-RL Safety Bridge main coordinator."""
    
    @pytest.fixture
    def safety_bridge_config(self):
        """Create safety bridge configuration."""
        return SafetyBridgeConfig(
            enable_ml_validation=True,
            enable_rl_validation=True,
            enable_cross_validation=True,
            fallback_on_violation=True,
            log_all_validations=True,
            validation_timeout_seconds=30.0,
            max_concurrent_validations=10
        )
    
    @pytest.fixture
    def safety_bridge(self, safety_constraints, safety_bridge_config, mock_safety_manager, 
                     mock_emergency_stop, mock_fallback_integration):
        """Create ML-RL Safety Bridge for testing."""
        return MLRLSafetyBridge(
            constraints=safety_constraints,
            config=safety_bridge_config,
            safety_state_manager=mock_safety_manager,
            emergency_stop_controller=mock_emergency_stop,
            fallback_integration=mock_fallback_integration
        )
    
    @pytest.mark.asyncio
    async def test_bridge_initialization(self, safety_bridge):
        """Test safety bridge initialization."""
        await safety_bridge.initialize()
        
        assert safety_bridge.ml_validator is not None
        assert safety_bridge.rl_validator is not None
        assert safety_bridge.cross_validator is not None
        assert safety_bridge.is_initialized is True
    
    @pytest.mark.asyncio  
    async def test_validate_ml_prediction(self, safety_bridge, sample_ml_prediction, validation_context):
        """Test ML prediction validation through bridge."""
        await safety_bridge.initialize()
        
        validation = await safety_bridge.validate_ml_prediction(sample_ml_prediction, validation_context)
        
        assert isinstance(validation, MLPredictionValidation)
        assert validation.validation_result.result == ValidationResult.APPROVED
        
        # Check metrics were recorded
        metrics = await safety_bridge.get_validation_metrics()
        assert metrics['total_ml_validations'] > 0
    
    @pytest.mark.asyncio
    async def test_validate_rl_action(self, safety_bridge, validation_context):
        """Test RL action validation through bridge."""
        await safety_bridge.initialize()
        
        action = TradeAction.BUY
        validation = await safety_bridge.validate_rl_action(action, validation_context)
        
        assert isinstance(validation, RLActionValidation)
        assert validation.validation_result.result == ValidationResult.APPROVED
        
        # Check metrics were recorded
        metrics = await safety_bridge.get_validation_metrics()
        assert metrics['total_rl_validations'] > 0
    
    @pytest.mark.asyncio
    async def test_validate_cross_system_consistency(self, safety_bridge, sample_ml_prediction, validation_context):
        """Test cross-system consistency validation through bridge."""
        await safety_bridge.initialize()
        
        action = TradeAction.BUY
        consistency_check = await safety_bridge.validate_cross_system_consistency(
            sample_ml_prediction, action, validation_context
        )
        
        assert isinstance(consistency_check, CrossSystemSafetyCheck)
        assert consistency_check.is_consistent is True
        
        # Check metrics were recorded
        metrics = await safety_bridge.get_validation_metrics()
        assert metrics['total_consistency_checks'] > 0
    
    @pytest.mark.asyncio
    async def test_emergency_stop_integration(self, safety_bridge, sample_ml_prediction, validation_context):
        """Test emergency stop integration."""
        await safety_bridge.initialize()
        
        # Mock emergency stop active
        safety_bridge.emergency_stop_controller.is_emergency_active.return_value = True
        
        validation = await safety_bridge.validate_ml_prediction(sample_ml_prediction, validation_context)
        
        # Should be rejected due to emergency stop
        assert validation.validation_result.result == ValidationResult.REJECTED
    
    @pytest.mark.asyncio
    async def test_bridge_status_reporting(self, safety_bridge):
        """Test bridge status reporting."""
        await safety_bridge.initialize()
        
        status = await safety_bridge.get_bridge_status()
        
        assert status is not None
        assert 'is_initialized' in status
        assert 'validation_stats' in status
        assert 'safety_status' in status
        assert 'active_constraints' in status
        assert status['is_initialized'] is True
    
    @pytest.mark.asyncio
    async def test_safety_metrics_tracking(self, safety_bridge, sample_ml_prediction, validation_context):
        """Test safety metrics tracking."""
        await safety_bridge.initialize()
        
        # Perform multiple validations
        await safety_bridge.validate_ml_prediction(sample_ml_prediction, validation_context)
        await safety_bridge.validate_rl_action(TradeAction.BUY, validation_context)
        await safety_bridge.validate_cross_system_consistency(sample_ml_prediction, TradeAction.BUY, validation_context)
        
        metrics = await safety_bridge.get_validation_metrics()
        
        assert metrics is not None
        assert 'total_validations' in metrics
        assert 'successful_validations' in metrics
        assert 'failed_validations' in metrics
        assert 'validation_latency_ms' in metrics
        assert metrics['total_validations'] >= 3


class TestValidationResults:
    """Test validation result classes and enums."""
    
    def test_safety_validation_result_creation(self):
        """Test safety validation result creation."""
        result = SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=0.85,
            violation_types=[],
            safety_score=0.9,
            risk_assessment={'volatility_risk': 0.2},
            recommended_actions=['monitor_closely'],
            fallback_suggestions=[],
            validation_details={'processing_time_ms': 45.2},
            requires_human_approval=False
        )
        
        assert result.result == ValidationResult.APPROVED
        assert result.confidence_score == 0.85
        assert result.safety_score == 0.9
        assert len(result.violation_types) == 0
        assert 'volatility_risk' in result.risk_assessment
    
    def test_ml_prediction_validation_creation(self, sample_ml_prediction):
        """Test ML prediction validation creation."""
        validation_result = SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=0.85,
            safety_score=0.9
        )
        
        ml_validation = MLPredictionValidation(
            original_prediction=sample_ml_prediction,
            validation_result=validation_result,
            adjusted_prediction=None,
            safety_adjustments={}
        )
        
        assert ml_validation.original_prediction == sample_ml_prediction
        assert ml_validation.validation_result == validation_result
        assert ml_validation.adjusted_prediction is None
    
    def test_rl_action_validation_creation(self):
        """Test RL action validation creation."""
        validation_result = SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=0.75,
            safety_score=0.85
        )
        
        rl_validation = RLActionValidation(
            original_action=TradeAction.BUY,
            validation_result=validation_result,
            adjusted_action=None,
            safety_adjustments={}
        )
        
        assert rl_validation.original_action == TradeAction.BUY
        assert rl_validation.validation_result == validation_result
        assert rl_validation.adjusted_action is None


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_concurrent_validations(self, safety_bridge, sample_ml_prediction, validation_context):
        """Test concurrent validation handling."""
        await safety_bridge.initialize()
        
        # Create multiple validation tasks
        tasks = []
        for i in range(5):
            task = safety_bridge.validate_ml_prediction(sample_ml_prediction, validation_context)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result.validation_result.result == ValidationResult.APPROVED
        
        # Check metrics reflect all validations
        metrics = await safety_bridge.get_validation_metrics()
        assert metrics['total_ml_validations'] >= 5
    
    @pytest.mark.asyncio
    async def test_validation_timeout_handling(self, safety_bridge, sample_ml_prediction, validation_context):
        """Test validation timeout handling."""
        await safety_bridge.initialize()
        
        # Set very short timeout for testing
        safety_bridge.config.validation_timeout_seconds = 0.001
        
        # This should either complete quickly or timeout gracefully
        validation = await safety_bridge.validate_ml_prediction(sample_ml_prediction, validation_context)
        
        # Should have a validation result regardless
        assert isinstance(validation, MLPredictionValidation)
        assert validation.validation_result is not None


if __name__ == "__main__":
    pytest.main([__file__])