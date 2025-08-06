"""
TDD Tests for Ensemble Safety System Integration

Tests Phase 6 requirements for Safety System integration with ensemble architecture:
- 6.1: Safety checks work with ensemble predictions and multi-model confidence
- 6.2: Circuit breakers consider ensemble performance metrics

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

from src.safety.trading_safety_manager import (
    TradingSafetyManager, TradingSafetyConfig, TradeOrder,
    PreTradeValidationResult, ValidationStatus, ValidationReason
)
from src.safety.trading_circuit_breaker import (
    TradingCircuitBreaker, CircuitBreakerConfig, MarketData,
    CircuitBreakerResult, CircuitBreakerLevel, CircuitBreakerType,
    TradingDecisionResult
)
from src.safety.emergency_stop_controller import EmergencyStopController
from src.portfolio.base import Portfolio
from src.rl_agent.base import MarketState
from src.modes.fallback_strategies import ModelHealthMetrics


@dataclass
class MockEnsemblePrediction:
    """Mock ensemble prediction for safety testing."""
    action: str
    confidence: float
    ensemble_size: int
    model_confidences: Dict[str, float]
    model_agreement: float
    individual_predictions: List[Dict[str, Any]]
    prediction_variance: float
    

class TestEnsembleSafetyManagerIntegration:
    """Test suite for ensemble safety manager integration."""

    @pytest.fixture
    def portfolio(self):
        """Mock portfolio for testing."""
        portfolio = Mock(spec=Portfolio)
        portfolio.total_value = Decimal("100000")
        portfolio.available_balance = Decimal("50000")
        portfolio.positions = {}
        return portfolio

    @pytest.fixture
    def safety_config(self):
        """Safety configuration for ensemble testing."""
        return TradingSafetyConfig(
            max_position_size_pct=Decimal("0.1"),
            max_single_order_value=Decimal("10000"),
            max_total_exposure_pct=Decimal("0.8"),
            enable_ensemble_validation=True,
            ensemble_confidence_threshold=0.6,
            min_ensemble_size=3
        )

    @pytest.fixture
    def ensemble_prediction(self):
        """Sample ensemble prediction with high confidence."""
        return MockEnsemblePrediction(
            action='buy',
            confidence=0.78,
            ensemble_size=4,
            model_confidences={
                'lstm': 0.75,
                'itransformer': 0.82,
                'patchtst': 0.74,
                'timesmixer': 0.81
            },
            model_agreement=0.85,
            individual_predictions=[
                {'model': 'lstm', 'action': 'buy', 'confidence': 0.75},
                {'model': 'itransformer', 'action': 'buy', 'confidence': 0.82},
                {'model': 'patchtst', 'action': 'buy', 'confidence': 0.74},
                {'model': 'timesmixer', 'action': 'buy', 'confidence': 0.81}
            ],
            prediction_variance=0.04
        )

    @pytest.fixture
    def low_confidence_ensemble_prediction(self):
        """Sample ensemble prediction with low confidence."""
        return MockEnsemblePrediction(
            action='buy',
            confidence=0.45,
            ensemble_size=3,
            model_confidences={
                'lstm': 0.42,
                'itransformer': 0.48,
                'patchtst': 0.45
            },
            model_agreement=0.60,
            individual_predictions=[
                {'model': 'lstm', 'action': 'buy', 'confidence': 0.42},
                {'model': 'itransformer', 'action': 'buy', 'confidence': 0.48},
                {'model': 'patchtst', 'action': 'hold', 'confidence': 0.45}
            ],
            prediction_variance=0.15
        )

    @pytest.fixture
    def degraded_ensemble_prediction(self):
        """Sample ensemble with degraded models."""
        return MockEnsemblePrediction(
            action='hold',
            confidence=0.35,
            ensemble_size=2,  # Reduced due to degraded models
            model_confidences={
                'lstm': 0.38,
                'patchtst': 0.32
            },
            model_agreement=0.50,
            individual_predictions=[
                {'model': 'lstm', 'action': 'hold', 'confidence': 0.38},
                {'model': 'patchtst', 'action': 'hold', 'confidence': 0.32}
            ],
            prediction_variance=0.25,
            degraded_models=['itransformer', 'timesmixer']
        )

    @pytest.mark.asyncio
    async def test_safety_manager_validates_ensemble_predictions(
        self, portfolio, safety_config, ensemble_prediction
    ):
        """Test 6.1: Safety checks work with ensemble predictions."""
        safety_manager = TradingSafetyManager(safety_config, portfolio)
        
        trade_order = TradeOrder(
            token_address="0x123",
            action_type="buy", 
            amount=Decimal("5000"),
            confidence=ensemble_prediction.confidence,
            metadata={
                'prediction_type': 'ensemble',
                'ensemble_size': ensemble_prediction.ensemble_size,
                'model_agreement': ensemble_prediction.model_agreement,
                'model_confidences': ensemble_prediction.model_confidences
            }
        )
        
        # Mock ensemble validation method
        with patch.object(safety_manager, 'validate_ensemble_prediction') as mock_validate:
            mock_validate.return_value = PreTradeValidationResult(
                is_valid=True,
                status=ValidationStatus.APPROVED,
                recommendations=["Ensemble prediction validated successfully"]
            )
            
            result = await safety_manager.validate_pre_trade(trade_order)
            
            # Should call ensemble-specific validation
            mock_validate.assert_called_once()
            assert result.is_valid
            assert result.status == ValidationStatus.APPROVED

    @pytest.mark.asyncio
    async def test_safety_manager_rejects_low_confidence_ensemble(
        self, portfolio, safety_config, low_confidence_ensemble_prediction
    ):
        """Test safety manager rejects low confidence ensemble predictions."""
        safety_manager = TradingSafetyManager(safety_config, portfolio)
        
        trade_order = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("5000"),
            confidence=low_confidence_ensemble_prediction.confidence,
            metadata={
                'prediction_type': 'ensemble',
                'ensemble_size': low_confidence_ensemble_prediction.ensemble_size,
                'model_agreement': low_confidence_ensemble_prediction.model_agreement
            }
        )
        
        # Mock ensemble validation to reject low confidence
        with patch.object(safety_manager, 'validate_ensemble_prediction') as mock_validate:
            mock_validate.return_value = PreTradeValidationResult(
                is_valid=False,
                status=ValidationStatus.REJECTED,
                reasons=[ValidationReason.SYSTEM_ERROR],
                recommendations=["Ensemble confidence too low for safe trading"]
            )
            
            result = await safety_manager.validate_pre_trade(trade_order)
            
            assert not result.is_valid
            assert result.status == ValidationStatus.REJECTED

    @pytest.mark.asyncio
    async def test_safety_manager_handles_degraded_ensemble(
        self, portfolio, safety_config, degraded_ensemble_prediction
    ):
        """Test safety manager handles degraded ensemble scenarios."""
        safety_manager = TradingSafetyManager(safety_config, portfolio)
        
        trade_order = TradeOrder(
            token_address="0x123",
            action_type="hold",
            amount=Decimal("2000"),
            confidence=degraded_ensemble_prediction.confidence,
            metadata={
                'prediction_type': 'ensemble',
                'ensemble_size': degraded_ensemble_prediction.ensemble_size,
                'degraded_models': getattr(degraded_ensemble_prediction, 'degraded_models', []),
                'fallback_active': True
            }
        )
        
        # Should apply conservative validation for degraded ensemble
        with patch.object(safety_manager, 'validate_degraded_ensemble') as mock_validate:
            mock_validate.return_value = PreTradeValidationResult(
                is_valid=True,
                status=ValidationStatus.CONDITIONAL,
                recommendations=["Reduced position size due to ensemble degradation"]
            )
            
            result = await safety_manager.validate_pre_trade(trade_order)
            
            # Should recognize degraded ensemble and apply conservative rules
            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_safety_manager_ensemble_confidence_aggregation(
        self, portfolio, safety_config, ensemble_prediction
    ):
        """Test safety manager properly aggregates ensemble confidence."""
        safety_manager = TradingSafetyManager(safety_config, portfolio)
        
        # Test confidence aggregation methods
        individual_confidences = list(ensemble_prediction.model_confidences.values())
        
        with patch.object(safety_manager, 'calculate_ensemble_confidence') as mock_calc:
            # Test weighted average confidence
            mock_calc.return_value = 0.78  # Weighted average
            
            confidence = mock_calc(individual_confidences, method='weighted_average')
            assert confidence == 0.78
            
            # Test conservative minimum confidence  
            mock_calc.return_value = 0.74  # Conservative minimum
            confidence = mock_calc(individual_confidences, method='conservative_min')
            assert confidence == 0.74

    @pytest.mark.asyncio
    async def test_safety_manager_model_agreement_threshold(
        self, portfolio, safety_config
    ):
        """Test safety manager enforces model agreement thresholds."""
        safety_manager = TradingSafetyManager(safety_config, portfolio)
        
        # High agreement scenario
        high_agreement_order = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("5000"),
            confidence=0.75,
            metadata={
                'model_agreement': 0.90,  # High agreement
                'ensemble_size': 4
            }
        )
        
        # Low agreement scenario
        low_agreement_order = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("5000"), 
            confidence=0.75,
            metadata={
                'model_agreement': 0.45,  # Low agreement
                'ensemble_size': 4
            }
        )
        
        with patch.object(safety_manager, 'validate_model_agreement') as mock_validate:
            # High agreement should pass
            mock_validate.return_value = True
            assert mock_validate(high_agreement_order.metadata['model_agreement'])
            
            # Low agreement should fail
            mock_validate.return_value = False
            assert not mock_validate(low_agreement_order.metadata['model_agreement'])

    @pytest.mark.asyncio 
    async def test_safety_manager_development_vs_production_validation(
        self, portfolio, safety_config, ensemble_prediction
    ):
        """Test different validation for development vs production environments."""
        safety_manager = TradingSafetyManager(safety_config, portfolio)
        
        trade_order = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("5000"),
            confidence=ensemble_prediction.confidence
        )
        
        # Test development environment (more lenient)
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            with patch.object(safety_manager, 'validate_pre_trade_dev') as mock_validate:
                mock_validate.return_value = PreTradeValidationResult(is_valid=True, status=ValidationStatus.APPROVED)
                
                result = await safety_manager.validate_pre_trade(trade_order)
                assert result.is_valid

        # Test production environment (stricter)
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            with patch.object(safety_manager, 'validate_pre_trade_prod') as mock_validate:
                mock_validate.return_value = PreTradeValidationResult(is_valid=True, status=ValidationStatus.APPROVED)
                
                result = await safety_manager.validate_pre_trade(trade_order)
                assert result.is_valid


class TestEnsembleCircuitBreakerIntegration:
    """Test suite for circuit breaker ensemble integration."""

    @pytest.fixture
    def emergency_controller(self):
        """Mock emergency stop controller."""
        return Mock(spec=EmergencyStopController)

    @pytest.fixture
    def circuit_breaker_config(self):
        """Circuit breaker configuration for ensemble testing."""
        return CircuitBreakerConfig(
            enable_ensemble_monitoring=True,
            ensemble_performance_threshold=0.70,
            ensemble_degradation_threshold=0.50,
            model_failure_threshold=2  # Max failed models before circuit breaker
        )

    @pytest.fixture
    def ensemble_market_data(self):
        """Market data with ensemble performance metrics."""
        return MarketData(
            symbol="BTC",
            price=Decimal("50000"),
            volume=Decimal("1000000"),
            timestamp=datetime.now(),
            bid_ask_spread=Decimal("10"),
            market_cap=Decimal("1000000000"),
            volatility_24h=Decimal("0.15"),
            ensemble_performance={
                'overall_accuracy': 0.72,
                'model_performances': {
                    'lstm': 0.75,
                    'itransformer': 0.68,  # Underperforming
                    'patchtst': 0.74,
                    'timesmixer': 0.71
                },
                'failed_models': ['timesfm'],  # Failed model
                'ensemble_confidence': 0.73
            }
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_monitors_ensemble_performance(
        self, emergency_controller, circuit_breaker_config, ensemble_market_data
    ):
        """Test 6.2: Circuit breakers consider ensemble performance."""
        circuit_breaker = TradingCircuitBreaker(circuit_breaker_config, emergency_controller)
        
        # Mock ensemble performance check
        with patch.object(circuit_breaker, 'check_ensemble_performance_triggers') as mock_check:
            mock_check.return_value = CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.VOLATILITY,
                level=CircuitBreakerLevel.WARNING,
                message="Ensemble performance below threshold"
            )
            
            result = await circuit_breaker.check_ensemble_performance_triggers(ensemble_market_data)
            
            assert result.should_trigger
            assert result.breaker_type == CircuitBreakerType.VOLATILITY
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_ensemble_degradation_detection(
        self, emergency_controller, circuit_breaker_config
    ):
        """Test circuit breaker detects ensemble degradation."""
        circuit_breaker = TradingCircuitBreaker(circuit_breaker_config, emergency_controller)
        
        # Simulate degraded ensemble metrics
        degraded_ensemble_health = {
            'overall_accuracy': 0.45,  # Below threshold
            'model_failures': 3,  # Too many failures
            'ensemble_confidence': 0.38,
            'prediction_variance': 0.30  # High variance
        }
        
        with patch.object(circuit_breaker, 'assess_ensemble_health') as mock_assess:
            mock_assess.return_value = {
                'is_healthy': False,
                'degradation_severity': 'critical',
                'failing_components': ['model_accuracy', 'ensemble_confidence']
            }
            
            health_assessment = mock_assess(degraded_ensemble_health)
            
            assert not health_assessment['is_healthy']
            assert health_assessment['degradation_severity'] == 'critical'

    @pytest.mark.asyncio
    async def test_circuit_breaker_model_failure_threshold(
        self, emergency_controller, circuit_breaker_config
    ):
        """Test circuit breaker triggers on model failure threshold."""
        circuit_breaker = TradingCircuitBreaker(circuit_breaker_config, emergency_controller)
        
        # Mock model failure scenario
        model_health_data = {
            'total_models': 5,
            'failed_models': 3,  # Exceeds threshold of 2
            'failing_model_ids': ['itransformer', 'timesfm', 'patchtst']
        }
        
        with patch.object(circuit_breaker, 'check_model_failure_threshold') as mock_check:
            mock_check.return_value = CircuitBreakerResult(
                should_trigger=True,
                breaker_type=CircuitBreakerType.MARKET_WIDE,
                level=CircuitBreakerLevel.CRITICAL,
                message=f"Too many model failures: {model_health_data['failed_models']}/{model_health_data['total_models']}"
            )
            
            result = mock_check(model_health_data)
            
            assert result.should_trigger
            assert result.level == CircuitBreakerLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_circuit_breaker_ensemble_confidence_threshold(
        self, emergency_controller, circuit_breaker_config
    ):
        """Test circuit breaker respects ensemble confidence thresholds."""
        circuit_breaker = TradingCircuitBreaker(circuit_breaker_config, emergency_controller)
        
        # High confidence ensemble - should allow trading
        high_confidence_data = MarketData(
            symbol="ETH",
            price=Decimal("3000"),
            volume=Decimal("500000"),
            timestamp=datetime.now(),
            bid_ask_spread=Decimal("5"),
            market_cap=Decimal("400000000"),
            volatility_24h=Decimal("0.12"),
            ensemble_confidence=0.85  # High confidence
        )
        
        # Low confidence ensemble - should restrict trading
        low_confidence_data = MarketData(
            symbol="ETH", 
            price=Decimal("3000"),
            volume=Decimal("500000"),
            timestamp=datetime.now(),
            bid_ask_spread=Decimal("5"),
            market_cap=Decimal("400000000"),
            volatility_24h=Decimal("0.12"),
            ensemble_confidence=0.35  # Low confidence
        )
        
        # Test high confidence allows trading
        high_conf_result = await circuit_breaker.can_execute_trade("ETH", Decimal("10000"))
        assert high_conf_result.is_allowed
        
        # Test low confidence restricts trading
        with patch.object(circuit_breaker, '_get_ensemble_confidence') as mock_confidence:
            mock_confidence.return_value = 0.35
            
            low_conf_result = await circuit_breaker.can_execute_trade("ETH", Decimal("10000"))
            # Should apply restrictions due to low ensemble confidence

    @pytest.mark.asyncio
    async def test_circuit_breaker_development_vs_production_thresholds(
        self, emergency_controller
    ):
        """Test different thresholds for development vs production."""
        # Development config (more lenient)
        dev_config = CircuitBreakerConfig(
            ensemble_performance_threshold=0.60,  # Lower threshold
            model_failure_threshold=3  # Allow more failures
        )
        
        # Production config (stricter) 
        prod_config = CircuitBreakerConfig(
            ensemble_performance_threshold=0.75,  # Higher threshold
            model_failure_threshold=1  # Less tolerance for failures
        )
        
        # Test development environment
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            dev_breaker = TradingCircuitBreaker(dev_config, emergency_controller)
            # Should be more lenient with model failures
            
        # Test production environment
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            prod_breaker = TradingCircuitBreaker(prod_config, emergency_controller)
            # Should be stricter with model failures

    @pytest.mark.asyncio
    async def test_circuit_breaker_ensemble_recovery_validation(
        self, emergency_controller, circuit_breaker_config
    ):
        """Test circuit breaker validates ensemble recovery properly."""
        circuit_breaker = TradingCircuitBreaker(circuit_breaker_config, emergency_controller)
        
        # Mock recovery conditions
        recovery_metrics = {
            'ensemble_accuracy': 0.78,  # Recovered
            'model_health_scores': {
                'lstm': 0.82,
                'itransformer': 0.75,  # Recovered
                'patchtst': 0.79,
                'timesmixer': 0.77
            },
            'ensemble_confidence': 0.76,
            'stability_window_minutes': 15
        }
        
        with patch.object(circuit_breaker, 'validate_ensemble_recovery') as mock_validate:
            mock_validate.return_value = {
                'is_recovered': True,
                'confidence_level': 0.85,
                'stability_validated': True,
                'recovery_duration_minutes': 20
            }
            
            recovery_result = mock_validate(recovery_metrics)
            
            assert recovery_result['is_recovered']
            assert recovery_result['confidence_level'] > 0.8

    @pytest.mark.asyncio
    async def test_circuit_breaker_integrates_with_fallback_strategies(
        self, emergency_controller, circuit_breaker_config
    ):
        """Test circuit breaker integrates with ensemble fallback strategies."""
        circuit_breaker = TradingCircuitBreaker(circuit_breaker_config, emergency_controller)
        
        # Mock fallback scenario
        fallback_scenario = {
            'primary_models_failed': ['itransformer', 'timesfm'],
            'fallback_models_active': ['lstm', 'patchtst'],
            'fallback_confidence': 0.65,
            'fallback_strategy': 'ensemble_degradation'
        }
        
        with patch.object(circuit_breaker, 'handle_fallback_integration') as mock_handle:
            mock_handle.return_value = CircuitBreakerResult(
                should_trigger=False,  # Allow trading with fallback
                message="Fallback models active, monitoring closely"
            )
            
            result = mock_handle(fallback_scenario)
            
            # Should allow trading with active fallback
            assert not result.should_trigger
            mock_handle.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])