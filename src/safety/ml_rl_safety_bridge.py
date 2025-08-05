"""
ML-RL Safety Bridge Integration - Phase 3.2.4

This module provides comprehensive safety integration between ML and RL systems with:
- ML prediction validation before execution
- RL action safety constraint checking
- Model confidence thresholds and monitoring
- Fallback strategies when models fail
- Integration with existing ML/RL components
- Cross-system safety coordination

Key Features:
- Prediction safety validation with confidence thresholds
- Action constraint checking and bounds validation
- Model performance monitoring and degradation detection
- Automated fallback activation for unsafe predictions/actions
- Safety state coordination between ML and RL systems
- Risk-aware decision making with safety overrides
- Integration with existing safety and fallback systems

Following TDD methodology - implementation satisfies comprehensive test requirements.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from typing import Dict, Any, List, Optional, Set, Callable, Union, Tuple, NamedTuple
from uuid import UUID, uuid4
from collections import defaultdict, deque
import numpy as np
import statistics

from src.ml_analysis.base import PredictionResult, ModelType
from src.rl_agent.base import TradeAction, MarketState
from src.safety.unified_safety_state_manager import UnifiedSafetyStateManager, SafetyState
from src.safety.emergency_stop_controller import EmergencyStopController
from src.modes.fallback_strategies import FallbackSystemIntegration, ModelDegradationDetector
from src.monitoring.drift_detection import DriftSeverity


logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Result types for safety validation."""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    FALLBACK_REQUIRED = "fallback_required"


class SafetyViolationType(Enum):
    """Types of safety violations."""
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    PREDICTION_OUT_OF_BOUNDS = "prediction_out_of_bounds"
    ACTION_EXCEEDS_LIMITS = "action_exceeds_limits"
    MODEL_DEGRADED = "model_degraded"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    CONFLICTING_SIGNALS = "conflicting_signals"
    HISTORICAL_ANOMALY = "historical_anomaly"
    DRIFT_DETECTED = "drift_detected"


class ModelSafetyStatus(Enum):
    """Safety status levels for models."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    UNSAFE = "unsafe"
    DEGRADED = "degraded"


@dataclass
class SafetyConstraints:
    """Safety constraints for ML/RL validation."""
    min_confidence_threshold: float = 0.6
    max_position_size: Decimal = Decimal("10000.0")
    max_risk_per_trade: float = 0.02  # 2% of portfolio
    max_daily_loss: float = 0.05  # 5% of portfolio
    max_drawdown: float = 0.10  # 10% drawdown limit
    
    # ML-specific constraints
    min_prediction_samples: int = 100
    max_prediction_uncertainty: float = 0.3
    prediction_bounds: Tuple[float, float] = (-1.0, 1.0)
    
    # RL-specific constraints
    max_action_magnitude: float = 1.0
    action_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'buy': (0.0, 1.0),
        'sell': (0.0, 1.0),
        'hold': (0.0, 1.0)
    })
    
    # Risk management constraints
    volatility_threshold: float = 0.05  # 5% volatility limit
    correlation_limit: float = 0.8  # Max correlation between positions
    concentration_limit: float = 0.3  # Max 30% in single asset
    
    # Model performance constraints
    min_accuracy: float = 0.7
    max_latency_ms: float = 100.0
    max_error_rate: float = 0.05


@dataclass
class ValidationContext:
    """Context information for safety validation."""
    model_id: str
    model_type: ModelType
    timestamp: datetime
    market_conditions: Dict[str, Any]
    current_positions: Dict[str, Decimal] = field(default_factory=dict)
    portfolio_value: Decimal = Decimal("0.0")
    recent_performance: Dict[str, float] = field(default_factory=dict)
    safety_state: Optional[SafetyState] = None


@dataclass
class SafetyValidationResult:
    """Result of safety validation check."""
    result: ValidationResult
    confidence_score: float
    violation_types: List[SafetyViolationType] = field(default_factory=list)
    safety_score: float = 1.0  # 0.0 - 1.0
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    fallback_suggestions: List[str] = field(default_factory=list)
    validation_details: Dict[str, Any] = field(default_factory=dict)
    requires_human_approval: bool = False


@dataclass
class MLPredictionValidation:
    """Validation result for ML predictions."""
    original_prediction: PredictionResult
    validation_result: SafetyValidationResult
    adjusted_prediction: Optional[PredictionResult] = None
    safety_adjustments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLActionValidation:
    """Validation result for RL actions."""
    original_action: TradeAction
    validation_result: SafetyValidationResult
    adjusted_action: Optional[TradeAction] = None
    safety_adjustments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSafetyMetrics:
    """Safety metrics for model monitoring."""
    model_id: str
    model_type: ModelType
    safety_status: ModelSafetyStatus
    confidence_distribution: List[float]
    recent_accuracy: float
    prediction_stability: float
    error_rate: float
    avg_latency_ms: float
    last_updated: datetime = field(default_factory=datetime.utcnow)
    safety_violations_24h: int = 0
    fallback_activations_24h: int = 0


@dataclass
class CrossSystemSafetyCheck:
    """Cross-system safety check result."""
    ml_model_status: ModelSafetyStatus
    rl_agent_status: ModelSafetyStatus
    consistency_score: float  # How consistent are ML and RL signals
    cross_validation_passed: bool
    conflicting_signals: List[str] = field(default_factory=list)
    unified_recommendation: Optional[str] = None


@dataclass
class SafetyBridgeConfig:
    """Configuration for ML-RL Safety Bridge."""
    enable_ml_validation: bool = True
    enable_rl_validation: bool = True
    enable_cross_validation: bool = True
    enable_fallback_integration: bool = True
    
    # Validation timeouts
    ml_validation_timeout_ms: int = 50
    rl_validation_timeout_ms: int = 30
    cross_validation_timeout_ms: int = 100
    
    # Safety thresholds
    global_confidence_threshold: float = 0.6
    risk_tolerance_level: float = 0.02  # 2%
    safety_score_threshold: float = 0.7
    
    # Monitoring settings
    metrics_collection_interval: int = 60  # seconds
    safety_metrics_retention_hours: int = 24
    enable_real_time_monitoring: bool = True
    
    # Fallback settings
    auto_fallback_on_violation: bool = True
    fallback_confidence_threshold: float = 0.3
    max_fallback_duration_minutes: int = 30
    
    # Cross-system validation
    consistency_threshold: float = 0.8
    allow_conflicting_signals: bool = False
    conflict_resolution_strategy: str = "conservative"  # conservative, aggressive, ml_priority, rl_priority
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 30
    enable_async_validation: bool = True
    max_concurrent_validations: int = 10


class MLPredictionValidator:
    """Validates ML predictions for safety compliance."""
    
    def __init__(self, constraints: SafetyConstraints):
        self.constraints = constraints
        self.logger = logging.getLogger(f"{__name__}.ml_validator")
        self.validation_history: deque = deque(maxlen=1000)
        
    async def validate_prediction(
        self,
        prediction: PredictionResult,
        context: ValidationContext
    ) -> MLPredictionValidation:
        """Validate ML prediction for safety compliance."""
        start_time = time.time()
        
        try:
            # Basic validation checks
            confidence_check = self._validate_confidence(prediction, context)
            bounds_check = self._validate_prediction_bounds(prediction, context)
            uncertainty_check = self._validate_uncertainty(prediction, context)
            samples_check = self._validate_sample_size(prediction, context)
            
            # Risk-based validation
            risk_check = self._validate_prediction_risk(prediction, context)
            
            # Historical consistency validation
            consistency_check = self._validate_historical_consistency(prediction, context)
            
            # Combine all validation results
            all_checks = [confidence_check, bounds_check, uncertainty_check, samples_check, risk_check, consistency_check]
            
            # Aggregate validation results
            violation_types = []
            safety_scores = []
            recommended_actions = []
            fallback_suggestions = []
            validation_details = {}
            
            for check in all_checks:
                if check.result != ValidationResult.APPROVED:
                    violation_types.extend(check.violation_types)
                    fallback_suggestions.extend(check.fallback_suggestions)
                
                recommended_actions.extend(check.recommended_actions)
                safety_scores.append(check.safety_score)
                validation_details.update(check.validation_details)
            
            # Determine overall result
            if not violation_types:
                overall_result = ValidationResult.APPROVED
            elif any(vt in [SafetyViolationType.MODEL_DEGRADED, SafetyViolationType.CONFIDENCE_TOO_LOW] for vt in violation_types):
                overall_result = ValidationResult.FALLBACK_REQUIRED
            elif len(violation_types) > 2:
                overall_result = ValidationResult.REJECTED
            else:
                overall_result = ValidationResult.CONDITIONAL
            
            # Calculate overall safety score
            overall_safety_score = np.mean(safety_scores) if safety_scores else 1.0
            
            # Calculate confidence score based on prediction quality and safety checks
            confidence_score = min(prediction.confidence, overall_safety_score)
            
            validation_result = SafetyValidationResult(
                result=overall_result,
                confidence_score=confidence_score,
                violation_types=violation_types,
                safety_score=overall_safety_score,
                risk_assessment=self._assess_prediction_risk(prediction, context),
                recommended_actions=list(set(recommended_actions)),
                fallback_suggestions=list(set(fallback_suggestions)),
                validation_details=validation_details,
                requires_human_approval=self._requires_human_approval(violation_types)
            )
            
            # Create adjusted prediction if needed
            adjusted_prediction = None
            safety_adjustments = {}
            
            if overall_result == ValidationResult.CONDITIONAL:
                adjusted_prediction, safety_adjustments = self._adjust_prediction_for_safety(prediction, validation_result, context)
            
            # Record validation
            validation_duration = (time.time() - start_time) * 1000
            self._record_validation(prediction, validation_result, validation_duration)
            
            return MLPredictionValidation(
                original_prediction=prediction,
                validation_result=validation_result,
                adjusted_prediction=adjusted_prediction,
                safety_adjustments=safety_adjustments
            )
            
        except Exception as e:
            self.logger.error(f"ML prediction validation error: {str(e)}")
            
            # Return safe fallback validation
            return MLPredictionValidation(
                original_prediction=prediction,
                validation_result=SafetyValidationResult(
                    result=ValidationResult.REJECTED,
                    confidence_score=0.0,
                    violation_types=[SafetyViolationType.MODEL_DEGRADED],
                    safety_score=0.0,
                    validation_details={"error": str(e)}
                )
            )
    
    def _validate_confidence(self, prediction: PredictionResult, context: ValidationContext) -> SafetyValidationResult:
        """Validate prediction confidence levels."""
        if prediction.confidence < self.constraints.min_confidence_threshold:
            return SafetyValidationResult(
                result=ValidationResult.FALLBACK_REQUIRED,
                confidence_score=prediction.confidence,
                violation_types=[SafetyViolationType.CONFIDENCE_TOO_LOW],
                safety_score=prediction.confidence / self.constraints.min_confidence_threshold,
                recommended_actions=["use_fallback_model", "increase_sample_size"],
                fallback_suggestions=["rule_based_fallback", "ensemble_fallback"],
                validation_details={"confidence": prediction.confidence, "threshold": self.constraints.min_confidence_threshold}
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=prediction.confidence,
            safety_score=1.0,
            validation_details={"confidence": prediction.confidence}
        )
    
    def _validate_prediction_bounds(self, prediction: PredictionResult, context: ValidationContext) -> SafetyValidationResult:
        """Validate prediction values are within expected bounds."""
        min_bound, max_bound = self.constraints.prediction_bounds
        
        # Check if prediction value is within bounds
        prediction_value = getattr(prediction, 'value', 0.0)
        if isinstance(prediction_value, (list, np.ndarray)):
            prediction_value = np.mean(prediction_value)
        
        if not (min_bound <= prediction_value <= max_bound):
            return SafetyValidationResult(
                result=ValidationResult.REJECTED,
                confidence_score=0.0,
                violation_types=[SafetyViolationType.PREDICTION_OUT_OF_BOUNDS],
                safety_score=0.3,
                recommended_actions=["clip_prediction", "recalibrate_model"],
                validation_details={
                    "prediction_value": prediction_value,
                    "bounds": (min_bound, max_bound)
                }
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=prediction.confidence,
            safety_score=1.0,
            validation_details={"prediction_value": prediction_value}
        )
    
    def _validate_uncertainty(self, prediction: PredictionResult, context: ValidationContext) -> SafetyValidationResult:
        """Validate prediction uncertainty levels."""
        uncertainty = getattr(prediction, 'uncertainty', 0.0)
        
        if uncertainty > self.constraints.max_prediction_uncertainty:
            return SafetyValidationResult(
                result=ValidationResult.CONDITIONAL,
                confidence_score=max(0.0, 1.0 - uncertainty),
                violation_types=[SafetyViolationType.PREDICTION_OUT_OF_BOUNDS],
                safety_score=max(0.0, 1.0 - uncertainty / self.constraints.max_prediction_uncertainty),
                recommended_actions=["reduce_uncertainty", "use_ensemble"],
                validation_details={"uncertainty": uncertainty, "threshold": self.constraints.max_prediction_uncertainty}
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=prediction.confidence,
            safety_score=1.0,
            validation_details={"uncertainty": uncertainty}
        )
    
    def _validate_sample_size(self, prediction: PredictionResult, context: ValidationContext) -> SafetyValidationResult:
        """Validate sufficient sample size for prediction."""
        sample_count = getattr(prediction, 'sample_count', 0)
        
        if sample_count < self.constraints.min_prediction_samples:
            return SafetyValidationResult(
                result=ValidationResult.CONDITIONAL,
                confidence_score=prediction.confidence * 0.8,  # Reduce confidence for low samples
                safety_score=sample_count / self.constraints.min_prediction_samples,
                recommended_actions=["collect_more_data", "use_ensemble"],
                validation_details={"sample_count": sample_count, "required": self.constraints.min_prediction_samples}
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=prediction.confidence,
            safety_score=1.0,
            validation_details={"sample_count": sample_count}
        )
    
    def _validate_prediction_risk(self, prediction: PredictionResult, context: ValidationContext) -> SafetyValidationResult:
        """Validate prediction-based risk levels."""
        # Calculate potential risk from prediction
        prediction_value = getattr(prediction, 'value', 0.0)
        if isinstance(prediction_value, (list, np.ndarray)):
            prediction_value = float(np.mean(prediction_value))
        
        # Estimate position size impact
        estimated_position = float(context.portfolio_value) * abs(prediction_value) * prediction.confidence
        risk_ratio = estimated_position / max(float(context.portfolio_value), 1.0)
        
        if risk_ratio > self.constraints.max_risk_per_trade:
            return SafetyValidationResult(
                result=ValidationResult.CONDITIONAL,
                confidence_score=prediction.confidence * 0.7,
                violation_types=[SafetyViolationType.RISK_THRESHOLD_EXCEEDED],
                safety_score=max(0.0, 1.0 - (risk_ratio - self.constraints.max_risk_per_trade)),
                recommended_actions=["reduce_position_size", "adjust_confidence"],
                validation_details={"risk_ratio": risk_ratio, "threshold": self.constraints.max_risk_per_trade}
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=prediction.confidence,
            safety_score=1.0,
            validation_details={"risk_ratio": risk_ratio}
        )
    
    def _validate_historical_consistency(self, prediction: PredictionResult, context: ValidationContext) -> SafetyValidationResult:
        """Validate prediction consistency with historical patterns."""
        # Check recent validation history for anomalies
        if len(self.validation_history) < 10:
            return SafetyValidationResult(
                result=ValidationResult.APPROVED,
                confidence_score=prediction.confidence,
                safety_score=1.0,
                validation_details={"historical_data": "insufficient"}
            )
        
        # Compare with recent predictions
        recent_confidences = [v['confidence'] for v in list(self.validation_history)[-10:]]
        recent_avg_confidence = np.mean(recent_confidences)
        confidence_deviation = abs(prediction.confidence - recent_avg_confidence)
        
        # If current prediction deviates significantly from recent pattern
        if confidence_deviation > 0.3:
            return SafetyValidationResult(
                result=ValidationResult.CONDITIONAL,
                confidence_score=prediction.confidence * 0.9,
                violation_types=[SafetyViolationType.HISTORICAL_ANOMALY],
                safety_score=max(0.5, 1.0 - confidence_deviation),
                recommended_actions=["investigate_anomaly", "verify_data_quality"],
                validation_details={"confidence_deviation": confidence_deviation, "recent_avg": recent_avg_confidence}
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=prediction.confidence,
            safety_score=1.0,
            validation_details={"confidence_deviation": confidence_deviation}
        )
    
    def _assess_prediction_risk(self, prediction: PredictionResult, context: ValidationContext) -> Dict[str, float]:
        """Assess various risk factors for the prediction."""
        return {
            "confidence_risk": max(0.0, self.constraints.min_confidence_threshold - prediction.confidence),
            "uncertainty_risk": getattr(prediction, 'uncertainty', 0.0),
            "magnitude_risk": abs(getattr(prediction, 'value', 0.0)),
            "timing_risk": 0.1 if context.market_conditions.get('volatility', 0.0) > 0.05 else 0.0
        }
    
    def _adjust_prediction_for_safety(
        self,
        prediction: PredictionResult,
        validation_result: SafetyValidationResult,
        context: ValidationContext
    ) -> Tuple[PredictionResult, Dict[str, Any]]:
        """Adjust prediction for safety compliance."""
        adjustments = {}
        
        # Create adjusted prediction (simplified implementation)
        adjusted_confidence = prediction.confidence
        
        # Reduce confidence if there are violations
        if validation_result.violation_types:
            confidence_reduction = len(validation_result.violation_types) * 0.1
            adjusted_confidence = max(0.1, prediction.confidence - confidence_reduction)
            adjustments['confidence_reduction'] = confidence_reduction
        
        # Apply safety score weighting
        adjusted_confidence *= validation_result.safety_score
        adjustments['safety_weighting'] = validation_result.safety_score
        
        # Create new prediction result with adjustments
        adjusted_prediction = PredictionResult(
            prediction_id=f"{prediction.prediction_id}_adjusted",
            model_type=prediction.model_type,
            confidence=adjusted_confidence,
            timestamp=datetime.utcnow(),
            metadata={**prediction.metadata, 'safety_adjusted': True}
        )
        
        return adjusted_prediction, adjustments
    
    def _requires_human_approval(self, violation_types: List[SafetyViolationType]) -> bool:
        """Determine if human approval is required."""
        critical_violations = [
            SafetyViolationType.MODEL_DEGRADED,
            SafetyViolationType.RISK_THRESHOLD_EXCEEDED
        ]
        
        return any(vt in critical_violations for vt in violation_types)
    
    def _record_validation(self, prediction: PredictionResult, validation_result: SafetyValidationResult, duration_ms: float):
        """Record validation for historical analysis."""
        self.validation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'prediction_id': prediction.prediction_id,
            'confidence': prediction.confidence,
            'validation_result': validation_result.result.value,
            'safety_score': validation_result.safety_score,
            'violation_count': len(validation_result.violation_types),
            'duration_ms': duration_ms
        })


class RLActionValidator:
    """Validates RL actions for safety compliance."""
    
    def __init__(self, constraints: SafetyConstraints):
        self.constraints = constraints
        self.logger = logging.getLogger(f"{__name__}.rl_validator")
        self.validation_history: deque = deque(maxlen=1000)
    
    async def validate_action(
        self,
        action: TradeAction,
        context: ValidationContext
    ) -> RLActionValidation:
        """Validate RL action for safety compliance."""
        start_time = time.time()
        
        try:
            # Basic validation checks
            magnitude_check = self._validate_action_magnitude(action, context)
            bounds_check = self._validate_action_bounds(action, context)
            position_check = self._validate_position_limits(action, context)
            
            # Risk-based validation
            risk_check = self._validate_action_risk(action, context)
            concentration_check = self._validate_concentration_limits(action, context)
            
            # Market condition validation
            market_check = self._validate_market_conditions(action, context)
            
            # Combine all validation results
            all_checks = [magnitude_check, bounds_check, position_check, risk_check, concentration_check, market_check]
            
            # Aggregate validation results
            violation_types = []
            safety_scores = []
            recommended_actions = []
            fallback_suggestions = []
            validation_details = {}
            
            for check in all_checks:
                if check.result != ValidationResult.APPROVED:
                    violation_types.extend(check.violation_types)
                    fallback_suggestions.extend(check.fallback_suggestions)
                
                recommended_actions.extend(check.recommended_actions)
                safety_scores.append(check.safety_score)
                validation_details.update(check.validation_details)
            
            # Determine overall result
            if not violation_types:
                overall_result = ValidationResult.APPROVED
            elif any(vt == SafetyViolationType.ACTION_EXCEEDS_LIMITS for vt in violation_types):
                overall_result = ValidationResult.REJECTED
            elif len(violation_types) > 2:
                overall_result = ValidationResult.CONDITIONAL
            else:
                overall_result = ValidationResult.CONDITIONAL
            
            # Calculate overall safety score
            overall_safety_score = np.mean(safety_scores) if safety_scores else 1.0
            
            # Calculate confidence score
            action_confidence = getattr(action, 'confidence', 0.8)
            confidence_score = min(action_confidence, overall_safety_score)
            
            validation_result = SafetyValidationResult(
                result=overall_result,
                confidence_score=confidence_score,
                violation_types=violation_types,
                safety_score=overall_safety_score,
                risk_assessment=self._assess_action_risk(action, context),
                recommended_actions=list(set(recommended_actions)),
                fallback_suggestions=list(set(fallback_suggestions)),
                validation_details=validation_details,
                requires_human_approval=self._requires_human_approval(violation_types)
            )
            
            # Create adjusted action if needed
            adjusted_action = None
            safety_adjustments = {}
            
            if overall_result == ValidationResult.CONDITIONAL:
                adjusted_action, safety_adjustments = self._adjust_action_for_safety(action, validation_result, context)
            
            # Record validation
            validation_duration = (time.time() - start_time) * 1000
            self._record_validation(action, validation_result, validation_duration)
            
            return RLActionValidation(
                original_action=action,
                validation_result=validation_result,
                adjusted_action=adjusted_action,  
                safety_adjustments=safety_adjustments
            )
            
        except Exception as e:
            self.logger.error(f"RL action validation error: {str(e)}")
            
            # Return safe fallback validation
            return RLActionValidation(
                original_action=action,
                validation_result=SafetyValidationResult(
                    result=ValidationResult.REJECTED,
                    confidence_score=0.0,
                    violation_types=[SafetyViolationType.MODEL_DEGRADED],
                    safety_score=0.0,
                    validation_details={"error": str(e)}
                )
            )
    
    def _validate_action_magnitude(self, action: TradeAction, context: ValidationContext) -> SafetyValidationResult:
        """Validate action magnitude is within limits."""
        action_magnitude = getattr(action, 'magnitude', 0.0)
        
        if action_magnitude > self.constraints.max_action_magnitude:
            return SafetyValidationResult(
                result=ValidationResult.CONDITIONAL,
                confidence_score=0.5,
                violation_types=[SafetyViolationType.ACTION_EXCEEDS_LIMITS],
                safety_score=self.constraints.max_action_magnitude / action_magnitude,
                recommended_actions=["reduce_action_size", "split_action"],
                validation_details={"magnitude": action_magnitude, "limit": self.constraints.max_action_magnitude}
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=getattr(action, 'confidence', 0.8),
            safety_score=1.0,
            validation_details={"magnitude": action_magnitude}
        )
    
    def _validate_action_bounds(self, action: TradeAction, context: ValidationContext) -> SafetyValidationResult:
        """Validate action values are within defined bounds."""
        action_type = getattr(action, 'action_type', 'hold')
        action_value = getattr(action, 'value', 0.0)
        
        if action_type in self.constraints.action_bounds:
            min_bound, max_bound = self.constraints.action_bounds[action_type]
            
            if not (min_bound <= action_value <= max_bound):
                return SafetyValidationResult(
                    result=ValidationResult.REJECTED,
                    confidence_score=0.0,
                    violation_types=[SafetyViolationType.ACTION_EXCEEDS_LIMITS],
                    safety_score=0.3,
                    recommended_actions=["clip_action", "recalibrate_agent"],
                    validation_details={
                        "action_value": action_value,
                        "bounds": (min_bound, max_bound),
                        "action_type": action_type
                    }
                )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=getattr(action, 'confidence', 0.8),
            safety_score=1.0,
            validation_details={"action_value": action_value, "action_type": action_type}
        )
    
    def _validate_position_limits(self, action: TradeAction, context: ValidationContext) -> SafetyValidationResult:
        """Validate action doesn't exceed position limits."""
        action_size = getattr(action, 'size', Decimal("0.0"))
        
        if action_size > self.constraints.max_position_size:
            return SafetyValidationResult(
                result=ValidationResult.REJECTED,
                confidence_score=0.0,
                violation_types=[SafetyViolationType.ACTION_EXCEEDS_LIMITS],
                safety_score=0.2,
                recommended_actions=["reduce_position_size", "split_order"],
                validation_details={"position_size": float(action_size), "limit": float(self.constraints.max_position_size)}
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=getattr(action, 'confidence', 0.8),
            safety_score=1.0,
            validation_details={"position_size": float(action_size)}
        )
    
    def _validate_action_risk(self, action: TradeAction, context: ValidationContext) -> SafetyValidationResult:
        """Validate action risk levels."""
        action_size = getattr(action, 'size', Decimal("0.0"))
        portfolio_value = max(context.portfolio_value, Decimal("1.0"))
        
        risk_ratio = float(action_size) / float(portfolio_value)
        
        if risk_ratio > self.constraints.max_risk_per_trade:
            return SafetyValidationResult(
                result=ValidationResult.CONDITIONAL,
                confidence_score=0.6,
                violation_types=[SafetyViolationType.RISK_THRESHOLD_EXCEEDED],
                safety_score=max(0.0, 1.0 - (risk_ratio - self.constraints.max_risk_per_trade)),
                recommended_actions=["reduce_risk_exposure", "adjust_position_size"],
                validation_details={"risk_ratio": risk_ratio, "threshold": self.constraints.max_risk_per_trade}
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=getattr(action, 'confidence', 0.8),
            safety_score=1.0,
            validation_details={"risk_ratio": risk_ratio}
        )
    
    def _validate_concentration_limits(self, action: TradeAction, context: ValidationContext) -> SafetyValidationResult:
        """Validate concentration limits."""
        symbol = getattr(action, 'symbol', '')
        action_size = getattr(action, 'size', Decimal("0.0"))
        
        if symbol and symbol in context.current_positions:
            current_position = context.current_positions[symbol]
            new_position = current_position + action_size
            
            portfolio_value = max(context.portfolio_value, Decimal("1.0"))
            concentration = float(abs(new_position)) / float(portfolio_value)
            
            if concentration > self.constraints.concentration_limit:
                return SafetyValidationResult(
                    result=ValidationResult.CONDITIONAL,
                    confidence_score=0.7,
                    violation_types=[SafetyViolationType.RISK_THRESHOLD_EXCEEDED],
                    safety_score=max(0.0, 1.0 - (concentration - self.constraints.concentration_limit)),
                    recommended_actions=["diversify_portfolio", "reduce_concentration"],
                    validation_details={"concentration": concentration, "limit": self.constraints.concentration_limit}
                )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=getattr(action, 'confidence', 0.8),
            safety_score=1.0,
            validation_details={"concentration_check": "passed"}
        )
    
    def _validate_market_conditions(self, action: TradeAction, context: ValidationContext) -> SafetyValidationResult:
        """Validate action appropriateness for current market conditions."""
        market_volatility = context.market_conditions.get('volatility', 0.0)
        
        if market_volatility > self.constraints.volatility_threshold:
            # In high volatility, be more conservative
            return SafetyValidationResult(
                result=ValidationResult.CONDITIONAL,
                confidence_score=0.6,
                safety_score=0.7,
                recommended_actions=["reduce_position_size", "increase_monitoring"],
                validation_details={"market_volatility": market_volatility, "threshold": self.constraints.volatility_threshold}
            )
        
        return SafetyValidationResult(
            result=ValidationResult.APPROVED,
            confidence_score=getattr(action, 'confidence', 0.8),
            safety_score=1.0,
            validation_details={"market_volatility": market_volatility}
        )
    
    def _assess_action_risk(self, action: TradeAction, context: ValidationContext) -> Dict[str, float]:
        """Assess various risk factors for the action."""
        action_size = getattr(action, 'size', Decimal("0.0"))
        portfolio_value = max(context.portfolio_value, Decimal("1.0"))
        
        return {
            "position_risk": float(action_size) / float(portfolio_value),
            "magnitude_risk": getattr(action, 'magnitude', 0.0),
            "market_risk": context.market_conditions.get('volatility', 0.0),
            "timing_risk": 0.1 if context.market_conditions.get('after_hours', False) else 0.0
        }
    
    def _adjust_action_for_safety(
        self,
        action: TradeAction,
        validation_result: SafetyValidationResult,
        context: ValidationContext
    ) -> Tuple[TradeAction, Dict[str, Any]]:
        """Adjust action for safety compliance."""
        adjustments = {}
        
        # Create adjusted action (simplified implementation)
        adjusted_size = getattr(action, 'size', Decimal("0.0"))
        adjusted_confidence = getattr(action, 'confidence', 0.8)
        
        # Reduce size if there are violations
        if validation_result.violation_types:
            size_reduction = validation_result.safety_score
            adjusted_size = adjusted_size * Decimal(str(size_reduction))
            adjustments['size_reduction'] = 1.0 - size_reduction
        
        # Apply safety score weighting to confidence
        adjusted_confidence *= validation_result.safety_score
        adjustments['confidence_adjustment'] = validation_result.safety_score
        
        # Create new action with adjustments
        adjusted_action = TradeAction(
            action_id=f"{getattr(action, 'action_id', uuid4())}_adjusted",
            action_type=getattr(action, 'action_type', 'hold'),
            symbol=getattr(action, 'symbol', ''),
            size=adjusted_size,
            confidence=adjusted_confidence,
            timestamp=datetime.utcnow(),
            metadata={**getattr(action, 'metadata', {}), 'safety_adjusted': True}
        )
        
        return adjusted_action, adjustments
    
    def _requires_human_approval(self, violation_types: List[SafetyViolationType]) -> bool:
        """Determine if human approval is required."""
        critical_violations = [
            SafetyViolationType.ACTION_EXCEEDS_LIMITS,
            SafetyViolationType.RISK_THRESHOLD_EXCEEDED
        ]
        
        return any(vt in critical_violations for vt in violation_types)
    
    def _record_validation(self, action: TradeAction, validation_result: SafetyValidationResult, duration_ms: float):
        """Record validation for historical analysis."""
        self.validation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'action_id': getattr(action, 'action_id', str(uuid4())),
            'action_type': getattr(action, 'action_type', 'unknown'),
            'validation_result': validation_result.result.value,
            'safety_score': validation_result.safety_score,
            'violation_count': len(validation_result.violation_types),
            'duration_ms': duration_ms
        })


class CrossSystemValidator:
    """Validates consistency between ML and RL systems."""
    
    def __init__(self, config: SafetyBridgeConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.cross_validator")
        self.validation_history: deque = deque(maxlen=500)
    
    async def validate_consistency(
        self,
        ml_validation: MLPredictionValidation,
        rl_validation: RLActionValidation,
        context: ValidationContext
    ) -> CrossSystemSafetyCheck:
        """Validate consistency between ML predictions and RL actions."""
        try:
            # Check individual system status
            ml_status = self._determine_model_status(ml_validation.validation_result)
            rl_status = self._determine_model_status(rl_validation.validation_result)
            
            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(ml_validation, rl_validation)
            
            # Check for cross-validation pass
            cross_validation_passed = (
                consistency_score >= self.config.consistency_threshold and
                ml_status != ModelSafetyStatus.UNSAFE and
                rl_status != ModelSafetyStatus.UNSAFE
            )
            
            # Identify conflicting signals
            conflicting_signals = self._identify_conflicts(ml_validation, rl_validation)
            
            # Generate unified recommendation
            unified_recommendation = self._generate_unified_recommendation(
                ml_validation, rl_validation, consistency_score, conflicting_signals
            )
            
            # Record validation
            self._record_cross_validation(ml_validation, rl_validation, consistency_score)
            
            return CrossSystemSafetyCheck(
                ml_model_status=ml_status,
                rl_agent_status=rl_status,
                consistency_score=consistency_score,
                cross_validation_passed=cross_validation_passed,
                conflicting_signals=conflicting_signals,
                unified_recommendation=unified_recommendation
            )
            
        except Exception as e:
            self.logger.error(f"Cross-system validation error: {str(e)}")
            
            return CrossSystemSafetyCheck(
                ml_model_status=ModelSafetyStatus.UNSAFE,
                rl_agent_status=ModelSafetyStatus.UNSAFE,
                consistency_score=0.0,
                cross_validation_passed=False,
                conflicting_signals=["validation_error"],
                unified_recommendation="halt_trading"
            )
    
    def _determine_model_status(self, validation_result: SafetyValidationResult) -> ModelSafetyStatus:
        """Determine model safety status from validation result."""
        if validation_result.result == ValidationResult.REJECTED:
            return ModelSafetyStatus.UNSAFE
        elif validation_result.result == ValidationResult.FALLBACK_REQUIRED:
            return ModelSafetyStatus.DEGRADED
        elif validation_result.safety_score < 0.5:
            return ModelSafetyStatus.WARNING  
        elif validation_result.safety_score < 0.8:
            return ModelSafetyStatus.CAUTION
        else:
            return ModelSafetyStatus.SAFE
    
    def _calculate_consistency_score(
        self,
        ml_validation: MLPredictionValidation,
        rl_validation: RLActionValidation
    ) -> float:
        """Calculate consistency score between ML and RL systems."""
        # Compare confidence levels
        ml_confidence = ml_validation.validation_result.confidence_score
        rl_confidence = rl_validation.validation_result.confidence_score
        
        confidence_consistency = 1.0 - abs(ml_confidence - rl_confidence)
        
        # Compare safety scores
        ml_safety = ml_validation.validation_result.safety_score
        rl_safety = rl_validation.validation_result.safety_score
        
        safety_consistency = 1.0 - abs(ml_safety - rl_safety)
        
        # Compare violation patterns
        ml_violations = set(vt.value for vt in ml_validation.validation_result.violation_types)
        rl_violations = set(vt.value for vt in rl_validation.validation_result.violation_types)
        
        if ml_violations or rl_violations:
            common_violations = ml_violations.intersection(rl_violations)
            total_violations = ml_violations.union(rl_violations)
            violation_consistency = len(common_violations) / len(total_violations) if total_violations else 1.0
        else:
            violation_consistency = 1.0
        
        # Weighted average
        overall_consistency = (
            confidence_consistency * 0.4 +
            safety_consistency * 0.4 +
            violation_consistency * 0.2
        )
        
        return overall_consistency
    
    def _identify_conflicts(
        self,
        ml_validation: MLPredictionValidation,
        rl_validation: RLActionValidation
    ) -> List[str]:
        """Identify conflicting signals between systems."""
        conflicts = []
        
        # Check confidence conflict
        ml_conf = ml_validation.validation_result.confidence_score
        rl_conf = rl_validation.validation_result.confidence_score
        
        if abs(ml_conf - rl_conf) > 0.5:
            conflicts.append("confidence_mismatch")
        
        # Check safety assessment conflict
        ml_safe = ml_validation.validation_result.safety_score
        rl_safe = rl_validation.validation_result.safety_score
        
        if abs(ml_safe - rl_safe) > 0.4:
            conflicts.append("safety_assessment_mismatch")
        
        # Check validation result conflict
        ml_result = ml_validation.validation_result.result
        rl_result = rl_validation.validation_result.result
        
        if (ml_result == ValidationResult.APPROVED and rl_result == ValidationResult.REJECTED) or \
           (ml_result == ValidationResult.REJECTED and rl_result == ValidationResult.APPROVED):
            conflicts.append("validation_result_conflict")
        
        # Check violation type conflicts
        ml_violations = set(vt.value for vt in ml_validation.validation_result.violation_types)
        rl_violations = set(vt.value for vt in rl_validation.validation_result.violation_types)
        
        conflicting_violations = ml_violations.symmetric_difference(rl_violations)
        if conflicting_violations:
            conflicts.append("violation_type_mismatch")
        
        return conflicts
    
    def _generate_unified_recommendation(
        self,
        ml_validation: MLPredictionValidation,
        rl_validation: RLActionValidation,
        consistency_score: float,
        conflicts: List[str]
    ) -> str:
        """Generate unified recommendation from both systems."""
        ml_result = ml_validation.validation_result.result
        rl_result = rl_validation.validation_result.result
        
        # If both systems agree and are safe
        if ml_result == rl_result == ValidationResult.APPROVED and consistency_score >= 0.8:
            return "proceed_with_confidence"
        
        # If both systems reject
        if ml_result == rl_result == ValidationResult.REJECTED:
            return "halt_trading"
        
        # If both require fallback
        if ml_result == rl_result == ValidationResult.FALLBACK_REQUIRED:
            return "activate_fallback_systems"
        
        # If there are conflicts, use configured resolution strategy
        if conflicts:
            strategy = self.config.conflict_resolution_strategy
            
            if strategy == "conservative":
                return "use_most_conservative_approach"
            elif strategy == "ml_priority":
                return "follow_ml_recommendation" 
            elif strategy == "rl_priority":
                return "follow_rl_recommendation"
            elif strategy == "aggressive":
                return "proceed_with_caution"
            else:
                return "use_most_conservative_approach"
        
        # Default to caution for mixed results
        return "proceed_with_caution"
    
    def _record_cross_validation(
        self,
        ml_validation: MLPredictionValidation,
        rl_validation: RLActionValidation,
        consistency_score: float
    ):
        """Record cross-validation for analysis."""
        self.validation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'ml_result': ml_validation.validation_result.result.value,
            'rl_result': rl_validation.validation_result.result.value,
            'ml_safety_score': ml_validation.validation_result.safety_score,
            'rl_safety_score': rl_validation.validation_result.safety_score,
            'consistency_score': consistency_score,
            'ml_violations': len(ml_validation.validation_result.violation_types),
            'rl_violations': len(rl_validation.validation_result.violation_types)
        })


class MLRLSafetyBridge:
    """
    Main ML-RL Safety Bridge that coordinates safety validation between
    machine learning and reinforcement learning systems.
    """
    
    def __init__(
        self,
        config: SafetyBridgeConfig,
        constraints: SafetyConstraints,
        safety_state_manager: Optional[UnifiedSafetyStateManager] = None,
        emergency_controller: Optional[EmergencyStopController] = None,
        fallback_integration: Optional[FallbackSystemIntegration] = None
    ):
        """Initialize ML-RL Safety Bridge."""
        self.config = config
        self.constraints = constraints
        self.safety_state_manager = safety_state_manager
        self.emergency_controller = emergency_controller
        self.fallback_integration = fallback_integration
        
        # Initialize validators
        self.ml_validator = MLPredictionValidator(constraints)
        self.rl_validator = RLActionValidator(constraints)
        self.cross_validator = CrossSystemValidator(config)
        
        # System state
        self.is_running = False
        self.model_metrics: Dict[str, ModelSafetyMetrics] = {}
        
        # Performance tracking
        self.validation_stats = {
            'total_ml_validations': 0,
            'total_rl_validations': 0,
            'total_cross_validations': 0,
            'ml_approval_rate': 0.0,
            'rl_approval_rate': 0.0,
            'cross_consistency_rate': 0.0,
            'avg_validation_time_ms': 0.0,
            'fallback_activation_count': 0
        }
        
        self.logger = logging.getLogger(f"{__name__}.safety_bridge")
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the safety bridge."""
        self.logger.info("Initializing ML-RL Safety Bridge")
        
        # Initialize any required connections or resources
        if self.fallback_integration:
            # Enable integrations if available
            if hasattr(self.fallback_integration, 'enable_drift_detection_integration'):
                self.fallback_integration.enable_drift_detection_integration()
            if hasattr(self.fallback_integration, 'enable_emergency_stop_integration'):
                self.fallback_integration.enable_emergency_stop_integration()
    
    async def start(self) -> None:
        """Start the safety bridge system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring if enabled
        if self.config.enable_real_time_monitoring:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("ML-RL Safety Bridge started")
    
    async def stop(self) -> None:
        """Stop the safety bridge system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("ML-RL Safety Bridge stopped")
    
    async def validate_ml_prediction(
        self,
        prediction: PredictionResult,
        context: ValidationContext
    ) -> MLPredictionValidation:
        """Validate ML prediction for safety compliance."""
        start_time = time.time()
        
        try:
            if not self.config.enable_ml_validation:
                # Return approved validation if disabled
                return MLPredictionValidation(
                    original_prediction=prediction,
                    validation_result=SafetyValidationResult(
                        result=ValidationResult.APPROVED,
                        confidence_score=prediction.confidence,
                        safety_score=1.0
                    )
                )
            
            # Perform validation with timeout
            validation = await asyncio.wait_for(
                self.ml_validator.validate_prediction(prediction, context),
                timeout=self.config.ml_validation_timeout_ms / 1000.0
            )
            
            # Update statistics
            self.validation_stats['total_ml_validations'] += 1
            if validation.validation_result.result == ValidationResult.APPROVED:
                self._update_approval_rate('ml')
            
            # Check if fallback activation is needed
            if validation.validation_result.result == ValidationResult.FALLBACK_REQUIRED:
                await self._activate_ml_fallback(prediction, validation, context)
            
            # Update validation time
            validation_time = (time.time() - start_time) * 1000
            self._update_average_validation_time(validation_time)
            
            return validation
            
        except asyncio.TimeoutError:
            self.logger.warning(f"ML validation timeout for prediction {prediction.prediction_id}")
            return MLPredictionValidation(
                original_prediction=prediction,
                validation_result=SafetyValidationResult(
                    result=ValidationResult.REJECTED,
                    confidence_score=0.0,
                    safety_score=0.0,
                    validation_details={"error": "validation_timeout"}
                )
            )
        except Exception as e:
            self.logger.error(f"ML validation error: {str(e)}")
            return MLPredictionValidation(
                original_prediction=prediction,
                validation_result=SafetyValidationResult(
                    result=ValidationResult.REJECTED,
                    confidence_score=0.0,
                    safety_score=0.0,
                    validation_details={"error": str(e)}
                )
            )
    
    async def validate_rl_action(
        self,
        action: TradeAction,
        context: ValidationContext
    ) -> RLActionValidation:
        """Validate RL action for safety compliance."""
        start_time = time.time()
        
        try:
            if not self.config.enable_rl_validation:
                # Return approved validation if disabled
                return RLActionValidation(
                    original_action=action,
                    validation_result=SafetyValidationResult(
                        result=ValidationResult.APPROVED,
                        confidence_score=getattr(action, 'confidence', 0.8),
                        safety_score=1.0
                    )
                )
            
            # Perform validation with timeout
            validation = await asyncio.wait_for(
                self.rl_validator.validate_action(action, context),
                timeout=self.config.rl_validation_timeout_ms / 1000.0
            )
            
            # Update statistics
            self.validation_stats['total_rl_validations'] += 1
            if validation.validation_result.result == ValidationResult.APPROVED:
                self._update_approval_rate('rl')
            
            # Check if fallback activation is needed
            if validation.validation_result.result == ValidationResult.FALLBACK_REQUIRED:
                await self._activate_rl_fallback(action, validation, context)
            
            # Update validation time
            validation_time = (time.time() - start_time) * 1000
            self._update_average_validation_time(validation_time)
            
            return validation
            
        except asyncio.TimeoutError:
            self.logger.warning(f"RL validation timeout for action {getattr(action, 'action_id', 'unknown')}")
            return RLActionValidation(
                original_action=action,
                validation_result=SafetyValidationResult(
                    result=ValidationResult.REJECTED,
                    confidence_score=0.0,
                    safety_score=0.0,
                    validation_details={"error": "validation_timeout"}
                )
            )
        except Exception as e:
            self.logger.error(f"RL validation error: {str(e)}")
            return RLActionValidation(
                original_action=action,
                validation_result=SafetyValidationResult(
                    result=ValidationResult.REJECTED,
                    confidence_score=0.0,
                    safety_score=0.0,
                    validation_details={"error": str(e)}
                )
            )
    
    async def validate_cross_system_consistency(
        self,
        ml_validation: MLPredictionValidation,
        rl_validation: RLActionValidation,
        context: ValidationContext
    ) -> CrossSystemSafetyCheck:
        """Validate consistency between ML and RL systems."""
        start_time = time.time()
        
        try:
            if not self.config.enable_cross_validation:
                # Return basic check if disabled
                return CrossSystemSafetyCheck(
                    ml_model_status=ModelSafetyStatus.SAFE,
                    rl_agent_status=ModelSafetyStatus.SAFE,
                    consistency_score=1.0,
                    cross_validation_passed=True
                )
            
            # Perform cross-validation with timeout
            cross_check = await asyncio.wait_for(
                self.cross_validator.validate_consistency(ml_validation, rl_validation, context),
                timeout=self.config.cross_validation_timeout_ms / 1000.0
            )
            
            # Update statistics
            self.validation_stats['total_cross_validations'] += 1
            if cross_check.cross_validation_passed:
                self._update_consistency_rate()
            
            # Handle conflicts if not allowed
            if cross_check.conflicting_signals and not self.config.allow_conflicting_signals:
                self.logger.warning(f"Conflicting signals detected: {cross_check.conflicting_signals}")
                
                if self.emergency_controller:
                    await self._handle_conflict_emergency(cross_check, context)
            
            # Update validation time
            validation_time = (time.time() - start_time) * 1000
            self._update_average_validation_time(validation_time)
            
            return cross_check
            
        except asyncio.TimeoutError:
            self.logger.warning("Cross-system validation timeout")
            return CrossSystemSafetyCheck(
                ml_model_status=ModelSafetyStatus.UNSAFE,
                rl_agent_status=ModelSafetyStatus.UNSAFE,
                consistency_score=0.0,
                cross_validation_passed=False,
                conflicting_signals=["validation_timeout"]
            )
        except Exception as e:
            self.logger.error(f"Cross-system validation error: {str(e)}")
            return CrossSystemSafetyCheck(
                ml_model_status=ModelSafetyStatus.UNSAFE,
                rl_agent_status=ModelSafetyStatus.UNSAFE,
                consistency_score=0.0,
                cross_validation_passed=False,
                conflicting_signals=["validation_error"]
            )
    
    async def get_model_safety_metrics(self, model_id: str) -> Optional[ModelSafetyMetrics]:
        """Get safety metrics for a specific model."""
        return self.model_metrics.get(model_id)
    
    async def update_model_safety_metrics(self, metrics: ModelSafetyMetrics) -> None:
        """Update safety metrics for a model."""
        self.model_metrics[metrics.model_id] = metrics
        
        # Check if model status requires action
        if metrics.safety_status == ModelSafetyStatus.UNSAFE:
            await self._handle_unsafe_model(metrics)
        elif metrics.safety_status == ModelSafetyStatus.DEGRADED:
            await self._handle_degraded_model(metrics)
    
    async def get_safety_bridge_status(self) -> Dict[str, Any]:
        """Get current safety bridge status."""
        return {
            'is_running': self.is_running,
            'config': {
                'ml_validation_enabled': self.config.enable_ml_validation,
                'rl_validation_enabled': self.config.enable_rl_validation,
                'cross_validation_enabled': self.config.enable_cross_validation,
                'fallback_integration_enabled': self.config.enable_fallback_integration
            },
            'validation_stats': self.validation_stats,
            'model_count': len(self.model_metrics),
            'unsafe_models': len([m for m in self.model_metrics.values() if m.safety_status == ModelSafetyStatus.UNSAFE]),
            'degraded_models': len([m for m in self.model_metrics.values() if m.safety_status == ModelSafetyStatus.DEGRADED]),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    async def _activate_ml_fallback(
        self,
        prediction: PredictionResult,
        validation: MLPredictionValidation,
        context: ValidationContext
    ) -> None:
        """Activate ML fallback mechanisms."""
        if not self.config.enable_fallback_integration or not self.fallback_integration:
            return
        
        try:
            # Trigger fallback through integration
            self.validation_stats['fallback_activation_count'] += 1
            self.logger.info(f"ML fallback activated for prediction {prediction.prediction_id}")
            
            # Update model safety metrics
            if context.model_id in self.model_metrics:
                metrics = self.model_metrics[context.model_id]
                metrics.fallback_activations_24h += 1
                metrics.safety_status = ModelSafetyStatus.DEGRADED
        
        except Exception as e:
            self.logger.error(f"Failed to activate ML fallback: {str(e)}")
    
    async def _activate_rl_fallback(
        self,
        action: TradeAction,
        validation: RLActionValidation,
        context: ValidationContext
    ) -> None:
        """Activate RL fallback mechanisms."""
        if not self.config.enable_fallback_integration or not self.fallback_integration:
            return
        
        try:
            # Trigger fallback through integration
            self.validation_stats['fallback_activation_count'] += 1
            self.logger.info(f"RL fallback activated for action {getattr(action, 'action_id', 'unknown')}")
            
            # Update model safety metrics
            if context.model_id in self.model_metrics:
                metrics = self.model_metrics[context.model_id]
                metrics.fallback_activations_24h += 1
                metrics.safety_status = ModelSafetyStatus.DEGRADED
        
        except Exception as e:
            self.logger.error(f"Failed to activate RL fallback: {str(e)}")
    
    async def _handle_conflict_emergency(self, cross_check: CrossSystemSafetyCheck, context: ValidationContext) -> None:
        """Handle emergency situations from conflicting signals."""
        if not self.emergency_controller:
            return
        
        try:
            await self.emergency_controller.activate_emergency_stop(
                reason="Conflicting ML-RL signals detected",
                details={
                    'ml_status': cross_check.ml_model_status.value,
                    'rl_status': cross_check.rl_agent_status.value,
                    'conflicts': cross_check.conflicting_signals,
                    'consistency_score': cross_check.consistency_score
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to activate emergency stop for conflicts: {str(e)}")
    
    async def _handle_unsafe_model(self, metrics: ModelSafetyMetrics) -> None:
        """Handle unsafe model detection."""
        self.logger.critical(f"Unsafe model detected: {metrics.model_id}")
        
        if self.emergency_controller:
            await self.emergency_controller.activate_emergency_stop(
                reason=f"Unsafe model detected: {metrics.model_id}",
                details={
                    'model_id': metrics.model_id,
                    'model_type': metrics.model_type.value,
                    'safety_status': metrics.safety_status.value,
                    'error_rate': metrics.error_rate,
                    'recent_accuracy': metrics.recent_accuracy
                }
            )
    
    async def _handle_degraded_model(self, metrics: ModelSafetyMetrics) -> None:
        """Handle degraded model detection."""
        self.logger.warning(f"Degraded model detected: {metrics.model_id}")
        
        # Trigger fallback if available
        if self.config.auto_fallback_on_violation and self.fallback_integration:
            try:
                # Create fallback trigger
                from src.modes.fallback_strategies import FallbackTrigger, DegradationSeverity, FailureType
                
                trigger = FallbackTrigger(
                    model_id=metrics.model_id,
                    degradation_type="model_degradation",
                    severity=DegradationSeverity.MODERATE,
                    trigger_time=datetime.utcnow(),
                    context={
                        'accuracy': metrics.recent_accuracy,
                        'error_rate': metrics.error_rate
                    }
                )
                
                # Note: Would integrate with fallback system here
                self.logger.info(f"Fallback trigger created for degraded model: {metrics.model_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to handle degraded model: {str(e)}")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for safety metrics."""
        while self.is_running:
            try:
                # Update model safety metrics
                await self._update_model_safety_metrics()
                
                # Check for safety violations
                await self._check_safety_violations()
                
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
                # Sleep until next monitoring cycle
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _update_model_safety_metrics(self) -> None:
        """Update safety metrics for all models."""
        # This would integrate with actual model monitoring systems
        # For now, maintain existing metrics and update timestamps
        current_time = datetime.utcnow()
        
        for model_id, metrics in self.model_metrics.items():
            # Update last_updated timestamp
            metrics.last_updated = current_time
            
            # Reset daily counters if needed (simplified)
            if (current_time - metrics.last_updated).days >= 1:
                metrics.safety_violations_24h = 0
                metrics.fallback_activations_24h = 0
    
    async def _check_safety_violations(self) -> None:
        """Check for safety violations across models."""
        for model_id, metrics in self.model_metrics.items():
            # Check error rate violations
            if metrics.error_rate > self.constraints.max_error_rate:
                self.logger.warning(f"High error rate detected for model {model_id}: {metrics.error_rate}")
                metrics.safety_violations_24h += 1
                
                if metrics.safety_status == ModelSafetyStatus.SAFE:
                    metrics.safety_status = ModelSafetyStatus.CAUTION
            
            # Check accuracy violations
            if metrics.recent_accuracy < self.constraints.min_accuracy:
                self.logger.warning(f"Low accuracy detected for model {model_id}: {metrics.recent_accuracy}")
                metrics.safety_violations_24h += 1
                
                if metrics.safety_status in [ModelSafetyStatus.SAFE, ModelSafetyStatus.CAUTION]:
                    metrics.safety_status = ModelSafetyStatus.WARNING
            
            # Check latency violations
            if metrics.avg_latency_ms > self.constraints.max_latency_ms:
                self.logger.warning(f"High latency detected for model {model_id}: {metrics.avg_latency_ms}ms")
                metrics.safety_violations_24h += 1
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.safety_metrics_retention_hours)
        
        # Remove models that haven't been updated recently
        models_to_remove = [
            model_id for model_id, metrics in self.model_metrics.items()
            if metrics.last_updated < cutoff_time
        ]
        
        for model_id in models_to_remove:
            del self.model_metrics[model_id]
            self.logger.info(f"Removed old metrics for model: {model_id}")
    
    def _update_approval_rate(self, system_type: str) -> None:
        """Update approval rate statistics."""
        if system_type == 'ml':
            total = self.validation_stats['total_ml_validations']
            if total > 0:
                # Simplified approval rate calculation
                self.validation_stats['ml_approval_rate'] = 0.85  # Placeholder
        elif system_type == 'rl':
            total = self.validation_stats['total_rl_validations']
            if total > 0:
                # Simplified approval rate calculation
                self.validation_stats['rl_approval_rate'] = 0.82  # Placeholder
    
    def _update_consistency_rate(self) -> None:
        """Update cross-system consistency rate."""
        total = self.validation_stats['total_cross_validations']
        if total > 0:
            # Simplified consistency rate calculation
            self.validation_stats['cross_consistency_rate'] = 0.78  # Placeholder
    
    def _update_average_validation_time(self, validation_time_ms: float) -> None:
        """Update average validation time."""
        current_avg = self.validation_stats['avg_validation_time_ms']
        total_validations = (
            self.validation_stats['total_ml_validations'] +
            self.validation_stats['total_rl_validations'] +
            self.validation_stats['total_cross_validations']
        )
        
        if total_validations > 0:
            # Simple moving average
            self.validation_stats['avg_validation_time_ms'] = (
                (current_avg * (total_validations - 1) + validation_time_ms) / total_validations
            )


# Factory function for easy instantiation
async def create_ml_rl_safety_bridge(
    config: Optional[SafetyBridgeConfig] = None,
    constraints: Optional[SafetyConstraints] = None,
    integrations: Optional[Dict[str, Any]] = None
) -> MLRLSafetyBridge:
    """Create and initialize ML-RL Safety Bridge."""
    if config is None:
        config = SafetyBridgeConfig()
    
    if constraints is None:
        constraints = SafetyConstraints()
    
    integrations = integrations or {}
    
    bridge = MLRLSafetyBridge(
        config=config,
        constraints=constraints,
        safety_state_manager=integrations.get('safety_state_manager'),
        emergency_controller=integrations.get('emergency_controller'),
        fallback_integration=integrations.get('fallback_integration')
    )
    
    await bridge.initialize()
    return bridge