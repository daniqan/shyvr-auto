"""
Transformer Safety Thresholds Configuration for Phase 3.2.3.3

This module defines comprehensive safety thresholds and configuration for
transformer-specific anomaly detection in the RLTE system.

Features:
- Configurable thresholds for each anomaly type
- Market condition-based threshold adjustment
- Production-ready threshold validation
- Integration with existing safety systems
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime

from .attention_safety_monitors import AttentionAnomalyType, SafetyAction

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classifications for threshold adjustment"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


class ThresholdLevel(Enum):
    """Threshold severity levels"""
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ThresholdConfig:
    """Configuration for a single threshold level"""
    value: float
    action: SafetyAction
    description: str = ""
    
    def __post_init__(self):
        """Validate threshold configuration"""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Threshold value must be between 0.0 and 1.0, got {self.value}")


@dataclass
class AnomalyThresholds:
    """Thresholds for a specific anomaly type"""
    warning: ThresholdConfig
    critical: ThresholdConfig
    emergency: ThresholdConfig
    
    def __post_init__(self):
        """Validate threshold ordering"""
        if not (self.warning.value < self.critical.value < self.emergency.value):
            raise ValueError("Thresholds must be in ascending order: warning < critical < emergency")


class TransformerSafetyThresholds:
    """Comprehensive transformer safety threshold management system"""
    
    def __init__(
        self,
        attention_weight_instability: Optional[Dict[str, float]] = None,
        cross_asset_anomaly: Optional[Dict[str, float]] = None,
        temporal_drift: Optional[Dict[str, float]] = None,
        feature_collapse: Optional[Dict[str, float]] = None,
        entropy_change: Optional[Dict[str, float]] = None,
        head_disagreement: Optional[Dict[str, float]] = None,
        sparsity_violation: Optional[Dict[str, float]] = None,
        market_adjustment_enabled: bool = True,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize transformer safety thresholds.
        
        Args:
            attention_weight_instability: Thresholds for attention weight instability
            cross_asset_anomaly: Thresholds for cross-asset attention anomalies
            temporal_drift: Thresholds for temporal attention drift
            feature_collapse: Thresholds for feature attention collapse
            entropy_change: Thresholds for attention entropy changes
            head_disagreement: Thresholds for head disagreement patterns
            sparsity_violation: Thresholds for attention sparsity violations
            market_adjustment_enabled: Enable market condition-based threshold adjustment
            custom_config: Custom configuration overrides
        """
        self.market_adjustment_enabled = market_adjustment_enabled
        self.custom_config = custom_config or {}
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Initialize default thresholds
        self.thresholds = self._initialize_default_thresholds()
        
        # Override with provided configurations
        threshold_configs = {
            AttentionAnomalyType.ATTENTION_WEIGHT_INSTABILITY: attention_weight_instability,
            AttentionAnomalyType.CROSS_ASSET_ANOMALY: cross_asset_anomaly,
            AttentionAnomalyType.TEMPORAL_DRIFT: temporal_drift,
            AttentionAnomalyType.FEATURE_COLLAPSE: feature_collapse,
            AttentionAnomalyType.ENTROPY_CHANGE: entropy_change,
            AttentionAnomalyType.HEAD_DISAGREEMENT: head_disagreement,
            AttentionAnomalyType.SPARSITY_VIOLATION: sparsity_violation
        }
        
        for anomaly_type, config in threshold_configs.items():
            if config:
                self._update_thresholds(anomaly_type, config)
        
        # Store original thresholds for market adjustment
        self.base_thresholds = self._deep_copy_thresholds(self.thresholds)
        self.current_market_regime = MarketRegime.SIDEWAYS
        self.last_adjustment = datetime.now()
    
    def _initialize_default_thresholds(self) -> Dict[AttentionAnomalyType, AnomalyThresholds]:
        """Initialize default thresholds for all anomaly types"""
        return {
            AttentionAnomalyType.ATTENTION_WEIGHT_INSTABILITY: AnomalyThresholds(
                warning=ThresholdConfig(0.3, SafetyAction.ALERT_GENERATION, "Moderate attention instability"),
                critical=ThresholdConfig(0.6, SafetyAction.POSITION_REDUCTION, "High attention instability"),
                emergency=ThresholdConfig(0.9, SafetyAction.EMERGENCY_STOP, "Critical attention instability")
            ),
            AttentionAnomalyType.CROSS_ASSET_ANOMALY: AnomalyThresholds(
                warning=ThresholdConfig(0.4, SafetyAction.ALERT_GENERATION, "Unexpected cross-asset correlations"),
                critical=ThresholdConfig(0.7, SafetyAction.POSITION_REDUCTION, "High cross-asset anomalies"),
                emergency=ThresholdConfig(0.85, SafetyAction.MODEL_FALLBACK, "Critical cross-asset anomalies")
            ),
            AttentionAnomalyType.TEMPORAL_DRIFT: AnomalyThresholds(
                warning=ThresholdConfig(0.3, SafetyAction.ALERT_GENERATION, "Temporal attention drift detected"),
                critical=ThresholdConfig(0.6, SafetyAction.MODEL_FALLBACK, "Significant temporal drift"),
                emergency=ThresholdConfig(0.8, SafetyAction.EMERGENCY_STOP, "Critical temporal drift")
            ),
            AttentionAnomalyType.FEATURE_COLLAPSE: AnomalyThresholds(
                warning=ThresholdConfig(0.5, SafetyAction.POSITION_REDUCTION, "Feature attention narrowing"),
                critical=ThresholdConfig(0.7, SafetyAction.MODEL_FALLBACK, "Feature attention collapse"),
                emergency=ThresholdConfig(0.9, SafetyAction.EMERGENCY_STOP, "Critical feature collapse")
            ),
            AttentionAnomalyType.ENTROPY_CHANGE: AnomalyThresholds(
                warning=ThresholdConfig(0.2, SafetyAction.ALERT_GENERATION, "Attention entropy deviation"),
                critical=ThresholdConfig(0.4, SafetyAction.POSITION_REDUCTION, "High entropy deviation"),
                emergency=ThresholdConfig(0.7, SafetyAction.MODEL_FALLBACK, "Critical entropy deviation")
            ),
            AttentionAnomalyType.HEAD_DISAGREEMENT: AnomalyThresholds(
                warning=ThresholdConfig(0.4, SafetyAction.ALERT_GENERATION, "Attention head disagreement"),
                critical=ThresholdConfig(0.6, SafetyAction.MODEL_FALLBACK, "High head disagreement"),
                emergency=ThresholdConfig(0.8, SafetyAction.EMERGENCY_STOP, "Critical head disagreement")
            ),
            AttentionAnomalyType.SPARSITY_VIOLATION: AnomalyThresholds(
                warning=ThresholdConfig(0.75, SafetyAction.POSITION_REDUCTION, "Attention sparsity anomaly"),
                critical=ThresholdConfig(0.85, SafetyAction.MODEL_FALLBACK, "High sparsity violation"),
                emergency=ThresholdConfig(0.95, SafetyAction.EMERGENCY_STOP, "Critical sparsity violation")
            )
        }
    
    def _update_thresholds(self, anomaly_type: AttentionAnomalyType, config: Dict[str, float]) -> None:
        """Update thresholds for a specific anomaly type"""
        if anomaly_type not in self.thresholds:
            self.logger.warning(f"Unknown anomaly type: {anomaly_type}")
            return
        
        current = self.thresholds[anomaly_type]
        
        # Update threshold values while preserving actions and descriptions
        if "warning" in config:
            current.warning.value = config["warning"]
        if "critical" in config:
            current.critical.value = config["critical"]
        if "emergency" in config:
            current.emergency.value = config["emergency"]
        
        # Validate updated thresholds
        try:
            current.__post_init__()
        except ValueError as e:
            self.logger.error(f"Invalid threshold configuration for {anomaly_type}: {e}")
            raise
    
    def get_threshold(self, anomaly_type: AttentionAnomalyType, level: str) -> float:
        """Get threshold value for specific anomaly type and level"""
        if anomaly_type not in self.thresholds:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        thresholds = self.thresholds[anomaly_type]
        
        if level == "warning":
            return thresholds.warning.value
        elif level == "critical":
            return thresholds.critical.value
        elif level == "emergency":
            return thresholds.emergency.value
        else:
            raise ValueError(f"Unknown threshold level: {level}")
    
    def get_action(self, anomaly_type: AttentionAnomalyType, severity: float) -> SafetyAction:
        """Get recommended action based on anomaly type and severity"""
        if anomaly_type not in self.thresholds:
            return SafetyAction.ALERT_GENERATION
        
        thresholds = self.thresholds[anomaly_type]
        
        if severity >= thresholds.emergency.value:
            return thresholds.emergency.action
        elif severity >= thresholds.critical.value:
            return thresholds.critical.action
        elif severity >= thresholds.warning.value:
            return thresholds.warning.action
        else:
            return SafetyAction.ALERT_GENERATION
    
    def get_threshold_level(self, anomaly_type: AttentionAnomalyType, severity: float) -> ThresholdLevel:
        """Get threshold level based on anomaly type and severity"""
        if anomaly_type not in self.thresholds:
            return ThresholdLevel.WARNING
        
        thresholds = self.thresholds[anomaly_type]
        
        if severity >= thresholds.emergency.value:
            return ThresholdLevel.EMERGENCY
        elif severity >= thresholds.critical.value:
            return ThresholdLevel.CRITICAL
        elif severity >= thresholds.warning.value:
            return ThresholdLevel.WARNING
        else:
            return ThresholdLevel.WARNING
    
    async def adjust_thresholds_for_market_conditions(
        self,
        volatility: float,
        trend: str,
        market_stress: Optional[float] = None
    ) -> None:
        """Adjust thresholds based on current market conditions"""
        if not self.market_adjustment_enabled:
            return
        
        # Determine market regime
        new_regime = self._classify_market_regime(volatility, trend, market_stress)
        
        if new_regime != self.current_market_regime:
            self.logger.info(
                "Market regime change detected",
                old_regime=self.current_market_regime.value,
                new_regime=new_regime.value,
                volatility=volatility,
                trend=trend
            )
            
            # Adjust thresholds based on regime
            adjustment_factors = self._get_adjustment_factors(new_regime)
            self._apply_threshold_adjustments(adjustment_factors)
            
            self.current_market_regime = new_regime
            self.last_adjustment = datetime.now()
    
    def _classify_market_regime(
        self,
        volatility: float,
        trend: str,
        market_stress: Optional[float] = None
    ) -> MarketRegime:
        """Classify current market regime based on conditions"""
        # Crisis conditions take precedence
        if market_stress and market_stress > 0.8:
            return MarketRegime.CRISIS
        
        # Volatility-based classification
        if volatility > 0.05:  # 5% daily volatility
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.01:  # 1% daily volatility
            return MarketRegime.LOW_VOLATILITY
        
        # Trend-based classification
        if trend.lower() in ["bull", "bullish", "up"]:
            return MarketRegime.BULL
        elif trend.lower() in ["bear", "bearish", "down"]:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    def _get_adjustment_factors(self, regime: MarketRegime) -> Dict[str, float]:
        """Get threshold adjustment factors for market regime"""
        if regime == MarketRegime.CRISIS:
            # Tighten all thresholds significantly during crisis
            return {"multiplier": 0.7, "description": "Crisis mode - tightened thresholds"}
        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Slightly tighten thresholds during high volatility
            return {"multiplier": 0.85, "description": "High volatility - tightened thresholds"}
        elif regime == MarketRegime.LOW_VOLATILITY:
            # Relax thresholds slightly during low volatility
            return {"multiplier": 1.15, "description": "Low volatility - relaxed thresholds"}
        elif regime == MarketRegime.BEAR:
            # Tighten thresholds during bear markets
            return {"multiplier": 0.9, "description": "Bear market - tightened thresholds"}
        elif regime == MarketRegime.BULL:
            # Slightly relax thresholds during bull markets
            return {"multiplier": 1.05, "description": "Bull market - slightly relaxed thresholds"}
        else:  # SIDEWAYS
            # Use base thresholds
            return {"multiplier": 1.0, "description": "Sideways market - base thresholds"}
    
    def _apply_threshold_adjustments(self, adjustment_factors: Dict[str, float]) -> None:
        """Apply adjustment factors to thresholds"""
        multiplier = adjustment_factors["multiplier"]
        
        for anomaly_type, base_thresholds in self.base_thresholds.items():
            current_thresholds = self.thresholds[anomaly_type]
            
            # Apply multiplier while ensuring thresholds stay within bounds
            current_thresholds.warning.value = min(0.99, max(0.01, base_thresholds.warning.value * multiplier))
            current_thresholds.critical.value = min(0.99, max(0.01, base_thresholds.critical.value * multiplier))
            current_thresholds.emergency.value = min(0.99, max(0.01, base_thresholds.emergency.value * multiplier))
            
            # Ensure threshold ordering is preserved
            if not (current_thresholds.warning.value < current_thresholds.critical.value < current_thresholds.emergency.value):
                # Reset to base values if ordering is violated
                current_thresholds.warning.value = base_thresholds.warning.value
                current_thresholds.critical.value = base_thresholds.critical.value
                current_thresholds.emergency.value = base_thresholds.emergency.value
        
        self.logger.info(
            "Thresholds adjusted for market conditions",
            adjustment=adjustment_factors["description"],
            multiplier=multiplier
        )
    
    def _deep_copy_thresholds(self, thresholds: Dict[AttentionAnomalyType, AnomalyThresholds]) -> Dict[AttentionAnomalyType, AnomalyThresholds]:
        """Create deep copy of thresholds for baseline storage"""
        copied = {}
        for anomaly_type, threshold_set in thresholds.items():
            copied[anomaly_type] = AnomalyThresholds(
                warning=ThresholdConfig(
                    threshold_set.warning.value,
                    threshold_set.warning.action,
                    threshold_set.warning.description
                ),
                critical=ThresholdConfig(
                    threshold_set.critical.value,
                    threshold_set.critical.action,
                    threshold_set.critical.description
                ),
                emergency=ThresholdConfig(
                    threshold_set.emergency.value,
                    threshold_set.emergency.action,
                    threshold_set.emergency.description
                )
            )
        return copied
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate entire threshold configuration"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        try:
            for anomaly_type, thresholds in self.thresholds.items():
                # Validate individual threshold set
                try:
                    thresholds.__post_init__()
                except ValueError as e:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"{anomaly_type.value}: {e}")
                
                # Check for reasonable threshold values
                if thresholds.warning.value < 0.1:
                    validation_results["warnings"].append(f"{anomaly_type.value}: Very low warning threshold ({thresholds.warning.value})")
                
                if thresholds.emergency.value > 0.95:
                    validation_results["warnings"].append(f"{anomaly_type.value}: Very high emergency threshold ({thresholds.emergency.value})")
                
                # Store threshold summary
                validation_results["summary"][anomaly_type.value] = {
                    "warning": thresholds.warning.value,
                    "critical": thresholds.critical.value,
                    "emergency": thresholds.emergency.value
                }
            
            if validation_results["valid"]:
                self.logger.info("Threshold configuration validation passed")
            else:
                self.logger.error("Threshold configuration validation failed", errors=validation_results["errors"])
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Configuration validation error: {e}")
            self.logger.error("Configuration validation exception", error=str(e))
        
        return validation_results
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary"""
        return {
            "thresholds": {
                anomaly_type.value: {
                    "warning": {
                        "value": thresholds.warning.value,
                        "action": thresholds.warning.action.value,
                        "description": thresholds.warning.description
                    },
                    "critical": {
                        "value": thresholds.critical.value,
                        "action": thresholds.critical.action.value,
                        "description": thresholds.critical.description
                    },
                    "emergency": {
                        "value": thresholds.emergency.value,
                        "action": thresholds.emergency.action.value,
                        "description": thresholds.emergency.description
                    }
                }
                for anomaly_type, thresholds in self.thresholds.items()
            },
            "market_adjustment": {
                "enabled": self.market_adjustment_enabled,
                "current_regime": self.current_market_regime.value,
                "last_adjustment": self.last_adjustment.isoformat()
            },
            "custom_config": self.custom_config
        }
    
    def reset_to_base_thresholds(self) -> None:
        """Reset all thresholds to their base values"""
        self.thresholds = self._deep_copy_thresholds(self.base_thresholds)
        self.current_market_regime = MarketRegime.SIDEWAYS
        self.last_adjustment = datetime.now()
        self.logger.info("Thresholds reset to base values")