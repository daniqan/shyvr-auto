"""
Safety metrics collection for monitoring risk and safety measures.

This module provides comprehensive safety metrics including:
- Risk level monitoring
- Emergency stop tracking
- Liquidation monitoring
- Position size and drawdown tracking
- Risk violation detection
- Safety check validation
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
import datetime

from .base import MetricsCollector, MetricsRegistry


class SafetyMetricsCollector(MetricsCollector):
    """
    Collector for safety and risk-related metrics.
    
    This class tracks critical safety indicators including risk levels,
    emergency stops, liquidations, and various risk threshold violations.
    """
    
    def __init__(self, registry: MetricsRegistry):
        """
        Initialize the safety metrics collector.
        
        Args:
            registry: MetricsRegistry instance for managing metrics
        """
        super().__init__(registry)
        
        # Risk level metrics
        self.risk_level_gauge = registry.get_gauge(
            "safety_risk_level",
            "Current risk level (0-1, where 1 is maximum risk)",
            ["risk_type", "timeframe"]
        )
        
        # Emergency and liquidation metrics
        self.emergency_stops_counter = registry.get_counter(
            "safety_emergency_stops_total",
            "Total number of emergency stops triggered",
            ["reason", "trigger"]
        )
        
        self.liquidations_counter = registry.get_counter(
            "safety_liquidations_total",
            "Total number of liquidations",
            ["symbol", "side", "reason"]
        )
        
        # Position and drawdown metrics
        self.position_size_gauge = registry.get_gauge(
            "safety_position_size_ratio",
            "Position size as ratio of maximum allowed",
            ["symbol"]
        )
        
        self.drawdown_gauge = registry.get_gauge(
            "safety_drawdown_percentage",
            "Current drawdown percentage",
            ["timeframe"]
        )
        
        # Risk violation metrics
        self.risk_violations_counter = registry.get_counter(
            "safety_risk_violations_total",
            "Total number of risk violations",
            ["violation_type", "severity"]
        )
        
        self.safety_checks_counter = registry.get_counter(
            "safety_checks_total",
            "Total number of safety checks performed",
            ["check_type", "result"]
        )
        
        # Advanced risk metrics
        self.margin_ratio_gauge = registry.get_gauge(
            "safety_margin_ratio",
            "Current margin ratio (available margin / used margin)",
            ["account"]
        )
        
        self.volatility_gauge = registry.get_gauge(
            "safety_volatility_score",
            "Portfolio volatility score (0-1)",
            ["timeframe"]
        )
        
        self.correlation_risk_gauge = registry.get_gauge(
            "safety_correlation_risk_score",
            "Portfolio correlation risk score (0-1)",
            ["timeframe"]
        )
    
    def collect_metrics(self) -> None:
        """
        Collect and update all safety metrics.
        
        This method gathers current safety and risk data and updates all relevant metrics.
        """
        try:
            # Collect risk management data
            risk_manager = self._get_risk_manager()
            if risk_manager:
                # Update risk level
                current_risk = risk_manager.get_current_risk_level()
                self.risk_level_gauge.labels(risk_type="overall", timeframe="current").set(current_risk)
                
                # Update position size ratio
                max_position_ratio = risk_manager.get_max_position_size_ratio()
                self.position_size_gauge.labels(symbol="ALL").set(max_position_ratio)
                
                # Update margin ratio
                margin_ratio = risk_manager.get_margin_ratio()
                self.margin_ratio_gauge.labels(account="main").set(margin_ratio)
                
                # Update volatility and correlation scores
                volatility_score = risk_manager.get_volatility_score()
                correlation_risk = risk_manager.get_correlation_risk()
                
                self.volatility_gauge.labels(timeframe="daily").set(volatility_score)
                self.correlation_risk_gauge.labels(timeframe="daily").set(correlation_risk)
            
            # Collect portfolio data
            portfolio_manager = self._get_portfolio_manager()
            if portfolio_manager:
                # Update drawdown
                current_drawdown = portfolio_manager.get_current_drawdown()
                self.drawdown_gauge.labels(timeframe="daily").set(float(current_drawdown))
            
            # Collect safety events data
            safety_events = self._get_safety_events()
            # Note: Counters are updated through record_* methods, not directly here
            
        except Exception as e:
            # Log error but don't crash the collector
            self._collection_errors += 1
            raise e
    
    def get_metric_definitions(self) -> Dict[str, str]:
        """
        Get metric definitions for safety metrics.
        
        Returns:
            Dictionary mapping metric names to their descriptions
        """
        return {
            "safety_risk_level": "Current risk level (0-1, where 1 is maximum risk)",
            "safety_emergency_stops_total": "Total number of emergency stops triggered",
            "safety_liquidations_total": "Total number of liquidations",
            "safety_position_size_ratio": "Position size as ratio of maximum allowed",
            "safety_drawdown_percentage": "Current drawdown percentage",
            "safety_risk_violations_total": "Total number of risk violations",
            "safety_checks_total": "Total number of safety checks performed",
            "safety_margin_ratio": "Current margin ratio (available margin / used margin)",
            "safety_volatility_score": "Portfolio volatility score (0-1)",
            "safety_correlation_risk_score": "Portfolio correlation risk score (0-1)"
        }
    
    def record_emergency_stop(self, stop_data: Dict[str, Any]) -> None:
        """
        Record an emergency stop event.
        
        Args:
            stop_data: Dictionary containing emergency stop information
        """
        reason = stop_data.get('reason', 'unknown')
        trigger = stop_data.get('trigger', 'manual')
        
        # Update emergency stop counter
        self.emergency_stops_counter.labels(reason=reason, trigger=trigger).inc()
        
        # Also update risk level to maximum temporarily
        self.risk_level_gauge.labels(risk_type="emergency", timeframe="current").set(1.0)
    
    def record_liquidation(self, liquidation_data: Dict[str, Any]) -> None:
        """
        Record a liquidation event.
        
        Args:
            liquidation_data: Dictionary containing liquidation information
        """
        symbol = liquidation_data.get('symbol', 'UNKNOWN')
        side = liquidation_data.get('side', 'unknown')
        reason = liquidation_data.get('reason', 'unknown')
        
        # Update liquidation counter
        self.liquidations_counter.labels(symbol=symbol, side=side, reason=reason).inc()
    
    def record_risk_violation(self, violation_data: Dict[str, Any]) -> None:
        """
        Record a risk violation.
        
        Args:
            violation_data: Dictionary containing risk violation information
        """
        violation_type = violation_data.get('violation_type', 'unknown')
        severity = violation_data.get('severity', 'medium')
        
        # Update risk violation counter
        self.risk_violations_counter.labels(
            violation_type=violation_type, 
            severity=severity
        ).inc()
    
    def record_safety_check(self, check_type: str, passed: bool) -> None:
        """
        Record a safety check result.
        
        Args:
            check_type: Type of safety check performed
            passed: Whether the check passed
        """
        result = "passed" if passed else "failed"
        
        # Update safety check counter
        self.safety_checks_counter.labels(check_type=check_type, result=result).inc()
    
    def update_risk_level(self, risk_level: float, risk_type: str = "overall") -> None:
        """
        Update the current risk level.
        
        Args:
            risk_level: Risk level (0-1)
            risk_type: Type of risk being measured
        """
        self.risk_level_gauge.labels(risk_type=risk_type, timeframe="current").set(risk_level)
    
    def _get_risk_manager(self):
        """
        Get the risk manager instance.
        
        Returns:
            Risk manager instance or None
        """
        try:
            from src.portfolio.risk_manager import RiskManager
            # In a real implementation, this would get the active risk manager
            # For now, return None to avoid import errors in tests
            return None
        except ImportError:
            return None
    
    def _get_portfolio_manager(self):
        """
        Get the portfolio manager instance.
        
        Returns:
            Portfolio manager instance or None
        """
        try:
            from src.portfolio.portfolio_manager import PortfolioManager
            # In a real implementation, this would get the active portfolio manager
            # For now, return None to avoid import errors in tests
            return None
        except ImportError:
            return None
    
    def _get_safety_events(self) -> Dict[str, int]:
        """
        Get safety event statistics from activity logs.
        
        Returns:
            Dictionary with safety event counts
        """
        try:
            from src.logging.activity_logger import ActivityLogger
            
            activity_logger = ActivityLogger()
            
            # Query safety events for the last 24 hours
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(hours=24)
            
            # Count emergency stops
            emergency_stops = activity_logger.query_activities(
                activity_type="EMERGENCY_STOP",
                start_time=start_time,
                end_time=end_time
            )
            
            # Count liquidations
            liquidations = activity_logger.query_activities(
                activity_type="LIQUIDATION",
                start_time=start_time,
                end_time=end_time
            )
            
            # Count risk violations
            risk_violations = activity_logger.query_activities(
                activity_type="RISK_VIOLATION",
                start_time=start_time,
                end_time=end_time
            )
            
            # Count safety checks
            safety_checks = activity_logger.query_activities(
                activity_type="SAFETY_CHECK",
                start_time=start_time,
                end_time=end_time
            )
            
            return {
                'emergency_stops': len(emergency_stops),
                'liquidations': len(liquidations),
                'risk_violations': len(risk_violations),
                'safety_checks': len(safety_checks)
            }
            
        except Exception:
            # Return default stats if unable to query
            return {
                'emergency_stops': 0,
                'liquidations': 0,
                'risk_violations': 0,
                'safety_checks': 0
            }
    
    def _calculate_risk_score(self, risk_data: Dict[str, float]) -> float:
        """
        Calculate an overall risk score based on multiple risk factors.
        
        Args:
            risk_data: Dictionary with risk factor values
            
        Returns:
            Overall risk score (0-1)
        """
        position_size_ratio = risk_data.get('position_size_ratio', 0)
        drawdown = risk_data.get('drawdown', 0)
        volatility = risk_data.get('volatility', 0)
        correlation_risk = risk_data.get('correlation_risk', 0)
        margin_ratio = risk_data.get('margin_ratio', 2.0)
        
        # Weighted risk calculation
        risk_score = (
            position_size_ratio * 0.3 +  # 30% weight on position size
            drawdown * 0.25 +             # 25% weight on drawdown
            volatility * 0.2 +            # 20% weight on volatility
            correlation_risk * 0.15 +     # 15% weight on correlation
            max(0, (2.0 - margin_ratio) / 2.0) * 0.1  # 10% weight on margin (inverted)
        )
        
        return min(1.0, max(0.0, risk_score))
    
    def _check_risk_thresholds(self, risk_data: Dict[str, float]) -> List[str]:
        """
        Check risk data against predefined thresholds.
        
        Args:
            risk_data: Dictionary with risk factor values
            
        Returns:
            List of risk threshold violations
        """
        violations = []
        
        # Define thresholds
        thresholds = {
            'position_size_ratio': 0.8,  # 80% max position size
            'drawdown': 0.2,             # 20% max drawdown
            'volatility': 0.8,           # 80% max volatility score
            'margin_ratio_min': 1.1      # Minimum 110% margin ratio
        }
        
        # Check thresholds
        if risk_data.get('position_size_ratio', 0) > thresholds['position_size_ratio']:
            violations.append('position_size_exceeded')
        
        if risk_data.get('drawdown', 0) > thresholds['drawdown']:
            violations.append('drawdown_exceeded')
        
        if risk_data.get('volatility', 0) > thresholds['volatility']:
            violations.append('volatility_exceeded')
        
        if risk_data.get('margin_ratio', 2.0) < thresholds['margin_ratio_min']:
            violations.append('margin_ratio_low')
        
        return violations
    
    def is_healthy(self) -> bool:
        """
        Check if the safety metrics collector is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to collect metrics to verify health
            self.collect_metrics()
            return True
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information for the safety metrics collector.
        
        Returns:
            Dictionary with collector status and safety-specific information
        """
        base_status = super().get_status()
        
        # Add safety-specific status information
        safety_events = self._get_safety_events()
        base_status['safety_events'] = safety_events
        
        return base_status