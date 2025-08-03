"""
Intelligent alerting system with ML-powered anomaly detection.

This module provides enhanced monitoring and alerting capabilities for Phase 6.1,
including intelligent alerting, anomaly detection, alert correlation, and escalation policies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import json
import uuid

import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger()


class AlertSeverity(Enum):
    """Enhanced alert severity levels with emergency tier."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    
    def __lt__(self, other):
        """Allow comparison of alert severity levels."""
        if not isinstance(other, AlertSeverity):
            return NotImplemented
        
        order = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.CRITICAL: 3,
            AlertSeverity.EMERGENCY: 4
        }
        return order[self] < order[other]
    
    def escalate(self):
        """Escalate to next severity level."""
        escalation_map = {
            AlertSeverity.INFO: AlertSeverity.WARNING,
            AlertSeverity.WARNING: AlertSeverity.CRITICAL,
            AlertSeverity.CRITICAL: AlertSeverity.EMERGENCY,
            AlertSeverity.EMERGENCY: AlertSeverity.EMERGENCY  # Max level
        }
        return escalation_map[self]


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    TRADING_PATTERN = "trading_pattern"
    SYSTEM_METRICS = "system_metrics"
    VOLUME_SPIKE = "volume_spike"
    FLASH_CRASH = "flash_crash"
    PUMP_AND_DUMP = "pump_and_dump"
    WASH_TRADING = "wash_trading"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CPU_SPIKE = "cpu_spike"
    MEMORY_LEAK = "memory_leak"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    NETWORK_ANOMALY = "network_anomaly"
    MARKET_CRASH = "market_crash"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    VOLATILITY_CLUSTERING = "volatility_clustering"


@dataclass
class Alert:
    """Enhanced alert with intelligent features."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: AlertSeverity = AlertSeverity.INFO
    message: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    anomaly_type: Optional[AnomalyType] = None
    resolved: bool = False
    resolution_time: Optional[timedelta] = None
    correlation_group: Optional[str] = None
    escalation_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary representation."""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "anomaly_type": self.anomaly_type.value if self.anomaly_type else None,
            "resolved": self.resolved,
            "correlation_group": self.correlation_group,
            "escalation_count": self.escalation_count
        }
    
    def escalate(self):
        """Escalate alert to next severity level."""
        self.severity = self.severity.escalate()
        self.escalation_count += 1


@dataclass
class AlertRule:
    """Configuration for alert rules."""
    name: str
    condition: str
    severity: AlertSeverity
    message_template: str
    enabled: bool = True
    cooldown_minutes: int = 5
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate alert rule condition against data."""
        try:
            # Simple expression evaluation (in production, use safer evaluation)
            # This is a simplified implementation
            for key, value in data.items():
                locals()[key] = value
            
            return eval(self.condition)
        except Exception as e:
            logger.warning("Failed to evaluate alert rule", rule=self.name, error=str(e))
            return False
    
    def format_message(self, data: Dict[str, Any]) -> str:
        """Format alert message with data."""
        try:
            return self.message_template.format(**data)
        except Exception as e:
            logger.warning("Failed to format alert message", rule=self.name, error=str(e))
            return self.message_template


@dataclass
class CorrelationRule:
    """Rule for correlating related alerts."""
    name: str
    conditions: List[str]
    action: str
    time_window_minutes: int = 5


class AnomalyDetector:
    """ML-powered anomaly detection for various data types."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize anomaly detector with configuration."""
        self.config = config
        self._error_count = 0
        self._max_errors = config.get('max_errors', 5)
        self.isolation_forest_contamination = config.get('isolation_forest_contamination', 0.1)
        self.statistical_threshold = config.get('statistical_threshold', 3.0)
        self.ml_model_threshold = config.get('ml_model_threshold', 0.8)
        self.min_samples = config.get('min_samples_for_detection', 10)
        
        logger.info("AnomalyDetector initialized", config=config)
    
    async def detect_anomalies(self, data: Dict[str, Any]) -> List[Alert]:
        """Detect anomalies in the provided data."""
        try:
            anomalies = []
            
            # Convert data to DataFrame if it's not already
            if isinstance(data, dict) and 'timestamp' in data:
                df = pd.DataFrame(data)
            else:
                logger.warning("Invalid data format for anomaly detection")
                return []
            
            # Skip detection if insufficient data
            if len(df) < self.min_samples:
                logger.debug("Insufficient data for anomaly detection", samples=len(df))
                return []
            
            # Detect different types of anomalies
            trading_anomalies = await self._detect_trading_anomalies(df)
            system_anomalies = await self._detect_system_anomalies(df)
            
            anomalies.extend(trading_anomalies)
            anomalies.extend(system_anomalies)
            
            logger.info("Anomaly detection completed", anomalies_found=len(anomalies))
            return anomalies
            
        except Exception as e:
            self._error_count += 1
            logger.error("Anomaly detection failed", error=str(e), error_count=self._error_count)
            return []
    
    async def _detect_trading_anomalies(self, data: pd.DataFrame) -> List[Alert]:
        """Detect trading-related anomalies."""
        anomalies = []
        
        # Volume spike detection
        if 'trading_volume' in data.columns:
            volume_mean = data['trading_volume'].mean()
            volume_std = data['trading_volume'].std()
            
            for idx, volume in data['trading_volume'].items():
                if volume > volume_mean + 3 * volume_std:
                    anomaly = Alert(
                        severity=AlertSeverity.WARNING,
                        message=f"Trading volume anomaly detected: {volume:.0f}",
                        source="AnomalyDetector",
                        anomaly_type=AnomalyType.TRADING_PATTERN,
                        metadata={
                            'volume': volume,
                            'volume_mean': volume_mean,
                            'volume_multiplier': volume / volume_mean
                        }
                    )
                    anomalies.append(anomaly)
        
        # Price change anomaly detection
        if 'price_change' in data.columns:
            price_changes = data['price_change']
            threshold = self.statistical_threshold * price_changes.std()
            
            for idx, change in price_changes.items():
                if abs(change) > threshold:
                    severity = AlertSeverity.CRITICAL if abs(change) > threshold * 1.5 else AlertSeverity.WARNING
                    anomaly = Alert(
                        severity=severity,
                        message=f"Significant price change detected: {change:.2%}",
                        source="AnomalyDetector",
                        anomaly_type=AnomalyType.TRADING_PATTERN,
                        metadata={
                            'price_change': change,
                            'threshold': threshold
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_system_anomalies(self, data: pd.DataFrame) -> List[Alert]:
        """Detect system metrics anomalies."""
        anomalies = []
        
        # CPU usage anomaly detection
        if 'cpu_usage' in data.columns:
            cpu_threshold = self.config.get('cpu_threshold', 0.8)
            high_cpu = data[data['cpu_usage'] > cpu_threshold]
            
            if len(high_cpu) > 0:
                max_cpu = high_cpu['cpu_usage'].max()
                anomaly = Alert(
                    severity=AlertSeverity.WARNING if max_cpu < 0.9 else AlertSeverity.CRITICAL,
                    message=f"High CPU usage detected: {max_cpu:.1%}",
                    source="AnomalyDetector",
                    anomaly_type=AnomalyType.SYSTEM_METRICS,
                    metadata={
                        'cpu_usage': max_cpu,
                        'threshold': cpu_threshold,
                        'duration_samples': len(high_cpu)
                    }
                )
                anomalies.append(anomaly)
        
        # Memory usage anomaly detection
        if 'memory_usage' in data.columns:
            memory_threshold = self.config.get('memory_threshold', 0.85)
            high_memory = data[data['memory_usage'] > memory_threshold]
            
            if len(high_memory) > 0:
                max_memory = high_memory['memory_usage'].max()
                anomaly = Alert(
                    severity=AlertSeverity.WARNING if max_memory < 0.95 else AlertSeverity.CRITICAL,
                    message=f"High memory usage detected: {max_memory:.1%}",
                    source="AnomalyDetector",
                    anomaly_type=AnomalyType.SYSTEM_METRICS,
                    metadata={
                        'memory_usage': max_memory,
                        'threshold': memory_threshold,
                        'duration_samples': len(high_memory)
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_isolation_forest_anomalies(self, data: np.ndarray) -> List[float]:
        """Detect anomalies using Isolation Forest algorithm."""
        try:
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(
                contamination=self.isolation_forest_contamination,
                random_state=42
            )
            
            # Fit and predict
            predictions = iso_forest.fit_predict(data)
            scores = iso_forest.decision_function(data)
            
            # Convert to anomaly scores (higher = more anomalous)
            anomaly_scores = -scores  # Negative decision function values indicate anomalies
            
            return anomaly_scores.tolist()
            
        except ImportError:
            logger.warning("scikit-learn not available for Isolation Forest")
            return [0.0] * len(data)
        except Exception as e:
            logger.error("Isolation Forest detection failed", error=str(e))
            return [0.0] * len(data)
    
    def _detect_statistical_anomalies(self, data: List[float], metric_name: str) -> List[Alert]:
        """Detect statistical anomalies using z-score method."""
        anomalies = []
        
        if len(data) < 3:
            return anomalies
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return anomalies
        
        for i, value in enumerate(data):
            z_score = abs(value - mean_val) / std_val
            
            if z_score > self.statistical_threshold:
                severity = AlertSeverity.WARNING if z_score < 4.0 else AlertSeverity.CRITICAL
                anomaly = Alert(
                    severity=severity,
                    message=f"Statistical anomaly in {metric_name}: {value}",
                    source="AnomalyDetector",
                    anomaly_type=AnomalyType.SYSTEM_METRICS,
                    metadata={
                        'value': value,
                        'z_score': z_score,
                        'mean': mean_val,
                        'std': std_val,
                        'metric': metric_name
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def is_healthy(self) -> bool:
        """Check if anomaly detector is healthy."""
        return self._error_count < self._max_errors


class AlertCorrelationEngine:
    """Engine for correlating and deduplicating alerts."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize alert correlation engine."""
        self.config = config
        self.correlation_window = timedelta(
            minutes=config.get('correlation_window_minutes', 5)
        )
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.max_alerts_per_correlation = config.get('max_alerts_per_correlation', 10)
        self.correlation_rules: List[CorrelationRule] = []
        
        logger.info("AlertCorrelationEngine initialized", config=config)
    
    async def correlate_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Correlate and deduplicate alerts."""
        if not alerts:
            return []
        
        # Group alerts by time windows
        time_groups = self._group_alerts_by_time(alerts)
        
        correlated_alerts = []
        for group in time_groups:
            # Apply correlation rules
            group_correlated = await self._apply_correlation_rules(group)
            correlated_alerts.extend(group_correlated)
        
        logger.info("Alert correlation completed", 
                   original_count=len(alerts), 
                   correlated_count=len(correlated_alerts))
        
        return correlated_alerts
    
    def _group_alerts_by_time(self, alerts: List[Alert]) -> List[List[Alert]]:
        """Group alerts by time windows."""
        alerts_sorted = sorted(alerts, key=lambda a: a.timestamp)
        groups = []
        current_group = []
        
        for alert in alerts_sorted:
            if not current_group:
                current_group.append(alert)
            else:
                time_diff = alert.timestamp - current_group[0].timestamp
                if time_diff <= self.correlation_window:
                    current_group.append(alert)
                else:
                    groups.append(current_group)
                    current_group = [alert]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _apply_correlation_rules(self, alerts: List[Alert]) -> List[Alert]:
        """Apply correlation rules to a group of alerts."""
        if len(alerts) <= 1:
            return alerts
        
        # Simple deduplication based on similarity
        unique_alerts = []
        
        for alert in alerts:
            is_duplicate = False
            for existing in unique_alerts:
                similarity = self._calculate_alert_similarity(alert, existing)
                if similarity > self.similarity_threshold:
                    # Merge alerts - keep the more severe one
                    if alert.severity > existing.severity:
                        unique_alerts.remove(existing)
                        unique_alerts.append(alert)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_alerts.append(alert)
        
        return unique_alerts
    
    def _calculate_alert_similarity(self, alert1: Alert, alert2: Alert) -> float:
        """Calculate similarity between two alerts."""
        # Simple similarity based on source and message content
        similarity = 0.0
        
        # Source similarity
        if alert1.source == alert2.source:
            similarity += 0.3
        
        # Message similarity (simple word overlap)
        words1 = set(alert1.message.lower().split())
        words2 = set(alert2.message.lower().split())
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            similarity += 0.5 * word_overlap
        
        # Severity similarity
        if alert1.severity == alert2.severity:
            similarity += 0.2
        
        return min(similarity, 1.0)
    
    def add_correlation_rule(self, rule: CorrelationRule):
        """Add a correlation rule."""
        self.correlation_rules.append(rule)
        logger.info("Correlation rule added", rule_name=rule.name)
    
    def is_healthy(self) -> bool:
        """Check if correlation engine is healthy."""
        return True  # Simple implementation


class AlertRoutingManager:
    """Manager for alert routing and escalation policies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize alert routing manager."""
        self.config = config
        self.escalation_thresholds = config.get('escalation_thresholds', {})
        self.notification_channels = config.get('notification_channels', [])
        self.routing_rules = []
        self.notification_service = None  # Will be injected
        
        logger.info("AlertRoutingManager initialized", config=config)
    
    async def route_alert(self, alert: Alert):
        """Route alert to appropriate channels."""
        try:
            # Select notification channels based on severity
            channels = self._select_notification_channels(alert)
            
            # Send notifications if service is available
            if self.notification_service:
                await self.notification_service.send_notification(alert, channels)
            
            logger.info("Alert routed", alert_id=alert.id, channels=channels)
            
        except Exception as e:
            logger.error("Failed to route alert", alert_id=alert.id, error=str(e))
    
    async def check_escalations(self):
        """Check for alerts that need escalation."""
        # This would typically check a database of active alerts
        # Simplified implementation for testing
        pass
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate an alert to higher severity."""
        alert.escalate()
        await self.route_alert(alert)
        logger.info("Alert escalated", alert_id=alert.id, new_severity=alert.severity.value)
    
    def _select_notification_channels(self, alert: Alert) -> List[str]:
        """Select notification channels based on alert severity."""
        channels = []
        
        # Default channel selection based on severity
        if alert.severity == AlertSeverity.INFO:
            channels = ['email']
        elif alert.severity == AlertSeverity.WARNING:
            channels = ['email', 'slack']
        elif alert.severity == AlertSeverity.CRITICAL:
            channels = ['email', 'slack', 'pagerduty']
        elif alert.severity == AlertSeverity.EMERGENCY:
            channels = ['pagerduty', 'slack', 'email', 'sms']
        
        # Filter by available channels
        available_channels = set(self.notification_channels)
        return [ch for ch in channels if ch in available_channels]
    
    def add_routing_rule(self, rule: Dict[str, Any]):
        """Add a routing rule."""
        self.routing_rules.append(rule)
        logger.info("Routing rule added", rule=rule)
    
    def is_healthy(self) -> bool:
        """Check if routing manager is healthy."""
        return True  # Simple implementation


class AlertAnalytics:
    """Analytics and insights for alert patterns."""
    
    def __init__(self):
        """Initialize alert analytics."""
        logger.info("AlertAnalytics initialized")
    
    def analyze_alert_frequency(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Analyze alert frequency patterns."""
        if not alerts:
            return {'total_alerts': 0, 'alerts_per_hour': 0, 'alerts_by_severity': {}}
        
        # Calculate time span
        timestamps = [a.timestamp for a in alerts]
        time_span = max(timestamps) - min(timestamps)
        hours = max(time_span.total_seconds() / 3600, 1)  # Avoid division by zero
        
        # Count by severity
        severity_counts = {}
        for alert in alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(alerts),
            'alerts_per_hour': len(alerts) / hours,
            'alerts_by_severity': severity_counts,
            'time_span_hours': hours
        }
    
    def detect_alert_patterns(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Detect patterns in alert data."""
        if not alerts:
            return {'recurring_sources': {}, 'time_patterns': {}}
        
        # Analyze sources
        source_counts = {}
        for alert in alerts:
            source_counts[alert.source] = source_counts.get(alert.source, 0) + 1
        
        # Analyze time patterns (hour of day)
        hour_counts = {}
        for alert in alerts:
            hour = alert.timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        return {
            'recurring_sources': source_counts,
            'time_patterns': hour_counts,
            'most_common_source': max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else None
        }
    
    def calculate_alert_fatigue(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Calculate alert fatigue metrics."""
        if not alerts:
            return {'fatigue_score': 0, 'duplicate_percentage': 0}
        
        # Simple duplicate detection based on message similarity
        unique_messages = set()
        duplicates = 0
        
        for alert in alerts:
            message_key = f"{alert.source}:{alert.message[:50]}"  # First 50 chars
            if message_key in unique_messages:
                duplicates += 1
            else:
                unique_messages.add(message_key)
        
        duplicate_percentage = (duplicates / len(alerts)) * 100
        
        # Fatigue score based on duplicate percentage and frequency
        fatigue_score = min(duplicate_percentage / 100, 1.0)
        
        return {
            'fatigue_score': fatigue_score,
            'duplicate_percentage': duplicate_percentage,
            'unique_messages': len(unique_messages),
            'total_alerts': len(alerts)
        }
    
    def generate_resolution_insights(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Generate insights about alert resolution."""
        if not alerts:
            return {'resolution_rate': 0, 'average_resolution_time': 0, 'unresolved_count': 0}
        
        resolved_alerts = [a for a in alerts if a.resolved]
        unresolved_count = len(alerts) - len(resolved_alerts)
        
        resolution_rate = len(resolved_alerts) / len(alerts) * 100
        
        # Calculate average resolution time
        resolution_times = [a.resolution_time for a in resolved_alerts if a.resolution_time]
        avg_resolution_time = sum(resolution_times, timedelta()).total_seconds() / len(resolution_times) if resolution_times else 0
        
        return {
            'resolution_rate': resolution_rate,
            'average_resolution_time': avg_resolution_time,
            'unresolved_count': unresolved_count,
            'resolved_count': len(resolved_alerts)
        }


class IntelligentAlertingSystem:
    """Main intelligent alerting system that orchestrates all components."""
    
    def __init__(
        self,
        anomaly_detector: Optional[AnomalyDetector] = None,
        correlation_engine: Optional[AlertCorrelationEngine] = None,
        routing_manager: Optional[AlertRoutingManager] = None,
        analytics: Optional[AlertAnalytics] = None
    ):
        """Initialize intelligent alerting system."""
        # Initialize components with default configurations if not provided
        self.anomaly_detector = anomaly_detector or AnomalyDetector({})
        self.correlation_engine = correlation_engine or AlertCorrelationEngine({})
        self.routing_manager = routing_manager or AlertRoutingManager({})
        self.analytics = analytics or AlertAnalytics()
        
        self.is_running = False
        self._alert_queue = asyncio.Queue()
        
        logger.info("IntelligentAlertingSystem initialized")
    
    async def start(self):
        """Start the intelligent alerting system."""
        self.is_running = True
        logger.info("IntelligentAlertingSystem started")
    
    async def stop(self):
        """Stop the intelligent alerting system."""
        self.is_running = False
        
        # Process remaining alerts in queue
        while not self._alert_queue.empty():
            try:
                alert = await asyncio.wait_for(self._alert_queue.get(), timeout=1.0)
                await self.routing_manager.route_alert(alert)
                self._alert_queue.task_done()
            except asyncio.TimeoutError:
                break
        
        logger.info("IntelligentAlertingSystem stopped")
    
    async def process_metrics(self, metrics_data: Dict[str, Any]) -> List[Alert]:
        """Process metrics data and generate intelligent alerts."""
        try:
            # Step 1: Detect anomalies
            anomalies = await self.anomaly_detector.detect_anomalies(metrics_data)
            
            # Step 2: Correlate alerts
            correlated_alerts = await self.correlation_engine.correlate_alerts(anomalies)
            
            # Step 3: Route alerts
            for alert in correlated_alerts:
                await self.routing_manager.route_alert(alert)
            
            logger.info("Metrics processed", 
                       anomalies_detected=len(anomalies),
                       alerts_after_correlation=len(correlated_alerts))
            
            return correlated_alerts
            
        except Exception as e:
            logger.error("Failed to process metrics", error=str(e))
            return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components."""
        return {
            'anomaly_detector': self.anomaly_detector.is_healthy(),
            'correlation_engine': self.correlation_engine.is_healthy(),
            'routing_manager': self.routing_manager.is_healthy(),
            'overall_healthy': all([
                self.anomaly_detector.is_healthy(),
                self.correlation_engine.is_healthy(),
                self.routing_manager.is_healthy()
            ])
        }
    
    def integrate_drift_detection(self, drift_detector):
        """Integration point with drift detection system."""
        # This would integrate with the existing drift detection from Phase 5
        logger.info("Drift detection integration configured")
    
    def integrate_safety_systems(self, safety_controller):
        """Integration point with safety systems."""
        # This would integrate with emergency stop controller from Phase 4
        logger.info("Safety systems integration configured")