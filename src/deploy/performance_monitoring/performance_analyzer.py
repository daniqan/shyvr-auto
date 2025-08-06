"""
Performance Analyzer

Advanced performance analysis system providing trend analysis,
anomaly detection, and performance insights for production systems.
"""

import asyncio
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Performance trend directions"""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


class AnomalyType(Enum):
    """Types of performance anomalies"""
    SPIKE = "spike"
    DIP = "dip"
    TREND_CHANGE = "trend_change"
    VARIANCE_INCREASE = "variance_increase"
    PATTERN_BREAK = "pattern_break"


@dataclass
class PerformanceAnomaly:
    """Performance anomaly detection result"""
    type: AnomalyType
    metric_name: str
    timestamp: float
    severity: str
    confidence: float
    description: str
    affected_value: float
    expected_range: Tuple[float, float]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Performance trend analysis result"""
    metric_name: str
    direction: TrendDirection
    magnitude: float
    confidence: float
    time_range_seconds: int
    data_points: int
    trend_line: List[float] = field(default_factory=list)
    seasonal_pattern: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceInsight:
    """Performance insight and recommendation"""
    insight_id: str
    category: str
    title: str
    description: str
    severity: str
    metrics_involved: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: float


class PerformanceAnalyzer:
    """
    Advanced Performance Analyzer
    
    Provides comprehensive performance analysis including trend detection,
    anomaly identification, and actionable insights generation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.anomaly_models: Dict[str, Dict[str, Any]] = {}
        self.baseline_models: Dict[str, Dict[str, Any]] = {}
        
        # Analysis parameters
        self.trend_window_minutes = self.config.get("trend_window_minutes", 60)
        self.anomaly_sensitivity = self.config.get("anomaly_sensitivity", 2.0)  # Standard deviations
        self.min_data_points = self.config.get("min_data_points", 10)
        
        logger.info("Performance analyzer initialized")
    
    def analyze_performance_trends(self, metric_data: Dict[str, List[Dict[str, Any]]], 
                                  time_range_seconds: int = 3600) -> Dict[str, TrendAnalysis]:
        """Analyze performance trends for multiple metrics"""
        trend_results = {}
        
        for metric_name, data_points in metric_data.items():
            try:
                trend_analysis = self._analyze_single_metric_trend(
                    metric_name, data_points, time_range_seconds
                )
                trend_results[metric_name] = trend_analysis
                
            except Exception as e:
                logger.error(f"Error analyzing trend for {metric_name}: {e}")
                trend_results[metric_name] = TrendAnalysis(
                    metric_name=metric_name,
                    direction=TrendDirection.INSUFFICIENT_DATA,
                    magnitude=0.0,
                    confidence=0.0,
                    time_range_seconds=time_range_seconds,
                    data_points=0
                )
        
        return trend_results
    
    def _analyze_single_metric_trend(self, metric_name: str, data_points: List[Dict[str, Any]], 
                                   time_range_seconds: int) -> TrendAnalysis:
        """Analyze trend for a single metric"""
        if len(data_points) < self.min_data_points:
            return TrendAnalysis(
                metric_name=metric_name,
                direction=TrendDirection.INSUFFICIENT_DATA,
                magnitude=0.0,
                confidence=0.0,
                time_range_seconds=time_range_seconds,
                data_points=len(data_points)
            )
        
        # Extract values and timestamps
        timestamps = [dp.get("timestamp", 0) for dp in data_points]
        values = [dp.get("value", 0) for dp in data_points]
        
        # Sort by timestamp
        sorted_pairs = sorted(zip(timestamps, values))
        timestamps, values = zip(*sorted_pairs)
        
        # Calculate trend using linear regression
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        # Calculate slope (trend direction and magnitude)
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Calculate correlation coefficient for confidence
        std_x = statistics.stdev(x) if n > 1 else 0
        std_y = statistics.stdev(values) if n > 1 else 0
        
        if std_x == 0 or std_y == 0:
            correlation = 0
        else:
            correlation = numerator / (n * std_x * std_y)
        
        confidence = abs(correlation)
        
        # Determine trend direction
        normalized_slope = slope / (y_mean if y_mean != 0 else 1)
        
        if abs(normalized_slope) < 0.01:  # Less than 1% change
            direction = TrendDirection.STABLE
        elif normalized_slope > 0.05:  # More than 5% increase
            direction = TrendDirection.IMPROVING if self._is_improving_metric(metric_name) else TrendDirection.DEGRADING
        elif normalized_slope < -0.05:  # More than 5% decrease
            direction = TrendDirection.DEGRADING if self._is_improving_metric(metric_name) else TrendDirection.IMPROVING
        else:
            # Check for volatility
            if len(values) > 5:
                recent_std = statistics.stdev(values[-5:])
                overall_std = statistics.stdev(values)
                if recent_std > overall_std * 1.5:
                    direction = TrendDirection.VOLATILE
                else:
                    direction = TrendDirection.STABLE
            else:
                direction = TrendDirection.STABLE
        
        # Generate trend line
        trend_line = [y_mean + slope * (i - x_mean) for i in range(n)]
        
        # Detect seasonal patterns (simplified)
        seasonal_pattern = self._detect_seasonal_pattern(timestamps, values)
        
        return TrendAnalysis(
            metric_name=metric_name,
            direction=direction,
            magnitude=abs(normalized_slope),
            confidence=confidence,
            time_range_seconds=time_range_seconds,
            data_points=n,
            trend_line=trend_line,
            seasonal_pattern=seasonal_pattern
        )
    
    def _is_improving_metric(self, metric_name: str) -> bool:
        """Determine if higher values are better for this metric"""
        # Metrics where higher values are better
        improving_metrics = [
            "throughput", "rps", "success_rate", "uptime", 
            "availability", "cache_hit_rate", "efficiency"
        ]
        
        # Metrics where lower values are better  
        degrading_metrics = [
            "response_time", "latency", "error_rate", "cpu_utilization",
            "memory_utilization", "disk_usage", "queue_length"
        ]
        
        metric_lower = metric_name.lower()
        
        for improving in improving_metrics:
            if improving in metric_lower:
                return True
        
        for degrading in degrading_metrics:
            if degrading in metric_lower:
                return False
        
        # Default: assume higher is better
        return True
    
    def _detect_seasonal_pattern(self, timestamps: List[float], values: List[float]) -> Optional[Dict[str, Any]]:
        """Detect seasonal patterns in metric data (simplified implementation)"""
        if len(values) < 24:  # Need at least 24 data points
            return None
        
        try:
            # Look for hourly patterns (simplified)
            hours = [(datetime.fromtimestamp(ts).hour) for ts in timestamps]
            hourly_averages = defaultdict(list)
            
            for hour, value in zip(hours, values):
                hourly_averages[hour].append(value)
            
            # Calculate average for each hour
            hour_stats = {}
            for hour in range(24):
                if hour in hourly_averages:
                    hour_values = hourly_averages[hour]
                    hour_stats[hour] = {
                        "avg": statistics.mean(hour_values),
                        "count": len(hour_values)
                    }
            
            # Check if there's significant variation between hours
            if len(hour_stats) >= 12:  # At least 12 hours of data
                hour_averages = [stats["avg"] for stats in hour_stats.values()]
                hour_std = statistics.stdev(hour_averages) if len(hour_averages) > 1 else 0
                overall_avg = statistics.mean(hour_averages)
                
                # If standard deviation is more than 20% of mean, consider it seasonal
                if hour_std > overall_avg * 0.2:
                    peak_hour = max(hour_stats.items(), key=lambda x: x[1]["avg"])[0]
                    low_hour = min(hour_stats.items(), key=lambda x: x[1]["avg"])[0]
                    
                    return {
                        "pattern_type": "hourly",
                        "peak_hour": peak_hour,
                        "low_hour": low_hour,
                        "variation_coefficient": hour_std / overall_avg,
                        "confidence": min(1.0, len(hour_stats) / 24)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting seasonal pattern: {e}")
            return None
    
    def detect_anomalies(self, metric_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[PerformanceAnomaly]]:
        """Detect anomalies in performance metrics"""
        anomaly_results = {}
        
        for metric_name, data_points in metric_data.items():
            try:
                anomalies = self._detect_metric_anomalies(metric_name, data_points)
                anomaly_results[metric_name] = anomalies
                
            except Exception as e:
                logger.error(f"Error detecting anomalies for {metric_name}: {e}")
                anomaly_results[metric_name] = []
        
        return anomaly_results
    
    def _detect_metric_anomalies(self, metric_name: str, data_points: List[Dict[str, Any]]) -> List[PerformanceAnomaly]:
        """Detect anomalies for a single metric"""
        if len(data_points) < self.min_data_points:
            return []
        
        anomalies = []
        values = [dp.get("value", 0) for dp in data_points]
        timestamps = [dp.get("timestamp", 0) for dp in data_points]
        
        # Update baseline model
        self._update_baseline_model(metric_name, values)
        
        # Get baseline statistics
        baseline = self.baseline_models.get(metric_name, {})
        if not baseline:
            return []
        
        mean = baseline.get("mean", 0)
        std_dev = baseline.get("std_dev", 0)
        
        if std_dev == 0:
            return []
        
        # Detect different types of anomalies
        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            
            # Z-score based anomaly detection
            z_score = abs(value - mean) / std_dev
            
            if z_score > self.anomaly_sensitivity:
                anomaly_type = AnomalyType.SPIKE if value > mean else AnomalyType.DIP
                severity = self._calculate_anomaly_severity(z_score)
                confidence = min(1.0, z_score / (self.anomaly_sensitivity * 2))
                
                expected_range = (
                    mean - self.anomaly_sensitivity * std_dev,
                    mean + self.anomaly_sensitivity * std_dev
                )
                
                anomaly = PerformanceAnomaly(
                    type=anomaly_type,
                    metric_name=metric_name,
                    timestamp=timestamp,
                    severity=severity,
                    confidence=confidence,
                    description=f"{metric_name} anomaly: value {value:.2f} deviates from expected range",
                    affected_value=value,
                    expected_range=expected_range,
                    context={
                        "z_score": z_score,
                        "baseline_mean": mean,
                        "baseline_std": std_dev
                    }
                )
                
                anomalies.append(anomaly)
        
        # Detect trend changes
        trend_anomalies = self._detect_trend_changes(metric_name, values, timestamps)
        anomalies.extend(trend_anomalies)
        
        # Detect variance changes
        variance_anomalies = self._detect_variance_changes(metric_name, values, timestamps)
        anomalies.extend(variance_anomalies)
        
        return anomalies
    
    def _update_baseline_model(self, metric_name: str, values: List[float]):
        """Update baseline statistical model for a metric"""
        if metric_name not in self.baseline_models:
            self.baseline_models[metric_name] = {}
        
        model = self.baseline_models[metric_name]
        
        # Update statistics
        model["mean"] = statistics.mean(values)
        model["std_dev"] = statistics.stdev(values) if len(values) > 1 else 0
        model["min"] = min(values)
        model["max"] = max(values)
        model["median"] = statistics.median(values)
        model["last_updated"] = time.time()
        
        # Keep historical statistics for trend detection
        if "historical_means" not in model:
            model["historical_means"] = []
        
        model["historical_means"].append(model["mean"])
        
        # Keep only recent history
        if len(model["historical_means"]) > 100:
            model["historical_means"] = model["historical_means"][-100:]
    
    def _calculate_anomaly_severity(self, z_score: float) -> str:
        """Calculate anomaly severity based on z-score"""
        if z_score > 4.0:
            return "critical"
        elif z_score > 3.0:
            return "high"
        elif z_score > 2.5:
            return "medium"
        else:
            return "low"
    
    def _detect_trend_changes(self, metric_name: str, values: List[float], 
                            timestamps: List[float]) -> List[PerformanceAnomaly]:
        """Detect sudden trend changes"""
        anomalies = []
        
        if len(values) < 20:  # Need enough data for trend analysis
            return anomalies
        
        # Split data into windows and compare trends
        window_size = max(10, len(values) // 4)
        
        for i in range(window_size, len(values) - window_size):
            before_window = values[i-window_size:i]
            after_window = values[i:i+window_size]
            
            # Calculate trends for both windows
            before_trend = self._calculate_simple_trend(before_window)
            after_trend = self._calculate_simple_trend(after_window)
            
            # Detect significant trend change
            trend_change = abs(after_trend - before_trend)
            
            # Threshold based on data variability
            data_std = statistics.stdev(values)
            change_threshold = data_std * 0.5
            
            if trend_change > change_threshold:
                anomaly = PerformanceAnomaly(
                    type=AnomalyType.TREND_CHANGE,
                    metric_name=metric_name,
                    timestamp=timestamps[i],
                    severity="medium",
                    confidence=min(1.0, trend_change / change_threshold),
                    description=f"Significant trend change detected in {metric_name}",
                    affected_value=values[i],
                    expected_range=(values[i] - data_std, values[i] + data_std),
                    context={
                        "before_trend": before_trend,
                        "after_trend": after_trend,
                        "trend_change": trend_change
                    }
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_simple_trend(self, values: List[float]) -> float:
        """Calculate simple trend (slope) for a window of values"""
        if len(values) < 2:
            return 0
        
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0
    
    def _detect_variance_changes(self, metric_name: str, values: List[float], 
                               timestamps: List[float]) -> List[PerformanceAnomaly]:
        """Detect changes in variance/volatility"""
        anomalies = []
        
        if len(values) < 20:
            return anomalies
        
        # Calculate rolling variance
        window_size = max(10, len(values) // 5)
        
        for i in range(window_size, len(values) - window_size):
            before_window = values[i-window_size:i]
            current_window = values[i:i+window_size]
            
            before_variance = statistics.variance(before_window) if len(before_window) > 1 else 0
            current_variance = statistics.variance(current_window) if len(current_window) > 1 else 0
            
            # Detect significant variance increase
            if before_variance > 0 and current_variance > before_variance * 2:
                anomaly = PerformanceAnomaly(
                    type=AnomalyType.VARIANCE_INCREASE,
                    metric_name=metric_name,
                    timestamp=timestamps[i],
                    severity="medium",
                    confidence=min(1.0, current_variance / (before_variance * 2)),
                    description=f"Increased volatility detected in {metric_name}",
                    affected_value=values[i],
                    expected_range=(values[i] - statistics.stdev(before_window), 
                                  values[i] + statistics.stdev(before_window)),
                    context={
                        "before_variance": before_variance,
                        "current_variance": current_variance,
                        "variance_ratio": current_variance / before_variance if before_variance > 0 else 0
                    }
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def generate_performance_insights(self, trend_results: Dict[str, TrendAnalysis], 
                                    anomaly_results: Dict[str, List[PerformanceAnomaly]]) -> List[PerformanceInsight]:
        """Generate actionable performance insights"""
        insights = []
        insight_counter = 0
        
        # Analyze trends for insights
        for metric_name, trend in trend_results.items():
            if trend.confidence > 0.7:  # High confidence trends
                if trend.direction == TrendDirection.DEGRADING:
                    insight = self._generate_degradation_insight(metric_name, trend)
                    if insight:
                        insight.insight_id = f"insight_{insight_counter}"
                        insights.append(insight)
                        insight_counter += 1
                
                elif trend.direction == TrendDirection.VOLATILE:
                    insight = self._generate_volatility_insight(metric_name, trend)
                    if insight:
                        insight.insight_id = f"insight_{insight_counter}"
                        insights.append(insight)
                        insight_counter += 1
        
        # Analyze anomalies for insights
        for metric_name, anomalies in anomaly_results.items():
            if anomalies:
                # Group anomalies by type
                anomaly_groups = defaultdict(list)
                for anomaly in anomalies:
                    anomaly_groups[anomaly.type].append(anomaly)
                
                for anomaly_type, group_anomalies in anomaly_groups.items():
                    if len(group_anomalies) >= 2:  # Multiple anomalies of same type
                        insight = self._generate_anomaly_pattern_insight(metric_name, anomaly_type, group_anomalies)
                        if insight:
                            insight.insight_id = f"insight_{insight_counter}"
                            insights.append(insight)
                            insight_counter += 1
        
        # Generate cross-metric insights
        cross_metric_insights = self._generate_cross_metric_insights(trend_results, anomaly_results)
        for insight in cross_metric_insights:
            insight.insight_id = f"insight_{insight_counter}"
            insights.append(insight)
            insight_counter += 1
        
        return insights
    
    def _generate_degradation_insight(self, metric_name: str, trend: TrendAnalysis) -> Optional[PerformanceInsight]:
        """Generate insight for degrading performance trends"""
        recommendations = []
        
        if "response_time" in metric_name.lower() or "latency" in metric_name.lower():
            recommendations.extend([
                "Review recent code deployments for performance regressions",
                "Check database query performance and indexing",
                "Monitor CPU and memory usage for resource constraints",
                "Consider implementing caching strategies"
            ])
            category = "latency_degradation"
            
        elif "throughput" in metric_name.lower() or "rps" in metric_name.lower():
            recommendations.extend([
                "Investigate potential bottlenecks in request processing",
                "Review auto-scaling configuration and thresholds",
                "Check for database connection pool exhaustion",
                "Monitor downstream service dependencies"
            ])
            category = "throughput_degradation"
            
        elif "error_rate" in metric_name.lower():
            recommendations.extend([
                "Analyze error logs for common failure patterns",
                "Review recent deployments and configuration changes",
                "Check service dependencies and external API status",
                "Implement circuit breakers for failing services"
            ])
            category = "reliability_degradation"
            
        else:
            recommendations.extend([
                "Investigate recent system changes",
                "Review resource utilization and scaling policies",
                "Check for external dependencies impact"
            ])
            category = "general_degradation"
        
        severity = "high" if trend.magnitude > 0.2 else "medium"
        
        return PerformanceInsight(
            insight_id="",  # Will be set by caller
            category=category,
            title=f"Performance Degradation Detected in {metric_name}",
            description=f"A degrading trend has been detected in {metric_name} with {trend.magnitude:.1%} decline over {trend.time_range_seconds//60} minutes. Confidence: {trend.confidence:.1%}",
            severity=severity,
            metrics_involved=[metric_name],
            recommendations=recommendations,
            confidence=trend.confidence,
            timestamp=time.time()
        )
    
    def _generate_volatility_insight(self, metric_name: str, trend: TrendAnalysis) -> Optional[PerformanceInsight]:
        """Generate insight for volatile performance metrics"""
        recommendations = [
            "Investigate sources of performance variability",
            "Review load balancing and traffic distribution",
            "Check for resource competition or contention",
            "Consider implementing performance smoothing mechanisms"
        ]
        
        return PerformanceInsight(
            insight_id="",  # Will be set by caller
            category="performance_volatility",
            title=f"High Volatility Detected in {metric_name}",
            description=f"High performance volatility detected in {metric_name}. This may indicate system instability or irregular load patterns.",
            severity="medium",
            metrics_involved=[metric_name],
            recommendations=recommendations,
            confidence=trend.confidence,
            timestamp=time.time()
        )
    
    def _generate_anomaly_pattern_insight(self, metric_name: str, anomaly_type: AnomalyType, 
                                        anomalies: List[PerformanceAnomaly]) -> Optional[PerformanceInsight]:
        """Generate insight for patterns in anomalies"""
        if anomaly_type == AnomalyType.SPIKE:
            title = f"Recurring Performance Spikes in {metric_name}"
            description = f"Multiple performance spikes detected in {metric_name}. This may indicate capacity issues or irregular load patterns."
            recommendations = [
                "Review capacity planning and auto-scaling thresholds",
                "Analyze traffic patterns for unexpected load spikes",
                "Implement rate limiting or load shedding mechanisms"
            ]
        
        elif anomaly_type == AnomalyType.DIP:
            title = f"Performance Drops Detected in {metric_name}"
            description = f"Multiple performance drops detected in {metric_name}. This may indicate service disruptions."
            recommendations = [
                "Check for service health and availability issues",
                "Review dependency status and connectivity",
                "Implement better error handling and retry mechanisms"
            ]
        
        else:
            return None
        
        avg_confidence = statistics.mean([a.confidence for a in anomalies])
        severity = "high" if len(anomalies) > 5 else "medium"
        
        return PerformanceInsight(
            insight_id="",  # Will be set by caller
            category="anomaly_pattern",
            title=title,
            description=description,
            severity=severity,
            metrics_involved=[metric_name],
            recommendations=recommendations,
            confidence=avg_confidence,
            timestamp=time.time()
        )
    
    def _generate_cross_metric_insights(self, trend_results: Dict[str, TrendAnalysis], 
                                      anomaly_results: Dict[str, List[PerformanceAnomaly]]) -> List[PerformanceInsight]:
        """Generate insights from cross-metric analysis"""
        insights = []
        
        # Look for correlated degradations
        degrading_metrics = [
            name for name, trend in trend_results.items()
            if trend.direction == TrendDirection.DEGRADING and trend.confidence > 0.6
        ]
        
        if len(degrading_metrics) >= 2:
            insight = PerformanceInsight(
                insight_id="",  # Will be set by caller
                category="system_wide_degradation",
                title="System-wide Performance Degradation",
                description=f"Multiple metrics showing degrading performance: {', '.join(degrading_metrics)}. This may indicate a system-wide issue.",
                severity="high",
                metrics_involved=degrading_metrics,
                recommendations=[
                    "Investigate recent system-wide changes or deployments",
                    "Check infrastructure health and capacity",
                    "Review resource utilization across all services",
                    "Consider rolling back recent changes if correlation is strong"
                ],
                confidence=statistics.mean([trend_results[m].confidence for m in degrading_metrics]),
                timestamp=time.time()
            )
            insights.append(insight)
        
        return insights
    
    def get_analysis_summary(self, time_range_seconds: int = 3600) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        cutoff_time = time.time() - time_range_seconds
        recent_analyses = [
            analysis for analysis in self.analysis_history
            if analysis.get("timestamp", 0) >= cutoff_time
        ]
        
        return {
            "time_range_seconds": time_range_seconds,
            "total_analyses": len(recent_analyses),
            "metrics_analyzed": len(self.baseline_models),
            "active_baselines": len([m for m in self.baseline_models.values() 
                                   if m.get("last_updated", 0) >= cutoff_time]),
            "analysis_capabilities": [
                "trend_analysis",
                "anomaly_detection", 
                "seasonal_pattern_detection",
                "performance_insights_generation",
                "cross_metric_correlation"
            ]
        }