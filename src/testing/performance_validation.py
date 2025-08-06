"""
Performance Validation Module

Comprehensive performance validation components for benchmarking,
SLA compliance testing, and performance regression detection.
Supports 99.9% uptime validation requirements.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceGrade(Enum):
    """Performance benchmark grades"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PerformanceBenchmark:
    """Performance benchmark definition"""
    name: str
    metric_type: str
    excellent_threshold: float
    good_threshold: float
    acceptable_threshold: float
    unit: str
    higher_is_better: bool = False
    
    def evaluate(self, value: float) -> PerformanceGrade:
        """Evaluate a value against this benchmark"""
        if self.higher_is_better:
            if value >= self.excellent_threshold:
                return PerformanceGrade.EXCELLENT
            elif value >= self.good_threshold:
                return PerformanceGrade.GOOD
            elif value >= self.acceptable_threshold:
                return PerformanceGrade.ACCEPTABLE
            else:
                return PerformanceGrade.POOR
        else:
            if value <= self.excellent_threshold:
                return PerformanceGrade.EXCELLENT
            elif value <= self.good_threshold:
                return PerformanceGrade.GOOD
            elif value <= self.acceptable_threshold:
                return PerformanceGrade.ACCEPTABLE
            else:
                return PerformanceGrade.POOR


@dataclass
class ValidationResult:
    """Performance validation result"""
    valid: bool
    score: float
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    metric_name: str
    threshold: float
    actual_value: float
    timestamp: float
    description: str
    resolution_suggestion: Optional[str] = None


class LatencyValidator:
    """
    Latency Performance Validator
    
    Validates response time performance against SLA requirements
    and industry benchmarks.
    """
    
    def __init__(self, sla_requirements: Dict[str, Any]):
        self.sla_requirements = sla_requirements
        self.benchmarks = self._initialize_latency_benchmarks()
    
    def _initialize_latency_benchmarks(self) -> List[PerformanceBenchmark]:
        """Initialize latency benchmarks"""
        return [
            PerformanceBenchmark(
                name="average_response_time",
                metric_type="latency",
                excellent_threshold=50.0,
                good_threshold=100.0,
                acceptable_threshold=200.0,
                unit="ms",
                higher_is_better=False
            ),
            PerformanceBenchmark(
                name="p95_response_time",
                metric_type="latency",
                excellent_threshold=100.0,
                good_threshold=200.0,
                acceptable_threshold=500.0,
                unit="ms",
                higher_is_better=False
            ),
            PerformanceBenchmark(
                name="p99_response_time",
                metric_type="latency",
                excellent_threshold=200.0,
                good_threshold=500.0,
                acceptable_threshold=1000.0,
                unit="ms",
                higher_is_better=False
            )
        ]
    
    def validate_latency_metrics(self, latency_data: Dict[str, Any]) -> ValidationResult:
        """Validate latency metrics against SLA and benchmarks"""
        violations = []
        warnings = []
        score = 1.0
        
        # Check SLA compliance
        max_response_time = self.sla_requirements.get("max_response_time_ms", 100)
        
        if "avg" in latency_data:
            avg_latency = latency_data["avg"]
            if avg_latency > max_response_time:
                violations.append({
                    "type": "sla_violation",
                    "metric": "average_response_time",
                    "expected": max_response_time,
                    "actual": avg_latency,
                    "severity": "high"
                })
                score *= 0.7
        
        # Check percentiles
        if "p95" in latency_data:
            p95_latency = latency_data["p95"]
            p95_limit = max_response_time * 2  # P95 can be 2x average limit
            if p95_latency > p95_limit:
                violations.append({
                    "type": "sla_violation",
                    "metric": "p95_response_time",
                    "expected": p95_limit,
                    "actual": p95_latency,
                    "severity": "medium"
                })
                score *= 0.85
        
        if "p99" in latency_data:
            p99_latency = latency_data["p99"]
            p99_limit = max_response_time * 3  # P99 can be 3x average limit
            if p99_latency > p99_limit:
                warnings.append({
                    "type": "performance_warning",
                    "metric": "p99_response_time",
                    "expected": p99_limit,
                    "actual": p99_latency,
                    "severity": "low"
                })
                score *= 0.95
        
        # Evaluate against benchmarks
        benchmark_scores = []
        for benchmark in self.benchmarks:
            if benchmark.name == "average_response_time" and "avg" in latency_data:
                grade = benchmark.evaluate(latency_data["avg"])
                benchmark_scores.append(self._grade_to_score(grade))
            elif benchmark.name == "p95_response_time" and "p95" in latency_data:
                grade = benchmark.evaluate(latency_data["p95"])
                benchmark_scores.append(self._grade_to_score(grade))
            elif benchmark.name == "p99_response_time" and "p99" in latency_data:
                grade = benchmark.evaluate(latency_data["p99"])
                benchmark_scores.append(self._grade_to_score(grade))
        
        # Combine SLA and benchmark scores
        if benchmark_scores:
            benchmark_score = statistics.mean(benchmark_scores)
            score = (score + benchmark_score) / 2
        
        return ValidationResult(
            valid=len(violations) == 0,
            score=score,
            violations=violations,
            warnings=warnings,
            metrics={"latency_score": score, "benchmark_grades": benchmark_scores}
        )
    
    def _grade_to_score(self, grade: PerformanceGrade) -> float:
        """Convert performance grade to numerical score"""
        grade_scores = {
            PerformanceGrade.EXCELLENT: 1.0,
            PerformanceGrade.GOOD: 0.8,
            PerformanceGrade.ACCEPTABLE: 0.6,
            PerformanceGrade.POOR: 0.3
        }
        return grade_scores.get(grade, 0.0)


class ThroughputValidator:
    """
    Throughput Performance Validator
    
    Validates throughput performance against capacity requirements
    and scaling expectations.
    """
    
    def __init__(self, capacity_requirements: Dict[str, Any]):
        self.capacity_requirements = capacity_requirements
        self.benchmarks = self._initialize_throughput_benchmarks()
    
    def _initialize_throughput_benchmarks(self) -> List[PerformanceBenchmark]:
        """Initialize throughput benchmarks"""
        return [
            PerformanceBenchmark(
                name="requests_per_second",
                metric_type="throughput",
                excellent_threshold=1000.0,
                good_threshold=500.0,
                acceptable_threshold=100.0,
                unit="rps",
                higher_is_better=True
            ),
            PerformanceBenchmark(
                name="concurrent_users",
                metric_type="capacity",
                excellent_threshold=2000.0,
                good_threshold=1000.0,
                acceptable_threshold=500.0,
                unit="users",
                higher_is_better=True
            )
        ]
    
    def validate_throughput_metrics(self, throughput_data: Dict[str, Any]) -> ValidationResult:
        """Validate throughput metrics"""
        violations = []
        warnings = []
        score = 1.0
        
        # Check minimum throughput requirement
        min_throughput = self.capacity_requirements.get("min_throughput_rps", 100)
        
        if "rps" in throughput_data:
            actual_throughput = throughput_data["rps"]
            if actual_throughput < min_throughput:
                violations.append({
                    "type": "capacity_violation",
                    "metric": "throughput_rps",
                    "expected": min_throughput,
                    "actual": actual_throughput,
                    "severity": "high"
                })
                score *= 0.6
        
        # Check capacity scaling
        max_users = self.capacity_requirements.get("max_concurrent_users", 1000)
        if "concurrent_users" in throughput_data:
            actual_users = throughput_data["concurrent_users"]
            if actual_users < max_users * 0.8:  # Should handle at least 80% of max
                warnings.append({
                    "type": "scaling_warning",
                    "metric": "concurrent_users",
                    "expected": max_users,
                    "actual": actual_users,
                    "severity": "medium"
                })
                score *= 0.9
        
        return ValidationResult(
            valid=len(violations) == 0,
            score=score,
            violations=violations,
            warnings=warnings,
            metrics={"throughput_score": score}
        )


class ResourceUsageValidator:
    """
    Resource Usage Validator
    
    Validates CPU, memory, and network resource utilization
    against efficiency and sustainability targets.
    """
    
    def __init__(self, resource_limits: Dict[str, Any]):
        self.resource_limits = resource_limits
    
    def validate_resource_usage(self, resource_data: Dict[str, Any]) -> ValidationResult:
        """Validate resource usage metrics"""
        violations = []
        warnings = []
        score = 1.0
        
        # CPU utilization validation
        max_cpu = self.resource_limits.get("max_cpu_utilization", 0.80)
        if "cpu" in resource_data:
            cpu_util = resource_data["cpu"] / 100.0 if resource_data["cpu"] > 1 else resource_data["cpu"]
            if cpu_util > max_cpu:
                violations.append({
                    "type": "resource_violation",
                    "metric": "cpu_utilization",
                    "expected": max_cpu,
                    "actual": cpu_util,
                    "severity": "high"
                })
                score *= 0.7
            elif cpu_util > max_cpu * 0.8:  # Warning at 80% of limit
                warnings.append({
                    "type": "resource_warning",
                    "metric": "cpu_utilization",
                    "threshold": max_cpu * 0.8,
                    "actual": cpu_util,
                    "severity": "medium"
                })
                score *= 0.9
        
        # Memory utilization validation
        max_memory = self.resource_limits.get("max_memory_utilization", 0.85)
        if "memory" in resource_data:
            memory_util = resource_data["memory"] / 100.0 if resource_data["memory"] > 1 else resource_data["memory"]
            if memory_util > max_memory:
                violations.append({
                    "type": "resource_violation",
                    "metric": "memory_utilization",
                    "expected": max_memory,
                    "actual": memory_util,
                    "severity": "high"
                })
                score *= 0.7
            elif memory_util > max_memory * 0.8:  # Warning at 80% of limit
                warnings.append({
                    "type": "resource_warning",
                    "metric": "memory_utilization",
                    "threshold": max_memory * 0.8,
                    "actual": memory_util,
                    "severity": "medium"
                })
                score *= 0.9
        
        # Network utilization validation
        max_network = self.resource_limits.get("max_network_utilization", 0.75)
        if "network" in resource_data:
            network_util = resource_data["network"] / 100.0 if resource_data["network"] > 1 else resource_data["network"]
            if network_util > max_network:
                warnings.append({
                    "type": "resource_warning",
                    "metric": "network_utilization",
                    "expected": max_network,
                    "actual": network_util,
                    "severity": "medium"
                })
                score *= 0.95
        
        return ValidationResult(
            valid=len(violations) == 0,
            score=score,
            violations=violations,
            warnings=warnings,
            metrics={"resource_efficiency_score": score}
        )


class ResponseTimeValidator:
    """
    Response Time Validator
    
    Specialized validator for response time patterns,
    distribution analysis, and anomaly detection.
    """
    
    def __init__(self, response_time_targets: Dict[str, Any]):
        self.response_time_targets = response_time_targets
        self.historical_data = []
    
    def validate_response_time_distribution(self, response_times: List[float]) -> ValidationResult:
        """Validate response time distribution characteristics"""
        if not response_times:
            return ValidationResult(valid=False, score=0.0, violations=[{"type": "data_error", "message": "No response time data"}])
        
        violations = []
        warnings = []
        score = 1.0
        
        # Calculate distribution statistics
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        p95_time = np.percentile(response_times, 95)
        p99_time = np.percentile(response_times, 99)
        
        # Check targets
        target_avg = self.response_time_targets.get("max_avg_ms", 100)
        target_p95 = self.response_time_targets.get("max_p95_ms", 200)
        target_p99 = self.response_time_targets.get("max_p99_ms", 500)
        
        if avg_time > target_avg:
            violations.append({
                "type": "response_time_violation",
                "metric": "average_response_time",
                "expected": target_avg,
                "actual": avg_time,
                "severity": "high"
            })
            score *= 0.7
        
        if p95_time > target_p95:
            violations.append({
                "type": "response_time_violation", 
                "metric": "p95_response_time",
                "expected": target_p95,
                "actual": p95_time,
                "severity": "medium"
            })
            score *= 0.85
        
        if p99_time > target_p99:
            warnings.append({
                "type": "response_time_warning",
                "metric": "p99_response_time",
                "expected": target_p99,
                "actual": p99_time,
                "severity": "low"
            })
            score *= 0.95
        
        # Check distribution health
        if std_dev > avg_time:  # High variability
            warnings.append({
                "type": "distribution_warning",
                "metric": "response_time_variability",
                "message": "High response time variability detected",
                "std_dev": std_dev,
                "avg": avg_time,
                "severity": "medium"
            })
            score *= 0.9
        
        # Check for outliers
        outlier_threshold = avg_time + (3 * std_dev)
        outliers = [t for t in response_times if t > outlier_threshold]
        outlier_rate = len(outliers) / len(response_times)
        
        if outlier_rate > 0.05:  # More than 5% outliers
            warnings.append({
                "type": "outlier_warning",
                "metric": "response_time_outliers",
                "outlier_rate": outlier_rate,
                "outlier_count": len(outliers),
                "severity": "medium"
            })
            score *= 0.9
        
        metrics = {
            "avg_response_time": avg_time,
            "median_response_time": median_time,
            "std_dev": std_dev,
            "p95_response_time": p95_time,
            "p99_response_time": p99_time,
            "outlier_rate": outlier_rate,
            "distribution_score": score
        }
        
        return ValidationResult(
            valid=len(violations) == 0,
            score=score,
            violations=violations,
            warnings=warnings,
            metrics=metrics
        )
    
    def detect_response_time_anomalies(self, response_times: List[float]) -> Dict[str, Any]:
        """Detect anomalies in response time patterns"""
        if len(self.historical_data) < 10:
            # Not enough historical data
            self.historical_data.extend(response_times)
            return {"anomalies_detected": False, "message": "Insufficient historical data"}
        
        # Simple anomaly detection using historical baseline
        historical_avg = statistics.mean(self.historical_data)
        historical_std = statistics.stdev(self.historical_data)
        
        current_avg = statistics.mean(response_times)
        anomaly_threshold = 2.0  # 2 standard deviations
        
        anomalies = []
        
        # Check for significant increase in average response time
        if current_avg > historical_avg + (anomaly_threshold * historical_std):
            anomalies.append({
                "type": "response_time_spike",
                "severity": "high",
                "current_avg": current_avg,
                "historical_avg": historical_avg,
                "increase_factor": current_avg / historical_avg
            })
        
        # Check for unusual distribution
        current_p95 = np.percentile(response_times, 95)
        historical_p95 = np.percentile(self.historical_data, 95)
        
        if current_p95 > historical_p95 * 1.5:  # 50% increase in P95
            anomalies.append({
                "type": "tail_latency_increase",
                "severity": "medium",
                "current_p95": current_p95,
                "historical_p95": historical_p95
            })
        
        # Update historical data
        self.historical_data.extend(response_times[-100:])  # Keep last 100 samples
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]  # Keep last 1000 samples
        
        return {
            "anomalies_detected": len(anomalies) > 0,
            "anomalies": anomalies,
            "baseline_avg": historical_avg,
            "current_avg": current_avg
        }


class ErrorRateValidator:
    """
    Error Rate Validator
    
    Validates error rates, error distribution patterns,
    and error rate trends for reliability assessment.
    """
    
    def __init__(self, error_rate_targets: Dict[str, Any]):
        self.error_rate_targets = error_rate_targets
        self.error_history = []
    
    def validate_error_rate(self, error_data: Dict[str, Any]) -> ValidationResult:
        """Validate error rate metrics"""
        violations = []
        warnings = []
        score = 1.0
        
        error_rate = error_data.get("error_rate", 0.0)
        total_requests = error_data.get("total_requests", 0)
        error_count = error_data.get("error_count", 0)
        
        # Validate overall error rate
        max_error_rate = self.error_rate_targets.get("max_error_rate", 0.01)  # 1% default
        if error_rate > max_error_rate:
            violations.append({
                "type": "error_rate_violation",
                "metric": "overall_error_rate",
                "expected": max_error_rate,
                "actual": error_rate,
                "severity": "critical"
            })
            score *= 0.5
        elif error_rate > max_error_rate * 0.5:  # Warning at 50% of limit
            warnings.append({
                "type": "error_rate_warning",
                "metric": "overall_error_rate",
                "threshold": max_error_rate * 0.5,
                "actual": error_rate,
                "severity": "medium"
            })
            score *= 0.8
        
        # Validate error distribution by type
        error_types = error_data.get("error_types", {})
        for error_type, count in error_types.items():
            type_rate = count / total_requests if total_requests > 0 else 0
            
            # Different thresholds for different error types
            if error_type in ["timeout", "connection_error"]:
                type_threshold = 0.005  # 0.5% for infrastructure errors
            elif error_type in ["4xx", "client_error"]:
                type_threshold = 0.02   # 2% for client errors
            else:
                type_threshold = 0.01   # 1% for other errors
            
            if type_rate > type_threshold:
                violations.append({
                    "type": "error_type_violation",
                    "metric": f"{error_type}_error_rate",
                    "expected": type_threshold,
                    "actual": type_rate,
                    "severity": "medium"
                })
                score *= 0.9
        
        # Check error rate trends
        self.error_history.append(error_rate)
        if len(self.error_history) > 10:
            recent_trend = self._calculate_error_trend()
            if recent_trend > 0.5:  # Increasing trend
                warnings.append({
                    "type": "error_trend_warning",
                    "metric": "error_rate_trend",
                    "trend": "increasing",
                    "trend_score": recent_trend,
                    "severity": "medium"
                })
                score *= 0.95
        
        metrics = {
            "error_rate": error_rate,
            "error_count": error_count,
            "total_requests": total_requests,
            "error_rate_score": score
        }
        
        return ValidationResult(
            valid=len(violations) == 0,
            score=score,
            violations=violations,
            warnings=warnings,
            metrics=metrics
        )
    
    def _calculate_error_trend(self) -> float:
        """Calculate error rate trend (0 = decreasing, 1 = increasing)"""
        if len(self.error_history) < 5:
            return 0.5  # Neutral
        
        recent_errors = self.error_history[-5:]
        earlier_errors = self.error_history[-10:-5] if len(self.error_history) >= 10 else self.error_history[:-5]
        
        if not earlier_errors:
            return 0.5
        
        recent_avg = statistics.mean(recent_errors)
        earlier_avg = statistics.mean(earlier_errors)
        
        if earlier_avg == 0:
            return 1.0 if recent_avg > 0 else 0.0
        
        # Normalize to 0-1 scale
        ratio = recent_avg / earlier_avg
        return min(1.0, max(0.0, (ratio - 0.5) * 2))


def create_performance_validator_suite(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a complete performance validation suite
    
    Args:
        config: Configuration containing SLA requirements, targets, and limits
    
    Returns:
        Dictionary containing all validator instances
    """
    
    sla_requirements = config.get("sla_requirements", {})
    capacity_requirements = config.get("capacity_requirements", {})
    resource_limits = config.get("resource_limits", {})
    response_time_targets = config.get("response_time_targets", {})
    error_rate_targets = config.get("error_rate_targets", {})
    
    validators = {
        "latency": LatencyValidator(sla_requirements),
        "throughput": ThroughputValidator(capacity_requirements),
        "resource_usage": ResourceUsageValidator(resource_limits),
        "response_time": ResponseTimeValidator(response_time_targets),
        "error_rate": ErrorRateValidator(error_rate_targets)
    }
    
    logger.info("Created performance validation suite with all validators")
    return validators


async def run_comprehensive_validation(validators: Dict[str, Any], performance_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive performance validation using all validators
    
    Args:
        validators: Dictionary of validator instances
        performance_data: Performance data to validate
    
    Returns:
        Comprehensive validation results
    """
    
    results = {}
    overall_score = 1.0
    total_violations = []
    total_warnings = []
    
    # Run latency validation
    if "latency" in validators and "response_time_ms" in performance_data:
        latency_result = validators["latency"].validate_latency_metrics(performance_data["response_time_ms"])
        results["latency"] = latency_result
        overall_score *= latency_result.score
        total_violations.extend(latency_result.violations)
        total_warnings.extend(latency_result.warnings)
    
    # Run throughput validation
    if "throughput" in validators and "throughput_data" in performance_data:
        throughput_result = validators["throughput"].validate_throughput_metrics(performance_data["throughput_data"])
        results["throughput"] = throughput_result
        overall_score *= throughput_result.score
        total_violations.extend(throughput_result.violations)
        total_warnings.extend(throughput_result.warnings)
    
    # Run resource usage validation
    if "resource_usage" in validators and "resource_usage" in performance_data:
        resource_result = validators["resource_usage"].validate_resource_usage(performance_data["resource_usage"])
        results["resource_usage"] = resource_result
        overall_score *= resource_result.score
        total_violations.extend(resource_result.violations)
        total_warnings.extend(resource_result.warnings)
    
    # Run response time distribution validation
    if "response_time" in validators and "response_times" in performance_data:
        response_time_result = validators["response_time"].validate_response_time_distribution(performance_data["response_times"])
        results["response_time_distribution"] = response_time_result
        overall_score *= response_time_result.score
        total_violations.extend(response_time_result.violations)
        total_warnings.extend(response_time_result.warnings)
    
    # Run error rate validation
    if "error_rate" in validators and "error_data" in performance_data:
        error_rate_result = validators["error_rate"].validate_error_rate(performance_data["error_data"])
        results["error_rate"] = error_rate_result
        overall_score *= error_rate_result.score
        total_violations.extend(error_rate_result.violations)
        total_warnings.extend(error_rate_result.warnings)
    
    # Calculate overall validation status
    validation_passed = len(total_violations) == 0
    
    comprehensive_result = {
        "validation_passed": validation_passed,
        "overall_score": overall_score,
        "total_violations": len(total_violations),
        "total_warnings": len(total_warnings),
        "individual_results": results,
        "violations": total_violations,
        "warnings": total_warnings,
        "validation_timestamp": time.time()
    }
    
    logger.info(f"Comprehensive validation completed. Passed: {validation_passed}, Score: {overall_score:.3f}")
    return comprehensive_result