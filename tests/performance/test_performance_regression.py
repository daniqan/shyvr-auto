"""
Performance Regression Detection Framework - Phase 7.2
Automated detection of performance regressions against established baselines.

Features:
- Baseline establishment and management
- Regression detection algorithms
- Performance trend analysis
- Automated alerting for performance degradation
- Historical performance tracking
- Statistical significance testing

Following TDD methodology - tests written first, then implementations.
"""
import pytest
import json
import statistics
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from scipy import stats

from tests.conftest_integration import (
    mock_environment_variables,
    performance_tracker
)


@dataclass
class PerformanceMeasurement:
    """Individual performance measurement."""
    operation: str
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any]
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation,
            'metric_name': self.metric_name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'success': self.success
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMeasurement':
        """Create from dictionary."""
        return cls(
            operation=data['operation'],
            metric_name=data['metric_name'],
            value=data['value'],
            unit=data['unit'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            context=data['context'],
            success=data['success']
        )


@dataclass
class PerformanceBaseline:
    """Performance baseline for an operation."""
    operation: str
    metric_name: str
    mean: float
    std_dev: float
    percentile_95: float
    percentile_99: float
    sample_count: int
    unit: str
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation,
            'metric_name': self.metric_name,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'percentile_95': self.percentile_95,
            'percentile_99': self.percentile_99,
            'sample_count': self.sample_count,
            'unit': self.unit,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """Create from dictionary."""
        return cls(
            operation=data['operation'],
            metric_name=data['metric_name'],
            mean=data['mean'],
            std_dev=data['std_dev'],
            percentile_95=data['percentile_95'],
            percentile_99=data['percentile_99'],
            sample_count=data['sample_count'],
            unit=data['unit'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated'])
        )


@dataclass
class RegressionAlert:
    """Performance regression alert."""
    operation: str
    metric_name: str
    baseline_value: float
    current_value: float
    regression_percentage: float
    severity: str  # 'minor', 'moderate', 'severe', 'critical'
    statistical_significance: float
    detected_at: datetime
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation,
            'metric_name': self.metric_name,
            'baseline_value': self.baseline_value,
            'current_value': self.current_value,
            'regression_percentage': self.regression_percentage,
            'severity': self.severity,
            'statistical_significance': self.statistical_significance,
            'detected_at': self.detected_at.isoformat(),
            'context': self.context
        }


class PerformanceBaselineManager:
    """Manages performance baselines and their persistence."""
    
    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = baseline_file
        self.baselines: Dict[Tuple[str, str], PerformanceBaseline] = {}
        
    def load_baselines(self) -> None:
        """Load baselines from file."""
        # This will initially fail - implementation needed
        pass
    
    def save_baselines(self) -> None:
        """Save baselines to file."""
        # This will initially fail - implementation needed
        pass
    
    def create_baseline(self, measurements: List[PerformanceMeasurement]) -> PerformanceBaseline:
        """Create baseline from measurements."""
        # This will initially fail - implementation needed
        pass
    
    def update_baseline(self, operation: str, metric_name: str, measurements: List[PerformanceMeasurement]) -> PerformanceBaseline:
        """Update existing baseline with new measurements."""
        # This will initially fail - implementation needed
        pass
    
    def get_baseline(self, operation: str, metric_name: str) -> Optional[PerformanceBaseline]:
        """Get baseline for operation and metric."""
        # This will initially fail - implementation needed
        pass
    
    def list_baselines(self) -> List[PerformanceBaseline]:
        """List all baselines."""
        # This will initially fail - implementation needed
        pass


class RegressionDetector:
    """Detects performance regressions against baselines."""
    
    def __init__(self, baseline_manager: PerformanceBaselineManager):
        self.baseline_manager = baseline_manager
        self.regression_thresholds = {
            'minor': 0.1,     # 10% regression
            'moderate': 0.2,  # 20% regression
            'severe': 0.35,   # 35% regression
            'critical': 0.5   # 50% regression
        }
        self.statistical_significance_threshold = 0.05  # p-value < 0.05
    
    def detect_regressions(self, measurements: List[PerformanceMeasurement]) -> List[RegressionAlert]:
        """Detect regressions in measurements against baselines."""
        # This will initially fail - implementation needed
        pass
    
    def analyze_single_measurement(self, measurement: PerformanceMeasurement) -> Optional[RegressionAlert]:
        """Analyze single measurement for regression."""
        # This will initially fail - implementation needed
        pass
    
    def calculate_statistical_significance(self, baseline: PerformanceBaseline, current_values: List[float]) -> float:
        """Calculate statistical significance of regression."""
        # This will initially fail - implementation needed
        pass
    
    def determine_severity(self, regression_percentage: float) -> str:
        """Determine severity level of regression."""
        # This will initially fail - implementation needed
        pass
    
    def batch_analyze_measurements(self, measurements: List[PerformanceMeasurement]) -> Dict[str, List[RegressionAlert]]:
        """Batch analyze measurements grouped by operation."""
        # This will initially fail - implementation needed
        pass


class TrendAnalyzer:
    """Analyzes performance trends over time."""
    
    def __init__(self):
        self.trend_window_days = 7  # Analyze trends over 7 days
        self.significance_threshold = 0.05
    
    def analyze_trends(self, measurements: List[PerformanceMeasurement], operation: str, metric_name: str) -> Dict[str, Any]:
        """Analyze performance trends for specific operation and metric."""
        # This will initially fail - implementation needed
        pass
    
    def detect_gradual_degradation(self, measurements: List[PerformanceMeasurement]) -> Optional[Dict[str, Any]]:
        """Detect gradual performance degradation over time."""
        # This will initially fail - implementation needed
        pass
    
    def forecast_performance(self, measurements: List[PerformanceMeasurement], days_ahead: int = 7) -> Dict[str, Any]:
        """Forecast performance trends."""
        # This will initially fail - implementation needed
        pass
    
    def calculate_trend_slope(self, measurements: List[PerformanceMeasurement]) -> Tuple[float, float]:
        """Calculate trend slope and R-squared value."""
        # This will initially fail - implementation needed
        pass


class PerformanceRegressionFramework:
    """Main framework for performance regression detection."""
    
    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_manager = PerformanceBaselineManager(baseline_file)
        self.regression_detector = RegressionDetector(self.baseline_manager)
        self.trend_analyzer = TrendAnalyzer()
        self.measurement_history: List[PerformanceMeasurement] = []
        
    def initialize(self) -> None:
        """Initialize the framework."""
        # This will initially fail - implementation needed
        pass
    
    def add_measurement(self, measurement: PerformanceMeasurement) -> Optional[RegressionAlert]:
        """Add measurement and check for immediate regressions."""
        # This will initially fail - implementation needed
        pass
    
    def batch_add_measurements(self, measurements: List[PerformanceMeasurement]) -> List[RegressionAlert]:
        """Add multiple measurements and detect regressions."""
        # This will initially fail - implementation needed
        pass
    
    def establish_baseline(self, operation: str, metric_name: str, measurements: List[PerformanceMeasurement]) -> PerformanceBaseline:
        """Establish new baseline for operation and metric."""
        # This will initially fail - implementation needed
        pass
    
    def update_baselines(self, auto_update: bool = True) -> List[PerformanceBaseline]:
        """Update baselines based on recent measurements."""
        # This will initially fail - implementation needed
        pass
    
    def generate_regression_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive regression report."""
        # This will initially fail - implementation needed
        pass
    
    def get_performance_health_score(self) -> float:
        """Calculate overall performance health score (0-100)."""
        # This will initially fail - implementation needed
        pass


class TestPerformanceBaselineManager:
    """Test performance baseline management."""
    
    @pytest.fixture
    def temp_baseline_file(self):
        """Provide temporary baseline file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            baseline_file = f.name
        yield baseline_file
        os.unlink(baseline_file)
    
    @pytest.fixture
    def baseline_manager(self, temp_baseline_file):
        """Provide baseline manager with temporary file."""
        return PerformanceBaselineManager(temp_baseline_file)
    
    @pytest.fixture
    def sample_measurements(self):
        """Provide sample performance measurements."""
        measurements = []
        for i in range(100):
            measurements.append(PerformanceMeasurement(
                operation="test_operation",
                metric_name="latency_ms",
                value=50.0 + np.random.normal(0, 5),  # 50ms ± 5ms
                unit="ms",
                timestamp=datetime.now() - timedelta(minutes=i),
                context={"test_id": i},
                success=True
            ))
        return measurements
    
    def test_baseline_creation_from_measurements(self, baseline_manager, sample_measurements):
        """Test creating baseline from measurements."""
        # This will initially fail - implementation needed
        baseline = baseline_manager.create_baseline(sample_measurements)
        
        assert baseline is not None, "Baseline should be created"
        assert baseline.operation == "test_operation", "Baseline operation should match"
        assert baseline.metric_name == "latency_ms", "Baseline metric should match"
        assert baseline.sample_count == 100, "Baseline should include all samples"
        assert 40 < baseline.mean < 60, f"Baseline mean {baseline.mean} outside expected range"
        assert baseline.std_dev > 0, "Baseline should have positive standard deviation"
        assert baseline.percentile_95 > baseline.mean, "P95 should be greater than mean"
        assert baseline.percentile_99 > baseline.percentile_95, "P99 should be greater than P95"
    
    def test_baseline_persistence(self, baseline_manager, sample_measurements):
        """Test baseline save and load functionality."""
        # Create and save baseline
        baseline = baseline_manager.create_baseline(sample_measurements)
        baseline_manager.baselines[("test_operation", "latency_ms")] = baseline
        baseline_manager.save_baselines()
        
        # Create new manager and load baselines
        new_manager = PerformanceBaselineManager(baseline_manager.baseline_file)
        new_manager.load_baselines()
        
        loaded_baseline = new_manager.get_baseline("test_operation", "latency_ms")
        assert loaded_baseline is not None, "Baseline should be loaded"
        assert loaded_baseline.mean == baseline.mean, "Loaded baseline should match original"
        assert loaded_baseline.sample_count == baseline.sample_count, "Sample count should match"
    
    def test_baseline_update(self, baseline_manager, sample_measurements):
        """Test updating existing baseline with new measurements."""
        # Create initial baseline
        initial_baseline = baseline_manager.create_baseline(sample_measurements[:50])
        baseline_manager.baselines[("test_operation", "latency_ms")] = initial_baseline
        
        # Update with additional measurements
        updated_baseline = baseline_manager.update_baseline(
            "test_operation", 
            "latency_ms", 
            sample_measurements[50:]
        )
        
        assert updated_baseline.sample_count == 100, "Updated baseline should include all samples"
        assert updated_baseline.last_updated > initial_baseline.created_at, "Update timestamp should be newer"
    
    def test_baseline_listing(self, baseline_manager, sample_measurements):
        """Test listing all baselines."""
        # Create multiple baselines
        latency_baseline = baseline_manager.create_baseline(sample_measurements)
        
        memory_measurements = [
            PerformanceMeasurement(
                operation="test_operation",
                metric_name="memory_mb",
                value=100.0 + np.random.normal(0, 10),
                unit="MB",
                timestamp=datetime.now(),
                context={},
                success=True
            )
            for _ in range(50)
        ]
        memory_baseline = baseline_manager.create_baseline(memory_measurements)
        
        baseline_manager.baselines[("test_operation", "latency_ms")] = latency_baseline
        baseline_manager.baselines[("test_operation", "memory_mb")] = memory_baseline
        
        all_baselines = baseline_manager.list_baselines()
        assert len(all_baselines) == 2, "Should have 2 baselines"
        assert any(b.metric_name == "latency_ms" for b in all_baselines), "Should include latency baseline"
        assert any(b.metric_name == "memory_mb" for b in all_baselines), "Should include memory baseline"


class TestRegressionDetector:
    """Test regression detection functionality."""
    
    @pytest.fixture
    def baseline_manager(self):
        """Provide baseline manager for testing."""
        return PerformanceBaselineManager("test_baselines.json")
    
    @pytest.fixture
    def regression_detector(self, baseline_manager):
        """Provide regression detector."""
        return RegressionDetector(baseline_manager)
    
    @pytest.fixture
    def baseline_measurements(self):
        """Provide baseline measurements (good performance)."""
        return [
            PerformanceMeasurement(
                operation="critical_operation",
                metric_name="latency_ms",
                value=50.0 + np.random.normal(0, 2),  # 50ms ± 2ms
                unit="ms",
                timestamp=datetime.now() - timedelta(days=7, minutes=i),
                context={},
                success=True
            )
            for i in range(100)
        ]
    
    @pytest.fixture
    def regression_measurements(self):
        """Provide measurements showing regression."""
        return [
            PerformanceMeasurement(
                operation="critical_operation",
                metric_name="latency_ms",
                value=75.0 + np.random.normal(0, 3),  # 75ms ± 3ms (50% increase)
                unit="ms",
                timestamp=datetime.now() - timedelta(minutes=i),
                context={},
                success=True
            )
            for i in range(50)
        ]
    
    def test_regression_detection_setup(self, baseline_manager, regression_detector, baseline_measurements):
        """Test regression detector setup with baseline."""
        # Create baseline
        baseline = baseline_manager.create_baseline(baseline_measurements)
        baseline_manager.baselines[("critical_operation", "latency_ms")] = baseline
        
        assert regression_detector.baseline_manager == baseline_manager, "Detector should use provided baseline manager"
        assert "minor" in regression_detector.regression_thresholds, "Should have minor threshold"
        assert "critical" in regression_detector.regression_thresholds, "Should have critical threshold"
    
    def test_single_measurement_regression_detection(self, baseline_manager, regression_detector, baseline_measurements):
        """Test detecting regression in single measurement."""
        # Establish baseline
        baseline = baseline_manager.create_baseline(baseline_measurements)
        baseline_manager.baselines[("critical_operation", "latency_ms")] = baseline
        
        # Create measurement showing regression
        regression_measurement = PerformanceMeasurement(
            operation="critical_operation",
            metric_name="latency_ms",
            value=80.0,  # 60% increase from 50ms baseline
            unit="ms",
            timestamp=datetime.now(),
            context={},
            success=True
        )
        
        # This will initially fail - implementation needed
        alert = regression_detector.analyze_single_measurement(regression_measurement)
        
        assert alert is not None, "Should detect regression"
        assert alert.operation == "critical_operation", "Alert should identify correct operation"
        assert alert.regression_percentage > 0.5, "Should detect significant regression"
        assert alert.severity in ["severe", "critical"], "Should classify as severe regression"
    
    def test_batch_regression_detection(self, baseline_manager, regression_detector, baseline_measurements, regression_measurements):
        """Test batch regression detection."""
        # Establish baseline
        baseline = baseline_manager.create_baseline(baseline_measurements)
        baseline_manager.baselines[("critical_operation", "latency_ms")] = baseline
        
        # This will initially fail - implementation needed
        alerts = regression_detector.detect_regressions(regression_measurements)
        
        assert len(alerts) > 0, "Should detect regressions in batch"
        assert all(alert.operation == "critical_operation" for alert in alerts), "All alerts should be for correct operation"
        assert any(alert.severity in ["severe", "critical"] for alert in alerts), "Should detect severe regressions"
    
    def test_statistical_significance_calculation(self, baseline_manager, regression_detector, baseline_measurements, regression_measurements):
        """Test statistical significance calculation."""
        baseline = baseline_manager.create_baseline(baseline_measurements)
        current_values = [m.value for m in regression_measurements]
        
        # This will initially fail - implementation needed
        p_value = regression_detector.calculate_statistical_significance(baseline, current_values)
        
        assert 0 <= p_value <= 1, "P-value should be between 0 and 1"
        assert p_value < 0.05, "Should detect statistically significant regression"
    
    def test_severity_classification(self, regression_detector):
        """Test regression severity classification."""
        # This will initially fail - implementation needed
        assert regression_detector.determine_severity(0.05) == "minor", "5% regression should be minor"
        assert regression_detector.determine_severity(0.15) == "moderate", "15% regression should be moderate"
        assert regression_detector.determine_severity(0.4) == "severe", "40% regression should be severe"
        assert regression_detector.determine_severity(0.6) == "critical", "60% regression should be critical"


class TestTrendAnalyzer:
    """Test performance trend analysis."""
    
    @pytest.fixture
    def trend_analyzer(self):
        """Provide trend analyzer."""
        return TrendAnalyzer()
    
    @pytest.fixture
    def trending_measurements(self):
        """Provide measurements showing performance trend."""
        measurements = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(168):  # 7 days of hourly measurements
            # Simulate gradual degradation: 50ms + 0.1ms per hour
            value = 50.0 + (i * 0.1) + np.random.normal(0, 1)
            measurements.append(PerformanceMeasurement(
                operation="trending_operation",
                metric_name="latency_ms",
                value=value,
                unit="ms",
                timestamp=base_time + timedelta(hours=i),
                context={"hour": i},
                success=True
            ))
        
        return measurements
    
    def test_trend_analysis(self, trend_analyzer, trending_measurements):
        """Test basic trend analysis."""
        # This will initially fail - implementation needed
        trend_result = trend_analyzer.analyze_trends(
            trending_measurements, 
            "trending_operation", 
            "latency_ms"
        )
        
        assert trend_result is not None, "Should return trend analysis"
        assert "trend_direction" in trend_result, "Should identify trend direction"
        assert "slope" in trend_result, "Should calculate trend slope"
        assert "r_squared" in trend_result, "Should calculate R-squared"
        assert trend_result["trend_direction"] in ["improving", "stable", "degrading"], "Should classify trend"
    
    def test_gradual_degradation_detection(self, trend_analyzer, trending_measurements):
        """Test detection of gradual performance degradation."""
        # This will initially fail - implementation needed
        degradation = trend_analyzer.detect_gradual_degradation(trending_measurements)
        
        assert degradation is not None, "Should detect gradual degradation"
        assert degradation["degradation_detected"] == True, "Should flag degradation"
        assert degradation["degradation_rate"] > 0, "Should calculate positive degradation rate"
    
    def test_performance_forecasting(self, trend_analyzer, trending_measurements):
        """Test performance forecasting."""
        # This will initially fail - implementation needed
        forecast = trend_analyzer.forecast_performance(trending_measurements, days_ahead=3)
        
        assert forecast is not None, "Should return forecast"
        assert "predicted_values" in forecast, "Should include predicted values"
        assert "confidence_interval" in forecast, "Should include confidence interval"
        assert len(forecast["predicted_values"]) > 0, "Should have predicted values"
    
    def test_trend_slope_calculation(self, trend_analyzer, trending_measurements):
        """Test trend slope calculation."""
        # This will initially fail - implementation needed
        slope, r_squared = trend_analyzer.calculate_trend_slope(trending_measurements)
        
        assert slope > 0, "Should detect positive slope (degradation)"
        assert 0 <= r_squared <= 1, "R-squared should be between 0 and 1"
        assert r_squared > 0.8, "Should have strong correlation for synthetic trend"


class TestPerformanceRegressionFramework:
    """Test integrated performance regression framework."""
    
    @pytest.fixture
    def temp_baseline_file(self):
        """Provide temporary baseline file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            baseline_file = f.name
        yield baseline_file
        os.unlink(baseline_file)
    
    @pytest.fixture
    def framework(self, temp_baseline_file):
        """Provide performance regression framework."""
        return PerformanceRegressionFramework(temp_baseline_file)
    
    @pytest.fixture
    def sample_measurements(self):
        """Provide sample measurements for framework testing."""
        measurements = []
        
        # Good baseline measurements
        for i in range(50):
            measurements.append(PerformanceMeasurement(
                operation="framework_test",
                metric_name="latency_ms",
                value=50.0 + np.random.normal(0, 2),
                unit="ms",
                timestamp=datetime.now() - timedelta(hours=48, minutes=i),
                context={"phase": "baseline"},
                success=True
            ))
        
        # Recent measurements with some regressions
        for i in range(30):
            base_value = 50.0 if i < 20 else 70.0  # Regression in last 10 measurements
            measurements.append(PerformanceMeasurement(
                operation="framework_test",
                metric_name="latency_ms",
                value=base_value + np.random.normal(0, 3),
                unit="ms",
                timestamp=datetime.now() - timedelta(minutes=i),
                context={"phase": "current"},
                success=True
            ))
        
        return measurements
    
    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        # This will initially fail - implementation needed
        framework.initialize()
        
        assert framework.baseline_manager is not None, "Should have baseline manager"
        assert framework.regression_detector is not None, "Should have regression detector"
        assert framework.trend_analyzer is not None, "Should have trend analyzer"
        assert isinstance(framework.measurement_history, list), "Should initialize measurement history"
    
    def test_baseline_establishment(self, framework, sample_measurements):
        """Test establishing baselines through framework."""
        framework.initialize()
        
        baseline_measurements = [m for m in sample_measurements if m.context.get("phase") == "baseline"]
        
        # This will initially fail - implementation needed
        baseline = framework.establish_baseline(
            "framework_test", 
            "latency_ms", 
            baseline_measurements
        )
        
        assert baseline is not None, "Should establish baseline"
        assert baseline.operation == "framework_test", "Baseline should match operation"
        assert baseline.sample_count == len(baseline_measurements), "Should use all baseline measurements"
    
    def test_measurement_addition_and_regression_detection(self, framework, sample_measurements):
        """Test adding measurements and detecting regressions."""
        framework.initialize()
        
        # Establish baseline
        baseline_measurements = [m for m in sample_measurements if m.context.get("phase") == "baseline"]
        framework.establish_baseline("framework_test", "latency_ms", baseline_measurements)
        
        # Add current measurements
        current_measurements = [m for m in sample_measurements if m.context.get("phase") == "current"]
        
        # This will initially fail - implementation needed
        alerts = framework.batch_add_measurements(current_measurements)
        
        assert len(alerts) > 0, "Should detect regressions"
        assert any(alert.severity in ["moderate", "severe", "critical"] for alert in alerts), "Should detect significant regressions"
    
    def test_regression_report_generation(self, framework, sample_measurements):
        """Test comprehensive regression report generation."""
        framework.initialize()
        
        # Setup baselines and measurements
        baseline_measurements = [m for m in sample_measurements if m.context.get("phase") == "baseline"]
        framework.establish_baseline("framework_test", "latency_ms", baseline_measurements)
        
        current_measurements = [m for m in sample_measurements if m.context.get("phase") == "current"]
        framework.batch_add_measurements(current_measurements)
        
        # This will initially fail - implementation needed
        report = framework.generate_regression_report(time_window_hours=24)
        
        assert report is not None, "Should generate report"
        assert "summary" in report, "Report should have summary"
        assert "regressions_detected" in report, "Report should list regressions"
        assert "baselines_status" in report, "Report should include baseline status"
        assert "recommendations" in report, "Report should include recommendations"
    
    def test_performance_health_score(self, framework, sample_measurements):
        """Test performance health score calculation."""
        framework.initialize()
        
        # Setup measurements
        framework.batch_add_measurements(sample_measurements)
        
        # This will initially fail - implementation needed
        health_score = framework.get_performance_health_score()
        
        assert 0 <= health_score <= 100, "Health score should be between 0 and 100"
        assert health_score < 100, "Health score should be less than 100 due to regressions"
    
    def test_automatic_baseline_updates(self, framework, sample_measurements):
        """Test automatic baseline updates."""
        framework.initialize()
        
        # Establish initial baseline
        baseline_measurements = [m for m in sample_measurements if m.context.get("phase") == "baseline"]
        framework.establish_baseline("framework_test", "latency_ms", baseline_measurements)
        
        # Add more measurements
        current_measurements = [m for m in sample_measurements if m.context.get("phase") == "current"]
        framework.batch_add_measurements(current_measurements)
        
        # This will initially fail - implementation needed
        updated_baselines = framework.update_baselines(auto_update=True)
        
        assert len(updated_baselines) > 0, "Should update baselines"
        assert any(b.operation == "framework_test" for b in updated_baselines), "Should update test operation baseline"


class TestRegressionDetectionIntegration:
    """Integration tests for regression detection framework."""
    
    def test_end_to_end_regression_detection_workflow(self, mock_environment_variables, performance_tracker):
        """Test complete end-to-end regression detection workflow."""
        with patch.dict('os.environ', mock_environment_variables):
            print("\n=== END-TO-END REGRESSION DETECTION TEST ===")
            
            # Initialize framework
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                baseline_file = f.name
            
            try:
                framework = PerformanceRegressionFramework(baseline_file)
                framework.initialize()
                
                # Simulate production performance measurements
                print("\n1. Establishing Performance Baselines:")
                
                # Create baseline measurements (good performance period)
                baseline_measurements = []
                for operation in ["order_execution", "ml_prediction", "safety_validation"]:
                    base_latency = {"order_execution": 50, "ml_prediction": 5, "safety_validation": 25}[operation]
                    
                    for i in range(100):
                        baseline_measurements.append(PerformanceMeasurement(
                            operation=operation,
                            metric_name="latency_ms",
                            value=base_latency + np.random.normal(0, base_latency * 0.1),
                            unit="ms",
                            timestamp=datetime.now() - timedelta(days=7, minutes=i),
                            context={"environment": "production"},
                            success=True
                        ))
                
                # Establish baselines
                for operation in ["order_execution", "ml_prediction", "safety_validation"]:
                    op_measurements = [m for m in baseline_measurements if m.operation == operation]
                    baseline = framework.establish_baseline(operation, "latency_ms", op_measurements)
                    print(f"   ✓ {operation}: {baseline.mean:.1f}ms ± {baseline.std_dev:.1f}ms")
                
                # Simulate current measurements with regressions
                print("\n2. Analyzing Current Performance:")
                
                current_measurements = []
                for operation in ["order_execution", "ml_prediction", "safety_validation"]:
                    base_latency = {"order_execution": 50, "ml_prediction": 5, "safety_validation": 25}[operation]
                    # Simulate 30% performance degradation
                    degraded_latency = base_latency * 1.3
                    
                    for i in range(50):
                        current_measurements.append(PerformanceMeasurement(
                            operation=operation,
                            metric_name="latency_ms",
                            value=degraded_latency + np.random.normal(0, degraded_latency * 0.1),
                            unit="ms",
                            timestamp=datetime.now() - timedelta(minutes=i),
                            context={"environment": "production"},
                            success=True
                        ))
                
                # Detect regressions
                alerts = framework.batch_add_measurements(current_measurements)
                print(f"   ✓ Detected {len(alerts)} performance regressions")
                
                for alert in alerts[:3]:  # Show first 3 alerts
                    print(f"   • {alert.operation}: {alert.regression_percentage:.1%} regression ({alert.severity})")
                
                # Generate comprehensive report
                print("\n3. Generating Regression Report:")
                
                report = framework.generate_regression_report(time_window_hours=24)
                print(f"   ✓ Report generated with {len(report.get('regressions_detected', []))} regressions")
                print(f"   ✓ Performance health score: {framework.get_performance_health_score():.1f}/100")
                
                # Validate regression detection
                assert len(alerts) >= 3, f"Should detect regressions in all 3 operations, got {len(alerts)}"
                assert all(alert.regression_percentage > 0.2 for alert in alerts), "Should detect significant regressions"
                assert any(alert.severity in ["moderate", "severe"] for alert in alerts), "Should classify regressions appropriately"
                
                # Validate report contents
                assert "summary" in report, "Report should contain summary"
                assert len(report["regressions_detected"]) > 0, "Report should list detected regressions"
                
                # Validate health score
                health_score = framework.get_performance_health_score()
                assert 0 <= health_score <= 100, "Health score should be valid percentage"
                assert health_score < 80, f"Health score {health_score} should reflect regressions"
                
                print("\n=== REGRESSION DETECTION FRAMEWORK VALIDATED ✓ ===")
                
            finally:
                os.unlink(baseline_file)


# Mark all tests as regression detection tests
pytestmark = [pytest.mark.performance, pytest.mark.regression]