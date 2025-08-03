"""
Comprehensive Performance Reporting - Phase 7.2
Creates detailed performance reports and dashboards for all testing categories.

Features:
- Unified performance report generation
- Multi-format export (JSON, HTML, PDF, CSV)
- Interactive dashboard data generation
- Performance trend visualization
- Comparative analysis reports
- Executive summary generation
- Automated report scheduling

Following TDD methodology - tests written first, then implementations.
"""
import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field, asdict
import statistics
import numpy as np

from tests.conftest_integration import (
    mock_environment_variables,
    performance_tracker
)


@dataclass
class PerformanceTestSuite:
    """Represents a complete performance test suite."""
    suite_name: str
    test_category: str  # 'gcp', 'hft', 'regression', 'stress', 'monitoring', 'baseline', 'load', 'efficiency'
    execution_timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration_seconds: float
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'suite_name': self.suite_name,
            'test_category': self.test_category,
            'execution_timestamp': self.execution_timestamp.isoformat(),
            'summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'skipped_tests': self.skipped_tests,
                'success_rate': self.success_rate,
                'total_duration_seconds': self.total_duration_seconds
            },
            'test_results': self.test_results,
            'metrics': self.metrics
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    report_title: str
    generation_timestamp: datetime
    test_suites: List[PerformanceTestSuite]
    executive_summary: Dict[str, Any]
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'report_id': self.report_id,
            'report_title': self.report_title,
            'generation_timestamp': self.generation_timestamp.isoformat(),
            'executive_summary': self.executive_summary,
            'test_suites': [suite.to_dict() for suite in self.test_suites],
            'detailed_analysis': self.detailed_analysis,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }


class PerformanceDataCollector:
    """Collects performance data from all testing categories."""
    
    def __init__(self):
        self.collected_data: Dict[str, Any] = {}
        self.test_suites: List[PerformanceTestSuite] = []
        
    def collect_gcp_performance_data(self) -> Dict[str, Any]:
        """Collect GCP performance test data."""
        # This will initially fail - implementation needed
        pass
    
    def collect_hft_performance_data(self) -> Dict[str, Any]:
        """Collect high-frequency trading performance data."""
        # This will initially fail - implementation needed
        pass
    
    def collect_regression_data(self) -> Dict[str, Any]:
        """Collect performance regression data."""
        # This will initially fail - implementation needed
        pass
    
    def collect_stress_test_data(self) -> Dict[str, Any]:
        """Collect stress testing data."""
        # This will initially fail - implementation needed
        pass
    
    def collect_monitoring_data(self) -> Dict[str, Any]:
        """Collect continuous monitoring data."""
        # This will initially fail - implementation needed
        pass
    
    def collect_baseline_validation_data(self) -> Dict[str, Any]:
        """Collect baseline validation data."""
        # This will initially fail - implementation needed
        pass
    
    def collect_load_testing_data(self) -> Dict[str, Any]:
        """Collect advanced load testing data."""
        # This will initially fail - implementation needed
        pass
    
    def collect_efficiency_data(self) -> Dict[str, Any]:
        """Collect resource efficiency data."""
        # This will initially fail - implementation needed
        pass
    
    def aggregate_all_performance_data(self) -> Dict[str, Any]:
        """Aggregate all performance data into unified structure."""
        # This will initially fail - implementation needed
        pass


class ReportGenerator:
    """Generates performance reports in various formats."""
    
    def __init__(self):
        self.templates: Dict[str, str] = {}
        self.generated_reports: List[str] = []
        
    def generate_json_report(self, performance_report: PerformanceReport) -> str:
        """Generate JSON format report."""
        # This will initially fail - implementation needed
        pass
    
    def generate_html_report(self, performance_report: PerformanceReport) -> str:
        """Generate HTML format report."""
        # This will initially fail - implementation needed
        pass
    
    def generate_csv_export(self, performance_data: Dict[str, Any]) -> str:
        """Generate CSV export of performance data."""
        # This will initially fail - implementation needed
        pass
    
    def generate_executive_summary(self, test_suites: List[PerformanceTestSuite]) -> Dict[str, Any]:
        """Generate executive summary from test suites."""
        # This will initially fail - implementation needed
        pass
    
    def generate_detailed_analysis(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed performance analysis."""
        # This will initially fail - implementation needed
        pass
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        # This will initially fail - implementation needed
        pass
    
    def create_dashboard_data(self, performance_report: PerformanceReport) -> Dict[str, Any]:
        """Create data structure for performance dashboard."""
        # This will initially fail - implementation needed
        pass


class PerformanceAnalyzer:
    """Analyzes performance data for insights and trends."""
    
    def __init__(self):
        self.analysis_cache: Dict[str, Any] = {}
        
    def analyze_performance_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # This will initially fail - implementation needed
        pass
    
    def identify_performance_bottlenecks(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks across all test categories."""
        # This will initially fail - implementation needed
        pass
    
    def calculate_performance_scores(self, test_suites: List[PerformanceTestSuite]) -> Dict[str, float]:
        """Calculate overall performance scores by category."""
        # This will initially fail - implementation needed
        pass
    
    def compare_performance_across_categories(self, test_suites: List[PerformanceTestSuite]) -> Dict[str, Any]:
        """Compare performance across different test categories."""
        # This will initially fail - implementation needed
        pass
    
    def generate_performance_insights(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate actionable performance insights."""
        # This will initially fail - implementation needed
        pass
    
    def predict_performance_issues(self, trend_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict potential future performance issues."""
        # This will initially fail - implementation needed
        pass


class VisualizationDataGenerator:
    """Generates data for performance visualizations."""
    
    def __init__(self):
        self.chart_data: Dict[str, Any] = {}
        
    def create_performance_timeline_data(self, test_suites: List[PerformanceTestSuite]) -> Dict[str, Any]:
        """Create timeline visualization data."""
        # This will initially fail - implementation needed
        pass
    
    def create_category_comparison_data(self, performance_scores: Dict[str, float]) -> Dict[str, Any]:
        """Create category comparison chart data."""
        # This will initially fail - implementation needed
        pass
    
    def create_trend_analysis_data(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create trend analysis visualization data."""
        # This will initially fail - implementation needed
        pass
    
    def create_resource_utilization_data(self, efficiency_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create resource utilization visualization data."""
        # This will initially fail - implementation needed
        pass
    
    def create_bottleneck_analysis_data(self, bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create bottleneck analysis visualization data."""
        # This will initially fail - implementation needed
        pass
    
    def create_heatmap_data(self, performance_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Create performance heatmap data."""
        # This will initially fail - implementation needed
        pass


class ReportScheduler:
    """Schedules and manages automated report generation."""
    
    def __init__(self):
        self.scheduled_reports: List[Dict[str, Any]] = []
        self.report_history: List[str] = []
        
    def schedule_daily_report(self, report_config: Dict[str, Any]) -> str:
        """Schedule daily performance report generation."""
        # This will initially fail - implementation needed
        pass
    
    def schedule_weekly_summary(self, report_config: Dict[str, Any]) -> str:
        """Schedule weekly performance summary."""
        # This will initially fail - implementation needed
        pass
    
    def schedule_monthly_analysis(self, report_config: Dict[str, Any]) -> str:
        """Schedule monthly performance analysis."""
        # This will initially fail - implementation needed
        pass
    
    def execute_scheduled_reports(self) -> List[str]:
        """Execute all due scheduled reports."""
        # This will initially fail - implementation needed
        pass
    
    def get_report_schedule(self) -> List[Dict[str, Any]]:
        """Get current report schedule."""
        # This will initially fail - implementation needed
        pass


class PerformanceReportingFramework:
    """Main framework for comprehensive performance reporting."""
    
    def __init__(self):
        self.data_collector = PerformanceDataCollector()
        self.report_generator = ReportGenerator()
        self.analyzer = PerformanceAnalyzer()
        self.visualizer = VisualizationDataGenerator()
        self.scheduler = ReportScheduler()
        self.generated_reports: List[PerformanceReport] = []
        
    def generate_comprehensive_report(self, report_title: str = "Phase 7.2 Performance Report") -> PerformanceReport:
        """Generate comprehensive performance report."""
        # This will initially fail - implementation needed
        pass
    
    def generate_category_specific_report(self, category: str) -> PerformanceReport:
        """Generate report for specific performance category."""
        # This will initially fail - implementation needed
        pass
    
    def generate_comparison_report(self, baseline_data: Dict[str, Any], current_data: Dict[str, Any]) -> PerformanceReport:
        """Generate comparison report between baseline and current performance."""
        # This will initially fail - implementation needed
        pass
    
    def export_report(self, report: PerformanceReport, format_type: str, output_path: str) -> str:
        """Export report in specified format."""
        # This will initially fail - implementation needed
        pass
    
    def create_performance_dashboard(self, report: PerformanceReport) -> Dict[str, Any]:
        """Create performance dashboard from report."""
        # This will initially fail - implementation needed
        pass
    
    def archive_report(self, report: PerformanceReport, archive_path: str) -> str:
        """Archive performance report."""
        # This will initially fail - implementation needed
        pass


class TestPerformanceDataCollection:
    """Test performance data collection functionality."""
    
    @pytest.fixture
    def data_collector(self):
        """Provide performance data collector."""
        return PerformanceDataCollector()
    
    def test_gcp_performance_data_collection(self, mock_environment_variables, data_collector):
        """Test GCP performance data collection."""
        with patch.dict('os.environ', mock_environment_variables):
            # This will initially fail - implementation needed
            gcp_data = data_collector.collect_gcp_performance_data()
            
            assert gcp_data is not None, "Should collect GCP performance data"
            assert 'cloud_sql_metrics' in gcp_data, "Should include Cloud SQL metrics"
            assert 'secret_manager_metrics' in gcp_data, "Should include Secret Manager metrics"
            assert 'cloud_run_metrics' in gcp_data, "Should include Cloud Run metrics"
            assert 'network_metrics' in gcp_data, "Should include network metrics"
    
    def test_hft_performance_data_collection(self, mock_environment_variables, data_collector):
        """Test HFT performance data collection."""
        with patch.dict('os.environ', mock_environment_variables):
            # This will initially fail - implementation needed
            hft_data = data_collector.collect_hft_performance_data()
            
            assert hft_data is not None, "Should collect HFT performance data"
            assert 'trading_latency_metrics' in hft_data, "Should include trading latency metrics"
            assert 'throughput_metrics' in hft_data, "Should include throughput metrics"
            assert 'concurrent_trading_metrics' in hft_data, "Should include concurrent trading metrics"
            assert 'emergency_stop_metrics' in hft_data, "Should include emergency stop metrics"
    
    def test_comprehensive_data_aggregation(self, mock_environment_variables, data_collector):
        """Test comprehensive data aggregation."""
        with patch.dict('os.environ', mock_environment_variables):
            # This will initially fail - implementation needed
            aggregated_data = data_collector.aggregate_all_performance_data()
            
            assert aggregated_data is not None, "Should aggregate all performance data"
            assert 'categories' in aggregated_data, "Should organize data by categories"
            assert 'summary_metrics' in aggregated_data, "Should include summary metrics"
            assert 'timestamp' in aggregated_data, "Should include collection timestamp"
            
            # Validate expected categories
            expected_categories = ['gcp', 'hft', 'regression', 'stress', 'monitoring', 'baseline', 'load', 'efficiency']
            for category in expected_categories:
                assert category in aggregated_data['categories'], f"Should include {category} category data"


class TestReportGeneration:
    """Test report generation functionality."""
    
    @pytest.fixture
    def report_generator(self):
        """Provide report generator."""
        return ReportGenerator()
    
    @pytest.fixture
    def sample_test_suites(self):
        """Provide sample test suites for testing."""
        suites = []
        categories = ['gcp', 'hft', 'stress', 'baseline']
        
        for i, category in enumerate(categories):
            suite = PerformanceTestSuite(
                suite_name=f"{category}_performance_tests",
                test_category=category,
                execution_timestamp=datetime.now() - timedelta(hours=i),
                total_tests=50 + i * 10,
                passed_tests=45 + i * 8,
                failed_tests=3 + i,
                skipped_tests=2 + i,
                total_duration_seconds=300 + i * 60,
                test_results=[],
                metrics={
                    'avg_latency_ms': 50 + i * 10,
                    'success_rate': 0.9 + i * 0.02,
                    'throughput_ops_sec': 100 + i * 50
                }
            )
            suites.append(suite)
        
        return suites
    
    def test_executive_summary_generation(self, report_generator, sample_test_suites):
        """Test executive summary generation."""
        # This will initially fail - implementation needed
        summary = report_generator.generate_executive_summary(sample_test_suites)
        
        assert summary is not None, "Should generate executive summary"
        assert 'overall_success_rate' in summary, "Should include overall success rate"
        assert 'total_tests_executed' in summary, "Should include total tests executed"
        assert 'performance_highlights' in summary, "Should include performance highlights"
        assert 'areas_of_concern' in summary, "Should include areas of concern"
        
        # Validate summary calculations
        total_tests = sum(suite.total_tests for suite in sample_test_suites)
        assert summary['total_tests_executed'] == total_tests, "Should calculate correct total tests"
        
        overall_success = sum(suite.passed_tests for suite in sample_test_suites) / total_tests
        assert abs(summary['overall_success_rate'] - overall_success) < 0.01, "Should calculate correct success rate"
    
    def test_detailed_analysis_generation(self, report_generator):
        """Test detailed analysis generation."""
        sample_performance_data = {
            'gcp': {
                'cloud_sql_latency_ms': [45, 50, 55, 48, 52],
                'secret_manager_latency_ms': [20, 25, 22, 24, 21],
                'network_latency_ms': [5, 7, 6, 8, 5]
            },
            'hft': {
                'order_execution_latency_ms': [75, 80, 85, 78, 82],
                'ml_prediction_throughput': [1100, 1050, 1200, 1150, 1075],
                'emergency_stop_latency_ms': [8, 10, 9, 11, 7]
            }
        }
        
        # This will initially fail - implementation needed
        analysis = report_generator.generate_detailed_analysis(sample_performance_data)
        
        assert analysis is not None, "Should generate detailed analysis"
        assert 'category_analysis' in analysis, "Should include category-specific analysis"
        assert 'performance_statistics' in analysis, "Should include performance statistics"
        assert 'trend_analysis' in analysis, "Should include trend analysis"
        
        # Validate analysis content
        assert 'gcp' in analysis['category_analysis'], "Should analyze GCP performance"
        assert 'hft' in analysis['category_analysis'], "Should analyze HFT performance"
    
    def test_json_report_generation(self, report_generator, sample_test_suites):
        """Test JSON report generation."""
        # Create a sample performance report
        report = PerformanceReport(
            report_id="test_report_001",
            report_title="Test Performance Report",
            generation_timestamp=datetime.now(),
            test_suites=sample_test_suites,
            executive_summary={'overall_success_rate': 0.92},
            detailed_analysis={'categories': len(sample_test_suites)},
            recommendations=["Optimize memory usage", "Improve error handling"]
        )
        
        # This will initially fail - implementation needed
        json_report = report_generator.generate_json_report(report)
        
        assert json_report is not None, "Should generate JSON report"
        assert len(json_report) > 0, "JSON report should not be empty"
        
        # Validate JSON structure
        try:
            parsed_json = json.loads(json_report)
            assert 'report_id' in parsed_json, "Should include report ID"
            assert 'test_suites' in parsed_json, "Should include test suites"
            assert 'executive_summary' in parsed_json, "Should include executive summary"
            assert 'recommendations' in parsed_json, "Should include recommendations"
        except json.JSONDecodeError:
            pytest.fail("Generated report should be valid JSON")
    
    def test_html_report_generation(self, report_generator, sample_test_suites):
        """Test HTML report generation."""
        report = PerformanceReport(
            report_id="test_report_002",
            report_title="Test HTML Report",
            generation_timestamp=datetime.now(),
            test_suites=sample_test_suites,
            executive_summary={'overall_success_rate': 0.92},
            detailed_analysis={'categories': len(sample_test_suites)},
            recommendations=["Optimize database queries", "Increase cache size"]
        )
        
        # This will initially fail - implementation needed
        html_report = report_generator.generate_html_report(report)
        
        assert html_report is not None, "Should generate HTML report"
        assert len(html_report) > 0, "HTML report should not be empty"
        assert '<html>' in html_report.lower(), "Should be valid HTML"
        assert '<title>' in html_report.lower(), "Should include title"
        assert 'test html report' in html_report.lower(), "Should include report title"


class TestPerformanceAnalysis:
    """Test performance analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Provide performance analyzer."""
        return PerformanceAnalyzer()
    
    def test_performance_trend_analysis(self, analyzer):
        """Test performance trend analysis."""
        # Create historical performance data
        historical_data = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            data_point = {
                'timestamp': (base_time + timedelta(days=i)).isoformat(),
                'latency_ms': 50 + np.random.normal(0, 5) + (i * 0.5),  # Gradual increase
                'throughput': 1000 - (i * 2) + np.random.normal(0, 10),  # Gradual decrease
                'success_rate': 0.95 - (i * 0.001) + np.random.normal(0, 0.01)  # Slight decline
            }
            historical_data.append(data_point)
        
        # This will initially fail - implementation needed
        trend_analysis = analyzer.analyze_performance_trends(historical_data)
        
        assert trend_analysis is not None, "Should analyze performance trends"
        assert 'latency_trend' in trend_analysis, "Should analyze latency trends"
        assert 'throughput_trend' in trend_analysis, "Should analyze throughput trends"
        assert 'success_rate_trend' in trend_analysis, "Should analyze success rate trends"
        
        # Validate trend detection
        latency_trend = trend_analysis['latency_trend']
        assert latency_trend['direction'] in ['increasing', 'decreasing', 'stable'], "Should classify trend direction"
        assert 'slope' in latency_trend, "Should calculate trend slope"
        assert 'confidence' in latency_trend, "Should provide trend confidence"
    
    def test_bottleneck_identification(self, analyzer):
        """Test performance bottleneck identification."""
        performance_data = {
            'gcp': {
                'cloud_sql_latency_ms': [150, 160, 170, 165, 175],  # High latency
                'secret_manager_latency_ms': [20, 22, 21, 23, 19],  # Normal
                'network_latency_ms': [5, 6, 5, 7, 6]  # Normal
            },
            'hft': {
                'order_execution_latency_ms': [45, 48, 47, 46, 49],  # Normal
                'ml_prediction_throughput': [800, 750, 780, 760, 770],  # Below target
                'emergency_stop_latency_ms': [8, 9, 8, 10, 9]  # Normal
            },
            'efficiency': {
                'memory_usage_mb': [2500, 2600, 2700, 2650, 2750],  # High usage
                'cpu_utilization': [95, 97, 96, 98, 95]  # Very high
            }
        }
        
        # This will initially fail - implementation needed
        bottlenecks = analyzer.identify_performance_bottlenecks(performance_data)
        
        assert bottlenecks is not None, "Should identify bottlenecks"
        assert len(bottlenecks) > 0, "Should find performance bottlenecks"
        
        # Validate bottleneck identification
        bottleneck_types = [b['type'] for b in bottlenecks]
        assert 'database_latency' in bottleneck_types, "Should identify database latency bottleneck"
        assert 'ml_throughput' in bottleneck_types, "Should identify ML throughput bottleneck"
        assert 'resource_usage' in bottleneck_types, "Should identify resource usage bottleneck"
    
    def test_performance_score_calculation(self, analyzer, sample_test_suites):
        """Test performance score calculation."""
        # This will initially fail - implementation needed
        performance_scores = analyzer.calculate_performance_scores(sample_test_suites)
        
        assert performance_scores is not None, "Should calculate performance scores"
        assert len(performance_scores) > 0, "Should have performance scores"
        
        # Validate score categories
        for category in ['gcp', 'hft', 'stress', 'baseline']:
            assert category in performance_scores, f"Should have score for {category} category"
            score = performance_scores[category]
            assert 0 <= score <= 100, f"{category} score {score} should be between 0 and 100"


class TestVisualizationDataGeneration:
    """Test visualization data generation."""
    
    @pytest.fixture
    def visualizer(self):
        """Provide visualization data generator."""
        return VisualizationDataGenerator()
    
    def test_performance_timeline_data(self, visualizer, sample_test_suites):
        """Test performance timeline data generation."""
        # This will initially fail - implementation needed
        timeline_data = visualizer.create_performance_timeline_data(sample_test_suites)
        
        assert timeline_data is not None, "Should create timeline data"
        assert 'timestamps' in timeline_data, "Should include timestamps"
        assert 'metrics' in timeline_data, "Should include metrics"
        assert 'categories' in timeline_data, "Should include categories"
        
        # Validate data structure
        assert len(timeline_data['timestamps']) > 0, "Should have timeline data points"
        assert len(timeline_data['metrics']) > 0, "Should have metrics data"
    
    def test_category_comparison_data(self, visualizer):
        """Test category comparison chart data."""
        performance_scores = {
            'gcp': 85.0,
            'hft': 92.0,
            'stress': 78.0,
            'baseline': 88.0,
            'efficiency': 82.0
        }
        
        # This will initially fail - implementation needed
        comparison_data = visualizer.create_category_comparison_data(performance_scores)
        
        assert comparison_data is not None, "Should create comparison data"
        assert 'categories' in comparison_data, "Should include categories"
        assert 'scores' in comparison_data, "Should include scores"
        assert 'chart_type' in comparison_data, "Should specify chart type"
        
        # Validate data consistency
        assert len(comparison_data['categories']) == len(performance_scores), "Should match input categories"
        assert len(comparison_data['scores']) == len(performance_scores), "Should match input scores"


class TestComprehensiveReportingFramework:
    """Test comprehensive reporting framework integration."""
    
    @pytest.fixture
    def reporting_framework(self):
        """Provide performance reporting framework."""
        return PerformanceReportingFramework()
    
    def test_comprehensive_report_generation(self, mock_environment_variables, reporting_framework, performance_tracker):
        """Test comprehensive performance report generation."""
        with patch.dict('os.environ', mock_environment_variables):
            print("\n=== COMPREHENSIVE PERFORMANCE REPORTING TEST ===")
            
            performance_tracker.start_timing('comprehensive_report_generation')
            
            try:
                # This will initially fail - implementation needed
                comprehensive_report = reporting_framework.generate_comprehensive_report(
                    "Phase 7.2 Comprehensive Performance Report"
                )
                
                assert comprehensive_report is not None, "Should generate comprehensive report"
                assert comprehensive_report.report_title == "Phase 7.2 Comprehensive Performance Report", "Should have correct title"
                assert len(comprehensive_report.test_suites) > 0, "Should include test suites"
                assert comprehensive_report.executive_summary is not None, "Should have executive summary"
                assert comprehensive_report.detailed_analysis is not None, "Should have detailed analysis"
                assert len(comprehensive_report.recommendations) > 0, "Should have recommendations"
                
                print(f"   ✓ Generated report with {len(comprehensive_report.test_suites)} test suites")
                print(f"   ✓ Executive summary created")
                print(f"   ✓ {len(comprehensive_report.recommendations)} recommendations generated")
                
                # Test report export
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Export as JSON
                    json_path = os.path.join(temp_dir, "performance_report.json")
                    json_export = reporting_framework.export_report(comprehensive_report, 'json', json_path)
                    
                    assert os.path.exists(json_export), "Should create JSON export file"
                    
                    # Validate JSON export
                    with open(json_export, 'r') as f:
                        json_data = json.load(f)
                        assert 'report_id' in json_data, "JSON export should contain report ID"
                        assert 'test_suites' in json_data, "JSON export should contain test suites"
                    
                    print(f"   ✓ JSON export created: {os.path.basename(json_export)}")
                    
                    # Export as HTML
                    html_path = os.path.join(temp_dir, "performance_report.html")
                    html_export = reporting_framework.export_report(comprehensive_report, 'html', html_path)
                    
                    assert os.path.exists(html_export), "Should create HTML export file"
                    print(f"   ✓ HTML export created: {os.path.basename(html_export)}")
                
                # Test dashboard creation
                dashboard_data = reporting_framework.create_performance_dashboard(comprehensive_report)
                
                assert dashboard_data is not None, "Should create dashboard data"
                assert 'overview' in dashboard_data, "Dashboard should include overview"
                assert 'charts' in dashboard_data, "Dashboard should include chart data"
                assert 'metrics' in dashboard_data, "Dashboard should include metrics"
                
                print(f"   ✓ Dashboard data created with {len(dashboard_data.get('charts', []))} charts")
                
                performance_tracker.end_timing('comprehensive_report_generation')
                
                # Validate report quality
                print("\n=== REPORT VALIDATION ===")
                
                # Check test suite coverage
                test_categories = {suite.test_category for suite in comprehensive_report.test_suites}
                expected_categories = {'gcp', 'hft', 'regression', 'stress', 'monitoring', 'baseline', 'load', 'efficiency'}
                coverage = len(test_categories.intersection(expected_categories)) / len(expected_categories)
                
                print(f"✓ Test category coverage: {coverage:.1%}")
                assert coverage >= 0.5, f"Test category coverage {coverage:.1%} below 50%"
                
                # Check executive summary completeness
                summary_keys = set(comprehensive_report.executive_summary.keys())
                required_summary_keys = {'overall_success_rate', 'total_tests_executed', 'performance_highlights'}
                summary_completeness = len(summary_keys.intersection(required_summary_keys)) / len(required_summary_keys)
                
                print(f"✓ Executive summary completeness: {summary_completeness:.1%}")
                assert summary_completeness >= 0.8, f"Executive summary completeness {summary_completeness:.1%} below 80%"
                
                # Check recommendations quality
                recommendations_count = len(comprehensive_report.recommendations)
                print(f"✓ Recommendations generated: {recommendations_count}")
                assert recommendations_count >= 3, f"Should generate at least 3 recommendations, got {recommendations_count}"
                
                print("\n=== COMPREHENSIVE PERFORMANCE REPORTING COMPLETE ✓ ===")
                
                return comprehensive_report
                
            except Exception as e:
                print(f"   ✗ Report generation error: {e}")
                raise


# Mark all tests as performance reporting tests
pytestmark = [pytest.mark.performance, pytest.mark.reporting]