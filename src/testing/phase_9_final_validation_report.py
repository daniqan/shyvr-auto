"""
Phase 9 Final Validation Report Generator

This module generates the comprehensive final validation report for Phase 9 - Final Integration and Validation.
It orchestrates all validation components and produces a complete assessment of the ensemble deployment.
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import subprocess

# Import validation modules
from .end_to_end_validation import run_phase_9_validation, EndToEndValidationReport
from .ensemble_performance_validation import run_phase_9_2_performance_validation, EnsemblePerformanceReport
from .ensemble_safety_validation import run_phase_9_3_safety_validation, EnsembleSafetyReport

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase9Status(Enum):
    """Overall Phase 9 validation status"""
    SUCCESS = "success"
    WARNING = "warning"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class ValidationSummary:
    """Summary of validation results"""
    category: str
    status: str
    tests_passed: int
    tests_failed: int
    tests_total: int
    execution_time: float
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class Phase9FinalValidationReport:
    """Complete Phase 9 final validation report"""
    validation_id: str
    timestamp: str
    phase: str
    overall_status: Phase9Status
    validation_duration: float
    
    # Validation summaries
    end_to_end_summary: ValidationSummary
    performance_summary: ValidationSummary
    safety_summary: ValidationSummary
    
    # Detailed reports
    end_to_end_report: Optional[EndToEndValidationReport] = None
    performance_report: Optional[EnsemblePerformanceReport] = None
    safety_report: Optional[EnsembleSafetyReport] = None
    
    # Overall metrics and recommendations
    key_achievements: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    performance_benchmarks: Dict[str, Any] = field(default_factory=dict)
    safety_validations: Dict[str, Any] = field(default_factory=dict)
    deployment_readiness: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # System information
    system_info: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)


class Phase9FinalValidator:
    """
    Orchestrates all Phase 9 validation components and generates final report
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Phase 9 final validator."""
        self.config = config or {}
        self.project_root = self.config.get("project_root", "/Users/kendo/daniqan/shyvrai-rlte")
        self.validation_start_time = time.time()
    
    async def run_complete_phase_9_validation(self) -> Phase9FinalValidationReport:
        """Run complete Phase 9 validation and generate final report."""
        logger.info("="*80)
        logger.info("STARTING PHASE 9 - FINAL INTEGRATION AND VALIDATION")
        logger.info("="*80)
        
        validation_id = f"phase_9_final_{int(time.time())}"
        
        try:
            # Initialize reports
            end_to_end_report = None
            performance_report = None
            safety_report = None
            
            # 9.1 - End-to-End Testing
            logger.info("Phase 9.1 - Running End-to-End Testing")
            try:
                end_to_end_report = await run_phase_9_validation()
                logger.info("✓ End-to-End Testing completed")
            except Exception as e:
                logger.error(f"✗ End-to-End Testing failed: {e}")
            
            # 9.2 - Performance Validation
            logger.info("Phase 9.2 - Running Performance Validation")
            try:
                performance_report = await run_phase_9_2_performance_validation()
                logger.info("✓ Performance Validation completed")
            except Exception as e:
                logger.error(f"✗ Performance Validation failed: {e}")
            
            # 9.3 - Safety Validation
            logger.info("Phase 9.3 - Running Safety Validation")
            try:
                safety_report = await run_phase_9_3_safety_validation()
                logger.info("✓ Safety Validation completed")
            except Exception as e:
                logger.error(f"✗ Safety Validation failed: {e}")
            
            # Generate summaries
            end_to_end_summary = self._create_end_to_end_summary(end_to_end_report)
            performance_summary = self._create_performance_summary(performance_report)
            safety_summary = self._create_safety_summary(safety_report)
            
            # Determine overall status
            overall_status = self._determine_overall_status(
                end_to_end_summary, performance_summary, safety_summary
            )
            
            # Generate key achievements and issues
            key_achievements = self._extract_key_achievements(
                end_to_end_report, performance_report, safety_report
            )
            critical_issues = self._extract_critical_issues(
                end_to_end_report, performance_report, safety_report
            )
            
            # Generate performance benchmarks
            performance_benchmarks = self._extract_performance_benchmarks(performance_report)
            
            # Generate safety validations
            safety_validations = self._extract_safety_validations(safety_report)
            
            # Generate deployment readiness assessment
            deployment_readiness = self._assess_deployment_readiness(
                end_to_end_summary, performance_summary, safety_summary
            )
            
            # Compile recommendations
            recommendations = self._compile_recommendations(
                end_to_end_report, performance_report, safety_report
            )
            
            # Collect system information
            system_info = await self._collect_system_info()
            environment_info = await self._collect_environment_info()
            
            validation_duration = time.time() - self.validation_start_time
            
            # Create final report
            final_report = Phase9FinalValidationReport(
                validation_id=validation_id,
                timestamp=datetime.now().isoformat(),
                phase="Phase 9 - Final Integration and Validation",
                overall_status=overall_status,
                validation_duration=validation_duration,
                
                end_to_end_summary=end_to_end_summary,
                performance_summary=performance_summary,
                safety_summary=safety_summary,
                
                end_to_end_report=end_to_end_report,
                performance_report=performance_report,
                safety_report=safety_report,
                
                key_achievements=key_achievements,
                critical_issues=critical_issues,
                performance_benchmarks=performance_benchmarks,
                safety_validations=safety_validations,
                deployment_readiness=deployment_readiness,
                recommendations=recommendations,
                
                system_info=system_info,
                environment_info=environment_info
            )
            
            logger.info("="*80)
            logger.info("PHASE 9 FINAL VALIDATION COMPLETED")
            logger.info(f"Overall Status: {overall_status.value.upper()}")
            logger.info(f"Validation Duration: {validation_duration:.2f}s")
            logger.info("="*80)
            
            return final_report
        
        except Exception as e:
            logger.error(f"Phase 9 validation failed with error: {e}")
            
            # Create error report
            return Phase9FinalValidationReport(
                validation_id=validation_id,
                timestamp=datetime.now().isoformat(),
                phase="Phase 9 - Final Integration and Validation",
                overall_status=Phase9Status.ERROR,
                validation_duration=time.time() - self.validation_start_time,
                
                end_to_end_summary=ValidationSummary("End-to-End", "error", 0, 0, 0, 0.0),
                performance_summary=ValidationSummary("Performance", "error", 0, 0, 0, 0.0),
                safety_summary=ValidationSummary("Safety", "error", 0, 0, 0, 0.0),
                
                critical_issues=[f"Phase 9 validation failed: {str(e)}"],
                recommendations=["Investigate and fix critical validation errors before proceeding"],
                system_info={},
                environment_info={}
            )
    
    def _create_end_to_end_summary(self, report: Optional[EndToEndValidationReport]) -> ValidationSummary:
        """Create summary for end-to-end testing."""
        if not report:
            return ValidationSummary("End-to-End", "error", 0, 0, 0, 0.0)
        
        status = "success" if report.overall_success else "failure"
        
        key_metrics = {
            "deployment_scenarios_tested": len(report.test_results),
            "environment_configurations_validated": 2,  # development, production
            "integration_points_tested": report.total_tests
        }
        
        recommendations = []
        if not report.overall_success:
            recommendations.append("Fix failing end-to-end tests before production deployment")
            recommendations.extend([
                result.error_message for result in report.test_results
                if result.error_message
            ][:3])  # Top 3 errors
        
        return ValidationSummary(
            category="End-to-End",
            status=status,
            tests_passed=report.passed_tests,
            tests_failed=report.failed_tests,
            tests_total=report.total_tests,
            execution_time=report.execution_time,
            key_metrics=key_metrics,
            recommendations=recommendations
        )
    
    def _create_performance_summary(self, report: Optional[EnsemblePerformanceReport]) -> ValidationSummary:
        """Create summary for performance validation."""
        if not report:
            return ValidationSummary("Performance", "error", 0, 0, 0, 0.0)
        
        status_map = {
            "passed": "success",
            "warning": "warning", 
            "failed": "failure",
            "error": "error"
        }
        status = status_map.get(report.overall_status.value, "error")
        
        # Extract key performance metrics
        key_metrics = {}
        for result in report.test_results:
            for metric in result.metrics:
                if metric.name in ["total_ensemble_memory", "average_latency", "average_cpu_utilization", "average_throughput"]:
                    key_metrics[metric.name] = {
                        "value": metric.value,
                        "unit": metric.unit,
                        "threshold": metric.threshold,
                        "passed": metric.status.value == "passed"
                    }
        
        return ValidationSummary(
            category="Performance", 
            status=status,
            tests_passed=report.passed_tests,
            tests_failed=report.failed_tests,
            tests_total=report.total_tests,
            execution_time=report.execution_time,
            key_metrics=key_metrics,
            recommendations=report.recommendations
        )
    
    def _create_safety_summary(self, report: Optional[EnsembleSafetyReport]) -> ValidationSummary:
        """Create summary for safety validation."""
        if not report:
            return ValidationSummary("Safety", "error", 0, 0, 0, 0.0)
        
        status_map = {
            "passed": "success",
            "warning": "warning",
            "failed": "failure", 
            "error": "error",
            "critical": "failure"
        }
        status = status_map.get(report.overall_status.value, "error")
        
        key_metrics = {
            "safety_systems_validated": len(report.safety_systems_validated),
            "emergency_procedures_tested": len(report.emergency_procedures_tested),
            "critical_failures": report.critical_tests
        }
        
        return ValidationSummary(
            category="Safety",
            status=status,
            tests_passed=report.passed_tests,
            tests_failed=report.failed_tests + report.critical_tests,
            tests_total=report.total_tests,
            execution_time=report.execution_time,
            key_metrics=key_metrics,
            recommendations=report.recommendations
        )
    
    def _determine_overall_status(self, e2e: ValidationSummary, perf: ValidationSummary, safety: ValidationSummary) -> Phase9Status:
        """Determine overall Phase 9 validation status."""
        # Critical: any safety failures or errors
        if safety.status in ["error", "failure"] or perf.status == "error" or e2e.status == "error":
            return Phase9Status.FAILURE
        
        # Warning: any warnings or performance issues
        if safety.status == "warning" or perf.status in ["warning", "failure"] or e2e.status == "failure":
            return Phase9Status.WARNING
        
        # Success: all passed
        if all(s.status == "success" for s in [e2e, perf, safety]):
            return Phase9Status.SUCCESS
        
        return Phase9Status.WARNING
    
    def _extract_key_achievements(self, e2e_report, perf_report, safety_report) -> List[str]:
        """Extract key achievements from validation reports."""
        achievements = []
        
        # End-to-end achievements
        if e2e_report and e2e_report.overall_success:
            achievements.append("✅ All deployment scenarios validated successfully")
            achievements.append("✅ Environment-based configuration working correctly")
            achievements.append("✅ LSTM-only development mode operational")
            achievements.append("✅ Full ensemble production mode operational")
        
        # Performance achievements
        if perf_report and perf_report.overall_status.value == "passed":
            achievements.append("✅ Ensemble memory usage within 8GB limit")
            achievements.append("✅ Ensemble latency below 100ms threshold")
            achievements.append("✅ CPU utilization within 80% limit")
            achievements.append("✅ Throughput exceeds 500 RPS requirement")
        
        # Safety achievements
        if safety_report and safety_report.overall_status.value in ["passed", "warning"]:
            achievements.append("✅ All safety systems integrated with ensemble")
            achievements.append("✅ Circuit breakers operational with ensemble")
            achievements.append("✅ Risk management validated with ensemble predictions")
            achievements.append("✅ Emergency stop procedures functional")
        
        return achievements
    
    def _extract_critical_issues(self, e2e_report, perf_report, safety_report) -> List[str]:
        """Extract critical issues from validation reports."""
        issues = []
        
        # End-to-end issues
        if e2e_report and not e2e_report.overall_success:
            issues.append("❌ End-to-end deployment validation failed")
            failed_tests = [r for r in e2e_report.test_results if r.status.value in ["failed", "error"]]
            for test in failed_tests[:3]:  # Top 3 failures
                if test.error_message:
                    issues.append(f"  • {test.test_name}: {test.error_message}")
        
        # Performance issues
        if perf_report and perf_report.overall_status.value in ["failed", "error"]:
            issues.append("❌ Performance validation failed")
            for result in perf_report.test_results:
                if result.status.value in ["failed", "error"]:
                    issues.append(f"  • {result.test_name}")
        
        # Safety issues
        if safety_report and safety_report.overall_status.value in ["failed", "error", "critical"]:
            issues.append("❌ Safety validation failed")
            if safety_report.critical_tests > 0:
                issues.append(f"  • {safety_report.critical_tests} critical safety failures detected")
        
        return issues
    
    def _extract_performance_benchmarks(self, perf_report) -> Dict[str, Any]:
        """Extract performance benchmarks from performance report."""
        if not perf_report:
            return {}
        
        benchmarks = {
            "memory_usage": {
                "target": "≤8GB",
                "status": "unknown"
            },
            "latency": {
                "target": "<100ms",
                "status": "unknown"
            },
            "cpu_utilization": {
                "target": "≤80%",
                "status": "unknown"
            },
            "throughput": {
                "target": ">500 RPS",
                "status": "unknown"
            }
        }
        
        # Extract actual values from test results
        for result in perf_report.test_results:
            for metric in result.metrics:
                if metric.name == "total_ensemble_memory":
                    benchmarks["memory_usage"]["actual"] = f"{metric.value:.1f}GB"
                    benchmarks["memory_usage"]["status"] = "passed" if metric.status.value == "passed" else "failed"
                elif metric.name == "average_latency":
                    benchmarks["latency"]["actual"] = f"{metric.value:.1f}ms"
                    benchmarks["latency"]["status"] = "passed" if metric.status.value == "passed" else "failed"
                elif metric.name == "average_cpu_utilization":
                    benchmarks["cpu_utilization"]["actual"] = f"{metric.value*100:.1f}%"
                    benchmarks["cpu_utilization"]["status"] = "passed" if metric.status.value == "passed" else "failed"
                elif metric.name == "average_throughput":
                    benchmarks["throughput"]["actual"] = f"{metric.value:.0f} RPS"
                    benchmarks["throughput"]["status"] = "passed" if metric.status.value == "passed" else "failed"
        
        return benchmarks
    
    def _extract_safety_validations(self, safety_report) -> Dict[str, Any]:
        """Extract safety validations from safety report."""
        if not safety_report:
            return {}
        
        return {
            "systems_validated": safety_report.safety_systems_validated,
            "procedures_tested": safety_report.emergency_procedures_tested,
            "critical_failures": safety_report.critical_tests,
            "overall_status": safety_report.overall_status.value
        }
    
    def _assess_deployment_readiness(self, e2e_summary, perf_summary, safety_summary) -> Dict[str, Any]:
        """Assess deployment readiness based on validation results."""
        readiness_score = 0
        max_score = 3
        
        if e2e_summary.status == "success":
            readiness_score += 1
        if perf_summary.status in ["success", "warning"]:
            readiness_score += 1
        if safety_summary.status in ["success", "warning"]:
            readiness_score += 1
        
        readiness_percentage = (readiness_score / max_score) * 100
        
        if readiness_percentage >= 100:
            readiness_status = "ready"
            readiness_message = "System is ready for production deployment"
        elif readiness_percentage >= 67:
            readiness_status = "ready_with_warnings"
            readiness_message = "System is ready for deployment with monitoring of warning conditions"
        elif readiness_percentage >= 33:
            readiness_status = "not_ready"
            readiness_message = "System requires fixes before production deployment"
        else:
            readiness_status = "critical_issues"
            readiness_message = "Critical issues must be resolved before any deployment"
        
        return {
            "status": readiness_status,
            "score": readiness_score,
            "percentage": readiness_percentage,
            "message": readiness_message,
            "e2e_ready": e2e_summary.status == "success",
            "performance_ready": perf_summary.status in ["success", "warning"],
            "safety_ready": safety_summary.status in ["success", "warning"]
        }
    
    def _compile_recommendations(self, e2e_report, perf_report, safety_report) -> List[str]:
        """Compile recommendations from all validation reports."""
        recommendations = []
        
        if e2e_report:
            for result in e2e_report.test_results:
                if hasattr(result, 'recommendations'):
                    recommendations.extend(getattr(result, 'recommendations', []))
        
        if perf_report:
            recommendations.extend(perf_report.recommendations)
        
        if safety_report:
            recommendations.extend(safety_report.recommendations)
        
        # Remove duplicates and add general recommendations
        recommendations = list(set(recommendations))
        
        # Add general Phase 9 recommendations
        recommendations.extend([
            "Monitor ensemble performance metrics closely in production",
            "Maintain regular safety system validation schedules",
            "Keep emergency procedures documentation updated",
            "Implement continuous monitoring for all validated metrics"
        ])
        
        return recommendations
    
    async def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        try:
            import psutil
            import platform
            
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information."""
        try:
            env_info = {
                "project_root": self.project_root,
                "environment": os.environ.get("ENVIRONMENT", "not_set"),
                "python_path": os.environ.get("PYTHONPATH", "not_set")
            }
            
            # Check if key files exist
            key_files = [
                "deploy/configs/environments.json",
                "src/ml_analysis/model_manager.py",
                "src/monitoring/transformer_monitoring_dashboard.py",
                "src/safety/trading_safety_manager.py"
            ]
            
            file_status = {}
            for file_path in key_files:
                full_path = os.path.join(self.project_root, file_path)
                file_status[file_path] = os.path.exists(full_path)
            
            env_info["key_files_exist"] = file_status
            
            return env_info
        except Exception as e:
            return {"error": str(e)}


def save_validation_report(report: Phase9FinalValidationReport, output_path: str):
    """Save validation report to file."""
    try:
        # Convert report to dict for JSON serialization
        report_dict = asdict(report)
        
        # Save JSON report
        json_path = output_path.replace('.json', '') + '.json'
        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {json_path}")
        
        # Save human-readable report
        txt_path = output_path.replace('.json', '') + '_summary.txt'
        with open(txt_path, 'w') as f:
            f.write(generate_human_readable_report(report))
        
        logger.info(f"Human-readable report saved to: {txt_path}")
        
        return json_path, txt_path
    
    except Exception as e:
        logger.error(f"Failed to save validation report: {e}")
        return None, None


def generate_human_readable_report(report: Phase9FinalValidationReport) -> str:
    """Generate human-readable validation report."""
    lines = []
    
    lines.append("="*80)
    lines.append("PHASE 9 - FINAL INTEGRATION AND VALIDATION REPORT")
    lines.append("="*80)
    lines.append(f"Validation ID: {report.validation_id}")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(f"Overall Status: {report.overall_status.value.upper()}")
    lines.append(f"Duration: {report.validation_duration:.2f}s")
    lines.append("")
    
    # Deployment readiness
    readiness = report.deployment_readiness
    lines.append("DEPLOYMENT READINESS ASSESSMENT")
    lines.append("-" * 40)
    lines.append(f"Status: {readiness.get('status', 'unknown').upper()}")
    lines.append(f"Readiness Score: {readiness.get('score', 0)}/3 ({readiness.get('percentage', 0):.0f}%)")
    lines.append(f"Assessment: {readiness.get('message', 'No assessment available')}")
    lines.append("")
    
    # Validation summaries
    lines.append("VALIDATION SUMMARIES")
    lines.append("-" * 40)
    
    for summary_name, summary in [
        ("End-to-End Testing", report.end_to_end_summary),
        ("Performance Validation", report.performance_summary),
        ("Safety Validation", report.safety_summary)
    ]:
        status_symbol = {"success": "✅", "warning": "⚠️", "failure": "❌", "error": "❌"}.get(summary.status, "❓")
        lines.append(f"{status_symbol} {summary_name}: {summary.status.upper()}")
        lines.append(f"   Tests: {summary.tests_passed}/{summary.tests_total} passed")
        lines.append(f"   Duration: {summary.execution_time:.2f}s")
        lines.append("")
    
    # Performance benchmarks
    if report.performance_benchmarks:
        lines.append("PERFORMANCE BENCHMARKS")
        lines.append("-" * 40)
        for benchmark, data in report.performance_benchmarks.items():
            status_symbol = "✅" if data.get("status") == "passed" else "❌"
            actual = data.get("actual", "N/A")
            target = data.get("target", "N/A")
            lines.append(f"{status_symbol} {benchmark.replace('_', ' ').title()}: {actual} (target: {target})")
        lines.append("")
    
    # Key achievements
    if report.key_achievements:
        lines.append("KEY ACHIEVEMENTS")
        lines.append("-" * 40)
        for achievement in report.key_achievements:
            lines.append(achievement)
        lines.append("")
    
    # Critical issues
    if report.critical_issues:
        lines.append("CRITICAL ISSUES")
        lines.append("-" * 40)
        for issue in report.critical_issues:
            lines.append(issue)
        lines.append("")
    
    # Recommendations
    if report.recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(report.recommendations[:10], 1):  # Top 10 recommendations
            lines.append(f"{i}. {rec}")
        lines.append("")
    
    lines.append("="*80)
    
    return "\n".join(lines)


async def run_complete_phase_9_validation(output_dir: Optional[str] = None) -> Phase9FinalValidationReport:
    """
    Run complete Phase 9 validation and generate final report.
    
    Args:
        output_dir: Directory to save validation reports
    
    Returns:
        Complete Phase 9 validation report
    """
    config = {
        "project_root": "/Users/kendo/daniqan/shyvrai-rlte"
    }
    
    validator = Phase9FinalValidator(config)
    report = await validator.run_complete_phase_9_validation()
    
    # Save reports if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"phase_9_validation_report_{int(time.time())}")
        save_validation_report(report, output_path)
    
    return report


if __name__ == "__main__":
    # Run complete Phase 9 validation when script is executed directly
    async def main():
        output_dir = "/Users/kendo/daniqan/shyvrai-rlte/reports"
        report = await run_complete_phase_9_validation(output_dir)
        
        # Print summary
        print(generate_human_readable_report(report))
        
        return report.overall_status == Phase9Status.SUCCESS
    
    success = asyncio.run(main())
    exit(0 if success else 1)