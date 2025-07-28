"""
Analysis metrics collection for monitoring backtesting performance.

This module provides comprehensive analysis metrics including:
- Backtest execution time and success rates
- Analysis quality metrics (result completeness, data coverage)
- Computational efficiency (CPU usage, memory consumption)
- Report generation performance
"""

import time
import psutil
import datetime
import structlog
from typing import Dict, Any, Optional
from decimal import Decimal

from .base import MetricsCollector, MetricsRegistry

logger = structlog.get_logger(__name__)


class AnalysisMetricsCollector(MetricsCollector):
    """
    Collector for analysis and backtesting-related metrics.
    
    This class tracks key performance indicators for backtesting operations,
    analysis quality, computational efficiency, and report generation.
    """
    
    def __init__(self, registry: MetricsRegistry):
        """
        Initialize the analysis metrics collector.
        
        Args:
            registry: MetricsRegistry instance for managing metrics
        """
        super().__init__(registry)
        
        # Backtest execution metrics
        self.backtest_execution_time_histogram = registry.get_histogram(
            "analysis_backtest_execution_time_seconds",
            "Distribution of backtest execution times in seconds",
            ["strategy", "dataset_size_bucket"],
            buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1200, 3600)
        )
        
        self.backtest_success_counter = registry.get_counter(
            "analysis_backtest_success_total",
            "Total number of successful backtests",
            ["strategy", "dataset_size_bucket"]
        )
        
        self.backtest_failure_counter = registry.get_counter(
            "analysis_backtest_failure_total",
            "Total number of failed backtests",
            ["strategy", "error_type"]
        )
        
        # Analysis quality metrics
        self.analysis_completeness_gauge = registry.get_gauge(
            "analysis_completeness_ratio",
            "Ratio of analysis completeness (0-1)",
            ["analysis_type"]
        )
        
        self.data_coverage_gauge = registry.get_gauge(
            "analysis_data_coverage_ratio",
            "Ratio of data coverage in analysis (0-1)",
            ["timeframe", "symbol"]
        )
        
        # Computational efficiency metrics
        self.cpu_usage_gauge = registry.get_gauge(
            "analysis_cpu_usage_percent",
            "CPU usage percentage during analysis operations",
            ["operation_type"]
        )
        
        self.memory_usage_gauge = registry.get_gauge(
            "analysis_memory_usage_bytes",
            "Memory usage in bytes during analysis operations",
            ["operation_type"]
        )
        
        self.disk_io_counter = registry.get_counter(
            "analysis_disk_io_bytes_total",
            "Total disk I/O bytes during analysis operations",
            ["operation", "direction"]
        )
        
        # Report generation metrics
        self.report_generation_time_histogram = registry.get_histogram(
            "analysis_report_generation_time_seconds",
            "Distribution of report generation times in seconds",
            ["report_type", "data_points_bucket"],
            buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 120, 300)
        )
        
        self.report_success_counter = registry.get_counter(
            "analysis_report_success_total",
            "Total number of successful report generations",
            ["report_type"]
        )
        
        self.report_failure_counter = registry.get_counter(
            "analysis_report_failure_total",
            "Total number of failed report generations",
            ["report_type", "error_type"]
        )
        
        # Internal tracking for active operations
        self._active_backtests: Dict[str, Dict[str, Any]] = {}
        self._active_reports: Dict[str, Dict[str, Any]] = {}
        self._cleanup_interval = 3600  # Clean up stale tracking every hour
        self._last_cleanup = time.time()
    
    def collect_metrics(self) -> None:
        """
        Collect and update all analysis metrics.
        
        This method gathers current system performance, backtest statistics,
        analysis quality metrics, and report generation data.
        """
        try:
            # Collect system performance metrics
            self._collect_system_metrics()
            
            # Collect backtest statistics
            backtest_stats = self._get_backtest_statistics()
            if backtest_stats:
                logger.debug("Collected backtest statistics", stats=backtest_stats)
            
            # Collect analysis quality metrics
            quality_metrics = self._get_analysis_quality_metrics()
            if quality_metrics:
                self.analysis_completeness_gauge.labels(
                    analysis_type="general"
                ).set(quality_metrics.get('completeness_ratio', 0))
                
                self.data_coverage_gauge.labels(
                    timeframe="daily",
                    symbol="ALL"
                ).set(quality_metrics.get('data_coverage_ratio', 0))
                
                logger.debug("Updated analysis quality metrics", metrics=quality_metrics)
            
            # Collect report statistics
            report_stats = self._get_report_statistics()
            if report_stats:
                logger.debug("Collected report statistics", stats=report_stats)
            
            # Cleanup stale tracking entries periodically
            current_time = time.time()
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup_stale_tracking()
                self._last_cleanup = current_time
            
        except Exception as e:
            logger.error("Error collecting analysis metrics", error=str(e))
            self._collection_errors += 1
            raise e
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics (CPU, memory, disk I/O)."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage_gauge.labels(operation_type="analysis").set(cpu_percent)
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.memory_usage_gauge.labels(operation_type="analysis").set(memory_info.used)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # Note: These are cumulative counters, so we increment by current values
                # In a real implementation, you'd track deltas
                self.disk_io_counter.labels(operation="analysis", direction="read").inc(
                    disk_io.read_bytes
                )
                self.disk_io_counter.labels(operation="analysis", direction="write").inc(
                    disk_io.write_bytes
                )
            
        except Exception as e:
            logger.warning("Failed to collect system metrics", error=str(e))
    
    def get_metric_definitions(self) -> Dict[str, str]:
        """
        Get metric definitions for analysis metrics.
        
        Returns:
            Dictionary mapping metric names to their descriptions
        """
        return {
            "analysis_backtest_execution_time_seconds": "Distribution of backtest execution times in seconds",
            "analysis_backtest_success_total": "Total number of successful backtests",
            "analysis_backtest_failure_total": "Total number of failed backtests",
            "analysis_completeness_ratio": "Ratio of analysis completeness (0-1)",
            "analysis_data_coverage_ratio": "Ratio of data coverage in analysis (0-1)",
            "analysis_cpu_usage_percent": "CPU usage percentage during analysis operations",
            "analysis_memory_usage_bytes": "Memory usage in bytes during analysis operations",
            "analysis_disk_io_bytes_total": "Total disk I/O bytes during analysis operations",
            "analysis_report_generation_time_seconds": "Distribution of report generation times in seconds",
            "analysis_report_success_total": "Total number of successful report generations",
            "analysis_report_failure_total": "Total number of failed report generations"
        }
    
    def record_backtest_start(self, backtest_id: str, strategy: str, dataset_size: int) -> None:
        """
        Record the start of a backtest operation.
        
        Args:
            backtest_id: Unique identifier for the backtest
            strategy: Name of the strategy being tested
            dataset_size: Size of the dataset being processed
        """
        self._active_backtests[backtest_id] = {
            'start_time': time.time(),
            'strategy': strategy,
            'dataset_size': dataset_size
        }
        
        logger.debug(
            "Started tracking backtest",
            backtest_id=backtest_id,
            strategy=strategy,
            dataset_size=dataset_size
        )
    
    def record_backtest_success(self, backtest_id: str, results: Dict[str, Any]) -> None:
        """
        Record a successful backtest completion.
        
        Args:
            backtest_id: Unique identifier for the backtest
            results: Dictionary containing backtest results
        """
        if backtest_id not in self._active_backtests:
            logger.warning("Backtest not found in active tracking", backtest_id=backtest_id)
            return
        
        backtest_info = self._active_backtests.pop(backtest_id)
        execution_time = time.time() - backtest_info['start_time']
        strategy = backtest_info['strategy']
        dataset_size = backtest_info['dataset_size']
        
        # Determine dataset size bucket
        dataset_bucket = self._get_dataset_size_bucket(dataset_size)
        
        # Update metrics
        self.backtest_execution_time_histogram.labels(
            strategy=strategy,
            dataset_size_bucket=dataset_bucket
        ).observe(execution_time)
        
        self.backtest_success_counter.labels(
            strategy=strategy,
            dataset_size_bucket=dataset_bucket
        ).inc()
        
        logger.info(
            "Recorded successful backtest",
            backtest_id=backtest_id,
            strategy=strategy,
            execution_time=execution_time,
            results_summary=self._summarize_results(results)
        )
    
    def record_backtest_failure(self, backtest_id: str, error_type: str, error_message: str) -> None:
        """
        Record a failed backtest.
        
        Args:
            backtest_id: Unique identifier for the backtest
            error_type: Type of error that occurred
            error_message: Detailed error message
        """
        if backtest_id not in self._active_backtests:
            logger.warning("Backtest not found in active tracking", backtest_id=backtest_id)
            return
        
        backtest_info = self._active_backtests.pop(backtest_id)
        execution_time = time.time() - backtest_info['start_time']
        strategy = backtest_info['strategy']
        
        # Update failure metrics
        self.backtest_failure_counter.labels(
            strategy=strategy,
            error_type=error_type
        ).inc()
        
        logger.error(
            "Recorded failed backtest",
            backtest_id=backtest_id,
            strategy=strategy,
            execution_time=execution_time,
            error_type=error_type,
            error_message=error_message
        )
    
    def record_analysis_quality(self, quality_metrics: Dict[str, Any]) -> None:
        """
        Record analysis quality metrics.
        
        Args:
            quality_metrics: Dictionary containing quality measurements
        """
        completeness_ratio = quality_metrics.get('completeness_ratio', 0)
        data_coverage_ratio = quality_metrics.get('data_coverage_ratio', 0)
        
        self.analysis_completeness_gauge.labels(
            analysis_type="general"
        ).set(completeness_ratio)
        
        self.data_coverage_gauge.labels(
            timeframe="daily",
            symbol="ALL"
        ).set(data_coverage_ratio)
        
        logger.debug(
            "Recorded analysis quality metrics",
            completeness_ratio=completeness_ratio,
            data_coverage_ratio=data_coverage_ratio
        )
    
    def record_report_generation_start(self, report_id: str, report_type: str, data_points: int) -> None:
        """
        Record the start of report generation.
        
        Args:
            report_id: Unique identifier for the report
            report_type: Type of report being generated
            data_points: Number of data points to process
        """
        self._active_reports[report_id] = {
            'start_time': time.time(),
            'report_type': report_type,
            'data_points': data_points
        }
        
        logger.debug(
            "Started tracking report generation",
            report_id=report_id,
            report_type=report_type,
            data_points=data_points
        )
    
    def record_report_generation_success(self, report_id: str, output_size: int) -> None:
        """
        Record successful report generation.
        
        Args:
            report_id: Unique identifier for the report
            output_size: Size of generated report in bytes
        """
        if report_id not in self._active_reports:
            logger.warning("Report not found in active tracking", report_id=report_id)
            return
        
        report_info = self._active_reports.pop(report_id)
        generation_time = time.time() - report_info['start_time']
        report_type = report_info['report_type']
        data_points = report_info['data_points']
        
        # Determine data points bucket
        data_bucket = self._get_data_points_bucket(data_points)
        
        # Update metrics
        self.report_generation_time_histogram.labels(
            report_type=report_type,
            data_points_bucket=data_bucket
        ).observe(generation_time)
        
        self.report_success_counter.labels(
            report_type=report_type
        ).inc()
        
        logger.info(
            "Recorded successful report generation",
            report_id=report_id,
            report_type=report_type,
            generation_time=generation_time,
            output_size=output_size
        )
    
    def record_report_generation_failure(self, report_id: str, error_type: str, error_message: str) -> None:
        """
        Record failed report generation.
        
        Args:
            report_id: Unique identifier for the report
            error_type: Type of error that occurred
            error_message: Detailed error message
        """
        if report_id not in self._active_reports:
            logger.warning("Report not found in active tracking", report_id=report_id)
            return
        
        report_info = self._active_reports.pop(report_id)
        generation_time = time.time() - report_info['start_time']
        report_type = report_info['report_type']
        
        # Update failure metrics
        self.report_failure_counter.labels(
            report_type=report_type,
            error_type=error_type
        ).inc()
        
        logger.error(
            "Recorded failed report generation",
            report_id=report_id,
            report_type=report_type,
            generation_time=generation_time,
            error_type=error_type,
            error_message=error_message
        )
    
    def _get_backtest_statistics(self) -> Dict[str, Any]:
        """
        Get backtest statistics from activity logs.
        
        Returns:
            Dictionary with backtest statistics
        """
        try:
            from src.logging.activity_logger import ActivityLogger
            
            activity_logger = ActivityLogger()
            
            # Query backtest activities for the last 24 hours
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(hours=24)
            
            backtest_activities = activity_logger.query_activities(
                activity_type="BACKTEST_EXECUTION",
                start_time=start_time,
                end_time=end_time
            )
            
            total_backtests = len(backtest_activities)
            successful_backtests = sum(
                1 for activity in backtest_activities 
                if activity.get('metadata', {}).get('success', False)
            )
            failed_backtests = total_backtests - successful_backtests
            
            # Calculate average execution time
            execution_times = [
                activity.get('metadata', {}).get('execution_time', 0)
                for activity in backtest_activities
                if activity.get('metadata', {}).get('execution_time')
            ]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            return {
                'total_backtests': total_backtests,
                'successful_backtests': successful_backtests,
                'failed_backtests': failed_backtests,
                'avg_execution_time': avg_execution_time
            }
            
        except Exception as e:
            logger.warning("Failed to get backtest statistics", error=str(e))
            return {
                'total_backtests': 0,
                'successful_backtests': 0,
                'failed_backtests': 0,
                'avg_execution_time': 0
            }
    
    def _get_analysis_quality_metrics(self) -> Dict[str, Any]:
        """
        Get analysis quality metrics from activity logs.
        
        Returns:
            Dictionary with analysis quality metrics
        """
        try:
            from src.logging.activity_logger import ActivityLogger
            
            activity_logger = ActivityLogger()
            
            # Query analysis quality check activities for the last 24 hours
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(hours=24)
            
            quality_activities = activity_logger.query_activities(
                activity_type="ANALYSIS_QUALITY_CHECK",
                start_time=start_time,
                end_time=end_time
            )
            
            if not quality_activities:
                return {'completeness_ratio': 0, 'data_coverage_ratio': 0}
            
            # Calculate average quality metrics
            completeness_ratios = [
                activity.get('metadata', {}).get('completeness_ratio', 0)
                for activity in quality_activities
            ]
            data_coverage_ratios = [
                activity.get('metadata', {}).get('data_coverage_ratio', 0)
                for activity in quality_activities
            ]
            
            avg_completeness = sum(completeness_ratios) / len(completeness_ratios)
            avg_data_coverage = sum(data_coverage_ratios) / len(data_coverage_ratios)
            
            return {
                'completeness_ratio': avg_completeness,
                'data_coverage_ratio': avg_data_coverage
            }
            
        except Exception as e:
            logger.warning("Failed to get analysis quality metrics", error=str(e))
            return {'completeness_ratio': 0, 'data_coverage_ratio': 0}
    
    def _get_report_statistics(self) -> Dict[str, Any]:
        """
        Get report generation statistics from activity logs.
        
        Returns:
            Dictionary with report statistics
        """
        try:
            from src.logging.activity_logger import ActivityLogger
            
            activity_logger = ActivityLogger()
            
            # Query report generation activities for the last 24 hours
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(hours=24)
            
            report_activities = activity_logger.query_activities(
                activity_type="REPORT_GENERATION",
                start_time=start_time,
                end_time=end_time
            )
            
            total_reports = len(report_activities)
            successful_reports = sum(
                1 for activity in report_activities 
                if activity.get('metadata', {}).get('success', False)
            )
            failed_reports = total_reports - successful_reports
            
            # Calculate average generation time
            generation_times = [
                activity.get('metadata', {}).get('generation_time', 0)
                for activity in report_activities
                if activity.get('metadata', {}).get('generation_time')
            ]
            avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
            
            return {
                'total_reports': total_reports,
                'successful_reports': successful_reports,
                'failed_reports': failed_reports,
                'avg_generation_time': avg_generation_time
            }
            
        except Exception as e:
            logger.warning("Failed to get report statistics", error=str(e))
            return {
                'total_reports': 0,
                'successful_reports': 0,
                'failed_reports': 0,
                'avg_generation_time': 0
            }
    
    def _get_dataset_size_bucket(self, size: int) -> str:
        """
        Get dataset size bucket for metrics labeling.
        
        Args:
            size: Dataset size
            
        Returns:
            String representing the size bucket
        """
        if size < 1000:
            return "small"
        elif size < 10000:
            return "medium"
        elif size < 100000:
            return "large"
        else:
            return "xlarge"
    
    def _get_data_points_bucket(self, points: int) -> str:
        """
        Get data points bucket for metrics labeling.
        
        Args:
            points: Number of data points
            
        Returns:
            String representing the data points bucket
        """
        if points < 100:
            return "tiny"
        elif points < 1000:
            return "small"
        elif points < 10000:
            return "medium"
        elif points < 100000:
            return "large"
        else:
            return "xlarge"
    
    def _summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of backtest results for logging.
        
        Args:
            results: Full results dictionary
            
        Returns:
            Summarized results for logging
        """
        return {
            'total_trades': results.get('total_trades', 0),
            'profitable_trades': results.get('profitable_trades', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0)
        }
    
    def _cleanup_stale_tracking(self) -> None:
        """Clean up stale tracking entries older than 1 hour."""
        current_time = time.time()
        stale_threshold = 3600  # 1 hour
        
        # Cleanup stale backtests
        stale_backtests = [
            backtest_id for backtest_id, info in self._active_backtests.items()
            if current_time - info['start_time'] > stale_threshold
        ]
        
        for backtest_id in stale_backtests:
            logger.warning("Cleaning up stale backtest tracking", backtest_id=backtest_id)
            del self._active_backtests[backtest_id]
        
        # Cleanup stale reports
        stale_reports = [
            report_id for report_id, info in self._active_reports.items()
            if current_time - info['start_time'] > stale_threshold
        ]
        
        for report_id in stale_reports:
            logger.warning("Cleaning up stale report tracking", report_id=report_id)
            del self._active_reports[report_id]
    
    def is_healthy(self) -> bool:
        """
        Check if the analysis metrics collector is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to collect basic system metrics to verify health
            cpu_percent = psutil.cpu_percent(interval=None)
            return cpu_percent >= 0  # Basic sanity check
        except Exception as e:
            logger.error("Health check failed for analysis metrics collector", error=str(e))
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information for the analysis metrics collector.
        
        Returns:
            Dictionary with collector status and analysis-specific information
        """
        base_status = super().get_status()
        
        # Add analysis-specific status information
        backtest_stats = self._get_backtest_statistics()
        analysis_quality = self._get_analysis_quality_metrics()
        report_stats = self._get_report_statistics()
        
        base_status.update({
            'backtest_stats': backtest_stats,
            'analysis_quality': analysis_quality,
            'report_stats': report_stats,
            'active_backtests': len(self._active_backtests),
            'active_reports': len(self._active_reports)
        })
        
        return base_status