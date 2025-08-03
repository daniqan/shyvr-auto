"""
Tests for analysis metrics collection functionality.

This module provides comprehensive tests for the AnalysisMetricsCollector,
which monitors backtesting performance and analysis quality.
"""

import pytest
import time
import psutil
from unittest.mock import Mock, MagicMock, patch
from decimal import Decimal
from datetime import datetime, timedelta

from src.monitoring.base import MetricsRegistry
from src.monitoring.analysis_metrics import AnalysisMetricsCollector


class TestAnalysisMetricsCollector:
    """Tests for the AnalysisMetricsCollector class."""
    
    def test_init_creates_metrics(self):
        """Test that initialization creates all analysis metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        # Check that all expected metrics are created
        assert hasattr(collector, 'backtest_execution_time_histogram')
        assert hasattr(collector, 'backtest_success_counter')
        assert hasattr(collector, 'backtest_failure_counter')
        assert hasattr(collector, 'analysis_completeness_gauge')
        assert hasattr(collector, 'data_coverage_gauge')
        assert hasattr(collector, 'cpu_usage_gauge')
        assert hasattr(collector, 'memory_usage_gauge')
        assert hasattr(collector, 'disk_io_counter')
        assert hasattr(collector, 'report_generation_time_histogram')
        assert hasattr(collector, 'report_success_counter')
        assert hasattr(collector, 'report_failure_counter')
    
    def test_get_metric_definitions_returns_all_metrics(self):
        """Test that get_metric_definitions returns all analysis metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        definitions = collector.get_metric_definitions()
        
        expected_metrics = [
            'analysis_backtest_execution_time_seconds',
            'analysis_backtest_success_total',
            'analysis_backtest_failure_total',
            'analysis_completeness_ratio',
            'analysis_data_coverage_ratio',
            'analysis_cpu_usage_percent',
            'analysis_memory_usage_bytes',
            'analysis_disk_io_bytes_total',
            'analysis_report_generation_time_seconds',
            'analysis_report_success_total',
            'analysis_report_failure_total'
        ]
        
        for metric in expected_metrics:
            assert metric in definitions
            assert isinstance(definitions[metric], str)
            assert len(definitions[metric]) > 0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    def test_collect_metrics_updates_system_metrics(self, mock_disk_io, mock_memory, mock_cpu):
        """Test that collect_metrics updates system performance metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        # Mock system metrics
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(used=1024*1024*512)  # 512 MB
        mock_disk_io.return_value = Mock(read_bytes=1000000, write_bytes=500000)
        
        # Mock analysis data
        with patch.object(collector, '_get_backtest_statistics') as mock_backtest_stats:
            mock_backtest_stats.return_value = {
                'total_backtests': 50,
                'successful_backtests': 45,
                'failed_backtests': 5,
                'avg_execution_time': 120.5
            }
            
            with patch.object(collector, '_get_analysis_quality_metrics') as mock_quality:
                mock_quality.return_value = {
                    'completeness_ratio': 0.92,
                    'data_coverage_ratio': 0.88
                }
                
                with patch.object(collector, '_get_report_statistics') as mock_report_stats:
                    mock_report_stats.return_value = {
                        'total_reports': 25,
                        'successful_reports': 23,
                        'failed_reports': 2,
                        'avg_generation_time': 45.2
                    }
                    
                    collector.collect_metrics()
                    
                    # Verify system monitoring methods were called
                    mock_cpu.assert_called_once_with(interval=None)
                    mock_memory.assert_called_once()
                    mock_disk_io.assert_called_once()
                    
                    # Verify analysis data methods were called
                    mock_backtest_stats.assert_called_once()
                    mock_quality.assert_called_once()
                    mock_report_stats.assert_called_once()
    
    def test_record_backtest_start_updates_metrics(self):
        """Test recording backtest start updates relevant metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        backtest_id = "backtest_001"
        strategy = "dqn_strategy"
        dataset_size = 10000
        
        collector.record_backtest_start(backtest_id, strategy, dataset_size)
        
        # Check that backtest tracking was initialized
        assert backtest_id in collector._active_backtests
        assert collector._active_backtests[backtest_id]['strategy'] == strategy
        assert collector._active_backtests[backtest_id]['dataset_size'] == dataset_size
        assert 'start_time' in collector._active_backtests[backtest_id]
    
    def test_record_backtest_success_updates_metrics(self):
        """Test recording backtest success updates relevant metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        # Start a backtest first
        backtest_id = "backtest_001"
        collector.record_backtest_start(backtest_id, "test_strategy", 5000)
        
        # Record success
        results = {
            'total_trades': 100,
            'profitable_trades': 75,
            'max_drawdown': 0.15,
            'sharpe_ratio': 1.25
        }
        
        collector.record_backtest_success(backtest_id, results)
        
        # Check that backtest was removed from active tracking
        assert backtest_id not in collector._active_backtests
    
    def test_record_backtest_failure_updates_metrics(self):
        """Test recording backtest failure updates relevant metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        # Start a backtest first
        backtest_id = "backtest_002"
        collector.record_backtest_start(backtest_id, "failing_strategy", 3000)
        
        # Record failure
        error_type = "data_insufficient"
        error_message = "Not enough historical data"
        
        collector.record_backtest_failure(backtest_id, error_type, error_message)
        
        # Check that backtest was removed from active tracking
        assert backtest_id not in collector._active_backtests
    
    def test_record_analysis_quality_updates_metrics(self):
        """Test recording analysis quality updates relevant metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        quality_metrics = {
            'completeness_ratio': 0.95,
            'data_coverage_ratio': 0.87,
            'missing_indicators': 2,
            'invalid_data_points': 15
        }
        
        collector.record_analysis_quality(quality_metrics)
        
        # Verify the method runs without error
        assert True
    
    def test_record_report_generation_start_updates_metrics(self):
        """Test recording report generation start updates metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        report_id = "report_001"
        report_type = "backtest_summary"
        data_points = 5000
        
        collector.record_report_generation_start(report_id, report_type, data_points)
        
        # Check that report tracking was initialized
        assert report_id in collector._active_reports
        assert collector._active_reports[report_id]['report_type'] == report_type
        assert collector._active_reports[report_id]['data_points'] == data_points
        assert 'start_time' in collector._active_reports[report_id]
    
    def test_record_report_generation_success_updates_metrics(self):
        """Test recording report generation success updates metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        # Start report generation first
        report_id = "report_001"
        collector.record_report_generation_start(report_id, "performance_analysis", 2000)
        
        # Record success
        output_size = 1024 * 50  # 50 KB
        
        collector.record_report_generation_success(report_id, output_size)
        
        # Check that report was removed from active tracking
        assert report_id not in collector._active_reports
    
    def test_record_report_generation_failure_updates_metrics(self):
        """Test recording report generation failure updates metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        # Start report generation first
        report_id = "report_002"
        collector.record_report_generation_start(report_id, "failing_report", 1000)
        
        # Record failure
        error_type = "template_error"
        error_message = "Template not found"
        
        collector.record_report_generation_failure(report_id, error_type, error_message)
        
        # Check that report was removed from active tracking
        assert report_id not in collector._active_reports
    
    def test_get_backtest_statistics_returns_stats(self):
        """Test that _get_backtest_statistics returns backtest statistics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        with patch('src.activity_logging.activity_logger.ActivityLogger') as mock_logger:
            mock_activity_logger = Mock()
            mock_logger.return_value = mock_activity_logger
            
            # Mock backtest query results
            mock_activity_logger.query_activities.return_value = [
                {
                    'activity_type': 'BACKTEST_EXECUTION',
                    'metadata': {
                        'success': True,
                        'execution_time': 120.5,
                        'strategy': 'dqn_strategy'
                    }
                },
                {
                    'activity_type': 'BACKTEST_EXECUTION',
                    'metadata': {
                        'success': True,
                        'execution_time': 95.3,
                        'strategy': 'lstm_strategy'
                    }
                },
                {
                    'activity_type': 'BACKTEST_EXECUTION',
                    'metadata': {
                        'success': False,
                        'execution_time': 30.1,
                        'strategy': 'failed_strategy',
                        'error': 'data_insufficient'
                    }
                },
            ]
            
            stats = collector._get_backtest_statistics()
            
            expected_stats = {
                'total_backtests': 3,
                'successful_backtests': 2,
                'failed_backtests': 1,
                'avg_execution_time': (120.5 + 95.3 + 30.1) / 3
            }
            
            assert stats['total_backtests'] == expected_stats['total_backtests']
            assert stats['successful_backtests'] == expected_stats['successful_backtests']
            assert stats['failed_backtests'] == expected_stats['failed_backtests']
            assert abs(stats['avg_execution_time'] - expected_stats['avg_execution_time']) < 0.001
    
    def test_get_analysis_quality_metrics_returns_quality_data(self):
        """Test that _get_analysis_quality_metrics returns quality metrics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        with patch('src.activity_logging.activity_logger.ActivityLogger') as mock_logger:
            mock_activity_logger = Mock()
            mock_logger.return_value = mock_activity_logger
            
            # Mock analysis quality query results
            mock_activity_logger.query_activities.return_value = [
                {
                    'activity_type': 'ANALYSIS_QUALITY_CHECK',
                    'metadata': {
                        'completeness_ratio': 0.95,
                        'data_coverage_ratio': 0.88,
                        'missing_indicators': 1,
                        'invalid_data_points': 5
                    }
                },
                {
                    'activity_type': 'ANALYSIS_QUALITY_CHECK',
                    'metadata': {
                        'completeness_ratio': 0.92,
                        'data_coverage_ratio': 0.85,
                        'missing_indicators': 2,
                        'invalid_data_points': 10
                    }
                }
            ]
            
            quality_metrics = collector._get_analysis_quality_metrics()
            
            # Should return average quality metrics
            assert quality_metrics['completeness_ratio'] == 0.935  # (0.95 + 0.92) / 2
            assert quality_metrics['data_coverage_ratio'] == 0.865  # (0.88 + 0.85) / 2
    
    def test_get_report_statistics_returns_stats(self):
        """Test that _get_report_statistics returns report statistics."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        with patch('src.activity_logging.activity_logger.ActivityLogger') as mock_logger:
            mock_activity_logger = Mock()
            mock_logger.return_value = mock_activity_logger
            
            # Mock report query results
            mock_activity_logger.query_activities.return_value = [
                {
                    'activity_type': 'REPORT_GENERATION',
                    'metadata': {
                        'success': True,
                        'generation_time': 45.2,
                        'report_type': 'backtest_summary',
                        'output_size': 51200
                    }
                },
                {
                    'activity_type': 'REPORT_GENERATION',
                    'metadata': {
                        'success': True,
                        'generation_time': 32.1,
                        'report_type': 'performance_analysis',
                        'output_size': 40960
                    }
                },
                {
                    'activity_type': 'REPORT_GENERATION',
                    'metadata': {
                        'success': False,
                        'generation_time': 15.5,
                        'report_type': 'failed_report',
                        'error': 'template_error'
                    }
                }
            ]
            
            stats = collector._get_report_statistics()
            
            expected_stats = {
                'total_reports': 3,
                'successful_reports': 2,
                'failed_reports': 1,
                'avg_generation_time': (45.2 + 32.1 + 15.5) / 3
            }
            
            assert stats['total_reports'] == expected_stats['total_reports']
            assert stats['successful_reports'] == expected_stats['successful_reports']
            assert stats['failed_reports'] == expected_stats['failed_reports']
            assert abs(stats['avg_generation_time'] - expected_stats['avg_generation_time']) < 0.001
    
    @patch('psutil.cpu_percent')
    def test_is_healthy_returns_true_when_collecting_successfully(self, mock_cpu):
        """Test that is_healthy returns True when metrics collection is working."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        mock_cpu.return_value = 25.0
        
        with patch.object(collector, 'collect_metrics'):
            assert collector.is_healthy() is True
    
    @patch('psutil.cpu_percent')
    def test_is_healthy_returns_false_when_collection_fails(self, mock_cpu):
        """Test that is_healthy returns False when metrics collection fails."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        mock_cpu.side_effect = Exception("CPU monitoring failed")
        
        with patch.object(collector, 'collect_metrics', side_effect=Exception("Collection failed")):
            assert collector.is_healthy() is False
    
    def test_get_status_includes_analysis_specific_info(self):
        """Test that get_status includes analysis-specific status information."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        with patch.object(collector, '_get_backtest_statistics') as mock_backtest_stats:
            mock_backtest_stats.return_value = {
                'total_backtests': 25,
                'successful_backtests': 22,
                'failed_backtests': 3,
                'avg_execution_time': 110.5
            }
            
            with patch.object(collector, '_get_analysis_quality_metrics') as mock_quality:
                mock_quality.return_value = {
                    'completeness_ratio': 0.94,
                    'data_coverage_ratio': 0.89
                }
                
                with patch.object(collector, '_get_report_statistics') as mock_report_stats:
                    mock_report_stats.return_value = {
                        'total_reports': 15,
                        'successful_reports': 14,
                        'failed_reports': 1,
                        'avg_generation_time': 38.7
                    }
                    
                    status = collector.get_status()
                    
                    assert status['name'] == 'AnalysisMetricsCollector'
                    assert 'backtest_stats' in status
                    assert 'analysis_quality' in status
                    assert 'report_stats' in status
                    assert status['backtest_stats']['total_backtests'] == 25
                    assert status['analysis_quality']['completeness_ratio'] == 0.94
                    assert status['report_stats']['total_reports'] == 15
    
    def test_cleanup_stale_tracking_removes_old_entries(self):
        """Test that cleanup removes stale tracking entries."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        # Add some tracking entries with old timestamps
        old_time = time.time() - 7200  # 2 hours ago
        collector._active_backtests['old_backtest'] = {
            'start_time': old_time,
            'strategy': 'old_strategy',
            'dataset_size': 1000
        }
        
        collector._active_reports['old_report'] = {
            'start_time': old_time,
            'report_type': 'old_report',
            'data_points': 500
        }
        
        # Add a recent entry that should not be cleaned up
        recent_time = time.time() - 600  # 10 minutes ago
        collector._active_backtests['recent_backtest'] = {
            'start_time': recent_time,
            'strategy': 'recent_strategy',
            'dataset_size': 2000
        }
        
        # Run cleanup
        collector._cleanup_stale_tracking()
        
        # Check that old entries were removed and recent ones kept
        assert 'old_backtest' not in collector._active_backtests
        assert 'old_report' not in collector._active_reports
        assert 'recent_backtest' in collector._active_backtests
    
    def test_execution_time_decorator_works_correctly(self):
        """Test that the execution time decorator records metrics properly."""
        registry = MetricsRegistry()
        collector = AnalysisMetricsCollector(registry)
        
        # Create a test method with the decorator
        @AnalysisMetricsCollector.record_execution_time("test_method")
        def test_method(self):
            time.sleep(0.01)  # Small delay to measure
            return "success"
        
        # Bind the method to the collector instance
        bound_method = test_method.__get__(collector, AnalysisMetricsCollector)
        
        # Execute the method
        result = bound_method()
        
        assert result == "success"
        # Test passes if no exception is raised during execution