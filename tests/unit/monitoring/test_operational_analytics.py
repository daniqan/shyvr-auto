"""
Test suite for operational analytics system (Phase 6.2)

This test module follows TDD methodology with failing tests first,
then implementing real functionality to make them pass.
"""

import pytest
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, Mock, patch

# Set required environment variables for tests
os.environ['SECRET_KEY'] = 'test-secret-key-for-operational-analytics-tests'
os.environ['JWT_SECRET_KEY'] = 'test-jwt-secret-key-for-operational-analytics'

# Test the operational analytics system
class TestOperationalAnalyticsCore:
    """Test core operational analytics functionality."""
    
    def test_operational_analytics_init(self):
        """Test OperationalAnalytics initialization."""
        from src.monitoring.operational_analytics import OperationalAnalytics
        
        analytics = OperationalAnalytics()
        assert analytics.config is not None
        assert analytics.activity_logger is not None
        assert analytics.enhanced_logger is not None
        assert analytics.metrics_collectors is not None
        assert len(analytics.analytics_engines) == 4  # log, trading, compliance, system
    
    @patch('src.activity_logging.activity_logger.get_database_pool')
    def test_operational_analytics_start_stop(self, mock_db_pool):
        """Test analytics system start and stop."""
        from src.monitoring.operational_analytics import OperationalAnalytics
        
        # Mock database pool
        mock_db_pool.return_value = AsyncMock()
        
        analytics = OperationalAnalytics()
        
        # Test start
        assert not analytics.is_running
        asyncio.run(analytics.start())
        assert analytics.is_running
        
        # Test stop
        asyncio.run(analytics.stop())
        assert not analytics.is_running
    
    def test_log_analytics_engine_init(self):
        """Test LogAnalyticsEngine initialization."""
        from src.monitoring.operational_analytics import LogAnalyticsEngine
        
        engine = LogAnalyticsEngine()
        assert engine.time_windows == [
            timedelta(hours=1), timedelta(hours=24), timedelta(days=7), timedelta(days=30)
        ]
        assert engine.log_patterns is not None
        assert engine.anomaly_detectors is not None
    
    def test_log_analytics_analysis_methods(self):
        """Test log analytics analysis methods exist."""
        from src.monitoring.operational_analytics import LogAnalyticsEngine
        
        engine = LogAnalyticsEngine()
        
        # Test method existence
        assert hasattr(engine, 'analyze_log_patterns')
        assert hasattr(engine, 'detect_log_anomalies')
        assert hasattr(engine, 'generate_log_insights')
        assert hasattr(engine, 'calculate_error_rates')
        assert hasattr(engine, 'analyze_performance_trends')
    
    @pytest.mark.asyncio
    async def test_log_pattern_analysis(self):
        """Test log pattern analysis functionality."""
        from src.monitoring.operational_analytics import LogAnalyticsEngine
        
        engine = LogAnalyticsEngine()
        
        # Mock log data
        log_data = [
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=30),
                'level': 'ERROR',
                'category': 'trading',
                'message': 'Trade execution failed: insufficient balance',
                'metadata': {'symbol': 'ETH', 'amount': 100}
            },
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=25),
                'level': 'ERROR',
                'category': 'trading',
                'message': 'Trade execution failed: market closed',
                'metadata': {'symbol': 'BTC', 'amount': 50}
            },
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=20),
                'level': 'INFO',
                'category': 'trading',
                'message': 'Trade executed successfully',
                'metadata': {'symbol': 'ETH', 'amount': 25}
            }
        ]
        
        patterns = await engine.analyze_log_patterns(log_data)
        
        assert patterns is not None
        assert 'error_patterns' in patterns
        assert 'success_patterns' in patterns
        assert 'frequency_analysis' in patterns
        assert patterns['total_logs'] == 3
        assert patterns['error_rate'] > 0


class TestTradingPerformanceAnalytics:
    """Test trading performance analytics engine."""
    
    def test_trading_analytics_engine_init(self):
        """Test TradingAnalyticsEngine initialization."""
        from src.monitoring.operational_analytics import TradingAnalyticsEngine
        
        engine = TradingAnalyticsEngine()
        assert engine.performance_metrics is not None
        assert engine.trend_analyzers is not None
        assert engine.risk_calculators is not None
    
    def test_trading_analytics_methods(self):
        """Test trading analytics methods exist."""
        from src.monitoring.operational_analytics import TradingAnalyticsEngine
        
        engine = TradingAnalyticsEngine()
        
        # Test method existence
        assert hasattr(engine, 'analyze_trading_performance')
        assert hasattr(engine, 'calculate_profitability_metrics')
        assert hasattr(engine, 'analyze_risk_metrics')
        assert hasattr(engine, 'detect_trading_patterns')
        assert hasattr(engine, 'generate_performance_insights')
    
    @pytest.mark.asyncio
    async def test_trading_performance_analysis(self):
        """Test trading performance analysis."""
        from src.monitoring.operational_analytics import TradingAnalyticsEngine
        
        engine = TradingAnalyticsEngine()
        
        # Mock trading data
        trading_data = [
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(hours=2),
                'symbol': 'ETH',
                'side': 'buy',
                'amount': Decimal('100'),
                'price': Decimal('2000'),
                'pnl': Decimal('50'),
                'success': True
            },
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
                'symbol': 'BTC',
                'side': 'sell',
                'amount': Decimal('1'),
                'price': Decimal('45000'),
                'pnl': Decimal('-100'),
                'success': False
            }
        ]
        
        performance = await engine.analyze_trading_performance(trading_data)
        
        assert performance is not None
        assert 'total_trades' in performance
        assert 'success_rate' in performance
        assert 'total_pnl' in performance
        assert 'average_trade_size' in performance
        assert performance['total_trades'] == 2
    
    @pytest.mark.asyncio
    async def test_profitability_metrics(self):
        """Test profitability metrics calculation."""
        from src.monitoring.operational_analytics import TradingAnalyticsEngine
        
        engine = TradingAnalyticsEngine()
        
        # Mock P&L data
        pnl_data = [
            {'timestamp': datetime.now(timezone.utc) - timedelta(days=1), 'pnl': Decimal('100')},
            {'timestamp': datetime.now(timezone.utc) - timedelta(hours=12), 'pnl': Decimal('-50')},
            {'timestamp': datetime.now(timezone.utc) - timedelta(hours=1), 'pnl': Decimal('200')}
        ]
        
        metrics = await engine.calculate_profitability_metrics(pnl_data)
        
        assert metrics is not None
        assert 'total_pnl' in metrics
        assert 'average_daily_pnl' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics


class TestComplianceAnalytics:
    """Test compliance analytics engine."""
    
    def test_compliance_analytics_engine_init(self):
        """Test ComplianceAnalyticsEngine initialization."""
        from src.monitoring.operational_analytics import ComplianceAnalyticsEngine
        
        engine = ComplianceAnalyticsEngine()
        assert engine.regulatory_frameworks is not None
        assert 'MIFID_II' in engine.regulatory_frameworks
        assert 'GDPR' in engine.regulatory_frameworks
        assert 'SEC_RULE_3A4' in engine.regulatory_frameworks
        assert engine.audit_processors is not None
    
    def test_compliance_analytics_methods(self):
        """Test compliance analytics methods exist."""
        from src.monitoring.operational_analytics import ComplianceAnalyticsEngine
        
        engine = ComplianceAnalyticsEngine()
        
        # Test method existence
        assert hasattr(engine, 'analyze_compliance_status')
        assert hasattr(engine, 'generate_regulatory_reports')
        assert hasattr(engine, 'detect_compliance_violations')
        assert hasattr(engine, 'audit_trail_analysis')
        assert hasattr(engine, 'risk_assessment_analytics')
    
    @pytest.mark.asyncio
    async def test_compliance_status_analysis(self):
        """Test compliance status analysis."""
        from src.monitoring.operational_analytics import ComplianceAnalyticsEngine
        
        engine = ComplianceAnalyticsEngine()
        
        # Mock audit data
        audit_data = [
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
                'event_type': 'trade_execution',
                'user_id': 'user123',
                'regulation': 'MIFID_II',
                'compliant': True,
                'details': {'explanation_provided': True, 'audit_trail': True}
            },
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=30),
                'event_type': 'data_processing',
                'user_id': 'user456',
                'regulation': 'GDPR',
                'compliant': False,
                'details': {'consent_obtained': False, 'data_minimization': True}
            }
        ]
        
        status = await engine.analyze_compliance_status(audit_data)
        
        assert status is not None
        assert 'overall_compliance_rate' in status
        assert 'regulation_breakdown' in status
        assert 'violations' in status
        assert 'MIFID_II' in status['regulation_breakdown']
        assert 'GDPR' in status['regulation_breakdown']
    
    @pytest.mark.asyncio
    async def test_regulatory_report_generation(self):
        """Test regulatory report generation."""
        from src.monitoring.operational_analytics import ComplianceAnalyticsEngine
        
        engine = ComplianceAnalyticsEngine()
        
        # Mock data for report
        report_data = {
            'period_start': datetime.now(timezone.utc) - timedelta(days=30),
            'period_end': datetime.now(timezone.utc),
            'regulation': 'MIFID_II'
        }
        
        report = await engine.generate_regulatory_reports(report_data)
        
        assert report is not None
        assert 'regulation' in report
        assert 'compliance_summary' in report
        assert 'audit_trail_completeness' in report
        assert 'recommendations' in report


class TestSystemHealthAnalytics:
    """Test system health analytics engine."""
    
    def test_system_health_analytics_init(self):
        """Test SystemHealthAnalyticsEngine initialization."""
        from src.monitoring.operational_analytics import SystemHealthAnalyticsEngine
        
        engine = SystemHealthAnalyticsEngine()
        assert engine.health_indicators is not None
        assert engine.predictive_models is not None
        assert engine.threshold_monitors is not None
    
    def test_system_health_methods(self):
        """Test system health analytics methods exist."""
        from src.monitoring.operational_analytics import SystemHealthAnalyticsEngine
        
        engine = SystemHealthAnalyticsEngine()
        
        # Test method existence
        assert hasattr(engine, 'analyze_system_health')
        assert hasattr(engine, 'predict_system_issues')
        assert hasattr(engine, 'generate_health_insights')
        assert hasattr(engine, 'monitor_resource_usage')
        assert hasattr(engine, 'detect_performance_degradation')
    
    @pytest.mark.asyncio
    async def test_system_health_analysis(self):
        """Test system health analysis."""
        from src.monitoring.operational_analytics import SystemHealthAnalyticsEngine
        
        engine = SystemHealthAnalyticsEngine()
        
        # Mock system metrics
        metrics_data = [
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=5),
                'cpu_usage': 75.5,
                'memory_usage': 68.2,
                'disk_usage': 45.1,
                'network_latency': 25.0,
                'error_rate': 0.02
            },
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=10),
                'cpu_usage': 80.1,
                'memory_usage': 72.4,
                'disk_usage': 45.2,
                'network_latency': 30.5,
                'error_rate': 0.05
            }
        ]
        
        health = await engine.analyze_system_health(metrics_data)
        
        assert health is not None
        assert 'overall_health_score' in health
        assert 'component_scores' in health
        assert 'trend_analysis' in health
        assert 'recommendations' in health
        assert 0 <= health['overall_health_score'] <= 100


class TestOperationalDashboardData:
    """Test operational dashboard data generation."""
    
    def test_dashboard_data_generator_init(self):
        """Test DashboardDataGenerator initialization."""
        from src.monitoring.operational_analytics import DashboardDataGenerator
        
        generator = DashboardDataGenerator()
        assert generator.analytics_engines is not None
        assert generator.refresh_interval > 0
        assert generator.data_cache is not None
    
    def test_dashboard_data_methods(self):
        """Test dashboard data methods exist."""
        from src.monitoring.operational_analytics import DashboardDataGenerator
        
        generator = DashboardDataGenerator()
        
        # Test method existence
        assert hasattr(generator, 'generate_overview_data')
        assert hasattr(generator, 'generate_trading_dashboard')
        assert hasattr(generator, 'generate_compliance_dashboard')
        assert hasattr(generator, 'generate_system_health_dashboard')
        assert hasattr(generator, 'refresh_dashboard_cache')
    
    @pytest.mark.asyncio
    async def test_overview_dashboard_generation(self):
        """Test overview dashboard data generation."""
        from src.monitoring.operational_analytics import DashboardDataGenerator
        
        generator = DashboardDataGenerator()
        
        overview = await generator.generate_overview_data()
        
        assert overview is not None
        assert 'system_status' in overview
        assert 'trading_summary' in overview
        assert 'compliance_status' in overview
        assert 'recent_alerts' in overview
        assert 'performance_metrics' in overview
    
    @pytest.mark.asyncio
    async def test_trading_dashboard_generation(self):
        """Test trading-specific dashboard data generation."""
        from src.monitoring.operational_analytics import DashboardDataGenerator
        
        generator = DashboardDataGenerator()
        
        trading_dashboard = await generator.generate_trading_dashboard()
        
        assert trading_dashboard is not None
        assert 'pnl_summary' in trading_dashboard
        assert 'trade_statistics' in trading_dashboard
        assert 'performance_charts' in trading_dashboard
        assert 'risk_metrics' in trading_dashboard


class TestAnalyticsInsights:
    """Test analytics insights generation."""
    
    def test_insights_generator_init(self):
        """Test InsightsGenerator initialization."""
        from src.monitoring.operational_analytics import InsightsGenerator
        
        generator = InsightsGenerator()
        assert generator.insight_templates is not None
        assert generator.ml_analyzers is not None
        assert generator.trend_detectors is not None
    
    def test_insights_generation_methods(self):
        """Test insights generation methods exist."""
        from src.monitoring.operational_analytics import InsightsGenerator
        
        generator = InsightsGenerator()
        
        # Test method existence
        assert hasattr(generator, 'generate_operational_insights')
        assert hasattr(generator, 'detect_trends_and_patterns')
        assert hasattr(generator, 'recommend_optimizations')
        assert hasattr(generator, 'predict_future_issues')
        assert hasattr(generator, 'generate_executive_summary')
    
    @pytest.mark.asyncio
    async def test_operational_insights_generation(self):
        """Test operational insights generation."""
        from src.monitoring.operational_analytics import InsightsGenerator
        
        generator = InsightsGenerator()
        
        # Mock analytics data
        analytics_data = {
            'log_analytics': {'error_rate': 0.05, 'performance_degradation': True},
            'trading_analytics': {'success_rate': 0.85, 'pnl_trend': 'positive'},
            'compliance_analytics': {'compliance_rate': 0.95, 'violations': 2},
            'system_health': {'health_score': 78, 'trending': 'down'}
        }
        
        insights = await generator.generate_operational_insights(analytics_data)
        
        assert insights is not None
        assert 'priority_issues' in insights
        assert 'recommendations' in insights
        assert 'trends' in insights
        assert 'success_metrics' in insights
        assert len(insights['priority_issues']) >= 0
        assert len(insights['recommendations']) >= 0


class TestAnalyticsConfiguration:
    """Test analytics configuration and settings."""
    
    def test_analytics_config_init(self):
        """Test AnalyticsConfig initialization."""
        from src.monitoring.operational_analytics import AnalyticsConfig
        
        config = AnalyticsConfig()
        assert config.analysis_intervals is not None
        assert config.retention_policies is not None
        assert config.alert_thresholds is not None
        assert config.dashboard_settings is not None
    
    def test_analytics_config_validation(self):
        """Test analytics configuration validation."""
        from src.monitoring.operational_analytics import AnalyticsConfig
        
        config = AnalyticsConfig()
        
        # Test validation methods
        assert hasattr(config, 'validate_config')
        assert hasattr(config, 'get_analysis_interval')
        assert hasattr(config, 'get_retention_policy')
        assert hasattr(config, 'get_alert_threshold')
        
        # Test default values
        assert config.get_analysis_interval('log_analytics') > 0
        assert config.get_retention_policy('trading_data') > 0
        assert config.get_alert_threshold('error_rate') > 0


class TestAnalyticsIntegration:
    """Test analytics integration with existing systems."""
    
    @pytest.mark.asyncio
    async def test_activity_logger_integration(self):
        """Test integration with activity logger."""
        from src.monitoring.operational_analytics import OperationalAnalytics
        
        analytics = OperationalAnalytics()
        
        # Test that analytics can access activity logger data
        assert hasattr(analytics, 'activity_logger')
        assert hasattr(analytics, 'get_activity_data')
        
        # Mock getting recent activities
        activities = await analytics.get_activity_data(
            hours_back=24,
            categories=['trading', 'system', 'security']
        )
        
        assert activities is not None
        assert isinstance(activities, list)
    
    @pytest.mark.asyncio
    async def test_enhanced_logging_integration(self):
        """Test integration with enhanced logging."""
        from src.monitoring.operational_analytics import OperationalAnalytics
        
        analytics = OperationalAnalytics()
        
        # Test that analytics can access enhanced logs
        assert hasattr(analytics, 'enhanced_logger')
        assert hasattr(analytics, 'get_structured_logs')
        
        # Mock getting structured log data
        logs = await analytics.get_structured_logs(
            time_range=timedelta(hours=1),
            log_levels=['INFO', 'WARNING', 'ERROR']
        )
        
        assert logs is not None
        assert isinstance(logs, list)
    
    @pytest.mark.asyncio
    async def test_monitoring_metrics_integration(self):
        """Test integration with monitoring metrics."""
        from src.monitoring.operational_analytics import OperationalAnalytics
        
        analytics = OperationalAnalytics()
        
        # Test that analytics can access monitoring data
        assert hasattr(analytics, 'metrics_collectors')
        assert hasattr(analytics, 'get_metrics_data')
        
        # Mock getting metrics data
        metrics = await analytics.get_metrics_data(
            metric_types=['trading', 'system', 'performance'],
            time_range=timedelta(hours=6)
        )
        
        assert metrics is not None
        assert isinstance(metrics, dict)


class TestAnalyticsPerformance:
    """Test analytics system performance."""
    
    @pytest.mark.asyncio
    async def test_analytics_processing_performance(self):
        """Test analytics processing performance."""
        from src.monitoring.operational_analytics import OperationalAnalytics
        import time
        
        analytics = OperationalAnalytics()
        
        # Test processing time for analytics
        start_time = time.time()
        
        # Run analytics on mock data
        await analytics.run_analytics_cycle()
        
        processing_time = time.time() - start_time
        
        # Analytics should complete within reasonable time (5 seconds max)
        assert processing_time < 5.0
    
    @pytest.mark.asyncio
    async def test_dashboard_data_caching(self):
        """Test dashboard data caching performance."""
        from src.monitoring.operational_analytics import DashboardDataGenerator
        import time
        
        generator = DashboardDataGenerator()
        
        # First call should populate cache
        start_time = time.time()
        overview1 = await generator.generate_overview_data()
        first_call_time = time.time() - start_time
        
        # Second call should use cache (much faster)
        start_time = time.time()
        overview2 = await generator.generate_overview_data()
        second_call_time = time.time() - start_time
        
        # Cached call should be significantly faster
        assert second_call_time < first_call_time / 2
        assert overview1 == overview2


if __name__ == "__main__":
    pytest.main([__file__])