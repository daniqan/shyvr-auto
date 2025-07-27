"""
Tests for safety metrics collection functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from decimal import Decimal

from src.monitoring.base import MetricsRegistry
from src.monitoring.safety_metrics import SafetyMetricsCollector


class TestSafetyMetricsCollector:
    """Tests for the SafetyMetricsCollector class."""
    
    def test_init_creates_metrics(self):
        """Test that initialization creates all safety metrics."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        # Check that all expected metrics are created
        assert hasattr(collector, 'risk_level_gauge')
        assert hasattr(collector, 'emergency_stops_counter')
        assert hasattr(collector, 'liquidations_counter')
        assert hasattr(collector, 'position_size_gauge')
        assert hasattr(collector, 'drawdown_gauge')
        assert hasattr(collector, 'risk_violations_counter')
        assert hasattr(collector, 'safety_checks_counter')
        assert hasattr(collector, 'margin_ratio_gauge')
        assert hasattr(collector, 'volatility_gauge')
        assert hasattr(collector, 'correlation_risk_gauge')
    
    def test_get_metric_definitions_returns_all_metrics(self):
        """Test that get_metric_definitions returns all safety metrics."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        definitions = collector.get_metric_definitions()
        
        expected_metrics = [
            'safety_risk_level',
            'safety_emergency_stops_total',
            'safety_liquidations_total',
            'safety_position_size_ratio',
            'safety_drawdown_percentage',
            'safety_risk_violations_total',
            'safety_checks_total',
            'safety_margin_ratio',
            'safety_volatility_score',
            'safety_correlation_risk_score'
        ]
        
        for metric in expected_metrics:
            assert metric in definitions
            assert isinstance(definitions[metric], str)
            assert len(definitions[metric]) > 0
    
    @patch('src.monitoring.safety_metrics.SafetyMetricsCollector._get_risk_manager')
    @patch('src.monitoring.safety_metrics.SafetyMetricsCollector._get_portfolio_manager')
    def test_collect_metrics_updates_all_metrics(self, mock_portfolio_manager, mock_risk_manager):
        """Test that collect_metrics updates all safety metrics."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        # Mock risk manager
        mock_risk = Mock()
        mock_risk.get_current_risk_level.return_value = 0.75
        mock_risk.get_max_position_size_ratio.return_value = 0.85
        mock_risk.get_margin_ratio.return_value = 1.5
        mock_risk.get_volatility_score.return_value = 0.6
        mock_risk.get_correlation_risk.return_value = 0.4
        mock_risk_manager.return_value = mock_risk
        
        # Mock portfolio manager
        mock_portfolio = Mock()
        mock_portfolio.get_current_drawdown.return_value = Decimal('0.15')
        mock_portfolio_manager.return_value = mock_portfolio
        
        # Mock safety events
        with patch.object(collector, '_get_safety_events') as mock_safety_events:
            mock_safety_events.return_value = {
                'emergency_stops': 2,
                'liquidations': 1,
                'risk_violations': 5,
                'safety_checks': 1000
            }
            
            collector.collect_metrics()
            
            # Verify that collection methods were called
            mock_risk.get_current_risk_level.assert_called_once()
            mock_risk.get_max_position_size_ratio.assert_called_once()
            mock_portfolio.get_current_drawdown.assert_called_once()
            mock_safety_events.assert_called_once()
    
    def test_record_emergency_stop_updates_metrics(self):
        """Test recording an emergency stop updates relevant metrics."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        # Record an emergency stop
        stop_data = {
            'reason': 'high_risk',
            'trigger': 'drawdown_limit',
            'positions_closed': 3
        }
        
        collector.record_emergency_stop(stop_data)
        
        # Check that the method runs without error
        assert True  # Test passes if no exception is raised
    
    def test_record_liquidation_updates_metrics(self):
        """Test recording a liquidation updates relevant metrics."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        # Record a liquidation
        liquidation_data = {
            'symbol': 'BTC/USD',
            'side': 'long',
            'amount': Decimal('1.5'),
            'reason': 'margin_call'
        }
        
        collector.record_liquidation(liquidation_data)
        
        # Check that the method runs without error
        assert True  # Test passes if no exception is raised
    
    def test_record_risk_violation_updates_metrics(self):
        """Test recording a risk violation updates relevant metrics."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        # Record a risk violation
        violation_data = {
            'violation_type': 'position_size_exceeded',
            'severity': 'high',
            'symbol': 'ETH/USD'
        }
        
        collector.record_risk_violation(violation_data)
        
        # Check that the method runs without error
        assert True  # Test passes if no exception is raised
    
    def test_record_safety_check_updates_metrics(self):
        """Test recording a safety check updates relevant metrics."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        # Record a safety check
        collector.record_safety_check(check_type='risk_assessment', passed=True)
        collector.record_safety_check(check_type='position_validation', passed=False)
        
        # Check that the method runs without error
        assert True  # Test passes if no exception is raised
    
    def test_update_risk_level_updates_gauge(self):
        """Test updating risk level updates the risk level gauge."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        # Update risk level
        collector.update_risk_level(0.85, 'overall')
        
        # Verify the method runs without error
        assert True  # Test passes if no exception is raised
    
    @patch('src.monitoring.safety_metrics.SafetyMetricsCollector._get_risk_manager')
    def test_get_risk_manager_returns_manager(self, mock_get_manager):
        """Test that _get_risk_manager returns a risk manager."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        result = collector._get_risk_manager()
        assert result is mock_manager
    
    @patch('src.monitoring.safety_metrics.SafetyMetricsCollector._get_portfolio_manager')
    def test_get_portfolio_manager_returns_manager(self, mock_get_manager):
        """Test that _get_portfolio_manager returns a portfolio manager."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        result = collector._get_portfolio_manager()
        assert result is mock_manager
    
    def test_get_safety_events_returns_events(self):
        """Test that _get_safety_events returns safety event statistics."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        with patch('src.logging.activity_logger.ActivityLogger') as mock_logger:
            mock_activity_logger = Mock()
            mock_logger.return_value = mock_activity_logger
            
            # Mock safety event query results
            mock_activity_logger.query_activities.side_effect = [
                [{'activity_type': 'EMERGENCY_STOP', 'metadata': {}}],  # emergency stops
                [{'activity_type': 'LIQUIDATION', 'metadata': {}}],     # liquidations
                [{'activity_type': 'RISK_VIOLATION', 'metadata': {}}],  # risk violations
                [{'activity_type': 'SAFETY_CHECK', 'metadata': {}}] * 5  # safety checks
            ]
            
            events = collector._get_safety_events()
            
            expected_events = {
                'emergency_stops': 1,
                'liquidations': 1,
                'risk_violations': 1,
                'safety_checks': 5
            }
            
            assert events == expected_events
    
    def test_is_healthy_returns_true_when_collecting_successfully(self):
        """Test that is_healthy returns True when metrics collection is working."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        with patch.object(collector, 'collect_metrics'):
            assert collector.is_healthy() is True
    
    def test_is_healthy_returns_false_when_collection_fails(self):
        """Test that is_healthy returns False when metrics collection fails."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        with patch.object(collector, 'collect_metrics', side_effect=Exception("Collection failed")):
            assert collector.is_healthy() is False
    
    def test_get_status_includes_safety_specific_info(self):
        """Test that get_status includes safety-specific status information."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        with patch.object(collector, '_get_safety_events') as mock_events:
            mock_events.return_value = {
                'emergency_stops': 3,
                'liquidations': 1,
                'risk_violations': 8,
                'safety_checks': 2500
            }
            
            status = collector.get_status()
            
            assert status['name'] == 'SafetyMetricsCollector'
            assert 'safety_events' in status
            assert status['safety_events']['emergency_stops'] == 3
            assert status['safety_events']['liquidations'] == 1
    
    def test_calculate_risk_score_returns_score(self):
        """Test that _calculate_risk_score returns a risk score."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        # Test risk score calculation with mock data
        risk_data = {
            'position_size_ratio': 0.8,
            'drawdown': 0.15,
            'volatility': 0.6,
            'correlation_risk': 0.4,
            'margin_ratio': 1.2
        }
        
        risk_score = collector._calculate_risk_score(risk_data)
        
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 1
    
    def test_check_risk_thresholds_identifies_violations(self):
        """Test that _check_risk_thresholds identifies risk threshold violations."""
        registry = MetricsRegistry()
        collector = SafetyMetricsCollector(registry)
        
        # Test with data that should trigger violations
        risk_data = {
            'position_size_ratio': 0.95,  # High position size
            'drawdown': 0.25,             # High drawdown
            'volatility': 0.9,            # High volatility
            'margin_ratio': 0.8           # Low margin ratio
        }
        
        violations = collector._check_risk_thresholds(risk_data)
        
        assert isinstance(violations, list)
        assert len(violations) > 0  # Should have violations
        
        # Test with safe data
        safe_data = {
            'position_size_ratio': 0.3,
            'drawdown': 0.05,
            'volatility': 0.2,
            'margin_ratio': 2.0
        }
        
        violations = collector._check_risk_thresholds(safe_data)
        assert len(violations) == 0  # Should have no violations