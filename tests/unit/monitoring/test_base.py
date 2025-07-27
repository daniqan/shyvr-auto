"""
Tests for base metrics collection functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import REGISTRY

from src.monitoring.base import MetricsCollector, MetricsRegistry


class TestMetricsRegistry:
    """Tests for the MetricsRegistry class."""
    
    def test_init_creates_custom_registry(self):
        """Test that initialization creates a custom registry."""
        registry = MetricsRegistry()
        assert registry.registry is not None
        assert isinstance(registry.registry, CollectorRegistry)
        assert registry.registry != REGISTRY  # Should be separate from default
    
    def test_get_counter_creates_new_counter(self):
        """Test creating a new counter metric."""
        registry = MetricsRegistry()
        counter = registry.get_counter(
            "test_counter", 
            "Test counter description",
            ["label1", "label2"]
        )
        
        assert isinstance(counter, Counter)
        assert counter._name == "test_counter"
        assert counter._documentation == "Test counter description"
        assert counter._labelnames == ("label1", "label2")
    
    def test_get_counter_returns_existing_counter(self):
        """Test that requesting the same counter returns the existing instance."""
        registry = MetricsRegistry()
        counter1 = registry.get_counter("test_counter", "Description")
        counter2 = registry.get_counter("test_counter", "Description")
        
        assert counter1 is counter2
    
    def test_get_gauge_creates_new_gauge(self):
        """Test creating a new gauge metric."""
        registry = MetricsRegistry()
        gauge = registry.get_gauge(
            "test_gauge",
            "Test gauge description", 
            ["environment", "service"]
        )
        
        assert isinstance(gauge, Gauge)
        assert gauge._name == "test_gauge"
        assert gauge._documentation == "Test gauge description"
        assert gauge._labelnames == ("environment", "service")
    
    def test_get_histogram_creates_new_histogram(self):
        """Test creating a new histogram metric."""
        registry = MetricsRegistry()
        histogram = registry.get_histogram(
            "test_histogram",
            "Test histogram description",
            ["method", "endpoint"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        assert isinstance(histogram, Histogram)
        assert histogram._name == "test_histogram"
        assert histogram._documentation == "Test histogram description"
        assert histogram._labelnames == ("method", "endpoint")
    
    def test_clear_metrics_resets_registry(self):
        """Test that clear_metrics creates a new registry."""
        registry = MetricsRegistry()
        original_registry = registry.registry
        
        # Add a metric
        registry.get_counter("test_counter", "Description")
        
        # Clear metrics
        registry.clear_metrics()
        
        # Should have a new registry
        assert registry.registry is not original_registry
        assert isinstance(registry.registry, CollectorRegistry)
    
    def test_generate_output_returns_metrics_text(self):
        """Test that generate_output returns Prometheus format text."""
        registry = MetricsRegistry()
        counter = registry.get_counter("test_counter", "Test counter")
        counter.inc(5)
        
        output = registry.generate_output()
        
        assert isinstance(output, bytes)
        assert b"test_counter" in output
        assert b"5.0" in output


class TestMetricsCollector:
    """Tests for the abstract MetricsCollector base class."""
    
    def test_init_requires_registry(self):
        """Test that MetricsCollector requires a registry."""
        registry = MetricsRegistry()
        
        # Create a concrete implementation for testing
        class TestCollector(MetricsCollector):
            def collect_metrics(self):
                pass
            
            def get_metric_definitions(self):
                return {}
        
        collector = TestCollector(registry)
        assert collector.registry is registry
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""
        registry = MetricsRegistry()
        
        # This should raise TypeError when trying to instantiate
        with pytest.raises(TypeError):
            MetricsCollector(registry)
    
    def test_start_collection_calls_collect_metrics(self):
        """Test that start_collection properly calls collect_metrics."""
        registry = MetricsRegistry()
        
        # Create a concrete implementation for testing
        class TestCollector(MetricsCollector):
            def __init__(self, registry):
                super().__init__(registry)
                self.collect_called = False
            
            def collect_metrics(self):
                self.collect_called = True
            
            def get_metric_definitions(self):
                return {}
        
        collector = TestCollector(registry)
        collector.start_collection()
        
        assert collector.collect_called
    
    def test_metric_definitions_property(self):
        """Test that metric_definitions property calls get_metric_definitions."""
        registry = MetricsRegistry()
        
        class TestCollector(MetricsCollector):
            def collect_metrics(self):
                pass
            
            def get_metric_definitions(self):
                return {"test_metric": "Test metric definition"}
        
        collector = TestCollector(registry)
        definitions = collector.metric_definitions
        
        assert definitions == {"test_metric": "Test metric definition"}
    
    @patch('src.monitoring.base.time')
    def test_record_execution_time_decorator(self, mock_time):
        """Test the record_execution_time decorator functionality."""
        registry = MetricsRegistry()
        
        class TestCollector(MetricsCollector):
            def __init__(self, registry):
                super().__init__(registry)
                self.execution_histogram = registry.get_histogram(
                    "execution_time", "Execution time", ["method"]
                )
            
            def collect_metrics(self):
                pass
            
            def get_metric_definitions(self):
                return {}
            
            @MetricsCollector.record_execution_time("test_method")
            def test_method(self):
                return "success"
        
        # Mock time to return specific values
        mock_time.time.side_effect = [100.0, 101.5]  # Start and end times
        
        collector = TestCollector(registry)
        result = collector.test_method()
        
        assert result == "success"
        # Verify the decorator was called
        assert mock_time.time.call_count == 2
    
    def test_is_healthy_default_implementation(self):
        """Test that is_healthy returns True by default."""
        registry = MetricsRegistry()
        
        class TestCollector(MetricsCollector):
            def collect_metrics(self):
                pass
            
            def get_metric_definitions(self):
                return {}
        
        collector = TestCollector(registry)
        assert collector.is_healthy() is True
    
    def test_get_status_returns_collector_info(self):
        """Test that get_status returns collector status information."""
        registry = MetricsRegistry()
        
        class TestCollector(MetricsCollector):
            def collect_metrics(self):
                pass
            
            def get_metric_definitions(self):
                return {"metric1": "Description 1", "metric2": "Description 2"}
        
        collector = TestCollector(registry)
        status = collector.get_status()
        
        assert status["healthy"] is True
        assert status["name"] == "TestCollector"
        assert status["metrics_count"] == 2
        assert "metric1" in status["metrics"]
        assert "metric2" in status["metrics"]