"""
Tests for the /metrics endpoint integration with FastAPI.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.monitoring.base import MetricsRegistry
from src.monitoring.trading_metrics import TradingMetricsCollector
from src.monitoring.safety_metrics import SafetyMetricsCollector


class TestMetricsEndpoint:
    """Tests for the metrics endpoint integration."""
    
    def test_metrics_endpoint_returns_prometheus_format(self):
        """Test that /metrics endpoint returns data in Prometheus format."""
        # Create a simple FastAPI app for testing
        app = FastAPI()
        
        # Create metrics registry and collectors
        registry = MetricsRegistry()
        trading_collector = TradingMetricsCollector(registry)
        safety_collector = SafetyMetricsCollector(registry)
        
        # Add some test data to metrics
        trading_collector.total_pnl_gauge.labels(currency="USD").set(1500.0)
        safety_collector.risk_level_gauge.labels(risk_type="overall", timeframe="current").set(0.65)
        
        # Define the metrics endpoint
        @app.get("/metrics")
        async def get_metrics():
            from fastapi import Response
            content = registry.generate_output()
            return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
        
        # Test the endpoint
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check that the response contains Prometheus format data
        content = response.content.decode('utf-8')
        assert "trading_total_pnl" in content
        assert "safety_risk_level" in content
        assert "1500.0" in content
        assert "0.65" in content
    
    def test_metrics_endpoint_includes_help_text(self):
        """Test that metrics endpoint includes help text for metrics."""
        app = FastAPI()
        registry = MetricsRegistry()
        
        # Create a counter with help text
        counter = registry.get_counter("test_counter", "This is a test counter")
        counter.inc(5)
        
        @app.get("/metrics")
        async def get_metrics():
            from fastapi import Response
            content = registry.generate_output()
            return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        content = response.content.decode('utf-8')
        assert "# HELP test_counter_total This is a test counter" in content
        assert "# TYPE test_counter_total counter" in content
        assert "test_counter_total 5.0" in content
    
    def test_metrics_endpoint_with_labels(self):
        """Test that metrics endpoint properly formats labels."""
        app = FastAPI()
        registry = MetricsRegistry()
        
        # Create metrics with labels
        gauge = registry.get_gauge("test_gauge", "Test gauge with labels", ["environment", "service"])
        gauge.labels(environment="production", service="trading").set(42.5)
        
        @app.get("/metrics")
        async def get_metrics():
            from fastapi import Response
            content = registry.generate_output()
            return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        content = response.content.decode('utf-8')
        assert 'test_gauge{environment="production",service="trading"} 42.5' in content
    
    def test_metrics_endpoint_with_multiple_collectors(self):
        """Test metrics endpoint with multiple metric collectors."""
        app = FastAPI()
        registry = MetricsRegistry()
        
        # Create multiple collectors
        trading_collector = TradingMetricsCollector(registry)
        safety_collector = SafetyMetricsCollector(registry)
        
        # Add test data
        trading_collector.trade_count_counter.labels(symbol="BTC/USD", side="buy").inc(10)
        safety_collector.emergency_stops_counter.labels(reason="risk", trigger="auto").inc(2)
        
        @app.get("/metrics")
        async def get_metrics():
            from fastapi import Response
            content = registry.generate_output()
            return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        content = response.content.decode('utf-8')
        
        # Should contain metrics from both collectors
        assert "trading_trade_count_total" in content
        assert "safety_emergency_stops_total" in content
        assert '10.0' in content  # Trade count
        assert '2.0' in content   # Emergency stops
    
    def test_metrics_endpoint_handles_empty_registry(self):
        """Test that metrics endpoint handles an empty registry gracefully."""
        app = FastAPI()
        registry = MetricsRegistry()
        
        @app.get("/metrics")
        async def get_metrics():
            return registry.generate_output()
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Empty registry should still return valid Prometheus format (just empty)
        content = response.content.decode('utf-8')
        assert isinstance(content, str)
    
    def test_metrics_endpoint_with_histogram(self):
        """Test that histogram metrics are properly formatted."""
        app = FastAPI()
        registry = MetricsRegistry()
        
        # Create histogram and add observations
        histogram = registry.get_histogram(
            "test_duration", 
            "Test duration histogram",
            ["method"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0)
        )
        
        # Add some observations
        histogram.labels(method="GET").observe(0.3)
        histogram.labels(method="GET").observe(1.2)
        histogram.labels(method="GET").observe(0.8)
        
        @app.get("/metrics")
        async def get_metrics():
            from fastapi import Response
            content = registry.generate_output()
            return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        content = response.content.decode('utf-8')
        
        # Check for histogram buckets
        assert "test_duration_bucket" in content
        assert "test_duration_sum" in content
        assert "test_duration_count" in content
        assert 'le="0.5"' in content
        assert 'le="1.0"' in content
        assert 'le="+Inf"' in content
    
    def test_metrics_endpoint_error_handling(self):
        """Test metrics endpoint error handling."""
        app = FastAPI()
        
        # Create a registry that will cause an error
        @app.get("/metrics")
        async def get_metrics():
            raise Exception("Test error")
        
        client = TestClient(app)
        
        # The endpoint should handle errors gracefully
        # (In a real implementation, we'd want proper error handling)
        with pytest.raises(Exception):
            response = client.get("/metrics")
    
    def test_metrics_content_type_header(self):
        """Test that metrics endpoint sets correct content type header."""
        app = FastAPI()
        registry = MetricsRegistry()
        
        @app.get("/metrics")
        async def get_metrics():
            from fastapi import Response
            content = registry.generate_output()
            return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "version=0.0.4" in response.headers["content-type"]
        assert "charset=utf-8" in response.headers["content-type"]
    
    @patch('src.monitoring.trading_metrics.TradingMetricsCollector.collect_metrics')
    @patch('src.monitoring.safety_metrics.SafetyMetricsCollector.collect_metrics')
    def test_metrics_endpoint_triggers_collection(self, mock_safety_collect, mock_trading_collect):
        """Test that accessing metrics endpoint triggers metrics collection."""
        app = FastAPI()
        registry = MetricsRegistry()
        
        trading_collector = TradingMetricsCollector(registry)
        safety_collector = SafetyMetricsCollector(registry)
        
        @app.get("/metrics")
        async def get_metrics():
            # Trigger collection before generating output
            trading_collector.collect_metrics()
            safety_collector.collect_metrics()
            return registry.generate_output()
        
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        mock_trading_collect.assert_called_once()
        mock_safety_collect.assert_called_once()
    
    def test_metrics_endpoint_concurrent_access(self):
        """Test that metrics endpoint handles concurrent access safely."""
        import threading
        import time
        
        app = FastAPI()
        registry = MetricsRegistry()
        counter = registry.get_counter("concurrent_test", "Concurrent test counter")
        
        @app.get("/metrics")
        async def get_metrics():
            # Simulate some processing time
            counter.inc()
            return registry.generate_output()
        
        client = TestClient(app)
        
        # Make multiple concurrent requests
        results = []
        def make_request():
            response = client.get("/metrics")
            results.append(response.status_code)
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5