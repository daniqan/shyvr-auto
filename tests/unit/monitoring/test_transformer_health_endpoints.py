#!/usr/bin/env python3

"""
Transformer Health Monitoring Endpoints Tests

Tests for transformer health monitoring endpoints and API integration following TDD methodology.

This test suite creates failing tests FIRST before any implementation.
All tests are designed to fail initially as the components don't exist yet.

Key Requirements:
1. Health check endpoints for transformer models
2. Memory usage monitoring endpoints 
3. Inference latency monitoring endpoints
4. Cache performance monitoring endpoints
5. Model loading status endpoints
6. Integration with main.py health endpoint
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi.testclient import TestClient

# These imports will FAIL initially - that's expected for TDD
try:
    from src.monitoring.transformer_health_endpoints import (
        TransformerHealthEndpoints,
        TransformerHealthAPI,
        TransformerMemoryMonitor,
        TransformerLatencyMonitor,
        TransformerCacheMonitor,
        TransformerLoadingMonitor
    )
    from main import app  # Main FastAPI application
except ImportError:
    # Expected to fail initially - components don't exist yet
    pass


class TestTransformerHealthEndpoints:
    """Test transformer health monitoring endpoints."""
    
    @pytest.fixture
    def mock_transformer_models(self):
        """Mock transformer models for testing."""
        models = {}
        model_configs = [
            ('itransformer', 'ITRANSFORMER', 1200.0, 85.0, 0.75, 2500.0, True),
            ('patchtst', 'PATCHTST', 950.0, 120.0, 0.82, 1800.0, True),
            ('timesmixer', 'TIMESMIXER', 1600.0, 95.0, 0.45, 3200.0, False),  # Low cache rate
            ('timesfm', 'TIMESFM', 7200.0, 1100.0, 0.68, 4500.0, False)  # High memory/latency
        ]
        
        for name, model_type, memory, latency, cache_rate, loading_time, healthy in model_configs:
            model = Mock()
            model.model_type = model_type
            model.get_memory_usage_mb.return_value = memory
            model.get_peak_memory_usage_mb.return_value = memory * 1.2
            model.get_inference_latency_ms.return_value = latency
            model.get_cache_hit_rate.return_value = cache_rate
            model.get_loading_time_ms.return_value = loading_time
            model.is_healthy.return_value = healthy
            model.get_model_info.return_value = {
                'name': name,
                'type': model_type,
                'loaded': True,
                'parameters': 125000000 if model_type != 'TIMESFM' else 2000000000
            }
            models[name] = model
            
        return models
    
    @pytest.fixture
    def health_endpoints(self, mock_transformer_models):
        """Initialize transformer health endpoints."""
        # This will FAIL initially - TransformerHealthEndpoints doesn't exist
        return TransformerHealthEndpoints(models=mock_transformer_models)
    
    def test_transformer_health_endpoints_initialization(self, mock_transformer_models):
        """Test TransformerHealthEndpoints initialization."""
        # This test will FAIL initially - TransformerHealthEndpoints doesn't exist
        endpoints = TransformerHealthEndpoints(models=mock_transformer_models)
        
        assert endpoints.models == mock_transformer_models
        assert hasattr(endpoints, 'memory_monitor')
        assert hasattr(endpoints, 'latency_monitor')
        assert hasattr(endpoints, 'cache_monitor')
        assert hasattr(endpoints, 'loading_monitor')
        
        # Verify monitors initialized with models
        assert endpoints.memory_monitor.models == mock_transformer_models
        assert endpoints.latency_monitor.models == mock_transformer_models
        assert endpoints.cache_monitor.models == mock_transformer_models
        assert endpoints.loading_monitor.models == mock_transformer_models
    
    @pytest.mark.asyncio
    async def test_health_overview_endpoint(self, health_endpoints):
        """Test /health/transformers endpoint."""
        health_data = await health_endpoints.get_health_overview()
        
        # Verify health overview structure
        assert 'status' in health_data
        assert 'timestamp' in health_data
        assert 'models' in health_data
        assert 'summary' in health_data
        
        # Verify model health data
        models_data = health_data['models']
        assert len(models_data) == 4  # All models included
        
        for model_name, model_health in models_data.items():
            assert 'status' in model_health
            assert 'memory_usage_mb' in model_health
            assert 'memory_usage_percent' in model_health
            assert 'inference_latency_ms' in model_health
            assert 'cache_hit_rate' in model_health
            assert 'is_loaded' in model_health
            assert model_health['status'] in ['healthy', 'unhealthy', 'warning']
        
        # Verify summary statistics
        summary = health_data['summary']
        assert 'total_models' in summary
        assert 'healthy_models' in summary
        assert 'unhealthy_models' in summary
        assert 'total_memory_usage_mb' in summary
        assert 'average_latency_ms' in summary
        assert summary['total_models'] == 4
        assert summary['healthy_models'] == 2  # itransformer, patchtst
        assert summary['unhealthy_models'] == 2  # timesmixer, timesfm
    
    @pytest.mark.asyncio
    async def test_memory_usage_endpoint(self, health_endpoints):
        """Test /health/transformers/memory endpoint."""
        memory_data = await health_endpoints.get_memory_usage()
        
        # Verify memory usage structure
        assert 'timestamp' in memory_data
        assert 'total_memory_limit_mb' in memory_data
        assert 'total_memory_usage_mb' in memory_data
        assert 'memory_usage_percent' in memory_data
        assert 'models' in memory_data
        
        # Verify total memory limit (8Gi = 8192MB)
        assert memory_data['total_memory_limit_mb'] == 8192.0
        
        # Verify memory usage calculation
        expected_total = 1200.0 + 950.0 + 1600.0 + 7200.0  # Sum of all models
        assert abs(memory_data['total_memory_usage_mb'] - expected_total) < 1.0
        
        # Verify memory percentage
        expected_percent = (expected_total / 8192.0) * 100
        assert abs(memory_data['memory_usage_percent'] - expected_percent) < 1.0
        
        # Verify per-model memory data
        models_memory = memory_data['models']
        for model_name, memory_info in models_memory.items():
            assert 'current_usage_mb' in memory_info
            assert 'peak_usage_mb' in memory_info  
            assert 'usage_percent' in memory_info
            assert 'status' in memory_info
            assert memory_info['current_usage_mb'] > 0
            assert memory_info['peak_usage_mb'] >= memory_info['current_usage_mb']
    
    @pytest.mark.asyncio
    async def test_inference_latency_endpoint(self, health_endpoints):
        """Test /health/transformers/latency endpoint."""
        latency_data = await health_endpoints.get_inference_latency()
        
        # Verify latency data structure
        assert 'timestamp' in latency_data
        assert 'models' in latency_data
        assert 'aggregated_stats' in latency_data
        
        # Verify per-model latency data
        models_latency = latency_data['models']
        for model_name, latency_info in models_latency.items():
            assert 'current_latency_ms' in latency_info
            assert 'percentiles' in latency_info
            assert 'status' in latency_info
            
            # Verify percentiles
            percentiles = latency_info['percentiles']
            assert 'p50' in percentiles
            assert 'p95' in percentiles
            assert 'p99' in percentiles
            assert percentiles['p50'] <= percentiles['p95'] <= percentiles['p99']
        
        # Verify aggregated statistics
        agg_stats = latency_data['aggregated_stats']
        assert 'average_latency_ms' in agg_stats
        assert 'max_latency_ms' in agg_stats
        assert 'min_latency_ms' in agg_stats
        assert 'models_over_threshold' in agg_stats
        
        # Verify threshold detection (>1000ms)
        assert agg_stats['models_over_threshold'] == 1  # Only timesfm > 1000ms
    
    @pytest.mark.asyncio
    async def test_cache_performance_endpoint(self, health_endpoints):
        """Test /health/transformers/cache endpoint."""
        cache_data = await health_endpoints.get_cache_performance()
        
        # Verify cache data structure
        assert 'timestamp' in cache_data
        assert 'models' in cache_data
        assert 'aggregated_stats' in cache_data
        
        # Verify per-model cache data
        models_cache = cache_data['models']
        for model_name, cache_info in models_cache.items():
            assert 'hit_rate' in cache_info
            assert 'miss_rate' in cache_info
            assert 'total_requests' in cache_info
            assert 'status' in cache_info
            
            # Verify hit rate + miss rate = 1.0
            assert abs(cache_info['hit_rate'] + cache_info['miss_rate'] - 1.0) < 0.01
        
        # Verify aggregated statistics
        agg_stats = cache_data['aggregated_stats']
        assert 'overall_hit_rate' in agg_stats
        assert 'models_below_threshold' in agg_stats
        assert 'total_requests' in agg_stats
        
        # Verify threshold detection (<50%)
        assert agg_stats['models_below_threshold'] == 1  # Only timesmixer < 50%
    
    @pytest.mark.asyncio
    async def test_model_loading_endpoint(self, health_endpoints):
        """Test /health/transformers/loading endpoint."""
        loading_data = await health_endpoints.get_model_loading_status()
        
        # Verify loading data structure
        assert 'timestamp' in loading_data
        assert 'models' in loading_data
        assert 'aggregated_stats' in loading_data
        
        # Verify per-model loading data
        models_loading = loading_data['models']
        for model_name, loading_info in models_loading.items():
            assert 'loading_time_ms' in loading_info
            assert 'is_loaded' in loading_info
            assert 'loading_failures' in loading_info
            assert 'last_loading_time' in loading_info
            assert 'status' in loading_info
        
        # Verify aggregated statistics
        agg_stats = loading_data['aggregated_stats']
        assert 'average_loading_time_ms' in agg_stats
        assert 'max_loading_time_ms' in agg_stats
        assert 'total_failures' in agg_stats
        assert 'successfully_loaded' in agg_stats
        assert agg_stats['successfully_loaded'] == 4  # All models loaded
    
    @pytest.mark.asyncio
    async def test_individual_model_health_endpoint(self, health_endpoints):
        """Test /health/transformers/{model_name} endpoint."""
        model_health = await health_endpoints.get_model_health('itransformer')
        
        # Verify individual model health structure
        assert 'model_name' in model_health
        assert 'model_type' in model_health
        assert 'status' in model_health
        assert 'timestamp' in model_health
        assert 'metrics' in model_health
        
        assert model_health['model_name'] == 'itransformer'
        assert model_health['model_type'] == 'ITRANSFORMER'
        assert model_health['status'] == 'healthy'
        
        # Verify detailed metrics
        metrics = model_health['metrics']
        assert 'memory' in metrics
        assert 'latency' in metrics
        assert 'cache' in metrics
        assert 'loading' in metrics
        assert 'model_info' in metrics
        
        # Verify memory metrics
        memory = metrics['memory']
        assert memory['current_usage_mb'] == 1200.0
        assert memory['usage_percent'] == (1200.0 / 8192.0) * 100
        
        # Verify model info
        model_info = metrics['model_info']
        assert model_info['parameters'] == 125000000
        assert model_info['loaded'] is True
    
    @pytest.mark.asyncio
    async def test_health_alerts_endpoint(self, health_endpoints):
        """Test /health/transformers/alerts endpoint."""
        alerts_data = await health_endpoints.get_health_alerts()
        
        # Verify alerts structure
        assert 'timestamp' in alerts_data
        assert 'active_alerts' in alerts_data
        assert 'alert_summary' in alerts_data
        
        # Verify active alerts
        active_alerts = alerts_data['active_alerts']
        assert len(active_alerts) >= 2  # At least memory and latency alerts
        
        # Expected alerts based on mock data
        alert_types = [alert['type'] for alert in active_alerts]
        assert 'high_memory_usage' in alert_types  # timesfm > 90% of 8Gi
        assert 'slow_inference_latency' in alert_types  # timesfm > 1000ms
        assert 'low_cache_hit_rate' in alert_types  # timesmixer < 50%
        
        # Verify alert structure
        for alert in active_alerts:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'model_name' in alert
            assert 'message' in alert
            assert 'threshold' in alert
            assert 'current_value' in alert
            assert 'timestamp' in alert
            assert alert['severity'] in ['warning', 'critical']
        
        # Verify alert summary
        summary = alerts_data['alert_summary']
        assert 'total_alerts' in summary
        assert 'critical_alerts' in summary
        assert 'warning_alerts' in summary
        assert 'affected_models' in summary


class TestTransformerHealthAPI:
    """Test FastAPI integration for transformer health endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for FastAPI application."""
        # This will FAIL initially - health endpoints not integrated with main.py
        client = TestClient(app)
        return client
    
    def test_health_transformers_endpoint(self, test_client):
        """Test GET /health/transformers endpoint via FastAPI."""
        response = test_client.get("/health/transformers")
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert 'models' in data
        assert 'summary' in data
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']
    
    def test_health_transformers_memory_endpoint(self, test_client):
        """Test GET /health/transformers/memory endpoint via FastAPI."""
        response = test_client.get("/health/transformers/memory")
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'total_memory_usage_mb' in data
        assert 'memory_usage_percent' in data
        assert 'models' in data
        assert data['total_memory_limit_mb'] == 8192.0
    
    def test_health_transformers_latency_endpoint(self, test_client):
        """Test GET /health/transformers/latency endpoint via FastAPI."""
        response = test_client.get("/health/transformers/latency")
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'models' in data
        assert 'aggregated_stats' in data
        
        # Verify latency threshold detection
        assert 'models_over_threshold' in data['aggregated_stats']
    
    def test_health_transformers_cache_endpoint(self, test_client):
        """Test GET /health/transformers/cache endpoint via FastAPI."""
        response = test_client.get("/health/transformers/cache")
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'models' in data
        assert 'aggregated_stats' in data
        
        # Verify cache threshold detection
        assert 'models_below_threshold' in data['aggregated_stats']
    
    def test_health_transformers_loading_endpoint(self, test_client):
        """Test GET /health/transformers/loading endpoint via FastAPI."""
        response = test_client.get("/health/transformers/loading")
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'models' in data
        assert 'aggregated_stats' in data
        assert 'successfully_loaded' in data['aggregated_stats']
    
    def test_health_transformers_alerts_endpoint(self, test_client):
        """Test GET /health/transformers/alerts endpoint via FastAPI."""
        response = test_client.get("/health/transformers/alerts")
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'active_alerts' in data
        assert 'alert_summary' in data
    
    def test_health_transformer_individual_endpoint(self, test_client):
        """Test GET /health/transformers/{model_name} endpoint via FastAPI."""
        response = test_client.get("/health/transformers/itransformer")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data['model_name'] == 'itransformer'
        assert data['model_type'] == 'ITRANSFORMER'
        assert 'metrics' in data
    
    def test_health_transformer_nonexistent_model(self, test_client):
        """Test GET /health/transformers/{nonexistent_model} endpoint."""
        response = test_client.get("/health/transformers/nonexistent")
        
        assert response.status_code == 404
        
        data = response.json()
        assert 'detail' in data
        assert 'not found' in data['detail'].lower()
    
    def test_main_health_endpoint_includes_transformers(self, test_client):
        """Test that main /health endpoint includes transformer health."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'transformers' in data
        
        # Verify transformer section
        transformers = data['transformers']
        assert 'status' in transformers
        assert 'models_healthy' in transformers
        assert 'total_models' in transformers
        assert 'total_memory_usage_mb' in transformers
        assert 'alerts_count' in transformers


class TestTransformerMonitors:
    """Test individual transformer monitoring components."""
    
    @pytest.fixture
    def mock_models(self):
        """Mock transformer models."""
        return {
            'test_model': Mock(
                get_memory_usage_mb=Mock(return_value=1500.0),
                get_peak_memory_usage_mb=Mock(return_value=1800.0),
                get_inference_latency_ms=Mock(return_value=95.0),
                get_cache_hit_rate=Mock(return_value=0.75),
                get_loading_time_ms=Mock(return_value=2500.0),
                is_healthy=Mock(return_value=True)
            )
        }
    
    def test_transformer_memory_monitor(self, mock_models):
        """Test TransformerMemoryMonitor functionality."""
        # This test will FAIL initially - TransformerMemoryMonitor doesn't exist
        monitor = TransformerMemoryMonitor(models=mock_models)
        
        memory_data = monitor.get_memory_status()
        
        # Verify memory monitoring
        assert 'current_usage_mb' in memory_data
        assert 'peak_usage_mb' in memory_data
        assert 'usage_percent' in memory_data
        assert 'status' in memory_data
        
        assert memory_data['current_usage_mb'] == 1500.0
        assert memory_data['peak_usage_mb'] == 1800.0
        assert memory_data['usage_percent'] == (1500.0 / 8192.0) * 100
        
        # Verify status determination
        if memory_data['usage_percent'] > 90:
            assert memory_data['status'] == 'critical'
        elif memory_data['usage_percent'] > 80:
            assert memory_data['status'] == 'warning'
        else:
            assert memory_data['status'] == 'healthy'
    
    def test_transformer_latency_monitor(self, mock_models):
        """Test TransformerLatencyMonitor functionality."""
        # This test will FAIL initially - TransformerLatencyMonitor doesn't exist
        monitor = TransformerLatencyMonitor(models=mock_models)
        
        latency_data = monitor.get_latency_status()
        
        # Verify latency monitoring
        assert 'current_latency_ms' in latency_data
        assert 'percentiles' in latency_data
        assert 'status' in latency_data
        
        assert latency_data['current_latency_ms'] == 95.0
        
        # Verify percentiles calculation
        percentiles = latency_data['percentiles']
        assert 'p50' in percentiles
        assert 'p95' in percentiles
        assert 'p99' in percentiles
        
        # Verify status determination
        if latency_data['current_latency_ms'] > 1000:
            assert latency_data['status'] == 'critical'
        elif latency_data['current_latency_ms'] > 500:
            assert latency_data['status'] == 'warning'
        else:
            assert latency_data['status'] == 'healthy'
    
    def test_transformer_cache_monitor(self, mock_models):
        """Test TransformerCacheMonitor functionality."""
        # This test will FAIL initially - TransformerCacheMonitor doesn't exist
        monitor = TransformerCacheMonitor(models=mock_models)
        
        cache_data = monitor.get_cache_status()
        
        # Verify cache monitoring
        assert 'hit_rate' in cache_data
        assert 'miss_rate' in cache_data
        assert 'total_requests' in cache_data
        assert 'status' in cache_data
        
        assert cache_data['hit_rate'] == 0.75
        assert cache_data['miss_rate'] == 0.25
        
        # Verify status determination
        if cache_data['hit_rate'] < 0.5:
            assert cache_data['status'] == 'critical'
        elif cache_data['hit_rate'] < 0.7:
            assert cache_data['status'] == 'warning'
        else:
            assert cache_data['status'] == 'healthy'
    
    def test_transformer_loading_monitor(self, mock_models):
        """Test TransformerLoadingMonitor functionality."""
        # This test will FAIL initially - TransformerLoadingMonitor doesn't exist
        monitor = TransformerLoadingMonitor(models=mock_models)
        
        loading_data = monitor.get_loading_status()
        
        # Verify loading monitoring
        assert 'loading_time_ms' in loading_data
        assert 'is_loaded' in loading_data
        assert 'loading_failures' in loading_data
        assert 'status' in loading_data
        
        assert loading_data['loading_time_ms'] == 2500.0
        assert loading_data['is_loaded'] is True
        
        # Verify status determination
        if loading_data['loading_time_ms'] > 5000:
            assert loading_data['status'] == 'critical'
        elif loading_data['loading_time_ms'] > 3000:
            assert loading_data['status'] == 'warning'
        else:
            assert loading_data['status'] == 'healthy'


class TestTransformerHealthIntegration:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_performance(self):
        """Test health endpoint performance requirements."""
        models = {f'model_{i}': Mock() for i in range(4)}
        
        endpoints = TransformerHealthEndpoints(models=models)
        
        start_time = datetime.now()
        health_data = await endpoints.get_health_overview()
        end_time = datetime.now()
        
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance requirements
        assert response_time_ms < 100  # <100ms response time
        assert health_data is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_health_requests(self):
        """Test handling concurrent health check requests."""
        models = {'test_model': Mock()}
        endpoints = TransformerHealthEndpoints(models=models)
        
        # Simulate concurrent requests
        import asyncio
        tasks = [
            endpoints.get_health_overview(),
            endpoints.get_memory_usage(),
            endpoints.get_inference_latency(),
            endpoints.get_cache_performance()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed successfully
        assert len(results) == 4
        for result in results:
            assert result is not None
            assert 'timestamp' in result
    
    @pytest.mark.asyncio
    async def test_error_handling_unhealthy_models(self):
        """Test error handling when models are unhealthy."""
        unhealthy_model = Mock()
        unhealthy_model.get_memory_usage_mb.side_effect = Exception("Memory unavailable")
        unhealthy_model.get_inference_latency_ms.side_effect = Exception("Latency unavailable")
        unhealthy_model.is_healthy.return_value = False
        
        models = {'unhealthy_model': unhealthy_model}
        endpoints = TransformerHealthEndpoints(models=models)
        
        # Should handle errors gracefully
        health_data = await endpoints.get_health_overview()
        
        assert health_data is not None
        assert 'models' in health_data
        assert 'unhealthy_model' in health_data['models']
        
        model_health = health_data['models']['unhealthy_model']
        assert model_health['status'] == 'unhealthy'
        assert 'error' in model_health or 'errors' in model_health


class TestTransformerHealthConfiguration:
    """Test configuration and customization of health monitoring."""
    
    def test_health_thresholds_configuration(self):
        """Test configurable health thresholds."""
        custom_thresholds = {
            'memory_warning_percent': 75.0,
            'memory_critical_percent': 85.0,
            'latency_warning_ms': 200.0,
            'latency_critical_ms': 500.0,
            'cache_warning_rate': 0.6,
            'cache_critical_rate': 0.4,
            'loading_warning_ms': 2000.0,
            'loading_critical_ms': 4000.0
        }
        
        models = {'test_model': Mock()}
        endpoints = TransformerHealthEndpoints(
            models=models,
            thresholds=custom_thresholds
        )
        
        assert endpoints.thresholds == custom_thresholds
        
        # Verify thresholds are used in monitoring
        assert endpoints.memory_monitor.warning_threshold == 75.0
        assert endpoints.memory_monitor.critical_threshold == 85.0
        assert endpoints.latency_monitor.warning_threshold == 200.0
        assert endpoints.latency_monitor.critical_threshold == 500.0
    
    def test_health_monitoring_intervals(self):
        """Test configurable monitoring intervals."""
        intervals = {
            'memory_check_interval_seconds': 30,
            'latency_check_interval_seconds': 15,
            'cache_check_interval_seconds': 60,
            'loading_check_interval_seconds': 300
        }
        
        models = {'test_model': Mock()}
        endpoints = TransformerHealthEndpoints(
            models=models,
            monitoring_intervals=intervals
        )
        
        assert endpoints.monitoring_intervals == intervals
        
        # Verify intervals are applied to monitors
        assert endpoints.memory_monitor.check_interval == 30
        assert endpoints.latency_monitor.check_interval == 15
        assert endpoints.cache_monitor.check_interval == 60
        assert endpoints.loading_monitor.check_interval == 300


if __name__ == '__main__':
    pytest.main([__file__, '-v'])