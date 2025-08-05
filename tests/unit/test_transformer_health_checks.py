#!/usr/bin/env python3
"""
Test suite for transformer-specific health check endpoints
Following TDD methodology - failing tests first
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import json

# Test the main health endpoint with transformer monitoring
class TestTransformerHealthEndpoint:
    """Test transformer health check integration in main.py"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI app for testing"""
        from main import app
        return app
    
    @pytest.fixture
    def client(self, mock_app):
        """Test client for API endpoints"""
        return TestClient(mock_app)
    
    def test_health_endpoint_includes_transformers(self, client):
        """Test that /health endpoint includes transformer health status"""
        # This test should FAIL initially as transformer health is not implemented
        with patch('main.get_config') as mock_config, \
             patch('main.check_database_health') as mock_db_health, \
             patch('main.get_ml_model_health') as mock_ml_health, \
             patch('main.get_rl_agent_health') as mock_rl_health, \
             patch('main.get_transformer_health') as mock_transformer_health:
            
            # Mock configuration
            mock_config_obj = MagicMock()
            mock_config_obj.app.version = "1.0.0"
            mock_config_obj.app.environment = "test"
            mock_config.return_value = mock_config_obj
            
            # Mock database health
            mock_db_health.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: {
                    "status": "healthy",
                    "timestamp": "2025-08-05T10:00:00"
                })()
            )
            
            # Mock ML health
            mock_ml_health.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: {"status": "healthy"})()
            )
            
            # Mock RL health  
            mock_rl_health.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: {"status": "healthy"})()
            )
            
            # Mock transformer health (this should not exist yet)
            expected_transformer_health = {
                "status": "healthy",
                "models": {
                    "itransformer": {
                        "loaded": True,
                        "memory_usage_mb": 256.5,
                        "avg_inference_latency_ms": 15.2,
                        "cache_hit_rate": 0.85
                    },
                    "patchtst": {
                        "loaded": True,
                        "memory_usage_mb": 189.3,
                        "avg_inference_latency_ms": 12.8,
                        "cache_hit_rate": 0.92
                    },
                    "timesmixer": {
                        "loaded": False,
                        "error": "Model not loaded"
                    },
                    "timesfm": {
                        "loaded": True,
                        "memory_usage_mb": 512.8,
                        "avg_inference_latency_ms": 28.4,
                        "cache_hit_rate": 0.76
                    }
                },
                "total_memory_usage_mb": 958.6,
                "models_loaded": 3,
                "models_healthy": 3
            }
            mock_transformer_health.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: expected_transformer_health)()
            )
            
            # Make request
            response = client.get("/health")
            
            # This assertion should FAIL initially
            assert response.status_code == 200
            data = response.json()
            
            # Check that transformers are included in components
            assert "transformers" in data["components"]
            
            # Check transformer health structure
            transformer_health = data["components"]["transformers"]
            assert transformer_health["status"] == "healthy"
            assert transformer_health["models_loaded"] == 3
            assert transformer_health["models_healthy"] == 3
            assert transformer_health["total_memory_usage_mb"] == 958.6
            
            # Check individual model health
            assert "itransformer" in transformer_health["models"]
            assert transformer_health["models"]["itransformer"]["loaded"] is True
            assert transformer_health["models"]["itransformer"]["memory_usage_mb"] == 256.5
            
            assert "patchtst" in transformer_health["models"]
            assert transformer_health["models"]["patchtst"]["avg_inference_latency_ms"] == 12.8
            
            assert "timesmixer" in transformer_health["models"]
            assert transformer_health["models"]["timesmixer"]["loaded"] is False
            
            assert "timesfm" in transformer_health["models"]
            assert transformer_health["models"]["timesfm"]["cache_hit_rate"] == 0.76
    
    def test_health_endpoint_transformer_error_handling(self, client):
        """Test health endpoint handles transformer health check failures"""
        # This test should FAIL initially
        with patch('main.get_config') as mock_config, \
             patch('main.check_database_health') as mock_db_health, \
             patch('main.get_ml_model_health') as mock_ml_health, \
             patch('main.get_rl_agent_health') as mock_rl_health, \
             patch('main.get_transformer_health') as mock_transformer_health:
            
            # Mock basic components
            mock_config_obj = MagicMock()
            mock_config_obj.app.version = "1.0.0"
            mock_config_obj.app.environment = "test"
            mock_config.return_value = mock_config_obj
            
            mock_db_health.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: {"status": "healthy"})()
            )
            mock_ml_health.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: {"status": "healthy"})()
            )
            mock_rl_health.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: {"status": "healthy"})()
            )
            
            # Mock transformer health failure
            async def failing_transformer_health():
                raise Exception("Transformer initialization failed")
            
            mock_transformer_health.return_value = asyncio.create_task(failing_transformer_health())
            
            # Make request
            response = client.get("/health")
            
            # Should still return 200 but with error status for transformers
            assert response.status_code == 200
            data = response.json()
            
            # Overall status should be degraded due to transformer failure
            assert data["status"] == "degraded"
            
            # Transformer component should show error
            assert data["components"]["transformers"]["status"] == "error"
            assert "error" in data["components"]["transformers"]


class TestTransformerHealthFunctions:
    """Test individual transformer health check functions"""
    
    @pytest.mark.asyncio
    async def test_get_transformer_health_function_exists(self):
        """Test that get_transformer_health function exists"""
        # This test should FAIL initially
        from main import get_transformer_health
        
        # Function should exist and be callable
        assert callable(get_transformer_health)
        
        # Should return proper health structure
        health_result = await get_transformer_health()
        
        # Check required fields
        assert "status" in health_result
        assert "models" in health_result
        assert "total_memory_usage_mb" in health_result
        assert "models_loaded" in health_result
        assert "models_healthy" in health_result
    
    @pytest.mark.asyncio
    async def test_transformer_health_model_details(self):
        """Test transformer health includes detailed model information"""
        # This test should FAIL initially
        from main import get_transformer_health
        
        with patch('src.ml_analysis.transformers.itransformer.iTransformerPredictor') as mock_itransformer, \
             patch('src.ml_analysis.transformers.patchtst.PatchTSTPredictor') as mock_patchtst, \
             patch('src.ml_analysis.transformers.timesmixer.TimesMixerPredictor') as mock_timesmixer, \
             patch('src.ml_analysis.transformers.timesfm_wrapper.TimesFMWrapper') as mock_timesfm:
            
            # Mock individual transformer health
            mock_itransformer_instance = MagicMock()
            mock_itransformer_instance.health_check.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: True)()
            )
            mock_itransformer_instance.get_memory_usage.return_value = 256.5
            mock_itransformer_instance.get_avg_inference_latency.return_value = 15.2
            mock_itransformer_instance.get_cache_hit_rate.return_value = 0.85
            mock_itransformer.return_value = mock_itransformer_instance
            
            mock_patchtst_instance = MagicMock()
            mock_patchtst_instance.health_check.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: True)()
            )
            mock_patchtst_instance.get_memory_usage.return_value = 189.3
            mock_patchtst_instance.get_avg_inference_latency.return_value = 12.8
            mock_patchtst_instance.get_cache_hit_rate.return_value = 0.92
            mock_patchtst.return_value = mock_patchtst_instance
            
            # TimesMixer should fail to load
            mock_timesmixer.side_effect = Exception("Model initialization failed")
            
            mock_timesfm_instance = MagicMock()
            mock_timesfm_instance.health_check.return_value = asyncio.create_task(
                asyncio.coroutine(lambda: True)()
            )
            mock_timesfm_instance.get_memory_usage.return_value = 512.8
            mock_timesfm_instance.get_avg_inference_latency.return_value = 28.4
            mock_timesfm_instance.get_cache_hit_rate.return_value = 0.76
            mock_timesfm.return_value = mock_timesfm_instance
            
            # Get health status
            health_result = await get_transformer_health()
            
            # Check model details
            assert health_result["models_loaded"] == 3
            assert health_result["models_healthy"] == 3
            assert health_result["total_memory_usage_mb"] == 958.6
            
            # Check individual models
            itransformer_health = health_result["models"]["itransformer"]
            assert itransformer_health["loaded"] is True
            assert itransformer_health["memory_usage_mb"] == 256.5
            assert itransformer_health["avg_inference_latency_ms"] == 15.2
            assert itransformer_health["cache_hit_rate"] == 0.85
            
            # TimesMixer should show as failed
            timesmixer_health = health_result["models"]["timesmixer"]
            assert timesmixer_health["loaded"] is False
            assert "error" in timesmixer_health


class TestTransformerModelHealthMethods:
    """Test individual transformer model health check methods"""
    
    @pytest.mark.asyncio
    async def test_itransformer_health_methods(self):
        """Test iTransformer health check methods"""
        # This test should FAIL initially as methods don't exist
        from src.ml_analysis.transformers.itransformer import iTransformerPredictor
        from src.ml_analysis.transformers.base import TransformerConfig
        from src.ml_analysis.base import ModelType
        
        config = {
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6
        }
        
        predictor = iTransformerPredictor(ModelType.ITRANSFORMER, config)
        
        # These methods should exist and return proper values
        assert hasattr(predictor, 'get_memory_usage')
        assert hasattr(predictor, 'get_avg_inference_latency')
        assert hasattr(predictor, 'get_cache_hit_rate')
        assert hasattr(predictor, 'get_model_cache_status')
        
        # Memory usage should return float in MB
        memory_usage = predictor.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        assert memory_usage > 0
        
        # Latency should return float in milliseconds
        latency = predictor.get_avg_inference_latency()
        assert isinstance(latency, (int, float))
        assert latency >= 0
        
        # Cache hit rate should return float between 0 and 1
        hit_rate = predictor.get_cache_hit_rate()
        assert isinstance(hit_rate, (int, float))
        assert 0 <= hit_rate <= 1
        
        # Cache status should return dict with cache info
        cache_status = predictor.get_model_cache_status()
        assert isinstance(cache_status, dict)
        assert "size" in cache_status
        assert "max_size" in cache_status
        assert "hit_rate" in cache_status
    
    @pytest.mark.asyncio
    async def test_patchtst_health_methods(self):
        """Test PatchTST health check methods"""
        # This test should FAIL initially
        from src.ml_analysis.transformers.patchtst import PatchTSTPredictor
        from src.ml_analysis.base import ModelType
        
        config = {
            'd_model': 512,
            'n_heads': 8,
            'patch_size': 16
        }
        
        predictor = PatchTSTPredictor(ModelType.PATCHTST, config)
        
        # Test health methods
        memory_usage = predictor.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        
        latency = predictor.get_avg_inference_latency()
        assert isinstance(latency, (int, float))
        
        hit_rate = predictor.get_cache_hit_rate()
        assert 0 <= hit_rate <= 1
    
    @pytest.mark.asyncio
    async def test_timesmixer_health_methods(self):
        """Test TimesMixer health check methods"""
        # This test should FAIL initially
        from src.ml_analysis.transformers.timesmixer import TimesMixerPredictor
        from src.ml_analysis.base import ModelType
        
        config = {
            'd_model': 512,
            'seq_len': 336,
            'pred_len': 96
        }
        
        predictor = TimesMixerPredictor(ModelType.TIMESMIXER, config)
        
        # Test health methods
        memory_usage = predictor.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        
        latency = predictor.get_avg_inference_latency()
        assert isinstance(latency, (int, float))
        
        cache_status = predictor.get_model_cache_status()
        assert isinstance(cache_status, dict)
    
    @pytest.mark.asyncio
    async def test_timesfm_health_methods(self):
        """Test TimesFM health check methods"""
        # This test should FAIL initially
        from src.ml_analysis.transformers.timesfm_wrapper import TimesFMWrapper
        from src.ml_analysis.base import ModelType
        
        config = {
            'model_size': 'small',
            'horizon_length': 128
        }
        
        wrapper = TimesFMWrapper(ModelType.TIMESFM, config)
        
        # Test health methods
        memory_usage = wrapper.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        
        latency = wrapper.get_avg_inference_latency()
        assert isinstance(latency, (int, float))
        
        hit_rate = wrapper.get_cache_hit_rate()
        assert 0 <= hit_rate <= 1


class TestTransformerInferenceTracking:
    """Test transformer inference latency and performance tracking"""
    
    @pytest.mark.asyncio
    async def test_inference_latency_tracking(self):
        """Test that transformer models track inference latency"""
        # This test should FAIL initially
        from src.ml_analysis.transformers.base import TransformerBase
        from src.ml_analysis.base import ModelType
        
        # Create a concrete transformer instance for testing
        class TestTransformer(TransformerBase):
            def forward(self, x, attention_mask=None):
                return x.mean(dim=1, keepdim=True)
            
            def get_attention_weights(self):
                return None
        
        config = {
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6
        }
        
        transformer = TestTransformer(ModelType.ITRANSFORMER, config)
        
        # Test inference tracking
        assert hasattr(transformer, '_inference_times')
        assert hasattr(transformer, 'track_inference_latency')
        
        # Simulate inference with tracking
        import torch
        test_input = torch.randn(1, 10, 512)
        
        with transformer.track_inference_latency():
            output = transformer.forward(test_input)
        
        # Check that latency was recorded
        assert len(transformer._inference_times) > 0
        assert transformer._inference_times[-1] > 0
        
        # Test average latency calculation
        avg_latency = transformer.get_avg_inference_latency()
        assert isinstance(avg_latency, (int, float))
        assert avg_latency > 0


class TestTransformerMemoryTracking:
    """Test transformer memory usage monitoring"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self):
        """Test that transformer models track memory usage"""
        # This test should FAIL initially
        from src.ml_analysis.transformers.base import TransformerBase
        from src.ml_analysis.base import ModelType
        
        class TestTransformer(TransformerBase):
            def forward(self, x, attention_mask=None):
                return x.mean(dim=1, keepdim=True)
            
            def get_attention_weights(self):
                return None
        
        config = {
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6
        }
        
        transformer = TestTransformer(ModelType.ITRANSFORMER, config)
        
        # Test memory tracking methods
        assert hasattr(transformer, 'get_memory_usage')
        assert hasattr(transformer, 'get_peak_memory_usage')
        assert hasattr(transformer, 'reset_memory_tracking')
        
        # Get current memory usage
        memory_usage = transformer.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        assert memory_usage > 0
        
        # Get peak memory usage
        peak_memory = transformer.get_peak_memory_usage()
        assert isinstance(peak_memory, (int, float))
        assert peak_memory >= memory_usage
        
        # Test memory tracking reset
        transformer.reset_memory_tracking()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])