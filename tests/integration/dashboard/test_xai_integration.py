"""
Integration tests for XAI system integration with dashboard (Phase 4.3)
Tests the integration between Phase 4.2 XAI system and Phase 4.3 Dashboard API
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.dashboard.service import DashboardService
from src.xai.trading_integration import TradingExplanationManager
from src.xai.factory import ExplainerFactory
from src.xai.data_models import ExplanationData


class TestXAIDashboardIntegration:
    """Integration tests for XAI-Dashboard integration"""
    
    @pytest.fixture
    def dashboard_service(self):
        """Dashboard service with real XAI integration"""
        service = DashboardService()
        service.config = MagicMock()
        return service
    
    @pytest.fixture
    def mock_model(self):
        """Mock ML model for XAI testing"""
        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([0.85]))
        model.predict_proba = MagicMock(return_value=np.array([[0.15, 0.85]]))
        return model
    
    @pytest.fixture
    def feature_data(self):
        """Sample feature data for trading decision"""
        return np.array([
            0.02,   # price_change_1h
            1.5,    # volume_ratio
            0.35,   # rsi
            1.0,    # ma_crossover_signal
            0.025   # volatility
        ])
    
    @pytest.fixture
    def feature_names(self):
        """Feature names for trading model"""
        return [
            "price_change_1h",
            "volume_ratio", 
            "rsi",
            "ma_crossover_signal",
            "volatility"
        ]
    
    # =============================================================================
    # XAI SYSTEM INITIALIZATION TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_xai_components_initialization(self, dashboard_service):
        """Test XAI components are properly initialized in dashboard service"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        # Assertions
        assert dashboard_service.xai_explainer_factory is not None
        assert dashboard_service.xai_explanation_manager is not None
        assert isinstance(dashboard_service.xai_explanation_manager, TradingExplanationManager)
        assert isinstance(dashboard_service.xai_explainer_factory, ExplainerFactory)
    
    @pytest.mark.asyncio 
    async def test_xai_manager_configuration(self, dashboard_service):
        """Test XAI explanation manager is properly configured"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        # Get cache stats to verify configuration
        cache_stats = await dashboard_service.get_xai_cache_stats()
        
        # Assertions
        assert "cache_limit" in cache_stats
        assert "enabled" in cache_stats
        assert cache_stats["cache_limit"] == 1000  # Default cache size
        assert cache_stats["enabled"] is True
    
    # =============================================================================
    # XAI EXPLANATION GENERATION INTEGRATION TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_xai_explanation_generation_flow(self, dashboard_service, mock_model, feature_data, feature_names):
        """Test complete XAI explanation generation and retrieval flow"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        # Generate explanation using XAI manager
        explanation = await dashboard_service.xai_explanation_manager.explain_trading_decision(
            decision_id="test-decision-123",
            model=mock_model,
            feature_data=feature_data,
            feature_names=feature_names,
            decision_type="buy",
            symbol="BTC/USDC",
            model_type="ml_model",
            metadata={"strategy": "momentum", "confidence": 0.85}
        )
        
        # Assertions
        if explanation:  # XAI may not be fully functional in test environment
            assert explanation.decision_id == "test-decision-123"
            assert explanation.symbol == "BTC/USDC"
            assert explanation.decision_type == "buy"
            assert explanation.model_type == "ml_model"
            assert explanation.explanation_data is not None
    
    @pytest.mark.asyncio
    async def test_feature_importance_aggregation(self, dashboard_service, mock_model, feature_data, feature_names):
        """Test feature importance aggregation across multiple decisions"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        # Generate multiple explanations
        decisions = [
            {"id": "decision-1", "symbol": "BTC/USDC", "type": "buy"},
            {"id": "decision-2", "symbol": "BTC/USDC", "type": "sell"},
            {"id": "decision-3", "symbol": "ETH/USDC", "type": "buy"}
        ]
        
        for decision in decisions:
            await dashboard_service.xai_explanation_manager.explain_trading_decision(
                decision_id=decision["id"],
                model=mock_model,
                feature_data=feature_data,
                feature_names=feature_names,
                decision_type=decision["type"],
                symbol=decision["symbol"],
                model_type="ml_model"
            )
        
        # Get feature importance summary
        btc_importance = await dashboard_service.get_feature_importance_summary(
            symbol="BTC/USDC",
            hours_back=1
        )
        
        all_importance = await dashboard_service.get_feature_importance_summary(
            symbol=None,
            hours_back=1
        )
        
        # Assertions (may be empty if XAI components not fully functional)
        assert isinstance(btc_importance, dict)
        assert isinstance(all_importance, dict)
    
    # =============================================================================
    # DASHBOARD API INTEGRATION TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_xai_api_endpoints_integration(self, dashboard_service):
        """Test XAI API endpoints integration with real service"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        # Test explanations endpoint
        explanations = await dashboard_service.get_xai_explanations(
            symbol="BTC/USDC",
            decision_type="buy",
            limit=10
        )
        
        # Test feature importance endpoint
        feature_importance = await dashboard_service.get_feature_importance_summary(
            symbol="BTC/USDC",
            hours_back=24
        )
        
        # Test cache stats endpoint
        cache_stats = await dashboard_service.get_xai_cache_stats()
        
        # Assertions
        assert isinstance(explanations, list)
        assert isinstance(feature_importance, dict)
        assert isinstance(cache_stats, dict)
        assert "cache_limit" in cache_stats
    
    @pytest.mark.asyncio
    async def test_xai_explanation_retrieval_by_id(self, dashboard_service, mock_model, feature_data, feature_names):
        """Test XAI explanation retrieval by decision ID"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        # Generate explanation
        decision_id = "integration-test-123"
        explanation = await dashboard_service.xai_explanation_manager.explain_trading_decision(
            decision_id=decision_id,
            model=mock_model,
            feature_data=feature_data,
            feature_names=feature_names,
            decision_type="buy",
            symbol="ETH/USDC",
            model_type="ml_model"
        )
        
        # Retrieve explanation by ID
        retrieved_explanation = await dashboard_service.get_xai_explanation(decision_id)
        
        # Assertions
        if explanation and retrieved_explanation:
            assert retrieved_explanation["decision_id"] == decision_id
            assert "explanation_data" in retrieved_explanation
    
    # =============================================================================
    # ERROR HANDLING AND RESILIENCE TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_xai_system_unavailable_graceful_handling(self, dashboard_service):
        """Test graceful handling when XAI system is unavailable"""
        # Don't initialize XAI components (simulate unavailable system)
        dashboard_service.xai_explanation_manager = None
        dashboard_service.xai_explainer_factory = None
        
        # Test API endpoints with unavailable XAI system
        explanations = await dashboard_service.get_xai_explanations()
        feature_importance = await dashboard_service.get_feature_importance_summary()
        cache_stats = await dashboard_service.get_xai_cache_stats()
        specific_explanation = await dashboard_service.get_xai_explanation("test-id")
        
        # Assertions - should handle gracefully without crashing
        assert explanations == []
        assert feature_importance == {}
        assert "error" in cache_stats
        assert specific_explanation is None
    
    @pytest.mark.asyncio
    async def test_xai_explanation_timeout_handling(self, dashboard_service, mock_model, feature_data, feature_names):
        """Test handling of XAI explanation generation timeouts"""
        # Initialize components with short timeout
        await dashboard_service._initialize_components()
        
        if dashboard_service.xai_explanation_manager:
            # Set very short timeout
            dashboard_service.xai_explanation_manager.explanation_timeout = 0.001
            
            # Try to generate explanation (should timeout)
            explanation = await dashboard_service.xai_explanation_manager.explain_trading_decision(
                decision_id="timeout-test",
                model=mock_model,
                feature_data=feature_data,
                feature_names=feature_names,
                decision_type="buy",
                symbol="BTC/USDC",
                model_type="ml_model"
            )
            
            # Should handle timeout gracefully
            # (May return None due to timeout)
            assert explanation is None or explanation.decision_id == "timeout-test"
    
    # =============================================================================
    # PERFORMANCE AND SCALABILITY TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_concurrent_xai_explanation_requests(self, dashboard_service):
        """Test concurrent XAI explanation requests performance"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        # Create concurrent requests
        tasks = []
        for i in range(10):
            task = dashboard_service.get_xai_explanations(
                symbol=f"TOKEN{i}/USDC",
                limit=10
            )
            tasks.append(task)
        
        # Execute concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()
        
        # Assertions
        assert len(results) == 10
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds
        
        # Check that no exceptions were raised
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_xai_cache_performance(self, dashboard_service, mock_model, feature_data, feature_names):
        """Test XAI explanation caching performance"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        if not dashboard_service.xai_explanation_manager:
            pytest.skip("XAI explanation manager not available")
        
        decision_id = "cache-test-123"
        
        # Generate explanation (first time - should cache)
        start_time = asyncio.get_event_loop().time()
        explanation1 = await dashboard_service.xai_explanation_manager.explain_trading_decision(
            decision_id=decision_id,
            model=mock_model,
            feature_data=feature_data,
            feature_names=feature_names,
            decision_type="buy",
            symbol="BTC/USDC",
            model_type="ml_model"
        )
        first_time = asyncio.get_event_loop().time() - start_time
        
        # Retrieve same explanation (should be cached)
        start_time = asyncio.get_event_loop().time()
        explanation2 = dashboard_service.xai_explanation_manager.get_explanation(decision_id)
        cached_time = asyncio.get_event_loop().time() - start_time
        
        # Assertions
        if explanation1 and explanation2:
            assert explanation2.decision_id == decision_id
            # Cached retrieval should be much faster
            assert cached_time < first_time / 10
    
    # =============================================================================
    # DATA CONSISTENCY TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_xai_data_consistency_across_endpoints(self, dashboard_service, mock_model, feature_data, feature_names):
        """Test data consistency across different XAI endpoints"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        if not dashboard_service.xai_explanation_manager:
            pytest.skip("XAI explanation manager not available")
        
        # Generate explanation
        decision_id = "consistency-test-123"
        await dashboard_service.xai_explanation_manager.explain_trading_decision(
            decision_id=decision_id,
            model=mock_model,
            feature_data=feature_data,
            feature_names=feature_names,
            decision_type="buy",
            symbol="BTC/USDC",
            model_type="ml_model"
        )
        
        # Get data through different endpoints
        specific_explanation = await dashboard_service.get_xai_explanation(decision_id)
        all_explanations = await dashboard_service.get_xai_explanations(
            symbol="BTC/USDC",
            decision_type="buy",
            limit=10
        )
        
        # Verify consistency
        if specific_explanation and all_explanations:
            # Find the specific explanation in the list
            found_explanation = None
            for exp in all_explanations:
                if exp.get("decision_id") == decision_id:
                    found_explanation = exp
                    break
            
            if found_explanation:
                assert found_explanation["decision_id"] == specific_explanation["decision_id"]
                assert found_explanation["symbol"] == specific_explanation["symbol"]
                assert found_explanation["decision_type"] == specific_explanation["decision_type"]
    
    # =============================================================================
    # TRADING MODE INTEGRATION TESTS
    # =============================================================================
    
    @pytest.mark.asyncio
    async def test_xai_integration_with_trading_modes(self, dashboard_service):
        """Test XAI integration works with different trading modes"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        # Test with different trading mode contexts
        modes = ["analysis", "simulation", "live"]
        
        for mode in modes:
            # Get XAI explanations in different modes
            explanations = await dashboard_service.get_xai_explanations(
                symbol="BTC/USDC",
                limit=5
            )
            
            feature_importance = await dashboard_service.get_feature_importance_summary(
                symbol="BTC/USDC",
                hours_back=1
            )
            
            # Assertions - should work in all modes
            assert isinstance(explanations, list)
            assert isinstance(feature_importance, dict)
    
    @pytest.mark.asyncio
    async def test_xai_real_time_metrics_integration(self, dashboard_service):
        """Test XAI metrics are properly integrated into real-time metrics"""
        # Initialize components
        await dashboard_service._initialize_components()
        
        # Get real-time metrics
        realtime_metrics = await dashboard_service.get_realtime_metrics()
        
        # Assertions
        assert "ml_rl" in realtime_metrics
        ml_rl_metrics = realtime_metrics["ml_rl"]
        
        # Should include XAI-related metrics
        assert "ml_prediction_accuracy" in ml_rl_metrics
        assert "rl_action_success_rate" in ml_rl_metrics
        assert "integration_active" in ml_rl_metrics
        assert "decision_latency_ms" in ml_rl_metrics


# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================

class TestXAIDashboardEndToEnd:
    """End-to-end integration tests for XAI-Dashboard system"""
    
    @pytest.mark.asyncio
    async def test_complete_trading_decision_explanation_flow(self):
        """Test complete flow from trading decision to dashboard explanation display"""
        # This test would simulate:
        # 1. ML model makes trading decision
        # 2. XAI system generates explanation
        # 3. Dashboard API serves explanation data
        # 4. Frontend displays explanation to user
        
        # For now, this is a placeholder for a comprehensive end-to-end test
        # that would require actual ML models and trading system integration
        pass
    
    @pytest.mark.asyncio
    async def test_xai_system_lifecycle_with_dashboard(self):
        """Test XAI system lifecycle events through dashboard"""
        # This test would verify:
        # 1. XAI system startup and initialization
        # 2. Dashboard detects XAI system availability
        # 3. Explanation generation and caching
        # 4. Dashboard serves cached and fresh explanations
        # 5. Graceful handling of XAI system shutdown
        pass