"""
Tests for XAI Trading Integration.

Following TDD principles, these tests define the expected behavior
of the trading explanation manager before/after implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import numpy as np

from src.xai.trading_integration import TradingExplanationManager, TradingExplanation
from src.xai.data_models import ExplanationData
from src.xai.factory import ExplainerFactory


class TestTradingExplanationManager:
    """Test suite for trading explanation manager."""
    
    @pytest.fixture
    def mock_explainer(self):
        """Create mock explainer."""
        explainer = Mock()
        explainer.explain_instance.return_value = ExplanationData(
            feature_importance={'price_change': 0.8, 'volume': 0.3},
            explanation_type='permutation',
            instance_data=[1.0, 2.0, 3.0],
            model_prediction=0.7,
            confidence_score=0.9
        )
        return explainer
    
    @pytest.fixture
    def mock_factory(self, mock_explainer):
        """Create mock explainer factory."""
        factory = Mock(spec=ExplainerFactory)
        factory.is_supported.return_value = True
        factory.create_explainer.return_value = mock_explainer
        return factory
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics collector."""
        metrics = Mock()
        return metrics
    
    @pytest.fixture
    def explanation_manager(self, mock_factory, mock_metrics):
        """Create explanation manager with mocks."""
        return TradingExplanationManager(
            explainer_factory=mock_factory,
            metrics_collector=mock_metrics,
            cache_size=10
        )
    
    def test_explanation_manager_initialization(self):
        """Test explanation manager initialization."""
        manager = TradingExplanationManager()
        
        assert manager.explainer_factory is not None
        assert manager.cache_size == 1000
        assert manager.explanation_timeout == 5.0
        assert manager._enabled is True
    
    @pytest.mark.asyncio
    async def test_explain_trading_decision_success(self, explanation_manager, mock_factory, mock_metrics):
        """Test successful trading decision explanation."""
        model = Mock()
        feature_data = [1.0, 2.0, 3.0]
        feature_names = ['price_change', 'volume', 'momentum']
        
        result = await explanation_manager.explain_trading_decision(
            decision_id='test_decision_1',
            model=model,
            feature_data=feature_data,
            feature_names=feature_names,
            decision_type='buy',
            symbol='BTC/USD'
        )
        
        assert result is not None
        assert isinstance(result, TradingExplanation)
        assert result.decision_id == 'test_decision_1'
        assert result.decision_type == 'buy'
        assert result.symbol == 'BTC/USD'
        assert result.explanation_data is not None
        
        # Verify factory was called
        mock_factory.create_explainer.assert_called_once()
        
        # Verify metrics were updated
        mock_metrics.increment_counter.assert_called()
    
    @pytest.mark.asyncio
    async def test_explain_trading_decision_caching(self, explanation_manager):
        """Test that explanations are cached properly."""
        model = Mock()
        feature_data = [1.0, 2.0, 3.0]
        feature_names = ['price_change', 'volume', 'momentum']
        
        # First call
        result1 = await explanation_manager.explain_trading_decision(
            decision_id='test_decision_cache',
            model=model,
            feature_data=feature_data,
            feature_names=feature_names,
            decision_type='buy',
            symbol='BTC/USD'
        )
        
        # Second call with same decision_id should return cached result
        result2 = await explanation_manager.explain_trading_decision(
            decision_id='test_decision_cache',
            model=model,
            feature_data=feature_data,
            feature_names=feature_names,
            decision_type='buy',
            symbol='BTC/USD'
        )
        
        assert result1 is result2  # Same object from cache
    
    @pytest.mark.asyncio
    async def test_explain_trading_decision_disabled(self, explanation_manager):
        """Test explanation generation when disabled."""
        explanation_manager.set_enabled(False)
        
        result = await explanation_manager.explain_trading_decision(
            decision_id='test_disabled',
            model=Mock(),
            feature_data=[1.0, 2.0],
            feature_names=['feature1', 'feature2'],
            decision_type='sell',
            symbol='ETH/USD'
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_explain_trading_decision_timeout(self, explanation_manager, mock_factory):
        """Test explanation generation timeout handling."""
        # Mock explainer that takes too long
        slow_explainer = Mock()
        slow_explainer.explain_instance = AsyncMock(side_effect=asyncio.sleep(10))
        mock_factory.create_explainer.return_value = slow_explainer
        
        explanation_manager.explanation_timeout = 0.1  # Very short timeout
        
        result = await explanation_manager.explain_trading_decision(
            decision_id='test_timeout',
            model=Mock(),
            feature_data=[1.0, 2.0],
            feature_names=['feature1', 'feature2'],
            decision_type='buy',
            symbol='BTC/USD'
        )
        
        assert result is None
    
    @pytest.mark.asyncio  
    async def test_explain_trading_decision_fallback_explainers(self, explanation_manager, mock_factory):
        """Test fallback to alternative explainers when primary fails."""
        # First explainer fails, second succeeds
        explainer1 = Mock()
        explainer1.explain_instance.side_effect = Exception("Primary explainer failed")
        
        explainer2 = Mock()
        explainer2.explain_instance.return_value = ExplanationData(
            feature_importance={'price': 0.5},
            explanation_type='lime',
            instance_data=[1.0],
            model_prediction=0.6
        )
        
        mock_factory.create_explainer.side_effect = [explainer1, explainer2]
        mock_factory.is_supported.return_value = True
        
        result = await explanation_manager.explain_trading_decision(
            decision_id='test_fallback',
            model=Mock(),
            feature_data=[1.0],
            feature_names=['price'],
            decision_type='buy',
            symbol='BTC/USD'
        )
        
        assert result is not None
        assert result.explanation_data.explanation_type == 'lime'
    
    def test_get_explanation(self, explanation_manager):
        """Test retrieving cached explanation."""
        # Add explanation to cache manually
        explanation = TradingExplanation(
            decision_id='test_get',
            timestamp=datetime.utcnow().isoformat() + 'Z',
            decision_type='buy',
            symbol='BTC/USD',
            explanation_data=Mock(),
            model_type='ml_model',
            confidence=0.8,
            metadata={}
        )
        explanation_manager._explanation_cache['test_get'] = explanation
        
        result = explanation_manager.get_explanation('test_get')
        assert result is explanation
        
        # Test non-existent explanation
        result = explanation_manager.get_explanation('non_existent')
        assert result is None
    
    def test_get_recent_explanations(self, explanation_manager):
        """Test filtering recent explanations."""
        # Add multiple explanations to cache
        explanations = []
        for i in range(5):
            explanation = TradingExplanation(
                decision_id=f'decision_{i}',
                timestamp=datetime.utcnow().isoformat() + 'Z',
                decision_type='buy' if i % 2 == 0 else 'sell',
                symbol='BTC/USD' if i < 3 else 'ETH/USD',
                explanation_data=Mock(),
                model_type='ml_model',
                confidence=0.8,
                metadata={}
            )
            explanations.append(explanation)
            explanation_manager._explanation_cache[f'decision_{i}'] = explanation
        
        # Test no filters
        recent = explanation_manager.get_recent_explanations()
        assert len(recent) == 5
        
        # Test symbol filter
        btc_explanations = explanation_manager.get_recent_explanations(symbol='BTC/USD')
        assert len(btc_explanations) == 3
        assert all(e.symbol == 'BTC/USD' for e in btc_explanations)
        
        # Test decision_type filter
        buy_explanations = explanation_manager.get_recent_explanations(decision_type='buy')
        assert len(buy_explanations) == 3
        assert all(e.decision_type == 'buy' for e in buy_explanations)
        
        # Test limit
        limited = explanation_manager.get_recent_explanations(limit=2)
        assert len(limited) == 2
    
    def test_get_feature_importance_summary(self, explanation_manager):
        """Test aggregated feature importance calculation."""
        # Create explanations with known feature importance
        for i in range(3):
            explanation_data = ExplanationData(
                feature_importance={'price': 0.5 + i * 0.1, 'volume': 0.3 - i * 0.1},
                explanation_type='test',
                instance_data=[1.0],
                model_prediction=0.5
            )
            
            explanation = TradingExplanation(
                decision_id=f'summary_test_{i}',
                timestamp=datetime.utcnow().isoformat() + 'Z',
                decision_type='buy',
                symbol='BTC/USD',
                explanation_data=explanation_data,
                model_type='ml_model',
                confidence=0.8,
                metadata={}
            )
            explanation_manager._explanation_cache[f'summary_test_{i}'] = explanation
        
        summary = explanation_manager.get_feature_importance_summary()
        
        assert 'price' in summary
        assert 'volume' in summary
        
        # Check that values are averaged (using absolute values)
        expected_price_avg = (0.5 + 0.6 + 0.7) / 3
        expected_volume_avg = (0.3 + 0.2 + 0.1) / 3
        
        assert abs(summary['price'] - expected_price_avg) < 0.01
        assert abs(summary['volume'] - expected_volume_avg) < 0.01
    
    def test_cache_size_limit(self, explanation_manager):
        """Test that cache respects size limit."""
        explanation_manager.cache_size = 3
        
        # Add more explanations than cache limit
        for i in range(5):
            explanation = TradingExplanation(
                decision_id=f'cache_limit_{i}',
                timestamp=(datetime.utcnow() + timedelta(seconds=i)).isoformat() + 'Z',
                decision_type='buy',
                symbol='BTC/USD',
                explanation_data=Mock(),
                model_type='ml_model',
                confidence=0.8,
                metadata={}
            )
            explanation_manager._cache_explanation(f'cache_limit_{i}', explanation)
        
        # Should only have 3 explanations (the most recent ones)
        assert len(explanation_manager._explanation_cache) == 3
        
        # Should have the last 3 explanations
        for i in range(2, 5):
            assert f'cache_limit_{i}' in explanation_manager._explanation_cache
    
    def test_clear_cache(self, explanation_manager):
        """Test cache clearing functionality."""
        # Add some explanations
        explanation_manager._explanation_cache['test1'] = Mock()
        explanation_manager._explainer_cache['explainer1'] = Mock()
        
        explanation_manager.clear_cache()
        
        assert len(explanation_manager._explanation_cache) == 0
        assert len(explanation_manager._explainer_cache) == 0
    
    def test_get_cache_stats(self, explanation_manager):
        """Test cache statistics reporting."""
        # Add some cached items
        explanation_manager._explanation_cache['test1'] = Mock()
        explanation_manager._explainer_cache['explainer1'] = Mock()
        
        stats = explanation_manager.get_cache_stats()
        
        assert stats['explanation_cache_size'] == 1
        assert stats['explainer_cache_size'] == 1
        assert stats['cache_limit'] == 10
        assert stats['enabled'] is True
        assert 'default_explainer_type' in stats
    
    def test_to_dict_serialization(self, explanation_manager):
        """Test trading explanation serialization."""
        explanation_data = ExplanationData(
            feature_importance={'price': 0.8},
            explanation_type='test',
            instance_data=[1.0],
            model_prediction=0.7
        )
        
        trading_explanation = TradingExplanation(
            decision_id='serialize_test',
            timestamp='2024-01-01T12:00:00Z',
            decision_type='buy',
            symbol='BTC/USD',
            explanation_data=explanation_data,
            model_type='ml_model',
            confidence=0.9,
            metadata={'test': 'value'}
        )
        
        result_dict = explanation_manager.to_dict(trading_explanation)
        
        assert isinstance(result_dict, dict)
        assert result_dict['decision_id'] == 'serialize_test'
        assert result_dict['decision_type'] == 'buy'
        assert result_dict['symbol'] == 'BTC/USD'
        assert isinstance(result_dict['explanation_data'], dict)
        assert result_dict['explanation_data']['feature_importance'] == {'price': 0.8}