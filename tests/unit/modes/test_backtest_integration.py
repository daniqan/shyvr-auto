"""
Tests for Backtest Result Integration with Continuous Learning

Following TDD methodology - comprehensive failing tests before implementation.
Tests define requirements for lightweight integration between backtest results
and the continuous learning system for validation purposes.
"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

import numpy as np

from src.modes.backtest_integration import (
    BacktestResultIntegrator,
    BacktestIntegrationConfig,
    ValidationDataPoint,
    ModelPerformanceComparison,
    IntegrationStatus,
    BacktestValidationError
)
from src.modes.analysis_mode import BacktestResult
from src.modes.continuous_learning import ContinuousLearningEngine
from src.rl_agent.base import TradeAction, MarketState
from src.rl_agent.experience_replay import Experience
from src.discovery.base import DiscoveredToken, TokenStatus
from src.utils.base import Chain


class TestBacktestIntegrationConfig:
    """Test configuration for backtest integration"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = BacktestIntegrationConfig()
        
        assert config.enabled == False  # Disabled by default for minimal coupling
        assert config.validation_enabled == True
        assert config.min_backtest_trades == 10
        assert config.max_validation_age_days == 30
        assert config.performance_comparison_window == 100
        assert config.validation_sample_ratio == 0.1
        assert config.store_validation_history == True
        
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid sample ratio
        with pytest.raises(ValueError, match="validation_sample_ratio must be between 0.0 and 1.0"):
            BacktestIntegrationConfig(validation_sample_ratio=1.5)
            
        # Test invalid age days
        with pytest.raises(ValueError, match="max_validation_age_days must be positive"):
            BacktestIntegrationConfig(max_validation_age_days=-1)
            
        # Test invalid min trades
        with pytest.raises(ValueError, match="min_backtest_trades must be positive"):
            BacktestIntegrationConfig(min_backtest_trades=0)


class TestValidationDataPoint:
    """Test validation data point structure"""
    
    def test_validation_data_point_creation(self):
        """Test creation of validation data points"""
        token = DiscoveredToken(
            address="test_address",
            chain=Chain.SOLANA,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=100.0,
            volume_24h=1000.0
        )
        
        market_state = MarketState(
            token=token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000.0,
            rsi=70.0,
            macd=0.5,
            timestamp=datetime.now()
        )
        
        action = TradeAction.BUY
        actual_outcome = 0.05  # 5% return
        confidence = 0.8
        
        data_point = ValidationDataPoint(
            market_state=market_state,
            action=action,
            actual_outcome=actual_outcome,
            confidence=confidence,
            timestamp=datetime.now(),
            source="backtest"
        )
        
        assert data_point.market_state == market_state
        assert data_point.action == action
        assert data_point.actual_outcome == actual_outcome
        assert data_point.confidence == confidence
        assert data_point.source == "backtest"
        
    def test_validation_data_point_serialization(self):
        """Test serialization of validation data points"""
        token = DiscoveredToken(
            address="test_address",
            chain=Chain.SOLANA,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test"
        )
        
        market_state = MarketState(
            token=token,
            price_usd=100.0,
            price_change_24h=0.0,
            volume_24h=1000.0,
            timestamp=datetime.now()
        )
        
        data_point = ValidationDataPoint(
            market_state=market_state,
            action=TradeAction.HOLD,
            actual_outcome=0.0,
            confidence=0.6,
            timestamp=datetime.now(),
            source="backtest"
        )
        
        serialized = data_point.to_dict()
        assert isinstance(serialized, dict)
        assert "market_state" in serialized
        assert "action" in serialized
        assert "actual_outcome" in serialized


class TestModelPerformanceComparison:
    """Test model performance comparison functionality"""
    
    def test_performance_comparison_creation(self):
        """Test creation of performance comparison"""
        backtest_metrics = {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "win_rate": 0.6,
            "max_drawdown": 0.08
        }
        
        live_metrics = {
            "total_return": 0.12,
            "sharpe_ratio": 1.3,
            "win_rate": 0.55,
            "max_drawdown": 0.10
        }
        
        comparison = ModelPerformanceComparison(
            backtest_metrics=backtest_metrics,
            live_metrics=live_metrics,
            comparison_window=30,
            created_at=datetime.now()
        )
        
        assert comparison.backtest_metrics == backtest_metrics
        assert comparison.live_metrics == live_metrics
        assert comparison.comparison_window == 30
        
    def test_performance_deviation_calculation(self):
        """Test calculation of performance deviations"""
        backtest_metrics = {"sharpe_ratio": 1.5, "win_rate": 0.6}
        live_metrics = {"sharpe_ratio": 1.2, "win_rate": 0.55}
        
        comparison = ModelPerformanceComparison(
            backtest_metrics=backtest_metrics,
            live_metrics=live_metrics,
            comparison_window=30,
            created_at=datetime.now()
        )
        
        deviations = comparison.calculate_deviations()
        
        assert abs(deviations["sharpe_ratio"] - -0.2) < 0.01  # (1.2 - 1.5) / 1.5
        assert abs(deviations["win_rate"] - (-0.0833)) < 0.01  # (0.55 - 0.6) / 0.6


class TestBacktestResultIntegrator:
    """Test main BacktestResultIntegrator class"""
    
    @pytest.fixture
    def mock_continuous_learning_engine(self):
        """Mock continuous learning engine"""
        engine = Mock(spec=ContinuousLearningEngine)
        engine.add_validation_data = AsyncMock()
        engine.get_performance_metrics = AsyncMock(return_value={
            "total_return": 0.12,
            "sharpe_ratio": 1.3,
            "win_rate": 0.55
        })
        return engine
    
    @pytest.fixture
    def sample_backtest_result(self):
        """Sample backtest result for testing"""
        return BacktestResult(
            strategy_name="test_strategy",
            total_return=0.15,
            annual_return=0.18,
            max_drawdown=0.08,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            win_rate=0.6,
            total_trades=50,
            profit_factor=1.3,
            trade_history=[
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "BUY",
                    "price": 100.0,
                    "return": 0.05
                },
                {
                    "timestamp": (datetime.now() + timedelta(hours=1)).isoformat(),
                    "action": "SELL",
                    "price": 105.0,
                    "return": 0.05
                }
            ],
            performance_metrics={
                "volatility": 0.15,
                "beta": 1.1,
                "alpha": 0.03
            }
        )
    
    @pytest.fixture
    def integrator_config(self):
        """Default integrator configuration"""
        return BacktestIntegrationConfig(enabled=True)
    
    @pytest.fixture
    def integrator(self, integrator_config, mock_continuous_learning_engine):
        """BacktestResultIntegrator instance for testing"""
        return BacktestResultIntegrator(
            config=integrator_config,
            learning_engine=mock_continuous_learning_engine
        )
    
    def test_integrator_initialization(self, integrator_config):
        """Test integrator initialization"""
        integrator = BacktestResultIntegrator(config=integrator_config)
        
        assert integrator.config == integrator_config
        assert integrator.learning_engine is None  # Optional dependency
        assert integrator.status == IntegrationStatus.INITIALIZED
        assert integrator.validation_history == []
        
    def test_integrator_initialization_with_learning_engine(self, integrator_config, mock_continuous_learning_engine):
        """Test integrator initialization with learning engine"""
        integrator = BacktestResultIntegrator(
            config=integrator_config,
            learning_engine=mock_continuous_learning_engine
        )
        
        assert integrator.learning_engine == mock_continuous_learning_engine
        assert integrator.status == IntegrationStatus.INITIALIZED
        
    def test_disabled_integration(self):
        """Test that disabled integration works gracefully"""
        config = BacktestIntegrationConfig(enabled=False)
        integrator = BacktestResultIntegrator(config=config)
        
        assert integrator.is_enabled() == False
        
    @pytest.mark.asyncio
    async def test_convert_backtest_to_validation_data(self, integrator, sample_backtest_result):
        """Test conversion of backtest results to validation data"""
        validation_data = await integrator.convert_backtest_to_validation_data(sample_backtest_result)
        
        assert isinstance(validation_data, list)
        assert len(validation_data) > 0
        
        # Check first validation data point
        data_point = validation_data[0]
        assert isinstance(data_point, ValidationDataPoint)
        assert data_point.source == "backtest"
        assert data_point.action in [TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD]
        assert isinstance(data_point.actual_outcome, float)
        assert 0.0 <= data_point.confidence <= 1.0
        
    @pytest.mark.asyncio
    async def test_filter_validation_data_by_quality(self, integrator, sample_backtest_result):
        """Test filtering validation data by quality criteria"""
        validation_data = await integrator.convert_backtest_to_validation_data(sample_backtest_result)
        
        # Apply quality filters
        filtered_data = integrator.filter_validation_data_by_quality(validation_data)
        
        # Should filter based on confidence and other quality metrics
        assert len(filtered_data) <= len(validation_data)
        
        # All filtered data should meet quality criteria
        for data_point in filtered_data:
            assert data_point.confidence >= 0.5  # Minimum confidence threshold
            
    @pytest.mark.asyncio
    async def test_feed_validation_data_to_learning_system(self, integrator, sample_backtest_result):
        """Test feeding validation data to learning system"""
        validation_data = await integrator.convert_backtest_to_validation_data(sample_backtest_result)
        
        result = await integrator.feed_validation_data_to_learning_system(validation_data)
        
        assert result is True
        integrator.learning_engine.add_validation_data.assert_called_once()
        
        # Verify the data passed to learning engine
        call_args = integrator.learning_engine.add_validation_data.call_args[0]
        assert len(call_args[0]) > 0  # Should pass validation data
        
    @pytest.mark.asyncio
    async def test_feed_validation_data_without_learning_engine(self):
        """Test feeding validation data without learning engine"""
        config = BacktestIntegrationConfig(enabled=True)
        integrator = BacktestResultIntegrator(config=config)  # No learning engine
        
        token = DiscoveredToken(
            address="test_address",
            chain=Chain.SOLANA,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test"
        )
        
        market_state = MarketState(
            token=token,
            price_usd=100.0,
            price_change_24h=0.0,
            volume_24h=1000.0,
            timestamp=datetime.now()
        )
        
        validation_data = [
            ValidationDataPoint(
                market_state=market_state,
                action=TradeAction.BUY,
                actual_outcome=0.05,
                confidence=0.8,
                timestamp=datetime.now(),
                source="backtest"
            )
        ]
        
        result = await integrator.feed_validation_data_to_learning_system(validation_data)
        
        assert result is False  # Should handle gracefully
        
    @pytest.mark.asyncio
    async def test_compare_model_performance(self, integrator, sample_backtest_result):
        """Test model performance comparison"""
        comparison = await integrator.compare_model_performance(sample_backtest_result)
        
        assert isinstance(comparison, ModelPerformanceComparison)
        assert comparison.backtest_metrics is not None
        assert comparison.live_metrics is not None
        
        # Verify learning engine was called
        integrator.learning_engine.get_performance_metrics.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_track_model_performance_over_time(self, integrator, sample_backtest_result):
        """Test tracking model performance over time"""
        # Process multiple backtest results
        for i in range(3):
            await integrator.process_backtest_result(sample_backtest_result)
            
        performance_history = integrator.get_performance_tracking_history()
        
        assert len(performance_history) > 0
        assert all(isinstance(comparison, ModelPerformanceComparison) for comparison in performance_history)
        
    @pytest.mark.asyncio
    async def test_process_backtest_result_end_to_end(self, integrator, sample_backtest_result):
        """Test end-to-end processing of backtest result"""
        initial_status = integrator.status
        
        result = await integrator.process_backtest_result(sample_backtest_result)
        
        assert result is True
        assert integrator.status == IntegrationStatus.ACTIVE
        
        # Verify validation data was created and fed to learning system
        integrator.learning_engine.add_validation_data.assert_called()
        
        # Verify performance comparison was created
        assert len(integrator.validation_history) > 0
        
    @pytest.mark.asyncio
    async def test_process_insufficient_backtest_trades(self, integrator):
        """Test processing backtest with insufficient trades"""
        # Create backtest result with too few trades
        insufficient_backtest = BacktestResult(
            strategy_name="insufficient_strategy",
            total_return=0.05,
            annual_return=0.06,
            max_drawdown=0.02,
            sharpe_ratio=1.0,
            sortino_ratio=1.2,
            win_rate=0.5,
            total_trades=5,  # Below minimum threshold
            profit_factor=1.1,
            trade_history=[],
            performance_metrics={}
        )
        
        with pytest.raises(BacktestValidationError, match="Insufficient trades"):
            await integrator.process_backtest_result(insufficient_backtest)
            
    @pytest.mark.asyncio
    async def test_validation_data_sampling(self, integrator, sample_backtest_result):
        """Test validation data sampling based on configuration"""
        integrator.config.validation_sample_ratio = 0.1  # 10% sampling
        
        validation_data = await integrator.convert_backtest_to_validation_data(sample_backtest_result)
        sampled_data = integrator.sample_validation_data(validation_data)
        
        # Should sample approximately 10% of data
        expected_sample_size = max(1, int(len(validation_data) * 0.1))
        assert len(sampled_data) <= expected_sample_size + 1  # Allow for rounding
        
    def test_get_integration_metrics(self, integrator):
        """Test getting integration metrics"""
        metrics = integrator.get_integration_metrics()
        
        assert isinstance(metrics, dict)
        assert "status" in metrics
        assert "validation_data_points_processed" in metrics
        assert "performance_comparisons_created" in metrics
        assert "last_integration_time" in metrics
        
    def test_integration_status_management(self, integrator):
        """Test integration status management"""
        assert integrator.status == IntegrationStatus.INITIALIZED
        
        integrator.set_status(IntegrationStatus.ACTIVE)
        assert integrator.status == IntegrationStatus.ACTIVE
        
        integrator.set_status(IntegrationStatus.ERROR)
        assert integrator.status == IntegrationStatus.ERROR
        
    @pytest.mark.asyncio
    async def test_error_handling_in_integration(self, integrator, sample_backtest_result):
        """Test error handling during integration"""
        # Mock learning engine to raise exception
        integrator.learning_engine.add_validation_data.side_effect = Exception("Learning engine error")
        
        # Integration should handle the error gracefully
        result = await integrator.process_backtest_result(sample_backtest_result)
        
        # Should still succeed but log the error
        assert result is True
        
        # Status should remain ACTIVE since overall processing succeeded
        assert integrator.status == IntegrationStatus.ACTIVE
        
    @pytest.mark.asyncio
    async def test_integration_with_validation_disabled(self):
        """Test integration with validation disabled"""
        config = BacktestIntegrationConfig(enabled=True, validation_enabled=False)
        integrator = BacktestResultIntegrator(config=config)
        
        sample_backtest = BacktestResult(
            strategy_name="test",
            total_return=0.1,
            annual_return=0.12,
            max_drawdown=0.05,
            sharpe_ratio=1.2,
            sortino_ratio=1.4,
            win_rate=0.55,
            total_trades=20,
            profit_factor=1.2,
            trade_history=[],
            performance_metrics={}
        )
        
        result = await integrator.process_backtest_result(sample_backtest)
        
        # Should process but skip validation
        assert result is True
        assert integrator.status == IntegrationStatus.ACTIVE


class TestIntegrationWithAnalysisMode:
    """Test integration with AnalysisMode"""
    
    @pytest.mark.asyncio
    async def test_analysis_mode_integration_hook(self):
        """Test that AnalysisMode can optionally call BacktestResultIntegrator"""
        # This test will be implemented after the main class is created
        # It should test the optional integration point in AnalysisMode
        pass
        
    @pytest.mark.asyncio
    async def test_configuration_driven_integration(self):
        """Test that integration is driven by configuration"""
        # Test that integration only happens when enabled in config
        pass


class TestLightweightCoupling:
    """Test that integration maintains lightweight coupling"""
    
    def test_optional_learning_engine_dependency(self):
        """Test that learning engine is optional"""
        config = BacktestIntegrationConfig(enabled=True)
        integrator = BacktestResultIntegrator(config=config)
        
        # Should work without learning engine
        assert integrator.learning_engine is None
        assert integrator.is_enabled() == True
        
    def test_graceful_degradation_without_learning_engine(self):
        """Test graceful degradation when learning engine unavailable"""
        config = BacktestIntegrationConfig(enabled=True)
        integrator = BacktestResultIntegrator(config=config)
        
        # Should handle missing learning engine gracefully
        assert integrator.can_feed_to_learning_system() == False
        
    def test_minimal_configuration_requirements(self):
        """Test minimal configuration requirements"""
        # Should work with minimal configuration
        config = BacktestIntegrationConfig()
        integrator = BacktestResultIntegrator(config=config)
        
        assert integrator is not None
        assert integrator.config == config