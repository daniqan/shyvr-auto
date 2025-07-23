"""
Tests for Advanced Reward Engineering system
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import deque

from src.rl_agent.base import TradeAction, TradingResult
from src.rl_agent.reward_engineering import (
    RiskMetrics, RewardConfig, AdvancedRewardCalculator,
    create_reward_calculator
)
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestRiskMetrics:
    """Test RiskMetrics data structure"""
    
    def test_risk_metrics_creation(self):
        """Test risk metrics creation"""
        metrics = RiskMetrics(
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            max_drawdown=0.12,
            volatility=0.18,
            value_at_risk=-0.03,
            expected_shortfall=-0.05,
            calmar_ratio=2.1
        )
        
        assert metrics.sharpe_ratio == 1.5
        assert metrics.sortino_ratio == 1.8
        assert metrics.max_drawdown == 0.12
        assert metrics.volatility == 0.18
        assert metrics.value_at_risk == -0.03
        assert metrics.expected_shortfall == -0.05
        assert metrics.calmar_ratio == 2.1
    
    def test_risk_metrics_defaults(self):
        """Test risk metrics default values"""
        metrics = RiskMetrics()
        
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.volatility == 0.0
        assert metrics.value_at_risk == 0.0
        assert metrics.expected_shortfall == 0.0
        assert metrics.calmar_ratio == 0.0
    
    def test_risk_metrics_to_dict(self):
        """Test risk metrics dictionary conversion"""
        metrics = RiskMetrics(
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            volatility=0.15
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['sharpe_ratio'] == 1.2
        assert metrics_dict['max_drawdown'] == 0.08
        assert metrics_dict['volatility'] == 0.15
        assert len(metrics_dict) == 7  # All 7 metrics


class TestRewardConfig:
    """Test RewardConfig configuration"""
    
    def test_reward_config_defaults(self):
        """Test default reward configuration"""
        config = RewardConfig()
        
        assert config.risk_free_rate == 0.02
        assert config.target_return == 0.15
        assert config.return_weight == 0.4
        assert config.risk_weight == 0.3
        assert config.consistency_weight == 0.2
        assert config.efficiency_weight == 0.1
        assert config.max_drawdown_threshold == 0.15
        assert config.volatility_threshold == 0.25
        assert config.lookback_window == 252
        assert config.min_observations == 30
        assert config.reward_scale == 100.0
    
    def test_reward_config_custom_values(self):
        """Test custom reward configuration"""
        config = RewardConfig(
            risk_free_rate=0.03,
            target_return=0.20,
            return_weight=0.5,
            risk_weight=0.4,
            lookback_window=100,
            reward_scale=50.0
        )
        
        assert config.risk_free_rate == 0.03
        assert config.target_return == 0.20
        assert config.return_weight == 0.5
        assert config.risk_weight == 0.4
        assert config.lookback_window == 100
        assert config.reward_scale == 50.0
        
        # Check that other values retain defaults
        assert config.consistency_weight == 0.2
        assert config.efficiency_weight == 0.1
    
    def test_reward_config_weight_validation(self):
        """Test that reward component weights are reasonable"""
        config = RewardConfig()
        
        total_weight = (config.return_weight + config.risk_weight + 
                       config.consistency_weight + config.efficiency_weight)
        
        assert abs(total_weight - 1.0) < 0.01  # Should sum to approximately 1.0


class TestAdvancedRewardCalculator:
    """Test Advanced Reward Calculator"""
    
    @pytest.fixture
    def reward_config(self):
        """Create reward configuration for testing"""
        return RewardConfig(
            lookback_window=50,
            min_observations=5,
            reward_scale=10.0  # Smaller scale for testing
        )
    
    @pytest.fixture
    def sample_token(self):
        """Create sample token"""
        return DiscoveredToken(
            address="0x123...",
            symbol="TEST",
            name="Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=1.50,
            volume_24h=100000
        )
    
    @pytest.fixture
    def sample_trading_result(self, sample_token):
        """Create sample trading result"""
        return TradingResult(
            action=TradeAction.BUY,
            token=sample_token,
            executed_at=datetime.now(),
            price=1.50,
            quantity=100,
            value_usd=150.0,
            success=True,
            portfolio_value_before=10000.0,
            portfolio_value_after=10150.0,
            realized_pnl=25.0
        )
    
    def test_calculator_initialization(self, reward_config):
        """Test reward calculator initialization"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        assert calculator.config == reward_config
        assert len(calculator.returns_history) == 0
        assert len(calculator.portfolio_values) == 0
        assert len(calculator.drawdown_history) == 0
        assert len(calculator.trade_results) == 0
        assert calculator.last_metrics is None
    
    def test_simple_reward_calculation(self, reward_config, sample_trading_result):
        """Test simple reward calculation"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        portfolio_before = 10000.0
        portfolio_after = 10150.0  # 1.5% gain
        
        reward, metrics = calculator.calculate_reward(
            portfolio_before, portfolio_after, sample_trading_result
        )
        
        assert isinstance(reward, float)
        assert isinstance(metrics, RiskMetrics)
        assert len(calculator.returns_history) == 1
        assert len(calculator.portfolio_values) == 1
    
    def test_insufficient_data_handling(self, reward_config, sample_trading_result):
        """Test handling of insufficient data"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        # Single observation - should not crash
        reward, metrics = calculator.calculate_reward(
            10000.0, 10100.0, sample_trading_result
        )
        
        assert isinstance(reward, float)
        assert metrics.sharpe_ratio == 0.0  # Not enough data
        assert metrics.max_drawdown == 0.0
    
    def test_multiple_observations(self, reward_config, sample_token):
        """Test reward calculation with multiple observations"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        # Add multiple observations
        base_value = 10000.0
        for i in range(10):
            # Simulate varying returns
            return_pct = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% volatility
            value_after = base_value * (1 + return_pct)
            
            trading_result = TradingResult(
                action=TradeAction.BUY,
                token=sample_token,
                executed_at=datetime.now(),
                price=1.50,
                quantity=100,
                value_usd=150.0,
                success=True,
                portfolio_value_before=base_value,
                portfolio_value_after=value_after
            )
            
            reward, metrics = calculator.calculate_reward(
                base_value, value_after, trading_result
            )
            
            base_value = value_after
        
        assert len(calculator.returns_history) == 10
        assert isinstance(reward, float)
        
        # With enough data, should calculate metrics
        if len(calculator.returns_history) >= reward_config.min_observations:
            assert calculator.last_metrics is not None
    
    def test_risk_metrics_calculation(self, reward_config, sample_token):
        """Test comprehensive risk metrics calculation"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        # Generate returns with known statistics
        np.random.seed(42)  # For reproducible tests
        base_value = 10000.0
        
        # Add enough observations for metrics calculation
        for i in range(reward_config.min_observations + 5):
            return_pct = np.random.normal(0.001, 0.015)  # Positive mean return
            value_after = base_value * (1 + return_pct)
            
            trading_result = TradingResult(
                action=TradeAction.HOLD,
                token=sample_token,
                executed_at=datetime.now(),
                price=1.50,
                quantity=0,
                value_usd=0,
                success=True,
                portfolio_value_before=base_value,
                portfolio_value_after=value_after
            )
            
            reward, metrics = calculator.calculate_reward(
                base_value, value_after, trading_result
            )
            
            base_value = value_after
        
        # Check that metrics were calculated
        assert calculator.last_metrics is not None
        assert calculator.last_metrics.volatility > 0
        assert calculator.last_metrics.sharpe_ratio != 0
        assert calculator.last_metrics.max_drawdown >= 0
    
    def test_drawdown_tracking(self, reward_config, sample_token):
        """Test drawdown calculation and tracking"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        # Simulate portfolio with drawdown
        values = [10000, 10500, 10200, 9800, 9500, 9800, 10200, 10800]
        
        for i in range(len(values) - 1):
            trading_result = TradingResult(
                action=TradeAction.HOLD,
                token=sample_token,
                executed_at=datetime.now(),
                price=1.50,
                quantity=0,
                value_usd=0,
                success=True,
                portfolio_value_before=values[i],
                portfolio_value_after=values[i + 1]
            )
            
            reward, metrics = calculator.calculate_reward(
                values[i], values[i + 1], trading_result
            )
        
        # Maximum value was 10500, minimum after that was 9500
        # So max drawdown should be (10500 - 9500) / 10500 ≈ 0.095
        expected_max_dd = (10500 - 9500) / 10500
        
        if len(calculator.returns_history) >= reward_config.min_observations:
            assert calculator.last_metrics.max_drawdown > 0
            assert abs(calculator.last_metrics.max_drawdown - expected_max_dd) < 0.01
    
    def test_successful_vs_failed_trades(self, reward_config, sample_token):
        """Test reward difference between successful and failed trades"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        # Successful trade
        successful_result = TradingResult(
            action=TradeAction.BUY,
            token=sample_token,
            executed_at=datetime.now(),
            price=1.50,
            quantity=100,
            value_usd=150.0,
            success=True,
            portfolio_value_before=10000.0,
            portfolio_value_after=10150.0,
            realized_pnl=25.0
        )
        
        reward_success, _ = calculator.calculate_reward(
            10000.0, 10150.0, successful_result
        )
        
        # Reset calculator
        calculator.reset()
        
        # Failed trade
        failed_result = TradingResult(
            action=TradeAction.BUY,
            token=sample_token,
            executed_at=datetime.now(),
            price=1.50,
            quantity=100,
            value_usd=150.0,
            success=False,
            error_message="Insufficient funds",
            portfolio_value_before=10000.0,
            portfolio_value_after=10000.0  # No change due to failure
        )
        
        reward_failure, _ = calculator.calculate_reward(
            10000.0, 10000.0, failed_result
        )
        
        # Successful trade should have higher reward
        assert reward_success > reward_failure
    
    def test_market_condition_adjustments(self, reward_config, sample_trading_result):
        """Test market condition adjustments"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        # Test high volatility regime
        market_conditions_high_vol = {
            'volatility_regime': 'high',
            'market_trend': 'bear',
            'fear_greed_index': 15  # Extreme fear
        }
        
        reward_high_vol, _ = calculator.calculate_reward(
            10000.0, 10100.0, sample_trading_result, market_conditions_high_vol
        )
        
        # Reset for comparison
        calculator.reset()
        
        # Test low volatility regime
        market_conditions_low_vol = {
            'volatility_regime': 'low',
            'market_trend': 'bull',
            'fear_greed_index': 85  # Extreme greed
        }
        
        reward_low_vol, _ = calculator.calculate_reward(
            10000.0, 10100.0, sample_trading_result, market_conditions_low_vol
        )
        
        # Rewards should be different due to market adjustments
        assert reward_high_vol != reward_low_vol
    
    def test_performance_summary(self, reward_config, sample_token):
        """Test performance summary generation"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        # Insufficient data case
        summary = calculator.get_performance_summary()
        assert summary['status'] == 'insufficient_data'
        assert summary['observations'] < reward_config.min_observations
        
        # Add sufficient data
        base_value = 10000.0
        for i in range(reward_config.min_observations + 2):
            return_pct = np.random.normal(0.002, 0.01)
            value_after = base_value * (1 + return_pct)
            
            trading_result = TradingResult(
                action=TradeAction.HOLD,
                token=sample_token,
                executed_at=datetime.now(),
                price=1.50,
                quantity=0,
                value_usd=0,
                success=True,
                portfolio_value_before=base_value,
                portfolio_value_after=value_after
            )
            
            calculator.calculate_reward(base_value, value_after, trading_result)
            base_value = value_after
        
        # Active data case
        summary = calculator.get_performance_summary()
        assert summary['status'] == 'active'
        assert summary['observations'] >= reward_config.min_observations
        assert 'total_return' in summary
        assert 'annualized_return' in summary
        assert 'risk_metrics' in summary
        assert 'trade_statistics' in summary
    
    def test_calculator_reset(self, reward_config, sample_trading_result):
        """Test calculator reset functionality"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        # Add some data
        calculator.calculate_reward(10000.0, 10100.0, sample_trading_result)
        
        assert len(calculator.returns_history) > 0
        assert len(calculator.portfolio_values) > 0
        
        # Reset
        calculator.reset()
        
        assert len(calculator.returns_history) == 0
        assert len(calculator.portfolio_values) == 0
        assert len(calculator.drawdown_history) == 0
        assert len(calculator.trade_results) == 0
        assert calculator.last_metrics is None
        assert calculator.metrics_timestamp is None
    
    def test_get_current_metrics(self, reward_config, sample_token):
        """Test getting current metrics"""
        calculator = AdvancedRewardCalculator(reward_config)
        
        # No metrics initially
        assert calculator.get_current_metrics() is None
        
        # Add sufficient data to generate metrics
        base_value = 10000.0
        for i in range(reward_config.min_observations + 1):
            return_pct = np.random.normal(0.001, 0.02)
            value_after = base_value * (1 + return_pct)
            
            trading_result = TradingResult(
                action=TradeAction.HOLD,
                token=sample_token,
                executed_at=datetime.now(),
                price=1.50,
                quantity=0,
                value_usd=0,
                success=True,
                portfolio_value_before=base_value,
                portfolio_value_after=value_after
            )
            
            calculator.calculate_reward(base_value, value_after, trading_result)
            base_value = value_after
        
        # Should have metrics now
        metrics = calculator.get_current_metrics()
        assert metrics is not None
        assert isinstance(metrics, RiskMetrics)


class TestRewardEngineeringFactory:
    """Test reward engineering factory function"""
    
    def test_create_reward_calculator_default(self):
        """Test creating reward calculator with default config"""
        calculator = create_reward_calculator()
        
        assert isinstance(calculator, AdvancedRewardCalculator)
        assert isinstance(calculator.config, RewardConfig)
        assert calculator.config.risk_free_rate == 0.02  # Default value
    
    def test_create_reward_calculator_custom_config(self):
        """Test creating reward calculator with custom config"""
        config = RewardConfig(
            risk_free_rate=0.05,
            target_return=0.25,
            lookback_window=100
        )
        
        calculator = create_reward_calculator(config)
        
        assert isinstance(calculator, AdvancedRewardCalculator)
        assert calculator.config == config
        assert calculator.config.risk_free_rate == 0.05
        assert calculator.config.target_return == 0.25
        assert calculator.config.lookback_window == 100


class TestRewardEngineeringIntegration:
    """Integration tests for reward engineering system"""
    
    @pytest.fixture
    def comprehensive_setup(self):
        """Create comprehensive test setup"""
        config = RewardConfig(
            lookback_window=100,
            min_observations=20,
            reward_scale=50.0,
            return_weight=0.4,
            risk_weight=0.3,
            consistency_weight=0.2,
            efficiency_weight=0.1
        )
        
        calculator = AdvancedRewardCalculator(config)
        
        # Create test token
        token = DiscoveredToken(
            address="0xtest...",
            symbol="TEST",
            name="Test Token",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=2.0,
            volume_24h=500000
        )
        
        return calculator, token, config
    
    def test_full_reward_calculation_cycle(self, comprehensive_setup):
        """Test complete reward calculation cycle"""
        calculator, token, config = comprehensive_setup
        
        # Simulate trading episode
        base_value = 50000.0
        total_reward = 0.0
        
        np.random.seed(123)  # For reproducible test
        
        for i in range(50):
            # Simulate different market conditions
            if i < 10:
                return_pct = np.random.normal(0.002, 0.01)  # Good period
                action = TradeAction.BUY
            elif i < 30:
                return_pct = np.random.normal(-0.001, 0.015)  # Volatile period
                action = TradeAction.HOLD
            else:
                return_pct = np.random.normal(0.001, 0.008)  # Recovery period
                action = TradeAction.SELL
            
            value_after = base_value * (1 + return_pct)
            
            # Create trading result
            success = np.random.random() > 0.1  # 90% success rate
            realized_pnl = np.random.normal(0, 100) if success else 0
            
            trading_result = TradingResult(
                action=action,
                token=token,
                executed_at=datetime.now(),
                price=2.0 + np.random.normal(0, 0.1),
                quantity=100 if action != TradeAction.HOLD else 0,
                value_usd=200 if action != TradeAction.HOLD else 0,
                success=success,
                portfolio_value_before=base_value,
                portfolio_value_after=value_after,
                realized_pnl=realized_pnl
            )
            
            # Market conditions
            market_conditions = {
                'volatility_regime': 'high' if i < 30 else 'low',
                'market_trend': 'bull' if i > 20 else 'bear',
                'fear_greed_index': 30 + i  # Gradual improvement
            }
            
            reward, metrics = calculator.calculate_reward(
                base_value, value_after, trading_result, market_conditions
            )
            
            total_reward += reward
            base_value = value_after
            
            # Validate reward and metrics
            assert isinstance(reward, float)
            assert isinstance(metrics, RiskMetrics)
            assert not np.isnan(reward)
            assert not np.isinf(reward)
        
        # Check final state
        assert len(calculator.returns_history) == 50
        assert calculator.last_metrics is not None
        
        # Get performance summary
        summary = calculator.get_performance_summary()
        assert summary['status'] == 'active'
        assert summary['observations'] == 50
        assert isinstance(summary['total_return'], float)
        assert isinstance(summary['risk_metrics'], dict)
        
        # Validate risk metrics
        metrics = calculator.last_metrics
        assert metrics.volatility >= 0
        assert metrics.max_drawdown >= 0
        assert -3.0 <= metrics.sharpe_ratio <= 3.0  # Reasonable range
    
    def test_extreme_scenarios(self, comprehensive_setup):
        """Test reward calculation in extreme scenarios"""
        calculator, token, config = comprehensive_setup
        
        # Scenario 1: Large loss
        large_loss_result = TradingResult(
            action=TradeAction.SELL,
            token=token,
            executed_at=datetime.now(),
            price=1.5,
            quantity=1000,
            value_usd=1500.0,
            success=True,
            portfolio_value_before=10000.0,
            portfolio_value_after=8000.0,  # 20% loss
            realized_pnl=-2000.0
        )
        
        reward_loss, metrics_loss = calculator.calculate_reward(
            10000.0, 8000.0, large_loss_result
        )
        
        # Should heavily penalize large losses
        assert reward_loss < 0
        
        # Scenario 2: Large gain
        calculator.reset()
        
        large_gain_result = TradingResult(
            action=TradeAction.SELL,
            token=token,
            executed_at=datetime.now(),
            price=2.5,
            quantity=1000,
            value_usd=2500.0,
            success=True,
            portfolio_value_before=10000.0,
            portfolio_value_after=12000.0,  # 20% gain
            realized_pnl=2000.0
        )
        
        reward_gain, metrics_gain = calculator.calculate_reward(
            10000.0, 12000.0, large_gain_result
        )
        
        # Should reward large gains
        assert reward_gain > 0
        assert reward_gain > reward_loss  # Gain should be much better than loss
        
        # Scenario 3: No change (HOLD)
        calculator.reset()
        
        hold_result = TradingResult(
            action=TradeAction.HOLD,
            token=token,
            executed_at=datetime.now(),
            price=2.0,
            quantity=0,
            value_usd=0,
            success=True,
            portfolio_value_before=10000.0,
            portfolio_value_after=10000.0  # No change
        )
        
        reward_hold, metrics_hold = calculator.calculate_reward(
            10000.0, 10000.0, hold_result
        )
        
        # Hold should have minimal reward/penalty
        assert abs(reward_hold) < abs(reward_loss)
        assert abs(reward_hold) < abs(reward_gain)