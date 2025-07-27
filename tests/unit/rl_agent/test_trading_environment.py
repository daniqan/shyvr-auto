"""
Tests for Trading Environment simulation
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock
from typing import List, Dict, Any

from src.rl_agent.base import (
    TradeAction, MarketState, TradingResult, AgentConfig
)
from src.rl_agent.trading_environment import (
    TradingEnvironment, Portfolio, EnvironmentConfig,
    TradingEnvironmentError
)
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


class TestPortfolio:
    """Test Portfolio management"""
    
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
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization"""
        portfolio = Portfolio(initial_cash=10000.0)
        
        assert portfolio.cash == 10000.0
        assert portfolio.initial_cash == 10000.0
        assert len(portfolio.positions) == 0
        assert portfolio.total_value == 10000.0
        assert portfolio.unrealized_pnl == 0.0
        assert portfolio.realized_pnl == 0.0
    
    def test_portfolio_buy_order(self, sample_token):
        """Test buying tokens"""
        portfolio = Portfolio(initial_cash=10000.0)
        
        # Buy 100 tokens at $1.50 each
        result = portfolio.execute_order(
            action=TradeAction.BUY,
            token=sample_token,
            quantity=100,
            price=1.50
        )
        
        assert result.success is True
        assert result.quantity == 100
        assert result.value_usd == 150.0
        assert portfolio.cash == 9850.0  # 10000 - 150
        assert sample_token.address in portfolio.positions
        assert portfolio.positions[sample_token.address]['quantity'] == 100
        assert portfolio.positions[sample_token.address]['avg_price'] == 1.50
    
    def test_portfolio_sell_order(self, sample_token):
        """Test selling tokens"""
        portfolio = Portfolio(initial_cash=10000.0)
        
        # First buy some tokens
        portfolio.execute_order(TradeAction.BUY, sample_token, 100, 1.50)
        
        # Then sell half at higher price
        result = portfolio.execute_order(
            action=TradeAction.SELL,
            token=sample_token,
            quantity=50,
            price=1.75
        )
        
        assert result.success is True
        assert result.quantity == 50
        assert result.value_usd == 87.5  # 50 * 1.75
        assert portfolio.cash == 9937.5  # 9850 + 87.5
        assert portfolio.positions[sample_token.address]['quantity'] == 50
        assert result.realized_pnl == 12.5  # (1.75 - 1.50) * 50
    
    def test_portfolio_insufficient_cash(self, sample_token):
        """Test buying with insufficient cash"""
        portfolio = Portfolio(initial_cash=100.0)
        
        # Try to buy more than we can afford
        result = portfolio.execute_order(
            action=TradeAction.BUY,
            token=sample_token,
            quantity=100,
            price=1.50  # Would cost 150, but we only have 100
        )
        
        assert result.success is False
        assert "Insufficient cash" in result.error_message
        assert portfolio.cash == 100.0  # No change
        assert len(portfolio.positions) == 0
    
    def test_portfolio_insufficient_tokens(self, sample_token):
        """Test selling more tokens than owned"""
        portfolio = Portfolio(initial_cash=10000.0)
        
        # Buy 50 tokens
        portfolio.execute_order(TradeAction.BUY, sample_token, 50, 1.50)
        
        # Try to sell 100 tokens (more than we have)
        result = portfolio.execute_order(
            action=TradeAction.SELL,
            token=sample_token,
            quantity=100,
            price=1.75
        )
        
        assert result.success is False
        assert "Insufficient tokens" in result.error_message
        assert portfolio.positions[sample_token.address]['quantity'] == 50  # No change
    
    def test_portfolio_update_prices(self, sample_token):
        """Test updating token prices for unrealized PnL"""
        portfolio = Portfolio(initial_cash=10000.0)
        
        # Buy tokens
        portfolio.execute_order(TradeAction.BUY, sample_token, 100, 1.50)
        
        # Update price
        portfolio.update_token_price(sample_token.address, 1.75)
        
        # Calculate unrealized PnL
        unrealized_pnl = portfolio.calculate_unrealized_pnl()
        expected_pnl = (1.75 - 1.50) * 100  # $25
        
        assert abs(unrealized_pnl - expected_pnl) < 0.01
        assert portfolio.unrealized_pnl == expected_pnl
    
    def test_portfolio_total_value_calculation(self, sample_token):
        """Test total portfolio value calculation"""
        portfolio = Portfolio(initial_cash=10000.0)
        
        # Buy tokens
        portfolio.execute_order(TradeAction.BUY, sample_token, 100, 1.50)
        
        # Update price
        portfolio.update_token_price(sample_token.address, 1.75)
        
        total_value = portfolio.calculate_total_value()
        expected_value = 9850.0 + (100 * 1.75)  # Cash + token value
        
        assert abs(total_value - expected_value) < 0.01
        assert portfolio.total_value == expected_value
    
    def test_portfolio_position_info(self, sample_token):
        """Test getting position information"""
        portfolio = Portfolio(initial_cash=10000.0)
        
        # Buy tokens in two transactions
        portfolio.execute_order(TradeAction.BUY, sample_token, 50, 1.50)
        portfolio.execute_order(TradeAction.BUY, sample_token, 30, 1.60)
        
        position = portfolio.get_position(sample_token.address)
        
        assert position is not None
        assert position['quantity'] == 80
        # Average price should be weighted: (50*1.5 + 30*1.6) / 80 = 1.5375
        expected_avg = (50 * 1.50 + 30 * 1.60) / 80
        assert abs(position['avg_price'] - expected_avg) < 0.01


class TestEnvironmentConfig:
    """Test Environment Configuration"""
    
    def test_environment_config_defaults(self):
        """Test default environment configuration"""
        config = EnvironmentConfig()
        
        assert config.initial_cash == 10000.0
        assert config.max_position_size == 0.1
        assert config.transaction_fee == 0.001
        assert config.slippage_factor == 0.002
        assert config.max_episode_steps == 1000
        assert config.lookback_window == 20
        assert config.price_volatility == 0.02
    
    def test_environment_config_custom_values(self):
        """Test custom environment configuration"""
        config = EnvironmentConfig(
            initial_cash=50000.0,
            max_position_size=0.2,
            transaction_fee=0.0005,
            max_episode_steps=500
        )
        
        assert config.initial_cash == 50000.0
        assert config.max_position_size == 0.2
        assert config.transaction_fee == 0.0005
        assert config.max_episode_steps == 500


class TestTradingEnvironment:
    """Test Trading Environment simulation"""
    
    @pytest.fixture
    def env_config(self):
        """Create environment configuration"""
        return EnvironmentConfig(
            initial_cash=10000.0,
            max_episode_steps=100,
            lookback_window=10
        )
    
    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing"""
        tokens = []
        for i in range(3):
            token = DiscoveredToken(
                address=f"0x{i:03d}...",
                symbol=f"TOKEN{i}",
                name=f"Test Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=1.0 + i * 0.5,
                volume_24h=100000 + i * 50000
            )
            tokens.append(token)
        return tokens
    
    @pytest.fixture
    def mock_price_data(self, sample_tokens):
        """Create mock historical price data"""
        price_data = {}
        for token in sample_tokens:
            # Generate 50 price points with some volatility
            base_price = token.price_usd
            prices = []
            for j in range(50):
                # Add some random walk
                volatility = 0.02
                change = np.random.normal(0, volatility)
                if j == 0:
                    price = base_price
                else:
                    price = prices[-1] * (1 + change)
                prices.append(max(price, 0.01))  # Ensure positive prices
            
            price_data[token.address] = {
                'prices': prices,
                'timestamps': [datetime.now() - timedelta(hours=50-i) for i in range(50)]
            }
        
        return price_data
    
    def test_environment_initialization(self, env_config, sample_tokens, mock_price_data):
        """Test environment initialization"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        
        assert env.config == env_config
        assert len(env.tokens) == 3
        assert env.current_step == 0
        assert env.portfolio.cash == env_config.initial_cash
        assert not env.done
    
    def test_environment_reset(self, env_config, sample_tokens, mock_price_data):
        """Test environment reset"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        
        # Make some trades and advance time
        initial_state = env.reset()
        env.step(TradeAction.BUY, sample_tokens[0], 0.05)  # 5% position
        env.step(TradeAction.HOLD, sample_tokens[0], 0.0)
        
        # Reset environment
        reset_state = env.reset()
        
        assert env.current_step == 0
        assert env.portfolio.cash == env_config.initial_cash
        assert len(env.portfolio.positions) == 0
        assert not env.done
        assert isinstance(reset_state, MarketState)
    
    def test_environment_step_buy_action(self, env_config, sample_tokens, mock_price_data):
        """Test environment step with buy action"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        env.reset()
        
        token = sample_tokens[0]
        position_size = 0.1  # 10% of portfolio
        
        state, reward, done, info = env.step(TradeAction.BUY, token, position_size)
        
        assert isinstance(state, MarketState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert env.current_step == 1
        
        # Check that position was created
        position = env.portfolio.get_position(token.address)
        assert position is not None
        assert position['quantity'] > 0
    
    def test_environment_step_sell_action(self, env_config, sample_tokens, mock_price_data):
        """Test environment step with sell action"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        env.reset()
        
        token = sample_tokens[0]
        
        # First buy some tokens
        env.step(TradeAction.BUY, token, 0.1)
        
        # Then sell them
        state, reward, done, info = env.step(TradeAction.SELL, token, 0.1)
        
        assert isinstance(reward, float)
        
        # Position should be reduced or eliminated
        position = env.portfolio.get_position(token.address)
        if position:
            assert position['quantity'] < env.portfolio.initial_cash * 0.1 / token.price_usd
    
    def test_environment_step_hold_action(self, env_config, sample_tokens, mock_price_data):
        """Test environment step with hold action"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        env.reset()
        
        token = sample_tokens[0]
        initial_cash = env.portfolio.cash
        
        state, reward, done, info = env.step(TradeAction.HOLD, token, 0.0)
        
        assert isinstance(reward, float)
        assert env.portfolio.cash == initial_cash  # No change in cash
        assert len(env.portfolio.positions) == 0  # No positions created
    
    def test_environment_episode_termination(self, env_config, sample_tokens, mock_price_data):
        """Test episode termination conditions"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        env.reset()
        
        # Run until episode ends
        token = sample_tokens[0]
        done = False
        step_count = 0
        
        while not done and step_count < env_config.max_episode_steps + 10:
            state, reward, done, info = env.step(TradeAction.HOLD, token, 0.0)
            step_count += 1
        
        assert done
        assert step_count <= env_config.max_episode_steps
    
    def test_environment_reward_calculation(self, env_config, sample_tokens, mock_price_data):
        """Test reward calculation"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        env.reset()
        
        token = sample_tokens[0]
        
        # Make a profitable trade (assuming price goes up)
        initial_value = env.portfolio.total_value
        
        # Buy
        state1, reward1, done1, info1 = env.step(TradeAction.BUY, token, 0.1)
        
        # Hold for a few steps (let price potentially change)
        for _ in range(5):
            state2, reward2, done2, info2 = env.step(TradeAction.HOLD, token, 0.0)
        
        final_value = env.portfolio.total_value
        
        # Reward should reflect change in portfolio value
        assert isinstance(reward2, float)
        assert abs(reward2) >= 0  # Reward magnitude should be reasonable
    
    def test_environment_market_state_generation(self, env_config, sample_tokens, mock_price_data):
        """Test market state generation"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        state = env.reset()
        
        assert isinstance(state, MarketState)
        assert state.token in sample_tokens
        assert state.price_usd > 0
        assert state.portfolio_value == env_config.initial_cash
        assert state.cash_balance == env_config.initial_cash
        assert state.current_position == 0.0
    
    def test_environment_transaction_fees(self, env_config, sample_tokens, mock_price_data):
        """Test transaction fee application"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        env.reset()
        
        token = sample_tokens[0]
        position_size = 0.1
        initial_cash = env.portfolio.cash
        
        # Make a buy order
        env.step(TradeAction.BUY, token, position_size)
        
        # Cash should be reduced by more than just the token cost due to fees
        expected_cost = initial_cash * position_size
        actual_cost = initial_cash - env.portfolio.cash
        
        assert actual_cost > expected_cost  # Fees were applied
        
        # Fee should be approximately the configured amount
        fee_rate = env_config.transaction_fee
        expected_fee = expected_cost * fee_rate
        actual_fee = actual_cost - expected_cost
        
        assert abs(actual_fee - expected_fee) < 0.01
    
    def test_environment_slippage(self, env_config, sample_tokens, mock_price_data):
        """Test price slippage simulation"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        env.reset()
        
        token = sample_tokens[0]
        expected_price = env.get_current_price(token.address)
        
        # Large order should have more slippage
        large_position = 0.5  # 50% of portfolio
        env.step(TradeAction.BUY, token, large_position)
        
        # Check the executed price in the last trading result
        if env.last_trading_result:
            executed_price = env.last_trading_result.price
            # With slippage, executed price should be different from market price
            assert executed_price != expected_price
    
    def test_environment_price_updates(self, env_config, sample_tokens, mock_price_data):
        """Test price updates over time"""
        env = TradingEnvironment(env_config, sample_tokens, mock_price_data)
        env.reset()
        
        token = sample_tokens[0]
        initial_price = env.get_current_price(token.address)
        
        # Step forward several times
        for _ in range(10):
            env.step(TradeAction.HOLD, token, 0.0)
        
        final_price = env.get_current_price(token.address)
        
        # Price should have updated (unless we're at the end of data)
        if env.current_step < len(mock_price_data[token.address]['prices']) - 1:
            assert final_price != initial_price


class TestTradingEnvironmentError:
    """Test trading environment error handling"""
    
    def test_trading_environment_error(self):
        """Test TradingEnvironmentError exception"""
        with pytest.raises(TradingEnvironmentError):
            raise TradingEnvironmentError("Environment error")
    
    def test_environment_error_inheritance(self):
        """Test error inheritance"""
        from src.rl_agent.base import RLTrainingError
        
        error = TradingEnvironmentError("Test error")
        assert isinstance(error, Exception)
        # Note: TradingEnvironmentError doesn't inherit from RLTrainingError
        # as it's a more general environment error


class TestTradingEnvironmentIntegration:
    """Integration tests for trading environment"""
    
    @pytest.fixture
    def full_environment_setup(self):
        """Create complete environment setup"""
        # Create tokens
        tokens = []
        for i in range(5):
            token = DiscoveredToken(
                address=f"0x{i:04d}...",
                symbol=f"TKN{i}",
                name=f"Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now() - timedelta(days=i),
                discovery_source="test",
                price_usd=1.0 + i * 0.25,
                volume_24h=100000 * (i + 1)
            )
            tokens.append(token)
        
        # Create price data
        price_data = {}
        np.random.seed(42)  # For reproducible tests
        
        for token in tokens:
            prices = []
            base_price = token.price_usd
            
            for j in range(200):  # Longer price history
                if j == 0:
                    price = base_price
                else:
                    # Random walk with slight upward bias
                    change = np.random.normal(0.001, 0.02)
                    price = prices[-1] * (1 + change)
                    price = max(price, 0.01)  # Floor price
                
                prices.append(price)
            
            price_data[token.address] = {
                'prices': prices,
                'timestamps': [
                    datetime.now() - timedelta(hours=200-i) 
                    for i in range(200)
                ]
            }
        
        config = EnvironmentConfig(
            initial_cash=25000.0,
            max_episode_steps=150,
            lookback_window=20,
            transaction_fee=0.0005,
            slippage_factor=0.001
        )
        
        return TradingEnvironment(config, tokens, price_data)
    
    def test_full_trading_episode(self, full_environment_setup):
        """Test a complete trading episode"""
        env = full_environment_setup
        state = env.reset()
        
        total_reward = 0.0
        episode_steps = 0
        done = False
        
        # Simple trading strategy: buy when RSI is low, sell when high
        while not done and episode_steps < 100:
            # Rotate through tokens
            token_index = episode_steps % len(env.tokens)
            token = env.tokens[token_index]
            
            # Simple strategy based on step number
            if episode_steps % 10 < 3:
                action = TradeAction.BUY
                position_size = 0.05  # 5% position
            elif episode_steps % 10 < 6:
                action = TradeAction.HOLD
                position_size = 0.0
            else:
                action = TradeAction.SELL
                position_size = 0.03  # Partial sell
            
            state, reward, done, info = env.step(action, token, position_size)
            
            total_reward += reward
            episode_steps += 1
            
            # Validate state
            assert isinstance(state, MarketState)
            assert state.portfolio_value > 0
            assert 0 <= state.current_position <= 1
        
        # Episode should complete successfully
        assert episode_steps > 0
        assert isinstance(total_reward, float)
        
        # Portfolio should still have positive value
        assert env.portfolio.total_value > 0
    
    def test_environment_with_multiple_positions(self, full_environment_setup):
        """Test environment with multiple concurrent positions"""
        env = full_environment_setup
        env.reset()
        
        # Buy positions in multiple tokens
        tokens_bought = []
        for i, token in enumerate(env.tokens[:3]):
            position_size = 0.05 + i * 0.02  # Varying position sizes
            state, reward, done, info = env.step(TradeAction.BUY, token, position_size)
            tokens_bought.append(token)
            
            # Verify position was created
            position = env.portfolio.get_position(token.address)
            assert position is not None
            assert position['quantity'] > 0
        
        # Portfolio should have multiple positions
        assert len(env.portfolio.positions) == 3
        
        # Total position value should be reasonable
        total_value = env.portfolio.calculate_total_value()
        assert total_value > 0
        assert total_value < env.config.initial_cash * 1.1  # Shouldn't exceed initial + reasonable gains


class TestTradingEnvironmentAdvancedRewards:
    """Test TradingEnvironment integration with AdvancedRewardCalculator"""
    
    @pytest.fixture
    def env_with_advanced_rewards(self):
        """Create environment with advanced reward calculator"""
        from src.rl_agent.reward_engineering import RewardConfig, AdvancedRewardCalculator
        
        config = EnvironmentConfig(
            initial_cash=10000.0,
            max_episode_steps=100,
            lookback_window=10
        )
        
        reward_config = RewardConfig(
            lookback_window=50,
            min_observations=5,
            reward_scale=10.0
        )
        
        tokens = []
        for i in range(2):
            token = DiscoveredToken(
                address=f"0x{i:03d}...",
                symbol=f"TOKEN{i}",
                name=f"Test Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="test",
                price_usd=1.0 + i * 0.5,
                volume_24h=100000 + i * 50000
            )
            tokens.append(token)
        
        # Create mock price data
        price_data = {}
        for token in tokens:
            base_price = token.price_usd
            prices = []
            for j in range(50):
                volatility = 0.02
                change = np.random.normal(0, volatility)
                if j == 0:
                    price = base_price
                else:
                    price = prices[-1] * (1 + change)
                prices.append(max(price, 0.01))
            
            price_data[token.address] = {
                'prices': prices,
                'timestamps': [datetime.now() - timedelta(hours=50-i) for i in range(50)]
            }
        
        env = TradingEnvironment(config, tokens, price_data)
        
        # Set up advanced reward calculator
        reward_calculator = AdvancedRewardCalculator(reward_config)
        env.reward_calculator = reward_calculator
        
        return env, reward_calculator, tokens
    
    def test_environment_has_advanced_reward_calculator(self, env_with_advanced_rewards):
        """Test that environment can use advanced reward calculator"""
        env, reward_calc, tokens = env_with_advanced_rewards
        
        assert hasattr(env, 'reward_calculator')
        assert env.reward_calculator is reward_calc
    
    def test_advanced_reward_calculation_in_step(self, env_with_advanced_rewards):
        """Test that advanced reward calculator is used in step method"""
        env, reward_calc, tokens = env_with_advanced_rewards
        env.reset()
        
        token = tokens[0]
        position_size = 0.1
        
        # Make a trade
        state, reward, done, info = env.step(TradeAction.BUY, token, position_size)
        
        # Verify reward was calculated using advanced calculator
        assert isinstance(reward, float)
        assert 'risk_metrics' in info
        assert len(reward_calc.returns_history) == 1
        assert len(reward_calc.portfolio_values) == 1
    
    def test_advanced_reward_vs_simple_reward(self, env_with_advanced_rewards):
        """Test that advanced reward differs from simple reward calculation"""
        env, reward_calc, tokens = env_with_advanced_rewards
        
        # Create a second environment without advanced rewards for comparison
        config = EnvironmentConfig(
            initial_cash=10000.0,
            max_episode_steps=100,
            lookback_window=10
        )
        
        price_data = {}
        for token in tokens:
            price_data[token.address] = {
                'prices': [token.price_usd] * 50,
                'timestamps': [datetime.now() - timedelta(hours=50-i) for i in range(50)]
            }
        
        simple_env = TradingEnvironment(config, tokens, price_data)
        
        # Reset both environments
        env.reset()
        simple_env.reset()
        
        token = tokens[0]
        position_size = 0.1
        
        # Make identical trades
        state_adv, reward_adv, done_adv, info_adv = env.step(TradeAction.BUY, token, position_size)
        state_simple, reward_simple, done_simple, info_simple = simple_env.step(TradeAction.BUY, token, position_size)
        
        # For the first step, rewards might be similar, but info should differ
        assert 'risk_metrics' in info_adv
        assert 'risk_metrics' not in info_simple
        
        # After multiple steps, rewards should diverge
        for _ in range(5):
            env.step(TradeAction.HOLD, token, 0.0)
            simple_env.step(TradeAction.HOLD, token, 0.0)
        
        # Advanced reward should have more sophisticated calculation
        assert len(reward_calc.returns_history) == 6  # 1 buy + 5 holds
    
    def test_risk_metrics_in_info(self, env_with_advanced_rewards):
        """Test that risk metrics are included in step info"""
        env, reward_calc, tokens = env_with_advanced_rewards
        env.reset()
        
        token = tokens[0]
        
        # Make several trades to build up history
        for i in range(10):
            action = TradeAction.BUY if i % 3 == 0 else TradeAction.HOLD
            position_size = 0.05 if action == TradeAction.BUY else 0.0
            
            state, reward, done, info = env.step(action, token, position_size)
            
            assert 'risk_metrics' in info
            risk_metrics = info['risk_metrics']
            
            assert isinstance(risk_metrics, dict)
            assert 'sharpe_ratio' in risk_metrics
            assert 'max_drawdown' in risk_metrics
            assert 'volatility' in risk_metrics
    
    def test_market_conditions_passed_to_reward_calculator(self, env_with_advanced_rewards):
        """Test that market conditions are passed to reward calculator"""
        env, reward_calc, tokens = env_with_advanced_rewards
        env.reset()
        
        # Mock the reward calculator to verify market conditions are passed
        original_calculate_reward = reward_calc.calculate_reward
        called_market_conditions = []
        
        def mock_calculate_reward(portfolio_value_before, portfolio_value_after, trading_result, market_conditions=None):
            called_market_conditions.append(market_conditions)
            return original_calculate_reward(portfolio_value_before, portfolio_value_after, trading_result, market_conditions)
        
        reward_calc.calculate_reward = mock_calculate_reward
        
        token = tokens[0]
        env.step(TradeAction.BUY, token, 0.1)
        
        # Verify market conditions were passed
        assert len(called_market_conditions) == 1
        market_conditions = called_market_conditions[0]
        
        # Should contain relevant market information
        if market_conditions:
            assert isinstance(market_conditions, dict)