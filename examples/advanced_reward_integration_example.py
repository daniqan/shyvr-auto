#!/usr/bin/env python3
"""
Example: TradingEnvironment with AdvancedRewardCalculator Integration

This example demonstrates how to use the TradingEnvironment with the sophisticated
AdvancedRewardCalculator for better RL agent training with risk-adjusted rewards.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.rl_agent.trading_environment import TradingEnvironment, EnvironmentConfig
from src.rl_agent.reward_engineering import AdvancedRewardCalculator, RewardConfig
from src.rl_agent.base import TradeAction
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


def create_sample_tokens() -> List[DiscoveredToken]:
    """Create sample tokens for demonstration"""
    tokens = []
    
    # Create different types of tokens
    token_configs = [
        ("ETH", "Ethereum", 2000.0, 1000000),
        ("BTC", "Bitcoin", 45000.0, 800000),
        ("MATIC", "Polygon", 0.85, 500000),
    ]
    
    for i, (symbol, name, price, volume) in enumerate(token_configs):
        token = DiscoveredToken(
            address=f"0x{i:04d}{'a' * 36}",
            symbol=symbol,
            name=name,
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now() - timedelta(hours=i),
            discovery_source="example",
            price_usd=price,
            volume_24h=volume,
            market_cap=price * 21000000  # Simplified market cap
        )
        tokens.append(token)
    
    return tokens


def create_sample_price_data(tokens: List[DiscoveredToken], num_periods: int = 200) -> Dict[str, Dict[str, Any]]:
    """Create realistic price data with trends and volatility"""
    price_data = {}
    
    np.random.seed(42)  # For reproducible example
    
    for token in tokens:
        prices = []
        timestamps = []
        base_price = token.price_usd
        
        # Different market regimes
        for i in range(num_periods):
            timestamp = datetime.now() - timedelta(hours=num_periods - i)
            timestamps.append(timestamp)
            
            if i == 0:
                price = base_price
            else:
                # Create different market phases
                if i < 50:
                    # Bull market phase
                    trend = 0.002  # 0.2% positive trend
                    volatility = 0.015  # 1.5% volatility
                elif i < 100:
                    # High volatility phase
                    trend = -0.001  # Slight negative trend
                    volatility = 0.035  # 3.5% volatility
                elif i < 150:
                    # Bear market phase
                    trend = -0.003  # -0.3% negative trend
                    volatility = 0.025  # 2.5% volatility
                else:
                    # Recovery phase
                    trend = 0.0015  # 0.15% positive trend
                    volatility = 0.02  # 2% volatility
                
                # Random walk with trend
                change = np.random.normal(trend, volatility)
                price = prices[-1] * (1 + change)
                price = max(price, base_price * 0.1)  # Minimum 10% of original price
            
            prices.append(price)
        
        price_data[token.address] = {
            'prices': prices,
            'timestamps': timestamps
        }
    
    return price_data


def run_trading_simulation():
    """Run a complete trading simulation with advanced rewards"""
    
    print("🚀 Starting TradingEnvironment with AdvancedRewardCalculator Integration Example")
    print("=" * 80)
    
    # Create sample data
    tokens = create_sample_tokens()
    price_data = create_sample_price_data(tokens)
    
    print(f"📊 Created {len(tokens)} tokens with {len(price_data[tokens[0].address]['prices'])} price periods each")
    
    # Configure environment
    env_config = EnvironmentConfig(
        initial_cash=50000.0,
        max_position_size=0.15,  # 15% max position size
        transaction_fee=0.0005,  # 0.05% transaction fee
        slippage_factor=0.001,   # 0.1% slippage
        max_episode_steps=150,
        lookback_window=20,
        price_volatility=0.02,
        max_drawdown_limit=0.25  # 25% max drawdown
    )
    
    # Configure advanced reward calculator
    reward_config = RewardConfig(
        risk_free_rate=0.03,      # 3% risk-free rate
        target_return=0.20,       # 20% target annual return
        return_weight=0.4,        # 40% weight on returns
        risk_weight=0.3,          # 30% weight on risk metrics
        consistency_weight=0.2,   # 20% weight on consistency
        efficiency_weight=0.1,    # 10% weight on efficiency
        max_drawdown_threshold=0.15,  # 15% drawdown threshold
        volatility_threshold=0.20,    # 20% volatility threshold
        lookback_window=100,      # 100 periods lookback
        min_observations=20,      # 20 minimum observations
        reward_scale=50.0         # Scale rewards to reasonable range
    )
    
    # Create advanced reward calculator
    reward_calculator = AdvancedRewardCalculator(reward_config)
    
    # Create environments for comparison
    print("\n🔧 Setting up environments...")
    
    # Environment with advanced rewards
    advanced_env = TradingEnvironment(
        config=env_config,
        tokens=tokens,
        historical_data=price_data,
        reward_calculator=reward_calculator
    )
    
    # Environment with simple rewards (for comparison)
    simple_env = TradingEnvironment(
        config=env_config,
        tokens=tokens,
        historical_data=price_data
        # No reward_calculator = simple rewards
    )
    
    print("✅ Environments created successfully")
    
    # Run simulation episodes
    print("\n🎯 Running trading simulations...")
    
    results = {
        'advanced': run_episode(advanced_env, "Advanced Rewards"),
        'simple': run_episode(simple_env, "Simple Rewards")
    }
    
    # Compare results
    print("\n📈 Simulation Results Comparison")
    print("=" * 80)
    
    for env_type, result in results.items():
        print(f"\n{result['name']}:")
        print(f"  Final Portfolio Value: ${result['final_value']:,.2f}")
        print(f"  Total Return: {result['total_return']:+.2%}")
        print(f"  Total Steps: {result['steps']}")
        print(f"  Average Reward: {result['avg_reward']:+.4f}")
        print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
        
        if 'risk_metrics' in result:
            metrics = result['risk_metrics']
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
            print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    
    print("\n🎯 Key Observations:")
    print("- Advanced rewards provide risk-adjusted feedback")
    print("- Risk metrics help evaluate trading quality beyond simple returns")
    print("- Market conditions influence reward calculations")
    print("- Both environments maintain the same trading mechanics")


def run_episode(env: TradingEnvironment, name: str) -> Dict[str, Any]:
    """Run a single trading episode"""
    print(f"\n  Running {name} episode...")
    
    state = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    rewards = []
    initial_cash = env.config.initial_cash
    
    # Simple trading strategy for demonstration
    while not done and step < 100:
        # Select token (rotate through available tokens)
        token = env.tokens[step % len(env.tokens)]
        
        # Simple momentum strategy
        if step > 10:
            # Look at recent rewards to determine action
            recent_rewards = rewards[-5:]
            avg_recent_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            
            if avg_recent_reward > 0.5:
                # Recent positive performance - buy
                action = TradeAction.BUY
                position_size = 0.08  # 8% position
            elif avg_recent_reward < -0.5:
                # Recent negative performance - sell
                action = TradeAction.SELL
                position_size = 0.5   # Sell 50% of position
            else:
                # Neutral - hold
                action = TradeAction.HOLD
                position_size = 0.0
        else:
            # Initial phase - small buys
            action = TradeAction.BUY
            position_size = 0.05  # 5% position
        
        # Execute step
        next_state, reward, done, info = env.step(action, token, position_size)
        
        total_reward += reward
        rewards.append(reward)
        step += 1
        
        # Print progress occasionally
        if step % 20 == 0:
            portfolio_value = info.get('portfolio_value', 0)
            print(f"    Step {step}: Portfolio=${portfolio_value:,.0f}, Reward={reward:+.3f}")
    
    # Collect final metrics
    final_value = env.portfolio.calculate_total_value()
    total_return = (final_value - initial_cash) / initial_cash
    avg_reward = total_reward / step if step > 0 else 0
    max_drawdown = env.portfolio.max_drawdown
    
    result = {
        'name': name,
        'final_value': final_value,
        'total_return': total_return,
        'steps': step,
        'avg_reward': avg_reward,
        'max_drawdown': max_drawdown,
        'total_reward': total_reward
    }
    
    # Add risk metrics if available
    if hasattr(env, 'reward_calculator') and env.reward_calculator:
        metrics = env.reward_calculator.get_current_metrics()
        if metrics:
            result['risk_metrics'] = metrics.to_dict()
    
    return result


def demonstrate_reward_components():
    """Demonstrate different reward components"""
    print("\n🔍 Reward Components Analysis")
    print("=" * 80)
    
    # Create simple setup for analysis
    tokens = create_sample_tokens()[:1]  # Just one token
    price_data = create_sample_price_data(tokens, 50)
    
    reward_config = RewardConfig(
        lookback_window=30,
        min_observations=5,
        reward_scale=10.0  # Smaller scale for clearer analysis
    )
    
    reward_calculator = AdvancedRewardCalculator(reward_config)
    
    print("\nReward Component Weights:")
    print(f"  Return Weight: {reward_config.return_weight:.1%}")
    print(f"  Risk Weight: {reward_config.risk_weight:.1%}")
    print(f"  Consistency Weight: {reward_config.consistency_weight:.1%}")
    print(f"  Efficiency Weight: {reward_config.efficiency_weight:.1%}")
    
    # Simulate different scenarios
    scenarios = [
        ("High Return, High Risk", 0.15, 0.20),
        ("Moderate Return, Low Risk", 0.08, 0.10),
        ("Low Return, Very Low Risk", 0.03, 0.05),
        ("Negative Return, High Risk", -0.10, 0.25),
    ]
    
    print("\nScenario Analysis:")
    for name, return_pct, volatility in scenarios:
        print(f"\n  {name}:")
        print(f"    Simulated Return: {return_pct:+.1%}")
        print(f"    Simulated Volatility: {volatility:.1%}")
        
        # Note: This is a simplified demonstration
        # In practice, you'd run actual trading simulations
        estimated_sharpe = return_pct / volatility if volatility > 0 else 0
        print(f"    Estimated Sharpe Ratio: {estimated_sharpe:.2f}")


if __name__ == "__main__":
    # Run the complete example
    run_trading_simulation()
    
    # Demonstrate reward components
    demonstrate_reward_components()
    
    print("\n🎉 Example completed successfully!")
    print("\nTo use this integration in your own code:")
    print("1. Create an AdvancedRewardCalculator with your desired configuration")
    print("2. Pass it to TradingEnvironment during initialization")
    print("3. The environment will automatically use sophisticated reward calculation")
    print("4. Risk metrics will be included in the step info dictionary")