#!/usr/bin/env python3
"""
Model Ensemble Usage Example
Demonstrates how to use the EnsemblePredictor for LSTM + Transformer predictions
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.ml_analysis.model_ensemble import (
    EnsemblePredictor, EnsembleConfig, EnsembleStrategy
)
from src.discovery.base import DiscoveredToken
from src.utils.base import Chain


async def main():
    """Main example function"""
    print("🤖 Model Ensemble Usage Example")
    print("=" * 50)

    # 1. Create a sample token for analysis
    sample_token = DiscoveredToken(
        chain=Chain.ETHEREUM,
        address="0xA0b86a33E6441efa50Ef0bb09e99b2dC37E2d0b6",  # WBTC
        symbol="WBTC",
        name="Wrapped Bitcoin",
        price_usd=45000.0,
        market_cap=7500000000.0,
        volume_24h=125000000.0
    )

    print(f"📊 Analyzing token: {sample_token.symbol} ({sample_token.name})")
    print(f"💰 Current price: ${sample_token.price_usd:,.2f}")

    # 2. Create ensemble configurations for different strategies
    configs = [
        EnsembleConfig(
            lstm_weight=0.6,
            transformer_weight=0.4,
            strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
            checkpoint_dir="tmp/checkpoints",
            fallback_model_dir="models"
        ),
        EnsembleConfig(
            strategy=EnsembleStrategy.SIMPLE_AVERAGE
        ),
        EnsembleConfig(
            strategy=EnsembleStrategy.CONFIDENCE_WEIGHTED
        ),
        EnsembleConfig(
            strategy=EnsembleStrategy.STACKING,
            meta_learner_type="random_forest"
        )
    ]

    # 3. Demonstrate each ensemble strategy
    for i, config in enumerate(configs, 1):
        print(f"\n🔧 Strategy {i}: {config.strategy.value.upper()}")
        print("-" * 30)

        try:
            # Create ensemble predictor
            ensemble = EnsemblePredictor(config)

            # Check if models can be loaded
            print("📥 Loading models...")
            load_results = await ensemble.load_models()
            print(f"   LSTM loaded: {load_results.get('lstm', False)}")
            print(f"   Transformer loaded: {load_results.get('transformer', False)}")

            if not any(load_results.values()):
                print("⚠️  No models loaded - this is expected in demo environment")
                print("   In production, models would be loaded from checkpoints")
                continue

            # Get ensemble prediction
            print("🔮 Generating ensemble prediction...")
            prediction = await ensemble.analyze_token(sample_token)

            # Display results
            print(f"📈 Prediction Results:")
            print(f"   1h prediction: ${prediction.price_prediction_1h:,.2f}" if prediction.price_prediction_1h else "   1h prediction: N/A")
            print(f"   4h prediction: ${prediction.price_prediction_4h:,.2f}" if prediction.price_prediction_4h else "   4h prediction: N/A")
            print(f"   24h prediction: ${prediction.price_prediction_24h:,.2f}" if prediction.price_prediction_24h else "   24h prediction: N/A")
            print(f"   Direction: {prediction.direction.value}")
            print(f"   Confidence: {prediction.confidence:.2%}")
            print(f"   Probability Up: {prediction.probability_up:.2%}")

            # Show performance metrics
            metrics = ensemble.get_performance_metrics()
            print(f"📊 Performance Metrics:")
            print(f"   Strategy: {metrics['strategy']}")
            print(f"   LSTM Weight: {metrics['lstm_weight']:.1%}")
            print(f"   Transformer Weight: {metrics['transformer_weight']:.1%}")

        except Exception as e:
            print(f"❌ Error with {config.strategy.value}: {str(e)}")

    # 4. Demonstrate weight adjustment
    print(f"\n⚖️  Weight Adjustment Example")
    print("-" * 30)

    config = EnsembleConfig(strategy=EnsembleStrategy.WEIGHTED_AVERAGE)
    ensemble = EnsemblePredictor(config)

    print("Original weights:")
    print(f"   LSTM: {ensemble.ensemble_config.lstm_weight:.1%}")
    print(f"   Transformer: {ensemble.ensemble_config.transformer_weight:.1%}")

    # Adjust weights
    ensemble.set_weights(lstm_weight=0.8, transformer_weight=0.2)

    print("Adjusted weights:")
    print(f"   LSTM: {ensemble.ensemble_config.lstm_weight:.1%}")
    print(f"   Transformer: {ensemble.ensemble_config.transformer_weight:.1%}")

    # 5. Demonstrate strategy switching
    print(f"\n🔄 Strategy Switching Example")
    print("-" * 30)

    original_strategy = ensemble.ensemble_config.strategy
    print(f"Original strategy: {original_strategy.value}")

    ensemble.set_strategy(EnsembleStrategy.CONFIDENCE_WEIGHTED)
    print(f"New strategy: {ensemble.ensemble_config.strategy.value}")

    # 6. Show required features
    print(f"\n📋 Required Features")
    print("-" * 30)

    try:
        features = ensemble.get_required_features()
        print(f"Required features: {', '.join(features)}")
    except Exception as e:
        print(f"Features would be aggregated from loaded models: {str(e)}")

    # 7. Health check example
    print(f"\n🏥 Health Check Example")
    print("-" * 30)

    try:
        is_healthy = await ensemble.health_check()
        print(f"Ensemble healthy: {is_healthy}")
    except Exception as e:
        print(f"Health check: {str(e)}")

    print(f"\n✅ Example completed!")
    print("💡 In production, ensure models are trained and checkpoints exist.")


def create_sample_training_data():
    """Create sample training data for demonstration"""
    print(f"\n📚 Creating Sample Training Data")
    print("-" * 30)

    # Generate synthetic price data
    np.random.seed(42)
    n_samples = 1000

    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    base_price = 45000

    # Generate realistic price movements
    returns = np.random.normal(0, 0.02, n_samples)  # 2% volatility
    prices = [base_price]

    for i in range(1, n_samples):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.lognormal(10, 1, n_samples),  # Log-normal volume
        'market_cap': np.array(prices) * 19000000,  # Approximate WBTC supply
        'returns': [0] + [np.log(prices[i] / prices[i-1]) for i in range(1, n_samples)]
    })

    # Add technical indicators
    data['sma_20'] = data['price'].rolling(20).mean()
    data['ema_12'] = data['price'].ewm(span=12).mean()
    data['volatility'] = data['returns'].rolling(24).std()

    print(f"📊 Generated {len(data)} samples")
    print(f"💰 Price range: ${data['price'].min():,.0f} - ${data['price'].max():,.0f}")
    print(f"📈 Mean return: {data['returns'].mean():.4f}")
    print(f"📊 Volatility: {data['returns'].std():.4f}")

    return data


async def training_example():
    """Demonstrate ensemble training"""
    print(f"\n🎓 Training Example")
    print("=" * 30)

    # Create sample data
    training_data = create_sample_training_data()

    # Create ensemble for training
    config = EnsembleConfig(
        strategy=EnsembleStrategy.STACKING,
        retrain_meta_learner=True
    )

    ensemble = EnsemblePredictor(config)

    print("🚀 Starting ensemble training...")

    try:
        success = await ensemble.train_model(training_data)
        print(f"✅ Training completed: {success}")

        if success:
            print("📊 Training results:")
            metrics = ensemble.get_performance_metrics()
            print(f"   Models loaded: {metrics['models_loaded']}")
            print(f"   LSTM trained: {metrics['lstm_trained']}")
            print(f"   Transformer trained: {metrics['transformer_trained']}")
            print(f"   Meta-learner available: {metrics['meta_learner_available']}")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("💡 This is expected in demo - requires actual model implementations")


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Run training example
    asyncio.run(training_example())