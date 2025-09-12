# ML Analysis Module

**Status**: Phases 1-3 Complete | Enhanced for 90%+ Accuracy Target

## Module Overview

Core ML analysis module with enhanced feature engineering, hyperparameter optimization, and multiple model architectures for cryptocurrency price prediction.

## Key Components

### FeatureEngineer (Enhanced)
- **Token Normalization**: `normalize_features_by_token()` - Handles WBTC vs PEPE scales
- **Advanced Features**: `calculate_advanced_features()` - Microstructure, volatility regime
- **Data Augmentation**: `augment_training_data()` - 20% augmentation with constraints
- **Missing Values**: `handle_missing_values()` - Multiple strategies, time-aware

### HyperparameterOptimizer (New)
- **Bayesian Optimization**: Gaussian Process with acquisition functions
- **Parameter Spaces**: Defined for all models (LSTM, Transformers)
- **Auto Search**: Finds optimal parameters in 20 trials
- **Persistent Storage**: Saves best params to JSON

### Model Architectures

#### LSTM
- Hidden size: 256 units
- Layers: 3 LSTM layers
- Advanced scheduling: CosineAnnealingWarmRestarts

#### Transformer
- D_model: 128 (configurable)
- Heads: 4-8
- OneCycleLR scheduling

#### iTransformer (Fixed)
- **40 Features**: Expanded from 5
- Cross-variate attention
- Inverted architecture for multivariate

#### PatchTST (Fixed)
- **Channel Aggregation**: Average instead of concatenate
- Patch-based tokenization
- Multi-channel support

#### TimesMixer (Optimized)
- **55x Speedup**: Vectorized operations
- Decomposable mixing
- Reduced to 2 decomposition layers

## Usage Examples

```python
# Enhanced feature engineering
from src.ml_analysis.feature_engineer import FeatureEngineer

fe = FeatureEngineer(
    enable_token_normalization=True,
    enable_advanced_features=True
)

# Process data with all enhancements
data = fe.calculate_advanced_features(data)
data = fe.handle_missing_values(data)
data = fe.normalize_features_by_token(data)
data = fe.augment_training_data(data, augmentation_factor=0.2)

# Hyperparameter optimization
from src.ml_analysis.hyperparameter_optimizer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(n_trials=20)
best_params = await optimizer.optimize_lstm(train_func)
```

## Configuration

Hyperparameters loaded from `config/hyperparameters.yaml`:
- Priority: Optimized → Default → Environment
- Auto-loaded by training pipeline
- Environment-specific overrides supported

## Performance Improvements

| Model | Before | After | Target |
|-------|--------|-------|--------|
| LSTM | 67.83% | TBD | 90% |
| Transformer | 38.44% | TBD | 90% |
| iTransformer | 2.89% | TBD | 90% |
| PatchTST | 4.61% | Fixed | 90% |
| TimesMixer | 59.17% | Optimized | 90% |

## Files in Module

```
src/ml_analysis/
├── feature_engineer.py         # Core feature engineering (enhanced)
├── hyperparameter_optimizer.py # Bayesian optimization (new)
├── lstm_model.py               # LSTM implementation
├── model_manager.py            # Model ensemble management
├── training_report_generator.py # Report generation
└── transformers/
    ├── transformer_predictor.py
    ├── itransformer.py         # 40 features
    ├── patchtst.py             # Fixed aggregation
    └── timesmixer.py           # Vectorized operations
```

## Recent Changes

### Phase 1 (Complete)
- Fixed architecture bugs in PatchTST and TimesMixer
- Expanded iTransformer to 40 features
- Implemented advanced learning rate scheduling

### Phase 2 (Complete)
- Added Bayesian hyperparameter optimization
- Created automated search functionality
- Integrated with training pipeline

### Phase 3 (Complete)
- Enhanced FeatureEngineer with 4 new methods
- Added token-specific normalization
- Implemented data augmentation
- Advanced feature engineering (8+ new features)

## Next Steps

- Implement relative positional encoding
- Add cross-validation for time series
- Integrate Sharpe ratio metrics
- JIT compilation for performance