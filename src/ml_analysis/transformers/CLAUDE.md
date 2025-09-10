# Transformer Models for Time-Series Prediction

This directory contains advanced transformer architectures optimized for cryptocurrency price prediction.

## Optimization Progress (2025-09-06)

### Performance Improvements Applied
1. **Epochs Increased**: All models now train for 80-150 epochs (was 10-20)
2. **Learning Rate Optimization**: 
   - Transformer: 5e-5 with OneCycleLR
   - iTransformer: 1e-4 with 40 features
   - PatchTST: 5e-4 with 15 channels
   - TimesMixer: 5e-4 with 2 decomposition layers
3. **Architecture Enhancements**:
   - LSTM: 256 hidden units, 3 layers
   - Transformer: Batch size 64, warmup scheduling
   - iTransformer: Expanded to 40 features from 5
4. **Advanced Scheduling**: OneCycleLR for transformers, CosineAnnealingWarmRestarts for LSTM

## Recent Fixes (2025-09-05)

### TimesMixer Output Fix
- **Issue**: Model was returning wrong tensor during training (`predictions` key contained raw future representations)
- **Root Cause**: The forward method was using `outputs['predictions']` for internal future representations
- **Fix**: Renamed `predictions` to `future_representations` to avoid confusion with training loop
- **Training**: Now correctly uses `price_1h`, `price_4h`, `price_24h` keys for loss calculation
- **Impact**: TimesMixer can now train properly without shape mismatch errors
- **Commit**: 870a73a - fix TimesMixer output key to avoid confusion during training

### iTransformer Shape Fix (Previously Applied)
- **Issue**: Prediction heads outputting wrong shape (batch_size, n_variates) instead of (batch_size, 1)
- **Fix**: Changed prediction heads to output single value per horizon
- **Status**: Fix already applied and working

## Model Status (Updated 2025-09-06)

| Model | Previous | Current | Target | Improvements Applied |
|-------|----------|---------|--------|---------------------|
| Transformer | 31.65% | Testing | 90%+ | 150 epochs, OneCycleLR, batch 64 |
| iTransformer | 3.22% | Testing | 85%+ | 40 features, 100 epochs, optimized LR |
| PatchTST | 4.61% | Testing | 85%+ | 15 channels, 100 epochs |
| TimesMixer | Fixed | Testing | 85%+ | 2 layers, 80 epochs, optimized |
| LSTM | 67.83% | Testing | 90%+ | 256 units, 3 layers, 100 epochs |

## Architecture Details

### TimesMixer
- Uses decomposable mixing for temporal patterns
- Separates seasonal and trend components
- Multiple predictor heads for different horizons
- Key output structure:
  - `price_1h`, `price_4h`, `price_24h`: Price predictions for training
  - `future_representations`: Internal representations (not for training)
  - `confidence`, `direction_probs`: Additional outputs for inference

### iTransformer
- Inverted attention mechanism (time as channels, features as tokens)
- Cross-variate attention for multivariate correlation
- Fixed prediction heads now output single values per horizon

### PatchTST
- Patch-based tokenization for efficiency
- Currently configured for single channel (close price only)
- Needs multi-channel configuration for better performance

### Standard Transformer
- Traditional multi-head attention
- Working but needs hyperparameter optimization
- Current accuracy: 31.65% R²

## Training Configuration

All models use the `prepare_training_from_corpus()` method to prepare data:
- Standardized input format from corpus data
- Returns (X, y, feature_names) tuple
- X shape: (n_samples, sequence_length, n_features)
- y shape: (n_samples, 3) for [1h, 4h, 24h] predictions

## Important Notes for Zero-Shot Resume

1. **TimesMixer Training**: The key issue was that `outputs['predictions']` was being used by the training loop but contained wrong tensor shape. Now uses price horizon keys.

2. **Shape Validation**: Training loop includes automatic shape validation and reshaping when possible (lines 700-707 in train_all_models.py)

3. **Output Handling**: Training loop handles multiple output formats:
   - Dict with `price_1h`, `price_4h`, `price_24h` keys (preferred)
   - Dict with `predictions` key (legacy, now avoided)
   - Direct tensor output

4. **Next Priority**: Test TimesMixer training with the fix, then work on improving accuracy for all transformer models through hyperparameter tuning.