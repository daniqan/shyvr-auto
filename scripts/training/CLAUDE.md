# Training Scripts

**Status**: Fully Integrated Pipeline with Phases 1-5 Complete - Dual Optimization & Model Ensemble

## Overview

Comprehensive training pipeline for cryptocurrency price prediction models with automated hyperparameter optimization, advanced feature engineering, multi-model support, and automatic checkpointing for long training sessions.

## Main Scripts

### train_all_models.py
**Purpose**: Unified training pipeline for all models
**Features**:
- Loads GCS corpus data
- Applies advanced feature engineering
- Token-specific normalization
- Data augmentation (20%)
- Hyperparameter auto-loading
- Multi-model training (LSTM + 4 Transformers)
- Automatic checkpointing and resume training
- Best model tracking based on validation loss

**Usage**:
```bash
python scripts/training/train_all_models.py
ENVIRONMENT=production python scripts/training/train_all_models.py
```

### hyperparameter_search.py
**Purpose**: Dual optimization for hyperparameters (Bayesian + Grid Search)
**Features**:
- Bayesian: Gaussian Process optimization (default)
- Grid Search: Exhaustive parameter space exploration (new)
- Method selection via --method parameter
- 20 trials per model (Bayesian) or configurable grid (Grid Search)
- Quick evaluation (5-10 epochs)
- Enhanced metadata saving for both methods

**Usage**:
```bash
# Bayesian optimization (default)
python scripts/training/hyperparameter_search.py \
    --n-trials 20 \
    --timeframe daily \
    --token WBTC

# Grid Search optimization (new)
python scripts/training/hyperparameter_search.py \
    --method grid \
    --model lstm \
    --timeframe daily \
    --token WBTC
```

### demonstrate_grid_search.py
**Purpose**: Demonstrate Grid search capabilities and showcase optimization methods
**Features**:
- Comprehensive grid search demonstration
- Model-specific parameter optimization
- Performance comparison between methods
- Enhanced metadata and logging

**Usage**:
```bash
# Demonstrate Grid Search capabilities across models
python scripts/training/demonstrate_grid_search.py
```

### train_transformers.py
**Purpose**: Dedicated transformer training  
**Features**:
- Environment-based configuration
- LSTM baseline + 4 transformer variants
- Progress tracking
- GCS model storage

**Usage**:
```bash
python scripts/training/train_transformers.py --environment production
```

## Training Pipeline Flow

```
1. Load Corpus (GCS)
2. Split Data (80/10/10)
3. Feature Engineering
   - Advanced features (microstructure, volatility)
   - Missing value handling
4. Token Normalization
   - Auto-detect token from price
   - Scale appropriately
5. Data Augmentation
   - 20% synthetic samples
   - Maintain constraints
6. Load Hyperparameters
   - Check optimized → default → environment
7. Train Models (with Checkpointing)
   - Auto-resume from checkpoints
   - Save best model on validation improvement
   - Save regular checkpoints every 10 epochs
   - LSTM with CosineAnnealingWarmRestarts
   - Transformers with OneCycleLR
8. Generate Report
   - HTML format
   - Performance metrics
   - Training curves
```

## Configuration

### hyperparameters.yaml Structure
```yaml
default:           # Baseline parameters
  lstm: {...}
  transformer: {...}
  
optimized:         # From Bayesian search
  lstm: {...}
  
environments:      # Environment overrides
  production:
    all:
      epochs: 150
```

### Environment Variables
```bash
ENVIRONMENT=development|staging|production
GCS_BUCKET=shyvr-models-prod
CUDA_VISIBLE_DEVICES=0
```

## Model Configurations

| Model | Epochs | Batch Size | Learning Rate | Special Features |
|-------|--------|------------|---------------|------------------|
| LSTM | 100 | 32 | 1e-3 | 3 layers, 256 hidden |
| Transformer | 150 | 64 | 5e-4 | OneCycleLR |
| iTransformer | 100 | 32 | 5e-4 | 40 features |
| PatchTST | 100 | 32 | 5e-4 | Fixed aggregation |
| TimesMixer | 80 | 32 | 5e-4 | Vectorized ops |

## Recent Improvements

### Phase 1: Architecture Fixes
- PatchTST: Fixed channel aggregation bug
- TimesMixer: 55x performance improvement
- iTransformer: Expanded to 40 features

### Phase 2: Hyperparameter System
- Bayesian optimization implementation
- Automated search script
- YAML configuration management
- Pipeline integration

### Phase 3: Training Enhancements
- Token-specific normalization
- Data augmentation (20%)
- Advanced features (+8 microstructure)
- Improved missing value handling

### Phase 4: Model Checkpointing
- Automatic checkpoint saving every 10 epochs
- Best model tracking based on validation loss
- Resume training from latest checkpoint
- Organized checkpoint storage structure

### Phase 5: Dual Hyperparameter Optimization & Model Ensemble
- Grid Search integration alongside Bayesian optimization
- Method selection (--method bayesian|grid)
- Enhanced metadata saving with detailed tracking
- Model ensemble capabilities (LSTM + Transformer combination)
- Critical fixes: DatetimeIndex handling, negative scale prevention
- Early stopping implementation for intelligent training termination

## Performance Tracking

Current vs Target (After Phase 1-5 Optimizations):
- LSTM: 68% → 90% (stable with checkpointing)
- Transformer: 38% → 90% (ensemble integration)
- iTransformer: 40%+ → 90% (expanded to 40 features - significant improvement)
- PatchTST: 25%+ → 90% (shape mismatch fixed - improved performance)
- TimesMixer: 59% → 90% (55x speedup achieved, vectorized operations)

## Files

```
scripts/training/
├── train_all_models.py         # Main pipeline with checkpointing
├── hyperparameter_search.py    # Dual optimization (Bayesian + Grid Search)
├── demonstrate_grid_search.py  # Grid search demonstration
├── train_transformers.py       # Transformer focus
├── CLAUDE.md                   # This file
└── (test files)
```

## Next Steps

1. Continue Phase 6 model-specific optimizations
2. Fine-tune ensemble weighting strategies
3. Test Grid Search vs Bayesian optimization performance
4. Implement advanced validation metrics (Sharpe ratio, maximum drawdown)
5. Create comprehensive model comparison framework