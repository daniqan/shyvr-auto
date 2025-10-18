# Training Scripts

**Status**: Fully Integrated Pipeline with Phases 1-4 Complete

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
**Purpose**: Automated Bayesian optimization for hyperparameters  
**Features**:
- Gaussian Process optimization
- 20 trials per model
- Quick evaluation (5-10 epochs)
- Saves best parameters

**Usage**:
```bash
python scripts/training/hyperparameter_search.py \
    --n-trials 20 \
    --timeframe daily \
    --token WBTC
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

## Performance Tracking

Current vs Target:
- LSTM: 68% → 90%
- Transformer: 38% → 90%
- iTransformer: 3% → 90%
- PatchTST: 5% → 90%
- TimesMixer: 59% → 90%

## Files

```
scripts/training/
├── train_all_models.py         # Main pipeline
├── hyperparameter_search.py    # Optimization
├── train_transformers.py       # Transformer focus
├── CLAUDE.md                   # This file
└── (test files)
```

## Next Steps

1. Run hyperparameter search for optimal parameters
2. Train with optimized parameters
3. Evaluate on test set
4. Implement Phase 4 optimizations if needed