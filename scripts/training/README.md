# Transformer Training Scripts

This directory contains scripts and configurations for training the transformer ensemble models used in the RLTE system.

## Overview

The training system supports multiple transformer architectures:

- **LSTM**: Baseline recurrent model for time-series prediction
- **iTransformer**: Inverted attention transformer optimized for multivariate time-series
- **PatchTST**: Patch-based transformer for long sequence modeling  
- **TimesMixer**: Decomposition-based mixing transformer for temporal patterns
- **TimesFM**: Google's pre-trained foundation model (zero-shot, no training needed)

## Files

- `train_transformers.py` - Main training script
- `config.yaml` - Training configuration file
- `README.md` - This documentation

## Usage

### Basic Training

```bash
# Development environment (LSTM only)
python scripts/training/train_transformers.py --environment development

# Production environment (full ensemble)
python scripts/training/train_transformers.py --environment production
```

### Advanced Options

```bash
# Train with custom data source
python scripts/training/train_transformers.py \
    --environment production \
    --data-source initial \
    --split-ratio 0.8 0.1 0.1

# Train without GCS (local only)  
python scripts/training/train_transformers.py \
    --environment development \
    --no-gcs

# Full production training
ENVIRONMENT=production python scripts/training/train_transformers.py \
    --data-source initial
```

## Environment Configuration

### Development Mode
- **Purpose**: Fast iteration and testing
- **Models**: LSTM only
- **Resources**: Low memory/compute requirements
- **Duration**: ~5-10 minutes
- **Data**: Reduced model dimensions and epochs

### Production Mode  
- **Purpose**: Full ensemble training
- **Models**: LSTM, iTransformer, PatchTST, TimesMixer
- **Resources**: Full GPU/memory utilization
- **Duration**: 2-4 hours depending on data size
- **Data**: Full model dimensions and training epochs

## Model Architecture Details

### LSTM (Baseline)
- **Purpose**: Established baseline for comparison
- **Strengths**: Simple, reliable, good for sequential patterns
- **Configuration**: 2-3 layers, 64-128 hidden units
- **Training Time**: Fastest (~10-30 minutes)

### iTransformer
- **Purpose**: Multivariate time-series with inverted attention
- **Strengths**: Better correlation capture across features
- **Innovation**: Time points as tokens, features as channels
- **Training Time**: Medium (~30-60 minutes)

### PatchTST
- **Purpose**: Long sequence modeling with patch-based attention
- **Strengths**: Efficient for long horizons, reduced memory
- **Innovation**: Patch tokenization reduces sequence length
- **Training Time**: Medium (~45-75 minutes)

### TimesMixer
- **Purpose**: Temporal pattern decomposition and mixing
- **Strengths**: Handles seasonality and trends explicitly  
- **Innovation**: Multi-scale temporal mixing layers
- **Training Time**: Medium (~40-70 minutes)

### TimesFM (Zero-Shot)
- **Purpose**: Google's pre-trained foundation model
- **Strengths**: No training needed, strong generalization
- **Usage**: Direct inference on formatted data
- **Training Time**: None (pre-trained)

## Data Pipeline

The training script uses the existing infrastructure:

1. **ModelManager**: Coordinates ensemble training and weights
2. **InitialCorpusCollector**: Loads real market data from database  
3. **Database**: CloudSQL with `data_source='initial'` corpus
4. **Features**: 130+ technical indicators and market features
5. **Storage**: Local checkpoints + GCS bucket `gs://shyvr-models-prod`

## Training Process

1. **Initialization**: Load configuration and initialize infrastructure
2. **Data Loading**: Fetch training data from database (no mocks)
3. **Data Splitting**: 80/10/10 train/validation/test split
4. **Model Training**: Train each model with progress tracking
5. **Validation**: Evaluate on validation set during training
6. **Checkpointing**: Save model states with metadata
7. **Testing**: Final evaluation on holdout test set
8. **Summary**: Generate comprehensive training report

## Output Files

### Model Checkpoints
```
models/
├── lstm_checkpoint_20240808_143022_acc0.857.pt
├── lstm_checkpoint_20240808_143022_acc0.857.json
├── itransformer_checkpoint_20240808_144511_acc0.892.pt
└── itransformer_checkpoint_20240808_144511_acc0.892.json
```

### Training Reports  
```
reports/
├── training_summary_20240808_145030.json
└── training_logs_20240808_143022.log
```

### GCS Structure
```
gs://shyvr-models-prod/
├── models/
│   ├── lstm/checkpoints/
│   ├── itransformer/checkpoints/
│   ├── patchtst/checkpoints/
│   └── timesmixer/checkpoints/
└── metadata/
    └── training_sessions/
```

## Monitoring and Logging

The training script provides comprehensive monitoring:

- **Progress Tracking**: Real-time epoch and loss reporting
- **Performance Metrics**: Accuracy, MAE, RMSE, R² score  
- **Resource Usage**: Memory consumption and inference latency
- **Health Checks**: Model validation and error detection
- **Activity Logging**: Integration with existing logging system

## Configuration

Training behavior is controlled through:

1. **Environment Variables**: `ENVIRONMENT=development|production`
2. **Command Line Args**: Override default settings
3. **Config File**: `config.yaml` with detailed model settings
4. **Database Config**: Existing CloudSQL configuration

## Requirements

### Software Dependencies
- Python 3.8+
- PyTorch 1.9+
- Transformers library
- Existing RLTE infrastructure

### Hardware Requirements

#### Development
- **Memory**: 4GB+ RAM
- **Storage**: 2GB free space  
- **Compute**: CPU-only training supported

#### Production  
- **Memory**: 16GB+ RAM (32GB recommended)
- **Storage**: 10GB+ free space
- **Compute**: GPU with 8GB+ VRAM recommended
- **Network**: High-bandwidth for GCS uploads

## Troubleshooting

### Common Issues

**Out of Memory Errors**
```bash
# Reduce batch size in development mode
python train_transformers.py --environment development
```

**Database Connection Issues**
```bash
# Check CloudSQL connectivity
python -c "from src.utils.database import get_database_connection; print('OK')"
```

**GCS Upload Failures**  
```bash
# Train with local storage only
python train_transformers.py --no-gcs
```

**Missing Training Data**
```bash
# Collect initial corpus first
python -c "from src.data_pipeline.initial_corpus_collector import InitialCorpusCollector; import asyncio; asyncio.run(InitialCorpusCollector().collect_standardized_corpus())"
```

### Performance Optimization

1. **Use GPU**: Set `CUDA_VISIBLE_DEVICES=0` for GPU training
2. **Mixed Precision**: Enabled automatically in production mode
3. **Memory Optimization**: Use development mode for resource constraints
4. **Parallel Training**: Models are trained sequentially (by design)
5. **Checkpoint Recovery**: Resume from failed training (manual process)

## Integration

The training script integrates with:

- **Model Manager**: Existing ensemble coordination
- **Activity Logger**: Training event tracking  
- **Model Preservation**: Automatic backup and versioning
- **Database**: Real market data storage and retrieval
- **GCS**: Cloud storage for model artifacts
- **Monitoring**: Performance and health metrics

## Next Steps

After successful training:

1. **Validate Models**: Run comprehensive validation tests
2. **Deploy Models**: Update production model weights  
3. **Monitor Performance**: Track ensemble accuracy in live trading
4. **Retrain**: Schedule periodic retraining with new data
5. **Optimize**: Fine-tune model configurations based on results