# ML Training Pipeline Scripts

This directory contains scripts and configurations for training all ML models used in the RLTE (Reinforcement Learning Trading Engine) system, providing a unified training pipeline for both traditional and transformer-based architectures.

## Overview

The training system supports multiple ML model architectures through a unified pipeline:

- **LSTM**: Baseline recurrent model for time-series prediction
- **Transformer**: Standard attention-based transformer for sequence modeling
- **iTransformer**: Inverted attention transformer optimized for multivariate time-series
- **PatchTST**: Patch-based transformer for efficient long sequence modeling  
- **TimesMixer**: Decomposition-based mixing transformer for temporal patterns
- **TimesFM**: Google's pre-trained foundation model (zero-shot, no training needed)

## Files

- `train_all_models.py` - **NEW**: Unified training pipeline for all models
- `train_transformers.py` - Legacy transformer-specific training script  
- `config.yaml` - Training configuration file with environment-specific settings
- `README.md` - This comprehensive documentation

## Primary Training Script: train_all_models.py

### Overview

The `train_all_models.py` script provides a complete unified training pipeline that:

- Loads real corpus data from GCS using GCSCorpusLoader
- Trains LSTM and all Transformer variants in a coordinated fashion
- Uses time-series aware train/validation/test splitting (80/10/10)
- Integrates with ModelManager for ensemble coordination
- Tracks training history in the `model_training_history` database table
- Saves trained models to GCS at `gs://shyvr-models-prod/trained-models/`
- Generates comprehensive training reports via TrainingReportGenerator
- Provides comprehensive error handling and logging

### Usage

#### Basic Training

```bash
# Train all models with default settings (Bitcoin data, daily timeframe)
python scripts/training/train_all_models.py

# The script runs asynchronously and provides real-time progress updates
```

#### Advanced Usage

The script can be imported and used programmatically:

```python
from scripts.training.train_all_models import UnifiedTrainingPipeline
import asyncio

async def custom_training():
    # Initialize pipeline with custom settings
    pipeline = UnifiedTrainingPipeline(
        gcs_bucket="shyvr-models-prod",
        model_save_bucket="shyvr-models-prod",
        cache_dir="/tmp/custom_training_cache"
    )
    
    # Train specific models on custom data
    results = await pipeline.train_all_models(
        corpus_version=None,  # Use latest corpus
        timeframe="hourly",   # Use hourly data
        token="ETH",         # Train on Ethereum data only
        models=['lstm', 'itransformer']  # Train specific models
    )
    
    print(f"Training completed: {results['session_id']}")
    return results

# Run custom training
results = asyncio.run(custom_training())
```

### Key Features

#### Real GCS Data Integration
- Loads actual corpus data from GCS (no mocks or test data)
- Supports multiple corpus versions and timeframes
- Handles large datasets with efficient caching

#### Time-Series Aware Data Splitting
- Preserves temporal order in train/validation/test splits
- Configurable split ratios (default: 80/10/10)
- Validates split integrity and completeness

#### Comprehensive Model Support
- **LSTM**: Traditional recurrent architecture
- **Transformer**: Standard attention-based model
- **iTransformer**: Inverted transformer for multivariate time-series
- **PatchTST**: Patch-based efficient transformer
- **TimesMixer**: Temporal decomposition transformer

#### Training Lifecycle Management
- Database tracking in `model_training_history` table
- GCS model persistence with metadata
- Training session management with unique session IDs
- Performance metrics tracking and reporting

## Legacy Training Script: train_transformers.py

### Usage

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

## Configuration Details

### Configuration File: config.yaml

The training system uses a comprehensive YAML configuration file that defines environment-specific settings, model configurations, and training parameters.

#### Environment-Specific Settings

**Development Mode:**
- **Purpose**: Fast iteration and testing
- **Models**: LSTM only for rapid development
- **Resources**: Optimized for low memory/compute requirements
- **Duration**: ~10-20 minutes
- **Configuration**:
  ```yaml
  development:
    models_to_train: ['lstm']
    batch_size: 16
    max_epochs: 10
    memory_optimization: true
    model_configs:
      lstm:
        hidden_dim: 64
        num_layers: 2
        sequence_length: 60
  ```

**Production Mode:**
- **Purpose**: Full ensemble training for production deployment
- **Models**: LSTM, iTransformer, PatchTST, TimesMixer (full ensemble)
- **Resources**: Full GPU/memory utilization with mixed precision
- **Duration**: 2-4 hours depending on data size and model complexity
- **Configuration**:
  ```yaml
  production:
    models_to_train: ['lstm', 'itransformer', 'patchtst', 'timesmixer']
    batch_size: 32
    max_epochs: 50
    enable_mixed_precision: true
    memory_optimization: false
  ```

#### Model-Specific Configurations

Each model architecture has tailored configurations optimized for performance:

**LSTM Configuration:**
```yaml
lstm:
  hidden_dim: 128          # Hidden layer dimension
  num_layers: 3            # Number of LSTM layers
  sequence_length: 120     # Input sequence length
  dropout: 0.1             # Dropout rate
  learning_rate: 0.001     # Learning rate
```

**iTransformer Configuration:**
```yaml
itransformer:
  d_model: 512             # Model dimension
  n_heads: 8               # Number of attention heads
  n_layers: 6              # Number of transformer layers
  n_variates: 12           # Number of time series variates
  use_inverted_attention: true
  cross_variate_attention: true
```

**PatchTST Configuration:**
```yaml
patchtst:
  d_model: 512             # Model dimension
  n_heads: 8               # Number of attention heads
  n_layers: 6              # Number of transformer layers
  patch_size: 16           # Patch size for tokenization
  stride: 8                # Patch stride
  prediction_horizons: ['1h', '4h', '24h']
```

**TimesMixer Configuration:**
```yaml
timesmixer:
  d_model: 512             # Model dimension
  n_heads: 8               # Number of attention heads
  n_layers: 6              # Number of layers
  mixing_factor: 0.5       # Temporal mixing factor
  decomposition_layers: 2  # Number of decomposition layers
  seasonal_periods: [24, 168]  # Seasonal patterns (daily, weekly)
```

## Training Pipeline Workflow

### UnifiedTrainingPipeline Process

The `train_all_models.py` script follows a comprehensive training workflow:

```python
async def train_all_models():
    """
    Complete training pipeline workflow:
    
    1. Data Loading
       - Load corpus data from GCS using GCSCorpusLoader
       - Support for specific corpus versions, timeframes, and tokens
       - Data validation and integrity checks
       
    2. Data Preparation
       - Time-series aware train/validation/test splitting (80/10/10)
       - Temporal order preservation
       - Feature engineering and normalization
       
    3. Model Training
       - Sequential training of LSTM and Transformer models
       - Real-time progress monitoring and logging
       - Comprehensive error handling and recovery
       
    4. Model Persistence
       - Save trained models to GCS with metadata
       - Database tracking in model_training_history table
       - Training session management
       
    5. Report Generation
       - Comprehensive training reports via TrainingReportGenerator
       - Performance metrics and model comparison
       - GCS upload of reports and artifacts
    """
```

### Training Session Management

Each training session is tracked with comprehensive metadata:

- **Session ID**: Unique identifier for training session
- **Training Metrics**: Performance metrics for each model
- **Data Statistics**: Information about training data used
- **Model Artifacts**: GCS paths to saved models
- **Reports**: Generated training reports and analysis

## Model Architecture Details

### LSTM (Baseline)
- **Purpose**: Established baseline for time-series prediction
- **Strengths**: Simple, reliable, excellent for sequential patterns
- **Architecture**: 2-3 LSTM layers with 64-128 hidden units
- **Training Time**: Fastest (~15-30 minutes)
- **Use Case**: Baseline comparison and rapid prototyping
- **Configuration**: Optimized for both development and production modes

### Transformer (Standard)
- **Purpose**: Standard attention-based sequence modeling
- **Strengths**: Parallel processing, long-range dependencies
- **Architecture**: Multi-head attention with 4-8 heads, 3-6 layers
- **Training Time**: Medium (~45-90 minutes)
- **Use Case**: General-purpose sequence modeling with attention mechanisms

### iTransformer (Inverted Attention)
- **Purpose**: Multivariate time-series with inverted attention mechanism
- **Strengths**: Superior correlation capture across multiple time series
- **Innovation**: Time points as tokens, features as channels (inverted approach)
- **Architecture**: Inverted attention with cross-variate modeling
- **Training Time**: Medium (~30-60 minutes)
- **Use Case**: Multi-asset correlation modeling and cross-market analysis

### PatchTST (Patch-Based)
- **Purpose**: Efficient long sequence modeling with patch tokenization
- **Strengths**: Memory efficient for long horizons, reduced computational complexity
- **Innovation**: Patch-based tokenization significantly reduces sequence length
- **Architecture**: Patch size 8-16, stride 4-8, efficient attention
- **Training Time**: Medium (~45-75 minutes)
- **Use Case**: Long-term forecasting with limited computational resources

### TimesMixer (Temporal Decomposition)
- **Purpose**: Temporal pattern decomposition and multi-scale mixing
- **Strengths**: Explicitly handles seasonality, trends, and irregular patterns
- **Innovation**: Multi-scale temporal mixing layers with decomposition
- **Architecture**: Mixing factor 0.3-0.7, decomposition layers 2-3
- **Training Time**: Medium (~40-70 minutes)
- **Use Case**: Complex temporal pattern recognition and seasonal modeling

### TimesFM (Pre-trained Foundation Model)
- **Purpose**: Google's pre-trained foundation model for time-series
- **Strengths**: Zero-shot capability, strong generalization, no training required
- **Usage**: Direct inference on formatted time-series data
- **Architecture**: Pre-trained 200M parameter model
- **Training Time**: None (inference only)
- **Use Case**: Quick deployment and comparison baseline

## Data Pipeline Integration

The training system integrates with the comprehensive RLTE data infrastructure:

### Core Data Components

1. **GCSCorpusLoader**: Primary data loading from Google Cloud Storage
   - Loads versioned corpus data as Parquet files
   - Supports multiple timeframes (daily, 4-hour, hourly)
   - Efficient caching with configurable TTL
   - Automatic data validation and integrity checks

2. **ModelManager**: Ensemble coordination and model management
   - Orchestrates training across multiple model architectures
   - Manages model weights and ensemble configuration
   - Provides unified interface for model access
   - Handles model versioning and deployment coordination

3. **Database Integration**: CloudSQL PostgreSQL database
   - Training history tracking in `model_training_history` table
   - Corpus versioning in `training_corpus_versions` table
   - Performance metrics storage and retrieval
   - Training session metadata management

4. **Feature Pipeline**: 130+ technical indicators and market features
   - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Price action features (Support/Resistance levels)
   - Volume analysis (Volume Profile, OBV)
   - Market structure (Volatility, Momentum indicators)
   - Cross-asset correlations and sentiment data

5. **GCS Storage**: Distributed model and data storage
   - Model artifacts: `gs://shyvr-models-prod/trained-models/`
   - Corpus data: `gs://shyvr-models-prod/corpus-data/`
   - Training reports: `gs://shyvr-models-prod/reports/`
   - Model checkpoints with metadata preservation

### Training Process Workflow

The unified training pipeline follows a systematic process:

#### 1. **Initialization Phase**
```python
# Configuration loading and validation
pipeline = UnifiedTrainingPipeline(
    gcs_bucket="shyvr-models-prod",
    model_save_bucket="shyvr-models-prod",
    cache_dir="/tmp/training_cache"
)

# Environment setup and authentication
await pipeline.initialize_environment()
```

#### 2. **Data Loading Phase**
```python
# Load corpus data from GCS
data = await pipeline.load_corpus_data(
    corpus_version=None,  # Latest version
    timeframe="daily",    # Data granularity
    token="BTC"          # Specific token or None for all
)

# Data validation and preprocessing
validated_data = await pipeline.validate_and_preprocess(data)
```

#### 3. **Data Splitting Phase**
```python
# Time-series aware splitting (preserves temporal order)
train_data, val_data, test_data = pipeline.prepare_train_val_test_split(
    data, 
    train_ratio=0.8,
    val_ratio=0.1, 
    test_ratio=0.1
)
```

#### 4. **Model Training Phase**
```python
# Sequential model training with progress tracking
lstm_results = await pipeline.train_lstm_model(train_data)
transformer_results = await pipeline.train_transformer_models(
    train_data, 
    models=['transformer', 'itransformer', 'patchtst', 'timesmixer']
)
```

#### 5. **Validation and Testing Phase**
```python
# Continuous validation during training
# Early stopping based on validation performance
# Final evaluation on holdout test set
```

#### 6. **Model Persistence Phase**
```python
# Save models to GCS with metadata
saved_paths = await pipeline.save_trained_models(
    training_results, 
    model_instances
)

# Database tracking
training_ids = await pipeline.track_training_history(
    training_results, 
    saved_paths
)
```

#### 7. **Report Generation Phase**
```python
# Comprehensive training report generation
report_results = await pipeline.report_generator.generate_reports_for_training_session(
    training_metrics=all_results,
    pipeline=pipeline
)
```

### Data Quality and Validation

The training pipeline includes comprehensive data quality assurance:

- **Input Validation**: Schema validation, missing data detection
- **Feature Quality**: Statistical validation of engineered features  
- **Temporal Consistency**: Ensuring proper time-series ordering
- **Split Validation**: Verifying train/val/test split integrity
- **Model Input Validation**: Ensuring compatibility with model architectures

## Output Files and Artifacts

### Model Checkpoints (Local)
```
models/
├── lstm_checkpoint_20250826_143022_acc0.857.pt
├── lstm_checkpoint_20250826_143022_acc0.857.json
├── transformer_checkpoint_20250826_144511_acc0.892.pt
├── transformer_checkpoint_20250826_144511_acc0.892.json
├── itransformer_checkpoint_20250826_145030_acc0.901.pt
├── itransformer_checkpoint_20250826_145030_acc0.901.json
├── patchtst_checkpoint_20250826_150145_acc0.888.pt
├── patchtst_checkpoint_20250826_150145_acc0.888.json
├── timesmixer_checkpoint_20250826_151200_acc0.895.pt
└── timesmixer_checkpoint_20250826_151200_acc0.895.json
```

### Training Reports (Local)
```
reports/
├── training_summary_20250826_152030.json
├── training_logs_20250826_143022.log
├── model_comparison_20250826.pdf
├── performance_analysis_20250826.html
└── training_session_20250826_metadata.json
```

### GCS Storage Structure
```
gs://shyvr-models-prod/
├── trained-models/                    # NEW: Unified training pipeline models
│   ├── lstm/
│   │   └── 20250826_143022/
│   │       ├── model.pt
│   │       └── metadata.json
│   ├── transformer/
│   │   └── 20250826_144511/
│   ├── itransformer/
│   │   └── 20250826_145030/
│   ├── patchtst/
│   │   └── 20250826_150145/
│   └── timesmixer/
│       └── 20250826_151200/
├── models/                            # Legacy: transformer training models
│   ├── lstm/checkpoints/
│   ├── itransformer/checkpoints/
│   ├── patchtst/checkpoints/
│   └── timesmixer/checkpoints/
├── reports/                           # Training reports and analysis
│   ├── training_sessions/
│   │   └── 20250826_152030/
│   │       ├── training_report.pdf
│   │       ├── model_comparison.html
│   │       └── session_metadata.json
│   └── performance_analysis/
└── metadata/
    └── training_sessions/
```

### Database Schema Integration

#### model_training_history Table
```sql
CREATE TABLE model_training_history (
    training_id BIGSERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,           -- 'lstm', 'transformer', 'itransformer', etc.
    corpus_version_id INTEGER,                 -- Links to training_corpus_versions
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_mode VARCHAR(20) DEFAULT 'initial',
    
    -- Performance metrics
    performance_metrics JSONB,                 -- Accuracy, loss, validation scores
    training_duration_seconds INTEGER,
    model_checkpoint_path VARCHAR(500),        -- GCS path to saved model
    model_version VARCHAR(100),                -- Version identifier
    training_config JSONB,                     -- Model configuration used
    
    -- Session tracking
    session_id UUID,                           -- UnifiedTrainingPipeline session ID
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### training_corpus_versions Table
```sql
CREATE TABLE training_corpus_versions (
    version_id BIGSERIAL PRIMARY KEY,
    version_name VARCHAR(100) NOT NULL,        -- 'v2.0', 'daily_20250826', etc.
    corpus_source VARCHAR(100),                -- 'multi_granularity', 'initial'
    timeframe VARCHAR(20),                     -- 'daily', '4hour', 'hourly'
    
    -- Data statistics
    total_records INTEGER,
    feature_count INTEGER,
    token_list TEXT[],                         -- List of tokens included
    date_range_start TIMESTAMP,
    date_range_end TIMESTAMP,
    
    -- Metadata
    collection_metadata JSONB,
    data_quality_score DECIMAL(4,3),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Monitoring and Logging

The training pipeline provides comprehensive monitoring and observability:

### Real-time Progress Tracking
- **Training Progress**: Real-time epoch and batch-level progress reporting
- **Loss Monitoring**: Training and validation loss tracking with visualization
- **Performance Metrics**: Accuracy, MAE, RMSE, R² score calculation and reporting
- **Resource Usage**: Memory consumption, GPU utilization, and inference latency monitoring
- **Session Tracking**: Unique session IDs for complete training run traceability

### Logging Infrastructure
```python
# Structured logging with context
logger.info("Training LSTM model", 
           model_type="lstm",
           session_id=session_id,
           data_samples=len(training_data),
           config=model_config)

# Performance metrics logging
logger.info("Model training completed",
           model_type="lstm",
           training_duration_seconds=duration,
           final_accuracy=accuracy,
           validation_loss=val_loss)
```

### Health Checks and Validation
- **Model Validation**: Automatic model architecture validation
- **Data Quality Checks**: Input data validation and integrity verification
- **Training Stability**: Loss convergence monitoring and divergence detection
- **Memory Management**: Out-of-memory detection and recovery
- **GCS Connectivity**: Upload success verification and retry logic

### Integration with Existing Systems
- **Activity Logger**: Seamless integration with RLTE activity logging system
- **Database Logging**: Training events stored in PostgreSQL for historical analysis
- **TrainingReportGenerator**: Automated report generation with performance analysis
- **Error Tracking**: Comprehensive error classification and reporting

## Requirements and Dependencies

### Software Requirements

#### Core Dependencies
- **Python**: 3.8+ (3.10+ recommended)
- **PyTorch**: 1.9+ with GPU support (2.0+ recommended)
- **Transformers**: HuggingFace transformers library for model architectures
- **Pandas/NumPy**: Data processing and numerical computation
- **Google Cloud SDK**: GCS integration and authentication
- **PostgreSQL Client**: Database connectivity (psycopg2-binary)
- **UV Package Manager**: For dependency management and virtual environments

#### RLTE-Specific Dependencies
```python
# Core RLTE modules required
from src.data_pipeline.gcs_corpus_loader import GCSCorpusLoader
from src.ml_analysis.lstm_model import LSTMPricePredictor
from src.ml_analysis.transformers import TransformerPredictor, iTransformerPredictor
from src.ml_analysis.model_manager import ModelManager
from src.ml_analysis.training_report_generator import TrainingReportGenerator
from src.utils.database import get_database_connection
```

### Hardware Requirements

#### Development Environment
- **Memory**: 8GB+ RAM (16GB recommended)
- **Storage**: 5GB+ free space for models and cache
- **Compute**: CPU-only training supported (slower)
- **Network**: Stable internet for GCS uploads

#### Production Environment  
- **Memory**: 32GB+ RAM (64GB recommended for large models)
- **Storage**: 50GB+ free space for models, checkpoints, and data cache
- **Compute**: GPU with 16GB+ VRAM (Tesla V100/A100 recommended)
- **Network**: High-bandwidth connection for large model and data transfers

#### Cloud Environment (Recommended)
- **GCP Compute Engine**: n1-highmem-8 or higher
- **GPU**: NVIDIA Tesla V100, T4, or A100
- **Storage**: SSD persistent disks for performance
- **Network**: Premium tier for faster GCS transfers

## Configuration Management

### Environment-Based Configuration

Training behavior is controlled through multiple configuration layers:

#### 1. Environment Variables
```bash
# Core environment settings
export GOOGLE_CLOUD_PROJECT="shvyr-ai-bots"
export ENVIRONMENT="development"  # or "production"
export GCS_BUCKET="shyvr-models-prod"

# Database configuration
export DB_HOST="localhost"
export DB_PORT="5433" 
export DB_USER="rlte_prod_user"
export DB_NAME="shyvr_rlte_prod"

# Optional optimization settings
export CUDA_VISIBLE_DEVICES="0"
export TORCH_HOME="/tmp/torch_cache"
```

#### 2. Configuration File (config.yaml)
```yaml
# Environment-specific model configurations
development:
  models_to_train: ['lstm']
  max_epochs: 10
  batch_size: 16

production:
  models_to_train: ['lstm', 'itransformer', 'patchtst', 'timesmixer']
  max_epochs: 50
  batch_size: 32
```

#### 3. Command Line Arguments (Legacy Script)
```bash
# Override configuration through CLI
python scripts/training/train_transformers.py \
    --environment production \
    --data-source initial \
    --split-ratio 0.8 0.1 0.1
```

#### 4. Programmatic Configuration
```python
# Direct configuration in code
pipeline = UnifiedTrainingPipeline(
    gcs_bucket="shyvr-models-prod",
    model_save_bucket="custom-bucket",
    cache_dir="/custom/cache/path",
    cache_ttl_hours=48
)
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Memory-Related Issues

**Out of Memory Errors**
```bash
# Solution 1: Use development mode with smaller models
python scripts/training/train_all_models.py  # Automatically uses appropriate config

# Solution 2: Reduce batch size in config.yaml
development:
  batch_size: 8  # Reduced from 16

# Solution 3: Enable memory optimization
production:
  memory_optimization: true
```

**Memory Leaks During Long Training**
```python
# Monitor memory usage
import psutil
process = psutil.Process()
logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Enable garbage collection
import gc
gc.collect()
torch.cuda.empty_cache()  # Clear GPU memory
```

#### Database Connection Issues

**Cloud SQL Proxy Connection Failed**
```bash
# Check proxy status
lsof -i:5433

# Restart proxy with proper authentication
~/cloud-sql-proxy --port=5433 \
    --credentials-file=$HOME/.config/gcloud/application_default_credentials.json \
    shvyr-ai-bots:us-central1:shyvr-rlte-db-prod
```

**Database Authentication Errors**
```bash
# Verify database connection
python -c "
import asyncio
from src.utils.database import get_database_connection
async def test(): 
    async with get_database_connection() as conn:
        result = await conn.fetchrow('SELECT 1 as test')
        print(f'Database OK: {result}')
asyncio.run(test())
"
```

#### GCS Integration Issues

**Authentication Failures**
```bash
# Check GCS authentication
gcloud auth list
gcloud auth application-default login

# Test GCS connectivity
python -c "
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('shyvr-models-prod')
print('GCS authentication OK')
"
```

**Upload Timeouts**
```python
# Configure retry policy for uploads
from google.cloud import storage
from google.api_core import retry

client = storage.Client()
bucket = client.bucket('shyvr-models-prod')
blob = bucket.blob('test-upload')

# Custom retry policy
retry_policy = retry.Retry(
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    deadline=300.0
)

blob.upload_from_filename('model.pt', retry=retry_policy)
```

#### Training Performance Issues

**Slow Training Progress**
```bash
# Check GPU utilization
nvidia-smi

# Enable mixed precision training (production mode)
production:
  enable_mixed_precision: true

# Use GPU acceleration
export CUDA_VISIBLE_DEVICES="0"
```

**Model Convergence Issues**
```yaml
# Adjust learning rates in config.yaml
optimization:
  learning_rates:
    lstm: 0.0001        # Reduced from 0.001
    transformer: 0.00005  # Reduced from 0.0001

# Enable early stopping
early_stopping:
  patience: 10          # Increased patience
  min_delta: 0.0001     # Reduced minimum improvement
```

### Debug Mode and Logging

**Enable Debug Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL="DEBUG"
```

**Training Session Debugging**
```python
# Access training session details
results = await pipeline.train_all_models()
session_id = results['session_id']
training_results = results['training_results']

# Check individual model results
for model_name, result in training_results.items():
    if not result.get('success'):
        print(f"Model {model_name} failed: {result.get('error')}")
    else:
        print(f"Model {model_name} trained successfully: {result['model_accuracy']:.4f}")
```

## Integration with RLTE Ecosystem

The training pipeline seamlessly integrates with the broader RLTE system:

### Core System Integration
- **ModelManager**: Ensemble coordination and model weight management
- **Database Schema**: Training history tracking and corpus versioning
- **GCS Storage**: Centralized model artifact storage and versioning
- **Activity Logging**: Comprehensive event tracking and monitoring
- **TrainingReportGenerator**: Automated performance analysis and reporting

### Production Deployment Integration
- **Model Versioning**: Automatic versioning for production deployment
- **Performance Tracking**: Integration with live trading performance monitoring
- **Ensemble Updates**: Coordination with production ensemble weight updates
- **Backup and Recovery**: Automatic model backup and disaster recovery

### Data Pipeline Integration
- **Corpus Loading**: Direct integration with GCSCorpusLoader for training data
- **Feature Engineering**: Utilizes existing 130+ feature pipeline
- **Data Validation**: Integration with corpus validation and quality assurance
- **Multi-granularity Support**: Seamless training across different timeframes

## Best Practices and Recommendations

### Development Workflow
1. **Start with Development Mode**: Always begin with development configuration for quick iteration
2. **Validate Data First**: Ensure corpus data quality before training
3. **Monitor Resource Usage**: Track memory and GPU utilization during training
4. **Use Checkpoints**: Save progress regularly for long training runs
5. **Validate Models**: Test trained models on holdout data before deployment

### Production Deployment
1. **Full Environment Setup**: Ensure all dependencies and authentication are configured
2. **Resource Allocation**: Allocate sufficient compute resources for production training
3. **Monitoring Setup**: Enable comprehensive monitoring and alerting
4. **Backup Strategy**: Implement automated backup of trained models
5. **Performance Validation**: Thoroughly validate models before production deployment

## Future Enhancements and Roadmap

### Short-term Improvements
- **Parallel Model Training**: Train multiple models concurrently where possible
- **Hyperparameter Optimization**: Automated hyperparameter tuning integration
- **Advanced Monitoring**: Real-time training dashboard with visualization
- **Model Comparison Tools**: Enhanced model performance comparison utilities

### Long-term Vision
- **AutoML Integration**: Automated model architecture search and optimization
- **Distributed Training**: Multi-node distributed training for large models
- **Real-time Model Updates**: Incremental training with streaming data
- **Advanced Ensembling**: Dynamic ensemble weight optimization based on market conditions