# Data Collection Scripts

This directory contains scripts and utilities for collecting, processing, and managing cryptocurrency market data used to train the RLTE (Reinforcement Learning Trading Engine) models.

## Overview

The data collection system supports both single-granularity and multi-granularity data collection, with comprehensive corpus management capabilities for reproducible ML model training.

### Key Features

- **Multi-timeframe collection**: Daily, 4-hour, and hourly data
- **Comprehensive feature extraction**: 130+ technical indicators and market features
- **Dynamic pagination**: Full historical data collection with checkpoint/resume capability
- **GCS integration**: Automatic export to Google Cloud Storage as Parquet files
- **Data validation**: Integrity checks and gap detection
- **Corpus versioning**: Version control for reproducible training datasets

## Scripts Overview

### Primary Collection Scripts

#### `collect_corpus.sh`
**Main unified corpus collection runner**

Supports both single-granularity and multi-granularity collection with various operation modes.

**Usage:**
```bash
# Clean all existing corpus data
./scripts/data_collection/collect_corpus.sh clean

# Test single granularity (2 tokens, 7 days)
./scripts/data_collection/collect_corpus.sh test

# Test multi-granularity (2 tokens, multiple timeframes)
./scripts/data_collection/collect_corpus.sh test-multi

# Production single granularity (10 tokens, 365 days)
./scripts/data_collection/collect_corpus.sh production

# Production multi-granularity with full capabilities
./scripts/data_collection/collect_corpus.sh production-multi

# Multi-granularity production (7 tokens, multiple timeframes)
./scripts/data_collection/collect_corpus.sh multi

# Dry run for multi-granularity
./scripts/data_collection/collect_corpus.sh multi-dry

# Custom configuration
./scripts/data_collection/collect_corpus.sh custom --tokens BTC ETH --timeframes daily hourly
```

**Available modes:**
- `clean`: Clean ALL existing corpus data
- `clean-multi`: Clean multi-granularity data only
- `test`: Test run (single granularity)
- `test-multi`: Test multi-granularity
- `production`: Production run (single granularity)
- `multi`: Multi-granularity production
- `production-multi`: Production multi with full pagination
- `multi-dry`: Multi-granularity dry run
- `custom`: Custom configuration

#### `collect_initial_corpus.sh`
**Legacy single-granularity collection runner**

**DEPRECATED:** Use `collect_corpus.sh` instead for unified support.

Maintained for backward compatibility only.

### Python Collection Scripts

#### `run_initial_corpus_collection.py`
**Initial corpus collection for single granularity**

Collects historical training data for transformer models with comprehensive feature extraction.

**Usage:**
```bash
# Test run (2 tokens, 7 days)
python scripts/data_collection/run_initial_corpus_collection.py --test

# Full production run (10 tokens, 180 days)
python scripts/data_collection/run_initial_corpus_collection.py --production

# Custom configuration
python scripts/data_collection/run_initial_corpus_collection.py \
    --tokens bitcoin ethereum solana \
    --days 30
```

**Features:**
- 6 months (180 days) of historical OHLCV data
- Market sentiment, DeFi metrics, and on-chain data
- Technical indicators and features calculation
- Production database storage with `data_source='initial'`

#### `run_parallel_corpus_collection.py`
**Parallel corpus collection for improved performance**

Optimized collection script supporting concurrent data fetching and processing.

### Data Management Scripts

#### `clean_corpus_data.py`
**Corpus data cleanup utility**

**Usage:**
```bash
# Clean all corpus data
python scripts/data_collection/clean_corpus_data.py --mode all

# Clean only multi-granularity data
python scripts/data_collection/clean_corpus_data.py --mode multi
```

#### `export_corpus_to_gcs.py`
**Export corpus data to Google Cloud Storage**

Exports collected corpus data as Parquet files to GCS for external analysis and backup.

**Features:**
- Parquet format for efficient storage
- Metadata preservation
- Batch processing for large datasets
- Compression optimization

#### `verify_corpus_data.py`
**Data integrity validation**

Validates collected corpus data for completeness, consistency, and quality.

**Checks performed:**
- Data completeness across timeframes
- Feature consistency
- Gap detection
- Statistical validation
- Schema validation

### Feature Management Scripts

#### `check_corpus_features.py`
**Feature analysis and validation**

Analyzes corpus features for training suitability and data quality.

#### `fix_feature_storage.py`
**Feature storage optimization**

Optimizes feature storage and resolves storage-related issues.

### Testing and Investigation Scripts

#### `test_coingecko_granularity.py`
**CoinGecko API granularity testing**

Tests different granularity options with CoinGecko API to ensure proper data collection.

#### `investigate_features.sh`
**Feature investigation utility**

Shell script for investigating corpus features issues with environment setup.

**Usage:**
```bash
./scripts/data_collection/investigate_features.sh
```

## Prerequisites and Dependencies

### Software Requirements

- **Python 3.8+** with asyncio support
- **PostgreSQL client** for database connections
- **Google Cloud SDK** for GCS integration
- **Cloud SQL Proxy** for secure database access
- **uv** package manager for Python dependencies

### Environment Variables

Required environment variables:
```bash
export GOOGLE_CLOUD_PROJECT="shvyr-ai-bots"
export ENVIRONMENT="development"
export SECRET_KEY="corpus_collection_secret_key_min_32_characters_long"
export DB_PORT="5433"
export DB_HOST="localhost"
export DB_USER="rlte_prod_user"
export DB_NAME="shyvr_rlte_prod"
```

### Cloud SQL Proxy Setup

Install and configure Cloud SQL Proxy:
```bash
# Download Cloud SQL Proxy
curl -o ~/cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.11.0/cloud-sql-proxy.darwin.amd64
chmod +x ~/cloud-sql-proxy

# The scripts will automatically start the proxy as needed
```

### API Keys

Configure API keys for external data sources:
- **CoinGecko API Key**: For cryptocurrency market data
- **LunarCrush API Key**: For social sentiment data
- **Helius API Key**: For Solana ecosystem data

## Common Workflows

### 1. Initial Setup and Testing

```bash
# 1. Test the collection system
./scripts/data_collection/collect_corpus.sh test

# 2. Verify collected data
python scripts/data_collection/verify_corpus_data.py

# 3. Check features
python scripts/data_collection/check_corpus_features.py
```

### 2. Production Data Collection

```bash
# 1. Clean existing data (if needed)
./scripts/data_collection/collect_corpus.sh clean

# 2. Run full production collection with multi-granularity
./scripts/data_collection/collect_corpus.sh production-multi

# 3. Export to GCS for backup
python scripts/data_collection/export_corpus_to_gcs.py

# 4. Validate collected data
python scripts/data_collection/verify_corpus_data.py
```

### 3. Multi-Granularity Collection

```bash
# 1. Test multi-granularity collection
./scripts/data_collection/collect_corpus.sh test-multi

# 2. Production multi-granularity with pagination
./scripts/data_collection/collect_corpus.sh production-multi

# 3. Verify data across all timeframes
python scripts/data_collection/verify_corpus_data.py --multi-granularity
```

### 4. Maintenance and Troubleshooting

```bash
# 1. Investigate feature issues
./scripts/data_collection/investigate_features.sh

# 2. Fix feature storage problems
python scripts/data_collection/fix_feature_storage.py

# 3. Clean partial/corrupted data
./scripts/data_collection/collect_corpus.sh clean-multi

# 4. Test API connectivity
python scripts/data_collection/test_coingecko_granularity.py
```

## Data Flow and Storage

### Collection Process

1. **Data Source APIs**: CoinGecko, LunarCrush, Helius
2. **Raw Data Processing**: OHLCV normalization and validation
3. **Feature Engineering**: Technical indicators and market features
4. **Database Storage**: PostgreSQL with appropriate granularity tags
5. **GCS Export**: Parquet files for external analysis
6. **Validation**: Data integrity and completeness checks

### Database Schema

Corpus data is stored with these key fields:
- `coin_id`: Token identifier
- `timestamp`: Data timestamp (UTC)
- `granularity`: Data timeframe (daily, 4hour, hourly)
- `data_source`: Collection source identifier
- Feature columns: 130+ technical indicators and market metrics

### GCS Storage Structure

```
gs://shyvr-models-prod/
├── corpus-data/
│   ├── v2.0/
│   │   ├── daily/
│   │   ├── 4hour/
│   │   └── hourly/
│   └── metadata/
│       ├── collection_logs/
│       └── validation_reports/
```

## Performance Considerations

### Collection Times

- **Test runs**: 5-10 minutes
- **Single granularity production**: 60-90 minutes
- **Multi-granularity production**: 2-4 hours
- **Production-multi with pagination**: 3-4 hours

### Resource Requirements

- **Memory**: 4GB+ RAM (8GB recommended for multi-granularity)
- **Storage**: 10GB+ free space for local caching
- **Network**: Stable internet for API calls and GCS uploads
- **Database**: PostgreSQL with sufficient storage and connection limits

### Optimization Tips

1. **Use test modes** for development and validation
2. **Enable pagination** for large historical collections
3. **Monitor API rate limits** to avoid throttling
4. **Use GCS caching** to reduce redundant data transfers
5. **Run collections during off-peak hours** for better performance

## Troubleshooting Guide

### Common Issues

#### Cloud SQL Proxy Connection Errors
```bash
# Check if proxy is running
lsof -i:5433

# Restart proxy manually
~/cloud-sql-proxy --port=5433 shvyr-ai-bots:us-central1:shyvr-rlte-db-prod &
```

#### API Rate Limiting
```bash
# Use test mode to reduce API calls
./scripts/data_collection/collect_corpus.sh test

# Check API key configuration
echo $COINGECKO_API_KEY
```

#### Incomplete Data Collection
```bash
# Resume collection from checkpoints (for paginated collections)
./scripts/data_collection/collect_corpus.sh production-multi

# Verify data completeness
python scripts/data_collection/verify_corpus_data.py
```

#### Feature Calculation Errors
```bash
# Investigate feature issues
./scripts/data_collection/investigate_features.sh

# Fix feature storage
python scripts/data_collection/fix_feature_storage.py
```

#### GCS Upload Failures
```bash
# Check GCS authentication
gcloud auth list

# Test GCS connectivity
python -c "from google.cloud import storage; print('GCS OK')"
```

### Error Codes and Solutions

- **Exit Code 1**: General script failure - check logs
- **Exit Code 2**: Database connection failed - verify Cloud SQL Proxy
- **Exit Code 3**: API authentication failed - check API keys
- **Exit Code 4**: Data validation failed - run verification scripts
- **Exit Code 5**: GCS upload failed - check authentication and permissions

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL="DEBUG"
./scripts/data_collection/collect_corpus.sh test
```

### Log Files

Collection logs are stored in:
- `/tmp/corpus_collection.log`
- GCS: `gs://shyvr-models-prod/corpus-data/logs/`

## Configuration Files

### Collection Configuration

Multi-granularity collection is configured via:
- `config/corpus_collection.yaml`: Production configuration
- `config/corpus_collection_test.yaml`: Test configuration

### Environment-Specific Settings

Development vs. Production settings are managed through:
- Environment variables
- Command-line arguments
- Configuration file selection

## Integration with Training Pipeline

The collected corpus data integrates with the training pipeline through:

1. **GCSCorpusLoader**: Loads data from GCS for training
2. **ModelManager**: Coordinates data access for ensemble training
3. **TrainingReportGenerator**: Uses corpus metadata for reporting
4. **Database tracking**: Records corpus versions for reproducibility

## Next Steps

After successful data collection:

1. **Verify data quality** using validation scripts
2. **Export to GCS** for backup and external access
3. **Run training pipeline** with collected corpus
4. **Monitor collection performance** and optimize as needed
5. **Schedule regular updates** to keep corpus current