# Training Data Corpus Implementation Plan - With Continuous Learning

## Implementation Status Summary

### ✅ Completed Phases
- **Phase 1.1**: Database schema design (migration 008 created and implemented)
- **Phase 2.0**: TDD test framework (comprehensive test suite with real API integration)
- **Phase 2.1**: Initial corpus collector (fully implemented with TDD compliance)

### 🚧 In Progress
- Phase 2.2-2.4: Continuous learning infrastructure
- Phase 3: Initial corpus collection (ready to execute)

### 📋 Pending
- Phases 4-9: Complete implementation and integration

### 🎯 Key Infrastructure Details
- **Database**: CloudSQL PostgreSQL with data lifecycle management
- **Storage**: GCS bucket `gs://shyvr-models-prod` with organized structure
- **Features**: 130+ features from real market data APIs (no mocks)
- **Real APIs**: CoinGecko, Alternative.me, DeFiLlama, LunarCrush*, Helius*
- **TDD**: All development follows strict test-first methodology

## Overview
Create a comprehensive training data corpus system that distinguishes between:
1. **Initial Training Data**: Standardized corpus for model initialization
2. **Live/Simulation Data**: Continuously collected data from production
3. **Online Learning**: Incremental training on new data
4. **Data Versioning**: Clear separation and tracking of data sources

### Important Clarification on Data Sources
- **ALL data is REAL market data** from APIs (CoinGecko, DeFiLlama, etc.)
- **"Simulation data"** = Real market data collected during paper trading mode
- **"Live data"** = Real market data collected during actual trading mode
- **The `data_source` flag indicates the MODE when data was collected, not data authenticity**
- **No synthetic or mock data is used in training**

## Phase 1: Database Infrastructure with Data Lifecycle Management (Day 1)

### 1.1 PostgreSQL Schema Design with Data Source Tracking - ✅ COMPLETED
- [x] Create `database/schemas/training_data_schema.sql` (COMPLETED - migration 008 created)
  
  #### Core Tables with Source Tracking
  - [ ] Table: `crypto_ohlcv`
    ```sql
    - data_source ENUM('initial', 'simulation', 'live', 'backtest')
    - collection_timestamp TIMESTAMPTZ
    - training_status ENUM('untrained', 'in_training', 'trained', 'archived')
    - model_version VARCHAR(50) -- Which model version used this data
    -- Note: Removed is_initial_corpus as redundant with data_source='initial'
    ```
  
  - [ ] Table: `crypto_features` - Technical indicators with source tracking
  - [ ] Table: `market_sentiment` - Fear & Greed with collection metadata
  - [ ] Table: `defi_metrics` - TVL data with source flags
  - [ ] Table: `onchain_metrics` - Transaction data with source tracking
  - [ ] Table: `social_sentiment` - LunarCrush data with metadata
  - [ ] Table: `token_metadata` - Token characteristics with update history
  
  #### Data Management Tables
  - [ ] Table: `training_corpus_versions`
    ```sql
    - version_id SERIAL PRIMARY KEY
    - version_name VARCHAR(100) -- e.g., 'initial_v1.0', 'live_2024_01'
    - created_at TIMESTAMPTZ
    - data_source ENUM('initial', 'simulation', 'live')
    - sample_count INTEGER
    - feature_count INTEGER
    - tokens TEXT[] -- Array of tokens included
    - is_active BOOLEAN -- Currently used for training
    ```
  
  - [ ] Table: `model_training_history`
    ```sql
    - training_id SERIAL PRIMARY KEY
    - model_type VARCHAR(50) -- 'lstm', 'itransformer', etc.
    - corpus_version_id INTEGER REFERENCES training_corpus_versions
    - trained_at TIMESTAMPTZ
    - training_mode ENUM('initial', 'incremental', 'fine_tune')
    - performance_metrics JSONB
    - model_checkpoint_path TEXT
    ```
  
  - [ ] Table: `continuous_learning_queue`
    ```sql
    - queue_id SERIAL PRIMARY KEY
    - data_batch_id VARCHAR(100)
    - collected_from TIMESTAMPTZ
    - collected_to TIMESTAMPTZ
    - data_source ENUM('simulation', 'live')
    - processing_status ENUM('pending', 'processing', 'completed', 'failed')
    - samples_count INTEGER
    ```

### 1.2 Data Partitioning Strategy
- [ ] Partition `crypto_ohlcv` by data_source and month
- [ ] Create indexes on (data_source, training_status, timestamp)
- [ ] Set up automated partition management
- [ ] Configure retention policies per data source:
  - Initial corpus: Permanent retention
  - Simulation data: 6 months rolling
  - Live data: 12 months rolling
  - Archived training data: 24 months

### 1.3 GCS Bucket Organization (Use Existing Infrastructure)
- [ ] Use existing `gs://shyvr-models-prod` bucket with new subdirectories:
  ```
  gs://shyvr-models-prod/
    ├── models/                    # Existing model storage
    ├── training-data/            # NEW: Training corpus data
    │   ├── initial-corpus/
    │   │   ├── v1.0/
    │   │   │   ├── raw/
    │   │   │   ├── processed/
    │   │   │   └── metadata.json
    │   │   └── v2.0/
    │   ├── continuous-learning/
    │   │   ├── simulation/
    │   │   │   ├── 2024-01/
    │   │   │   └── 2024-02/
    │   │   └── live/
    │   │       ├── 2024-01/
    │   │       └── 2024-02/
    │   └── archived/
    │       └── trained-batches/
    └── backups/                  # Existing backup infrastructure
  ```
- [ ] Leverage existing bucket configurations:
  - Versioning already enabled
  - Lifecycle policies configured
  - IAM permissions established
  - Monitoring and alerting active

## Phase 2: Data Collection Infrastructure with TDD Approach (Day 1-2)

### 2.0 Test-Driven Development Requirements - ✅ COMPLETED
- [x] **IMPORTANT**: All scripts must be developed using TDD methodology (COMPLETED)
- [x] Write tests FIRST before implementation (COMPLETED)
- [x] Tests must use REAL API calls, not mocks: (COMPLETED)
  - [ ] Real CoinGecko API calls for OHLCV data
  - [ ] Real Alternative.me API for Fear & Greed
  - [ ] Real DeFiLlama API for TVL data
  - [ ] Real LunarCrush API for social sentiment
  - [ ] Real Helius API for on-chain data
- [x] Create `tests/integration/data_pipeline/` directory structure:
  - [x] `test_initial_corpus_collector.py` - Test initial collection (COMPLETED - TDD tests created)
    - [x] Real API integration tests (CoinGecko, Alternative.me, DeFiLlama, LunarCrush*, Helius*)
    - [x] Data quality validation tests (130+ features, NaN checks, ranges)
    - [x] Rate limiting compliance tests
    - [x] Database storage with data_source='initial' flag tests
    - [x] Error handling and retry mechanism tests
    - [x] Feature completeness validation tests
    - [x] Corpus versioning management tests
    - Note: (*) Tests conditional on API key availability in environment
    - Status: FAILING as expected in TDD - implementation needed
  - [ ] `test_continuous_collector.py` - Test live/simulation collection
  - [ ] `test_feature_pipeline.py` - Test feature engineering
  - [ ] `test_storage_manager.py` - Test storage operations
- [ ] Each test must verify:
  - [ ] Data completeness (all 130+ features)
  - [ ] Data quality (no NaN, proper ranges)
  - [ ] API rate limiting compliance
  - [ ] Error handling and retries
  - [ ] Database storage integrity

## Phase 2.1: Data Collection Infrastructure with Source Management

### 2.1 Initial Corpus Collector - ✅ COMPLETED  
- [x] Create `src/data_pipeline/initial_corpus_collector.py` (COMPLETED - implementation created)
  - [x] Class: `InitialCorpusCollector` (COMPLETED)
  - [x] Method: `collect_standardized_corpus()` - One-time collection (COMPLETED)
  - [x] Method: `validate_corpus_completeness()` - Ensure all features (COMPLETED)
  - [x] Method: `mark_as_initial_corpus()` - Flag in database (COMPLETED)
  - [x] Method: `create_corpus_snapshot()` - Version control (COMPLETED)

### 2.2 Continuous Data Collector (Integrated with Existing Systems)
- [ ] Create `src/data_pipeline/continuous_collector.py`
  - [ ] Class: `ContinuousDataCollector`
  - [ ] Method: `collect_live_data()` - Real-time collection
  - [ ] Method: `collect_simulation_data()` - Simulation mode data
  - [ ] Method: `queue_for_training()` - Add to learning queue
  - [ ] Integration with existing drift detection:
    - Import `src.monitoring.drift_detection.EnhancedDriftDetector`
    - Import `src.modes.fallback_strategies.FallbackSystemIntegration`
    - Check drift before adding data to corpus
    - Trigger existing fallback mechanisms on significant drift

### 2.3 Online Learning Pipeline
- [ ] Create `src/data_pipeline/online_learning_pipeline.py`
  - [ ] Class: `OnlineLearningPipeline`
  - [ ] Method: `process_learning_queue()` - Process new data batches
  - [ ] Method: `incremental_feature_update()` - Update features
  - [ ] Method: `trigger_model_update()` - Initiate retraining
  - [ ] Method: `archive_trained_batch()` - Move to archive

### 2.4 Data Lifecycle Manager
- [ ] Create `src/data_pipeline/lifecycle_manager.py`
  - [ ] Class: `DataLifecycleManager`
  - [ ] Method: `separate_data_sources()` - Maintain separation
  - [ ] Method: `track_data_lineage()` - Track data flow
  - [ ] Method: `manage_versions()` - Version control
  - [ ] Method: `cleanup_old_data()` - Retention policies

## Phase 3: Initial Training Corpus Collection (Day 2)

### 3.1 One-Time Initial Corpus Script
- [ ] Create `scripts/collect_initial_corpus.py`
  ```python
  # Collects standardized training data for model initialization
  # Marks all data with data_source='initial' and is_initial_corpus=True
  ```
  - [ ] Collect 6 months historical data for 10 tokens
  - [ ] Calculate all 130+ features
  - [ ] Mark with `data_source='initial'`
  - [ ] Create immutable corpus version
  - [ ] Export to `gs://shyvr-models-prod/training-data/initial-corpus/v1.0/`

### 3.2 Initial Corpus Specifications
- [ ] **Tokens**: BTC, ETH, BNB, SOL, ADA, MATIC, AVAX, DOT, LINK, UNI
- [ ] **Time Range**: 6 months (4,320 hourly samples per token)
- [ ] **Total Samples**: 43,200 (10 tokens × 4,320 samples)
- [ ] **Features**: All 130+ standardized features
- [ ] **Storage**: Immutable, versioned, never modified

## Phase 4: Continuous Learning Infrastructure (Day 3)

### 4.1 Live Data Collection Service
- [ ] Create `src/services/live_data_service.py`
  - [ ] Class: `LiveDataService`
  - [ ] Method: `start_collection()` - Begin live collection
  - [ ] Method: `buffer_new_samples()` - Buffer before training
  - [ ] Method: `batch_for_training()` - Create training batches
  - [ ] Batch size: 24 hours of data (240 samples for 10 tokens)
  - [ ] Mark with `data_source='live'`

### 4.2 Simulation Mode Data Handler
- [ ] Create `src/services/simulation_data_handler.py`
  - [ ] Class: `SimulationDataHandler`
  - [ ] Method: `capture_simulation_trades()` - Real market data during paper trading
  - [ ] Method: `validate_data_quality()` - Quality checks on real data
  - [ ] Method: `prepare_for_training()` - Feature engineering
  - [ ] Mark with `data_source='simulation'` (real data collected during simulation mode)
  - [ ] Note: "Simulation data" = real market data collected while in simulation/paper trading mode

### 4.3 Incremental Training Manager
- [ ] Create `src/ml_analysis/incremental_training_manager.py`
  - [ ] Class: `IncrementalTrainingManager`
  - [ ] Method: `load_base_model()` - Load from initial training
  - [ ] Method: `prepare_incremental_batch()` - New data batch
  - [ ] Method: `update_model_weights()` - Incremental learning
  - [ ] Method: `validate_performance()` - Check for degradation
  - [ ] Method: `rollback_if_degraded()` - Safety mechanism
  - [ ] Method: `save_checkpoint()` - Version control

### 4.4 Model Versioning System
- [ ] Create `src/ml_analysis/model_versioning.py`
  - [ ] Class: `ModelVersionManager`
  - [ ] Track model lineage: initial → incremental updates
  - [ ] Maintain performance history
  - [ ] Support rollback to previous versions
  - [ ] A/B testing between versions

## Phase 5: Training Data Manager with Source Awareness (Day 3)

### 5.1 Enhanced Training Data Manager
- [ ] Update `src/ml_analysis/training_data_manager.py`
  - [ ] Class: `TrainingDataManager`
  - [ ] Method: `load_initial_corpus()` - Load only initial data
  - [ ] Method: `load_continuous_data()` - Load live/simulation data
  - [ ] Method: `create_mixed_batches()` - Combine sources for training
  - [ ] Method: `get_data_by_source()` - Filter by data source
  - [ ] Method: `track_data_usage()` - Log which data was used

### 5.2 Data Source Configuration
- [ ] Create `config/data_sources.yaml`
  ```yaml
  initial_corpus:
    version: "v1.0"
    path: "gs://shyvr-models-prod/training-data/initial-corpus/v1.0/"
    immutable: true
    
  continuous_learning:
    batch_size: 240  # samples
    update_frequency: "daily"
    sources:
      - simulation
      - live
    
  mixing_strategy:
    initial_weight: 0.7  # 70% initial corpus
    new_data_weight: 0.3  # 30% new data
  ```

## Phase 6: Model Training with Data Lifecycle (Day 4-5)

### 6.1 Initial Training Scripts
- [ ] Create `scripts/training/initial_training.py`
  - [ ] Use ONLY initial corpus data
  - [ ] Train all models from scratch
  - [ ] Save as base model versions
  - [ ] Record in `model_training_history`

### 6.2 Incremental Training Scripts
- [ ] Create `scripts/training/incremental_training.py`
  - [ ] Load base models
  - [ ] Use new data from continuous learning queue
  - [ ] Apply incremental learning techniques:
    - Elastic Weight Consolidation (EWC)
    - Learning rate scheduling
    - Replay buffer from initial corpus
  - [ ] Update model versions

### 6.3 Training Orchestrator
- [ ] Create `scripts/training/training_orchestrator.py`
  - [ ] Coordinate initial vs incremental training
  - [ ] Manage training queue
  - [ ] Monitor model performance
  - [ ] Trigger retraining when needed

## Phase 7: Monitoring & Data Quality (Day 5)

### 7.1 Data Drift Detection (Integration with Existing System)
- [ ] Extend existing `src/monitoring/drift_detection.py`:
  - [ ] Add training data specific drift monitoring
  - [ ] Use existing `EnhancedDriftDetector` class
  - [ ] Integrate with `FeatureDriftMonitor` for real-time monitoring
  - [ ] Connect to existing alerting via `src/monitoring/intelligent_alerting.py`
  - [ ] Leverage transformer-specific drift from `transformer_drift_detection.py`

### 7.2 Training Performance Dashboard
- [ ] Create `src/monitoring/training_dashboard.py`
  - [ ] Track data source usage
  - [ ] Monitor model performance by data source
  - [ ] Visualize incremental learning progress
  - [ ] Show data lifecycle status

### 7.3 Audit Trail System
- [ ] Create `src/monitoring/audit_trail.py`
  - [ ] Log all data movements
  - [ ] Track model-data associations
  - [ ] Maintain compliance records
  - [ ] Generate training reports

## Phase 8: Integration with Existing RLTE Systems

### 8.1 Safety System Integration
- [ ] Connect with `src/safety/ml_rl_safety_bridge.py`:
  - [ ] Register training data quality checks with safety monitors
  - [ ] Enable emergency stops on data corruption
  - [ ] Integrate with automated recovery systems

### 8.2 Model Registry Integration
- [ ] Update `src/ml_analysis/model_manager.py`:
  - [ ] Track which training corpus version each model used
  - [ ] Enable model rollback based on data issues
  - [ ] Support A/B testing with different training data

### 8.3 Mode System Integration
- [ ] Integrate with `src/modes/mode_manager.py`:
  - [ ] Different data labeling per mode (all using real market data):
    - [ ] Simulation mode: Real data labeled as `data_source='simulation'` (paper trading)
    - [ ] Production mode: Real data labeled as `data_source='live'` (actual trading)
    - [ ] Safety mode: Uses only initial corpus for predictions (no new collection)

### 8.4 Existing Database Integration
- [ ] Align with existing schemas in `database/migrations/`:
  - [ ] Follow naming conventions from existing tables
  - [ ] Use similar indexing strategies
  - [ ] Maintain foreign key relationships

## Phase 9: Archival & Cleanup (Day 5)

### 9.1 Data Archival Service
- [ ] Create `src/services/archival_service.py`
  - [ ] Move trained data to archive
  - [ ] Maintain data lineage
  - [ ] Compress old data
  - [ ] Update indices

### 9.2 Cleanup Policies
- [ ] Create `scripts/cleanup_old_data.py`
  - [ ] Remove processed simulation data > 6 months
  - [ ] Archive live data > 12 months
  - [ ] Never delete initial corpus
  - [ ] Clean up failed training batches

## Data Flow Architecture

```
Initial Corpus (One-time)
    ↓
[Initial Training] → Base Models v1.0
    ↓
                    
Live/Simulation Data (Continuous)
    ↓
[Buffer: 24 hours]
    ↓
[Continuous Learning Queue]
    ↓
[Incremental Training] → Updated Models v1.1, v1.2...
    ↓
[Archive Trained Data]
    ↓
[Cleanup Old Data]
```

## Storage Estimates with Lifecycle

### Initial Corpus (Permanent)
- **PostgreSQL**: 50GB (10 tokens, 6 months, all features)
- **GCS**: 20GB compressed
- **Status**: Never deleted, versioned

### Continuous Learning (Rolling)
- **Live Data Buffer**: 5GB/month
- **Simulation Data**: 3GB/month  
- **Archived Training Data**: 100GB/year
- **Active Window**: 10GB (last 30 days)

### Total Storage
- **Year 1**: ~200GB
- **Year 2**: ~350GB (with archival)
- **Cost**: ~$200/month

## Success Criteria

### Initial Training
- [ ] Initial corpus collected and versioned
- [ ] All models trained on standardized data
- [ ] Base model checkpoints saved
- [ ] Clear separation from continuous data

### Continuous Learning
- [ ] Live data collection running
- [ ] Daily incremental training working
- [ ] Model versions tracked
- [ ] Performance monitored

### Data Lifecycle
- [ ] Clear data source separation maintained
- [ ] Training history tracked
- [ ] Archival working
- [ ] No data mixing between sources

## Implementation Priority

1. **Week 1**: Initial corpus collection and base training
2. **Week 2**: Continuous data collection setup
3. **Week 3**: Incremental training implementation
4. **Week 4**: Monitoring and archival
5. **Ongoing**: Daily incremental updates

## Risk Mitigation

- **Data Contamination**: Strict source separation
- **Model Degradation**: Performance monitoring and rollback
- **Storage Growth**: Automated archival and cleanup
- **Training Failures**: Queue retry mechanism
- **Data Loss**: Multi-region backups