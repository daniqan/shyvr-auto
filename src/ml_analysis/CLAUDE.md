# ML Analysis Module Documentation

## Feature Engineering Pipeline

### Data Sources and APIs

The FeatureEngineer collects and processes data from multiple sources:

1. **CoinGecko API**
   - OHLCV data (Open, High, Low, Close, Volume)
   - Historical price data at multiple granularities
   - Market cap and circulating supply

2. **The Graph Protocol**
   - On-chain DeFi metrics
   - Liquidity pool data
   - Trading volume from DEXs

3. **LunarCrush API**
   - Social sentiment scores
   - Social volume metrics
   - Galaxy scores and alt ranks

4. **Helius API (Solana)**
   - Solana-specific on-chain data
   - Transaction metrics
   - Token holder information

### Complete Feature List (65 Features)

The FeatureEngineer creates the following features from raw data:

#### Core OHLCV Features (from CoinGecko)
1. `open` - Opening price
2. `high` - Highest price in period
3. `low` - Lowest price in period
4. `close` - Closing price
5. `volume` - Trading volume

#### Technical Indicators
6. `rsi_14` - Relative Strength Index (14 periods)
7. `rsi_7` - Relative Strength Index (7 periods)
8. `rsi_21` - Relative Strength Index (21 periods)
9. `macd` - MACD line
10. `macd_signal` - MACD signal line
11. `macd_histogram` - MACD histogram
12. `bb_upper` - Bollinger Band upper
13. `bb_middle` - Bollinger Band middle
14. `bb_lower` - Bollinger Band lower
15. `bb_position` - Position within Bollinger Bands (0-1)
16. `volume_sma_20` - Volume 20-period SMA
17. `ema_12` - 12-period EMA
18. `ema_26` - 26-period EMA
19. `ema_50` - 50-period EMA
20. `ema_200` - 200-period EMA
21. `sma_20` - 20-period SMA
22. `sma_50` - 50-period SMA
23. `sma_200` - 200-period SMA
24. `volume_ema` - Volume EMA
25. `atr` - Average True Range
26. `adx` - Average Directional Index
27. `cci` - Commodity Channel Index
28. `stoch_k` - Stochastic %K
29. `stoch_d` - Stochastic %D
30. `williams_r` - Williams %R
31. `obv` - On-Balance Volume
32. `vwap` - Volume Weighted Average Price

#### Returns and Price Changes
33. `returns_1h` - 1-hour returns
34. `returns_24h` - 24-hour returns
35. `returns_7d` - 7-day returns
36. `price_change` - Simple price change
37. `price_change_1h` - 1-hour price change
38. `price_change_4h` - 4-hour price change
39. `price_change_24h` - 24-hour price change
40. `price_change_7d` - 7-day price change

#### Volatility Metrics
41. `volatility` - Price volatility
42. `volatility_1h` - 1-hour volatility
43. `volatility_24h` - 24-hour volatility
44. `volatility_ratio` - Volatility ratio
45. `realized_volatility` - Realized volatility

#### Price Ratios and Relationships
46. `high_low_ratio` - High/Low ratio
47. `close_open_ratio` - Close/Open ratio
48. `close_to_high` - Distance from close to high
49. `close_to_low` - Distance from close to low
50. `volume_price_ratio` - Volume/Price ratio

#### Momentum Indicators
51. `momentum_5` - 5-period momentum
52. `momentum_10` - 10-period momentum
53. `momentum_20` - 20-period momentum
54. `price_momentum` - Price momentum score

#### Time-Based Features
55. `hour` - Hour of day (0-23)
56. `day_of_week` - Day of week (0-6)
57. `month` - Month (1-12)
58. `quarter` - Quarter (1-4)
59. `is_weekend` - Weekend indicator (0/1)
60. `trading_session` - Trading session (Asia/Europe/US)
61. `hour_sin` - Sine encoding of hour
62. `hour_cos` - Cosine encoding of hour
63. `day_sin` - Sine encoding of day
64. `day_cos` - Cosine encoding of day

#### ML-Derived Scores
65. `volatility_score` - Normalized volatility score (0-1)
66. `volume_score` - Normalized volume score (0-1)
67. `momentum_score` - Momentum strength score
68. `market_regime` - Market regime classification (0: Bear, 1: Neutral, 2: Bull)
69. `trend_strength` - Trend strength indicator (-1 to 1)

## Corpus Data Format

### Storage Format
- **Type**: Apache Parquet files
- **Compression**: Snappy
- **Location**: 
  - Local: `data/corpus/v2.0/{granularity}/{token}_{granularity}.parquet`
  - GCS: `gs://shyvr-models-prod/training-data/initial-corpus/v2.0/`

### DataFrame Structure
```python
# Shape: (n_timesteps, 65+ features)
# Index: DatetimeIndex or RangeIndex
# Columns: All 65+ features listed above plus metadata

# Metadata columns (not used for training):
- id
- ohlcv_id  
- token_id
- timestamp
- feature_version
- calculated_at
- data_source
- granularity
- created_at
- updated_at
- collection_timestamp
```

### Granularities Available
- `daily`: 365 days of data
- `four_hour`: 180 days of data  
- `hourly`: 90 days of data
- `fifteen_minute`: 30 days of data (optional)

## Model Input Requirements

### 1. LSTM Model

**Method**: `prepare_training_from_corpus(data, sequence_length=50, prediction_horizons=[1,4,24])`

**Input Shape**:
- Training: `(n_samples, 50, 65)` 
  - n_samples: Number of sequences
  - 50: Timesteps (sequence_length)
  - 65: All numeric features

**Output Shape**:
- Predictions: `(n_samples, 3)` - [1h, 4h, 24h price predictions]

**Features Used**: ALL available numeric features (excluding metadata)

**Preprocessing**:
- Forward fill then backward fill for NaN values
- StandardScaler normalization across all features
- Sliding window with stride 1

### 2. Transformer Model

**Method**: `prepare_training_from_corpus(data, sequence_length=192, prediction_horizons=[1,4,24])`

**Input Shape**:
- Training: `(n_samples, 192, 65)`
  - n_samples: Number of sequences
  - 192: Timesteps (longer context window)
  - 65: All numeric features

**Output Shape**:
- Predictions: `(n_samples, 3)` - [1h, 4h, 24h price changes as returns]

**Features Used**: ALL available numeric features

**Preprocessing**:
- StandardScaler normalization
- Returns calculated as (future_price - current_price) / current_price
- Sliding window with stride 1

### 3. iTransformer Model

**Method**: `prepare_training_from_corpus(data, sequence_length=96, prediction_horizons=[1,4,24], selected_features=None)`

**Input Shape**:
- Training: `(n_samples, 96, ~20)`
  - n_samples: Number of sequences
  - 96: Timesteps
  - ~20: Selected key variates (configurable)

**Default Selected Features** (treats each as separate variate):
- Core: close, volume, open, high, low
- RSI: rsi_14, rsi_7, rsi_21
- MACD: macd, macd_signal
- Bollinger: bb_upper, bb_lower, bb_position
- Momentum: momentum_5, momentum_10, momentum_20
- Scores: volatility_score, volume_score
- Market: market_regime, trend_strength

**Output Shape**:
- Predictions: `(n_samples, 3)` - [1h, 4h, 24h price changes]

**Preprocessing**:
- Each variate normalized independently
- Inverted attention mechanism treats features as separate channels

### 4. PatchTST Model

**Method**: `prepare_training_from_corpus(data, sequence_length=patch_length*4, prediction_horizons=[1,4,24], n_channels=None)`

**Input Shape**:
- Training: `(n_samples, patch_length*4, n_channels)`
  - n_samples: Number of sequences
  - patch_length*4: Typically 64 (if patch_length=16)
  - n_channels: Configurable, typically 10-20 channels

**Priority Features for Patches**:
1. OHLCV (5 features)
2. Multi-scale returns (returns_1h, returns_24h, returns_7d)
3. RSI variations (rsi_14, rsi_7, rsi_21)
4. MACD components
5. Bollinger bands
6. Volatility measures (atr, adx)
7. Momentum indicators
8. ML scores

**Output Shape**:
- Predictions: `(n_samples, 3)` - [1h, 4h, 24h price changes]

**Preprocessing**:
- StandardScaler normalization
- Patches created from continuous sequences
- Zero padding if sequence length insufficient

### 5. TimesMixer Model

**Method**: `prepare_training_from_corpus(data, sequence_length=336, prediction_horizons=[1,4,24], n_features=None)`

**Input Shape**:
- Training: `(n_samples, 336, n_features)`
  - n_samples: Number of sequences
  - 336: Two weeks of hourly data for seasonal patterns
  - n_features: Configurable, typically 30-40 features

**Priority Features for Temporal Decomposition**:
1. OHLCV (5 features)
2. Seasonal/cyclical: hour, day_of_week, month, hour_sin, hour_cos, day_sin, day_cos
3. Multi-scale returns: returns_1h, returns_24h, returns_7d
4. Trend indicators: Various EMAs and SMAs
5. Oscillators: RSI variations
6. Volatility at different scales
7. MACD components
8. Momentum features
9. Market microstructure
10. ML scores and regime

**Output Shape**:
- Predictions: `(n_samples, 3)` - [1h, 4h, 24h price changes]

**Preprocessing**:
- Each feature normalized independently for decomposition
- Long sequences for capturing seasonal patterns
- Temporal mixing across different time scales

## Data Flow

```
1. Raw Data Collection (APIs)
   ↓
2. FeatureEngineer.calculate_features_for_corpus()
   - Creates all 65+ features
   - Returns Dict[str, float]
   ↓
3. MultiGranularityCollector.collect_multi_granularity_corpus()
   - Collects at multiple timeframes
   - Stores in database (PostgreSQL)
   - Exports to Parquet files
   ↓
4. Model Training
   - Load parquet file as DataFrame
   - Call model.prepare_training_from_corpus(df)
   - Returns (X, y, feature_names)
   ↓
5. Model.train()
   - Uses prepared sequences
   - Trains on all available features
```

## Usage Example

```python
import pandas as pd
from src.ml_analysis.lstm_model import LSTMModel
from src.ml_analysis.transformers.transformer_predictor import TransformerPredictor

# Load corpus data
corpus_df = pd.read_parquet('data/corpus/v2.0/hourly/weth_hourly.parquet')

# LSTM Training
lstm_model = LSTMModel()
X, y, features = lstm_model.prepare_training_from_corpus(
    corpus_df,
    sequence_length=50,
    prediction_horizons=[1, 4, 24]
)
# X shape: (n_samples, 50, 65)

# Transformer Training  
transformer = TransformerPredictor()
X, y, features = transformer.prepare_training_from_corpus(
    corpus_df,
    sequence_length=192,
    prediction_horizons=[1, 4, 24]
)
# X shape: (n_samples, 192, 65)

# iTransformer with custom features
from src.ml_analysis.transformers.itransformer import iTransformerPredictor

itrans = iTransformerPredictor()
X, y, features = itrans.prepare_training_from_corpus(
    corpus_df,
    selected_features=['close', 'volume', 'rsi_14', 'momentum_5']
)
# X shape: (n_samples, 96, 4)
```

## Important Notes

1. **Feature Availability**: Not all features may be available for all tokens or timeframes
2. **NaN Handling**: Models use forward-fill, backward-fill, then zero-fill for missing values
3. **Normalization**: Each model handles normalization differently based on architecture needs
4. **Sequence Overlap**: All models use sliding windows with stride 1 for maximum training data
5. **Target Calculation**: All models predict price changes (returns) rather than absolute prices
6. **Memory Considerations**: Longer sequences (Transformer, TimesMixer) require more memory

## Feature Engineering Methods

Key methods in `src/ml_analysis/feature_engineer.py`:

- `calculate_features_for_corpus()`: Main entry point for corpus feature calculation
- `calculate_technical_indicators()`: RSI, MACD, Bollinger Bands, etc.
- `calculate_momentum_indicators()`: Momentum at various periods
- `calculate_volatility_features()`: Various volatility metrics
- `calculate_time_features()`: Hour, day, cyclical encodings
- `calculate_ml_scores()`: Volatility/volume scores, market regime
- `extract_stablecoin_features()`: Special features for stablecoins

## Model Configuration Defaults

| Model | Sequence Length | Features Used | Prediction Horizons |
|-------|----------------|---------------|-------------------|
| LSTM | 50 | All 65 | [1h, 4h, 24h] |
| Transformer | 192 | All 65 | [1h, 4h, 24h] |
| iTransformer | 96 | ~20 selected | [1h, 4h, 24h] |
| PatchTST | 64 | ~20 priority | [1h, 4h, 24h] |
| TimesMixer | 336 | ~40 temporal | [1h, 4h, 24h] |