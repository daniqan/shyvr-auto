"""
Model-specific data exporter for training-ready sequences

Exports corpus data in formats optimized for different model architectures:
- LSTM: Fixed sequence length (e.g., 50 timesteps)
- Transformers: Variable/longer sequences (e.g., 192 timesteps)
- Different feature subsets per model type
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import structlog
from datetime import datetime, timezone

logger = structlog.get_logger(__name__)


class ModelSpecificExporter:
    """Export corpus data in model-specific formats for training"""
    
    # Model-specific configuration
    MODEL_CONFIGS = {
        'lstm': {
            'sequence_length': 50,
            'prediction_horizons': [1, 24],  # 1h and 24h predictions
            'feature_subset': None,  # Use all features
            'min_samples': 100,
            'overlap_ratio': 0.5,  # 50% overlap for sequences
        },
        'transformer': {
            'sequence_length': 192,  # Longer context window
            'prediction_horizons': [1, 24, 168],  # 1h, 24h, 7d
            'feature_subset': None,  # Use all features
            'min_samples': 10,
            'overlap_ratio': 0.8,  # 80% overlap for more training data
        },
        'patchtst': {
            'sequence_length': 512,  # Even longer for patch-based
            'patch_length': 16,
            'prediction_horizons': [24, 168],  # 24h, 7d
            'feature_subset': None,
            'min_samples': 20,
            'overlap_ratio': 0.9,  # High overlap for patches
        },
        'itransformer': {
            'sequence_length': 96,  # Moderate length for inverted attention
            'prediction_horizons': [24],  # Focus on 24h
            'feature_subset': ['open', 'high', 'low', 'close', 'volume', 
                              'rsi_14', 'rsi_7', 'momentum_5', 'volatility_score'],  # Key variates
            'min_samples': 10,
            'overlap_ratio': 0.7,
        },
        'timesmixer': {
            'sequence_length': 336,  # 2 weeks for seasonal patterns
            'prediction_horizons': [24, 168],
            'feature_subset': None,
            'min_samples': 20,
            'overlap_ratio': 0.85,
        }
    }
    
    def __init__(self):
        """Initialize the model-specific exporter"""
        self.export_stats = {}
    
    def create_sequences(self, 
                        data: pd.DataFrame, 
                        sequence_length: int,
                        prediction_horizons: List[int],
                        overlap_ratio: float = 0.5,
                        feature_subset: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Create sequences for training from time series data
        
        Args:
            data: DataFrame with time series data
            sequence_length: Length of input sequences
            prediction_horizons: List of prediction horizons (e.g., [1, 24] for 1h and 24h)
            overlap_ratio: Overlap between sequences (0.5 = 50% overlap)
            feature_subset: Optional list of features to use (None = all)
            
        Returns:
            Dictionary with 'sequences', 'targets_{horizon}', and metadata
        """
        # Select features
        if feature_subset:
            # Ensure we have the required features
            available_features = [f for f in feature_subset if f in data.columns]
            if len(available_features) < len(feature_subset):
                missing = set(feature_subset) - set(available_features)
                logger.warning(f"Missing features: {missing}, using available: {available_features}")
            feature_data = data[available_features].values
            feature_names = available_features
        else:
            # Use all numeric columns except metadata
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['id', 'ohlcv_id', 'token_id']
            feature_cols = [c for c in numeric_cols if c not in exclude_cols]
            feature_data = data[feature_cols].values
            feature_names = feature_cols
        
        # Calculate stride based on overlap
        stride = max(1, int(sequence_length * (1 - overlap_ratio)))
        
        sequences = []
        targets = {f'targets_{h}h': [] for h in prediction_horizons}
        timestamps = []
        
        # Create sequences
        max_horizon = max(prediction_horizons)
        for i in range(0, len(feature_data) - sequence_length - max_horizon, stride):
            # Input sequence
            seq = feature_data[i:i + sequence_length]
            sequences.append(seq)
            
            # Timestamp for the sequence (end of sequence)
            if 'timestamp' in data.columns:
                timestamps.append(data.iloc[i + sequence_length - 1]['timestamp'])
            
            # Targets at different horizons (using close price)
            close_idx = feature_names.index('close') if 'close' in feature_names else 0
            base_idx = i + sequence_length
            
            for horizon in prediction_horizons:
                if base_idx + horizon < len(data):
                    # Price change as target
                    current_price = feature_data[base_idx - 1, close_idx]
                    future_price = feature_data[base_idx + horizon - 1, close_idx]
                    price_change = (future_price - current_price) / current_price if current_price > 0 else 0
                    targets[f'targets_{horizon}h'].append(price_change)
                else:
                    targets[f'targets_{horizon}h'].append(0.0)
        
        # Convert to numpy arrays
        result = {
            'sequences': np.array(sequences, dtype=np.float32),
            'feature_names': feature_names,
            'sequence_length': sequence_length,
            'n_features': len(feature_names),
            'n_samples': len(sequences),
        }
        
        # Add targets
        for key, values in targets.items():
            result[key] = np.array(values, dtype=np.float32)
        
        # Add timestamps if available
        if timestamps:
            result['timestamps'] = timestamps
        
        return result
    
    def export_for_model(self,
                        corpus_data: Dict[str, pd.DataFrame],
                        model_type: str,
                        output_dir: Path,
                        compress: bool = True) -> Dict[str, Path]:
        """
        Export corpus data in format optimized for specific model type
        
        Args:
            corpus_data: Dictionary of DataFrames from corpus collection
            model_type: Type of model ('lstm', 'transformer', 'patchtst', etc.)
            output_dir: Directory to save exported files
            compress: Whether to use compression
            
        Returns:
            Dictionary mapping dataset keys to exported file paths
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[model_type]
        model_dir = output_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        total_sequences = 0
        
        for key, df in corpus_data.items():
            logger.info(f"Exporting {key} for {model_type} model")
            
            # Skip if insufficient data
            if len(df) < config['sequence_length'] + max(config['prediction_horizons']):
                logger.warning(f"Skipping {key}: insufficient data "
                             f"({len(df)} rows < {config['sequence_length'] + max(config['prediction_horizons'])} required)")
                continue
            
            # Create sequences
            sequence_data = self.create_sequences(
                data=df,
                sequence_length=config['sequence_length'],
                prediction_horizons=config['prediction_horizons'],
                overlap_ratio=config['overlap_ratio'],
                feature_subset=config.get('feature_subset')
            )
            
            # Skip if too few sequences
            if sequence_data['n_samples'] < config['min_samples']:
                logger.warning(f"Skipping {key}: too few sequences "
                             f"({sequence_data['n_samples']} < {config['min_samples']} required)")
                continue
            
            # Create DataFrame for export more efficiently
            # Start with metadata columns
            # Round created_at to milliseconds for parquet compatibility
            created_at = datetime.now(timezone.utc)
            created_at = created_at.replace(microsecond=(created_at.microsecond // 1000) * 1000)
            
            metadata_dict = {
                'sequence_id': range(sequence_data['n_samples']),
                'sequence_length': [config['sequence_length']] * sequence_data['n_samples'],
                'n_features': [sequence_data['n_features']] * sequence_data['n_samples'],
                'token': [key.split('_')[0]] * sequence_data['n_samples'],
                'granularity': ['_'.join(key.split('_')[1:])] * sequence_data['n_samples'],
                'model_type': [model_type] * sequence_data['n_samples'],
                'created_at': [created_at] * sequence_data['n_samples'],
            }
            
            # Add targets
            for horizon in config['prediction_horizons']:
                target_key = f'targets_{horizon}h'
                if target_key in sequence_data:
                    metadata_dict[f'target_{horizon}h'] = sequence_data[target_key]
            
            # Add timestamps if available (convert to ms precision for parquet)
            if 'timestamps' in sequence_data:
                # Convert timestamps to millisecond precision to avoid pyarrow issue
                timestamps = sequence_data['timestamps']
                if timestamps and hasattr(timestamps[0], 'replace'):
                    # Round to milliseconds
                    timestamps = [ts.replace(microsecond=(ts.microsecond // 1000) * 1000) 
                                 for ts in timestamps]
                metadata_dict['timestamp'] = timestamps
            
            # Create sequence columns dictionary for efficient concatenation
            sequences = sequence_data['sequences']
            sequence_dict = {}
            for t in range(config['sequence_length']):
                for f, feat_name in enumerate(sequence_data['feature_names']):
                    col_name = f'seq_t{t:03d}_f{f:03d}_{feat_name}'
                    sequence_dict[col_name] = sequences[:, t, f]
            
            # Combine all dictionaries efficiently
            export_df = pd.DataFrame(metadata_dict)
            sequence_df = pd.DataFrame(sequence_dict)
            export_df = pd.concat([export_df, sequence_df], axis=1)
            
            # Save to parquet
            filename = f"{key}_{model_type}_sequences.parquet"
            filepath = model_dir / filename
            
            # Write with appropriate compression
            table = pa.Table.from_pandas(export_df)
            compression = 'snappy' if compress else None
            pq.write_table(
                table,
                filepath,
                compression=compression,
                use_dictionary=True,
                coerce_timestamps='ms'
            )
            
            exported_files[key] = filepath
            total_sequences += sequence_data['n_samples']
            
            logger.info(f"Exported {key} for {model_type}",
                       path=str(filepath),
                       sequences=sequence_data['n_samples'],
                       features=sequence_data['n_features'],
                       size_mb=filepath.stat().st_size / 1024 / 1024)
        
        # Store export statistics
        self.export_stats[model_type] = {
            'total_sequences': total_sequences,
            'files_exported': len(exported_files),
            'timestamp': datetime.now(timezone.utc)
        }
        
        return exported_files
    
    def export_all_models(self,
                         corpus_data: Dict[str, pd.DataFrame],
                         output_dir: Path) -> Dict[str, Dict[str, Path]]:
        """
        Export corpus data for all supported model types
        
        Args:
            corpus_data: Dictionary of DataFrames from corpus collection
            output_dir: Base directory for exports
            
        Returns:
            Nested dictionary: {model_type: {dataset_key: filepath}}
        """
        all_exports = {}
        
        for model_type in self.MODEL_CONFIGS.keys():
            logger.info(f"Exporting for {model_type} model")
            exports = self.export_for_model(corpus_data, model_type, output_dir)
            all_exports[model_type] = exports
        
        # Write summary metadata
        self._write_export_metadata(output_dir)
        
        return all_exports
    
    def _write_export_metadata(self, output_dir: Path):
        """Write metadata about the export process"""
        metadata = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'model_configs': self.MODEL_CONFIGS,
            'export_stats': self.export_stats
        }
        
        metadata_path = output_dir / 'export_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Export metadata written to {metadata_path}")
    
    def load_sequences(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        """
        Load sequences from exported parquet file back into training format
        
        Args:
            filepath: Path to the parquet file
            
        Returns:
            Tuple of (sequences, targets_dict, feature_names)
        """
        df = pd.read_parquet(filepath)
        
        # Extract metadata
        sequence_length = df['sequence_length'].iloc[0]
        n_features = df['n_features'].iloc[0]
        n_samples = len(df)
        
        # Reconstruct sequences from flattened columns
        seq_cols = [c for c in df.columns if c.startswith('seq_t')]
        
        # Extract feature names from column names
        feature_names = []
        for col in seq_cols[:n_features]:
            # Format: seq_t000_f000_feature_name
            parts = col.split('_', 3)
            if len(parts) >= 4:
                feature_names.append(parts[3])
        
        # Reconstruct 3D array
        sequences = np.zeros((n_samples, sequence_length, n_features), dtype=np.float32)
        for col in seq_cols:
            # Parse column name
            parts = col.split('_')
            t = int(parts[1][1:])  # Remove 't' prefix
            f = int(parts[2][1:])  # Remove 'f' prefix
            sequences[:, t, f] = df[col].values
        
        # Extract targets
        targets = {}
        target_cols = [c for c in df.columns if c.startswith('target_')]
        for col in target_cols:
            targets[col] = df[col].values
        
        logger.info(f"Loaded sequences from {filepath}",
                   shape=sequences.shape,
                   targets=list(targets.keys()))
        
        return sequences, targets, feature_names