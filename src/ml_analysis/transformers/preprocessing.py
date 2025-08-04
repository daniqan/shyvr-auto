"""
Transformer Preprocessing Module
Utilities for preparing time-series data for Transformer models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import structlog
from dataclasses import dataclass

from src.discovery.base import DiscoveredToken


logger = structlog.get_logger()


@dataclass
class TransformerPreprocessorConfig:
    """Configuration for Transformer preprocessing"""
    sequence_length: int = 60
    patch_length: int = 16
    patch_stride: int = 8
    prediction_horizons: List[int] = None
    max_sequence_length: int = 1000
    padding_value: float = 0.0
    mask_probability: float = 0.15
    augmentation_probability: float = 0.1
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 4, 24]  # 1h, 4h, 24h


class TransformerPreprocessor:
    """Preprocessing utilities for Transformer models"""
    
    def __init__(self, config: TransformerPreprocessorConfig = None):
        self.config = config or TransformerPreprocessorConfig()
        self.logger = structlog.get_logger().bind(component="TransformerPreprocessor")
    
    def create_patches(self, sequences: np.ndarray, patch_length: int = None, stride: int = None) -> np.ndarray:
        """
        Create patches for PatchTST model
        
        Args:
            sequences: Input sequences [batch, seq_len, features]
            patch_length: Length of each patch
            stride: Stride between patches
            
        Returns:
            Patched sequences [batch, num_patches, patch_length, features]
        """
        try:
            patch_length = patch_length or self.config.patch_length
            stride = stride or self.config.patch_stride
            
            if len(sequences.shape) != 3:
                raise ValueError(f"Expected 3D input [batch, seq_len, features], got {sequences.shape}")
            
            batch_size, seq_len, num_features = sequences.shape
            
            if seq_len < patch_length:
                self.logger.warning("Sequence length shorter than patch length", 
                                  seq_len=seq_len, patch_length=patch_length)
                # Pad sequence to minimum patch length
                pad_length = patch_length - seq_len
                sequences = np.pad(sequences, ((0, 0), (pad_length, 0), (0, 0)), 
                                 mode='constant', constant_values=self.config.padding_value)
                seq_len = patch_length
            
            # Calculate number of patches
            num_patches = max(1, (seq_len - patch_length) // stride + 1)
            
            # Create patches
            patches = np.zeros((batch_size, num_patches, patch_length, num_features))
            
            for i in range(num_patches):
                start_idx = i * stride
                end_idx = start_idx + patch_length
                if end_idx <= seq_len:
                    patches[:, i, :, :] = sequences[:, start_idx:end_idx, :]
                else:
                    # Handle last patch if it extends beyond sequence
                    remaining = seq_len - start_idx
                    patches[:, i, :remaining, :] = sequences[:, start_idx:, :]
                    # Pad the rest
                    patches[:, i, remaining:, :] = self.config.padding_value
            
            self.logger.info("Patches created", 
                           input_shape=sequences.shape,
                           output_shape=patches.shape,
                           num_patches=num_patches)
            
            return patches
            
        except Exception as e:
            self.logger.error("Failed to create patches", error=str(e))
            raise
    
    def pad_sequences(self, sequences: List[np.ndarray], max_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad sequences to uniform length and create attention masks
        
        Args:
            sequences: List of sequences with varying lengths
            max_length: Maximum sequence length (uses config default if None)
            
        Returns:
            Tuple of (padded_sequences, attention_masks)
        """
        try:
            if not sequences:
                raise ValueError("Empty sequence list provided")
            
            max_length = max_length or self.config.max_sequence_length
            
            # Get dimensions
            seq_lengths = [seq.shape[0] for seq in sequences]
            num_features = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
            
            # Determine actual max length to use
            actual_max_length = min(max_length, max(seq_lengths))
            
            # Initialize padded arrays
            padded_sequences = np.full(
                (len(sequences), actual_max_length, num_features), 
                self.config.padding_value, 
                dtype=np.float32
            )
            attention_masks = np.zeros((len(sequences), actual_max_length), dtype=bool)
            
            for i, seq in enumerate(sequences):
                seq_len = min(seq.shape[0], actual_max_length)
                
                if len(seq.shape) == 1:
                    padded_sequences[i, :seq_len, 0] = seq[:seq_len]
                else:
                    padded_sequences[i, :seq_len, :] = seq[:seq_len]
                
                # Create attention mask (True for real tokens, False for padding)
                attention_masks[i, :seq_len] = True
            
            self.logger.info("Sequences padded", 
                           num_sequences=len(sequences),
                           original_lengths=seq_lengths,
                           padded_length=actual_max_length)
            
            return padded_sequences, attention_masks
            
        except Exception as e:
            self.logger.error("Failed to pad sequences", error=str(e))
            raise
    
    def create_attention_mask(self, sequences: np.ndarray, mask_type: str = "causal") -> np.ndarray:
        """
        Create attention masks for different attention patterns
        
        Args:
            sequences: Input sequences [batch, seq_len, features]
            mask_type: Type of mask ('causal', 'bidirectional', 'random')
            
        Returns:
            Attention mask [batch, seq_len, seq_len]
        """
        try:
            batch_size, seq_len, _ = sequences.shape
            
            if mask_type == "causal":
                # Lower triangular mask for causal attention
                mask = np.tril(np.ones((seq_len, seq_len)))
                mask = np.tile(mask[None, :, :], (batch_size, 1, 1))
                
            elif mask_type == "bidirectional":
                # All-ones mask for bidirectional attention
                mask = np.ones((batch_size, seq_len, seq_len))
                
            elif mask_type == "random":
                # Random masking for robustness
                mask = np.random.rand(batch_size, seq_len, seq_len) > self.config.mask_probability
                mask = mask.astype(float)
                
            else:
                raise ValueError(f"Unknown mask type: {mask_type}")
            
            self.logger.info("Attention mask created", 
                           mask_type=mask_type,
                           shape=mask.shape)
            
            return mask
            
        except Exception as e:
            self.logger.error("Failed to create attention mask", error=str(e))
            raise
    
    def prepare_multi_horizon_targets(self, price_data: pd.DataFrame, 
                                    prediction_horizons: List[int] = None) -> np.ndarray:
        """
        Prepare multi-horizon prediction targets
        
        Args:
            price_data: DataFrame with price data
            prediction_horizons: List of horizons in hours
            
        Returns:
            Target array [seq_len, num_horizons]
        """
        try:
            prediction_horizons = prediction_horizons or self.config.prediction_horizons
            
            if 'close' not in price_data.columns:
                raise ValueError("Price data must contain 'close' column")
            
            prices = price_data['close'].values
            targets = np.zeros((len(prices), len(prediction_horizons)))
            
            for h_idx, horizon in enumerate(prediction_horizons):
                for i in range(len(prices) - horizon):
                    current_price = prices[i]
                    future_price = prices[i + horizon]
                    
                    # Calculate percentage change
                    if current_price > 0:
                        price_change = (future_price - current_price) / current_price
                        targets[i, h_idx] = price_change
            
            self.logger.info("Multi-horizon targets prepared", 
                           horizons=prediction_horizons,
                           target_shape=targets.shape)
            
            return targets
            
        except Exception as e:
            self.logger.error("Failed to prepare multi-horizon targets", error=str(e))
            raise
    
    def apply_transformer_normalization(self, features: np.ndarray, method: str = "layer_norm") -> np.ndarray:
        """
        Apply attention-friendly normalization to features
        
        Args:
            features: Feature array [batch, seq_len, features] or [batch, features]
            method: Normalization method ('layer_norm', 'batch_norm', 'robust_scale')
            
        Returns:
            Normalized features
        """
        try:
            if method == "layer_norm":
                # Layer normalization across feature dimension
                if len(features.shape) == 3:
                    # For sequences: normalize across last dimension
                    mean = np.mean(features, axis=-1, keepdims=True)
                    std = np.std(features, axis=-1, keepdims=True)
                else:
                    # For feature vectors: normalize across features
                    mean = np.mean(features, axis=-1, keepdims=True)
                    std = np.std(features, axis=-1, keepdims=True)
                
                normalized = (features - mean) / (std + 1e-8)
                
            elif method == "batch_norm":
                # Batch normalization across batch dimension
                mean = np.mean(features, axis=0, keepdims=True)
                std = np.std(features, axis=0, keepdims=True)
                normalized = (features - mean) / (std + 1e-8)
                
            elif method == "robust_scale":
                # Robust scaling using median and IQR
                if len(features.shape) == 3:
                    median = np.median(features, axis=-1, keepdims=True)
                    q75 = np.percentile(features, 75, axis=-1, keepdims=True)
                    q25 = np.percentile(features, 25, axis=-1, keepdims=True)
                else:
                    median = np.median(features, axis=-1, keepdims=True)
                    q75 = np.percentile(features, 75, axis=-1, keepdims=True)
                    q25 = np.percentile(features, 25, axis=-1, keepdims=True)
                
                iqr = q75 - q25
                normalized = (features - median) / (iqr + 1e-8)
                
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Clip extreme values for attention stability
            normalized = np.clip(normalized, -10.0, 10.0)
            
            self.logger.info("Transformer normalization applied", 
                           method=method,
                           input_shape=features.shape,
                           output_range=(float(normalized.min()), float(normalized.max())))
            
            return normalized
            
        except Exception as e:
            self.logger.error("Failed to apply transformer normalization", error=str(e))
            raise
    
    def augment_time_series(self, sequences: np.ndarray, augmentation_type: str = "noise") -> np.ndarray:
        """
        Apply data augmentation techniques for time-series
        
        Args:
            sequences: Input sequences [batch, seq_len, features]
            augmentation_type: Type of augmentation ('noise', 'scale', 'shift', 'mask')
            
        Returns:
            Augmented sequences
        """
        try:
            if np.random.rand() > self.config.augmentation_probability:
                return sequences  # No augmentation
            
            augmented = sequences.copy()
            
            if augmentation_type == "noise":
                # Add Gaussian noise
                noise_std = np.std(sequences) * 0.01  # 1% of signal std
                noise = np.random.normal(0, noise_std, sequences.shape)
                augmented += noise
                
            elif augmentation_type == "scale":
                # Random scaling
                scale_factor = np.random.uniform(0.95, 1.05)
                augmented *= scale_factor
                
            elif augmentation_type == "shift":
                # Random time shift (circular)
                shift_amount = np.random.randint(-5, 6)  # Shift by up to 5 positions
                if shift_amount != 0:
                    augmented = np.roll(augmented, shift_amount, axis=1)
                    
            elif augmentation_type == "mask":
                # Random masking of time steps
                mask_prob = 0.1
                mask = np.random.rand(*sequences.shape[:2]) > mask_prob
                augmented = augmented * mask[:, :, np.newaxis]
                
            else:
                raise ValueError(f"Unknown augmentation type: {augmentation_type}")
            
            self.logger.info("Time-series augmentation applied", 
                           augmentation_type=augmentation_type,
                           shape=augmented.shape)
            
            return augmented
            
        except Exception as e:
            self.logger.error("Failed to augment time-series", error=str(e))
            return sequences  # Return original on error
    
    def create_multivariate_features(self, tokens: List[DiscoveredToken], 
                                   price_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Create aligned multivariate features for cross-asset Transformer models
        
        Args:
            tokens: List of tokens
            price_data: Dictionary mapping token addresses to price DataFrames
            
        Returns:
            Aligned feature matrix [seq_len, num_assets, num_features]
        """
        try:
            if not tokens or not price_data:
                raise ValueError("Empty tokens or price data provided")
            
            # Collect all DataFrames with proper alignment
            aligned_data = {}
            min_length = float('inf')
            
            for token in tokens:
                if token.address not in price_data:
                    continue
                
                df = price_data[token.address].copy()
                
                # Ensure datetime index
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                # Extract features
                features = []
                feature_names = []
                
                for col in ['close', 'volume', 'high', 'low', 'open']:
                    if col in df.columns:
                        features.append(df[col].values)
                        feature_names.append(f"{token.symbol}_{col}")
                
                if features:
                    aligned_data[token.symbol or token.address] = np.column_stack(features)
                    min_length = min(min_length, len(features[0]))
            
            if not aligned_data:
                raise ValueError("No valid data found for any tokens")
            
            # Truncate all series to minimum length for alignment
            num_assets = len(aligned_data)
            num_features_per_asset = list(aligned_data.values())[0].shape[1]
            
            multivariate_features = np.zeros((min_length, num_assets, num_features_per_asset))
            
            for i, (symbol, data) in enumerate(aligned_data.items()):
                multivariate_features[:, i, :] = data[:min_length]
            
            self.logger.info("Multivariate features created", 
                           shape=multivariate_features.shape,
                           assets=list(aligned_data.keys()),
                           features_per_asset=num_features_per_asset)
            
            return multivariate_features
            
        except Exception as e:
            self.logger.error("Failed to create multivariate features", error=str(e))
            raise
    
    def create_sliding_windows(self, data: np.ndarray, 
                             window_size: int = None, 
                             step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows for sequence-to-sequence learning
        
        Args:
            data: Input data [seq_len, features]
            window_size: Size of each window
            step_size: Step size between windows
            
        Returns:
            Tuple of (input_windows, target_windows)
        """
        try:
            window_size = window_size or self.config.sequence_length
            
            if len(data) < window_size + 1:
                raise ValueError(f"Data length {len(data)} too short for window size {window_size}")
            
            num_windows = (len(data) - window_size) // step_size
            num_features = data.shape[1] if len(data.shape) > 1 else 1
            
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            input_windows = np.zeros((num_windows, window_size, num_features))
            target_windows = np.zeros((num_windows, num_features))
            
            for i in range(num_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                
                input_windows[i] = data[start_idx:end_idx]
                target_windows[i] = data[end_idx]  # Next time step as target
            
            self.logger.info("Sliding windows created", 
                           input_shape=input_windows.shape,
                           target_shape=target_windows.shape,
                           num_windows=num_windows)
            
            return input_windows, target_windows
            
        except Exception as e:
            self.logger.error("Failed to create sliding windows", error=str(e))
            raise


def create_patches(sequences: np.ndarray, patch_length: int = 16, stride: int = 8) -> np.ndarray:
    """Convenience function for creating patches"""
    preprocessor = TransformerPreprocessor()
    return preprocessor.create_patches(sequences, patch_length, stride)


def pad_sequences(sequences: List[np.ndarray], max_length: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for padding sequences"""
    preprocessor = TransformerPreprocessor()
    return preprocessor.pad_sequences(sequences, max_length)


def create_attention_mask(sequences: np.ndarray, mask_type: str = "causal") -> np.ndarray:
    """Convenience function for creating attention masks"""
    preprocessor = TransformerPreprocessor()
    return preprocessor.create_attention_mask(sequences, mask_type)


def prepare_multi_horizon_targets(price_data: pd.DataFrame, 
                                prediction_horizons: List[int] = None) -> np.ndarray:
    """Convenience function for preparing multi-horizon targets"""
    preprocessor = TransformerPreprocessor()
    return preprocessor.prepare_multi_horizon_targets(price_data, prediction_horizons)


def augment_time_series(sequences: np.ndarray, augmentation_type: str = "noise") -> np.ndarray:
    """Convenience function for time-series augmentation"""
    preprocessor = TransformerPreprocessor()
    return preprocessor.augment_time_series(sequences, augmentation_type)