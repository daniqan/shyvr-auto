"""
Token-specific normalization for cryptocurrency price data.

Handles different price scales across tokens (e.g., BTC ~$60K vs PEPE ~$0.00001)
by maintaining separate scalers for each token.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
from pathlib import Path
import logging
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class TokenNormalizationConfig:
    """Configuration for token-specific normalization"""
    
    # Normalization method per feature type
    price_normalizer: str = "robust"  # robust, standard, minmax, log
    volume_normalizer: str = "log_robust"  # log transform + robust scaler
    indicator_normalizer: str = "standard"  # for technical indicators
    
    # Token-specific price ranges (for validation)
    expected_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'BTC': (10000, 100000),
        'ETH': (500, 10000),
        'WBTC': (10000, 100000),
        'SOL': (1, 500),
        'PEPE': (0.000001, 0.001),
        'SHIB': (0.000001, 0.001),
        'DOGE': (0.01, 1.0),
    })
    
    # Feature groups for different normalization strategies
    price_features: List[str] = field(default_factory=lambda: [
        'close', 'open', 'high', 'low', 'vwap', 'twap'
    ])
    
    volume_features: List[str] = field(default_factory=lambda: [
        'volume', 'volume_24h', 'volume_7d', 'obv'
    ])
    
    indicator_features: List[str] = field(default_factory=lambda: [
        'rsi_14', 'rsi_7', 'rsi_21', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_position', 'atr', 'adx',
        'stoch_k', 'stoch_d', 'williams_r'
    ])
    
    # Features that should not be normalized
    skip_features: List[str] = field(default_factory=lambda: [
        'timestamp', 'token_symbol', 'chain', 'direction'
    ])


class TokenNormalizer:
    """
    Manages token-specific normalization for training data.
    
    Key features:
    - Separate scalers for each token
    - Different normalization strategies for price/volume/indicators
    - Persistent scaler storage for inference
    - Outlier detection and handling
    """
    
    def __init__(self, config: Optional[TokenNormalizationConfig] = None,
                 scaler_dir: Optional[Path] = None):
        """
        Initialize TokenNormalizer.
        
        Args:
            config: Normalization configuration
            scaler_dir: Directory to save/load fitted scalers
        """
        self.config = config or TokenNormalizationConfig()
        self.scaler_dir = Path(scaler_dir) if scaler_dir else Path("models/scalers")
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        
        # Store fitted scalers per token
        self.token_scalers: Dict[str, Dict[str, Any]] = {}
        self.is_fitted: Dict[str, bool] = {}
        
        logger.info(f"TokenNormalizer initialized with scaler_dir={self.scaler_dir}")
    
    def _create_scaler(self, method: str) -> Any:
        """Create appropriate scaler based on method."""
        if method == "standard":
            return StandardScaler()
        elif method == "robust":
            return RobustScaler(quantile_range=(5.0, 95.0))
        elif method == "minmax":
            return MinMaxScaler(feature_range=(-1, 1))
        elif method == "log_robust":
            # Will apply log transform then robust scaling
            return RobustScaler(quantile_range=(5.0, 95.0))
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _apply_log_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply log transform with handling for zeros/negatives."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        # Handle negative values by shifting
        min_val = np.min(data)
        if min_val <= 0:
            shift = abs(min_val) + epsilon
            data = data + shift
        
        return np.log1p(data)  # log(1 + x) for numerical stability
    
    def _inverse_log_transform(self, data: np.ndarray, original_min: float) -> np.ndarray:
        """Inverse log transform."""
        result = np.expm1(data)  # exp(x) - 1
        
        # Reverse shift if applied during transform
        if original_min <= 0:
            shift = abs(original_min) + 1e-8
            result = result - shift
        
        return result
    
    def fit(self, data: pd.DataFrame, token_symbol: str) -> 'TokenNormalizer':
        """
        Fit normalizers for a specific token.
        
        Args:
            data: Training data for the token
            token_symbol: Token identifier (e.g., 'BTC', 'ETH')
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting normalizers for token: {token_symbol}")
        
        if token_symbol not in self.token_scalers:
            self.token_scalers[token_symbol] = {
                'price_scaler': self._create_scaler(self.config.price_normalizer),
                'volume_scaler': self._create_scaler(self.config.volume_normalizer),
                'indicator_scaler': self._create_scaler(self.config.indicator_normalizer),
                'metadata': {}
            }
        
        scalers = self.token_scalers[token_symbol]
        
        # Fit price features
        price_cols = [col for col in self.config.price_features if col in data.columns]
        if price_cols:
            price_data = data[price_cols].values
            scalers['price_scaler'].fit(price_data)
            scalers['metadata']['price_features'] = price_cols
            
            # Store price range for validation
            scalers['metadata']['price_range'] = (
                float(np.min(price_data)),
                float(np.max(price_data))
            )
        
        # Fit volume features (with log transform if specified)
        volume_cols = [col for col in self.config.volume_features if col in data.columns]
        if volume_cols:
            volume_data = data[volume_cols].values
            
            if self.config.volume_normalizer.startswith("log"):
                scalers['metadata']['volume_min'] = float(np.min(volume_data))
                volume_data = self._apply_log_transform(volume_data)
            
            scalers['volume_scaler'].fit(volume_data)
            scalers['metadata']['volume_features'] = volume_cols
        
        # Fit indicator features
        indicator_cols = [col for col in self.config.indicator_features if col in data.columns]
        if indicator_cols:
            indicator_data = data[indicator_cols].values
            scalers['indicator_scaler'].fit(indicator_data)
            scalers['metadata']['indicator_features'] = indicator_cols
        
        self.is_fitted[token_symbol] = True
        
        # Validate price range if expected
        if token_symbol in self.config.expected_ranges:
            expected_min, expected_max = self.config.expected_ranges[token_symbol]
            actual_min, actual_max = scalers['metadata']['price_range']
            
            if actual_min < expected_min * 0.1 or actual_max > expected_max * 10:
                logger.warning(
                    f"Unusual price range for {token_symbol}: "
                    f"[{actual_min:.8f}, {actual_max:.8f}] vs expected "
                    f"[{expected_min:.8f}, {expected_max:.8f}]"
                )
        
        logger.info(f"Fitted normalizers for {token_symbol}: "
                   f"{len(price_cols)} price, {len(volume_cols)} volume, "
                   f"{len(indicator_cols)} indicator features")
        
        return self
    
    def transform(self, data: pd.DataFrame, token_symbol: str) -> pd.DataFrame:
        """
        Transform data using token-specific normalizers.
        
        Args:
            data: Data to transform
            token_symbol: Token identifier
        
        Returns:
            Normalized data
        """
        if token_symbol not in self.is_fitted or not self.is_fitted[token_symbol]:
            raise ValueError(f"Normalizer not fitted for token: {token_symbol}")
        
        result = data.copy()
        scalers = self.token_scalers[token_symbol]
        
        # Transform price features
        price_cols = scalers['metadata'].get('price_features', [])
        if price_cols:
            price_cols_present = [col for col in price_cols if col in result.columns]
            if price_cols_present:
                result[price_cols_present] = scalers['price_scaler'].transform(
                    result[price_cols_present].values
                )
        
        # Transform volume features
        volume_cols = scalers['metadata'].get('volume_features', [])
        if volume_cols:
            volume_cols_present = [col for col in volume_cols if col in result.columns]
            if volume_cols_present:
                volume_data = result[volume_cols_present].values
                
                if self.config.volume_normalizer.startswith("log"):
                    volume_data = self._apply_log_transform(volume_data)
                
                result[volume_cols_present] = scalers['volume_scaler'].transform(volume_data)
        
        # Transform indicator features
        indicator_cols = scalers['metadata'].get('indicator_features', [])
        if indicator_cols:
            indicator_cols_present = [col for col in indicator_cols if col in result.columns]
            if indicator_cols_present:
                result[indicator_cols_present] = scalers['indicator_scaler'].transform(
                    result[indicator_cols_present].values
                )
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame, token_symbol: str) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data
            token_symbol: Token identifier
        
        Returns:
            Data in original scale
        """
        if token_symbol not in self.is_fitted or not self.is_fitted[token_symbol]:
            raise ValueError(f"Normalizer not fitted for token: {token_symbol}")
        
        result = data.copy()
        scalers = self.token_scalers[token_symbol]
        
        # Inverse transform price features
        price_cols = scalers['metadata'].get('price_features', [])
        if price_cols:
            price_cols_present = [col for col in price_cols if col in result.columns]
            if price_cols_present:
                result[price_cols_present] = scalers['price_scaler'].inverse_transform(
                    result[price_cols_present].values
                )
        
        # Inverse transform volume features
        volume_cols = scalers['metadata'].get('volume_features', [])
        if volume_cols:
            volume_cols_present = [col for col in volume_cols if col in result.columns]
            if volume_cols_present:
                volume_data = scalers['volume_scaler'].inverse_transform(
                    result[volume_cols_present].values
                )
                
                if self.config.volume_normalizer.startswith("log"):
                    original_min = scalers['metadata'].get('volume_min', 0)
                    volume_data = self._inverse_log_transform(volume_data, original_min)
                
                result[volume_cols_present] = volume_data
        
        # Inverse transform indicator features
        indicator_cols = scalers['metadata'].get('indicator_features', [])
        if indicator_cols:
            indicator_cols_present = [col for col in indicator_cols if col in result.columns]
            if indicator_cols_present:
                result[indicator_cols_present] = scalers['indicator_scaler'].inverse_transform(
                    result[indicator_cols_present].values
                )
        
        return result
    
    def fit_transform(self, data: pd.DataFrame, token_symbol: str) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data, token_symbol).transform(data, token_symbol)
    
    def save_scalers(self, token_symbol: Optional[str] = None) -> None:
        """
        Save fitted scalers to disk.
        
        Args:
            token_symbol: Specific token to save, or None for all
        """
        tokens_to_save = [token_symbol] if token_symbol else list(self.token_scalers.keys())
        
        for token in tokens_to_save:
            if token not in self.token_scalers:
                continue
            
            token_dir = self.scaler_dir / token
            token_dir.mkdir(parents=True, exist_ok=True)
            
            scalers = self.token_scalers[token]
            
            # Save individual scalers
            joblib.dump(scalers['price_scaler'], token_dir / 'price_scaler.pkl')
            joblib.dump(scalers['volume_scaler'], token_dir / 'volume_scaler.pkl')
            joblib.dump(scalers['indicator_scaler'], token_dir / 'indicator_scaler.pkl')
            
            # Save metadata
            with open(token_dir / 'metadata.json', 'w') as f:
                json.dump(scalers['metadata'], f, indent=2)
            
            logger.info(f"Saved scalers for {token} to {token_dir}")
    
    def load_scalers(self, token_symbol: str) -> None:
        """
        Load fitted scalers from disk.
        
        Args:
            token_symbol: Token to load scalers for
        """
        token_dir = self.scaler_dir / token_symbol
        
        if not token_dir.exists():
            raise FileNotFoundError(f"No saved scalers found for {token_symbol}")
        
        self.token_scalers[token_symbol] = {
            'price_scaler': joblib.load(token_dir / 'price_scaler.pkl'),
            'volume_scaler': joblib.load(token_dir / 'volume_scaler.pkl'),
            'indicator_scaler': joblib.load(token_dir / 'indicator_scaler.pkl'),
            'metadata': {}
        }
        
        # Load metadata
        metadata_path = token_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.token_scalers[token_symbol]['metadata'] = json.load(f)
        
        self.is_fitted[token_symbol] = True
        logger.info(f"Loaded scalers for {token_symbol} from {token_dir}")
    
    def normalize_mixed_tokens(self, data: pd.DataFrame, 
                              token_column: str = 'token_symbol') -> pd.DataFrame:
        """
        Normalize data containing multiple tokens.
        
        Args:
            data: DataFrame with multiple tokens
            token_column: Column identifying the token
        
        Returns:
            Normalized DataFrame
        """
        if token_column not in data.columns:
            raise ValueError(f"Token column '{token_column}' not found in data")
        
        result_dfs = []
        
        for token in data[token_column].unique():
            token_data = data[data[token_column] == token].copy()
            
            if token in self.is_fitted and self.is_fitted[token]:
                token_data = self.transform(token_data, token)
            else:
                logger.warning(f"No fitted scaler for {token}, skipping normalization")
            
            result_dfs.append(token_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def detect_outliers(self, data: pd.DataFrame, token_symbol: str,
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in token data.
        
        Args:
            data: Data to check for outliers
            token_symbol: Token identifier
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
        
        Returns:
            Boolean DataFrame indicating outliers
        """
        if token_symbol not in self.is_fitted:
            raise ValueError(f"Normalizer not fitted for token: {token_symbol}")
        
        # Transform data first for consistent outlier detection
        normalized = self.transform(data, token_symbol)
        
        if method == 'zscore':
            # Z-score based outlier detection
            z_scores = np.abs(normalized.select_dtypes(include=[np.number]))
            outliers = z_scores > threshold
        
        elif method == 'iqr':
            # IQR based outlier detection
            Q1 = normalized.select_dtypes(include=[np.number]).quantile(0.25)
            Q3 = normalized.select_dtypes(include=[np.number]).quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            numeric_data = normalized.select_dtypes(include=[np.number])
            outliers = (numeric_data < lower_bound) | (numeric_data > upper_bound)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers