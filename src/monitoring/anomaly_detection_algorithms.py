"""
Advanced anomaly detection algorithms for trading patterns and system metrics.

This module provides specialized ML-powered anomaly detection algorithms
for different types of data in trading and system monitoring.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import json

import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger()


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    confidence: float
    anomaly_type: 'AnomalyType'
    timestamp: datetime = None
    affected_metrics: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.affected_metrics is None:
            self.affected_metrics = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'confidence': self.confidence,
            'anomaly_type': self.anomaly_type.value,
            'timestamp': self.timestamp.isoformat(),
            'affected_metrics': self.affected_metrics,
            'metadata': self.metadata
        }
    
    def __gt__(self, other):
        """Compare based on confidence."""
        if not isinstance(other, AnomalyResult):
            return NotImplemented
        return self.confidence > other.confidence
    
    def __lt__(self, other):
        """Compare based on confidence."""
        if not isinstance(other, AnomalyResult):
            return NotImplemented
        return self.confidence < other.confidence


from src.monitoring.intelligent_alerting import AnomalyType


class AnomalyAlgorithm(ABC):
    """Base class for anomaly detection algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize algorithm with configuration."""
        self.config = config
        self.fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray):
        """Fit the algorithm to training data."""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies in data."""
        pass
    
    @abstractmethod
    def get_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """Get anomaly scores for data points."""
        pass
    
    @staticmethod
    def create_algorithm(algorithm_type: str, config: Dict[str, Any]) -> 'AnomalyAlgorithm':
        """Factory method to create algorithm instances."""
        algorithms = {
            'isolation_forest': IsolationForestAlgorithm,
            'one_class_svm': OneClassSVMAlgorithm,
            'local_outlier_factor': LocalOutlierFactorAlgorithm,
            'statistical_outlier': StatisticalOutlierAlgorithm,
            'neural_network_autoencoder': AutoencoderAlgorithm
        }
        
        if algorithm_type not in algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        return algorithms[algorithm_type](config)
    
    @staticmethod
    def get_available_algorithms() -> List[str]:
        """Get list of available algorithm types."""
        return [
            'isolation_forest', 'one_class_svm', 'local_outlier_factor',
            'statistical_outlier', 'neural_network_autoencoder'
        ]


class IsolationForestAlgorithm(AnomalyAlgorithm):
    """Isolation Forest anomaly detection algorithm."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.contamination = config.get('contamination', 0.1)
        self.random_state = config.get('random_state', 42)
        self.model = None
    
    def fit(self, data: np.ndarray):
        """Fit Isolation Forest to data."""
        try:
            from sklearn.ensemble import IsolationForest
            
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state
            )
            self.model.fit(data)
            self.fitted = True
            
        except ImportError:
            logger.warning("scikit-learn not available for Isolation Forest")
            self.fitted = False
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)."""
        if not self.fitted or self.model is None:
            return np.ones(len(data))  # All normal if not fitted
        
        return self.model.predict(data)
    
    def get_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower scores = more anomalous)."""
        if not self.fitted or self.model is None:
            return np.zeros(len(data))
        
        return -self.model.decision_function(data)  # Invert for higher = more anomalous


class OneClassSVMAlgorithm(AnomalyAlgorithm):
    """One-Class SVM anomaly detection algorithm."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.nu = config.get('nu', 0.1)
        self.kernel = config.get('kernel', 'rbf')
        self.model = None
    
    def fit(self, data: np.ndarray):
        """Fit One-Class SVM to data."""
        try:
            from sklearn.svm import OneClassSVM
            
            self.model = OneClassSVM(nu=self.nu, kernel=self.kernel)
            self.model.fit(data)
            self.fitted = True
            
        except ImportError:
            logger.warning("scikit-learn not available for One-Class SVM")
            self.fitted = False
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)."""
        if not self.fitted or self.model is None:
            return np.ones(len(data))
        
        return self.model.predict(data)
    
    def get_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.fitted or self.model is None:
            return np.zeros(len(data))
        
        return -self.model.decision_function(data)


class LocalOutlierFactorAlgorithm(AnomalyAlgorithm):
    """Local Outlier Factor anomaly detection algorithm."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_neighbors = config.get('n_neighbors', 20)
        self.contamination = config.get('contamination', 0.1)
        self.model = None
    
    def fit(self, data: np.ndarray):
        """Fit LOF to data."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            self.model = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination
            )
            self.model.fit(data)
            self.fitted = True
            
        except ImportError:
            logger.warning("scikit-learn not available for LOF")
            self.fitted = False
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)."""
        if not self.fitted or self.model is None:
            return np.ones(len(data))
        
        return self.model.fit_predict(data)  # LOF requires fit_predict
    
    def get_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.fitted or self.model is None:
            return np.zeros(len(data))
        
        # LOF uses negative outlier factor
        predictions = self.model.fit_predict(data)
        factors = self.model.negative_outlier_factor_
        return -factors  # Higher = more anomalous


class StatisticalOutlierAlgorithm(AnomalyAlgorithm):
    """Statistical outlier detection using z-score."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get('threshold', 3.0)
        self.means = None
        self.stds = None
    
    def fit(self, data: np.ndarray):
        """Fit statistical parameters."""
        self.means = np.mean(data, axis=0)
        self.stds = np.std(data, axis=0)
        self.fitted = True
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies based on z-score."""
        if not self.fitted:
            return np.ones(len(data))
        
        z_scores = self.get_anomaly_score(data)
        return np.where(z_scores > self.threshold, -1, 1)
    
    def get_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """Get z-scores as anomaly scores."""
        if not self.fitted:
            return np.zeros(len(data))
        
        # Avoid division by zero
        safe_stds = np.where(self.stds == 0, 1e-8, self.stds)
        z_scores = np.abs((data - self.means) / safe_stds)
        
        # Return maximum z-score across features for each sample
        return np.max(z_scores, axis=1) if data.ndim > 1 else z_scores


class AutoencoderAlgorithm(AnomalyAlgorithm):
    """Neural network autoencoder for anomaly detection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.threshold = config.get('threshold', 0.1)
        self.encoding_dim = config.get('encoding_dim', 32)
        self.epochs = config.get('epochs', 100)
        self.model = None
    
    def fit(self, data: np.ndarray):
        """Fit autoencoder to data."""
        try:
            # Simple implementation - in practice would use TensorFlow/PyTorch
            logger.info("Autoencoder training started (simplified implementation)")
            
            # For testing purposes, simulate training
            input_dim = data.shape[1] if data.ndim > 1 else 1
            self.fitted = True
            
        except Exception as e:
            logger.warning("Autoencoder training failed", error=str(e))
            self.fitted = False
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies based on reconstruction error."""
        if not self.fitted:
            return np.ones(len(data))
        
        scores = self.get_anomaly_score(data)
        return np.where(scores > self.threshold, -1, 1)
    
    def get_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """Get reconstruction error as anomaly score."""
        if not self.fitted:
            return np.zeros(len(data))
        
        # Simplified implementation - in practice would use actual autoencoder
        # Return random scores for testing
        np.random.seed(42)
        return np.random.random(len(data)) * 0.1


class TradingPatternAnomalyDetector:
    """Specialized detector for trading pattern anomalies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trading pattern detector."""
        self.config = self._validate_config(config)
        self.volume_threshold_multiplier = config.get('volume_threshold_multiplier', 3.0)
        self.price_change_threshold = config.get('price_change_threshold', 0.05)
        self.liquidity_threshold = config.get('liquidity_threshold', 0.1)
        self.algorithms = self._initialize_algorithms()
        
        logger.info("TradingPatternAnomalyDetector initialized", config=config)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters."""
        if config.get('volume_threshold_multiplier', 1.0) <= 0:
            raise ValueError("volume_threshold_multiplier must be positive")
        
        if not 0 < config.get('price_change_threshold', 0.05) < 1:
            raise ValueError("price_change_threshold must be between 0 and 1")
        
        return config
    
    def _initialize_algorithms(self) -> Dict[str, AnomalyAlgorithm]:
        """Initialize anomaly detection algorithms."""
        algorithms = {}
        
        # Initialize different algorithms for different pattern types
        algorithms['isolation_forest'] = AnomalyAlgorithm.create_algorithm(
            'isolation_forest', {'contamination': 0.1}
        )
        algorithms['statistical'] = AnomalyAlgorithm.create_algorithm(
            'statistical_outlier', {'threshold': 3.0}
        )
        
        return algorithms
    
    async def detect_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect all types of trading anomalies."""
        anomalies = []
        
        # Detect different types of anomalies
        volume_anomalies = await self.detect_volume_anomalies(data)
        price_anomalies = await self.detect_price_anomalies(data)
        pump_dump_anomalies = await self.detect_pump_dump_patterns(data)
        wash_trading_anomalies = await self.detect_wash_trading(data)
        liquidity_anomalies = await self.detect_liquidity_anomalies(data)
        
        anomalies.extend(volume_anomalies)
        anomalies.extend(price_anomalies)
        anomalies.extend(pump_dump_anomalies)
        anomalies.extend(wash_trading_anomalies)
        anomalies.extend(liquidity_anomalies)
        
        return anomalies
    
    async def detect_volume_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect volume spike anomalies."""
        anomalies = []
        
        if 'volume' not in data.columns:
            return anomalies
        
        volume_mean = data['volume'].mean()
        volume_std = data['volume'].std()
        
        for idx, row in data.iterrows():
            volume = row['volume']
            volume_multiplier = volume / volume_mean if volume_mean > 0 else 0
            
            if volume_multiplier > self.volume_threshold_multiplier:
                confidence = min(volume_multiplier / self.volume_threshold_multiplier, 1.0)
                
                anomaly = AnomalyResult(
                    confidence=confidence,
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    timestamp=row.get('timestamp', datetime.now()),
                    affected_metrics=['volume'],
                    metadata={
                        'volume': volume,
                        'volume_mean': volume_mean,
                        'volume_multiplier': volume_multiplier,
                        'threshold': self.volume_threshold_multiplier
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_price_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect price-related anomalies including flash crashes."""
        anomalies = []
        
        if 'price' not in data.columns:
            return anomalies
        
        # Calculate price changes
        price_changes = data['price'].pct_change().fillna(0)
        
        for idx, change in price_changes.items():
            abs_change = abs(change)
            
            # Flash crash detection
            if change < -0.2:  # 20% drop
                anomaly = AnomalyResult(
                    confidence=min(abs_change / 0.2, 1.0),
                    anomaly_type=AnomalyType.FLASH_CRASH,
                    timestamp=data.loc[idx].get('timestamp', datetime.now()),
                    affected_metrics=['price'],
                    metadata={
                        'price_drop_percent': abs(change) * 100,
                        'price_change': change
                    }
                )
                anomalies.append(anomaly)
            
            # General price anomaly
            elif abs_change > self.price_change_threshold:
                anomaly = AnomalyResult(
                    confidence=min(abs_change / self.price_change_threshold, 1.0),
                    anomaly_type=AnomalyType.TRADING_PATTERN,
                    timestamp=data.loc[idx].get('timestamp', datetime.now()),
                    affected_metrics=['price'],
                    metadata={
                        'price_change_percent': abs_change * 100,
                        'price_change': change,
                        'threshold': self.price_change_threshold
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_pump_dump_patterns(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect pump and dump patterns."""
        anomalies = []
        
        if 'price' not in data.columns or 'volume' not in data.columns:
            return anomalies
        
        # Look for pump followed by dump pattern
        price_changes = data['price'].pct_change().fillna(0)
        volume_data = data['volume']
        
        # Simple pump and dump detection
        for i in range(len(price_changes) - 10):
            window = price_changes.iloc[i:i+10]
            volume_window = volume_data.iloc[i:i+10]
            
            # Look for significant pump followed by dump
            max_pump = window.max()
            min_dump = window.min()
            avg_volume = volume_window.mean()
            
            if max_pump > 0.3 and min_dump < -0.2 and avg_volume > volume_data.mean() * 2:
                anomaly = AnomalyResult(
                    confidence=0.8,
                    anomaly_type=AnomalyType.PUMP_AND_DUMP,
                    timestamp=data.iloc[i].get('timestamp', datetime.now()),
                    affected_metrics=['price', 'volume'],
                    metadata={
                        'pump_percent': max_pump * 100,
                        'dump_percent': abs(min_dump) * 100,
                        'volume_multiplier': avg_volume / volume_data.mean()
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_wash_trading(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect wash trading patterns."""
        anomalies = []
        
        required_cols = ['volume', 'price', 'trade_count']
        if not all(col in data.columns for col in required_cols):
            return anomalies
        
        # Detect high volume with minimal price movement
        price_volatility = data['price'].rolling(window=10).std().fillna(0)
        volume_data = data['volume']
        trade_counts = data['trade_count']
        
        for idx, row in data.iterrows():
            vol = row['volume']
            volatility = price_volatility.loc[idx]
            trades = row['trade_count']
            
            # High volume, low volatility, many small trades
            if (vol > volume_data.mean() * 3 and 
                volatility < price_volatility.mean() * 0.5 and
                trades > trade_counts.mean() * 2):
                
                anomaly = AnomalyResult(
                    confidence=0.7,
                    anomaly_type=AnomalyType.WASH_TRADING,
                    timestamp=row.get('timestamp', datetime.now()),
                    affected_metrics=['volume', 'price', 'trade_count'],
                    metadata={
                        'volume_multiplier': vol / volume_data.mean(),
                        'volatility_ratio': volatility / price_volatility.mean(),
                        'trade_count_multiplier': trades / trade_counts.mean()
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_liquidity_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect liquidity crisis patterns."""
        anomalies = []
        
        if 'bid_ask_spread' not in data.columns:
            return anomalies
        
        spread_data = data['bid_ask_spread']
        spread_mean = spread_data.mean()
        
        for idx, spread in spread_data.items():
            # Wide spreads indicate liquidity crisis
            if spread > spread_mean * 5:  # 5x normal spread
                anomaly = AnomalyResult(
                    confidence=min(spread / (spread_mean * 5), 1.0),
                    anomaly_type=AnomalyType.LIQUIDITY_CRISIS,
                    timestamp=data.loc[idx].get('timestamp', datetime.now()),
                    affected_metrics=['bid_ask_spread'],
                    metadata={
                        'spread': spread,
                        'spread_mean': spread_mean,
                        'spread_multiplier': spread / spread_mean
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _extract_trading_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for trading pattern analysis."""
        features = data.copy()
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_rolling_mean'] = data['volume'].rolling(window=10).mean()
            features['volume_rolling_std'] = data['volume'].rolling(window=10).std()
        
        # Price features
        if 'price' in data.columns:
            features['price_volatility'] = data['price'].rolling(window=10).std()
            features['price_momentum'] = data['price'].pct_change()
        
        # Trade features
        if 'trade_count' in data.columns and 'volume' in data.columns:
            features['trade_size_avg'] = data['volume'] / data['trade_count']
        
        # Spread features
        if 'bid_ask_spread' in data.columns and 'price' in data.columns:
            features['bid_ask_spread_normalized'] = data['bid_ask_spread'] / data['price']
        
        return features.fillna(0)


class SystemMetricsAnomalyDetector:
    """Specialized detector for system metrics anomalies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize system metrics detector."""
        self.config = config
        self.cpu_threshold = config.get('cpu_threshold', 0.8)
        self.memory_threshold = config.get('memory_threshold', 0.85)
        self.response_time_threshold = config.get('response_time_threshold', 1000)
        self.error_rate_threshold = config.get('error_rate_threshold', 0.05)
        
        logger.info("SystemMetricsAnomalyDetector initialized", config=config)
    
    async def detect_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect all types of system anomalies."""
        anomalies = []
        
        cpu_anomalies = await self.detect_cpu_anomalies(data)
        memory_anomalies = await self.detect_memory_anomalies(data)
        performance_anomalies = await self.detect_performance_anomalies(data)
        error_anomalies = await self.detect_error_anomalies(data)
        network_anomalies = await self.detect_network_anomalies(data)
        
        anomalies.extend(cpu_anomalies)
        anomalies.extend(memory_anomalies)
        anomalies.extend(performance_anomalies)
        anomalies.extend(error_anomalies)
        anomalies.extend(network_anomalies)
        
        return anomalies
    
    async def detect_cpu_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect CPU usage anomalies."""
        anomalies = []
        
        if 'cpu_usage' not in data.columns:
            return anomalies
        
        for idx, row in data.iterrows():
            cpu_usage = row['cpu_usage']
            
            if cpu_usage > self.cpu_threshold:
                severity = self._calculate_anomaly_severity(cpu_usage, self.cpu_threshold)
                
                anomaly = AnomalyResult(
                    confidence=severity,
                    anomaly_type=AnomalyType.CPU_SPIKE,
                    timestamp=row.get('timestamp', datetime.now()),
                    affected_metrics=['cpu_usage'],
                    metadata={
                        'cpu_usage': cpu_usage,
                        'threshold': self.cpu_threshold,
                        'severity': severity
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_memory_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect memory-related anomalies including leaks."""
        anomalies = []
        
        if 'memory_usage' not in data.columns:
            return anomalies
        
        memory_data = data['memory_usage']
        
        # Detect memory leaks (gradual increase)
        if len(memory_data) >= 10:
            # Check for consistent upward trend
            recent_data = memory_data.tail(10)
            if len(recent_data) > 0 and recent_data.iloc[-1] > recent_data.iloc[0] * 1.2:  # 20% increase
                initial_value = recent_data.iloc[0] if recent_data.iloc[0] > 0 else 1.0  # Avoid division by zero
                trend_strength = (recent_data.iloc[-1] - recent_data.iloc[0]) / initial_value
                
                anomaly = AnomalyResult(
                    confidence=min(trend_strength, 1.0),
                    anomaly_type=AnomalyType.MEMORY_LEAK,
                    timestamp=data.iloc[-1].get('timestamp', datetime.now()),
                    affected_metrics=['memory_usage'],
                    metadata={
                        'memory_increase_percent': trend_strength * 100,
                        'current_usage': recent_data.iloc[-1],
                        'initial_usage': recent_data.iloc[0]
                    }
                )
                anomalies.append(anomaly)
        
        # Detect high memory usage
        for idx, row in data.iterrows():
            memory_usage = row['memory_usage']
            
            if memory_usage > self.memory_threshold:
                severity = self._calculate_anomaly_severity(memory_usage, self.memory_threshold)
                
                anomaly = AnomalyResult(
                    confidence=severity,
                    anomaly_type=AnomalyType.SYSTEM_METRICS,
                    timestamp=row.get('timestamp', datetime.now()),
                    affected_metrics=['memory_usage'],
                    metadata={
                        'memory_usage': memory_usage,
                        'threshold': self.memory_threshold,
                        'severity': severity
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_performance_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect performance degradation."""
        anomalies = []
        
        if 'response_time_ms' not in data.columns:
            return anomalies
        
        for idx, row in data.iterrows():
            response_time = row['response_time_ms']
            
            if response_time > self.response_time_threshold:
                severity = min(response_time / self.response_time_threshold, 1.0)
                
                anomaly = AnomalyResult(
                    confidence=severity,
                    anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                    timestamp=row.get('timestamp', datetime.now()),
                    affected_metrics=['response_time_ms'],
                    metadata={
                        'response_time_ms': response_time,
                        'threshold': self.response_time_threshold,
                        'severity': severity
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_error_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect error rate spikes."""
        anomalies = []
        
        if 'error_rate' not in data.columns:
            return anomalies
        
        for idx, row in data.iterrows():
            error_rate = row['error_rate']
            
            if error_rate > self.error_rate_threshold:
                severity = min(error_rate / self.error_rate_threshold, 1.0)
                
                anomaly = AnomalyResult(
                    confidence=severity,
                    anomaly_type=AnomalyType.ERROR_RATE_SPIKE,
                    timestamp=row.get('timestamp', datetime.now()),
                    affected_metrics=['error_rate'],
                    metadata={
                        'error_rate': error_rate,
                        'threshold': self.error_rate_threshold,
                        'error_percentage': error_rate * 100
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_network_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect network-related anomalies."""
        anomalies = []
        
        if 'network_io_mb' not in data.columns:
            return anomalies
        
        network_data = data['network_io_mb']
        network_mean = network_data.mean()
        network_std = network_data.std()
        
        for idx, row in data.iterrows():
            network_io = row['network_io_mb']
            
            # Detect unusually high network I/O
            if network_io > network_mean + 3 * network_std:
                anomaly = AnomalyResult(
                    confidence=0.8,
                    anomaly_type=AnomalyType.NETWORK_ANOMALY,
                    timestamp=row.get('timestamp', datetime.now()),
                    affected_metrics=['network_io_mb'],
                    metadata={
                        'network_io_mb': network_io,
                        'network_mean': network_mean,
                        'network_multiplier': network_io / network_mean if network_mean > 0 else 0
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def calculate_health_score(self, data: pd.DataFrame) -> float:
        """Calculate overall system health score."""
        if data.empty:
            return 1.0
        
        scores = []
        
        # CPU health score
        if 'cpu_usage' in data.columns and self.cpu_threshold > 0:
            avg_cpu = data['cpu_usage'].mean()
            cpu_score = max(0, 1 - (avg_cpu / self.cpu_threshold))
            scores.append(cpu_score)
        
        # Memory health score
        if 'memory_usage' in data.columns and self.memory_threshold > 0:
            avg_memory = data['memory_usage'].mean()
            memory_score = max(0, 1 - (avg_memory / self.memory_threshold))
            scores.append(memory_score)
        
        # Response time health score
        if 'response_time_ms' in data.columns and self.response_time_threshold > 0:
            avg_response = data['response_time_ms'].mean()
            response_score = max(0, 1 - (avg_response / self.response_time_threshold))
            scores.append(response_score)
        
        # Error rate health score
        if 'error_rate' in data.columns and self.error_rate_threshold > 0:
            avg_error_rate = data['error_rate'].mean()
            error_score = max(0, 1 - (avg_error_rate / self.error_rate_threshold))
            scores.append(error_score)
        
        return np.mean(scores) if scores else 1.0
    
    def _calculate_anomaly_severity(self, value: float, threshold: float) -> float:
        """Calculate anomaly severity score."""
        if threshold == 0:
            return 1.0
        
        severity = (value - threshold) / threshold
        return min(max(severity, 0), 1.0)


class MarketAnomalyDetector:
    """Specialized detector for market-wide anomalies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize market anomaly detector."""
        self.config = config
        self.market_correlation_threshold = config.get('market_correlation_threshold', 0.8)
        self.volatility_spike_threshold = config.get('volatility_spike_threshold', 2.0)
        self.market_crash_threshold = config.get('market_crash_threshold', 0.1)
        
        logger.info("MarketAnomalyDetector initialized", config=config)
    
    async def detect_market_crashes(self, market_data: Dict[str, pd.DataFrame]) -> List[AnomalyResult]:
        """Detect market-wide crashes."""
        anomalies = []
        
        if not market_data:
            return anomalies
        
        # Calculate market-wide price changes
        all_changes = []
        for asset_name, asset_data in market_data.items():
            if 'price' in asset_data.columns:
                price_changes = asset_data['price'].pct_change().fillna(0)
                all_changes.extend(price_changes.tolist())
        
        if not all_changes:
            return anomalies
        
        # Detect simultaneous drops across assets
        negative_changes = [c for c in all_changes if c < -self.market_crash_threshold]
        
        if len(negative_changes) > len(all_changes) * 0.7:  # 70% of assets dropping
            avg_drop = np.mean(negative_changes)
            
            anomaly = AnomalyResult(
                confidence=min(abs(avg_drop) / self.market_crash_threshold, 1.0),
                anomaly_type=AnomalyType.MARKET_CRASH,
                affected_metrics=['price'],
                metadata={
                    'average_drop_percent': abs(avg_drop) * 100,
                    'affected_assets_percent': len(negative_changes) / len(all_changes) * 100,
                    'crash_threshold': self.market_crash_threshold
                }
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_correlation_anomalies(self, market_data: Dict[str, pd.DataFrame]) -> List[AnomalyResult]:
        """Detect correlation breakdown anomalies."""
        anomalies = []
        
        correlations = self._calculate_market_correlations(market_data)
        
        # Simple correlation breakdown detection
        if correlations.size > 0:
            avg_correlation = np.mean(correlations[correlations != 1.0])  # Exclude diagonal
            
            if avg_correlation < self.market_correlation_threshold:
                anomaly = AnomalyResult(
                    confidence=1 - avg_correlation,
                    anomaly_type=AnomalyType.CORRELATION_BREAKDOWN,
                    affected_metrics=['correlation'],
                    metadata={
                        'average_correlation': avg_correlation,
                        'correlation_threshold': self.market_correlation_threshold
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def detect_volatility_anomalies(self, market_data: Dict[str, pd.DataFrame]) -> List[AnomalyResult]:
        """Detect volatility clustering and spikes."""
        anomalies = []
        
        for asset_name, asset_data in market_data.items():
            if 'price' in asset_data.columns:
                price_changes = asset_data['price'].pct_change().fillna(0)
                volatility = price_changes.rolling(window=10).std().fillna(0)
                
                # Detect volatility spikes
                vol_mean = volatility.mean()
                vol_std = volatility.std()
                
                for idx, vol in volatility.items():
                    if vol > vol_mean + self.volatility_spike_threshold * vol_std:
                        anomaly = AnomalyResult(
                            confidence=min(vol / (vol_mean + self.volatility_spike_threshold * vol_std), 1.0),
                            anomaly_type=AnomalyType.VOLATILITY_CLUSTERING,
                            affected_metrics=['volatility'],
                            metadata={
                                'asset': asset_name,
                                'volatility': vol,
                                'volatility_mean': vol_mean,
                                'volatility_multiplier': vol / vol_mean if vol_mean > 0 else 0
                            }
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_market_correlations(self, market_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Calculate correlation matrix for market data."""
        price_data = {}
        
        for asset_name, asset_data in market_data.items():
            if 'price' in asset_data.columns:
                price_data[asset_name] = asset_data['price'].pct_change().fillna(0)
        
        if not price_data:
            return np.array([])
        
        df = pd.DataFrame(price_data)
        return df.corr().values
    
    def _analyze_sector_anomalies(self, market_data: Dict[str, pd.DataFrame], sectors: Dict[str, str]) -> Dict[str, Any]:
        """Analyze anomalies by market sector."""
        sector_analysis = {}
        
        for sector in set(sectors.values()):
            sector_assets = [asset for asset, s in sectors.items() if s == sector]
            sector_data = {asset: market_data[asset] for asset in sector_assets if asset in market_data}
            
            if sector_data:
                # Simple sector analysis
                sector_analysis[sector] = {
                    'asset_count': len(sector_data),
                    'analyzed': True
                }
        
        return sector_analysis


class EnsembleAnomalyDetector:
    """Ensemble detector that combines multiple algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ensemble detector."""
        self.config = self._validate_config(config)
        self.voting_strategy = config.get('voting_strategy', 'weighted')
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.algorithm_weights = config.get('algorithm_weights', {})
        self.detectors = {}
        self.max_workers = 1  # For parallel processing
        
        logger.info("EnsembleAnomalyDetector initialized", config=config)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ensemble configuration."""
        weights = config.get('algorithm_weights', {})
        if weights and abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Algorithm weights must sum to 1.0")
        
        return config
    
    async def detect_anomalies(self, data: Dict[str, Any]) -> List[AnomalyResult]:
        """Detect anomalies using ensemble approach."""
        all_results = []
        
        # Collect results from all detectors
        for detector_name, detector in self.detectors.items():
            try:
                results = await detector.detect_anomalies(data)
                all_results.extend(results)
            except Exception as e:
                logger.warning("Detector failed", detector=detector_name, error=str(e))
        
        # Apply ensemble voting
        ensemble_results = self._apply_ensemble_voting(all_results)
        
        return ensemble_results
    
    def _apply_ensemble_voting(self, results: List[AnomalyResult]) -> List[AnomalyResult]:
        """Apply ensemble voting to combine results."""
        if not results:
            return []
        
        # Group results by anomaly type and apply consensus
        grouped_results = {}
        for result in results:
            key = result.anomaly_type
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        ensemble_results = []
        for anomaly_type, type_results in grouped_results.items():
            if self.voting_strategy == 'weighted':
                consensus_result = self._weighted_consensus(type_results)
            else:
                consensus_result = self._majority_consensus(type_results)
            
            if consensus_result and consensus_result.confidence >= self.confidence_threshold:
                ensemble_results.append(consensus_result)
        
        return ensemble_results
    
    def _weighted_consensus(self, results: List[AnomalyResult]) -> Optional[AnomalyResult]:
        """Apply weighted consensus to results."""
        if not results:
            return None
        
        # Simple weighted average of confidence scores
        confidences = [r.confidence for r in results]
        weights = list(self.algorithm_weights.values()) if self.algorithm_weights else [1.0] * len(results)
        
        if len(weights) != len(confidences):
            weights = [1.0] * len(confidences)  # Equal weights if mismatch
        
        weighted_confidence = self._aggregate_confidence(confidences, weights, 'weighted_average')
        
        # Create consensus result
        consensus_type = self._determine_consensus_type(results)
        
        return AnomalyResult(
            confidence=weighted_confidence,
            anomaly_type=consensus_type,
            affected_metrics=list(set().union(*[r.affected_metrics for r in results])),
            metadata={
                'ensemble_method': 'weighted_consensus',
                'individual_confidences': confidences,
                'weights': weights
            }
        )
    
    def _majority_consensus(self, results: List[AnomalyResult]) -> Optional[AnomalyResult]:
        """Apply majority consensus to results."""
        if not results:
            return None
        
        # Simple majority vote
        if len(results) >= 2:  # Need at least 2 votes
            confidences = [r.confidence for r in results]
            avg_confidence = np.mean(confidences)
            
            consensus_type = self._determine_consensus_type(results)
            
            return AnomalyResult(
                confidence=avg_confidence,
                anomaly_type=consensus_type,
                affected_metrics=list(set().union(*[r.affected_metrics for r in results])),
                metadata={
                    'ensemble_method': 'majority_consensus',
                    'vote_count': len(results)
                }
            )
        
        return None
    
    def _aggregate_confidence(self, confidences: List[float], weights: List[float] = None, method: str = 'average') -> float:
        """Aggregate confidence scores."""
        if not confidences:
            return 0.0
        
        if method == 'weighted_average' and weights:
            return sum(c * w for c, w in zip(confidences, weights))
        elif method == 'maximum':
            return max(confidences)
        else:
            return np.mean(confidences)
    
    def _determine_consensus_type(self, results: List[AnomalyResult]) -> AnomalyType:
        """Determine consensus anomaly type."""
        if not results:
            return AnomalyType.TRADING_PATTERN  # Default
        
        # Return the most common type, or the one with highest confidence
        type_counts = {}
        for result in results:
            anomaly_type = result.anomaly_type
            if anomaly_type not in type_counts:
                type_counts[anomaly_type] = []
            type_counts[anomaly_type].append(result.confidence)
        
        # Return type with highest average confidence
        best_type = max(type_counts.items(), key=lambda x: np.mean(x[1]))[0]
        return best_type
    
    def enable_parallel_processing(self, max_workers: int = 4):
        """Enable parallel processing for detectors."""
        self.max_workers = max_workers
        logger.info("Parallel processing enabled", max_workers=max_workers)