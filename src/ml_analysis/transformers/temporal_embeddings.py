"""
Temporal embeddings optimized for financial time-series data
Specialized for cryptocurrency trading with market-aware features

Features:
- Trading session encoding (Asian, European, US markets)
- Market regime detection and encoding
- Volatility regime classification
- Crypto-specific event embeddings
- Block time and network congestion encoding
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class TemporalEmbeddingConfig:
    """Configuration for temporal embeddings"""
    
    # Model dimensions
    d_model: int = 512
    max_sequence_length: int = 1000  # GCP memory constraint
    dropout: float = 0.1
    
    # Feature flags
    enable_trading_sessions: bool = True
    enable_market_regimes: bool = True
    enable_volatility_regimes: bool = True
    enable_crypto_events: bool = True
    enable_seasonal_patterns: bool = True
    
    # Trading session parameters
    session_overlap_bonus: float = 1.2  # Extra weight for session overlaps
    
    # Market regime parameters
    regime_detection_window: int = 20  # Lookback window for regime detection
    
    # Volatility parameters
    volatility_window: int = 10  # Window for volatility calculation
    volatility_thresholds: Tuple[float, float] = (0.02, 0.08)  # Low/high vol thresholds
    
    def __post_init__(self):
        """Validate configuration"""
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        
        # GCP memory constraints
        if self.max_sequence_length > 1000:
            raise ValueError("max_sequence_length must be <= 1000 for memory constraints")


class TemporalEmbedding(nn.Module):
    """
    Basic temporal embedding for time-series data
    
    Converts timestamps into temporal features including:
    - Hour of day (0-23)
    - Day of week (0-6) 
    - Day of month (1-31)
    - Month (1-12)
    - Year
    """
    
    def __init__(self, config: TemporalEmbeddingConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_length = config.max_sequence_length
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        # Temporal component embeddings
        self.hour_embedding = nn.Embedding(24, config.d_model // 8)      # Hours 0-23
        self.day_embedding = nn.Embedding(7, config.d_model // 8)        # Days 0-6
        self.month_embedding = nn.Embedding(12, config.d_model // 8)     # Months 0-11
        self.year_embedding = nn.Embedding(50, config.d_model // 8)      # Years (relative)
        
        # Day of month (1-31) - using sin/cos encoding for cyclical nature
        self.day_of_month_dim = config.d_model // 8
        
        # Weekend indicator
        self.weekend_embedding = nn.Embedding(2, config.d_model // 8)    # 0=weekday, 1=weekend
        
        # Quarter embedding
        self.quarter_embedding = nn.Embedding(4, config.d_model // 8)    # Q1-Q4
        
        # Combine all temporal features
        total_dim = config.d_model // 8 * 6 + self.day_of_month_dim
        self.temporal_combiner = nn.Linear(total_dim, config.d_model)
        
        logger.info(
            "Initialized basic temporal embedding",
            d_model=config.d_model,
            total_feature_dim=total_dim
        )
    
    def _extract_temporal_components(self, timestamps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract temporal components from unix timestamps
        
        Args:
            timestamps: Unix timestamps tensor of shape (batch_size, seq_len)
            
        Returns:
            Dictionary of temporal components
        """
        batch_size, seq_len = timestamps.shape
        device = timestamps.device
        
        # Convert to datetime components (vectorized)
        # Note: This is a simplified version - in production would handle timezones properly
        
        # Assuming timestamps are in seconds since epoch
        seconds_per_hour = 3600
        seconds_per_day = 86400
        seconds_per_year = 31536000  # Approximate
        
        # Extract components
        hours = (timestamps % seconds_per_day) // seconds_per_hour
        days = (timestamps // seconds_per_day) % 7  # Day of week
        years = timestamps // seconds_per_year  # Years since epoch (approximate)
        
        # Month calculation (simplified)
        day_of_year = (timestamps % seconds_per_year) // seconds_per_day
        months = torch.clamp(day_of_year // 30, 0, 11)  # Rough month estimation
        
        # Day of month (cyclical encoding)
        day_of_month = (day_of_year % 30) + 1
        
        # Quarter
        quarters = months // 3
        
        # Weekend indicator
        is_weekend = (days >= 5).long()  # Saturday=5, Sunday=6
        
        return {
            'hours': hours.long(),
            'days': days.long(), 
            'months': months.long(),
            'years': (years - years.min()).long(),  # Relative years
            'day_of_month': day_of_month.float(),
            'quarters': quarters.long(),
            'is_weekend': is_weekend
        }
    
    def _encode_day_of_month(self, day_of_month: torch.Tensor) -> torch.Tensor:
        """
        Encode day of month using sinusoidal encoding (cyclical)
        
        Args:
            day_of_month: Day values (1-31)
            
        Returns:
            Sinusoidal encoding of shape (*day_of_month.shape, day_of_month_dim)
        """
        # Normalize to [0, 2π]
        normalized = (day_of_month - 1) / 30 * 2 * math.pi
        
        # Create frequency components
        freqs = torch.arange(self.day_of_month_dim // 2, device=day_of_month.device)
        angles = normalized.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        
        # Sin and cos components
        sin_components = torch.sin(angles)
        cos_components = torch.cos(angles)
        
        # Interleave sin and cos
        encoding = torch.stack([sin_components, cos_components], dim=-1)
        encoding = encoding.flatten(start_dim=-2)
        
        # Handle odd dimensions
        if self.day_of_month_dim % 2 == 1:
            extra = torch.zeros(*day_of_month.shape, 1, device=day_of_month.device)
            encoding = torch.cat([encoding, extra], dim=-1)
        
        return encoding
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Convert timestamps to temporal embeddings
        
        Args:
            timestamps: Unix timestamps of shape (batch_size, seq_len)
            
        Returns:
            Temporal embeddings of shape (batch_size, seq_len, d_model)
        """
        # Extract temporal components
        components = self._extract_temporal_components(timestamps)
        
        # Get embeddings for each component
        hour_emb = self.hour_embedding(components['hours'])
        day_emb = self.day_embedding(components['days'])
        month_emb = self.month_embedding(components['months'])
        year_emb = self.year_embedding(components['years'])
        quarter_emb = self.quarter_embedding(components['quarters'])
        weekend_emb = self.weekend_embedding(components['is_weekend'])
        
        # Encode day of month cyclically
        day_of_month_emb = self._encode_day_of_month(components['day_of_month'])
        
        # Concatenate all features
        features = torch.cat([
            hour_emb, day_emb, month_emb, year_emb, 
            quarter_emb, weekend_emb, day_of_month_emb
        ], dim=-1)
        
        # Combine and project to d_model
        embeddings = self.temporal_combiner(features)
        
        return self.dropout(embeddings)


class TradingSessionEncoder(nn.Module):
    """
    Encode trading sessions and their overlaps
    
    Trading sessions:
    - Asian: 00:00-08:00 UTC (Tokyo, Sydney)
    - European: 08:00-16:00 UTC (London, Frankfurt)
    - US: 16:00-24:00 UTC (New York)
    
    Overlaps are periods of higher liquidity and volatility.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Session embeddings
        self.session_embedding = nn.Embedding(3, d_model // 2)  # 3 sessions
        self.overlap_embedding = nn.Embedding(2, d_model // 2)  # Overlap yes/no
        
        self.combiner = nn.Linear(d_model, d_model)
    
    def detect_trading_sessions(self, timestamps: List[datetime]) -> List[int]:
        """
        Detect trading session from datetime objects
        
        Args:
            timestamps: List of datetime objects
            
        Returns:
            List of session IDs (0=Asian, 1=European, 2=US)
        """
        sessions = []
        for dt in timestamps:
            hour_utc = dt.hour
            
            if 0 <= hour_utc < 8:
                sessions.append(0)  # Asian
            elif 8 <= hour_utc < 16:
                sessions.append(1)  # European
            else:
                sessions.append(2)  # US
        
        return sessions
    
    def detect_session_overlaps(self, timestamps: List[datetime]) -> List[bool]:
        """
        Detect session overlap periods
        
        Overlaps:
        - Asian-European: 07:00-09:00 UTC
        - European-US: 15:00-17:00 UTC
        """
        overlaps = []
        for dt in timestamps:
            hour_utc = dt.hour
            
            # Asian-European overlap
            if 7 <= hour_utc <= 9:
                overlaps.append(True)
            # European-US overlap
            elif 15 <= hour_utc <= 17:
                overlaps.append(True)
            else:
                overlaps.append(False)
        
        return overlaps
    
    def forward(self, sessions: torch.Tensor, overlaps: torch.Tensor) -> torch.Tensor:
        """
        Encode trading sessions and overlaps
        
        Args:
            sessions: Session IDs of shape (batch_size, seq_len)
            overlaps: Overlap indicators of shape (batch_size, seq_len)
            
        Returns:
            Encoded features of shape (batch_size, seq_len, d_model)
        """
        session_emb = self.session_embedding(sessions)
        overlap_emb = self.overlap_embedding(overlaps.long())
        
        # Combine features
        combined = torch.cat([session_emb, overlap_emb], dim=-1)
        return self.combiner(combined)


class MarketRegimeEncoder(nn.Module):
    """
    Encode market regimes (bull, bear, sideways) based on price trends
    """
    
    def __init__(self, d_model: int, lookback_window: int = 20):
        super().__init__()
        self.d_model = d_model
        self.lookback_window = lookback_window
        
        # Regime embeddings: 0=bull, 1=bear, 2=sideways
        self.regime_embedding = nn.Embedding(3, d_model)
        
        # Trend strength embedding (continuous)
        self.trend_strength_encoder = nn.Linear(1, d_model // 2)
        
        self.combiner = nn.Linear(d_model + d_model // 2, d_model)
    
    def detect_regime(self, prices: torch.Tensor) -> int:
        """
        Detect market regime from price sequence
        
        Args:
            prices: Price sequence
            
        Returns:  
            Regime ID (0=bull, 1=bear, 2=sideways)
        """
        if len(prices) < 2:
            return 2  # Sideways for insufficient data
        
        # Calculate returns
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        
        # Simple trend detection
        mean_return = returns.mean().item()
        return_std = returns.std().item()
        
        # Classify regime
        if mean_return > 2 * return_std:
            return 0  # Bull market
        elif mean_return < -2 * return_std:
            return 1  # Bear market
        else:
            return 2  # Sideways market
    
    def detect_regime_sequence(self, prices: torch.Tensor) -> List[int]:
        """
        Detect regime for each position in sequence
        """
        regimes = []
        for i in range(len(prices)):
            start_idx = max(0, i - self.lookback_window)
            window_prices = prices[start_idx:i+1]
            regime = self.detect_regime(window_prices)
            regimes.append(regime)
        
        return regimes
    
    def detect_transitions(self, regimes: List[int]) -> List[bool]:
        """
        Detect regime transitions
        """
        transitions = [False]  # First position has no transition
        for i in range(1, len(regimes)):
            transitions.append(regimes[i] != regimes[i-1])
        
        return transitions
    
    def calculate_trend_strength(self, prices: torch.Tensor) -> float:
        """
        Calculate trend strength (0 = no trend, 1 = strong trend)
        """
        if len(prices) < 2:
            return 0.0
        
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        
        # R-squared of linear regression as trend strength measure
        x = torch.arange(len(returns), dtype=torch.float)
        x_mean = x.mean()
        y_mean = returns.mean()
        
        numerator = ((x - x_mean) * (returns - y_mean)).sum()
        denominator_x = ((x - x_mean) ** 2).sum()
        denominator_y = ((returns - y_mean) ** 2).sum()
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
        
        correlation = numerator / torch.sqrt(denominator_x * denominator_y)
        r_squared = correlation ** 2
        
        return torch.clamp(r_squared, 0, 1).item()
    
    def forward(self, regimes: torch.Tensor, trend_strengths: torch.Tensor) -> torch.Tensor:
        """
        Encode market regimes
        
        Args:
            regimes: Regime IDs of shape (batch_size, seq_len)
            trend_strengths: Trend strengths of shape (batch_size, seq_len)
            
        Returns:
            Encoded features of shape (batch_size, seq_len, d_model)
        """
        regime_emb = self.regime_embedding(regimes)
        strength_emb = self.trend_strength_encoder(trend_strengths.unsqueeze(-1))
        
        combined = torch.cat([regime_emb, strength_emb], dim=-1)
        return self.combiner(combined)


class VolatilityRegimeEncoder(nn.Module):
    """
    Encode volatility regimes (low, medium, high volatility periods)
    """
    
    def __init__(self, d_model: int, vol_window: int = 10):
        super().__init__()
        self.d_model = d_model
        self.vol_window = vol_window
        
        # Volatility regime embeddings: 0=low, 1=medium, 2=high
        self.vol_regime_embedding = nn.Embedding(3, d_model)
    
    def calculate_volatility(self, returns: torch.Tensor) -> float:
        """
        Calculate volatility from returns
        
        Args:
            returns: Return sequence
            
        Returns:
            Volatility (standard deviation of returns)
        """
        if len(returns) < 2:
            return 0.0
        
        return returns.std().item()
    
    def classify_volatility_regime(self, volatility: float) -> int:
        """
        Classify volatility into regime
        
        Args:
            volatility: Volatility value
            
        Returns:
            Regime ID (0=low, 1=medium, 2=high)
        """
        # These thresholds should be calibrated to crypto market data
        if volatility < 0.02:  # Less than 2% daily volatility
            return 0  # Low volatility
        elif volatility < 0.08:  # Less than 8% daily volatility  
            return 1  # Medium volatility
        else:
            return 2  # High volatility
    
    def forward(self, vol_regimes: torch.Tensor) -> torch.Tensor:
        """
        Encode volatility regimes
        
        Args:
            vol_regimes: Volatility regime IDs of shape (batch_size, seq_len)
            
        Returns:
            Encoded features of shape (batch_size, seq_len, d_model)
        """
        return self.vol_regime_embedding(vol_regimes)


class FinancialTemporalEmbedding(TemporalEmbedding):
    """
    Extended temporal embedding with financial market features
    """
    
    def __init__(self, config: TemporalEmbeddingConfig):
        super().__init__(config)
        
        # Additional financial features
        if config.enable_trading_sessions:
            self.session_encoder = TradingSessionEncoder(config.d_model // 4)
        
        if config.enable_market_regimes:
            self.regime_encoder = MarketRegimeEncoder(config.d_model // 4)
        
        if config.enable_volatility_regimes:
            self.vol_encoder = VolatilityRegimeEncoder(config.d_model // 4)
        
        # Feature combination
        base_dim = config.d_model
        additional_dims = 0
        if config.enable_trading_sessions:
            additional_dims += config.d_model // 4
        if config.enable_market_regimes:
            additional_dims += config.d_model // 4  
        if config.enable_volatility_regimes:
            additional_dims += config.d_model // 4
        
        if additional_dims > 0:
            self.financial_combiner = nn.Linear(base_dim + additional_dims, config.d_model)
        
        logger.info(
            "Initialized financial temporal embedding",
            base_dim=base_dim,
            additional_dims=additional_dims,
            total_dim=base_dim + additional_dims
        )
    
    def forward(self, 
                timestamps: torch.Tensor,
                market_regimes: Optional[torch.Tensor] = None,
                volatility_regimes: Optional[torch.Tensor] = None,
                trading_sessions: Optional[torch.Tensor] = None,
                session_overlaps: Optional[torch.Tensor] = None,
                trend_strengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced forward pass with financial features
        """
        # Get base temporal embeddings
        embeddings = super().forward(timestamps)
        
        additional_features = []
        
        # Add trading session features
        if hasattr(self, 'session_encoder') and trading_sessions is not None:
            if session_overlaps is None:
                session_overlaps = torch.zeros_like(trading_sessions)
            session_features = self.session_encoder(trading_sessions, session_overlaps)
            additional_features.append(session_features)
        
        # Add market regime features
        if hasattr(self, 'regime_encoder') and market_regimes is not None:
            if trend_strengths is None:
                trend_strengths = torch.zeros_like(market_regimes, dtype=torch.float)
            regime_features = self.regime_encoder(market_regimes, trend_strengths)
            additional_features.append(regime_features)
        
        # Add volatility regime features
        if hasattr(self, 'vol_encoder') and volatility_regimes is not None:
            vol_features = self.vol_encoder(volatility_regimes)
            additional_features.append(vol_features)
        
        # Combine all features
        if additional_features and hasattr(self, 'financial_combiner'):
            all_features = torch.cat([embeddings] + additional_features, dim=-1)
            embeddings = self.financial_combiner(all_features)
        
        return embeddings


class CryptoMarketEmbedding(FinancialTemporalEmbedding):
    """
    Crypto-specific temporal embedding with blockchain and DeFi features
    """
    
    def __init__(self, config: TemporalEmbeddingConfig):
        super().__init__(config)
        
        if config.enable_crypto_events:
            # Crypto event embeddings
            self.crypto_event_embedding = nn.Embedding(10, config.d_model // 8)  # 10 event types
            
            # Block time encoding
            self.block_time_encoder = nn.Linear(1, config.d_model // 8)
            
            # DeFi event embedding
            self.defi_event_embedding = nn.Embedding(5, config.d_model // 8)  # 5 DeFi event types
            
            # Network congestion embedding
            self.congestion_embedding = nn.Embedding(4, config.d_model // 8)  # 4 congestion levels
            
            # Combine crypto features
            crypto_dim = config.d_model // 8 * 4
            self.crypto_combiner = nn.Linear(crypto_dim, config.d_model // 4)
            
            # Update main combiner
            if hasattr(self, 'financial_combiner'):
                # Need to account for additional crypto features
                old_in_features = self.financial_combiner.in_features
                new_in_features = old_in_features + config.d_model // 4
                self.financial_combiner = nn.Linear(new_in_features, config.d_model)
        
        logger.info("Initialized crypto market embedding with blockchain features")
    
    def forward(self,
                timestamps: torch.Tensor,
                crypto_events: Optional[torch.Tensor] = None,
                block_times: Optional[torch.Tensor] = None,
                defi_events: Optional[torch.Tensor] = None,
                network_congestion: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass with crypto-specific features
        """
        # Get financial temporal embeddings
        embeddings = super().forward(timestamps, **kwargs)
        
        # Add crypto-specific features
        if hasattr(self, 'crypto_combiner'):
            crypto_features = []
            batch_size, seq_len = timestamps.shape
            device = timestamps.device
            
            # Crypto events
            if crypto_events is not None:
                crypto_event_emb = self.crypto_event_embedding(crypto_events)
            else:
                crypto_event_emb = torch.zeros(batch_size, seq_len, self.d_model // 8, device=device)
            crypto_features.append(crypto_event_emb)
            
            # Block times
            if block_times is not None:
                # Normalize block times (log scale)
                normalized_block_times = torch.log(block_times + 1e-6).unsqueeze(-1)
                block_time_emb = self.block_time_encoder(normalized_block_times)
            else:
                block_time_emb = torch.zeros(batch_size, seq_len, self.d_model // 8, device=device)
            crypto_features.append(block_time_emb)
            
            # DeFi events
            if defi_events is not None:
                defi_emb = self.defi_event_embedding(defi_events)
            else:
                defi_emb = torch.zeros(batch_size, seq_len, self.d_model // 8, device=device)
            crypto_features.append(defi_emb)
            
            # Network congestion
            if network_congestion is not None:
                congestion_emb = self.congestion_embedding(network_congestion)
            else:
                congestion_emb = torch.zeros(batch_size, seq_len, self.d_model // 8, device=device)
            crypto_features.append(congestion_emb)
            
            # Combine crypto features
            combined_crypto = torch.cat(crypto_features, dim=-1)
            crypto_embeddings = self.crypto_combiner(combined_crypto)
            
            # Add to main embeddings
            if hasattr(self, 'financial_combiner'):
                all_features = torch.cat([embeddings, crypto_embeddings], dim=-1)
                embeddings = self.financial_combiner(all_features)
        
        return embeddings


# Export all classes
__all__ = [
    'TemporalEmbeddingConfig',
    'TemporalEmbedding',
    'FinancialTemporalEmbedding', 
    'CryptoMarketEmbedding',
    'TradingSessionEncoder',
    'MarketRegimeEncoder',
    'VolatilityRegimeEncoder'
]