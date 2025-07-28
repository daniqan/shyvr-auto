"""
Backtest Result Integration with Continuous Learning

Provides lightweight interface to feed backtest results to the learning system
as validation data. Maintains minimal coupling while enabling model validation
against historical performance.

Following TDD methodology - implementation after comprehensive tests.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np
import structlog
import random

from src.modes.analysis_mode import BacktestResult
from src.rl_agent.base import TradeAction, MarketState
from src.rl_agent.experience_replay import Experience
from src.discovery.base import DiscoveredToken, TokenStatus
from src.utils.base import Chain

try:
    from src.modes.continuous_learning import ContinuousLearningEngine
except ImportError:
    # Handle optional dependency gracefully
    ContinuousLearningEngine = None


logger = structlog.get_logger()


class IntegrationStatus(Enum):
    """Status of backtest integration"""
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class BacktestValidationError(Exception):
    """Raised when backtest validation fails"""
    pass


@dataclass
class BacktestIntegrationConfig:
    """Configuration for backtest result integration"""
    
    # Core settings
    enabled: bool = False  # Disabled by default for minimal coupling
    validation_enabled: bool = True
    
    # Quality filters
    min_backtest_trades: int = 10
    max_validation_age_days: int = 30
    validation_sample_ratio: float = 0.1  # Sample 10% of validation data
    
    # Performance comparison
    performance_comparison_window: int = 100  # Episodes for comparison
    
    # Data management
    store_validation_history: bool = True
    max_validation_history_size: int = 1000
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not (0.0 <= self.validation_sample_ratio <= 1.0):
            raise ValueError("validation_sample_ratio must be between 0.0 and 1.0")
            
        if self.max_validation_age_days <= 0:
            raise ValueError("max_validation_age_days must be positive")
            
        if self.min_backtest_trades <= 0:
            raise ValueError("min_backtest_trades must be positive")


@dataclass
class ValidationDataPoint:
    """Single validation data point from backtest"""
    market_state: MarketState
    action: TradeAction
    actual_outcome: float  # Actual return achieved
    confidence: float  # Confidence in the data point (0.0 to 1.0)
    timestamp: datetime
    source: str = "backtest"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "market_state": asdict(self.market_state),
            "action": self.action.value if hasattr(self.action, 'value') else str(self.action),
            "actual_outcome": self.actual_outcome,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class ModelPerformanceComparison:
    """Comparison between backtest and live model performance"""
    backtest_metrics: Dict[str, float]
    live_metrics: Dict[str, float]
    comparison_window: int
    created_at: datetime
    deviations: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Calculate deviations after initialization"""
        if self.deviations is None:
            self.deviations = self.calculate_deviations()
    
    def calculate_deviations(self) -> Dict[str, float]:
        """Calculate percentage deviations between backtest and live metrics"""
        deviations = {}
        
        for metric, backtest_value in self.backtest_metrics.items():
            if metric in self.live_metrics and backtest_value != 0:
                live_value = self.live_metrics[metric]
                deviation = (live_value - backtest_value) / backtest_value
                deviations[metric] = deviation
                
        return deviations


class BacktestResultIntegrator:
    """
    Lightweight interface to feed backtest results to learning system as validation data.
    
    Provides minimal coupling while enabling model validation against historical data.
    Learning system dependency is optional - integration gracefully degrades when unavailable.
    """
    
    def __init__(self, config: BacktestIntegrationConfig, learning_engine: Optional[Any] = None):
        """
        Initialize backtest result integrator
        
        Args:
            config: Integration configuration
            learning_engine: Optional continuous learning engine for feeding validation data
        """
        self.config = config
        self.learning_engine = learning_engine
        self.status = IntegrationStatus.DISABLED if not config.enabled else IntegrationStatus.INITIALIZED
        
        # Tracking and metrics
        self.validation_history: List[ModelPerformanceComparison] = []
        self.validation_data_points_processed = 0
        self.performance_comparisons_created = 0
        self.last_integration_time: Optional[datetime] = None
        
        logger.info("BacktestResultIntegrator initialized", 
                   enabled=config.enabled, 
                   status=self.status.value)
    
    def is_enabled(self) -> bool:
        """Check if integration is enabled"""
        return self.config.enabled and self.status != IntegrationStatus.DISABLED
    
    def can_feed_to_learning_system(self) -> bool:
        """Check if can feed data to learning system"""
        return self.learning_engine is not None and self.is_enabled()
    
    def set_status(self, status: IntegrationStatus) -> None:
        """Set integration status"""
        self.status = status
        logger.info("Integration status updated", status=status.value)
    
    async def process_backtest_result(self, backtest_result: BacktestResult) -> bool:
        """
        Process backtest result and integrate with learning system
        
        Args:
            backtest_result: Backtest result to process
            
        Returns:
            True if processing successful, False otherwise
        """
        if not self.is_enabled():
            logger.debug("Integration disabled, skipping backtest result processing")
            return False
            
        try:
            logger.info("Processing backtest result", 
                       strategy=backtest_result.strategy_name,
                       trades=backtest_result.total_trades)
            
            # Validate backtest result
            self._validate_backtest_result(backtest_result)
            
            # Convert to validation data
            validation_data = await self.convert_backtest_to_validation_data(backtest_result)
            
            # Filter by quality if validation enabled
            if self.config.validation_enabled:
                validation_data = self.filter_validation_data_by_quality(validation_data)
            
            # Sample data based on configuration
            sampled_data = self.sample_validation_data(validation_data)
            
            # Feed to learning system if available
            if self.can_feed_to_learning_system():
                success = await self.feed_validation_data_to_learning_system(sampled_data)
                if not success:
                    # If feeding fails, we should still continue with the rest of the process
                    logger.warning("Failed to feed validation data to learning system, continuing")
            
            # Compare performance with live model
            if self.can_feed_to_learning_system():
                performance_comparison = await self.compare_model_performance(backtest_result)
                self._store_performance_comparison(performance_comparison)
            
            self.validation_data_points_processed += len(sampled_data)
            self.last_integration_time = datetime.now()
            self.set_status(IntegrationStatus.ACTIVE)
            
            logger.info("Backtest result processed successfully",
                       validation_points=len(sampled_data),
                       strategy=backtest_result.strategy_name)
            
            return True
            
        except Exception as e:
            logger.error("Error processing backtest result", error=str(e))
            self.set_status(IntegrationStatus.ERROR)
            raise
    
    def _validate_backtest_result(self, backtest_result: BacktestResult) -> None:
        """Validate backtest result meets minimum requirements"""
        if backtest_result.total_trades < self.config.min_backtest_trades:
            raise BacktestValidationError(
                f"Insufficient trades: {backtest_result.total_trades} < {self.config.min_backtest_trades}"
            )
    
    async def convert_backtest_to_validation_data(self, backtest_result: BacktestResult) -> List[ValidationDataPoint]:
        """
        Convert backtest result to validation data points
        
        Args:
            backtest_result: Backtest result to convert
            
        Returns:
            List of validation data points
        """
        validation_data = []
        
        try:
            for trade in backtest_result.trade_history:
                # Parse trade data
                timestamp = datetime.fromisoformat(trade["timestamp"]) if isinstance(trade["timestamp"], str) else trade["timestamp"]
                action = self._parse_trade_action(trade["action"])
                price = trade.get("price", 0.0)
                trade_return = trade.get("return", 0.0)
                
                # Create dummy token for backtest data
                dummy_token = DiscoveredToken(
                    address=f"backtest_{backtest_result.strategy_name}",
                    chain=Chain.SOLANA,  # Default to Solana for backtest
                    symbol="BACKTEST",
                    name=f"Backtest Token {backtest_result.strategy_name}",
                    discovered_at=timestamp,
                    discovery_source="backtest",
                    price_usd=price,
                    volume_24h=trade.get("volume", 0.0)
                )
                
                # Create market state from trade data
                market_state = MarketState(
                    token=dummy_token,
                    price_usd=price,
                    price_change_24h=trade.get("price_change_24h", 0.0),
                    volume_24h=trade.get("volume", 0.0),
                    timestamp=timestamp
                )
                
                # Calculate confidence based on trade characteristics
                confidence = self._calculate_confidence(trade, backtest_result)
                
                # Create validation data point
                data_point = ValidationDataPoint(
                    market_state=market_state,
                    action=action,
                    actual_outcome=trade_return,
                    confidence=confidence,
                    timestamp=timestamp,
                    source="backtest",
                    metadata={
                        "strategy_name": backtest_result.strategy_name,
                        "backtest_metrics": {
                            "sharpe_ratio": backtest_result.sharpe_ratio,
                            "win_rate": backtest_result.win_rate
                        }
                    }
                )
                
                validation_data.append(data_point)
                
        except Exception as e:
            logger.error("Error converting backtest to validation data", error=str(e))
            raise
        
        logger.debug("Converted backtest to validation data", 
                    points=len(validation_data),
                    strategy=backtest_result.strategy_name)
        
        return validation_data
    
    def _parse_trade_action(self, action_str: str) -> TradeAction:
        """Parse trade action from string"""
        action_str = action_str.upper()
        if action_str in ["BUY", "LONG"]:
            return TradeAction.BUY
        elif action_str in ["SELL", "SHORT"]:
            return TradeAction.SELL
        else:
            return TradeAction.HOLD
    
    def _calculate_confidence(self, trade: Dict[str, Any], backtest_result: BacktestResult) -> float:
        """Calculate confidence score for a trade based on various factors"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on backtest performance
        if backtest_result.sharpe_ratio > 1.0:
            confidence += 0.2
        if backtest_result.win_rate > 0.6:
            confidence += 0.2
        
        # Adjust based on trade characteristics
        trade_return = abs(trade.get("return", 0.0))
        if trade_return > 0.1:  # High return trades are less reliable
            confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
    
    def filter_validation_data_by_quality(self, validation_data: List[ValidationDataPoint]) -> List[ValidationDataPoint]:
        """Filter validation data by quality criteria"""
        filtered_data = []
        
        for data_point in validation_data:
            # Filter by confidence threshold
            if data_point.confidence < 0.5:
                continue
                
            # Filter by age
            age_days = (datetime.now() - data_point.timestamp).days
            if age_days > self.config.max_validation_age_days:
                continue
                
            # Filter by outcome reasonableness (avoid extreme outliers)
            if abs(data_point.actual_outcome) > 1.0:  # 100% return is suspicious
                continue
                
            filtered_data.append(data_point)
        
        logger.debug("Filtered validation data by quality",
                    original=len(validation_data),
                    filtered=len(filtered_data))
        
        return filtered_data
    
    def sample_validation_data(self, validation_data: List[ValidationDataPoint]) -> List[ValidationDataPoint]:
        """Sample validation data based on configuration"""
        if self.config.validation_sample_ratio >= 1.0:
            return validation_data
            
        sample_size = max(1, int(len(validation_data) * self.config.validation_sample_ratio))
        sampled_data = random.sample(validation_data, min(sample_size, len(validation_data)))
        
        logger.debug("Sampled validation data",
                    original=len(validation_data),
                    sampled=len(sampled_data),
                    ratio=self.config.validation_sample_ratio)
        
        return sampled_data
    
    async def feed_validation_data_to_learning_system(self, validation_data: List[ValidationDataPoint]) -> bool:
        """
        Feed validation data to the continuous learning system
        
        Args:
            validation_data: Validation data points to feed
            
        Returns:
            True if successful, False otherwise
        """
        if not self.can_feed_to_learning_system():
            logger.debug("Cannot feed to learning system - engine unavailable or integration disabled")
            return False
            
        try:
            # Convert validation data to format expected by learning engine
            learning_data = self._convert_to_learning_format(validation_data)
            
            # Feed to learning engine
            await self.learning_engine.add_validation_data(learning_data)
            
            logger.info("Fed validation data to learning system",
                       data_points=len(validation_data))
            
            return True
            
        except Exception as e:
            logger.error("Error feeding validation data to learning system", error=str(e))
            return False
    
    def _convert_to_learning_format(self, validation_data: List[ValidationDataPoint]) -> List[Any]:
        """Convert validation data to format expected by learning engine"""
        # Convert to Experience objects or similar format expected by learning engine
        learning_data = []
        
        for data_point in validation_data:
            # Create experience-like object for learning engine
            experience_data = {
                "state": data_point.market_state,
                "action": data_point.action,
                "reward": data_point.actual_outcome,
                "confidence": data_point.confidence,
                "timestamp": data_point.timestamp,
                "source": data_point.source
            }
            learning_data.append(experience_data)
            
        return learning_data
    
    async def compare_model_performance(self, backtest_result: BacktestResult) -> ModelPerformanceComparison:
        """
        Compare backtest performance with live model performance
        
        Args:
            backtest_result: Backtest result to compare
            
        Returns:
            Performance comparison object
        """
        # Extract backtest metrics
        backtest_metrics = {
            "total_return": backtest_result.total_return,
            "sharpe_ratio": backtest_result.sharpe_ratio,
            "win_rate": backtest_result.win_rate,
            "max_drawdown": backtest_result.max_drawdown,
            "profit_factor": backtest_result.profit_factor
        }
        
        # Get live model metrics from learning engine
        live_metrics = {}
        if self.learning_engine:
            try:
                live_metrics = await self.learning_engine.get_performance_metrics(
                    window=self.config.performance_comparison_window
                )
            except Exception as e:
                logger.warning("Could not get live metrics", error=str(e))
                # Use default metrics if unavailable
                live_metrics = {metric: 0.0 for metric in backtest_metrics.keys()}
        
        comparison = ModelPerformanceComparison(
            backtest_metrics=backtest_metrics,
            live_metrics=live_metrics,
            comparison_window=self.config.performance_comparison_window,
            created_at=datetime.now()
        )
        
        logger.info("Created model performance comparison",
                   backtest_sharpe=backtest_metrics.get("sharpe_ratio", 0.0),
                   live_sharpe=live_metrics.get("sharpe_ratio", 0.0))
        
        return comparison
    
    def _store_performance_comparison(self, comparison: ModelPerformanceComparison) -> None:
        """Store performance comparison in history"""
        if self.config.store_validation_history:
            self.validation_history.append(comparison)
            
            # Limit history size
            if len(self.validation_history) > self.config.max_validation_history_size:
                self.validation_history = self.validation_history[-self.config.max_validation_history_size:]
            
            self.performance_comparisons_created += 1
    
    def get_performance_tracking_history(self) -> List[ModelPerformanceComparison]:
        """Get performance tracking history"""
        return self.validation_history.copy()
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration metrics and statistics"""
        return {
            "status": self.status.value,
            "enabled": self.is_enabled(),
            "can_feed_to_learning_system": self.can_feed_to_learning_system(),
            "validation_data_points_processed": self.validation_data_points_processed,
            "performance_comparisons_created": self.performance_comparisons_created,
            "validation_history_size": len(self.validation_history),
            "last_integration_time": self.last_integration_time.isoformat() if self.last_integration_time else None,
            "config": asdict(self.config)
        }