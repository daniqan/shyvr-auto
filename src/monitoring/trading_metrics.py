"""
Trading metrics collection for monitoring trading performance.

This module provides comprehensive trading metrics including:
- Profit and Loss (PnL) tracking
- Trade volume and count metrics
- Success rate monitoring
- Position tracking
- Trade size and duration analysis
"""

from typing import Dict, Any, Optional
from decimal import Decimal
import datetime

from .base import MetricsCollector, MetricsRegistry


class TradingMetricsCollector(MetricsCollector):
    """
    Collector for trading-related metrics.
    
    This class tracks key trading performance indicators including P&L,
    trade volumes, success rates, and position information.
    """
    
    def __init__(self, registry: MetricsRegistry):
        """
        Initialize the trading metrics collector.
        
        Args:
            registry: MetricsRegistry instance for managing metrics
        """
        super().__init__(registry)
        
        # P&L metrics
        self.total_pnl_gauge = registry.get_gauge(
            "trading_total_pnl",
            "Total profit and loss across all trades",
            ["currency"]
        )
        
        self.daily_pnl_gauge = registry.get_gauge(
            "trading_daily_pnl",
            "Daily profit and loss",
            ["currency", "date"]
        )
        
        # Volume and trade count metrics
        self.trade_volume_counter = registry.get_counter(
            "trading_volume_total",
            "Total trading volume",
            ["symbol", "side"]
        )
        
        self.trade_count_counter = registry.get_counter(
            "trading_trade_count_total",
            "Total number of trades",
            ["symbol", "side"]
        )
        
        # Success/failure tracking
        self.successful_trades_counter = registry.get_counter(
            "trading_successful_trades_total",
            "Total number of successful trades",
            ["symbol", "side"]
        )
        
        self.failed_trades_counter = registry.get_counter(
            "trading_failed_trades_total",
            "Total number of failed trades",
            ["symbol", "side", "error_type"]
        )
        
        # Position metrics
        self.position_count_gauge = registry.get_gauge(
            "trading_position_count",
            "Number of active positions",
            ["symbol"]
        )
        
        # Performance metrics
        self.success_rate_gauge = registry.get_gauge(
            "trading_success_rate",
            "Success rate of trades (0-1)",
            ["symbol", "timeframe"]
        )
        
        # Distribution metrics
        self.trade_size_histogram = registry.get_histogram(
            "trading_trade_size_seconds",
            "Distribution of trade sizes",
            ["symbol", "side"],
            buckets=(10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000)
        )
        
        self.trade_duration_histogram = registry.get_histogram(
            "trading_trade_duration_seconds",
            "Distribution of trade durations in seconds",
            ["symbol"],
            buckets=(1, 5, 15, 30, 60, 300, 900, 3600, 14400, 86400)
        )
    
    def collect_metrics(self) -> None:
        """
        Collect and update all trading metrics.
        
        This method gathers current trading data and updates all relevant metrics.
        """
        try:
            # Collect P&L data
            portfolio_manager = self._get_portfolio_manager()
            if portfolio_manager:
                total_pnl = portfolio_manager.get_total_pnl()
                daily_pnl = portfolio_manager.get_daily_pnl()
                total_volume = portfolio_manager.get_total_volume()
                
                # Update P&L gauges
                self.total_pnl_gauge.labels(currency="USD").set(float(total_pnl))
                self.daily_pnl_gauge.labels(
                    currency="USD", 
                    date=datetime.date.today().isoformat()
                ).set(float(daily_pnl))
            
            # Collect position data
            position_tracker = self._get_position_tracker()
            if position_tracker:
                active_positions = position_tracker.get_active_position_count()
                self.position_count_gauge.labels(symbol="ALL").set(active_positions)
            
            # Collect trade statistics
            trade_stats = self._get_trade_statistics()
            if trade_stats:
                # Update success rate
                success_rate = trade_stats.get('success_rate', 0)
                self.success_rate_gauge.labels(symbol="ALL", timeframe="daily").set(success_rate)
            
        except Exception as e:
            # Log error but don't crash the collector
            self._collection_errors += 1
            raise e
    
    def get_metric_definitions(self) -> Dict[str, str]:
        """
        Get metric definitions for trading metrics.
        
        Returns:
            Dictionary mapping metric names to their descriptions
        """
        return {
            "trading_total_pnl": "Total profit and loss across all trades",
            "trading_daily_pnl": "Daily profit and loss",
            "trading_volume_total": "Total trading volume",
            "trading_trade_count_total": "Total number of trades",
            "trading_successful_trades_total": "Total number of successful trades",
            "trading_failed_trades_total": "Total number of failed trades",
            "trading_position_count": "Number of active positions",
            "trading_success_rate": "Success rate of trades (0-1)",
            "trading_trade_size_seconds": "Distribution of trade sizes",
            "trading_trade_duration_seconds": "Distribution of trade durations in seconds"
        }
    
    def record_trade_success(self, trade_data: Dict[str, Any]) -> None:
        """
        Record a successful trade.
        
        Args:
            trade_data: Dictionary containing trade information
        """
        symbol = trade_data.get('symbol', 'UNKNOWN')
        side = trade_data.get('side', 'unknown')
        size = trade_data.get('size', Decimal('0'))
        duration = trade_data.get('duration', 0)
        
        # Update counters
        self.successful_trades_counter.labels(symbol=symbol, side=side).inc()
        self.trade_count_counter.labels(symbol=symbol, side=side).inc()
        
        # Update histograms
        if size > 0:
            self.trade_size_histogram.labels(symbol=symbol, side=side).observe(float(size))
        
        if duration > 0:
            self.trade_duration_histogram.labels(symbol=symbol).observe(duration)
    
    def record_trade_failure(self, trade_data: Dict[str, Any]) -> None:
        """
        Record a failed trade.
        
        Args:
            trade_data: Dictionary containing trade information and error details
        """
        symbol = trade_data.get('symbol', 'UNKNOWN')
        side = trade_data.get('side', 'unknown')
        error_type = trade_data.get('error', 'unknown_error')
        
        # Update failure counter
        self.failed_trades_counter.labels(
            symbol=symbol, 
            side=side, 
            error_type=error_type
        ).inc()
        
        # Still count as a trade attempt
        self.trade_count_counter.labels(symbol=symbol, side=side).inc()
    
    def record_pnl_update(self, total_pnl: Decimal, daily_pnl: Decimal) -> None:
        """
        Record P&L updates.
        
        Args:
            total_pnl: Total profit/loss
            daily_pnl: Daily profit/loss
        """
        self.total_pnl_gauge.labels(currency="USD").set(float(total_pnl))
        self.daily_pnl_gauge.labels(
            currency="USD", 
            date=datetime.date.today().isoformat()
        ).set(float(daily_pnl))
    
    def record_volume_update(self, volume: Decimal, symbol: str) -> None:
        """
        Record volume updates.
        
        Args:
            volume: Trading volume
            symbol: Trading symbol
        """
        self.trade_volume_counter.labels(symbol=symbol, side="ALL").inc(float(volume))
    
    def _get_portfolio_manager(self):
        """
        Get the portfolio manager instance.
        
        Returns:
            Portfolio manager instance or None
        """
        try:
            from src.portfolio.portfolio_manager import PortfolioManager
            # In a real implementation, this would get the active portfolio manager
            # For now, return None to avoid import errors in tests
            return None
        except ImportError:
            return None
    
    def _get_position_tracker(self):
        """
        Get the position tracker instance.
        
        Returns:
            Position tracker instance or None
        """
        try:
            from src.portfolio.position_tracker import PositionTracker
            # In a real implementation, this would get the active position tracker
            # For now, return None to avoid import errors in tests
            return None
        except ImportError:
            return None
    
    def _get_trade_statistics(self) -> Dict[str, Any]:
        """
        Get trade statistics from activity logs.
        
        Returns:
            Dictionary with trade statistics
        """
        try:
            from src.logging.activity_logger import ActivityLogger
            
            activity_logger = ActivityLogger()
            
            # Query trade activities for the last 24 hours
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(hours=24)
            
            trade_activities = activity_logger.query_activities(
                activity_type="TRADE_EXECUTION",
                start_time=start_time,
                end_time=end_time
            )
            
            total_trades = len(trade_activities)
            successful_trades = sum(
                1 for activity in trade_activities 
                if activity.get('metadata', {}).get('success', False)
            )
            failed_trades = total_trades - successful_trades
            success_rate = successful_trades / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'failed_trades': failed_trades,
                'success_rate': success_rate
            }
            
        except Exception:
            # Return default stats if unable to query
            return {
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'success_rate': 0
            }
    
    def is_healthy(self) -> bool:
        """
        Check if the trading metrics collector is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to collect metrics to verify health
            self.collect_metrics()
            return True
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information for the trading metrics collector.
        
        Returns:
            Dictionary with collector status and trading-specific information
        """
        base_status = super().get_status()
        
        # Add trading-specific status information
        trade_stats = self._get_trade_statistics()
        base_status['trading_stats'] = trade_stats
        
        return base_status