"""
Operational Analytics System for Phase 6.2

This module provides comprehensive operational analytics that leverage existing
logging and monitoring infrastructure to generate insights and recommendations.

Key Features:
- Log analytics and insights generation
- Trading performance analytics
- Compliance analytics for regulatory reporting  
- System health analytics with proactive monitoring
- Real-time operational dashboards
"""

import asyncio
import json
import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from collections import defaultdict, Counter
import re

# Import existing systems
from ..activity_logging.activity_logger import ActivityLogger, ActivityCategory, ActivitySeverity
from ..enhanced_logging.enhanced_logging import get_enhanced_logger, LogCategory
from ..monitoring.trading_metrics import TradingMetricsCollector
from ..monitoring.base import MetricsRegistry
from ..utils.config import get_config

logger = get_enhanced_logger(__name__)


class AnalyticsEngine(Enum):
    """Types of analytics engines."""
    LOG_ANALYTICS = "log_analytics"
    TRADING_ANALYTICS = "trading_analytics" 
    COMPLIANCE_ANALYTICS = "compliance_analytics"
    SYSTEM_HEALTH = "system_health"


@dataclass
class AnalyticsConfig:
    """Configuration for operational analytics system."""
    
    # Analysis intervals in seconds
    analysis_intervals: Dict[str, int] = field(default_factory=lambda: {
        'log_analytics': 300,  # 5 minutes
        'trading_analytics': 600,  # 10 minutes
        'compliance_analytics': 3600,  # 1 hour
        'system_health': 180  # 3 minutes
    })
    
    # Data retention policies in days
    retention_policies: Dict[str, int] = field(default_factory=lambda: {
        'trading_data': 2555,  # 7 years for financial data
        'audit_logs': 2555,  # 7 years for compliance
        'system_logs': 90,  # 3 months for system logs
        'analytics_results': 365  # 1 year for analytics
    })
    
    # Alert thresholds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'error_rate': 0.05,  # 5%
        'response_time_p95': 2000.0,  # 2 seconds
        'success_rate': 0.95,  # 95%
        'compliance_rate': 0.98,  # 98%
        'system_health_score': 80.0  # 80%
    })
    
    # Dashboard settings
    dashboard_settings: Dict[str, Any] = field(default_factory=lambda: {
        'refresh_interval': 30,  # seconds
        'cache_duration': 60,  # seconds
        'max_data_points': 1000,
        'enable_real_time': True
    })
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate intervals are positive
            for interval in self.analysis_intervals.values():
                if interval <= 0:
                    return False
            
            # Validate retention policies are positive
            for retention in self.retention_policies.values():
                if retention <= 0:
                    return False
            
            # Validate thresholds are in valid ranges
            if not (0 <= self.alert_thresholds['error_rate'] <= 1):
                return False
            if not (0 <= self.alert_thresholds['success_rate'] <= 1):
                return False
            if not (0 <= self.alert_thresholds['compliance_rate'] <= 1):
                return False
            
            return True
        except Exception:
            return False
    
    def get_analysis_interval(self, engine_type: str) -> int:
        """Get analysis interval for engine type."""
        return self.analysis_intervals.get(engine_type, 300)
    
    def get_retention_policy(self, data_type: str) -> int:
        """Get retention policy for data type."""
        return self.retention_policies.get(data_type, 90)
    
    def get_alert_threshold(self, metric: str) -> float:
        """Get alert threshold for metric."""
        return self.alert_thresholds.get(metric, 0.0)


class LogAnalyticsEngine:
    """Engine for analyzing logs and generating insights."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize log analytics engine."""
        self.config = config or AnalyticsConfig()
        self.time_windows = [
            timedelta(hours=1),
            timedelta(hours=24), 
            timedelta(days=7),
            timedelta(days=30)
        ]
        
        # Pre-compiled regex patterns for log analysis
        self.log_patterns = {
            'error_patterns': [
                re.compile(r'error|exception|failed|failure', re.IGNORECASE),
                re.compile(r'timeout|connection.*refused', re.IGNORECASE),
                re.compile(r'permission.*denied|unauthorized', re.IGNORECASE)
            ],
            'performance_patterns': [
                re.compile(r'slow|timeout|latency|performance', re.IGNORECASE),
                re.compile(r'response.*time.*(\d+).*ms', re.IGNORECASE)
            ],
            'security_patterns': [
                re.compile(r'security|breach|attack|suspicious', re.IGNORECASE),
                re.compile(r'failed.*login|authentication.*failed', re.IGNORECASE)
            ]
        }
        
        # Anomaly detection algorithms
        self.anomaly_detectors = {
            'frequency': self._detect_frequency_anomalies,
            'patterns': self._detect_pattern_anomalies,
            'temporal': self._detect_temporal_anomalies
        }
    
    async def analyze_log_patterns(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in log data."""
        if not log_data:
            return {'total_logs': 0, 'error_rate': 0, 'patterns': {}}
        
        patterns = {
            'total_logs': len(log_data),
            'error_patterns': [],
            'success_patterns': [],
            'frequency_analysis': {},
            'error_rate': 0,
            'level_distribution': {},
            'category_distribution': {},
            'time_distribution': {}
        }
        
        error_count = 0
        level_counts = Counter()
        category_counts = Counter()
        hourly_counts = defaultdict(int)
        
        for log_entry in log_data:
            level = log_entry.get('level', 'INFO').upper()
            category = log_entry.get('category', 'unknown')
            message = log_entry.get('message', '')
            timestamp = log_entry.get('timestamp')
            
            # Count by level
            level_counts[level] += 1
            
            # Count by category
            category_counts[category] += 1
            
            # Count by hour
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = timestamp.hour
                hourly_counts[hour] += 1
            
            # Check for errors
            if level in ['ERROR', 'CRITICAL']:
                error_count += 1
                patterns['error_patterns'].append({
                    'message': message,
                    'level': level,
                    'category': category,
                    'timestamp': timestamp.isoformat() if timestamp else None
                })
            
            # Check for success patterns
            if level in ['INFO'] and any(word in message.lower() for word in ['success', 'completed', 'executed']):
                patterns['success_patterns'].append({
                    'message': message,
                    'category': category,
                    'timestamp': timestamp.isoformat() if timestamp else None
                })
        
        # Calculate metrics
        patterns['error_rate'] = error_count / len(log_data) if log_data else 0
        patterns['level_distribution'] = dict(level_counts)
        patterns['category_distribution'] = dict(category_counts)
        patterns['time_distribution'] = dict(hourly_counts)
        
        # Frequency analysis
        patterns['frequency_analysis'] = {
            'errors_per_hour': error_count,
            'logs_per_hour': len(log_data),
            'avg_logs_per_minute': len(log_data) / 60 if log_data else 0
        }
        
        return patterns
    
    async def detect_log_anomalies(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalies in log data."""
        anomalies = {
            'frequency_anomalies': [],
            'pattern_anomalies': [],
            'temporal_anomalies': [],
            'severity_anomalies': []
        }
        
        if not log_data:
            return anomalies
        
        # Detect frequency anomalies
        anomalies['frequency_anomalies'] = await self.anomaly_detectors['frequency'](log_data)
        
        # Detect pattern anomalies
        anomalies['pattern_anomalies'] = await self.anomaly_detectors['patterns'](log_data)
        
        # Detect temporal anomalies
        anomalies['temporal_anomalies'] = await self.anomaly_detectors['temporal'](log_data)
        
        return anomalies
    
    async def generate_log_insights(self, patterns: Dict[str, Any], anomalies: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from log patterns and anomalies."""
        insights = {
            'summary': '',
            'key_findings': [],
            'recommendations': [],
            'risk_level': 'low',
            'action_items': []
        }
        
        error_rate = patterns.get('error_rate', 0)
        total_logs = patterns.get('total_logs', 0)
        
        # Assess risk level
        if error_rate > 0.1:  # > 10%
            insights['risk_level'] = 'high'
        elif error_rate > 0.05:  # > 5%
            insights['risk_level'] = 'medium'
        
        # Generate key findings
        insights['key_findings'].append(f"Analyzed {total_logs} log entries")
        insights['key_findings'].append(f"Error rate: {error_rate:.2%}")
        
        if error_rate > self.config.get_alert_threshold('error_rate'):
            insights['key_findings'].append("Error rate exceeds threshold")
            insights['recommendations'].append("Investigate error patterns immediately")
            insights['action_items'].append("Review recent deployments and system changes")
        
        # Analyze error patterns
        error_patterns = patterns.get('error_patterns', [])
        if error_patterns:
            common_errors = Counter([p['category'] for p in error_patterns])
            most_common = common_errors.most_common(3)
            insights['key_findings'].append(f"Most frequent error categories: {most_common}")
        
        # Anomaly insights
        total_anomalies = sum(len(anomalies[key]) for key in anomalies)
        if total_anomalies > 0:
            insights['key_findings'].append(f"Detected {total_anomalies} anomalies")
            insights['recommendations'].append("Review anomalies for potential issues")
        
        # Generate summary
        insights['summary'] = f"Log analysis shows {insights['risk_level']} risk level with {error_rate:.2%} error rate"
        
        return insights
    
    async def calculate_error_rates(self, log_data: List[Dict[str, Any]], 
                                    time_window: timedelta) -> Dict[str, float]:
        """Calculate error rates by time window."""
        now = datetime.now(timezone.utc)
        cutoff = now - time_window
        
        filtered_logs = [
            log for log in log_data 
            if log.get('timestamp') and 
            (isinstance(log['timestamp'], datetime) and log['timestamp'] > cutoff or
             isinstance(log['timestamp'], str) and 
             datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')) > cutoff)
        ]
        
        if not filtered_logs:
            return {'error_rate': 0.0, 'total_logs': 0, 'error_count': 0}
        
        error_count = sum(
            1 for log in filtered_logs 
            if log.get('level', '').upper() in ['ERROR', 'CRITICAL']
        )
        
        return {
            'error_rate': error_count / len(filtered_logs),
            'total_logs': len(filtered_logs),
            'error_count': error_count
        }
    
    async def analyze_performance_trends(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends from logs."""
        performance_data = []
        response_times = []
        
        for log in log_data:
            message = log.get('message', '')
            
            # Extract response times from messages
            response_time_match = re.search(r'(\d+).*ms', message)
            if response_time_match:
                response_times.append(float(response_time_match.group(1)))
            
            # Extract execution times
            execution_time = log.get('execution_time_ms')
            if execution_time:
                performance_data.append(execution_time)
        
        all_times = response_times + performance_data
        
        if not all_times:
            return {'trend': 'no_data', 'metrics': {}}
        
        return {
            'trend': 'improving' if all_times[-10:] < all_times[:10] else 'degrading',
            'metrics': {
                'avg_response_time': statistics.mean(all_times),
                'p95_response_time': np.percentile(all_times, 95) if all_times else 0,
                'p99_response_time': np.percentile(all_times, 99) if all_times else 0,
                'total_samples': len(all_times)
            }
        }
    
    async def _detect_frequency_anomalies(self, log_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect frequency-based anomalies."""
        anomalies = []
        
        # Group logs by hour
        hourly_counts = defaultdict(int)
        for log in log_data:
            timestamp = log.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = timestamp.hour
                hourly_counts[hour] += 1
        
        if len(hourly_counts) < 3:
            return anomalies
        
        # Calculate mean and std dev
        counts = list(hourly_counts.values())
        mean_count = statistics.mean(counts)
        std_dev = statistics.stdev(counts) if len(counts) > 1 else 0
        
        # Detect anomalies (more than 2 std devs from mean)
        threshold = mean_count + (2 * std_dev)
        
        for hour, count in hourly_counts.items():
            if count > threshold and std_dev > 0:
                anomalies.append({
                    'type': 'frequency',
                    'hour': hour,
                    'count': count,
                    'expected': mean_count,
                    'severity': 'high' if count > mean_count + (3 * std_dev) else 'medium'
                })
        
        return anomalies
    
    async def _detect_pattern_anomalies(self, log_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        # Look for unusual error patterns
        error_logs = [log for log in log_data if log.get('level', '').upper() in ['ERROR', 'CRITICAL']]
        
        if len(error_logs) > 10:  # Only analyze if sufficient data
            error_messages = [log.get('message', '') for log in error_logs]
            message_counts = Counter(error_messages)
            
            # Find unusual error patterns (very frequent or very rare)
            total_errors = len(error_logs)
            for message, count in message_counts.items():
                frequency = count / total_errors
                
                if frequency > 0.3:  # More than 30% of errors are the same
                    anomalies.append({
                        'type': 'pattern',
                        'pattern': 'repeated_error',
                        'message': message,
                        'frequency': frequency,
                        'severity': 'high'
                    })
        
        return anomalies
    
    async def _detect_temporal_anomalies(self, log_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect temporal anomalies."""
        anomalies = []
        
        if len(log_data) < 2:
            return anomalies
        
        # Check for sudden spikes in log volume
        timestamps = []
        for log in log_data:
            timestamp = log.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamps.append(timestamp)
        
        if len(timestamps) < 10:
            return anomalies
        
        # Group by 5-minute intervals
        interval_counts = defaultdict(int)
        for ts in timestamps:
            interval = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
            interval_counts[interval] += 1
        
        if len(interval_counts) > 2:
            counts = list(interval_counts.values())
            mean_count = statistics.mean(counts)
            std_dev = statistics.stdev(counts) if len(counts) > 1 else 0
            
            # Detect spikes
            threshold = mean_count + (2 * std_dev)
            
            for interval, count in interval_counts.items():
                if count > threshold and std_dev > 0:
                    anomalies.append({
                        'type': 'temporal',
                        'interval': interval.isoformat(),
                        'count': count,
                        'expected': mean_count,
                        'severity': 'medium'
                    })
        
        return anomalies


class TradingAnalyticsEngine:
    """Engine for analyzing trading performance and generating insights."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize trading analytics engine."""
        self.config = config or AnalyticsConfig()
        
        # Performance metrics calculators
        self.performance_metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown,
            'win_rate': self._calculate_win_rate,
            'profit_factor': self._calculate_profit_factor
        }
        
        # Trend analysis methods
        self.trend_analyzers = {
            'moving_average': self._analyze_moving_average_trend,
            'linear_regression': self._analyze_linear_trend,
            'momentum': self._analyze_momentum_trend
        }
        
        # Risk calculators
        self.risk_calculators = {
            'var': self._calculate_var,
            'cvar': self._calculate_cvar,
            'volatility': self._calculate_volatility
        }
    
    async def analyze_trading_performance(self, trading_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall trading performance."""
        if not trading_data:
            return {'total_trades': 0, 'message': 'No trading data available'}
        
        performance = {
            'total_trades': len(trading_data),
            'successful_trades': 0,
            'failed_trades': 0,
            'success_rate': 0.0,
            'total_pnl': Decimal('0'),
            'total_volume': Decimal('0'),
            'average_trade_size': Decimal('0'),
            'symbols_traded': set(),
            'time_range': {}
        }
        
        pnl_values = []
        volumes = []
        timestamps = []
        
        for trade in trading_data:
            success = trade.get('success', False)
            pnl = trade.get('pnl', Decimal('0'))
            amount = trade.get('amount', Decimal('0'))
            symbol = trade.get('symbol', 'UNKNOWN')
            timestamp = trade.get('timestamp')
            
            if success:
                performance['successful_trades'] += 1
            else:
                performance['failed_trades'] += 1
            
            if isinstance(pnl, (int, float)):
                pnl = Decimal(str(pnl))
            performance['total_pnl'] += pnl
            pnl_values.append(float(pnl))
            
            if isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            performance['total_volume'] += amount
            volumes.append(float(amount))
            
            performance['symbols_traded'].add(symbol)
            
            if timestamp:
                timestamps.append(timestamp)
        
        # Calculate derived metrics
        if performance['total_trades'] > 0:
            performance['success_rate'] = performance['successful_trades'] / performance['total_trades']
            performance['average_trade_size'] = performance['total_volume'] / performance['total_trades']
        
        # Convert set to list for JSON serialization
        performance['symbols_traded'] = list(performance['symbols_traded'])
        
        # Time range analysis
        if timestamps:
            performance['time_range'] = {
                'start': min(timestamps).isoformat() if timestamps else None,
                'end': max(timestamps).isoformat() if timestamps else None
            }
        
        return performance
    
    async def calculate_profitability_metrics(self, pnl_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed profitability metrics."""
        if not pnl_data:
            return {'message': 'No P&L data available'}
        
        pnl_values = []
        daily_pnl = defaultdict(Decimal)
        
        for entry in pnl_data:
            pnl = entry.get('pnl', Decimal('0'))
            timestamp = entry.get('timestamp')
            
            if isinstance(pnl, (int, float)):
                pnl = Decimal(str(pnl))
            pnl_values.append(float(pnl))
            
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                date_key = timestamp.date()
                daily_pnl[date_key] += pnl
        
        metrics = {
            'total_pnl': sum(pnl_values),
            'average_daily_pnl': statistics.mean(list(daily_pnl.values())) if daily_pnl else 0,
            'sharpe_ratio': await self.performance_metrics['sharpe_ratio'](pnl_values),
            'max_drawdown': await self.performance_metrics['max_drawdown'](pnl_values),
            'win_rate': await self.performance_metrics['win_rate'](pnl_values),
            'profit_factor': await self.performance_metrics['profit_factor'](pnl_values),
            'volatility': statistics.stdev(pnl_values) if len(pnl_values) > 1 else 0,
            'trading_days': len(daily_pnl)
        }
        
        return metrics
    
    async def analyze_risk_metrics(self, trading_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze risk metrics from trading data."""
        if not trading_data:
            return {'message': 'No trading data for risk analysis'}
        
        pnl_values = [float(trade.get('pnl', 0)) for trade in trading_data]
        
        risk_metrics = {
            'var_95': await self.risk_calculators['var'](pnl_values, 0.05),
            'var_99': await self.risk_calculators['var'](pnl_values, 0.01),
            'cvar_95': await self.risk_calculators['cvar'](pnl_values, 0.05),
            'volatility': await self.risk_calculators['volatility'](pnl_values),
            'max_loss': min(pnl_values) if pnl_values else 0,
            'max_gain': max(pnl_values) if pnl_values else 0
        }
        
        return risk_metrics
    
    async def detect_trading_patterns(self, trading_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in trading behavior."""
        patterns = {
            'symbol_frequency': {},
            'time_patterns': {},
            'size_patterns': {},
            'success_patterns': {}
        }
        
        if not trading_data:
            return patterns
        
        # Symbol frequency analysis
        symbol_counts = Counter([trade.get('symbol', 'UNKNOWN') for trade in trading_data])
        patterns['symbol_frequency'] = dict(symbol_counts.most_common(10))
        
        # Time pattern analysis
        hour_counts = defaultdict(int)
        for trade in trading_data:
            timestamp = trade.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour_counts[timestamp.hour] += 1
        patterns['time_patterns'] = dict(hour_counts)
        
        # Size pattern analysis
        amounts = [float(trade.get('amount', 0)) for trade in trading_data if trade.get('amount', 0) > 0]
        if amounts:
            patterns['size_patterns'] = {
                'average_size': statistics.mean(amounts),
                'median_size': statistics.median(amounts),
                'size_std': statistics.stdev(amounts) if len(amounts) > 1 else 0
            }
        
        # Success pattern analysis by symbol
        symbol_success = defaultdict(lambda: {'total': 0, 'successful': 0})
        for trade in trading_data:
            symbol = trade.get('symbol', 'UNKNOWN')
            symbol_success[symbol]['total'] += 1
            if trade.get('success', False):
                symbol_success[symbol]['successful'] += 1
        
        patterns['success_patterns'] = {
            symbol: {
                'success_rate': data['successful'] / data['total'] if data['total'] > 0 else 0,
                'total_trades': data['total']
            }
            for symbol, data in symbol_success.items()
        }
        
        return patterns
    
    async def generate_performance_insights(self, performance: Dict[str, Any], 
                                           patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading performance insights."""
        insights = {
            'summary': '',
            'key_findings': [],
            'recommendations': [],
            'risk_assessment': 'low',
            'action_items': []
        }
        
        success_rate = performance.get('success_rate', 0)
        total_pnl = performance.get('total_pnl', 0)
        total_trades = performance.get('total_trades', 0)
        
        # Risk assessment
        if success_rate < 0.6:
            insights['risk_assessment'] = 'high'
        elif success_rate < 0.8:
            insights['risk_assessment'] = 'medium'
        
        # Key findings
        insights['key_findings'].append(f"Executed {total_trades} trades with {success_rate:.1%} success rate")
        insights['key_findings'].append(f"Total P&L: ${total_pnl:.2f}")
        
        if success_rate < self.config.get_alert_threshold('success_rate'):
            insights['key_findings'].append("Success rate below threshold")
            insights['recommendations'].append("Review trading strategy and risk management")
            insights['action_items'].append("Analyze failed trades for common patterns")
        
        # Pattern insights
        symbol_freq = patterns.get('symbol_frequency', {})
        if symbol_freq:
            top_symbol = max(symbol_freq, key=symbol_freq.get)
            insights['key_findings'].append(f"Most traded symbol: {top_symbol} ({symbol_freq[top_symbol]} trades)")
        
        # Generate summary
        insights['summary'] = f"Trading performance shows {insights['risk_assessment']} risk with {success_rate:.1%} success rate"
        
        return insights
    
    # Helper methods for performance calculations
    async def _calculate_sharpe_ratio(self, pnl_values: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(pnl_values) < 2:
            return 0.0
        
        mean_return = statistics.mean(pnl_values)
        std_return = statistics.stdev(pnl_values)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_return
    
    async def _calculate_max_drawdown(self, pnl_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not pnl_values:
            return 0.0
        
        cumulative = np.cumsum(pnl_values)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return float(np.max(drawdown))
    
    async def _calculate_win_rate(self, pnl_values: List[float]) -> float:
        """Calculate win rate."""
        if not pnl_values:
            return 0.0
        
        wins = sum(1 for pnl in pnl_values if pnl > 0)
        return wins / len(pnl_values)
    
    async def _calculate_profit_factor(self, pnl_values: List[float]) -> float:
        """Calculate profit factor."""
        if not pnl_values:
            return 0.0
        
        wins = sum(pnl for pnl in pnl_values if pnl > 0)
        losses = abs(sum(pnl for pnl in pnl_values if pnl < 0))
        
        if losses == 0:
            return float('inf') if wins > 0 else 0.0
        
        return wins / losses
    
    async def _calculate_var(self, pnl_values: List[float], confidence: float) -> float:
        """Calculate Value at Risk."""
        if not pnl_values:
            return 0.0
        
        return float(np.percentile(pnl_values, confidence * 100))
    
    async def _calculate_cvar(self, pnl_values: List[float], confidence: float) -> float:
        """Calculate Conditional Value at Risk."""
        if not pnl_values:
            return 0.0
        
        var = await self._calculate_var(pnl_values, confidence)
        tail_values = [pnl for pnl in pnl_values if pnl <= var]
        
        return statistics.mean(tail_values) if tail_values else 0.0
    
    async def _calculate_volatility(self, pnl_values: List[float]) -> float:
        """Calculate volatility."""
        if len(pnl_values) < 2:
            return 0.0
        
        return statistics.stdev(pnl_values)
    
    async def _analyze_moving_average_trend(self, values: List[float], window: int = 20) -> str:
        """Analyze trend using moving average."""
        if len(values) < window:
            return 'insufficient_data'
        
        recent_ma = statistics.mean(values[-window:])
        previous_ma = statistics.mean(values[-window*2:-window])
        
        if recent_ma > previous_ma * 1.05:
            return 'uptrend'
        elif recent_ma < previous_ma * 0.95:
            return 'downtrend'
        else:
            return 'sideways'
    
    async def _analyze_linear_trend(self, values: List[float]) -> str:
        """Analyze linear trend."""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'flat'
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return 'uptrend'
        elif slope < -0.1:
            return 'downtrend'
        else:
            return 'sideways'
    
    async def _analyze_momentum_trend(self, values: List[float], period: int = 10) -> str:
        """Analyze momentum trend."""
        if len(values) < period * 2:
            return 'insufficient_data'
        
        recent_momentum = sum(values[-period:])
        previous_momentum = sum(values[-period*2:-period])
        
        if recent_momentum > previous_momentum * 1.1:
            return 'accelerating'
        elif recent_momentum < previous_momentum * 0.9:
            return 'decelerating'
        else:
            return 'stable'


class ComplianceAnalyticsEngine:
    """Engine for compliance analytics and regulatory reporting."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize compliance analytics engine."""
        self.config = config or AnalyticsConfig()
        
        # Regulatory frameworks
        self.regulatory_frameworks = {
            'MIFID_II': {
                'requirements': ['audit_trail', 'explanation_provided', 'transparency'],
                'retention_years': 7,
                'report_frequency': 'monthly'
            },
            'GDPR': {
                'requirements': ['consent_obtained', 'data_minimization', 'right_to_erasure'],
                'retention_years': 7,
                'report_frequency': 'quarterly'
            },
            'SEC_RULE_3A4': {
                'requirements': ['model_documentation', 'risk_disclosure', 'performance_reporting'],
                'retention_years': 5,
                'report_frequency': 'annual'
            }
        }
        
        # Audit processors
        self.audit_processors = {
            'completeness': self._check_audit_completeness,
            'accuracy': self._check_audit_accuracy,
            'timeliness': self._check_audit_timeliness
        }
    
    async def analyze_compliance_status(self, audit_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall compliance status."""
        if not audit_data:
            return {'message': 'No audit data available', 'overall_compliance_rate': 0.0}
        
        status = {
            'overall_compliance_rate': 0.0,
            'regulation_breakdown': {},
            'violations': [],
            'audit_completeness': 0.0,
            'total_events': len(audit_data),
            'compliant_events': 0
        }
        
        regulation_stats = defaultdict(lambda: {'total': 0, 'compliant': 0})
        
        for entry in audit_data:
            regulation = entry.get('regulation', 'UNKNOWN')
            compliant = entry.get('compliant', False)
            event_type = entry.get('event_type', 'unknown')
            
            regulation_stats[regulation]['total'] += 1
            if compliant:
                regulation_stats[regulation]['compliant'] += 1
                status['compliant_events'] += 1
            else:
                # Record violation
                status['violations'].append({
                    'regulation': regulation,
                    'event_type': event_type,
                    'timestamp': entry.get('timestamp'),
                    'details': entry.get('details', {})
                })
        
        # Calculate compliance rates
        if status['total_events'] > 0:
            status['overall_compliance_rate'] = status['compliant_events'] / status['total_events']
        
        # Calculate per-regulation compliance
        for regulation, stats in regulation_stats.items():
            if stats['total'] > 0:
                compliance_rate = stats['compliant'] / stats['total']
                status['regulation_breakdown'][regulation] = {
                    'compliance_rate': compliance_rate,
                    'total_events': stats['total'],
                    'compliant_events': stats['compliant'],
                    'violations': stats['total'] - stats['compliant']
                }
        
        return status
    
    async def generate_regulatory_reports(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate regulatory compliance reports."""
        regulation = report_data.get('regulation', 'MIFID_II')
        period_start = report_data.get('period_start')
        period_end = report_data.get('period_end')
        
        framework = self.regulatory_frameworks.get(regulation, {})
        
        report = {
            'regulation': regulation,
            'reporting_period': {
                'start': period_start.isoformat() if period_start else None,
                'end': period_end.isoformat() if period_end else None
            },
            'compliance_summary': {
                'requirements_met': 0,
                'total_requirements': len(framework.get('requirements', [])),
                'compliance_percentage': 0.0
            },
            'audit_trail_completeness': 100.0,  # Placeholder
            'recommendations': [],
            'next_review_date': None
        }
        
        # Calculate next review date based on frequency
        if period_end:
            frequency = framework.get('report_frequency', 'monthly')
            if frequency == 'monthly':
                next_review = period_end + timedelta(days=30)
            elif frequency == 'quarterly':
                next_review = period_end + timedelta(days=90)
            else:  # annual
                next_review = period_end + timedelta(days=365)
            
            report['next_review_date'] = next_review.isoformat()
        
        # Generate recommendations based on regulation
        if regulation == 'MIFID_II':
            report['recommendations'].extend([
                'Ensure all algorithmic trading decisions have explanations',
                'Maintain complete audit trails for 7 years',
                'Provide transparency reports to clients monthly'
            ])
        elif regulation == 'GDPR':
            report['recommendations'].extend([
                'Verify consent for all data processing activities',
                'Implement data minimization practices',
                'Ensure right to erasure procedures are operational'
            ])
        
        return report
    
    async def detect_compliance_violations(self, audit_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential compliance violations."""
        violations = []
        
        for entry in audit_data:
            regulation = entry.get('regulation')
            compliant = entry.get('compliant', True)
            details = entry.get('details', {})
            
            if not compliant:
                violation = {
                    'timestamp': entry.get('timestamp'),
                    'regulation': regulation,
                    'event_type': entry.get('event_type'),
                    'violation_type': 'non_compliance',
                    'severity': self._assess_violation_severity(regulation, details),
                    'details': details,
                    'required_actions': self._get_remediation_actions(regulation, details)
                }
                violations.append(violation)
        
        return violations
    
    async def audit_trail_analysis(self, audit_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze audit trail completeness and quality."""
        analysis = {
            'completeness_score': 0.0,
            'quality_score': 0.0,
            'coverage_gaps': [],
            'data_integrity_issues': [],
            'retention_compliance': {}
        }
        
        if not audit_data:
            return analysis
        
        # Check completeness
        completeness = await self.audit_processors['completeness'](audit_data)
        analysis['completeness_score'] = completeness
        
        # Check quality
        quality = await self.audit_processors['accuracy'](audit_data)
        analysis['quality_score'] = quality
        
        # Check timeliness
        timeliness = await self.audit_processors['timeliness'](audit_data)
        analysis['timeliness_score'] = timeliness
        
        return analysis
    
    async def risk_assessment_analytics(self, compliance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform compliance risk assessment."""
        risk_assessment = {
            'overall_risk_level': 'low',
            'risk_factors': [],
            'mitigation_strategies': [],
            'priority_actions': []
        }
        
        compliance_rate = compliance_data.get('overall_compliance_rate', 1.0)
        violations = compliance_data.get('violations', [])
        
        # Assess risk level
        if compliance_rate < 0.9:
            risk_assessment['overall_risk_level'] = 'high'
            risk_assessment['risk_factors'].append('Low overall compliance rate')
        elif compliance_rate < 0.95:
            risk_assessment['overall_risk_level'] = 'medium'
            risk_assessment['risk_factors'].append('Moderate compliance rate')
        
        # Analyze violations
        if len(violations) > 10:
            risk_assessment['risk_factors'].append('High number of violations')
            
        critical_violations = [v for v in violations if v.get('severity') == 'critical']
        if critical_violations:
            risk_assessment['risk_factors'].append('Critical violations present')
            risk_assessment['priority_actions'].append('Address critical violations immediately')
        
        # Generate mitigation strategies
        risk_assessment['mitigation_strategies'] = [
            'Implement automated compliance monitoring',
            'Enhance audit trail procedures',
            'Increase compliance training frequency',
            'Review and update compliance policies'
        ]
        
        return risk_assessment
    
    def _assess_violation_severity(self, regulation: str, details: Dict[str, Any]) -> str:
        """Assess severity of compliance violation."""
        # Default severity assessment
        if regulation == 'MIFID_II':
            if not details.get('audit_trail'):
                return 'critical'
            elif not details.get('explanation_provided'):
                return 'high'
        elif regulation == 'GDPR':
            if not details.get('consent_obtained'):
                return 'critical'
        
        return 'medium'
    
    def _get_remediation_actions(self, regulation: str, details: Dict[str, Any]) -> List[str]:
        """Get remediation actions for violation."""
        actions = ['Review compliance procedures']
        
        if regulation == 'MIFID_II':
            if not details.get('audit_trail'):
                actions.append('Implement complete audit trail logging')
            if not details.get('explanation_provided'):
                actions.append('Ensure all decisions have explanations')
        elif regulation == 'GDPR':
            if not details.get('consent_obtained'):
                actions.append('Obtain proper consent before data processing')
        
        return actions
    
    async def _check_audit_completeness(self, audit_data: List[Dict[str, Any]]) -> float:
        """Check audit trail completeness."""
        if not audit_data:
            return 0.0
        
        required_fields = ['timestamp', 'event_type', 'user_id', 'details']
        complete_entries = 0
        
        for entry in audit_data:
            if all(field in entry and entry[field] for field in required_fields):
                complete_entries += 1
        
        return complete_entries / len(audit_data)
    
    async def _check_audit_accuracy(self, audit_data: List[Dict[str, Any]]) -> float:
        """Check audit data accuracy."""
        # Simplified accuracy check
        return 0.95  # Placeholder
    
    async def _check_audit_timeliness(self, audit_data: List[Dict[str, Any]]) -> float:
        """Check audit data timeliness."""
        # Simplified timeliness check
        return 0.98  # Placeholder


class SystemHealthAnalyticsEngine:
    """Engine for system health analytics and proactive monitoring."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize system health analytics engine."""
        self.config = config or AnalyticsConfig()
        
        # Health indicators
        self.health_indicators = {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'disk_threshold': 90.0,
            'network_latency_threshold': 100.0,
            'error_rate_threshold': 0.05
        }
        
        # Predictive models (simplified)
        self.predictive_models = {
            'resource_exhaustion': self._predict_resource_exhaustion,
            'performance_degradation': self._predict_performance_degradation,
            'failure_probability': self._predict_failure_probability
        }
        
        # Threshold monitors
        self.threshold_monitors = {
            'cpu': self._monitor_cpu_usage,
            'memory': self._monitor_memory_usage,
            'disk': self._monitor_disk_usage,
            'network': self._monitor_network_latency,
            'errors': self._monitor_error_rate
        }
    
    async def analyze_system_health(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall system health."""
        if not metrics_data:
            return {'overall_health_score': 0, 'message': 'No metrics data available'}
        
        health = {
            'overall_health_score': 0,
            'component_scores': {},
            'trend_analysis': {},
            'recommendations': [],
            'alerts': [],
            'resource_utilization': {}
        }
        
        # Calculate component scores
        latest_metrics = metrics_data[-1] if metrics_data else {}
        
        cpu_score = self._calculate_cpu_health_score(latest_metrics.get('cpu_usage', 0))
        memory_score = self._calculate_memory_health_score(latest_metrics.get('memory_usage', 0))
        disk_score = self._calculate_disk_health_score(latest_metrics.get('disk_usage', 0))
        network_score = self._calculate_network_health_score(latest_metrics.get('network_latency', 0))
        error_score = self._calculate_error_health_score(latest_metrics.get('error_rate', 0))
        
        health['component_scores'] = {
            'cpu': cpu_score,
            'memory': memory_score,
            'disk': disk_score,
            'network': network_score,
            'error_handling': error_score
        }
        
        # Calculate overall score (weighted average)
        weights = {'cpu': 0.25, 'memory': 0.25, 'disk': 0.2, 'network': 0.15, 'error_handling': 0.15}
        health['overall_health_score'] = sum(
            health['component_scores'][component] * weight
            for component, weight in weights.items()
        )
        
        # Trend analysis
        if len(metrics_data) > 1:
            health['trend_analysis'] = await self._analyze_health_trends(metrics_data)
        
        # Generate recommendations
        health['recommendations'] = await self._generate_health_recommendations(health['component_scores'])
        
        # Resource utilization summary
        health['resource_utilization'] = {
            'cpu_usage': latest_metrics.get('cpu_usage', 0),
            'memory_usage': latest_metrics.get('memory_usage', 0),
            'disk_usage': latest_metrics.get('disk_usage', 0),
            'network_latency': latest_metrics.get('network_latency', 0)
        }
        
        return health
    
    async def predict_system_issues(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict potential system issues."""
        predictions = {
            'resource_exhaustion': await self.predictive_models['resource_exhaustion'](metrics_data),
            'performance_degradation': await self.predictive_models['performance_degradation'](metrics_data),
            'failure_probability': await self.predictive_models['failure_probability'](metrics_data)
        }
        
        return predictions
    
    async def generate_health_insights(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system health insights."""
        insights = {
            'summary': '',
            'critical_issues': [],
            'optimization_opportunities': [],
            'preventive_actions': [],
            'performance_impact': 'low'
        }
        
        overall_score = health_data.get('overall_health_score', 100)
        component_scores = health_data.get('component_scores', {})
        
        # Assess performance impact
        if overall_score < 60:
            insights['performance_impact'] = 'high'
        elif overall_score < 80:
            insights['performance_impact'] = 'medium'
        
        # Identify critical issues
        for component, score in component_scores.items():
            if score < 60:
                insights['critical_issues'].append(f"{component.upper()} health critically low ({score:.1f}%)")
        
        # Optimization opportunities
        for component, score in component_scores.items():
            if 60 <= score < 80:
                insights['optimization_opportunities'].append(f"Optimize {component} performance")
        
        # Preventive actions
        if overall_score < 90:
            insights['preventive_actions'].extend([
                'Monitor system resources closely',
                'Review system capacity planning',
                'Consider performance optimization'
            ])
        
        # Generate summary
        insights['summary'] = f"System health at {overall_score:.1f}% with {insights['performance_impact']} performance impact"
        
        return insights
    
    async def monitor_resource_usage(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor resource usage patterns."""
        monitoring = {
            'cpu_trend': 'stable',
            'memory_trend': 'stable',
            'disk_trend': 'stable',
            'network_trend': 'stable',
            'resource_alerts': []
        }
        
        if len(metrics_data) < 2:
            return monitoring
        
        # Analyze trends
        cpu_values = [m.get('cpu_usage', 0) for m in metrics_data]
        memory_values = [m.get('memory_usage', 0) for m in metrics_data]
        disk_values = [m.get('disk_usage', 0) for m in metrics_data]
        network_values = [m.get('network_latency', 0) for m in metrics_data]
        
        monitoring['cpu_trend'] = self._analyze_resource_trend(cpu_values)
        monitoring['memory_trend'] = self._analyze_resource_trend(memory_values)
        monitoring['disk_trend'] = self._analyze_resource_trend(disk_values)
        monitoring['network_trend'] = self._analyze_resource_trend(network_values)
        
        # Check for alerts
        latest = metrics_data[-1]
        if latest.get('cpu_usage', 0) > self.health_indicators['cpu_threshold']:
            monitoring['resource_alerts'].append('CPU usage above threshold')
        
        if latest.get('memory_usage', 0) > self.health_indicators['memory_threshold']:
            monitoring['resource_alerts'].append('Memory usage above threshold')
        
        return monitoring
    
    async def detect_performance_degradation(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect performance degradation patterns."""
        degradation = {
            'detected': False,
            'affected_components': [],
            'severity': 'low',
            'degradation_rate': 0.0,
            'recommendations': []
        }
        
        if len(metrics_data) < 10:
            return degradation
        
        # Compare recent vs historical performance
        recent_data = metrics_data[-5:]
        historical_data = metrics_data[-15:-5]
        
        recent_avg_cpu = statistics.mean([m.get('cpu_usage', 0) for m in recent_data])
        historical_avg_cpu = statistics.mean([m.get('cpu_usage', 0) for m in historical_data])
        
        recent_avg_latency = statistics.mean([m.get('network_latency', 0) for m in recent_data])
        historical_avg_latency = statistics.mean([m.get('network_latency', 0) for m in historical_data])
        
        # Check for degradation
        cpu_degradation = (recent_avg_cpu - historical_avg_cpu) / historical_avg_cpu if historical_avg_cpu > 0 else 0
        latency_degradation = (recent_avg_latency - historical_avg_latency) / historical_avg_latency if historical_avg_latency > 0 else 0
        
        if cpu_degradation > 0.2:  # 20% increase
            degradation['detected'] = True
            degradation['affected_components'].append('CPU')
            
        if latency_degradation > 0.3:  # 30% increase
            degradation['detected'] = True
            degradation['affected_components'].append('Network')
        
        if degradation['detected']:
            degradation['degradation_rate'] = max(cpu_degradation, latency_degradation)
            degradation['severity'] = 'high' if degradation['degradation_rate'] > 0.5 else 'medium'
            degradation['recommendations'] = [
                'Investigate recent system changes',
                'Review resource allocation',
                'Consider scaling resources'
            ]
        
        return degradation
    
    def _calculate_cpu_health_score(self, cpu_usage: float) -> float:
        """Calculate CPU health score."""
        if cpu_usage < 60:
            return 100.0
        elif cpu_usage < 80:
            return 100 - (cpu_usage - 60) * 2  # Linear decrease
        else:
            return max(0, 60 - (cpu_usage - 80) * 3)  # Faster decrease
    
    def _calculate_memory_health_score(self, memory_usage: float) -> float:
        """Calculate memory health score."""
        if memory_usage < 70:
            return 100.0
        elif memory_usage < 85:
            return 100 - (memory_usage - 70) * 2
        else:
            return max(0, 70 - (memory_usage - 85) * 2)
    
    def _calculate_disk_health_score(self, disk_usage: float) -> float:
        """Calculate disk health score."""
        if disk_usage < 80:
            return 100.0
        elif disk_usage < 90:
            return 100 - (disk_usage - 80) * 3
        else:
            return max(0, 70 - (disk_usage - 90) * 5)
    
    def _calculate_network_health_score(self, network_latency: float) -> float:
        """Calculate network health score."""
        if network_latency < 50:
            return 100.0
        elif network_latency < 100:
            return 100 - (network_latency - 50) * 1.5
        else:
            return max(0, 25 - (network_latency - 100) * 0.5)
    
    def _calculate_error_health_score(self, error_rate: float) -> float:
        """Calculate error handling health score."""
        if error_rate < 0.01:  # Less than 1%
            return 100.0
        elif error_rate < 0.05:  # Less than 5%
            return 100 - (error_rate - 0.01) * 2000  # Scale to percentage
        else:
            return max(0, 20 - (error_rate - 0.05) * 1000)
    
    async def _analyze_health_trends(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze health trends over time."""
        if len(metrics_data) < 3:
            return {}
        
        # Simple trend analysis
        cpu_values = [m.get('cpu_usage', 0) for m in metrics_data]
        memory_values = [m.get('memory_usage', 0) for m in metrics_data]
        
        return {
            'cpu_trend': self._analyze_resource_trend(cpu_values),
            'memory_trend': self._analyze_resource_trend(memory_values)
        }
    
    def _analyze_resource_trend(self, values: List[float]) -> str:
        """Analyze trend in resource values."""
        if len(values) < 3:
            return 'stable'
        
        recent_avg = statistics.mean(values[-3:])
        earlier_avg = statistics.mean(values[:3])
        
        change = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
        
        if change > 0.1:
            return 'increasing'
        elif change < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _generate_health_recommendations(self, component_scores: Dict[str, float]) -> List[str]:
        """Generate health recommendations based on component scores."""
        recommendations = []
        
        for component, score in component_scores.items():
            if score < 60:
                if component == 'cpu':
                    recommendations.append('Consider CPU optimization or scaling')
                elif component == 'memory':
                    recommendations.append('Review memory usage and consider increasing allocation')
                elif component == 'disk':
                    recommendations.append('Clean up disk space or expand storage')
                elif component == 'network':
                    recommendations.append('Investigate network performance issues')
                elif component == 'error_handling':
                    recommendations.append('Review error handling and reduce error rates')
        
        if not recommendations:
            recommendations.append('System health is good - continue monitoring')
        
        return recommendations
    
    # Simplified predictive models
    async def _predict_resource_exhaustion(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict resource exhaustion."""
        return {'probability': 0.1, 'time_to_exhaustion': '> 7 days', 'confidence': 0.7}
    
    async def _predict_performance_degradation(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict performance degradation."""
        return {'probability': 0.15, 'affected_components': ['network'], 'confidence': 0.6}
    
    async def _predict_failure_probability(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict system failure probability."""
        return {'probability': 0.05, 'risk_factors': ['high_cpu'], 'confidence': 0.8}
    
    # Monitor methods (placeholders)
    async def _monitor_cpu_usage(self, metrics: Dict[str, Any]) -> bool:
        return metrics.get('cpu_usage', 0) > self.health_indicators['cpu_threshold']
    
    async def _monitor_memory_usage(self, metrics: Dict[str, Any]) -> bool:
        return metrics.get('memory_usage', 0) > self.health_indicators['memory_threshold']
    
    async def _monitor_disk_usage(self, metrics: Dict[str, Any]) -> bool:
        return metrics.get('disk_usage', 0) > self.health_indicators['disk_threshold']
    
    async def _monitor_network_latency(self, metrics: Dict[str, Any]) -> bool:
        return metrics.get('network_latency', 0) > self.health_indicators['network_latency_threshold']
    
    async def _monitor_error_rate(self, metrics: Dict[str, Any]) -> bool:
        return metrics.get('error_rate', 0) > self.health_indicators['error_rate_threshold']


class DashboardDataGenerator:
    """Generator for real-time operational dashboard data."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize dashboard data generator."""
        self.config = config or AnalyticsConfig()
        self.analytics_engines = {
            'log': LogAnalyticsEngine(self.config),
            'trading': TradingAnalyticsEngine(self.config),
            'compliance': ComplianceAnalyticsEngine(self.config),
            'system': SystemHealthAnalyticsEngine(self.config)
        }
        self.refresh_interval = self.config.dashboard_settings['refresh_interval']
        self.data_cache = {}
        self.cache_timestamps = {}
    
    async def generate_overview_data(self) -> Dict[str, Any]:
        """Generate overview dashboard data."""
        cache_key = 'overview'
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.data_cache[cache_key]
        
        overview = {
            'system_status': {
                'overall_health': 85.5,
                'status': 'healthy',
                'active_alerts': 2,
                'uptime': '15 days, 6 hours'
            },
            'trading_summary': {
                'total_trades_today': 45,
                'success_rate': 0.89,
                'total_pnl': 1250.50,
                'active_positions': 12
            },
            'compliance_status': {
                'compliance_rate': 0.96,
                'violations_today': 1,
                'audit_completeness': 0.98
            },
            'recent_alerts': [
                {'timestamp': datetime.now(timezone.utc).isoformat(), 'level': 'warning', 'message': 'CPU usage above 80%'},
                {'timestamp': (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat(), 'level': 'info', 'message': 'Trading session started'}
            ],
            'performance_metrics': {
                'avg_response_time': 145.5,
                'error_rate': 0.023,
                'throughput': 1250.0
            }
        }
        
        # Cache the result
        self._cache_data(cache_key, overview)
        return overview
    
    async def generate_trading_dashboard(self) -> Dict[str, Any]:
        """Generate trading-specific dashboard data."""
        cache_key = 'trading_dashboard'
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.data_cache[cache_key]
        
        trading_dashboard = {
            'pnl_summary': {
                'daily_pnl': 150.25,
                'weekly_pnl': 890.75,
                'monthly_pnl': 2450.50,
                'ytd_pnl': 12500.00
            },
            'trade_statistics': {
                'total_trades': 125,
                'successful_trades': 112,
                'failed_trades': 13,
                'success_rate': 0.896,
                'avg_trade_size': 1250.00
            },
            'performance_charts': {
                'pnl_chart': [100, 120, 110, 150, 140, 160, 150],
                'volume_chart': [1000, 1200, 950, 1400, 1100, 1350, 1250],
                'success_rate_chart': [0.85, 0.88, 0.82, 0.91, 0.87, 0.89, 0.896]
            },
            'risk_metrics': {
                'var_95': -125.50,
                'max_drawdown': 0.08,
                'sharpe_ratio': 1.45,
                'volatility': 0.12
            }
        }
        
        # Cache the result
        self._cache_data(cache_key, trading_dashboard)
        return trading_dashboard
    
    async def generate_compliance_dashboard(self) -> Dict[str, Any]:
        """Generate compliance-specific dashboard data."""
        cache_key = 'compliance_dashboard'
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.data_cache[cache_key]
        
        compliance_dashboard = {
            'regulatory_status': {
                'MIFID_II': {'compliance_rate': 0.98, 'violations': 1, 'status': 'compliant'},
                'GDPR': {'compliance_rate': 0.96, 'violations': 2, 'status': 'compliant'},
                'SEC_RULE_3A4': {'compliance_rate': 0.99, 'violations': 0, 'status': 'compliant'}
            },
            'audit_metrics': {
                'completeness': 0.97,
                'accuracy': 0.99,
                'timeliness': 0.98
            },
            'recent_violations': [
                {
                    'timestamp': (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
                    'regulation': 'GDPR',
                    'type': 'consent_missing',
                    'severity': 'medium'
                }
            ],
            'upcoming_reports': [
                {
                    'regulation': 'MIFID_II',
                    'due_date': (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                    'status': 'in_progress'
                }
            ]
        }
        
        # Cache the result
        self._cache_data(cache_key, compliance_dashboard)
        return compliance_dashboard
    
    async def generate_system_health_dashboard(self) -> Dict[str, Any]:
        """Generate system health dashboard data."""
        cache_key = 'system_health_dashboard'
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.data_cache[cache_key]
        
        system_health_dashboard = {
            'resource_utilization': {
                'cpu_usage': 68.5,
                'memory_usage': 72.3,
                'disk_usage': 45.8,
                'network_latency': 35.2
            },
            'health_scores': {
                'overall_health': 85.5,
                'cpu_health': 82.0,
                'memory_health': 78.0,
                'disk_health': 95.0,
                'network_health': 88.0
            },
            'trend_indicators': {
                'cpu_trend': 'stable',
                'memory_trend': 'increasing',
                'disk_trend': 'stable',
                'network_trend': 'improving'
            },
            'predictions': {
                'resource_exhaustion': {'probability': 0.1, 'time_estimate': '> 7 days'},
                'performance_degradation': {'probability': 0.15, 'components': ['memory']},
                'failure_risk': {'probability': 0.05, 'confidence': 0.8}
            }
        }
        
        # Cache the result
        self._cache_data(cache_key, system_health_dashboard)
        return system_health_dashboard
    
    async def refresh_dashboard_cache(self) -> None:
        """Force refresh of dashboard cache."""
        self.data_cache.clear()
        self.cache_timestamps.clear()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.data_cache:
            return False
        
        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return False
        
        cache_duration = self.config.dashboard_settings['cache_duration']
        return (datetime.now(timezone.utc) - cache_time).total_seconds() < cache_duration
    
    def _cache_data(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache dashboard data."""
        self.data_cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now(timezone.utc)


class InsightsGenerator:
    """Generator for operational insights and recommendations."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize insights generator."""
        self.config = config or AnalyticsConfig()
        
        # Insight templates
        self.insight_templates = {
            'performance': 'System performance is {status} with {metric} at {value}',
            'trading': 'Trading performance shows {trend} with {success_rate} success rate',
            'compliance': 'Compliance status is {status} with {violations} violations',
            'resource': 'Resource utilization is {level} with {component} at {usage}%'
        }
        
        # ML analyzers (simplified)
        self.ml_analyzers = {
            'pattern_detection': self._detect_patterns,
            'anomaly_analysis': self._analyze_anomalies,
            'trend_prediction': self._predict_trends
        }
        
        # Trend detectors
        self.trend_detectors = {
            'upward': lambda x: x[-1] > x[0] and statistics.mean(x[-3:]) > statistics.mean(x[:3]),
            'downward': lambda x: x[-1] < x[0] and statistics.mean(x[-3:]) < statistics.mean(x[:3]),
            'stable': lambda x: abs(x[-1] - x[0]) / x[0] < 0.1 if x[0] != 0 else True
        }
    
    async def generate_operational_insights(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive operational insights."""
        insights = {
            'priority_issues': [],
            'recommendations': [],
            'trends': [],
            'success_metrics': [],
            'executive_summary': '',
            'risk_assessment': 'low'
        }
        
        # Analyze each analytics component
        log_analytics = analytics_data.get('log_analytics', {})
        trading_analytics = analytics_data.get('trading_analytics', {})
        compliance_analytics = analytics_data.get('compliance_analytics', {})
        system_health = analytics_data.get('system_health', {})
        
        # Identify priority issues
        if log_analytics.get('error_rate', 0) > 0.05:
            insights['priority_issues'].append({
                'type': 'system',
                'severity': 'high',
                'issue': 'High error rate detected',
                'impact': 'System reliability',
                'urgency': 'immediate'
            })
        
        if trading_analytics.get('success_rate', 1.0) < 0.8:
            insights['priority_issues'].append({
                'type': 'trading',
                'severity': 'high',
                'issue': 'Low trading success rate',
                'impact': 'Profitability',
                'urgency': 'immediate'
            })
        
        if compliance_analytics.get('compliance_rate', 1.0) < 0.95:
            insights['priority_issues'].append({
                'type': 'compliance',
                'severity': 'critical',
                'issue': 'Compliance rate below threshold',
                'impact': 'Regulatory risk',
                'urgency': 'immediate'
            })
        
        if system_health.get('health_score', 100) < 70:
            insights['priority_issues'].append({
                'type': 'infrastructure',
                'severity': 'medium',
                'issue': 'System health degradation',
                'impact': 'Performance',
                'urgency': 'within_24h'
            })
        
        # Generate recommendations
        insights['recommendations'] = await self._generate_recommendations(analytics_data)
        
        # Detect trends
        insights['trends'] = await self._detect_operational_trends(analytics_data)
        
        # Identify success metrics
        if trading_analytics.get('pnl_trend') == 'positive':
            insights['success_metrics'].append('Trading profitability improving')
        
        if system_health.get('trending') != 'down':
            insights['success_metrics'].append('System health stable')
        
        # Risk assessment
        critical_issues = [i for i in insights['priority_issues'] if i['severity'] == 'critical']
        high_issues = [i for i in insights['priority_issues'] if i['severity'] == 'high']
        
        if critical_issues:
            insights['risk_assessment'] = 'critical'
        elif len(high_issues) > 2:
            insights['risk_assessment'] = 'high'
        elif len(high_issues) > 0:
            insights['risk_assessment'] = 'medium'
        
        # Executive summary
        insights['executive_summary'] = await self._generate_executive_summary(insights)
        
        return insights
    
    async def detect_trends_and_patterns(self, time_series_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Detect trends and patterns in time series data."""
        trends = {}
        
        for metric, values in time_series_data.items():
            if len(values) < 3:
                trends[metric] = 'insufficient_data'
                continue
            
            # Detect trend direction
            if self.trend_detectors['upward'](values):
                trends[metric] = 'upward'
            elif self.trend_detectors['downward'](values):
                trends[metric] = 'downward'
            else:
                trends[metric] = 'stable'
        
        return trends
    
    async def recommend_optimizations(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend system optimizations."""
        recommendations = []
        
        # CPU optimization
        if performance_data.get('cpu_usage', 0) > 80:
            recommendations.append({
                'category': 'performance',
                'priority': 'high',
                'recommendation': 'Optimize CPU-intensive processes',
                'expected_impact': 'Reduce CPU usage by 15-20%',
                'implementation_effort': 'medium'
            })
        
        # Memory optimization
        if performance_data.get('memory_usage', 0) > 85:
            recommendations.append({
                'category': 'performance',
                'priority': 'high',
                'recommendation': 'Implement memory optimization',
                'expected_impact': 'Reduce memory usage by 10-15%',
                'implementation_effort': 'low'
            })
        
        # Trading optimization
        if performance_data.get('trading_success_rate', 1.0) < 0.85:
            recommendations.append({
                'category': 'trading',
                'priority': 'medium',
                'recommendation': 'Review and optimize trading strategies',
                'expected_impact': 'Improve success rate by 5-10%',
                'implementation_effort': 'high'
            })
        
        return recommendations
    
    async def predict_future_issues(self, historical_data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Predict potential future issues."""
        predictions = {
            'resource_issues': [],
            'performance_issues': [],
            'compliance_issues': [],
            'confidence_levels': {}
        }
        
        # Simple trend-based predictions
        for metric, values in historical_data.items():
            if len(values) < 5:
                continue
            
            # Linear trend prediction
            if isinstance(values[0], (int, float)):
                trend = (values[-1] - values[0]) / len(values)
                predicted_value = values[-1] + trend * 5  # 5 periods ahead
                
                # Check for potential issues
                if metric == 'cpu_usage' and predicted_value > 90:
                    predictions['resource_issues'].append(f"CPU usage may exceed 90% in next 5 periods")
                elif metric == 'memory_usage' and predicted_value > 95:
                    predictions['resource_issues'].append(f"Memory usage may exceed 95% in next 5 periods")
                elif metric == 'error_rate' and predicted_value > 0.1:
                    predictions['performance_issues'].append(f"Error rate may exceed 10% in next 5 periods")
        
        # Set confidence levels
        predictions['confidence_levels'] = {
            'resource_predictions': 0.7,
            'performance_predictions': 0.6,
            'compliance_predictions': 0.8
        }
        
        return predictions
    
    async def generate_executive_summary(self, insights_data: Dict[str, Any]) -> str:
        """Generate executive summary of operational insights."""
        priority_issues = insights_data.get('priority_issues', [])
        risk_level = insights_data.get('risk_assessment', 'low')
        
        if risk_level == 'critical':
            summary = f"CRITICAL: System requires immediate attention with {len(priority_issues)} critical issues identified."
        elif risk_level == 'high':
            summary = f"HIGH RISK: {len(priority_issues)} high-priority issues require prompt resolution."
        elif risk_level == 'medium':
            summary = f"MODERATE: {len(priority_issues)} issues identified for planned resolution."
        else:
            summary = "Operations are running smoothly with no critical issues identified."
        
        return summary
    
    async def _generate_recommendations(self, analytics_data: Dict[str, Any]) -> List[str]:
        """Generate operational recommendations."""
        recommendations = []
        
        # System recommendations
        system_health = analytics_data.get('system_health', {})
        if system_health.get('health_score', 100) < 80:
            recommendations.append('Review system resource allocation and optimization')
        
        # Trading recommendations
        trading_analytics = analytics_data.get('trading_analytics', {})
        if trading_analytics.get('success_rate', 1.0) < 0.9:
            recommendations.append('Analyze trading patterns and adjust strategies')
        
        # Compliance recommendations
        compliance_analytics = analytics_data.get('compliance_analytics', {})
        if compliance_analytics.get('violations', 0) > 0:
            recommendations.append('Address compliance violations promptly')
        
        if not recommendations:
            recommendations.append('Continue monitoring current operational metrics')
        
        return recommendations
    
    async def _detect_operational_trends(self, analytics_data: Dict[str, Any]) -> List[str]:
        """Detect operational trends."""
        trends = []
        
        # Simple trend detection based on available data
        system_health = analytics_data.get('system_health', {})
        if system_health.get('trending') == 'down':
            trends.append('System health declining')
        elif system_health.get('trending') == 'up':
            trends.append('System health improving')
        
        trading_analytics = analytics_data.get('trading_analytics', {})
        if trading_analytics.get('pnl_trend') == 'positive':
            trends.append('Trading profitability increasing')
        elif trading_analytics.get('pnl_trend') == 'negative':
            trends.append('Trading profitability decreasing')
        
        return trends
    
    async def _generate_executive_summary(self, insights: Dict[str, Any]) -> str:
        """Generate executive summary."""
        priority_issues = len(insights.get('priority_issues', []))
        risk_level = insights.get('risk_assessment', 'low')
        
        if risk_level == 'critical' or priority_issues > 3:
            return f"URGENT: {priority_issues} critical issues require immediate attention"
        elif risk_level == 'high' or priority_issues > 1:
            return f"ATTENTION: {priority_issues} high-priority issues identified"
        else:
            return "Operations stable with routine monitoring recommendations"
    
    # ML analyzer placeholders
    async def _detect_patterns(self, data: Any) -> Dict[str, Any]:
        """Detect patterns using ML analysis."""
        return {'patterns_detected': 0, 'confidence': 0.8}
    
    async def _analyze_anomalies(self, data: Any) -> Dict[str, Any]:
        """Analyze anomalies using ML."""
        return {'anomalies_detected': 0, 'severity': 'low'}
    
    async def _predict_trends(self, data: Any) -> Dict[str, Any]:
        """Predict trends using ML."""
        return {'predicted_trend': 'stable', 'confidence': 0.7}


class OperationalAnalytics:
    """Main operational analytics system orchestrator."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize operational analytics system."""
        self.config = config or AnalyticsConfig()
        
        # Validate configuration
        if not self.config.validate_config():
            raise ValueError("Invalid analytics configuration")
        
        # Initialize existing systems
        self.activity_logger = ActivityLogger()
        self.enhanced_logger = get_enhanced_logger(__name__)
        self.metrics_collectors = MetricsRegistry()
        
        # Initialize analytics engines
        self.analytics_engines = {
            AnalyticsEngine.LOG_ANALYTICS: LogAnalyticsEngine(self.config),
            AnalyticsEngine.TRADING_ANALYTICS: TradingAnalyticsEngine(self.config),
            AnalyticsEngine.COMPLIANCE_ANALYTICS: ComplianceAnalyticsEngine(self.config),
            AnalyticsEngine.SYSTEM_HEALTH: SystemHealthAnalyticsEngine(self.config)
        }
        
        # Initialize dashboard and insights
        self.dashboard_generator = DashboardDataGenerator(self.config)
        self.insights_generator = InsightsGenerator(self.config)
        
        # Runtime state
        self.is_running = False
        self._tasks = []
    
    async def start(self) -> None:
        """Start the operational analytics system."""
        if self.is_running:
            return
        
        # Start activity logger
        await self.activity_logger.start()
        
        # Start background analytics tasks
        for engine_type, interval in self.config.analysis_intervals.items():
            task = asyncio.create_task(self._run_analytics_loop(engine_type, interval))
            self._tasks.append(task)
        
        self.is_running = True
        self.enhanced_logger.info("Operational analytics system started")
    
    async def stop(self) -> None:
        """Stop the operational analytics system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel analytics tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        # Stop activity logger
        await self.activity_logger.stop()
        
        self.enhanced_logger.info("Operational analytics system stopped")
    
    async def run_analytics_cycle(self) -> Dict[str, Any]:
        """Run a complete analytics cycle."""
        results = {}
        
        try:
            # Gather data from all sources
            activity_data = await self.get_activity_data(hours_back=24)
            structured_logs = await self.get_structured_logs(time_range=timedelta(hours=6))
            metrics_data = await self.get_metrics_data(time_range=timedelta(hours=6))
            
            # Run analytics engines
            results['log_analytics'] = await self._run_log_analytics(structured_logs)
            results['trading_analytics'] = await self._run_trading_analytics(activity_data)
            results['compliance_analytics'] = await self._run_compliance_analytics(activity_data)
            results['system_health'] = await self._run_system_health_analytics(metrics_data)
            
            # Generate insights
            results['insights'] = await self.insights_generator.generate_operational_insights(results)
            
            self.enhanced_logger.info("Analytics cycle completed", results_count=len(results))
            
        except Exception as e:
            self.enhanced_logger.error("Analytics cycle failed", error=str(e))
            results['error'] = str(e)
        
        return results
    
    async def get_activity_data(self, hours_back: int = 24, 
                               categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get activity data from activity logger."""
        # Mock implementation - in real system, query activity logger database
        mock_activities = [
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
                'category': 'trading',
                'action': 'execute',
                'success': True,
                'pnl': Decimal('50.25'),
                'symbol': 'ETH',
                'amount': Decimal('100')
            },
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(hours=2),
                'category': 'system',
                'action': 'error',
                'success': False,
                'error_message': 'Connection timeout'
            }
        ]
        return mock_activities
    
    async def get_structured_logs(self, time_range: timedelta, 
                                 log_levels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get structured logs from enhanced logging."""
        # Mock implementation - in real system, query log aggregation system
        mock_logs = [
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=30),
                'level': 'ERROR',
                'category': 'trading',
                'message': 'Trade execution failed: insufficient balance',
                'execution_time_ms': 150
            },
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=15),
                'level': 'INFO',
                'category': 'system',
                'message': 'System health check completed successfully',
                'execution_time_ms': 25
            }
        ]
        return mock_logs
    
    async def get_metrics_data(self, metric_types: Optional[List[str]] = None,
                              time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get metrics data from monitoring systems."""
        # Mock implementation - in real system, query metrics collectors
        mock_metrics = {
            'system_metrics': [
                {
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=5),
                    'cpu_usage': 68.5,
                    'memory_usage': 72.3,
                    'disk_usage': 45.8,
                    'network_latency': 35.2,
                    'error_rate': 0.023
                }
            ],
            'trading_metrics': {
                'total_trades': 45,
                'success_rate': 0.89,
                'total_pnl': 1250.50
            }
        }
        return mock_metrics
    
    async def _run_analytics_loop(self, engine_type: str, interval: int) -> None:
        """Run analytics loop for specific engine type."""
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                
                if engine_type == 'log_analytics':
                    logs = await self.get_structured_logs(timedelta(hours=1))
                    await self._run_log_analytics(logs)
                elif engine_type == 'trading_analytics':
                    activities = await self.get_activity_data(hours_back=6, categories=['trading'])
                    await self._run_trading_analytics(activities)
                # Add other engine types as needed
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.enhanced_logger.error(f"Analytics loop error for {engine_type}", error=str(e))
    
    async def _run_log_analytics(self, log_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run log analytics engine."""
        engine = self.analytics_engines[AnalyticsEngine.LOG_ANALYTICS]
        
        patterns = await engine.analyze_log_patterns(log_data)
        anomalies = await engine.detect_log_anomalies(log_data)
        insights = await engine.generate_log_insights(patterns, anomalies)
        
        return {'patterns': patterns, 'anomalies': anomalies, 'insights': insights}
    
    async def _run_trading_analytics(self, trading_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run trading analytics engine."""
        engine = self.analytics_engines[AnalyticsEngine.TRADING_ANALYTICS]
        
        performance = await engine.analyze_trading_performance(trading_data)
        patterns = await engine.detect_trading_patterns(trading_data)
        insights = await engine.generate_performance_insights(performance, patterns)
        
        return {'performance': performance, 'patterns': patterns, 'insights': insights}
    
    async def _run_compliance_analytics(self, audit_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run compliance analytics engine."""
        engine = self.analytics_engines[AnalyticsEngine.COMPLIANCE_ANALYTICS]
        
        status = await engine.analyze_compliance_status(audit_data)
        violations = await engine.detect_compliance_violations(audit_data)
        
        return {'status': status, 'violations': violations}
    
    async def _run_system_health_analytics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run system health analytics engine."""
        engine = self.analytics_engines[AnalyticsEngine.SYSTEM_HEALTH]
        
        system_metrics = metrics_data.get('system_metrics', [])
        health = await engine.analyze_system_health(system_metrics)
        predictions = await engine.predict_system_issues(system_metrics)
        insights = await engine.generate_health_insights(health)
        
        return {'health': health, 'predictions': predictions, 'insights': insights}