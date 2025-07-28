"""
AnalysisValidator - Data quality and result validation for analysis operations.

This module provides comprehensive validation for:
- Data quality validation (completeness, consistency, temporal coverage)
- Backtest result validation (realistic returns, proper statistics calculation)
- Computational resource monitoring (prevent runaway analysis, memory limits)
- Analysis quality scoring (0-1 scale based on multiple factors)

Designed to be lightweight and catch issues without adding significant overhead.
"""

import asyncio
import psutil
import statistics
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Union, Tuple
from uuid import UUID, uuid4
import structlog

logger = structlog.get_logger()


# Exception Classes
class ValidationError(Exception):
    """Base validation error."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class DataQualityError(ValidationError):
    """Error in data quality validation."""
    pass


class BacktestValidationError(ValidationError):
    """Error in backtest result validation."""
    pass


class ResourceLimitExceededError(ValidationError):
    """Error when resource limits are exceeded."""
    pass


class ExecutionTimeoutError(ValidationError):
    """Error when execution time limit is exceeded."""
    pass


class RunawayAnalysisError(ValidationError):
    """Error when runaway analysis is detected."""
    pass


class ValidationConfigError(ValidationError):
    """Error in validation configuration."""
    pass


class AnalysisValidator:
    """
    Comprehensive validator for analysis operations.
    
    Provides data quality validation, backtest result validation,
    computational resource monitoring, and analysis quality scoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AnalysisValidator with configuration.
        
        Args:
            config: Validation configuration dictionary
            
        Raises:
            ValidationConfigError: If configuration is invalid
        """
        self.config = self._validate_config(config)
        self.logger = logger.bind(component="analysis_validator")
        
        # Resource monitoring state
        self._active_monitors: Dict[str, Dict[str, Any]] = {}
        self._metrics_history: List[Dict[str, Any]] = []
        self._validation_stats = {
            "validations_performed": 0,
            "validation_success_count": 0,
            "total_validation_time": 0.0,
            "resource_usage_peaks": {}
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and set defaults for configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration with defaults
            
        Raises:
            ValidationConfigError: If configuration is invalid
        """
        default_config = {
            "data_quality": {
                "min_completeness_ratio": 0.8,
                "min_temporal_coverage_hours": 24,
                "max_missing_data_points": 100,
                "consistency_check_enabled": True,
                "outlier_detection_enabled": True,
                "outlier_threshold_std": 3.0
            },
            "backtest_validation": {
                "max_realistic_daily_return": 0.5,
                "min_realistic_daily_return": -0.5,
                "max_sharpe_ratio": 10.0,
                "min_trade_count": 1,
                "max_trade_count": 10000,
                "win_rate_bounds": [0.0, 1.0],
                "profit_factor_bounds": [0.0, 50.0]
            },
            "resource_monitoring": {
                "max_memory_usage_mb": 1024,
                "max_cpu_usage_percent": 80.0,
                "max_execution_time_seconds": 300,
                "memory_leak_detection": True,
                "check_interval_seconds": 5
            },
            "quality_scoring": {
                "weights": {
                    "data_completeness": 0.3,
                    "temporal_coverage": 0.2,
                    "result_consistency": 0.25,
                    "statistical_validity": 0.25
                },
                "min_acceptable_score": 0.6
            }
        }
        
        # Merge with defaults
        merged_config = self._deep_merge(default_config, config)
        
        # Validate specific constraints
        data_quality = merged_config["data_quality"]
        if not 0 <= data_quality["min_completeness_ratio"] <= 1:
            raise ValidationConfigError("min_completeness_ratio must be between 0 and 1")
        
        if data_quality["min_temporal_coverage_hours"] <= 0:
            raise ValidationConfigError("min_temporal_coverage_hours must be positive")
        
        backtest = merged_config["backtest_validation"]
        if backtest["max_realistic_daily_return"] <= backtest["min_realistic_daily_return"]:
            raise ValidationConfigError("max_realistic_daily_return must be greater than min_realistic_daily_return")
        
        weights = merged_config["quality_scoring"]["weights"]
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValidationConfigError(f"Quality scoring weights must sum to 1.0, got {weight_sum}")
        
        return merged_config
    
    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    # Data Quality Validation
    async def validate_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.
        
        Args:
            data: Market data to validate
            
        Returns:
            Validation results dictionary
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting data quality validation", data_keys=list(data.keys()))
            
            # Run all validation components
            completeness = await self.validate_data_completeness(data)
            temporal_coverage = await self.validate_temporal_coverage(data)
            consistency = await self.validate_data_consistency(data)
            outlier_analysis = await self.detect_outliers(data)
            
            # Calculate overall quality score
            quality_metrics = {
                "data_completeness": completeness.get("completeness_ratio", 0),
                "temporal_coverage": temporal_coverage.get("coverage_ratio", 0),
                "result_consistency": 1.0 - len(consistency.get("inconsistencies", [])) / 10.0,
                "statistical_validity": 1.0 - len(outlier_analysis.get("outlier_indices", [])) / 10.0
            }
            
            overall_score = await self.calculate_analysis_quality_score(quality_metrics)
            
            # Determine if validation passed
            validation_passed = (
                completeness["validation_passed"] and
                temporal_coverage["validation_passed"] and
                consistency["validation_passed"] and
                overall_score["overall_score"] >= self.config["quality_scoring"]["min_acceptable_score"]
            )
            
            # Collect issues and recommendations
            issues_found = []
            recommendations = []
            
            if not completeness["validation_passed"]:
                issues_found.append("Data completeness below threshold")
                recommendations.append("Ensure all required data points are available")
            
            if not temporal_coverage["validation_passed"]:
                issues_found.append("Insufficient temporal coverage")
                recommendations.append("Extend data collection period")
            
            if not consistency["validation_passed"]:
                issues_found.append("Data consistency issues detected")
                recommendations.append("Review data source quality and preprocessing")
            
            result = {
                "completeness": completeness,
                "temporal_coverage": temporal_coverage,
                "consistency": consistency,
                "outlier_analysis": outlier_analysis,
                "overall_quality_score": overall_score["overall_score"],
                "validation_passed": validation_passed,
                "issues_found": issues_found,
                "recommendations": recommendations
            }
            
            # Update statistics
            self._update_validation_stats(True, (datetime.now() - start_time).total_seconds())
            
            self.logger.info(
                "Data quality validation completed",
                validation_passed=validation_passed,
                quality_score=overall_score["overall_score"]
            )
            
            return result
            
        except Exception as e:
            self._update_validation_stats(False, (datetime.now() - start_time).total_seconds())
            self.logger.error("Data quality validation failed", error=str(e))
            raise DataQualityError(f"Data quality validation failed: {str(e)}")
    
    async def validate_data_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data completeness.
        
        Args:
            data: Market data to validate
            
        Returns:
            Completeness validation results
        """
        total_expected_points = 0
        missing_points = 0
        data_gaps = []
        
        # Check for required fields
        required_fields = ["timestamps", "prices"]
        for field in required_fields:
            if field not in data or not data[field]:
                return {
                    "completeness_ratio": 0.0,
                    "missing_data_points": float('inf'),
                    "data_gaps": [f"Missing required field: {field}"],
                    "validation_passed": False
                }
        
        timestamps = data.get("timestamps", [])
        prices = data.get("prices", [])
        volumes = data.get("volumes", [])
        
        if not timestamps:
            return {
                "completeness_ratio": 0.0,
                "missing_data_points": 0,
                "data_gaps": ["No timestamp data"],
                "validation_passed": False
            }
        
        # Calculate expected data points based on time range
        if len(timestamps) >= 2:
            time_diff = timestamps[-1] - timestamps[0]
            if isinstance(timestamps[0], datetime):
                expected_interval = time_diff / (len(timestamps) - 1)
                total_expected_points = int(time_diff.total_seconds() / expected_interval.total_seconds()) + 1
            else:
                total_expected_points = len(timestamps)
        else:
            total_expected_points = len(timestamps)
        
        # Count missing data points
        if len(prices) < len(timestamps):
            missing_points += len(timestamps) - len(prices)
            data_gaps.append(f"Missing {len(timestamps) - len(prices)} price points")
        
        if volumes and len(volumes) < len(timestamps):
            missing_points += len(timestamps) - len(volumes)
            data_gaps.append(f"Missing {len(timestamps) - len(volumes)} volume points")
        
        # Check for None values
        none_prices = sum(1 for p in prices if p is None)
        none_volumes = sum(1 for v in volumes if v is None) if volumes else 0
        
        missing_points += none_prices + none_volumes
        if none_prices > 0:
            data_gaps.append(f"{none_prices} null price values")
        if none_volumes > 0:
            data_gaps.append(f"{none_volumes} null volume values")
        
        # Calculate completeness ratio
        if total_expected_points > 0:
            completeness_ratio = max(0, (total_expected_points - missing_points) / total_expected_points)
        else:
            completeness_ratio = 0.0
        
        validation_passed = (
            completeness_ratio >= self.config["data_quality"]["min_completeness_ratio"] and
            missing_points <= self.config["data_quality"]["max_missing_data_points"]
        )
        
        return {
            "completeness_ratio": completeness_ratio,
            "missing_data_points": missing_points,
            "data_gaps": data_gaps,
            "validation_passed": validation_passed
        }
    
    async def validate_temporal_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate temporal coverage of data.
        
        Args:
            data: Market data to validate
            
        Returns:
            Temporal coverage validation results
        """
        timestamps = data.get("timestamps", [])
        
        if not timestamps:
            return {
                "coverage_hours": 0,
                "expected_data_points": 0,
                "actual_data_points": 0,
                "coverage_ratio": 0.0,
                "validation_passed": False
            }
        
        # Calculate time coverage
        if len(timestamps) >= 2:
            start_time = min(timestamps)
            end_time = max(timestamps)
            
            if isinstance(start_time, datetime):
                coverage_duration = end_time - start_time
                coverage_hours = coverage_duration.total_seconds() / 3600
            else:
                # Assume hourly intervals if not datetime
                coverage_hours = len(timestamps)
        else:
            coverage_hours = 0
        
        # Calculate expected vs actual data points
        expected_points = max(1, int(coverage_hours))
        actual_points = len(timestamps)
        coverage_ratio = actual_points / expected_points if expected_points > 0 else 0
        
        validation_passed = coverage_hours >= self.config["data_quality"]["min_temporal_coverage_hours"]
        
        return {
            "coverage_hours": coverage_hours,
            "expected_data_points": expected_points,
            "actual_data_points": actual_points,
            "coverage_ratio": coverage_ratio,
            "validation_passed": validation_passed
        }
    
    async def validate_data_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data consistency.
        
        Args:
            data: Market data to validate
            
        Returns:
            Consistency validation results
        """
        if not self.config["data_quality"]["consistency_check_enabled"]:
            return {
                "price_consistency": True,
                "volume_consistency": True,
                "timestamp_consistency": True,
                "outliers_detected": [],
                "validation_passed": True
            }
        
        inconsistencies = []
        timestamps = data.get("timestamps", [])
        prices = data.get("prices", [])
        volumes = data.get("volumes", [])
        
        # Check timestamp consistency
        if len(timestamps) > 1:
            timestamp_diffs = []
            for i in range(1, len(timestamps)):
                if isinstance(timestamps[i], datetime):
                    diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                    timestamp_diffs.append(diff)
            
            if timestamp_diffs:
                avg_diff = statistics.mean(timestamp_diffs)
                for i, diff in enumerate(timestamp_diffs):
                    if abs(diff - avg_diff) > avg_diff * 0.5:  # 50% tolerance
                        inconsistencies.append(f"Irregular timestamp interval at index {i+1}")
        
        # Check price consistency
        if prices:
            valid_prices = [p for p in prices if p is not None and p > 0]
            if len(valid_prices) != len(prices):
                inconsistencies.append("Invalid or negative prices detected")
            
            if len(valid_prices) > 1:
                price_changes = []
                for i in range(1, len(valid_prices)):
                    change = abs(valid_prices[i] - valid_prices[i-1]) / valid_prices[i-1]
                    price_changes.append(change)
                
                # Flag extremely large price changes (>50% in one period)
                large_changes = [i for i, change in enumerate(price_changes) if change > 0.5]
                if large_changes:
                    inconsistencies.append(f"Extreme price changes detected at indices: {large_changes}")
        
        # Check volume consistency
        if volumes:
            valid_volumes = [v for v in volumes if v is not None and v >= 0]
            if len(valid_volumes) != len(volumes):
                inconsistencies.append("Invalid or negative volumes detected")
        
        validation_passed = len(inconsistencies) == 0
        
        return {
            "price_consistency": "price" not in str(inconsistencies),
            "volume_consistency": "volume" not in str(inconsistencies),
            "timestamp_consistency": "timestamp" not in str(inconsistencies),
            "inconsistencies": inconsistencies,
            "validation_passed": validation_passed
        }
    
    async def detect_outliers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect outliers in market data.
        
        Args:
            data: Market data to analyze
            
        Returns:
            Outlier detection results
        """
        if not self.config["data_quality"]["outlier_detection_enabled"]:
            return {
                "outliers_found": False,
                "outlier_indices": [],
                "outlier_values": [],
                "outlier_scores": []
            }
        
        prices = data.get("prices", [])
        volumes = data.get("volumes", [])
        
        outlier_indices = []
        outlier_values = []
        outlier_scores = []
        
        # Detect price outliers using z-score
        if prices:
            valid_prices = [(i, p) for i, p in enumerate(prices) if p is not None and p > 0]
            if len(valid_prices) > 3:
                price_values = [p for _, p in valid_prices]
                mean_price = statistics.mean(price_values)
                std_price = statistics.stdev(price_values)
                threshold = self.config["data_quality"]["outlier_threshold_std"]
                
                for i, price in valid_prices:
                    z_score = abs(price - mean_price) / std_price if std_price > 0 else 0
                    if z_score > threshold:
                        outlier_indices.append(i)
                        outlier_values.append(price)
                        outlier_scores.append(z_score)
        
        # Detect volume outliers
        if volumes:
            valid_volumes = [(i, v) for i, v in enumerate(volumes) if v is not None and v >= 0]
            if len(valid_volumes) > 3:
                volume_values = [v for _, v in valid_volumes]
                mean_volume = statistics.mean(volume_values)
                std_volume = statistics.stdev(volume_values)
                threshold = self.config["data_quality"]["outlier_threshold_std"]
                
                for i, volume in valid_volumes:
                    z_score = abs(volume - mean_volume) / std_volume if std_volume > 0 else 0
                    if z_score > threshold and i not in outlier_indices:
                        outlier_indices.append(i)
                        outlier_values.append(volume)
                        outlier_scores.append(z_score)
        
        return {
            "outliers_found": len(outlier_indices) > 0,
            "outlier_indices": outlier_indices,
            "outlier_values": outlier_values,
            "outlier_scores": outlier_scores
        }
    
    # Backtest Result Validation
    async def validate_backtest_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive backtest result validation.
        
        Args:
            results: Backtest results to validate
            
        Returns:
            Validation results dictionary
        """
        try:
            self.logger.info("Starting backtest result validation", strategy=results.get("strategy_name"))
            
            return_validation = await self.validate_realistic_returns(results)
            statistics_validation = await self.validate_statistics_calculation(results)
            trade_validation = await self.validate_trade_history(results)
            
            # Calculate overall validation score
            validation_scores = [
                return_validation.get("validation_passed", False),
                statistics_validation.get("validation_passed", False),
                trade_validation.get("validation_passed", False)
            ]
            
            overall_score = sum(validation_scores) / len(validation_scores)
            validation_passed = overall_score >= 0.8  # Require 80% of validations to pass
            
            # Collect issues
            issues_found = []
            recommendations = []
            
            if not return_validation["validation_passed"]:
                issues_found.extend(return_validation.get("issues", []))
                recommendations.append("Review return calculations and market data")
            
            if not statistics_validation["validation_passed"]:
                issues_found.extend(statistics_validation.get("issues", []))
                recommendations.append("Verify statistical calculations")
            
            if not trade_validation["validation_passed"]:
                issues_found.extend(trade_validation.get("issues", []))
                recommendations.append("Check trade history consistency")
            
            result = {
                "return_validation": return_validation,
                "statistics_validation": statistics_validation,
                "trade_history_validation": trade_validation,
                "overall_validation_score": overall_score,
                "validation_passed": validation_passed,
                "issues_found": issues_found,
                "recommendations": recommendations
            }
            
            self.logger.info(
                "Backtest validation completed",
                validation_passed=validation_passed,
                overall_score=overall_score
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Backtest validation failed", error=str(e))
            raise BacktestValidationError(f"Backtest validation failed: {str(e)}")
    
    async def validate_realistic_returns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that returns are realistic.
        
        Args:
            results: Backtest results
            
        Returns:
            Return validation results
        """
        config = self.config["backtest_validation"]
        issues = []
        
        total_return = results.get("total_return", 0)
        annual_return = results.get("annual_return", 0)
        
        # Check daily return bounds
        daily_returns_valid = True
        if "trade_history" in results:
            trade_returns = [trade.get("pnl", 0) for trade in results["trade_history"]]
            if trade_returns:
                max_daily_return = max(trade_returns) / 10000  # Assume $10k portfolio
                min_daily_return = min(trade_returns) / 10000
                
                if max_daily_return > config["max_realistic_daily_return"]:
                    daily_returns_valid = False
                    issues.append(f"Daily return {max_daily_return:.2%} exceeds realistic maximum")
                
                if min_daily_return < config["min_realistic_daily_return"]:
                    daily_returns_valid = False
                    issues.append(f"Daily return {min_daily_return:.2%} below realistic minimum")
        
        # Check annual return bounds
        annual_return_valid = abs(annual_return) <= 10.0  # 1000% annual return threshold
        if not annual_return_valid:
            issues.append(f"Annual return {annual_return:.2%} is unrealistic")
        
        # Check total return bounds
        total_return_valid = abs(total_return) <= 5.0  # 500% total return threshold
        if not total_return_valid:
            issues.append(f"Total return {total_return:.2%} is unrealistic")
        
        validation_passed = daily_returns_valid and annual_return_valid and total_return_valid
        
        return {
            "daily_returns_valid": daily_returns_valid,
            "annual_return_valid": annual_return_valid,
            "total_return_valid": total_return_valid,
            "validation_passed": validation_passed,
            "outlier_returns": [],  # Could be enhanced to detect specific outlier trades
            "issues": issues
        }
    
    async def validate_statistics_calculation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate proper statistics calculation.
        
        Args:
            results: Backtest results
            
        Returns:
            Statistics validation results
        """
        config = self.config["backtest_validation"]
        issues = []
        
        # Validate Sharpe ratio
        sharpe_ratio = results.get("sharpe_ratio", 0)
        sharpe_ratio_valid = abs(sharpe_ratio) <= config["max_sharpe_ratio"]
        if not sharpe_ratio_valid:
            issues.append(f"Sharpe ratio {sharpe_ratio:.2f} exceeds realistic bounds")
        
        # Validate win rate
        win_rate = results.get("win_rate", 0)
        win_rate_bounds = config["win_rate_bounds"]
        win_rate_valid = win_rate_bounds[0] <= win_rate <= win_rate_bounds[1]
        if not win_rate_valid:
            issues.append(f"Win rate {win_rate:.2%} outside valid bounds")
        
        # Validate profit factor
        profit_factor = results.get("profit_factor", 0)
        pf_bounds = config["profit_factor_bounds"]
        profit_factor_valid = pf_bounds[0] <= profit_factor <= pf_bounds[1]
        if not profit_factor_valid:
            issues.append(f"Profit factor {profit_factor:.2f} outside valid bounds")
        
        # Validate drawdown
        max_drawdown = results.get("max_drawdown", 0)
        drawdown_valid = -1.0 <= max_drawdown <= 0  # Drawdown should be negative or zero
        if not drawdown_valid:
            issues.append(f"Max drawdown {max_drawdown:.2%} has invalid value")
        
        # Validate trade count
        total_trades = results.get("total_trades", 0)
        trade_count_valid = config["min_trade_count"] <= total_trades <= config["max_trade_count"]
        if not trade_count_valid:
            issues.append(f"Trade count {total_trades} outside valid bounds")
        
        validation_passed = all([
            sharpe_ratio_valid, win_rate_valid, profit_factor_valid, 
            drawdown_valid, trade_count_valid
        ])
        
        return {
            "sharpe_ratio_valid": sharpe_ratio_valid,
            "win_rate_valid": win_rate_valid,
            "profit_factor_valid": profit_factor_valid,
            "drawdown_valid": drawdown_valid,
            "trade_count_valid": trade_count_valid,
            "validation_passed": validation_passed,
            "calculation_errors": [],
            "issues": issues
        }
    
    async def validate_trade_history(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate trade history consistency.
        
        Args:
            results: Backtest results
            
        Returns:
            Trade history validation results
        """
        trade_history = results.get("trade_history", [])
        total_trades = results.get("total_trades", 0)
        issues = []
        
        # Check trade count consistency
        trade_count_matches = len(trade_history) == total_trades
        if not trade_count_matches:
            issues.append(f"Trade history length {len(trade_history)} doesn't match total_trades {total_trades}")
        
        # Validate timestamps
        timestamps_valid = True
        if trade_history:
            for i, trade in enumerate(trade_history):
                if "timestamp" not in trade:
                    timestamps_valid = False
                    issues.append(f"Trade {i} missing timestamp")
                elif not isinstance(trade["timestamp"], datetime):
                    timestamps_valid = False
                    issues.append(f"Trade {i} has invalid timestamp format")
        
        # Validate PnL consistency
        pnl_consistency = True
        if trade_history:
            total_pnl = sum(trade.get("pnl", 0) for trade in trade_history)
            expected_return = results.get("total_return", 0) * 10000  # Assume $10k portfolio
            
            if abs(total_pnl - expected_return) > expected_return * 0.1:  # 10% tolerance
                pnl_consistency = False
                issues.append("Trade PnL sum doesn't match total return")
        
        # Validate position sizes
        position_size_valid = True
        if trade_history:
            for i, trade in enumerate(trade_history):
                size = trade.get("size", 0)
                if size <= 0 or size > 1:  # Position size should be 0-100% of portfolio
                    position_size_valid = False
                    issues.append(f"Trade {i} has invalid position size {size}")
        
        validation_passed = all([
            trade_count_matches, timestamps_valid, pnl_consistency, position_size_valid
        ])
        
        return {
            "trade_count_matches": trade_count_matches,
            "timestamps_valid": timestamps_valid,
            "pnl_consistency": pnl_consistency,
            "position_size_valid": position_size_valid,
            "validation_passed": validation_passed,
            "issues": issues
        }
    
    # Computational Resource Monitoring
    async def start_resource_monitoring(self, monitor_id: str) -> None:
        """
        Start resource monitoring for an operation.
        
        Args:
            monitor_id: Unique identifier for the monitoring session
        """
        if monitor_id in self._active_monitors:
            self.logger.warning("Monitor already active", monitor_id=monitor_id)
            return
        
        self._active_monitors[monitor_id] = {
            "start_time": datetime.now(),
            "initial_memory": await self.get_current_memory_usage(),
            "peak_memory": 0,
            "peak_cpu": 0,
            "memory_samples": [],
            "cpu_samples": []
        }
        
        # Start monitoring task
        self._active_monitors[monitor_id]["monitor_task"] = asyncio.create_task(
            self._monitor_resources(monitor_id)
        )
        
        self.logger.info("Resource monitoring started", monitor_id=monitor_id)
    
    async def stop_resource_monitoring(self, monitor_id: str) -> Dict[str, Any]:
        """
        Stop resource monitoring and return results.
        
        Args:
            monitor_id: Monitor identifier
            
        Returns:
            Monitoring results
        """
        if monitor_id not in self._active_monitors:
            self.logger.warning("Monitor not found", monitor_id=monitor_id)
            return {}
        
        monitor = self._active_monitors[monitor_id]
        
        # Cancel monitoring task
        if "monitor_task" in monitor:
            monitor["monitor_task"].cancel()
            try:
                await monitor["monitor_task"]
            except asyncio.CancelledError:
                pass
        
        # Calculate final results
        end_time = datetime.now()
        duration = (end_time - monitor["start_time"]).total_seconds()
        final_memory = await self.get_current_memory_usage()
        
        results = {
            "duration_seconds": duration,
            "initial_memory_mb": monitor["initial_memory"]["memory_mb"],
            "final_memory_mb": final_memory["memory_mb"],
            "peak_memory_mb": monitor["peak_memory"],
            "peak_cpu_percent": monitor["peak_cpu"],
            "memory_growth": final_memory["memory_mb"] - monitor["initial_memory"]["memory_mb"]
        }
        
        # Clean up
        del self._active_monitors[monitor_id]
        
        self.logger.info(
            "Resource monitoring stopped",
            monitor_id=monitor_id,
            duration=duration,
            peak_memory=monitor["peak_memory"]
        )
        
        return results
    
    async def _monitor_resources(self, monitor_id: str) -> None:
        """
        Background task to monitor resources.
        
        Args:
            monitor_id: Monitor identifier
        """
        monitor = self._active_monitors[monitor_id]
        check_interval = self.config["resource_monitoring"]["check_interval_seconds"]
        
        try:
            while monitor_id in self._active_monitors:
                # Get current resource usage
                memory_usage = await self.get_current_memory_usage()
                cpu_usage = await self.get_current_cpu_usage()
                
                # Update peak values
                monitor["peak_memory"] = max(monitor["peak_memory"], memory_usage["memory_mb"])
                monitor["peak_cpu"] = max(monitor["peak_cpu"], cpu_usage["cpu_percent"])
                
                # Store samples
                monitor["memory_samples"].append(memory_usage["memory_mb"])
                monitor["cpu_samples"].append(cpu_usage["cpu_percent"])
                
                # Check limits
                await self._check_resource_limits(monitor_id, memory_usage, cpu_usage)
                
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error("Resource monitoring error", monitor_id=monitor_id, error=str(e))
    
    async def _check_resource_limits(self, monitor_id: str, memory_usage: Dict[str, Any], cpu_usage: Dict[str, Any]) -> None:
        """
        Check if resource limits are exceeded.
        
        Args:
            monitor_id: Monitor identifier
            memory_usage: Current memory usage
            cpu_usage: Current CPU usage
        """
        config = self.config["resource_monitoring"]
        
        # Check memory limit
        if memory_usage["memory_mb"] > config["max_memory_usage_mb"]:
            raise ResourceLimitExceededError(
                f"Memory usage {memory_usage['memory_mb']}MB exceeds limit {config['max_memory_usage_mb']}MB",
                context={"monitor_id": monitor_id, "memory_usage": memory_usage}
            )
        
        # Check CPU limit
        if cpu_usage["cpu_percent"] > config["max_cpu_usage_percent"]:
            raise ResourceLimitExceededError(
                f"CPU usage {cpu_usage['cpu_percent']}% exceeds limit {config['max_cpu_usage_percent']}%",
                context={"monitor_id": monitor_id, "cpu_usage": cpu_usage}
            )
        
        # Check execution time limit
        monitor = self._active_monitors.get(monitor_id)
        if monitor:
            duration = (datetime.now() - monitor["start_time"]).total_seconds()
            if duration > config["max_execution_time_seconds"]:
                raise ExecutionTimeoutError(
                    f"Execution time {duration}s exceeds limit {config['max_execution_time_seconds']}s",
                    context={"monitor_id": monitor_id, "duration": duration}
                )
    
    async def get_current_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage.
        
        Returns:
            Memory usage information
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    async def get_current_cpu_usage(self) -> Dict[str, Any]:
        """
        Get current CPU usage.
        
        Returns:
            CPU usage information
        """
        process = psutil.Process()
        
        return {
            "cpu_percent": process.cpu_percent(),
            "cpu_count": psutil.cpu_count(),
            "system_cpu_percent": psutil.cpu_percent()
        }
    
    @asynccontextmanager
    async def monitor_execution_time(self, operation_name: str, max_time_seconds: Optional[float] = None):
        """
        Context manager for monitoring execution time.
        
        Args:
            operation_name: Name of the operation
            max_time_seconds: Maximum allowed execution time
        """
        if max_time_seconds is None:
            max_time_seconds = self.config["resource_monitoring"]["max_execution_time_seconds"]
        
        start_time = datetime.now()
        
        try:
            # Start timeout task
            timeout_task = asyncio.create_task(asyncio.sleep(max_time_seconds))
            
            yield
            
            # Cancel timeout if operation completed
            timeout_task.cancel()
            
        except asyncio.CancelledError:
            raise ExecutionTimeoutError(
                f"Operation '{operation_name}' timed out after {max_time_seconds}s"
            )
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.debug(
                "Operation completed",
                operation=operation_name,
                duration=duration
            )
    
    async def enforce_resource_limits(self) -> None:
        """
        Check and enforce current resource limits.
        
        Raises:
            ResourceLimitExceededError: If limits are exceeded
        """
        memory_usage = await self.get_current_memory_usage()
        cpu_usage = await self.get_current_cpu_usage()
        
        config = self.config["resource_monitoring"]
        
        if memory_usage["memory_mb"] > config["max_memory_usage_mb"]:
            raise ResourceLimitExceededError(
                f"Current memory usage {memory_usage['memory_mb']}MB exceeds limit",
                context={"memory_usage": memory_usage}
            )
        
        if cpu_usage["cpu_percent"] > config["max_cpu_usage_percent"]:
            raise ResourceLimitExceededError(
                f"Current CPU usage {cpu_usage['cpu_percent']}% exceeds limit",
                context={"cpu_usage": cpu_usage}
            )
    
    async def check_runaway_analysis(self, analysis_metrics: Dict[str, Any]) -> None:
        """
        Check for runaway analysis operations.
        
        Args:
            analysis_metrics: Analysis execution metrics
            
        Raises:
            RunawayAnalysisError: If runaway analysis is detected
        """
        config = self.config["resource_monitoring"]
        issues = []
        
        execution_time = analysis_metrics.get("execution_time", 0)
        memory_usage = analysis_metrics.get("memory_usage", 0)
        cpu_usage = analysis_metrics.get("cpu_usage", 0)
        
        if execution_time > config["max_execution_time_seconds"]:
            issues.append(f"Execution time {execution_time}s exceeds limit")
        
        if memory_usage > config["max_memory_usage_mb"]:
            issues.append(f"Memory usage {memory_usage}MB exceeds limit")
        
        if cpu_usage > config["max_cpu_usage_percent"]:
            issues.append(f"CPU usage {cpu_usage}% exceeds limit")
        
        if issues:
            raise RunawayAnalysisError(
                f"Runaway analysis detected: {'; '.join(issues)}",
                context=analysis_metrics
            )
    
    async def analyze_memory_leak(self, monitor_id: str) -> Dict[str, Any]:
        """
        Analyze potential memory leaks.
        
        Args:
            monitor_id: Monitor identifier
            
        Returns:
            Memory leak analysis results
        """
        if monitor_id not in self._active_monitors:
            return {"memory_trend": "unknown", "leak_detected": False, "memory_growth_rate": 0}
        
        monitor = self._active_monitors[monitor_id]
        memory_samples = monitor["memory_samples"]
        
        if len(memory_samples) < 3:
            return {"memory_trend": "insufficient_data", "leak_detected": False, "memory_growth_rate": 0}
        
        # Calculate memory growth trend
        recent_samples = memory_samples[-min(10, len(memory_samples)):]
        if len(recent_samples) >= 2:
            growth_rate = (recent_samples[-1] - recent_samples[0]) / len(recent_samples)
        else:
            growth_rate = 0
        
        # Determine trend
        if growth_rate > 1.0:  # Growing more than 1MB per sample
            trend = "increasing"
            leak_detected = growth_rate > 5.0  # More than 5MB per sample
        elif growth_rate < -1.0:
            trend = "decreasing"
            leak_detected = False
        else:
            trend = "stable"
            leak_detected = False
        
        return {
            "memory_trend": trend,
            "leak_detected": leak_detected,
            "memory_growth_rate": growth_rate,
            "peak_memory": monitor["peak_memory"],
            "sample_count": len(memory_samples)
        }
    
    # Analysis Quality Scoring
    async def calculate_analysis_quality_score(self, quality_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive analysis quality score.
        
        Args:
            quality_inputs: Quality assessment inputs
            
        Returns:
            Quality score results
        """
        try:
            weights = self.config["quality_scoring"]["weights"]
            
            # Calculate component scores
            component_scores = await self.calculate_component_scores(quality_inputs)
            
            # Calculate weighted overall score
            weighted_score = 0.0
            for component, weight in weights.items():
                score_key = f"{component}_score"
                if score_key in component_scores:
                    weighted_score += component_scores[score_key] * weight
                else:
                    # Use direct value if available
                    weighted_score += quality_inputs.get(component, 0) * weight
            
            # Assign quality grade
            quality_grade = await self.assign_quality_grade(weighted_score)
            
            # Generate improvement suggestions
            improvement_suggestions = await self.generate_improvement_suggestions(quality_inputs)
            
            return {
                "overall_score": min(1.0, max(0.0, weighted_score)),
                "component_scores": component_scores,
                "weighted_score": weighted_score,
                "quality_grade": quality_grade,
                "improvement_suggestions": improvement_suggestions
            }
            
        except Exception as e:
            self.logger.error("Quality score calculation failed", error=str(e))
            return {
                "overall_score": 0.0,
                "component_scores": {},
                "weighted_score": 0.0,
                "quality_grade": "F",
                "improvement_suggestions": ["Error in quality calculation"]
            }
    
    async def calculate_component_scores(self, quality_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate individual component scores.
        
        Args:
            quality_inputs: Quality assessment inputs
            
        Returns:
            Component scores dictionary
        """
        scores = {}
        
        # Data completeness score
        data_completeness = quality_inputs.get("data_completeness", 0)
        scores["data_completeness_score"] = min(1.0, max(0.0, data_completeness))
        
        # Temporal coverage score
        temporal_coverage = quality_inputs.get("temporal_coverage", 0)
        scores["temporal_coverage_score"] = min(1.0, max(0.0, temporal_coverage))
        
        # Result consistency score
        result_consistency = quality_inputs.get("result_consistency", 0)
        scores["result_consistency_score"] = min(1.0, max(0.0, result_consistency))
        
        # Statistical validity score
        statistical_validity = quality_inputs.get("statistical_validity", 0)
        scores["statistical_validity_score"] = min(1.0, max(0.0, statistical_validity))
        
        # Execution efficiency bonus
        execution_metrics = quality_inputs.get("execution_metrics", {})
        if execution_metrics:
            memory_efficiency = execution_metrics.get("memory_efficiency", 0)
            execution_time = execution_metrics.get("execution_time", float('inf'))
            error_count = execution_metrics.get("error_count", 0)
            
            # Efficiency bonus (0-0.1 bonus points)
            efficiency_bonus = 0.0
            if memory_efficiency > 0.8:
                efficiency_bonus += 0.03
            if execution_time < 30:  # Fast execution
                efficiency_bonus += 0.03
            if error_count == 0:
                efficiency_bonus += 0.04
            
            scores["efficiency_bonus"] = efficiency_bonus
        
        return scores
    
    async def assign_quality_grade(self, score: float) -> str:
        """
        Assign quality grade based on score.
        
        Args:
            score: Overall quality score (0-1)
            
        Returns:
            Quality grade (A-F)
        """
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    async def generate_improvement_suggestions(self, quality_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate improvement suggestions based on quality inputs.
        
        Args:
            quality_inputs: Quality assessment inputs
            
        Returns:
            Improvement suggestions
        """
        suggestions = {
            "data_quality_suggestions": [],
            "temporal_coverage_suggestions": [],
            "statistical_suggestions": [],
            "prioritized_actions": []
        }
        
        # Data completeness suggestions
        data_completeness = quality_inputs.get("data_completeness", 1.0)
        if data_completeness < 0.8:
            suggestions["data_quality_suggestions"].append(
                "Improve data collection to increase completeness above 80%"
            )
            suggestions["prioritized_actions"].append({
                "action": "Enhance data collection",
                "priority": "high",
                "impact": "data_quality"
            })
        
        # Temporal coverage suggestions
        temporal_coverage = quality_inputs.get("temporal_coverage", 1.0)
        if temporal_coverage < 0.7:
            suggestions["temporal_coverage_suggestions"].append(
                "Extend data collection period to improve temporal coverage"
            )
            suggestions["prioritized_actions"].append({
                "action": "Extend data timeframe",
                "priority": "medium",
                "impact": "temporal_coverage"
            })
        
        # Statistical validity suggestions
        statistical_validity = quality_inputs.get("statistical_validity", 1.0)
        if statistical_validity < 0.8:
            suggestions["statistical_suggestions"].append(
                "Review statistical calculations and outlier handling"
            )
            suggestions["prioritized_actions"].append({
                "action": "Improve statistical methods",
                "priority": "medium",
                "impact": "statistical_validity"
            })
        
        # Execution efficiency suggestions
        execution_metrics = quality_inputs.get("execution_metrics", {})
        if execution_metrics:
            memory_efficiency = execution_metrics.get("memory_efficiency", 1.0)
            if memory_efficiency < 0.7:
                suggestions["prioritized_actions"].append({
                    "action": "Optimize memory usage",
                    "priority": "low",
                    "impact": "performance"
                })
        
        # Sort prioritized actions by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        suggestions["prioritized_actions"].sort(
            key=lambda x: priority_order.get(x["priority"], 0),
            reverse=True
        )
        
        return suggestions
    
    async def store_quality_score(self, analysis_id: str, quality_score: Dict[str, Any]) -> None:
        """
        Store quality score for tracking.
        
        Args:
            analysis_id: Analysis identifier
            quality_score: Quality score results
        """
        # Store in metrics history for now (could be enhanced with database storage)
        self._metrics_history.append({
            "analysis_id": analysis_id,
            "timestamp": datetime.now(),
            "quality_score": quality_score["overall_score"],
            "quality_grade": quality_score["quality_grade"],
            "component_scores": quality_score["component_scores"]
        })
        
        # Keep only recent history (last 100 entries)
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]
        
        self.logger.info(
            "Quality score stored",
            analysis_id=analysis_id,
            score=quality_score["overall_score"],
            grade=quality_score["quality_grade"]
        )
    
    async def get_quality_score(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored quality score.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Quality score results or None if not found
        """
        for entry in reversed(self._metrics_history):
            if entry["analysis_id"] == analysis_id:
                return {
                    "overall_score": entry["quality_score"],
                    "quality_grade": entry["quality_grade"],
                    "component_scores": entry["component_scores"],
                    "timestamp": entry["timestamp"]
                }
        return None
    
    # Integration and Utility Methods
    async def validate_analysis_operation(self, validation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete analysis operation.
        
        Args:
            validation_context: Context for validation
            
        Returns:
            Comprehensive validation results
        """
        operation = validation_context.get("operation", "unknown")
        
        self.logger.info("Starting analysis operation validation", operation=operation)
        
        # Pre-validation checks
        pre_validation = {
            "resource_availability": await self._check_resource_availability(),
            "configuration_valid": True,  # Assume valid for now
            "dependencies_available": True  # Assume available for now
        }
        
        # Runtime monitoring setup
        monitor_id = f"validation_{operation}_{uuid4().hex[:8]}"
        await self.start_resource_monitoring(monitor_id)
        
        try:
            # Simulate operation monitoring
            await asyncio.sleep(0.1)  # Placeholder for actual operation
            
            # Post-validation checks
            monitoring_results = await self.stop_resource_monitoring(monitor_id)
            
            post_validation = {
                "resource_usage_acceptable": monitoring_results.get("peak_memory_mb", 0) < 500,
                "execution_time_acceptable": monitoring_results.get("duration_seconds", 0) < 60,
                "no_errors_detected": True  # Placeholder
            }
            
            overall_validation_passed = (
                all(pre_validation.values()) and
                all(post_validation.values())
            )
            
            return {
                "pre_validation": pre_validation,
                "runtime_monitoring": monitoring_results,
                "post_validation": post_validation,
                "overall_validation_passed": overall_validation_passed
            }
            
        except Exception as e:
            await self.stop_resource_monitoring(monitor_id)
            self.logger.error("Analysis operation validation failed", operation=operation, error=str(e))
            raise ValidationError(f"Analysis operation validation failed: {str(e)}")
    
    async def _check_resource_availability(self) -> bool:
        """
        Check if sufficient resources are available.
        
        Returns:
            True if resources are available
        """
        try:
            memory_usage = await self.get_current_memory_usage()
            available_memory = memory_usage["available_memory_mb"]
            required_memory = self.config["resource_monitoring"]["max_memory_usage_mb"]
            
            return available_memory > required_memory * 1.5  # 50% buffer
        except Exception:
            return False
    
    def _update_validation_stats(self, success: bool, duration: float) -> None:
        """
        Update validation statistics.
        
        Args:
            success: Whether validation was successful
            duration: Validation duration in seconds
        """
        self._validation_stats["validations_performed"] += 1
        if success:
            self._validation_stats["validation_success_count"] += 1
        self._validation_stats["total_validation_time"] += duration
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """
        Get validation metrics.
        
        Returns:
            Validation metrics dictionary
        """
        stats = self._validation_stats
        total_validations = stats["validations_performed"]
        
        if total_validations == 0:
            return {
                "validations_performed": 0,
                "validation_success_rate": 0.0,
                "average_validation_time": 0.0,
                "resource_usage_stats": {}
            }
        
        return {
            "validations_performed": total_validations,
            "validation_success_rate": stats["validation_success_count"] / total_validations,
            "average_validation_time": stats["total_validation_time"] / total_validations,
            "resource_usage_stats": stats["resource_usage_peaks"]
        }
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Performance statistics dictionary
        """
        stats = self._validation_stats
        
        return {
            "total_validations": stats["validations_performed"],
            "average_execution_time": (
                stats["total_validation_time"] / stats["validations_performed"]
                if stats["validations_performed"] > 0 else 0
            ),
            "peak_memory_usage": max(
                stats["resource_usage_peaks"].values()
                if stats["resource_usage_peaks"] else [0]
            ),
            "active_monitors": len(self._active_monitors)
        }