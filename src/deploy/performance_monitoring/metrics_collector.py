"""
Metrics Collector

Advanced metrics collection system for performance monitoring.
Provides metrics aggregation, storage, and retrieval capabilities.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AggregationType(Enum):
    """Metric aggregation types"""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"


@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    type: MetricType
    unit: str
    description: str
    tags: Dict[str, str] = field(default_factory=dict)
    aggregation: AggregationType = AggregationType.AVERAGE
    retention_seconds: int = 3600  # 1 hour default


@dataclass
class MetricDataPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric result"""
    name: str
    aggregation_type: AggregationType
    value: float
    count: int
    start_time: float
    end_time: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Advanced Metrics Collector
    
    Collects, aggregates, and stores performance metrics with
    configurable retention policies and aggregation strategies.
    """
    
    def __init__(self, max_memory_mb: int = 100):
        self.max_memory_mb = max_memory_mb
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.raw_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, List[AggregatedMetric]] = defaultdict(list)
        self.collection_lock = threading.RLock()
        self.background_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.collection_stats = {
            "total_metrics_collected": 0,
            "total_aggregations_performed": 0,
            "memory_usage_mb": 0,
            "collection_errors": 0
        }
        
        logger.info(f"Metrics collector initialized with {max_memory_mb}MB memory limit")
    
    def define_metric(self, metric_def: MetricDefinition) -> Dict[str, Any]:
        """Define a new metric type"""
        try:
            with self.collection_lock:
                self.metric_definitions[metric_def.name] = metric_def
                
            logger.info(f"Defined metric: {metric_def.name} ({metric_def.type.value})")
            return {"success": True, "metric": metric_def.name}
            
        except Exception as e:
            logger.error(f"Failed to define metric {metric_def.name}: {e}")
            return {"success": False, "error": str(e)}
    
    def collect_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Collect a single metric data point"""
        try:
            timestamp = time.time()
            tags = tags or {}
            
            # Check if metric is defined
            if name not in self.metric_definitions:
                # Auto-define as gauge metric
                self.define_metric(MetricDefinition(
                    name=name,
                    type=MetricType.GAUGE,
                    unit="count",
                    description=f"Auto-defined metric: {name}"
                ))
            
            data_point = MetricDataPoint(
                name=name,
                value=value,
                timestamp=timestamp,
                tags=tags
            )
            
            with self.collection_lock:
                self.raw_metrics[name].append(data_point)
                self.collection_stats["total_metrics_collected"] += 1
            
            # Clean up old metrics if needed
            self._cleanup_old_metrics()
            
            return {"success": True, "metric": name, "timestamp": timestamp}
            
        except Exception as e:
            self.collection_stats["collection_errors"] += 1
            logger.error(f"Failed to collect metric {name}: {e}")
            return {"success": False, "error": str(e)}
    
    def collect_metrics_batch(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect multiple metrics in batch"""
        successful = 0
        failed = 0
        
        for metric in metrics:
            name = metric.get("name")
            value = metric.get("value")
            tags = metric.get("tags", {})
            
            if name is None or value is None:
                failed += 1
                continue
            
            result = self.collect_metric(name, value, tags)
            if result["success"]:
                successful += 1
            else:
                failed += 1
        
        return {
            "success": failed == 0,
            "successful_count": successful,
            "failed_count": failed,
            "total_count": len(metrics)
        }
    
    def get_raw_metrics(self, name: str, time_range_seconds: Optional[int] = None) -> List[MetricDataPoint]:
        """Get raw metric data points"""
        if name not in self.raw_metrics:
            return []
        
        with self.collection_lock:
            metrics = list(self.raw_metrics[name])
        
        if time_range_seconds:
            cutoff_time = time.time() - time_range_seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return sorted(metrics, key=lambda x: x.timestamp)
    
    def aggregate_metrics(self, name: str, aggregation: AggregationType, 
                         time_range_seconds: int, bucket_size_seconds: int = 60) -> List[AggregatedMetric]:
        """Aggregate metrics over time buckets"""
        try:
            raw_data = self.get_raw_metrics(name, time_range_seconds)
            if not raw_data:
                return []
            
            # Group data points into time buckets
            end_time = time.time()
            start_time = end_time - time_range_seconds
            
            buckets = {}
            bucket_count = int(time_range_seconds / bucket_size_seconds)
            
            for i in range(bucket_count):
                bucket_start = start_time + (i * bucket_size_seconds)
                bucket_end = bucket_start + bucket_size_seconds
                buckets[bucket_start] = {
                    "start": bucket_start,
                    "end": bucket_end,
                    "data": []
                }
            
            # Assign data points to buckets
            for data_point in raw_data:
                bucket_key = int((data_point.timestamp - start_time) / bucket_size_seconds) * bucket_size_seconds + start_time
                if bucket_key in buckets:
                    buckets[bucket_key]["data"].append(data_point)
            
            # Calculate aggregations for each bucket
            aggregated_results = []
            for bucket_start, bucket_data in buckets.items():
                if not bucket_data["data"]:
                    continue
                
                values = [dp.value for dp in bucket_data["data"]]
                aggregated_value = self._calculate_aggregation(values, aggregation)
                
                aggregated_metric = AggregatedMetric(
                    name=name,
                    aggregation_type=aggregation,
                    value=aggregated_value,
                    count=len(values),
                    start_time=bucket_data["start"],
                    end_time=bucket_data["end"]
                )
                
                aggregated_results.append(aggregated_metric)
            
            # Store aggregated results
            with self.collection_lock:
                if name not in self.aggregated_metrics:
                    self.aggregated_metrics[name] = []
                self.aggregated_metrics[name].extend(aggregated_results)
                self.collection_stats["total_aggregations_performed"] += len(aggregated_results)
                
                # Keep only recent aggregated data
                retention_limit = time.time() - 7200  # 2 hours
                self.aggregated_metrics[name] = [
                    am for am in self.aggregated_metrics[name]
                    if am.end_time >= retention_limit
                ]
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Failed to aggregate metrics for {name}: {e}")
            return []
    
    def _calculate_aggregation(self, values: List[float], aggregation: AggregationType) -> float:
        """Calculate aggregation value"""
        if not values:
            return 0.0
        
        if aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.AVERAGE:
            return statistics.mean(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.COUNT:
            return float(len(values))
        elif aggregation == AggregationType.PERCENTILE:
            # Default to P95
            return self._calculate_percentile(values, 95)
        else:
            return statistics.mean(values)
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def get_metric_statistics(self, name: str, time_range_seconds: int = 3600) -> Dict[str, Any]:
        """Get comprehensive statistics for a metric"""
        raw_data = self.get_raw_metrics(name, time_range_seconds)
        
        if not raw_data:
            return {"error": "No data available"}
        
        values = [dp.value for dp in raw_data]
        
        stats = {
            "metric_name": name,
            "time_range_seconds": time_range_seconds,
            "data_points": len(values),
            "latest_value": values[-1] if values else 0,
            "latest_timestamp": raw_data[-1].timestamp if raw_data else 0,
            "statistics": {
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "p50": self._calculate_percentile(values, 50),
                "p95": self._calculate_percentile(values, 95),
                "p99": self._calculate_percentile(values, 99)
            },
            "trend": self._calculate_trend(values),
            "quality": self._assess_data_quality(raw_data)
        }
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 10:
            return "insufficient_data"
        
        # Compare recent quarter with previous quarter
        quarter_size = len(values) // 4
        recent_quarter = values[-quarter_size:]
        previous_quarter = values[-(2*quarter_size):-quarter_size]
        
        if not previous_quarter:
            return "insufficient_data"
        
        recent_avg = statistics.mean(recent_quarter)
        previous_avg = statistics.mean(previous_quarter)
        
        change_percent = (recent_avg - previous_avg) / previous_avg * 100
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"
    
    def _assess_data_quality(self, data_points: List[MetricDataPoint]) -> Dict[str, Any]:
        """Assess quality of metric data"""
        if not data_points:
            return {"quality_score": 0.0, "issues": ["no_data"]}
        
        issues = []
        quality_score = 1.0
        
        # Check for data gaps
        if len(data_points) > 1:
            timestamps = [dp.timestamp for dp in data_points]
            time_gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_gap = statistics.mean(time_gaps)
            max_gap = max(time_gaps)
            
            if max_gap > avg_gap * 3:  # Large gaps
                issues.append("irregular_intervals")
                quality_score *= 0.9
        
        # Check for outliers
        values = [dp.value for dp in data_points]
        if len(values) > 10:
            q1 = self._calculate_percentile(values, 25)
            q3 = self._calculate_percentile(values, 75)
            iqr = q3 - q1
            
            outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            outliers = [v for v in values if v < outlier_bounds[0] or v > outlier_bounds[1]]
            
            outlier_rate = len(outliers) / len(values)
            if outlier_rate > 0.1:  # More than 10% outliers
                issues.append("high_outlier_rate")
                quality_score *= 0.8
        
        # Check for constant values
        if len(set(values)) == 1:
            issues.append("constant_values")
            quality_score *= 0.7
        
        return {
            "quality_score": quality_score,
            "issues": issues,
            "data_completeness": len(data_points) / (len(data_points) + len(issues))
        }
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to manage memory usage"""
        current_time = time.time()
        
        with self.collection_lock:
            for name, metric_def in self.metric_definitions.items():
                if name in self.raw_metrics:
                    cutoff_time = current_time - metric_def.retention_seconds
                    
                    # Filter out old data points
                    old_deque = self.raw_metrics[name]
                    new_deque = deque(
                        [dp for dp in old_deque if dp.timestamp >= cutoff_time],
                        maxlen=old_deque.maxlen
                    )
                    self.raw_metrics[name] = new_deque
        
        # Update memory usage estimate
        self._update_memory_usage()
    
    def _update_memory_usage(self):
        """Update memory usage estimate"""
        total_data_points = sum(len(deque_obj) for deque_obj in self.raw_metrics.values())
        estimated_mb = (total_data_points * 100) / (1024 * 1024)  # Rough estimate
        self.collection_stats["memory_usage_mb"] = estimated_mb
    
    def get_all_metric_names(self) -> List[str]:
        """Get list of all defined metric names"""
        return list(self.metric_definitions.keys())
    
    def get_metric_definition(self, name: str) -> Optional[MetricDefinition]:
        """Get metric definition by name"""
        return self.metric_definitions.get(name)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get metrics collection statistics"""
        with self.collection_lock:
            stats = self.collection_stats.copy()
            stats.update({
                "defined_metrics": len(self.metric_definitions),
                "active_metrics": len(self.raw_metrics),
                "aggregated_metrics": len(self.aggregated_metrics)
            })
        
        return stats
    
    def export_metrics(self, format_type: str = "json", time_range_seconds: int = 3600) -> Union[str, Dict[str, Any]]:
        """Export metrics data in specified format"""
        try:
            export_data = {}
            
            for name in self.get_all_metric_names():
                metrics_data = self.get_raw_metrics(name, time_range_seconds)
                export_data[name] = [
                    {
                        "timestamp": dp.timestamp,
                        "value": dp.value,
                        "tags": dp.tags
                    }
                    for dp in metrics_data
                ]
            
            export_metadata = {
                "export_timestamp": time.time(),
                "time_range_seconds": time_range_seconds,
                "metrics_count": len(export_data),
                "total_data_points": sum(len(data) for data in export_data.values())
            }
            
            full_export = {
                "metadata": export_metadata,
                "metrics": export_data
            }
            
            if format_type.lower() == "json":
                return json.dumps(full_export, indent=2)
            else:
                return full_export
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return {"error": str(e)}
    
    async def start_auto_aggregation(self, interval_seconds: int = 300):
        """Start automatic metric aggregation"""
        async def aggregation_loop():
            while True:
                try:
                    for name in self.get_all_metric_names():
                        metric_def = self.get_metric_definition(name)
                        if metric_def:
                            self.aggregate_metrics(
                                name=name,
                                aggregation=metric_def.aggregation,
                                time_range_seconds=3600,  # 1 hour
                                bucket_size_seconds=60    # 1 minute buckets
                            )
                    
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in auto-aggregation loop: {e}")
                    await asyncio.sleep(interval_seconds)
        
        task = asyncio.create_task(aggregation_loop())
        self.background_tasks.append(task)
        logger.info(f"Started auto-aggregation with {interval_seconds}s interval")
        
        return {"success": True, "interval_seconds": interval_seconds}
    
    async def stop_background_tasks(self):
        """Stop all background tasks"""
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
        logger.info("Stopped all background tasks")