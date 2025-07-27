"""
Base classes for metrics collection and monitoring.

This module provides the foundation for all metrics collection in the Shyvr RLTE system,
including Prometheus integration and common patterns for metrics collectors.
"""

import time
import functools
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable

from prometheus_client import (
    CollectorRegistry, 
    Counter, 
    Gauge, 
    Histogram, 
    generate_latest
)


class MetricsRegistry:
    """
    Registry for managing Prometheus metrics with a custom registry.
    
    This class provides a centralized way to create and manage metrics,
    ensuring consistent naming and avoiding conflicts with other metrics.
    """
    
    def __init__(self):
        """Initialize with a custom registry."""
        self.registry = CollectorRegistry()
        self._metrics_cache: Dict[str, Any] = {}
    
    def get_counter(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None
    ) -> Counter:
        """
        Get or create a Counter metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            
        Returns:
            Counter metric instance
        """
        if name in self._metrics_cache:
            return self._metrics_cache[name]
        
        counter = Counter(
            name=name,
            documentation=description,
            labelnames=labels or [],
            registry=self.registry
        )
        self._metrics_cache[name] = counter
        return counter
    
    def get_gauge(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None
    ) -> Gauge:
        """
        Get or create a Gauge metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            
        Returns:
            Gauge metric instance
        """
        if name in self._metrics_cache:
            return self._metrics_cache[name]
        
        gauge = Gauge(
            name=name,
            documentation=description,
            labelnames=labels or [],
            registry=self.registry
        )
        self._metrics_cache[name] = gauge
        return gauge
    
    def get_histogram(
        self, 
        name: str, 
        description: str, 
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None
    ) -> Histogram:
        """
        Get or create a Histogram metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            buckets: Histogram buckets (optional)
            
        Returns:
            Histogram metric instance
        """
        if name in self._metrics_cache:
            return self._metrics_cache[name]
        
        kwargs = {
            "name": name,
            "documentation": description,
            "labelnames": labels or [],
            "registry": self.registry
        }
        
        if buckets:
            kwargs["buckets"] = buckets
        
        histogram = Histogram(**kwargs)
        self._metrics_cache[name] = histogram
        return histogram
    
    def clear_metrics(self) -> None:
        """Clear all metrics and create a new registry."""
        self.registry = CollectorRegistry()
        self._metrics_cache.clear()
    
    def generate_output(self) -> bytes:
        """
        Generate Prometheus format output for all metrics.
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)


class MetricsCollector(ABC):
    """
    Abstract base class for all metrics collectors.
    
    This class provides common functionality for metrics collection,
    including timing decorators and health checking.
    """
    
    def __init__(self, registry: MetricsRegistry):
        """
        Initialize the metrics collector.
        
        Args:
            registry: MetricsRegistry instance for managing metrics
        """
        self.registry = registry
        self._last_collection_time: Optional[float] = None
        self._collection_errors: int = 0
    
    @abstractmethod
    def collect_metrics(self) -> None:
        """
        Collect and update metrics.
        
        This method should be implemented by subclasses to perform
        the actual metrics collection logic.
        """
        pass
    
    @abstractmethod
    def get_metric_definitions(self) -> Dict[str, str]:
        """
        Get metric definitions for this collector.
        
        Returns:
            Dictionary mapping metric names to their descriptions
        """
        pass
    
    @property
    def metric_definitions(self) -> Dict[str, str]:
        """Get metric definitions for this collector."""
        return self.get_metric_definitions()
    
    def start_collection(self) -> None:
        """
        Start metrics collection.
        
        This method handles the collection process and error tracking.
        """
        try:
            start_time = time.time()
            self.collect_metrics()
            self._last_collection_time = time.time() - start_time
            
        except Exception as e:
            self._collection_errors += 1
            raise e
    
    def is_healthy(self) -> bool:
        """
        Check if the metrics collector is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        return True  # Default implementation
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information for this collector.
        
        Returns:
            Dictionary with collector status information
        """
        definitions = self.get_metric_definitions()
        
        return {
            "name": self.__class__.__name__,
            "healthy": self.is_healthy(),
            "last_collection_time": self._last_collection_time,
            "collection_errors": self._collection_errors,
            "metrics_count": len(definitions),
            "metrics": definitions
        }
    
    @staticmethod
    def record_execution_time(method_name: str) -> Callable:
        """
        Decorator to record execution time of methods.
        
        Args:
            method_name: Name of the method for labeling
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                start_time = time.time()
                try:
                    result = func(self, *args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    # Record execution time if histogram exists
                    if hasattr(self, 'execution_histogram'):
                        self.execution_histogram.labels(method=method_name).observe(execution_time)
            return wrapper
        return decorator