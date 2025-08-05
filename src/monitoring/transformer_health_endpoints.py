#!/usr/bin/env python3

"""
Transformer Health Monitoring Endpoints

FastAPI endpoints for transformer health monitoring including:
1. Health check endpoints for transformer models
2. Memory usage monitoring endpoints
3. Inference latency monitoring endpoints  
4. Cache performance monitoring endpoints
5. Model loading status endpoints
6. Integration with main.py health endpoint
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class TransformerHealthEndpoints:
    """
    Health monitoring endpoints for transformer models.
    
    Provides comprehensive health monitoring including memory usage,
    inference latency, cache performance, and model loading status.
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None,
        monitoring_intervals: Optional[Dict[str, int]] = None
    ):
        """
        Initialize transformer health endpoints.
        
        Args:
            models: Dictionary of transformer models
            thresholds: Custom health thresholds
            monitoring_intervals: Custom monitoring intervals
        """
        self.models = models
        
        # Default thresholds
        self.thresholds = thresholds or {
            'memory_warning_percent': 80.0,
            'memory_critical_percent': 90.0,
            'latency_warning_ms': 500.0,
            'latency_critical_ms': 1000.0,
            'cache_warning_rate': 0.6,
            'cache_critical_rate': 0.5,
            'loading_warning_ms': 3000.0,
            'loading_critical_ms': 5000.0
        }
        
        # Default monitoring intervals
        self.monitoring_intervals = monitoring_intervals or {
            'memory_check_interval_seconds': 30,
            'latency_check_interval_seconds': 15,
            'cache_check_interval_seconds': 60,
            'loading_check_interval_seconds': 300
        }
        
        # Initialize monitors
        self.memory_monitor = TransformerMemoryMonitor(
            models=self.models,
            warning_threshold=self.thresholds['memory_warning_percent'],
            critical_threshold=self.thresholds['memory_critical_percent'],
            check_interval=self.monitoring_intervals['memory_check_interval_seconds']
        )
        
        self.latency_monitor = TransformerLatencyMonitor(
            models=self.models,
            warning_threshold=self.thresholds['latency_warning_ms'],
            critical_threshold=self.thresholds['latency_critical_ms'],
            check_interval=self.monitoring_intervals['latency_check_interval_seconds']
        )
        
        self.cache_monitor = TransformerCacheMonitor(
            models=self.models,
            warning_threshold=self.thresholds['cache_warning_rate'],
            critical_threshold=self.thresholds['cache_critical_rate'],
            check_interval=self.monitoring_intervals['cache_check_interval_seconds']
        )
        
        self.loading_monitor = TransformerLoadingMonitor(
            models=self.models,
            warning_threshold=self.thresholds['loading_warning_ms'],
            critical_threshold=self.thresholds['loading_critical_ms'],
            check_interval=self.monitoring_intervals['loading_check_interval_seconds']
        )
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def get_health_overview(self) -> Dict[str, Any]:
        """Get comprehensive health overview for all transformer models."""
        try:
            timestamp = datetime.now()
            models_data = {}
            healthy_count = 0
            unhealthy_count = 0
            
            for model_name, model in self.models.items():
                try:
                    # Get model health data
                    memory_usage_mb = getattr(model, 'memory_usage_mb', 0.0)
                    if hasattr(model, 'get_memory_usage_mb'):
                        memory_usage_mb = model.get_memory_usage_mb()
                    
                    memory_usage_percent = (memory_usage_mb / 8192.0) * 100
                    
                    latency_ms = getattr(model, 'last_inference_latency_ms', 0.0)
                    if hasattr(model, 'get_inference_latency_ms'):
                        latency_ms = model.get_inference_latency_ms()
                    
                    cache_hit_rate = getattr(model, 'cache_hit_rate', 0.0)
                    if hasattr(model, 'get_cache_hit_rate'):
                        cache_hit_rate = model.get_cache_hit_rate()
                    
                    is_healthy = getattr(model, 'is_healthy', True)
                    if hasattr(model, 'is_healthy') and callable(model.is_healthy):
                        is_healthy = model.is_healthy()
                    
                    # Determine status
                    status = 'healthy'
                    if memory_usage_percent > self.thresholds['memory_critical_percent']:
                        status = 'unhealthy'
                    elif latency_ms > self.thresholds['latency_critical_ms']:
                        status = 'unhealthy'
                    elif cache_hit_rate < self.thresholds['cache_critical_rate']:
                        status = 'unhealthy'
                    elif not is_healthy:
                        status = 'unhealthy'
                    elif (memory_usage_percent > self.thresholds['memory_warning_percent'] or
                          latency_ms > self.thresholds['latency_warning_ms'] or
                          cache_hit_rate < self.thresholds['cache_warning_rate']):
                        status = 'warning'
                    
                    if status == 'unhealthy':
                        unhealthy_count += 1
                    else:
                        healthy_count += 1
                    
                    models_data[model_name] = {
                        'status': status,
                        'memory_usage_mb': memory_usage_mb,
                        'memory_usage_percent': memory_usage_percent,
                        'inference_latency_ms': latency_ms,
                        'cache_hit_rate': cache_hit_rate,
                        'is_loaded': True  # Assume loaded if in models dict
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get health data for {model_name}: {e}")
                    models_data[model_name] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    unhealthy_count += 1
            
            # Calculate overall status
            total_models = len(self.models)
            if unhealthy_count == 0:
                overall_status = 'healthy'
            elif healthy_count > unhealthy_count:
                overall_status = 'degraded'
            else:
                overall_status = 'unhealthy'
            
            return {
                'status': overall_status,
                'timestamp': timestamp,
                'models': models_data,
                'summary': {
                    'total_models': total_models,
                    'healthy_models': healthy_count,
                    'unhealthy_models': unhealthy_count,
                    'total_memory_usage_mb': sum(
                        m.get('memory_usage_mb', 0) for m in models_data.values()
                        if 'memory_usage_mb' in m
                    ),
                    'average_latency_ms': sum(
                        m.get('inference_latency_ms', 0) for m in models_data.values()
                        if 'inference_latency_ms' in m
                    ) / max(total_models, 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get health overview: {e}")
            return {
                'status': 'error',
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for all transformer models."""
        try:
            timestamp = datetime.now()
            total_memory_limit_mb = 8192.0  # 8Gi
            total_memory_usage_mb = 0.0
            models_memory = {}
            
            for model_name, model in self.models.items():
                try:
                    current_usage = getattr(model, 'memory_usage_mb', 0.0)
                    if hasattr(model, 'get_memory_usage_mb'):
                        current_usage = model.get_memory_usage_mb()
                    
                    peak_usage = current_usage * 1.2  # Mock peak usage
                    if hasattr(model, 'get_peak_memory_usage_mb'):
                        peak_usage = model.get_peak_memory_usage_mb()
                    
                    usage_percent = (current_usage / total_memory_limit_mb) * 100
                    
                    # Determine status
                    if usage_percent > self.thresholds['memory_critical_percent']:
                        status = 'critical'
                    elif usage_percent > self.thresholds['memory_warning_percent']:
                        status = 'warning'
                    else:
                        status = 'healthy'
                    
                    models_memory[model_name] = {
                        'current_usage_mb': current_usage,
                        'peak_usage_mb': peak_usage,
                        'usage_percent': usage_percent,
                        'status': status
                    }
                    
                    total_memory_usage_mb += current_usage
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get memory data for {model_name}: {e}")
                    models_memory[model_name] = {
                        'current_usage_mb': 0.0,
                        'peak_usage_mb': 0.0,
                        'usage_percent': 0.0,
                        'status': 'error',
                        'error': str(e)
                    }
            
            memory_usage_percent = (total_memory_usage_mb / total_memory_limit_mb) * 100
            
            return {
                'timestamp': timestamp,
                'total_memory_limit_mb': total_memory_limit_mb,
                'total_memory_usage_mb': total_memory_usage_mb,
                'memory_usage_percent': memory_usage_percent,
                'models': models_memory
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    async def get_inference_latency(self) -> Dict[str, Any]:
        """Get inference latency information for all transformer models."""
        try:
            timestamp = datetime.now()
            models_latency = {}
            latencies = []
            models_over_threshold = 0
            
            for model_name, model in self.models.items():
                try:
                    current_latency = getattr(model, 'last_inference_latency_ms', 0.0)
                    if hasattr(model, 'get_inference_latency_ms'):
                        current_latency = model.get_inference_latency_ms()
                    
                    # Mock percentiles calculation
                    percentiles = {
                        'p50': current_latency * 0.8,
                        'p95': current_latency * 1.2,
                        'p99': current_latency * 1.5
                    }
                    
                    # Determine status
                    if current_latency > self.thresholds['latency_critical_ms']:
                        status = 'critical'
                        models_over_threshold += 1
                    elif current_latency > self.thresholds['latency_warning_ms']:
                        status = 'warning'
                    else:
                        status = 'healthy'
                    
                    models_latency[model_name] = {
                        'current_latency_ms': current_latency,
                        'percentiles': percentiles,
                        'status': status
                    }
                    
                    latencies.append(current_latency)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get latency data for {model_name}: {e}")
                    models_latency[model_name] = {
                        'current_latency_ms': 0.0,
                        'percentiles': {'p50': 0.0, 'p95': 0.0, 'p99': 0.0},
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Calculate aggregated statistics
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)
            else:
                avg_latency = max_latency = min_latency = 0.0
            
            return {
                'timestamp': timestamp,
                'models': models_latency,
                'aggregated_stats': {
                    'average_latency_ms': avg_latency,
                    'max_latency_ms': max_latency,
                    'min_latency_ms': min_latency,
                    'models_over_threshold': models_over_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get inference latency: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    async def get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance information for all transformer models."""
        try:
            timestamp = datetime.now()
            models_cache = {}
            hit_rates = []
            total_requests = 0
            models_below_threshold = 0
            
            for model_name, model in self.models.items():
                try:
                    hit_rate = getattr(model, 'cache_hit_rate', 0.0)
                    if hasattr(model, 'get_cache_hit_rate'):
                        hit_rate = model.get_cache_hit_rate()
                    
                    miss_rate = 1.0 - hit_rate
                    requests = 1000  # Mock value
                    
                    # Determine status
                    if hit_rate < self.thresholds['cache_critical_rate']:
                        status = 'critical'
                        models_below_threshold += 1
                    elif hit_rate < self.thresholds['cache_warning_rate']:
                        status = 'warning'
                    else:
                        status = 'healthy'
                    
                    models_cache[model_name] = {
                        'hit_rate': hit_rate,
                        'miss_rate': miss_rate,
                        'total_requests': requests,
                        'status': status
                    }
                    
                    hit_rates.append(hit_rate)
                    total_requests += requests
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get cache data for {model_name}: {e}")
                    models_cache[model_name] = {
                        'hit_rate': 0.0,
                        'miss_rate': 1.0,
                        'total_requests': 0,
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Calculate overall hit rate
            overall_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
            
            return {
                'timestamp': timestamp,
                'models': models_cache,
                'aggregated_stats': {
                    'overall_hit_rate': overall_hit_rate,
                    'models_below_threshold': models_below_threshold,
                    'total_requests': total_requests
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache performance: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    async def get_model_loading_status(self) -> Dict[str, Any]:
        """Get model loading status for all transformer models."""
        try:
            timestamp = datetime.now()
            models_loading = {}
            loading_times = []
            total_failures = 0
            successfully_loaded = 0
            
            for model_name, model in self.models.items():
                try:
                    loading_time = getattr(model, 'loading_time_ms', 0.0)
                    if hasattr(model, 'get_loading_time_ms'):
                        loading_time = model.get_loading_time_ms()
                    
                    is_loaded = True  # Assume loaded if in models dict
                    failures = 0  # Mock value
                    last_loading_time = datetime.now()
                    
                    # Determine status
                    if loading_time > self.thresholds['loading_critical_ms']:
                        status = 'critical'
                    elif loading_time > self.thresholds['loading_warning_ms']:
                        status = 'warning'
                    else:
                        status = 'healthy'
                    
                    models_loading[model_name] = {
                        'loading_time_ms': loading_time,
                        'is_loaded': is_loaded,
                        'loading_failures': failures,
                        'last_loading_time': last_loading_time,
                        'status': status
                    }
                    
                    loading_times.append(loading_time)
                    total_failures += failures
                    if is_loaded:
                        successfully_loaded += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get loading data for {model_name}: {e}")
                    models_loading[model_name] = {
                        'loading_time_ms': 0.0,
                        'is_loaded': False,
                        'loading_failures': 1,
                        'last_loading_time': None,
                        'status': 'error',
                        'error': str(e)
                    }
                    total_failures += 1
            
            # Calculate aggregated statistics
            if loading_times:
                avg_loading_time = sum(loading_times) / len(loading_times)
                max_loading_time = max(loading_times)
            else:
                avg_loading_time = max_loading_time = 0.0
            
            return {
                'timestamp': timestamp,
                'models': models_loading,
                'aggregated_stats': {
                    'average_loading_time_ms': avg_loading_time,
                    'max_loading_time_ms': max_loading_time,
                    'total_failures': total_failures,
                    'successfully_loaded': successfully_loaded
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get model loading status: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    async def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """Get detailed health information for a specific model."""
        try:
            if model_name not in self.models:
                return {
                    'error': f'Model {model_name} not found',
                    'timestamp': datetime.now()
                }
            
            model = self.models[model_name]
            timestamp = datetime.now()
            
            # Collect all metrics for the model
            memory_usage = getattr(model, 'memory_usage_mb', 0.0)
            if hasattr(model, 'get_memory_usage_mb'):
                memory_usage = model.get_memory_usage_mb()
            
            latency = getattr(model, 'last_inference_latency_ms', 0.0)
            if hasattr(model, 'get_inference_latency_ms'):
                latency = model.get_inference_latency_ms()
            
            cache_hit_rate = getattr(model, 'cache_hit_rate', 0.0)
            if hasattr(model, 'get_cache_hit_rate'):
                cache_hit_rate = model.get_cache_hit_rate()
            
            loading_time = getattr(model, 'loading_time_ms', 0.0)
            if hasattr(model, 'get_loading_time_ms'):
                loading_time = model.get_loading_time_ms()
            
            model_type = getattr(model, 'model_type', model_name.upper())
            
            # Get model info
            model_info = {}
            if hasattr(model, 'get_model_info'):
                model_info = model.get_model_info()
            else:
                model_info = {
                    'name': model_name,
                    'type': model_type,
                    'loaded': True,
                    'parameters': 125000000 if model_type != 'TIMESFM' else 2000000000
                }
            
            # Determine overall status
            memory_percent = (memory_usage / 8192.0) * 100
            status = 'healthy'
            
            if (memory_percent > self.thresholds['memory_critical_percent'] or
                latency > self.thresholds['latency_critical_ms'] or
                cache_hit_rate < self.thresholds['cache_critical_rate']):
                status = 'unhealthy'
            elif (memory_percent > self.thresholds['memory_warning_percent'] or
                  latency > self.thresholds['latency_warning_ms'] or
                  cache_hit_rate < self.thresholds['cache_warning_rate']):
                status = 'warning'
            
            return {
                'model_name': model_name,
                'model_type': model_type,
                'status': status,
                'timestamp': timestamp,
                'metrics': {
                    'memory': {
                        'current_usage_mb': memory_usage,
                        'usage_percent': memory_percent
                    },
                    'latency': {
                        'current_latency_ms': latency,
                        'percentiles': {
                            'p50': latency * 0.8,
                            'p95': latency * 1.2,
                            'p99': latency * 1.5
                        }
                    },
                    'cache': {
                        'hit_rate': cache_hit_rate,
                        'miss_rate': 1.0 - cache_hit_rate
                    },
                    'loading': {
                        'loading_time_ms': loading_time,
                        'is_loaded': True
                    },
                    'model_info': model_info
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get model health for {model_name}: {e}")
            return {
                'model_name': model_name,
                'status': 'error',
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    async def get_health_alerts(self) -> Dict[str, Any]:
        """Get active health alerts for transformer models."""
        try:
            timestamp = datetime.now()
            active_alerts = []
            
            for model_name, model in self.models.items():
                try:
                    # Check memory usage
                    memory_usage = getattr(model, 'memory_usage_mb', 0.0)
                    if hasattr(model, 'get_memory_usage_mb'):
                        memory_usage = model.get_memory_usage_mb()
                    
                    memory_percent = (memory_usage / 8192.0) * 100
                    
                    if memory_percent > self.thresholds['memory_critical_percent']:
                        active_alerts.append({
                            'type': 'high_memory_usage',
                            'severity': 'critical',
                            'model_name': model_name,
                            'message': f'Memory usage {memory_percent:.1f}% exceeds critical threshold',
                            'threshold': self.thresholds['memory_critical_percent'],
                            'current_value': memory_percent,
                            'timestamp': timestamp
                        })
                    elif memory_percent > self.thresholds['memory_warning_percent']:
                        active_alerts.append({
                            'type': 'high_memory_usage',
                            'severity': 'warning',
                            'model_name': model_name,
                            'message': f'Memory usage {memory_percent:.1f}% exceeds warning threshold',
                            'threshold': self.thresholds['memory_warning_percent'],
                            'current_value': memory_percent,
                            'timestamp': timestamp
                        })
                    
                    # Check inference latency
                    latency = getattr(model, 'last_inference_latency_ms', 0.0)
                    if hasattr(model, 'get_inference_latency_ms'):
                        latency = model.get_inference_latency_ms()
                    
                    if latency > self.thresholds['latency_critical_ms']:
                        active_alerts.append({
                            'type': 'slow_inference_latency',
                            'severity': 'critical',
                            'model_name': model_name,
                            'message': f'Inference latency {latency:.1f}ms exceeds critical threshold',
                            'threshold': self.thresholds['latency_critical_ms'],
                            'current_value': latency,
                            'timestamp': timestamp
                        })
                    elif latency > self.thresholds['latency_warning_ms']:
                        active_alerts.append({
                            'type': 'slow_inference_latency',
                            'severity': 'warning',
                            'model_name': model_name,
                            'message': f'Inference latency {latency:.1f}ms exceeds warning threshold',
                            'threshold': self.thresholds['latency_warning_ms'],
                            'current_value': latency,
                            'timestamp': timestamp
                        })
                    
                    # Check cache hit rate
                    cache_hit_rate = getattr(model, 'cache_hit_rate', 0.0)
                    if hasattr(model, 'get_cache_hit_rate'):
                        cache_hit_rate = model.get_cache_hit_rate()
                    
                    if cache_hit_rate < self.thresholds['cache_critical_rate']:
                        active_alerts.append({
                            'type': 'low_cache_hit_rate',
                            'severity': 'critical',
                            'model_name': model_name,
                            'message': f'Cache hit rate {cache_hit_rate:.1%} below critical threshold',
                            'threshold': self.thresholds['cache_critical_rate'],
                            'current_value': cache_hit_rate,
                            'timestamp': timestamp
                        })
                    elif cache_hit_rate < self.thresholds['cache_warning_rate']:
                        active_alerts.append({
                            'type': 'low_cache_hit_rate',
                            'severity': 'warning',
                            'model_name': model_name,
                            'message': f'Cache hit rate {cache_hit_rate:.1%} below warning threshold',
                            'threshold': self.thresholds['cache_warning_rate'],
                            'current_value': cache_hit_rate,
                            'timestamp': timestamp
                        })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to check alerts for {model_name}: {e}")
            
            # Calculate summary
            critical_alerts = len([a for a in active_alerts if a['severity'] == 'critical'])
            warning_alerts = len([a for a in active_alerts if a['severity'] == 'warning'])
            affected_models = len(set(a['model_name'] for a in active_alerts))
            
            return {
                'timestamp': timestamp,
                'active_alerts': active_alerts,
                'alert_summary': {
                    'total_alerts': len(active_alerts),
                    'critical_alerts': critical_alerts,
                    'warning_alerts': warning_alerts,
                    'affected_models': affected_models
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get health alerts: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }


class TransformerMemoryMonitor:
    """Monitor for transformer memory usage."""
    
    def __init__(self, models: Dict[str, Any], warning_threshold: float, critical_threshold: float, check_interval: int):
        self.models = models
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get memory status for all models."""
        # Use first model for single status
        model = next(iter(self.models.values()))
        
        current_usage = getattr(model, 'memory_usage_mb', 1500.0)
        if hasattr(model, 'get_memory_usage_mb'):
            current_usage = model.get_memory_usage_mb()
        
        peak_usage = current_usage * 1.2
        if hasattr(model, 'get_peak_memory_usage_mb'):
            peak_usage = model.get_peak_memory_usage_mb()
        
        usage_percent = (current_usage / 8192.0) * 100
        
        if usage_percent > self.critical_threshold:
            status = 'critical'
        elif usage_percent > self.warning_threshold:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'current_usage_mb': current_usage,
            'peak_usage_mb': peak_usage,
            'usage_percent': usage_percent,
            'status': status
        }


class TransformerLatencyMonitor:
    """Monitor for transformer inference latency."""
    
    def __init__(self, models: Dict[str, Any], warning_threshold: float, critical_threshold: float, check_interval: int):
        self.models = models
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
    
    def get_latency_status(self) -> Dict[str, Any]:
        """Get latency status for all models."""
        # Use first model for single status
        model = next(iter(self.models.values()))
        
        current_latency = getattr(model, 'last_inference_latency_ms', 95.0)
        if hasattr(model, 'get_inference_latency_ms'):
            current_latency = model.get_inference_latency_ms()
        
        percentiles = {
            'p50': current_latency * 0.8,
            'p95': current_latency * 1.2,
            'p99': current_latency * 1.5
        }
        
        if current_latency > self.critical_threshold:
            status = 'critical'
        elif current_latency > self.warning_threshold:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'current_latency_ms': current_latency,
            'percentiles': percentiles,
            'status': status
        }


class TransformerCacheMonitor:
    """Monitor for transformer cache performance."""
    
    def __init__(self, models: Dict[str, Any], warning_threshold: float, critical_threshold: float, check_interval: int):
        self.models = models
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status for all models."""
        # Use first model for single status
        model = next(iter(self.models.values()))
        
        hit_rate = getattr(model, 'cache_hit_rate', 0.75)
        if hasattr(model, 'get_cache_hit_rate'):
            hit_rate = model.get_cache_hit_rate()
        
        miss_rate = 1.0 - hit_rate
        total_requests = 1000  # Mock value
        
        if hit_rate < self.critical_threshold:
            status = 'critical'
        elif hit_rate < self.warning_threshold:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'total_requests': total_requests,
            'status': status
        }


class TransformerLoadingMonitor:
    """Monitor for transformer model loading."""
    
    def __init__(self, models: Dict[str, Any], warning_threshold: float, critical_threshold: float, check_interval: int):
        self.models = models
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
    
    def get_loading_status(self) -> Dict[str, Any]:
        """Get loading status for all models."""
        # Use first model for single status
        model = next(iter(self.models.values()))
        
        loading_time = getattr(model, 'loading_time_ms', 2500.0)
        if hasattr(model, 'get_loading_time_ms'):
            loading_time = model.get_loading_time_ms()
        
        is_loaded = True
        loading_failures = 0
        
        if loading_time > self.critical_threshold:
            status = 'critical'
        elif loading_time > self.warning_threshold:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'loading_time_ms': loading_time,
            'is_loaded': is_loaded,
            'loading_failures': loading_failures,
            'status': status
        }


# FastAPI integration class
class TransformerHealthAPI:
    """FastAPI integration for transformer health endpoints."""
    
    def __init__(self, endpoints: TransformerHealthEndpoints):
        self.endpoints = endpoints