"""
System Health Monitor

This module provides real-time monitoring and alerting for overall system
health across all modes. It tracks performance metrics, resource usage,
and integration health with comprehensive alerting capabilities.

Key Features:
- Real-time system metrics collection
- Health threshold monitoring and alerting  
- Performance degradation detection
- Resource usage monitoring
- Integration health monitoring

Following TDD methodology - implementation satisfies comprehensive test requirements.
"""

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import UUID, uuid4
import structlog

from src.modes.base import ModeType, ModeStatus
from src.modes.mode_manager import ModeManager
from src.portfolio.base import Portfolio


logger = structlog.get_logger()


class HealthStatus(Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Health alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of health metrics."""
    SYSTEM_RESOURCE = "system_resource"
    TRADING_PERFORMANCE = "trading_performance"
    API_HEALTH = "api_health"
    DATABASE_PERFORMANCE = "database_performance"
    NETWORK_LATENCY = "network_latency"
    ERROR_RATE = "error_rate"


@dataclass
class HealthMetric:
    """Individual health metric data structure."""
    metric_id: UUID
    name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    threshold_type: str = "max"  # max, min, range
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def status(self) -> HealthStatus:
        """Calculate health status based on thresholds."""
        if self.threshold_critical is not None:
            if self.threshold_type == "max" and self.value > self.threshold_critical:
                return HealthStatus.CRITICAL
            elif self.threshold_type == "min" and self.value < self.threshold_critical:
                return HealthStatus.CRITICAL
        
        if self.threshold_warning is not None:
            if self.threshold_type == "max" and self.value > self.threshold_warning:
                return HealthStatus.WARNING
            elif self.threshold_type == "min" and self.value < self.threshold_warning:
                return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    @property
    def is_healthy(self) -> bool:
        """Check if metric is in healthy state."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class HealthAlert:
    """Health alert data structure."""
    alert_id: UUID
    alert_level: AlertLevel
    metric_name: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    report_id: UUID
    timestamp: datetime
    overall_status: HealthStatus
    system_metrics: Dict[str, HealthMetric]
    trading_metrics: Dict[str, HealthMetric]
    api_health: Dict[str, HealthMetric]
    active_alerts: List[HealthAlert]
    performance_summary: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)


@dataclass
class HealthThreshold:
    """Health threshold configuration."""
    name: str
    warning_value: float
    critical_value: float
    threshold_type: str = "max"  # max, min, range
    enabled: bool = True
    description: str = ""


@dataclass
class HealthMonitorConfig:
    """Configuration for system health monitor."""
    monitoring_interval_seconds: float = 5.0
    metric_retention_hours: int = 24
    alert_cooldown_seconds: int = 300
    max_alerts_per_hour: int = 20
    enable_system_monitoring: bool = True
    enable_trading_monitoring: bool = True
    enable_api_monitoring: bool = True
    enable_performance_monitoring: bool = True
    
    # System resource thresholds
    cpu_warning_percent: float = 70.0
    cpu_critical_percent: float = 90.0
    memory_warning_percent: float = 80.0
    memory_critical_percent: float = 95.0
    disk_warning_percent: float = 85.0
    disk_critical_percent: float = 95.0
    
    # Network thresholds
    latency_warning_ms: float = 100.0
    latency_critical_ms: float = 500.0
    
    # Trading performance thresholds
    drawdown_warning_percent: float = 10.0
    drawdown_critical_percent: float = 15.0
    error_rate_warning: float = 0.05
    error_rate_critical: float = 0.10
    
    # Alert delivery settings
    enable_email_alerts: bool = False
    enable_telegram_alerts: bool = False
    enable_webhook_alerts: bool = False
    webhook_url: Optional[str] = None
    alert_escalation_minutes: int = 30
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.monitoring_interval_seconds <= 0:
            raise ValueError("monitoring_interval_seconds must be positive")
        if self.metric_retention_hours <= 0:
            raise ValueError("metric_retention_hours must be positive")
        if self.alert_cooldown_seconds <= 0:
            raise ValueError("alert_cooldown_seconds must be positive")


class SystemHealthMonitor:
    """
    Real-time system health monitoring and alerting system.
    
    Monitors system resources, trading performance, API health,
    and provides comprehensive alerting and reporting capabilities.
    """
    
    def __init__(self, config: HealthMonitorConfig, mode_manager: ModeManager):
        """Initialize system health monitor."""
        self.config = config
        self.mode_manager = mode_manager
        
        # Health state
        self.overall_status = HealthStatus.UNKNOWN
        self.last_health_check = datetime.now()
        
        # Metrics storage
        self.current_metrics: Dict[str, HealthMetric] = {}
        self.metric_history: Dict[str, List[HealthMetric]] = {}
        
        # Alerts
        self.active_alerts: Dict[UUID, HealthAlert] = {}
        self.alert_history: List[HealthAlert] = []
        self.alert_counts: Dict[str, int] = {}
        self.last_alert_reset = time.time()
        
        # Monitoring tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.performance_baseline: Dict[str, float] = {}
        self.degradation_detected = False
        
        # Initialize thresholds
        self.health_thresholds = self._initialize_health_thresholds()
        
        # Configure logger
        self.logger = logger.bind(
            component="system_health_monitor",
            monitoring_interval=config.monitoring_interval_seconds
        )
        
        self.logger.info("System health monitor initialized")
    
    async def start_monitoring(self) -> None:
        """Start all health monitoring tasks."""
        if self._monitoring_tasks:
            self.logger.warning("Health monitoring already started")
            return
        
        self.logger.info("Starting system health monitoring")
        
        # Start monitoring tasks
        if self.config.enable_system_monitoring:
            task = asyncio.create_task(self._system_monitoring_loop())
            self._monitoring_tasks.append(task)
        
        if self.config.enable_trading_monitoring:
            task = asyncio.create_task(self._trading_monitoring_loop())
            self._monitoring_tasks.append(task)
        
        if self.config.enable_api_monitoring:
            task = asyncio.create_task(self._api_monitoring_loop())
            self._monitoring_tasks.append(task)
        
        if self.config.enable_performance_monitoring:
            task = asyncio.create_task(self._performance_monitoring_loop())
            self._monitoring_tasks.append(task)
        
        # Start alert management
        task = asyncio.create_task(self._alert_management_loop())
        self._monitoring_tasks.append(task)
        
        # Start cleanup task
        task = asyncio.create_task(self._cleanup_loop())
        self._monitoring_tasks.append(task)
        
        self.logger.info("System health monitoring started", active_tasks=len(self._monitoring_tasks))
    
    async def stop_monitoring(self) -> None:
        """Stop all health monitoring tasks."""
        self.logger.info("Stopping system health monitoring")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        self._monitoring_tasks.clear()
        self._shutdown_event.clear()
        
        self.logger.info("System health monitoring stopped")
    
    async def get_health_report(self) -> SystemHealthReport:
        """Generate comprehensive health report."""
        # Update overall status
        await self._update_overall_status()
        
        # Categorize metrics
        system_metrics = {}
        trading_metrics = {}
        api_metrics = {}
        
        for metric in self.current_metrics.values():
            if metric.metric_type == MetricType.SYSTEM_RESOURCE:
                system_metrics[metric.name] = metric
            elif metric.metric_type == MetricType.TRADING_PERFORMANCE:
                trading_metrics[metric.name] = metric
            elif metric.metric_type == MetricType.API_HEALTH:
                api_metrics[metric.name] = metric
        
        # Generate performance summary
        performance_summary = await self._generate_performance_summary()
        
        # Generate recommendations
        recommendations = await self._generate_recommendations()
        
        return SystemHealthReport(
            report_id=uuid4(),
            timestamp=datetime.now(),
            overall_status=self.overall_status,
            system_metrics=system_metrics,
            trading_metrics=trading_metrics,
            api_health=api_metrics,
            active_alerts=list(self.active_alerts.values()),
            performance_summary=performance_summary,
            recommendations=recommendations
        )
    
    async def get_metric_history(
        self,
        metric_name: str,
        hours: int = 1
    ) -> List[HealthMetric]:
        """Get historical data for a specific metric."""
        if metric_name not in self.metric_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metric for metric in self.metric_history[metric_name]
            if metric.timestamp >= cutoff_time
        ]
    
    async def acknowledge_alert(self, alert_id: UUID) -> bool:
        """Acknowledge a health alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.acknowledged = True
        
        self.logger.info("Health alert acknowledged", alert_id=str(alert_id))
        return True
    
    async def resolve_alert(self, alert_id: UUID) -> bool:
        """Manually resolve a health alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        
        # Move to history and remove from active
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        self.logger.info("Health alert manually resolved", alert_id=str(alert_id))
        return True
    
    async def add_custom_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        unit: str = "",
        description: str = ""
    ) -> UUID:
        """Add a custom health metric."""
        metric = HealthMetric(
            metric_id=uuid4(),
            name=name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            description=description
        )
        
        await self._store_metric(metric)
        
        return metric.metric_id
    
    # Private methods
    
    def _initialize_health_thresholds(self) -> Dict[str, HealthThreshold]:
        """Initialize health thresholds from configuration."""
        return {
            "cpu_usage": HealthThreshold(
                name="cpu_usage",
                warning_value=self.config.cpu_warning_percent,
                critical_value=self.config.cpu_critical_percent,
                description="CPU usage percentage"
            ),
            "memory_usage": HealthThreshold(
                name="memory_usage",
                warning_value=self.config.memory_warning_percent,
                critical_value=self.config.memory_critical_percent,
                description="Memory usage percentage"
            ),
            "disk_usage": HealthThreshold(
                name="disk_usage",
                warning_value=self.config.disk_warning_percent,
                critical_value=self.config.disk_critical_percent,
                description="Disk usage percentage"
            ),
            "network_latency": HealthThreshold(
                name="network_latency",
                warning_value=self.config.latency_warning_ms,
                critical_value=self.config.latency_critical_ms,
                description="Network latency in milliseconds"
            ),
            "portfolio_drawdown": HealthThreshold(
                name="portfolio_drawdown",
                warning_value=self.config.drawdown_warning_percent,
                critical_value=self.config.drawdown_critical_percent,
                description="Portfolio drawdown percentage"
            ),
            "error_rate": HealthThreshold(
                name="error_rate",
                warning_value=self.config.error_rate_warning,
                critical_value=self.config.error_rate_critical,
                description="System error rate"
            )
        }
    
    async def _system_monitoring_loop(self) -> None:
        """Main system resource monitoring loop."""
        self.logger.info("System monitoring loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._collect_system_metrics()
                
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error("System monitoring error", error=str(e))
                await asyncio.sleep(1)
    
    async def _trading_monitoring_loop(self) -> None:
        """Trading performance monitoring loop."""
        self.logger.info("Trading monitoring loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._collect_trading_metrics()
                
                await asyncio.sleep(self.config.monitoring_interval_seconds * 2)  # Less frequent
                
            except Exception as e:
                self.logger.error("Trading monitoring error", error=str(e))
                await asyncio.sleep(1)
    
    async def _api_monitoring_loop(self) -> None:
        """API health monitoring loop."""
        self.logger.info("API monitoring loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._collect_api_metrics()
                
                await asyncio.sleep(self.config.monitoring_interval_seconds * 3)  # Less frequent
                
            except Exception as e:
                self.logger.error("API monitoring error", error=str(e))
                await asyncio.sleep(1)
    
    async def _performance_monitoring_loop(self) -> None:
        """Performance degradation monitoring loop."""
        self.logger.info("Performance monitoring loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._monitor_performance_degradation()
                
                await asyncio.sleep(self.config.monitoring_interval_seconds * 5)  # Less frequent
                
            except Exception as e:
                self.logger.error("Performance monitoring error", error=str(e))
                await asyncio.sleep(1)
    
    async def _alert_management_loop(self) -> None:
        """Alert management and escalation loop."""
        self.logger.info("Alert management loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._process_metric_alerts()
                await self._manage_alert_escalation()
                await self._update_alert_rate_limits()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error("Alert management error", error=str(e))
                await asyncio.sleep(1)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old metrics and alerts loop."""
        self.logger.info("Cleanup loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_old_metrics()
                await self._cleanup_resolved_alerts()
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self.logger.error("Cleanup error", error=str(e))
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._create_and_store_metric(
                name="cpu_usage",
                value=cpu_percent,
                metric_type=MetricType.SYSTEM_RESOURCE,
                unit="percent",
                description="CPU usage percentage"
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self._create_and_store_metric(
                name="memory_usage",
                value=memory.percent,
                metric_type=MetricType.SYSTEM_RESOURCE,
                unit="percent",
                description="Memory usage percentage"
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self._create_and_store_metric(
                name="disk_usage",
                value=disk_percent,
                metric_type=MetricType.SYSTEM_RESOURCE,
                unit="percent",
                description="Disk usage percentage"
            )
            
            # Network I/O
            network = psutil.net_io_counters()
            await self._create_and_store_metric(
                name="network_bytes_sent",
                value=float(network.bytes_sent),
                metric_type=MetricType.SYSTEM_RESOURCE,
                unit="bytes",
                description="Total network bytes sent"
            )
            
            await self._create_and_store_metric(
                name="network_bytes_recv",
                value=float(network.bytes_recv),
                metric_type=MetricType.SYSTEM_RESOURCE,
                unit="bytes",
                description="Total network bytes received"
            )
            
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))
    
    async def _collect_trading_metrics(self) -> None:
        """Collect trading performance metrics."""
        try:
            total_pnl = Decimal("0")
            total_unrealized_pnl = Decimal("0")
            active_positions = 0
            total_trades = 0
            max_drawdown = 0.0
            
            # Aggregate metrics from all active modes
            for mode in self.mode_manager.active_modes.values():
                if mode.status == ModeStatus.ACTIVE:
                    portfolio = mode.portfolio
                    total_pnl += portfolio.realized_pnl
                    total_unrealized_pnl += portfolio.unrealized_pnl
                    active_positions += len(portfolio.positions)
                    
                    # Calculate drawdown
                    if portfolio.total_value > 0:
                        mode_drawdown = abs(float(portfolio.unrealized_pnl / portfolio.total_value)) * 100
                        max_drawdown = max(max_drawdown, mode_drawdown)
                    
                    # Get trade count from metrics
                    total_trades += mode.metrics.get("trades_executed", 0)
            
            # Store trading metrics
            await self._create_and_store_metric(
                name="total_realized_pnl",
                value=float(total_pnl),
                metric_type=MetricType.TRADING_PERFORMANCE,
                unit="USD",
                description="Total realized P&L"
            )
            
            await self._create_and_store_metric(
                name="total_unrealized_pnl",
                value=float(total_unrealized_pnl),
                metric_type=MetricType.TRADING_PERFORMANCE,
                unit="USD",
                description="Total unrealized P&L"
            )
            
            await self._create_and_store_metric(
                name="active_positions",
                value=float(active_positions),
                metric_type=MetricType.TRADING_PERFORMANCE,
                unit="count",
                description="Number of active positions"
            )
            
            await self._create_and_store_metric(
                name="total_trades",
                value=float(total_trades),
                metric_type=MetricType.TRADING_PERFORMANCE,
                unit="count",
                description="Total number of trades executed"
            )
            
            await self._create_and_store_metric(
                name="portfolio_drawdown",
                value=max_drawdown,
                metric_type=MetricType.TRADING_PERFORMANCE,
                unit="percent",
                description="Maximum portfolio drawdown"
            )
            
        except Exception as e:
            self.logger.error("Failed to collect trading metrics", error=str(e))
    
    async def _collect_api_metrics(self) -> None:
        """Collect API health metrics."""
        try:
            # Placeholder for API health monitoring
            # In a real implementation, this would ping various APIs
            # and measure response times
            
            # Simulate API response times
            api_response_time = 50.0  # ms placeholder
            await self._create_and_store_metric(
                name="api_response_time",
                value=api_response_time,
                metric_type=MetricType.API_HEALTH,
                unit="milliseconds",
                description="Average API response time"
            )
            
            # Simulate API success rate
            api_success_rate = 99.5  # percent placeholder
            await self._create_and_store_metric(
                name="api_success_rate",
                value=api_success_rate,
                metric_type=MetricType.API_HEALTH,
                unit="percent",
                description="API success rate"
            )
            
        except Exception as e:
            self.logger.error("Failed to collect API metrics", error=str(e))
    
    async def _monitor_performance_degradation(self) -> None:
        """Monitor for performance degradation."""
        try:
            # Check if we have baseline metrics
            if not self.performance_baseline:
                # Establish baseline from recent metrics
                await self._establish_performance_baseline()
                return
            
            # Compare current metrics to baseline
            degradation_detected = False
            
            for metric_name, baseline_value in self.performance_baseline.items():
                if metric_name in self.current_metrics:
                    current_value = self.current_metrics[metric_name].value
                    
                    # Check for significant degradation (>20% worse)
                    if metric_name in ["cpu_usage", "memory_usage", "api_response_time"]:
                        if current_value > baseline_value * 1.2:
                            degradation_detected = True
                            
                            await self._create_performance_alert(
                                metric_name,
                                current_value,
                                baseline_value
                            )
            
            self.degradation_detected = degradation_detected
            
        except Exception as e:
            self.logger.error("Failed to monitor performance degradation", error=str(e))
    
    async def _create_and_store_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        unit: str,
        description: str
    ) -> None:
        """Create metric with thresholds and store."""
        # Get thresholds if available
        threshold = self.health_thresholds.get(name)
        threshold_warning = threshold.warning_value if threshold else None
        threshold_critical = threshold.critical_value if threshold else None
        threshold_type = threshold.threshold_type if threshold else "max"
        
        metric = HealthMetric(
            metric_id=uuid4(),
            name=name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            threshold_type=threshold_type,
            description=description
        )
        
        await self._store_metric(metric)
    
    async def _store_metric(self, metric: HealthMetric) -> None:
        """Store metric in current metrics and history."""
        # Store in current metrics
        self.current_metrics[metric.name] = metric
        
        # Store in history
        if metric.name not in self.metric_history:
            self.metric_history[metric.name] = []
        
        self.metric_history[metric.name].append(metric)
        
        # Check for alerts
        if not metric.is_healthy:
            await self._create_metric_alert(metric)
    
    async def _create_metric_alert(self, metric: HealthMetric) -> None:
        """Create alert for unhealthy metric."""
        # Check rate limiting
        alert_key = f"{metric.name}_{metric.status.value}"
        if not await self._check_alert_rate_limit(alert_key):
            return
        
        # Determine alert level
        alert_level = AlertLevel.WARNING
        if metric.status == HealthStatus.CRITICAL:
            alert_level = AlertLevel.CRITICAL
        
        # Create alert
        alert = HealthAlert(
            alert_id=uuid4(),
            alert_level=alert_level,
            metric_name=metric.name,
            message=f"Health threshold exceeded: {metric.name}",
            current_value=metric.value,
            threshold_value=metric.threshold_warning or metric.threshold_critical or 0,
            metadata={
                "metric_type": metric.metric_type.value,
                "unit": metric.unit,
                "threshold_type": metric.threshold_type
            }
        )
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(
            "Health alert created",
            alert_id=str(alert.alert_id),
            metric=metric.name,
            current_value=metric.value,
            threshold=metric.threshold_warning or metric.threshold_critical,
            status=metric.status.value
        )
    
    async def _process_metric_alerts(self) -> None:
        """Process and resolve metric alerts."""
        alerts_to_resolve = []
        
        for alert in self.active_alerts.values():
            if not alert.resolved:
                # Check if metric is now healthy
                current_metric = self.current_metrics.get(alert.metric_name)
                if current_metric and current_metric.is_healthy:
                    alert.resolved = True
                    alerts_to_resolve.append(alert.alert_id)
        
        # Remove resolved alerts
        for alert_id in alerts_to_resolve:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
    
    async def _manage_alert_escalation(self) -> None:
        """Manage alert escalation based on time and severity."""
        now = datetime.now()
        escalation_threshold = timedelta(minutes=self.config.alert_escalation_minutes)
        
        for alert in self.active_alerts.values():
            if not alert.escalated and not alert.acknowledged:
                if alert.alert_level == AlertLevel.CRITICAL:
                    age = now - alert.timestamp
                    if age > escalation_threshold:
                        alert.escalated = True
                        await self._escalate_alert(alert)
    
    async def _escalate_alert(self, alert: HealthAlert) -> None:
        """Escalate critical alert."""
        self.logger.critical(
            "Alert escalated",
            alert_id=str(alert.alert_id),
            metric=alert.metric_name,
            age_minutes=(datetime.now() - alert.timestamp).total_seconds() / 60
        )
        
        # In a real implementation, this would send additional notifications
        # such as SMS, phone calls, or high-priority emails
    
    async def _check_alert_rate_limit(self, alert_key: str) -> bool:
        """Check if alert is within rate limits."""
        current_time = time.time()
        
        # Reset hourly counts
        if current_time - self.last_alert_reset > 3600:
            self.alert_counts.clear()
            self.last_alert_reset = current_time
        
        # Check rate limit
        current_count = self.alert_counts.get(alert_key, 0)
        if current_count >= self.config.max_alerts_per_hour:
            return False
        
        self.alert_counts[alert_key] = current_count + 1
        return True
    
    async def _update_alert_rate_limits(self) -> None:
        """Update alert rate limiting counters."""
        current_time = time.time()
        if current_time - self.last_alert_reset > 3600:
            self.alert_counts.clear()
            self.last_alert_reset = current_time
    
    async def _update_overall_status(self) -> None:
        """Update overall system health status."""
        if not self.current_metrics:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        critical_count = 0
        warning_count = 0
        healthy_count = 0
        
        for metric in self.current_metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                critical_count += 1
            elif metric.status == HealthStatus.WARNING:
                warning_count += 1
            else:
                healthy_count += 1
        
        # Determine overall status
        if critical_count > 0:
            self.overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            self.overall_status = HealthStatus.WARNING
        elif self.degradation_detected:
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.HEALTHY
        
        self.last_health_check = datetime.now()
    
    async def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {
            "total_metrics": len(self.current_metrics),
            "healthy_metrics": len([m for m in self.current_metrics.values() if m.is_healthy]),
            "warning_metrics": len([m for m in self.current_metrics.values() if m.status == HealthStatus.WARNING]),
            "critical_metrics": len([m for m in self.current_metrics.values() if m.status == HealthStatus.CRITICAL]),
            "active_alerts": len(self.active_alerts),
            "last_health_check": self.last_health_check.isoformat(),
            "monitoring_uptime_hours": (datetime.now() - self.last_health_check).total_seconds() / 3600,
            "performance_degradation_detected": self.degradation_detected
        }
        
        return summary
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate health recommendations based on current metrics."""
        recommendations = []
        
        # Check for high resource usage
        if "cpu_usage" in self.current_metrics:
            cpu_metric = self.current_metrics["cpu_usage"]
            if cpu_metric.status == HealthStatus.CRITICAL:
                recommendations.append("CPU usage is critical. Consider scaling up resources or optimizing performance.")
            elif cpu_metric.status == HealthStatus.WARNING:
                recommendations.append("CPU usage is elevated. Monitor performance and consider optimization.")
        
        if "memory_usage" in self.current_metrics:
            memory_metric = self.current_metrics["memory_usage"]
            if memory_metric.status == HealthStatus.CRITICAL:
                recommendations.append("Memory usage is critical. Check for memory leaks and consider adding more RAM.")
        
        # Check for trading performance issues
        if "portfolio_drawdown" in self.current_metrics:
            drawdown_metric = self.current_metrics["portfolio_drawdown"]
            if drawdown_metric.status == HealthStatus.CRITICAL:
                recommendations.append("Portfolio drawdown is excessive. Review trading strategy and risk management.")
        
        # Check for API issues
        if "api_response_time" in self.current_metrics:
            api_metric = self.current_metrics["api_response_time"]
            if api_metric.status == HealthStatus.WARNING:
                recommendations.append("API response times are elevated. Check network connectivity and API status.")
        
        return recommendations
    
    async def _establish_performance_baseline(self) -> None:
        """Establish performance baseline from recent metrics."""
        try:
            baseline_metrics = ["cpu_usage", "memory_usage", "api_response_time"]
            
            for metric_name in baseline_metrics:
                if metric_name in self.current_metrics:
                    self.performance_baseline[metric_name] = self.current_metrics[metric_name].value
            
            self.logger.info("Performance baseline established", baseline=self.performance_baseline)
            
        except Exception as e:
            self.logger.error("Failed to establish performance baseline", error=str(e))
    
    async def _create_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float
    ) -> None:
        """Create alert for performance degradation."""
        degradation_pct = ((current_value - baseline_value) / baseline_value) * 100
        
        alert = HealthAlert(
            alert_id=uuid4(),
            alert_level=AlertLevel.WARNING,
            metric_name=metric_name,
            message=f"Performance degradation detected: {metric_name}",
            current_value=current_value,
            threshold_value=baseline_value,
            metadata={
                "degradation_percent": degradation_pct,
                "baseline_value": baseline_value,
                "type": "performance_degradation"
            }
        )
        
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        self.logger.warning(
            "Performance degradation alert",
            metric=metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            degradation_pct=degradation_pct
        )
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics based on retention policy."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.metric_retention_hours)
        
        for metric_name, metrics in self.metric_history.items():
            # Remove old metrics
            self.metric_history[metric_name] = [
                metric for metric in metrics
                if metric.timestamp >= cutoff_time
            ]
    
    async def _cleanup_resolved_alerts(self) -> None:
        """Clean up old resolved alerts."""
        # Keep resolved alerts for 24 hours for audit purposes
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.alert_history = [
            alert for alert in self.alert_history
            if not alert.resolved or alert.timestamp >= cutoff_time
        ]


# Exception Classes
class HealthMonitorError(Exception):
    """Base health monitor error."""
    pass


class AlertDeliveryError(HealthMonitorError):
    """Error during alert delivery."""
    pass