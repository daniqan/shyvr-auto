"""
Simulation Metrics Collection for Prometheus Integration

This module provides comprehensive metrics collection for simulation trading mode,
including performance metrics, risk metrics, and simulation-specific indicators.

Key Features:
- Prometheus metrics integration
- Real-time simulation performance tracking
- Risk metrics collection
- Dashboard-ready metrics formatting
- Alert generation based on simulation performance
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
import structlog

from src.monitoring.base import MetricsCollector, MetricsRegistry
from src.modes.simulation_mode import SimulationMode


logger = structlog.get_logger()


class SimulationMetricsCollector(MetricsCollector):
    """
    Metrics collector specifically for simulation trading mode.
    
    Collects and exposes simulation-specific metrics including:
    - Portfolio performance metrics
    - Trading activity metrics
    - Risk and safety metrics
    - Learning progress metrics
    - Simulation accuracy metrics
    """
    
    def __init__(self, registry: MetricsRegistry, simulation_mode: SimulationMode):
        """
        Initialize simulation metrics collector.
        
        Args:
            registry: MetricsRegistry for managing Prometheus metrics
            simulation_mode: SimulationMode instance to collect metrics from
        """
        super().__init__(registry)
        self.simulation_mode = simulation_mode
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Tracking state
        self.last_collection_time = None
        self.collection_count = 0
        
        self.logger = logger.bind(
            component="simulation_metrics_collector",
            mode_id=str(simulation_mode.mode_id)
        )
        
        self.logger.info("Simulation metrics collector initialized")
    
    def _initialize_metrics(self) -> None:
        """Initialize all simulation-specific Prometheus metrics."""
        mode_id_label = str(self.simulation_mode.mode_id)
        
        # Portfolio metrics
        self.portfolio_value_gauge = self.registry.get_gauge(
            "simulation_portfolio_value",
            "Current virtual portfolio value in USD",
            ["mode_id"]
        )
        
        self.unrealized_pnl_gauge = self.registry.get_gauge(
            "simulation_unrealized_pnl",
            "Current unrealized P&L in USD",
            ["mode_id"]
        )
        
        self.realized_pnl_gauge = self.registry.get_gauge(
            "simulation_realized_pnl",
            "Total realized P&L in USD",
            ["mode_id"]
        )
        
        self.cash_balance_gauge = self.registry.get_gauge(
            "simulation_cash_balance",
            "Current virtual cash balance in USD",
            ["mode_id"]
        )
        
        # Trading activity metrics
        self.trades_counter = self.registry.get_counter(
            "simulation_trades_total",
            "Total number of simulated trades executed",
            ["mode_id", "action_type", "success"]
        )
        
        self.win_rate_gauge = self.registry.get_gauge(
            "simulation_win_rate",
            "Percentage of profitable trades",
            ["mode_id"]
        )
        
        self.trade_execution_histogram = self.registry.get_histogram(
            "simulation_trade_execution_time",
            "Simulation trade execution time in seconds",
            ["mode_id"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )
        
        # Risk metrics
        self.drawdown_gauge = self.registry.get_gauge(
            "simulation_drawdown",
            "Current portfolio drawdown percentage",
            ["mode_id"]
        )
        
        self.position_count_gauge = self.registry.get_gauge(
            "simulation_active_positions",
            "Number of active positions",
            ["mode_id"]
        )
        
        self.risk_score_gauge = self.registry.get_gauge(
            "simulation_risk_score",
            "Current portfolio risk score",
            ["mode_id"]
        )
        
        # Performance metrics
        self.sharpe_ratio_gauge = self.registry.get_gauge(
            "simulation_sharpe_ratio",
            "Current Sharpe ratio",
            ["mode_id"]
        )
        
        self.volatility_gauge = self.registry.get_gauge(
            "simulation_volatility",
            "Portfolio volatility",
            ["mode_id"]
        )
        
        # Simulation-specific metrics
        self.slippage_gauge = self.registry.get_gauge(
            "simulation_slippage",
            "Average slippage in basis points",
            ["mode_id"]
        )
        
        self.fees_paid_gauge = self.registry.get_gauge(
            "simulation_fees_paid",
            "Total fees paid in USD",
            ["mode_id"]
        )
        
        self.execution_latency_histogram = self.registry.get_histogram(
            "simulation_execution_latency",
            "Trade execution latency in milliseconds",
            ["mode_id"],
            buckets=(1, 5, 10, 50, 100, 500, 1000, 5000)
        )
        
        # Learning metrics (if continuous learning enabled)
        if self.simulation_mode.enable_continuous_learning:
            self.learning_sessions_counter = self.registry.get_counter(
                "simulation_learning_sessions_total",
                "Total number of learning sessions triggered",
                ["mode_id", "success"]
            )
            
            self.model_updates_counter = self.registry.get_counter(
                "simulation_model_updates_total",
                "Total number of model updates",
                ["mode_id"]
            )
            
            self.learning_performance_gauge = self.registry.get_gauge(
                "simulation_learning_performance",
                "Current learning model performance score",
                ["mode_id", "model_version"]
            )
        
        # Safety metrics (if safety enabled)
        if self.simulation_mode.enable_simulation_safety:
            self.safety_violations_counter = self.registry.get_counter(
                "simulation_safety_violations_total",
                "Total number of safety violations",
                ["mode_id", "violation_type"]
            )
            
            self.safety_alerts_counter = self.registry.get_counter(
                "simulation_safety_alerts_total",
                "Total number of safety alerts generated",
                ["mode_id", "severity"]
            )
            
            self.circuit_breaker_gauge = self.registry.get_gauge(
                "simulation_circuit_breaker_active",
                "Circuit breaker status (1=active, 0=inactive)",
                ["mode_id"]
            )
    
    def collect_metrics(self) -> None:
        """Collect all simulation metrics."""
        try:
            self.collection_count += 1
            self.last_collection_time = time.time()
            
            mode_id = str(self.simulation_mode.mode_id)
            
            # Collect portfolio metrics
            self._collect_portfolio_metrics(mode_id)
            
            # Collect trading metrics
            self._collect_trading_metrics(mode_id)
            
            # Collect risk metrics
            self._collect_risk_metrics(mode_id)
            
            # Collect performance metrics
            self._collect_performance_metrics(mode_id)
            
            # Collect simulation-specific metrics
            self._collect_simulation_metrics(mode_id)
            
            # Collect learning metrics if enabled
            if self.simulation_mode.enable_continuous_learning:
                self._collect_learning_metrics(mode_id)
            
            # Collect safety metrics if enabled
            if self.simulation_mode.enable_simulation_safety:
                self._collect_safety_metrics(mode_id)
            
            self.logger.debug("Metrics collection completed", collection_count=self.collection_count)
            
        except Exception as e:
            self.logger.error("Error collecting simulation metrics", error=str(e))
            raise
    
    def _collect_portfolio_metrics(self, mode_id: str) -> None:
        """Collect portfolio-related metrics."""
        virtual_portfolio = self.simulation_mode.virtual_portfolio
        
        # Portfolio value metrics
        self.portfolio_value_gauge.labels(mode_id=mode_id).set(
            float(virtual_portfolio.equity)
        )
        
        self.unrealized_pnl_gauge.labels(mode_id=mode_id).set(
            float(virtual_portfolio.total_unrealized_pnl)
        )
        
        # Calculate realized P&L
        realized_pnl = sum(
            float(tx.net_value) for tx in virtual_portfolio.transaction_history
            if tx.transaction_type.value == "SELL"
        )
        self.realized_pnl_gauge.labels(mode_id=mode_id).set(realized_pnl)
        
        self.cash_balance_gauge.labels(mode_id=mode_id).set(
            float(virtual_portfolio.cash_balance)
        )
    
    def _collect_trading_metrics(self, mode_id: str) -> None:
        """Collect trading activity metrics."""
        simulation_metrics = self.simulation_mode.simulation_metrics
        
        # Win rate
        self.win_rate_gauge.labels(mode_id=mode_id).set(
            float(simulation_metrics.win_rate)
        )
        
        # Position count
        active_positions = len([
            pos for pos in self.simulation_mode.virtual_portfolio.positions.values()
            if pos.status.value == "OPEN"
        ])
        self.position_count_gauge.labels(mode_id=mode_id).set(active_positions)
    
    def _collect_risk_metrics(self, mode_id: str) -> None:
        """Collect risk-related metrics."""
        virtual_portfolio = self.simulation_mode.virtual_portfolio
        
        # Calculate drawdown
        if virtual_portfolio.peak_value > 0:
            current_drawdown = (
                (virtual_portfolio.peak_value - virtual_portfolio.equity) / 
                virtual_portfolio.peak_value
            ) * 100
        else:
            current_drawdown = 0.0
        
        self.drawdown_gauge.labels(mode_id=mode_id).set(current_drawdown)
        
        # Risk score (simplified calculation)
        risk_score = min(100.0, max(0.0, current_drawdown * 2))  # Simple risk score
        self.risk_score_gauge.labels(mode_id=mode_id).set(risk_score)
    
    def _collect_performance_metrics(self, mode_id: str) -> None:
        """Collect performance metrics."""
        simulation_metrics = self.simulation_mode.simulation_metrics
        
        # Sharpe ratio
        self.sharpe_ratio_gauge.labels(mode_id=mode_id).set(
            float(simulation_metrics.sharpe_ratio)
        )
        
        # Volatility (simplified calculation)
        volatility = 0.15  # Placeholder - would calculate from price history
        self.volatility_gauge.labels(mode_id=mode_id).set(volatility)
    
    def _collect_simulation_metrics(self, mode_id: str) -> None:
        """Collect simulation-specific metrics."""
        simulation_metrics = self.simulation_mode.simulation_metrics
        
        # Slippage
        self.slippage_gauge.labels(mode_id=mode_id).set(
            float(simulation_metrics.average_slippage_bps)
        )
        
        # Fees paid
        self.fees_paid_gauge.labels(mode_id=mode_id).set(
            float(simulation_metrics.total_fees_paid)
        )
        
        # Execution latency (if available)
        if simulation_metrics.execution_latency_ms:
            avg_latency = sum(simulation_metrics.execution_latency_ms) / len(simulation_metrics.execution_latency_ms)
            # Record histogram observation
            self.execution_latency_histogram.labels(mode_id=mode_id).observe(avg_latency)
    
    def _collect_learning_metrics(self, mode_id: str) -> None:
        """Collect continuous learning metrics."""
        if not self.simulation_mode.continuous_learning_engine:
            return
        
        try:
            # Get learning statistics
            learning_stats = self.simulation_mode.continuous_learning_engine.get_performance_statistics()
            
            # Model performance
            if self.simulation_mode.continuous_learning_engine.active_model_version:
                active_version = self.simulation_mode.continuous_learning_engine.active_model_version
                performance_score = active_version.performance_metrics.calculate_composite_score()
                
                self.learning_performance_gauge.labels(
                    mode_id=mode_id,
                    model_version=active_version.version_id
                ).set(performance_score)
            
        except Exception as e:
            self.logger.warning("Error collecting learning metrics", error=str(e))
    
    def _collect_safety_metrics(self, mode_id: str) -> None:
        """Collect safety-related metrics."""
        if not self.simulation_mode.simulation_safety_manager:
            return
        
        try:
            # Circuit breaker status
            circuit_breaker_active = 1.0 if self.simulation_mode.simulation_safety_manager.circuit_breaker_active else 0.0
            self.circuit_breaker_gauge.labels(mode_id=mode_id).set(circuit_breaker_active)
            
        except Exception as e:
            self.logger.warning("Error collecting safety metrics", error=str(e))
    
    async def update_real_time_metrics(self) -> None:
        """Update metrics in real-time during simulation."""
        try:
            # This method can be called more frequently for real-time updates
            mode_id = str(self.simulation_mode.mode_id)
            
            # Update key real-time metrics
            self.portfolio_value_gauge.labels(mode_id=mode_id).set(
                float(self.simulation_mode.virtual_portfolio.equity)
            )
            
            self.unrealized_pnl_gauge.labels(mode_id=mode_id).set(
                float(self.simulation_mode.virtual_portfolio.total_unrealized_pnl)
            )
            
        except Exception as e:
            self.logger.warning("Error updating real-time metrics", error=str(e))
    
    def record_trade_execution(self, action_type: str, success: bool, execution_time_ms: float) -> None:
        """Record trade execution metrics."""
        mode_id = str(self.simulation_mode.mode_id)
        
        # Increment trade counter
        self.trades_counter.labels(
            mode_id=mode_id,
            action_type=action_type,
            success=str(success).lower()
        ).inc()
        
        # Record execution time
        self.trade_execution_histogram.labels(mode_id=mode_id).observe(execution_time_ms / 1000.0)
    
    def record_safety_violation(self, violation_type: str) -> None:
        """Record safety violation."""
        mode_id = str(self.simulation_mode.mode_id)
        
        self.safety_violations_counter.labels(
            mode_id=mode_id,
            violation_type=violation_type
        ).inc()
    
    def record_safety_alert(self, severity: str) -> None:
        """Record safety alert."""
        mode_id = str(self.simulation_mode.mode_id)
        
        self.safety_alerts_counter.labels(
            mode_id=mode_id,
            severity=severity
        ).inc()
    
    def record_learning_session(self, success: bool) -> None:
        """Record learning session."""
        if not self.simulation_mode.enable_continuous_learning:
            return
        
        mode_id = str(self.simulation_mode.mode_id)
        
        self.learning_sessions_counter.labels(
            mode_id=mode_id,
            success=str(success).lower()
        ).inc()
    
    def record_model_update(self) -> None:
        """Record model update."""
        if not self.simulation_mode.enable_continuous_learning:
            return
        
        mode_id = str(self.simulation_mode.mode_id)
        self.model_updates_counter.labels(mode_id=mode_id).inc()
    
    def get_metric_definitions(self) -> Dict[str, str]:
        """Get metric definitions for this collector."""
        base_metrics = {
            "simulation_portfolio_value": "Current virtual portfolio value in USD",
            "simulation_unrealized_pnl": "Current unrealized P&L in USD", 
            "simulation_realized_pnl": "Total realized P&L in USD",
            "simulation_cash_balance": "Current virtual cash balance in USD",
            "simulation_trades_total": "Total number of simulated trades executed",
            "simulation_win_rate": "Percentage of profitable trades",
            "simulation_trade_execution_time": "Simulation trade execution time in seconds",
            "simulation_drawdown": "Current portfolio drawdown percentage",
            "simulation_active_positions": "Number of active positions",
            "simulation_risk_score": "Current portfolio risk score",
            "simulation_sharpe_ratio": "Current Sharpe ratio",
            "simulation_volatility": "Portfolio volatility",
            "simulation_slippage": "Average slippage in basis points",
            "simulation_fees_paid": "Total fees paid in USD",
            "simulation_execution_latency": "Trade execution latency in milliseconds"
        }
        
        # Add learning metrics if enabled
        if self.simulation_mode.enable_continuous_learning:
            base_metrics.update({
                "simulation_learning_sessions_total": "Total number of learning sessions triggered",
                "simulation_model_updates_total": "Total number of model updates",
                "simulation_learning_performance": "Current learning model performance score"
            })
        
        # Add safety metrics if enabled
        if self.simulation_mode.enable_simulation_safety:
            base_metrics.update({
                "simulation_safety_violations_total": "Total number of safety violations",
                "simulation_safety_alerts_total": "Total number of safety alerts generated",
                "simulation_circuit_breaker_active": "Circuit breaker status (1=active, 0=inactive)"
            })
        
        return base_metrics
    
    def is_healthy(self) -> bool:
        """Check if the metrics collector is healthy."""
        try:
            # Check if simulation mode is accessible
            if not self.simulation_mode:
                return False
            
            # Check if we've collected metrics recently
            if self.last_collection_time:
                time_since_collection = time.time() - self.last_collection_time
                if time_since_collection > 300:  # 5 minutes
                    return False
            
            # Check error rate
            if self._collection_errors > 10:
                return False
            
            return True
            
        except Exception:
            return False


class SimulationDashboard:
    """
    Dashboard integration for simulation metrics.
    
    Provides dashboard-ready data formatting and real-time updates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_source = None
        self.real_time_stream = None
        
        self.logger = logger.bind(component="simulation_dashboard")
    
    async def prepare_dashboard_data(self) -> Dict[str, Any]:
        """Prepare data for dashboard visualization."""
        try:
            if not self.data_source:
                return {"error": "No data source configured"}
            
            # Get metrics data
            metrics_data = await self.data_source.get_metrics_data()
            
            # Format for dashboard
            dashboard_data = {
                "panels": self._format_panels_data(metrics_data),
                "alerts": self._format_alerts_data(metrics_data),
                "last_updated": datetime.now().isoformat(),
                "refresh_interval": self.config.get("refresh_interval", 5)
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error("Error preparing dashboard data", error=str(e))
            return {"error": str(e)}
    
    def _format_panels_data(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format metrics data for dashboard panels."""
        panels = []
        
        for panel_config in self.config.get("panels", []):
            panel_data = {
                "type": panel_config["type"],
                "title": panel_config["title"],
                "data": {}
            }
            
            # Extract relevant metrics for this panel
            for metric_name in panel_config.get("metrics", []):
                if metric_name in metrics_data:
                    panel_data["data"][metric_name] = metrics_data[metric_name]
            
            panels.append(panel_data)
        
        return panels
    
    def _format_alerts_data(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format alerts data for dashboard."""
        alerts = []
        
        for alert_config in self.config.get("alerts", []):
            alert_data = {
                "name": alert_config["name"],
                "condition": alert_config["condition"],
                "severity": alert_config["severity"],
                "active": self._evaluate_alert_condition(alert_config, metrics_data)
            }
            alerts.append(alert_data)
        
        return alerts
    
    def _evaluate_alert_condition(self, alert_config: Dict[str, Any], metrics_data: Dict[str, Any]) -> bool:
        """Evaluate if alert condition is met."""
        try:
            condition = alert_config["condition"]
            
            # Simple condition evaluation (in production would use proper parser)
            if "simulation_drawdown > 0.15" in condition:
                drawdown = metrics_data.get("simulation_drawdown", [0])
                if isinstance(drawdown, list) and drawdown:
                    return drawdown[-1] > 15.0
            
            if "simulation_win_rate < 0.4" in condition:
                win_rate = metrics_data.get("simulation_win_rate", [1.0])
                if isinstance(win_rate, list) and win_rate:
                    return win_rate[-1] < 0.4
            
            return False
            
        except Exception:
            return False
    
    def start_real_time_updates(self) -> None:
        """Start real-time dashboard updates."""
        if self.real_time_stream:
            self.real_time_stream.subscribe(self._handle_real_time_update)
    
    def _handle_real_time_update(self, update_data: Dict[str, Any]) -> None:
        """Handle real-time metric updates."""
        self.logger.debug("Real-time dashboard update received", data_keys=list(update_data.keys()))


class SimulationAlertManager:
    """
    Alert manager for simulation performance monitoring.
    
    Handles alert condition evaluation, cooldowns, and notifications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history = []
        self.alert_cooldowns = {}
        
        # Notification channels
        self.console_notifier = None
        self.metrics_notifier = None
        
        self.logger = logger.bind(component="simulation_alert_manager")
    
    def evaluate_alert_conditions(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate alert conditions against current metrics."""
        triggered_alerts = []
        
        for alert_config in self.config.get("alerts", []):
            alert_name = alert_config["name"]
            
            # Check cooldown
            if self._is_in_cooldown(alert_name):
                continue
            
            # Evaluate condition
            if self._evaluate_condition(alert_config["condition"], current_metrics):
                alert = {
                    "name": alert_name,
                    "severity": alert_config["severity"],
                    "message": alert_config["message"],
                    "timestamp": datetime.now(),
                    "metrics": current_metrics
                }
                
                triggered_alerts.append(alert)
                self._set_cooldown(alert_name, alert_config.get("cooldown_minutes", 10))
        
        return triggered_alerts
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition."""
        try:
            # Simple condition evaluation
            if "simulation_drawdown >" in condition:
                threshold = float(condition.split(">")[1].strip())
                return metrics.get("simulation_drawdown", 0) > threshold
            
            if "simulation_win_rate <" in condition and "AND" in condition:
                parts = condition.split("AND")
                win_rate_condition = parts[0].strip()
                trades_condition = parts[1].strip()
                
                win_rate_threshold = float(win_rate_condition.split("<")[1].strip())
                trades_threshold = int(trades_condition.split(">")[1].strip())
                
                return (
                    metrics.get("simulation_win_rate", 1.0) < win_rate_threshold and
                    metrics.get("simulation_trades_total", 0) > trades_threshold
                )
            
            if "simulation_execution_errors >" in condition:
                threshold = int(condition.split(">")[1].strip())
                return metrics.get("simulation_execution_errors", 0) > threshold
            
            return False
            
        except Exception as e:
            self.logger.error("Error evaluating alert condition", condition=condition, error=str(e))
            return False
    
    def _is_in_cooldown(self, alert_name: str) -> bool:
        """Check if alert is in cooldown period."""
        if alert_name not in self.alert_cooldowns:
            return False
        
        cooldown_until = self.alert_cooldowns[alert_name]
        return datetime.now() < cooldown_until
    
    def _set_cooldown(self, alert_name: str, cooldown_minutes: int) -> None:
        """Set cooldown period for alert."""
        from datetime import timedelta
        self.alert_cooldowns[alert_name] = datetime.now() + timedelta(minutes=cooldown_minutes)
    
    async def send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert through configured notification channels."""
        try:
            # Send through all configured channels
            for channel in self.config.get("notification_channels", []):
                if channel == "console" and self.console_notifier:
                    await self.console_notifier.send_notification(alert)
                elif channel == "metrics" and self.metrics_notifier:
                    await self.metrics_notifier.send_notification(alert)
            
            # Record in history
            self.alert_history.append(alert)
            
            self.logger.info("Alert sent", alert_name=alert["name"], severity=alert["severity"])
            
        except Exception as e:
            self.logger.error("Error sending alert", error=str(e))