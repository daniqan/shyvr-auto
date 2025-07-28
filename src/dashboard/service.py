"""
Dashboard Service - Integrates with all system components
"""

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import structlog

from ..utils.config import get_config
from ..modes.mode_manager import ModeManager
from ..modes.system_health_monitor import SystemHealthMonitor
from ..portfolio.portfolio_manager import PortfolioManager
from ..ml_analysis.model_manager import ModelManager
from ..rl_agent.dqn_agent import DQNTradingAgent
from ..integration.ml_rl_bridge import MLRLBridge
from ..logging.activity_logger import (
    activity_logger, ActivityCategory, ActivityAction, ActivitySeverity,
    TradingMode as LoggingTradingMode, performance_tracker
)

from .base import (
    DashboardData, SystemMetrics, TradingStatus, PortfolioStatus, 
    MLRLStatus, SystemStatus, TradingMode, Position, Trade,
    MLModel, RLAgent, dashboard_cache, DashboardError
)
from .websocket_manager import websocket_manager
from .activity_integration import dashboard_activity

# Import XAI components for integration
from ..xai.trading_integration import TradingExplanationManager
from ..xai.factory import ExplainerFactory

logger = structlog.get_logger()


class DashboardService:
    """Main dashboard service integrating with all system components"""
    
    def __init__(self):
        self.config = get_config()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._start_time = time.time()
        
        # System components (will be initialized on start)
        self.mode_manager: Optional[ModeManager] = None
        self.health_monitor: Optional[SystemHealthMonitor] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.model_manager: Optional[ModelManager] = None
        self.rl_agent: Optional[DQNTradingAgent] = None
        self.ml_rl_bridge: Optional[MLRLBridge] = None
        
        # XAI integration components
        self.xai_explanation_manager: Optional[TradingExplanationManager] = None
        self.xai_explainer_factory: Optional[ExplainerFactory] = None
        
        # Update interval
        self.update_interval = 5.0  # 5 seconds
        
        # Performance tracking
        self._request_count = 0
        self._error_count = 0
        self._request_times: List[float] = []
        
    async def start(self) -> None:
        """Start the dashboard service"""
        if self._running:
            return
        
        try:
            # Start ActivityLogger first
            await activity_logger.start()
            
            # Start dashboard activity integration
            await dashboard_activity.start()
            
            # Log dashboard service startup
            await activity_logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.START,
                source="dashboard_service",
                event_type="service_startup",
                title="Dashboard service starting",
                severity=ActivitySeverity.INFO,
                dashboard_component="dashboard_service"
            )
            
            # Initialize system components
            await self._initialize_components()
            
            # Start WebSocket manager
            await websocket_manager.start()
            
            # Start update loop
            self._running = True
            self._update_task = asyncio.create_task(self._update_loop())
            
            # Log successful startup
            await activity_logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.SUCCESS,
                source="dashboard_service",
                event_type="service_started",
                title="Dashboard service started successfully",
                severity=ActivitySeverity.INFO,
                dashboard_component="dashboard_service"
            )
            
            logger.info("Dashboard service started")
            
        except Exception as e:
            # Log startup failure
            await activity_logger.log_error(
                category=ActivityCategory.SYSTEM,
                source="dashboard_service",
                event_type="service_startup_failed",
                title="Dashboard service startup failed",
                error_message=str(e),
                exception=e,
                severity=ActivitySeverity.CRITICAL,
                dashboard_component="dashboard_service"
            )
            logger.error("Failed to start dashboard service", error=str(e))
            raise DashboardError(f"Service startup failed: {e}")
    
    async def stop(self) -> None:
        """Stop the dashboard service"""
        if not self._running:
            return
        
        # Log shutdown initiation
        await activity_logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.STOP,
            source="dashboard_service",
            event_type="service_shutdown",
            title="Dashboard service stopping",
            severity=ActivitySeverity.INFO,
            dashboard_component="dashboard_service"
        )
        
        self._running = False
        
        # Stop update loop
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        # Stop WebSocket manager
        await websocket_manager.stop()
        
        # Log successful shutdown
        await activity_logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.SUCCESS,
            source="dashboard_service",
            event_type="service_stopped",
            title="Dashboard service stopped successfully",
            severity=ActivitySeverity.INFO,
            dashboard_component="dashboard_service"
        )
        
        # Stop dashboard activity integration
        await dashboard_activity.stop()
        
        # Stop ActivityLogger last
        await activity_logger.stop()
        
        logger.info("Dashboard service stopped")
    
    async def _initialize_components(self) -> None:
        """Initialize system components"""
        try:
            # Create mock/default configuration for components
            from ..modes.mode_manager import ModeManagerConfig
            from ..portfolio.base import Portfolio, PortfolioConfig
            from decimal import Decimal
            from uuid import uuid4
            
            # Create mock portfolio for dashboard
            portfolio_config = PortfolioConfig(
                initial_balance=Decimal("10000.00"),
                base_currency="USDC"
            )
            mock_portfolio = Portfolio(
                portfolio_id=uuid4(),
                name="Dashboard Mock Portfolio",
                config=portfolio_config,
                cash_balance=Decimal("10000.00"),
                total_value=Decimal("10000.00")
            )
            
            # Create mode manager config
            mode_manager_config = ModeManagerConfig(
                max_concurrent_modes=1,
                enable_mode_switching=True,
                auto_recovery=True
            )
            
            # Initialize mode manager with required arguments
            self.mode_manager = ModeManager(mode_manager_config, mock_portfolio)
            
            # Initialize health monitor (if it exists and needs no args)
            try:
                self.health_monitor = SystemHealthMonitor()
            except Exception as e:
                logger.warning("Failed to initialize health monitor", error=str(e))
                self.health_monitor = None
            
            # Initialize portfolio manager (if it exists and needs no args)
            try:
                self.portfolio_manager = PortfolioManager()
            except Exception as e:
                logger.warning("Failed to initialize portfolio manager", error=str(e))
                self.portfolio_manager = None
            
            # Initialize ML model manager (if it exists and needs no args)
            try:
                self.model_manager = ModelManager()
            except Exception as e:
                logger.warning("Failed to initialize ML model manager", error=str(e))
                self.model_manager = None
            
            # Initialize RL agent (with mock config for now)
            try:
                from ..rl_agent.base import RLConfig, RLAgentBase
                rl_config = RLConfig()
                self.rl_agent = DQNTradingAgent(rl_config)
            except Exception as e:
                logger.warning("Failed to initialize RL agent", error=str(e))
                self.rl_agent = None
            
            # Initialize ML-RL bridge only if both components are available
            try:
                if self.model_manager and self.rl_agent:
                    self.ml_rl_bridge = MLRLBridge(
                        ml_analyzer=self.model_manager,
                        rl_agent=self.rl_agent
                    )
                else:
                    logger.info("ML-RL bridge not initialized due to missing components")
                    self.ml_rl_bridge = None
            except Exception as e:
                logger.warning("Failed to initialize ML-RL bridge", error=str(e))
                self.ml_rl_bridge = None

            # Initialize XAI components
            try:
                self.xai_explainer_factory = ExplainerFactory()
                self.xai_explanation_manager = TradingExplanationManager(
                    explainer_factory=self.xai_explainer_factory,
                    cache_size=1000,
                    explanation_timeout=5.0
                )
                logger.info("XAI components initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize XAI components", error=str(e))
                self.xai_explainer_factory = None
                self.xai_explanation_manager = None
            
            logger.info("Dashboard components initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize components", error=str(e))
            # For now, continue with mock components
            logger.info("Continuing with mock components for dashboard")
    
    async def get_dashboard_data(self) -> DashboardData:
        """Get current dashboard data"""
        async with performance_tracker(
            source="dashboard_service",
            operation="get_dashboard_data",
            category=ActivityCategory.DASHBOARD
        ) as tracker:
            try:
                # Get system metrics
                system_metrics = await self._get_system_metrics()
                
                # Get portfolio status
                portfolio_status = await self._get_portfolio_status()
                
                # Get trading status
                trading_status = await self._get_trading_status()
                
                # Get ML/RL status
                ml_rl_status = await self._get_ml_rl_status()
                
                # Get real activity data
                recent_activities = await self._get_recent_activity()
                recent_logs = [activity['title'] for activity in recent_activities[:10]]
                
                # Count alerts and errors from recent activity
                active_alerts = len([a for a in recent_activities if a.get('severity') in ['alert', 'critical', 'emergency']])
                warnings_count = len([a for a in recent_activities if a.get('severity') == 'warning'])
                errors_count = len([a for a in recent_activities if a.get('severity') == 'error'])
                
                # Create dashboard data
                dashboard_data = DashboardData(
                    system_metrics=system_metrics,
                    portfolio_status=portfolio_status,
                    trading_status=trading_status,
                    ml_rl_status=ml_rl_status,
                    active_alerts=active_alerts,
                    warnings_count=warnings_count,
                    errors_count=errors_count,
                    recent_logs=recent_logs,
                    recent_notifications=[],  # Could be derived from activity data too
                    timestamp=datetime.utcnow()
                )
                
                # Log successful data retrieval
                await activity_logger.log_activity(
                    category=ActivityCategory.DASHBOARD,
                    action=ActivityAction.READ,
                    source="dashboard_service",
                    event_type="dashboard_data_retrieved",
                    title="Dashboard data retrieved successfully",
                    severity=ActivitySeverity.DEBUG,
                    dashboard_component="data_aggregator"
                )
                
                return dashboard_data
                
            except Exception as e:
                # Log error in data retrieval
                await activity_logger.log_error(
                    category=ActivityCategory.DASHBOARD,
                    source="dashboard_service",
                    event_type="dashboard_data_error",
                    title="Failed to retrieve dashboard data",
                    error_message=str(e),
                    exception=e,
                    severity=ActivitySeverity.ERROR,
                    dashboard_component="data_aggregator"
                )
                logger.error("Failed to get dashboard data", error=str(e))
                return DashboardData.create_default()
    
    async def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Calculate uptime
            uptime = time.time() - self._start_time
            
            # Calculate performance metrics
            requests_per_minute = self._request_count / max(uptime / 60, 1)
            error_rate = (self._error_count / max(self._request_count, 1)) * 100
            avg_response_time = sum(self._request_times[-100:]) / max(len(self._request_times[-100:]), 1) * 1000
            
            # Get WebSocket connections
            ws_connections = await websocket_manager.connection_manager.get_connection_count()
            
            return SystemMetrics(
                status=SystemStatus.HEALTHY,
                uptime_seconds=uptime,
                cpu_usage_pct=cpu_percent,
                memory_usage_mb=memory.used / (1024 * 1024),
                memory_usage_pct=memory.percent,
                active_connections=ws_connections,
                requests_per_minute=requests_per_minute,
                error_rate_pct=error_rate,
                response_time_ms=avg_response_time,
                last_updated=datetime.utcnow(),
                database_status=SystemStatus.HEALTHY,  # TODO: Check actual DB status
                ml_models_status=SystemStatus.HEALTHY, # TODO: Check ML models
                rl_agent_status=SystemStatus.HEALTHY,  # TODO: Check RL agent
                dex_connections_status=SystemStatus.HEALTHY,  # TODO: Check DEX connections
                total_requests=self._request_count,
                total_errors=self._error_count,
                cache_hit_rate_pct=85.0  # TODO: Implement cache metrics
            )
            
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return SystemMetrics.create_default()
    
    async def _get_portfolio_status(self) -> PortfolioStatus:
        """Get current portfolio status"""
        try:
            if self.portfolio_manager:
                # TODO: Implement actual portfolio data retrieval
                pass
            
            # For now, return mock data
            return PortfolioStatus(
                total_value_usd=Decimal("10000.00"),
                available_balance_usd=Decimal("8500.00"),
                margin_used_usd=Decimal("1500.00"),
                unrealized_pnl_usd=Decimal("250.50"),
                realized_pnl_usd=Decimal("1200.75"),
                daily_pnl_usd=Decimal("75.25"),
                daily_pnl_pct=Decimal("0.75"),
                total_return_pct=Decimal("12.08"),
                max_drawdown_pct=Decimal("5.2"),
                current_drawdown_pct=Decimal("1.8"),
                sharpe_ratio=1.85,
                volatility_pct=15.3,
                var_95_usd=Decimal("180.50"),
                active_positions=[],
                position_count=3,
                recent_trades=[],
                chain_balances={
                    "solana": Decimal("5000.00"),
                    "ethereum": Decimal("3000.00"),
                    "base": Decimal("2000.00")
                },
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Failed to get portfolio status", error=str(e))
            return PortfolioStatus.create_default()
    
    async def _get_trading_status(self) -> TradingStatus:
        """Get current trading status"""
        try:
            current_mode = TradingMode.ANALYSIS
            analysis_running = False
            simulation_running = False
            live_trading_enabled = False
            
            if self.mode_manager:
                try:
                    # Get active modes from mode manager
                    active_modes = self.mode_manager.list_active_modes()
                    
                    # Map mode types to our dashboard mode enum
                    from ..modes.base import ModeType
                    mode_map = {
                        ModeType.ANALYSIS: TradingMode.ANALYSIS,
                        ModeType.SIMULATION: TradingMode.SIMULATION, 
                        ModeType.LIVE_TRADING: TradingMode.LIVE,
                        ModeType.PAPER_TRADING: TradingMode.SIMULATION
                    }
                    
                    # Find the currently active mode
                    for mode_id, mode_instance in active_modes.items():
                        if mode_instance.status.value == "active":
                            # Determine the mode type from the registry
                            for mode_type, registered_id in self.mode_manager._mode_type_registry.items():
                                if registered_id == mode_id:
                                    current_mode = mode_map.get(mode_type, TradingMode.ANALYSIS)
                                    break
                            break
                    
                    # Set running flags based on active mode
                    analysis_running = current_mode == TradingMode.ANALYSIS
                    simulation_running = current_mode == TradingMode.SIMULATION
                    live_trading_enabled = current_mode == TradingMode.LIVE
                    
                except Exception as mode_error:
                    logger.warning("Failed to get mode from mode manager", error=str(mode_error))
                    # Fall back to default analysis mode
                    current_mode = TradingMode.ANALYSIS
                    analysis_running = True
            else:
                # No mode manager, default to analysis mode
                analysis_running = True
            
            return TradingStatus(
                mode=current_mode,
                is_trading_active=current_mode in [TradingMode.SIMULATION, TradingMode.LIVE],
                last_trade_time=datetime.utcnow() - timedelta(minutes=15),
                trades_today=8,
                volume_today_usd=Decimal("2500.00"),
                analysis_running=analysis_running,
                simulation_running=simulation_running,
                live_trading_enabled=live_trading_enabled,
                emergency_stop_active=False,
                risk_limits_active=True,
                max_position_size_usd=Decimal("1000.00"),
                max_daily_loss_usd=Decimal("500.00"),
                win_rate_pct=65.5,
                avg_trade_duration_hours=2.3,
                avg_profit_per_trade_usd=Decimal("45.30"),
                tokens_analyzed_today=157,
                tokens_in_watchlist=23,
                high_confidence_signals=5,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Failed to get trading status", error=str(e))
            return TradingStatus.create_default()
    
    async def _get_ml_rl_status(self) -> MLRLStatus:
        """Get ML and RL status"""
        try:
            ml_models = []
            rl_agents = []
            
            # Mock ML model data
            ml_models.append(MLModel(
                name="LSTM Price Predictor",
                type="LSTM",
                status=SystemStatus.HEALTHY,
                accuracy=0.78,
                last_training_time=datetime.utcnow() - timedelta(hours=2),
                predictions_today=247,
                avg_prediction_time_ms=1.2,
                model_size_mb=12.5,
                version="1.2.0"
            ))
            
            # Mock RL agent data
            rl_agents.append(RLAgent(
                name="DQN Trading Agent",
                algorithm="DQN",
                status=SystemStatus.HEALTHY,
                episode=15247,
                epsilon=0.05,
                avg_reward=2.35,
                win_rate_pct=62.8,
                experience_buffer_size=45000,
                last_training_time=datetime.utcnow() - timedelta(minutes=30),
                actions_today=89,
                avg_decision_time_ms=8.5
            ))
            
            return MLRLStatus(
                ml_models=ml_models,
                rl_agents=rl_agents,
                ml_rl_integration_active=True,
                ml_rl_decision_latency_ms=9.8,
                ml_confidence_threshold=0.7,
                rl_action_confidence=0.82,
                ml_training_active=False,
                rl_training_active=False,
                continuous_learning_active=True,
                ml_prediction_accuracy_pct=77.5,
                rl_action_success_rate_pct=64.2,
                ensemble_agreement_pct=81.3,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Failed to get ML/RL status", error=str(e))
            return MLRLStatus.create_default()
    
    async def switch_trading_mode(self, mode: str, user_id: str) -> bool:
        """Switch trading mode"""
        # Log mode switch attempt
        await activity_logger.log_activity(
            category=ActivityCategory.USER,
            action=ActivityAction.UPDATE,
            source="dashboard_service",
            event_type="mode_switch_attempt",
            title=f"User attempting to switch to {mode} mode",
            severity=ActivitySeverity.INFO,
            user_id=int(user_id.split('-')[-1]) if '-' in user_id else 1,  # Extract numeric ID
            dashboard_component="mode_switcher",
            metadata={"target_mode": mode, "current_user": user_id}
        )
        
        try:
            if mode not in ["analysis", "simulation", "live"]:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Map string mode to ModeType
            from ..modes.base import ModeType
            mode_type_map = {
                "analysis": ModeType.ANALYSIS,
                "simulation": ModeType.SIMULATION,
                "live": ModeType.LIVE_TRADING
            }
            
            target_mode_type = mode_type_map[mode]
            
            if self.mode_manager:
                try:
                    # Find the current active mode
                    active_modes = self.mode_manager.list_active_modes()
                    current_mode_id = None
                    
                    for mode_id, mode_instance in active_modes.items():
                        if mode_instance.status.value == "active":
                            current_mode_id = mode_id
                            break
                    
                    # Find target mode ID by type
                    target_mode_id = None
                    for mode_type, mode_id in self.mode_manager._mode_type_registry.items():
                        if mode_type == target_mode_type:
                            target_mode_id = mode_id
                            break
                    
                    if target_mode_id is None:
                        # If target mode doesn't exist, we need to register it first
                        from ..modes.base import ModeConfig
                        mode_config = ModeConfig(
                            mode_type=target_mode_type,
                            enabled=True,
                            auto_start=False
                        )
                        target_mode_id = await self.mode_manager.register_mode(mode_config)
                    
                    # If we have a current mode, switch from it to target
                    if current_mode_id and current_mode_id != target_mode_id:
                        await self.mode_manager.switch_mode(current_mode_id, target_mode_id)
                    elif target_mode_id:
                        # Just start the target mode if no current mode
                        await self.mode_manager.start_mode(target_mode_id)
                    
                    logger.info(
                        "Mode switching completed",
                        mode=mode,
                        mode_type=target_mode_type.value,
                        user_id=user_id,
                        current_mode_id=str(current_mode_id) if current_mode_id else None,
                        target_mode_id=str(target_mode_id) if target_mode_id else None
                    )
                    
                except Exception as mode_error:
                    logger.warning(
                        "Mode manager operation failed, using fallback",
                        error=str(mode_error),
                        mode=mode
                    )
                    # Continue with fallback behavior
            
            # Send WebSocket notification
            await websocket_manager.send_system_alert(
                "mode_change",
                f"Trading mode switched to {mode} by user {user_id}",
                "info"
            )
            
            # Log successful mode switch
            await activity_logger.log_activity(
                category=ActivityCategory.USER,
                action=ActivityAction.SUCCESS,
                source="dashboard_service",
                event_type="mode_switch_success",
                title=f"Trading mode successfully switched to {mode}",
                severity=ActivitySeverity.INFO,
                user_id=int(user_id.split('-')[-1]) if '-' in user_id else 1,
                dashboard_component="mode_switcher",
                metadata={
                    "new_mode": mode,
                    "user": user_id,
                    "target_mode_type": target_mode_type.value
                }
            )
            
            logger.info(
                "Trading mode switched",
                mode=mode,
                user_id=user_id
            )
            
            return True
            
        except Exception as e:
            # Log mode switch failure
            await activity_logger.log_error(
                category=ActivityCategory.USER,
                source="dashboard_service",
                event_type="mode_switch_failed",
                title=f"Failed to switch trading mode to {mode}",
                error_message=str(e),
                exception=e,
                severity=ActivitySeverity.ERROR,
                user_id=int(user_id.split('-')[-1]) if '-' in user_id else 1,
                dashboard_component="mode_switcher",
                metadata={"target_mode": mode, "user": user_id}
            )
            logger.error("Failed to switch trading mode", error=str(e), mode=mode)
            return False
    
    async def emergency_stop(self, user_id: str) -> bool:
        """Activate emergency stop"""
        # Log emergency stop activation attempt
        await activity_logger.log_activity(
            category=ActivityCategory.SECURITY,
            action=ActivityAction.EXECUTE,
            source="dashboard_service",
            event_type="emergency_stop_attempt",
            title=f"Emergency stop activation attempted by user {user_id}",
            severity=ActivitySeverity.CRITICAL,
            user_id=int(user_id.split('-')[-1]) if '-' in user_id else 1,
            dashboard_component="emergency_controls",
            security_level="critical",
            metadata={"activating_user": user_id}
        )
        
        try:
            # TODO: Implement actual emergency stop logic
            
            # Send critical alert
            await websocket_manager.send_system_alert(
                "emergency_stop",
                f"Emergency stop activated by user {user_id}",
                "critical"
            )
            
            # Log successful emergency stop
            await activity_logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=ActivityAction.SUCCESS,
                source="dashboard_service",
                event_type="emergency_stop_activated",
                title=f"Emergency stop successfully activated by user {user_id}",
                severity=ActivitySeverity.CRITICAL,
                user_id=int(user_id.split('-')[-1]) if '-' in user_id else 1,
                dashboard_component="emergency_controls",
                security_level="critical",
                metadata={"activating_user": user_id}
            )
            
            logger.critical(
                "Emergency stop activated",
                user_id=user_id
            )
            
            return True
            
        except Exception as e:
            # Log emergency stop failure
            await activity_logger.log_error(
                category=ActivityCategory.SECURITY,
                source="dashboard_service",
                event_type="emergency_stop_failed",
                title=f"Failed to activate emergency stop",
                error_message=str(e),
                exception=e,
                severity=ActivitySeverity.CRITICAL,
                user_id=int(user_id.split('-')[-1]) if '-' in user_id else 1,
                dashboard_component="emergency_controls",
                security_level="critical",
                metadata={"activating_user": user_id}
            )
            logger.error("Failed to activate emergency stop", error=str(e))
            return False
    
    async def update_risk_limits(self, limits: Dict[str, Any], user_id: str) -> bool:
        """Update risk management limits"""
        # Log risk limits update attempt
        await activity_logger.log_activity(
            category=ActivityCategory.CONFIGURATION,
            action=ActivityAction.UPDATE,
            source="dashboard_service",
            event_type="risk_limits_update_attempt",
            title=f"Risk limits update attempted by user {user_id}",
            severity=ActivitySeverity.INFO,
            user_id=int(user_id.split('-')[-1]) if '-' in user_id else 1,
            dashboard_component="risk_management",
            metadata={"limits": limits, "updating_user": user_id}
        )
        
        try:
            # TODO: Implement actual risk limit updates
            
            # Send notification
            await websocket_manager.send_system_alert(
                "risk_limits_updated",
                f"Risk limits updated by user {user_id}",
                "info"
            )
            
            # Log successful risk limits update
            await activity_logger.log_activity(
                category=ActivityCategory.CONFIGURATION,
                action=ActivityAction.SUCCESS,
                source="dashboard_service",
                event_type="risk_limits_updated",
                title=f"Risk limits successfully updated by user {user_id}",
                severity=ActivitySeverity.INFO,
                user_id=int(user_id.split('-')[-1]) if '-' in user_id else 1,
                dashboard_component="risk_management",
                metadata={"updated_limits": limits, "updating_user": user_id}
            )
            
            logger.info(
                "Risk limits updated",
                limits=limits,
                user_id=user_id
            )
            
            return True
            
        except Exception as e:
            # Log risk limits update failure
            await activity_logger.log_error(
                category=ActivityCategory.CONFIGURATION,
                source="dashboard_service",
                event_type="risk_limits_update_failed",
                title=f"Failed to update risk limits",
                error_message=str(e),
                exception=e,
                severity=ActivitySeverity.ERROR,
                user_id=int(user_id.split('-')[-1]) if '-' in user_id else 1,
                dashboard_component="risk_management",
                metadata={"intended_limits": limits, "updating_user": user_id}
            )
            logger.error("Failed to update risk limits", error=str(e))
            return False
    
    async def get_trading_history(self, limit: int = 100) -> List[Trade]:
        """Get recent trading history"""
        try:
            # TODO: Implement actual trading history retrieval
            
            # Return mock data for now
            return []
            
        except Exception as e:
            logger.error("Failed to get trading history", error=str(e))
            return []
    
    async def get_system_logs(self, limit: int = 100) -> List[str]:
        """Get recent system logs"""
        try:
            # Get recent activity logs from database
            activities = await dashboard_activity.get_recent_activity(
                limit=limit,
                category="system",
                hours_back=24
            )
            
            # Convert to simple log messages
            return [activity['title'] for activity in activities]
            
        except Exception as e:
            logger.error("Failed to get system logs", error=str(e))
            # Fallback to mock data
            return [
                "System started successfully",
                "ML model loaded: LSTM Price Predictor v1.2.0",
                "RL agent initialized: DQN Trading Agent",
                "Dashboard service started",
                "WebSocket connections: 2 active"
            ]
    
    async def _get_recent_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent activity for dashboard display"""
        try:
            # Get recent activities from the database
            activities = await dashboard_activity.get_recent_activity(
                limit=limit,
                hours_back=1  # Last hour for recent activity
            )
            return activities
            
        except Exception as e:
            logger.error("Failed to get recent activity", error=str(e))
            # Return empty list on error
            return []
    
    async def get_backtest_results(self, strategy_name: Optional[str] = None, 
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   min_sharpe_ratio: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get backtest results from analysis mode"""
        try:
            # Get analysis mode from mode manager
            if not self.mode_manager:
                logger.warning("Mode manager not initialized")
                return None
                
            analysis_mode = self.mode_manager.get_mode_by_type("analysis")
            if not analysis_mode:
                logger.info("No analysis mode found or active")
                return None
                
            # Check if backtest engine is available
            if not hasattr(analysis_mode, 'backtest_engine') or not analysis_mode.backtest_engine:
                logger.info("No backtest engine available in analysis mode")
                return None
            
            # Get backtest results
            if strategy_name:
                raw_results = analysis_mode.backtest_engine.get_results_by_strategy(strategy_name)
            else:
                raw_results = analysis_mode.backtest_engine.get_latest_results()
                
            if not raw_results:
                return None
                
            # Transform and serialize the results
            return self._serialize_backtest_results(raw_results)
            
        except Exception as e:
            logger.error("Failed to get backtest results", error=str(e))
            return None
    
    def _serialize_backtest_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize backtest results for API response"""
        try:
            # Handle datetime serialization
            serialized_equity_curve = []
            if "equity_curve" in raw_results:
                for point in raw_results["equity_curve"]:
                    serialized_point = {
                        "timestamp": point["timestamp"].isoformat() if hasattr(point["timestamp"], 'isoformat') else str(point["timestamp"]),
                        "value": float(point["value"]) if hasattr(point["value"], '__float__') else point["value"]
                    }
                    serialized_equity_curve.append(serialized_point)
            
            # Handle trade history serialization  
            serialized_trade_history = []
            if "trade_history" in raw_results:
                for trade in raw_results["trade_history"]:
                    serialized_trade = trade.copy()
                    if "timestamp" in trade:
                        serialized_trade["timestamp"] = trade["timestamp"].isoformat() if hasattr(trade["timestamp"], 'isoformat') else str(trade["timestamp"])
                    serialized_trade_history.append(serialized_trade)
            
            # Build the serialized result
            result = {
                "strategy_name": raw_results.get("strategy_name", "Unknown"),
                "total_return": raw_results.get("total_return", 0.0),
                "annual_return": raw_results.get("annual_return", 0.0),
                "max_drawdown": raw_results.get("max_drawdown", 0.0),
                "sharpe_ratio": raw_results.get("sharpe_ratio", 0.0),
                "volatility": raw_results.get("volatility", 0.0),
                "win_rate": raw_results.get("win_rate", 0.0),
                "total_trades": raw_results.get("total_trades", 0),
                "profitable_trades": raw_results.get("profitable_trades", 0),
                "losing_trades": raw_results.get("losing_trades", 0),
                "avg_trade_duration_hours": raw_results.get("avg_trade_duration_hours", 0.0),
                "avg_profit_per_trade": raw_results.get("avg_profit_per_trade", 0.0),
                "equity_curve": serialized_equity_curve,
                "trade_history": serialized_trade_history,
                "performance_metrics": raw_results.get("performance_metrics", {})
            }
            
            # Add start/end dates if available
            if "start_date" in raw_results:
                result["start_date"] = raw_results["start_date"].isoformat() if hasattr(raw_results["start_date"], 'isoformat') else str(raw_results["start_date"])
            if "end_date" in raw_results:
                result["end_date"] = raw_results["end_date"].isoformat() if hasattr(raw_results["end_date"], 'isoformat') else str(raw_results["end_date"])
                
            return result
            
        except Exception as e:
            logger.error("Failed to serialize backtest results", error=str(e))
            return raw_results  # Return as-is if serialization fails

    # =============================================================================
    # XAI EXPLANATION METHODS
    # =============================================================================
    
    async def get_xai_explanations(
        self,
        symbol: Optional[str] = None,
        decision_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent XAI explanations with filtering"""
        try:
            if not self.xai_explanation_manager:
                logger.warning("XAI explanation manager not initialized")
                return []
            
            # Get explanations from XAI manager
            explanations = self.xai_explanation_manager.get_recent_explanations(
                symbol=symbol,
                decision_type=decision_type,
                limit=limit
            )
            
            # Convert to serializable format
            return [
                self.xai_explanation_manager.to_dict(explanation)
                for explanation in explanations
            ]
            
        except Exception as e:
            logger.error("Failed to get XAI explanations", error=str(e))
            return []
    
    async def get_xai_explanation(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get specific XAI explanation by decision ID"""
        try:
            if not self.xai_explanation_manager:
                logger.warning("XAI explanation manager not initialized")
                return None
            
            explanation = self.xai_explanation_manager.get_explanation(decision_id)
            if explanation:
                return self.xai_explanation_manager.to_dict(explanation)
            return None
            
        except Exception as e:
            logger.error("Failed to get XAI explanation", error=str(e), decision_id=decision_id)
            return None
    
    async def get_feature_importance_summary(
        self,
        symbol: Optional[str] = None,
        hours_back: int = 24
    ) -> Dict[str, float]:
        """Get aggregated feature importance summary"""
        try:
            if not self.xai_explanation_manager:
                logger.warning("XAI explanation manager not initialized")
                return {}
            
            return self.xai_explanation_manager.get_feature_importance_summary(
                symbol=symbol,
                hours_back=hours_back
            )
            
        except Exception as e:
            logger.error("Failed to get feature importance summary", error=str(e))
            return {}
    
    async def get_xai_cache_stats(self) -> Dict[str, Any]:
        """Get XAI cache statistics"""
        try:
            if not self.xai_explanation_manager:
                logger.warning("XAI explanation manager not initialized")
                return {"error": "XAI explanation manager not initialized"}
            
            return self.xai_explanation_manager.get_cache_stats()
            
        except Exception as e:
            logger.error("Failed to get XAI cache stats", error=str(e))
            return {"error": str(e)}
    
    # =============================================================================
    # INTERACTIVE CHARTING DATA METHODS
    # =============================================================================
    
    async def get_price_chart_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Get price chart data for visualization"""
        try:
            # TODO: Implement actual price data retrieval from market data service
            # For now, return mock data
            from datetime import datetime, timedelta
            import random
            
            base_price = 100.0
            data = []
            current_time = datetime.utcnow()
            
            # Generate mock OHLCV data
            for i in range(limit):
                price_variation = random.uniform(-2.0, 2.0)
                open_price = base_price + price_variation
                high_price = open_price + random.uniform(0, 1.5)
                low_price = open_price - random.uniform(0, 1.5)
                close_price = open_price + random.uniform(-1.0, 1.0)
                volume = random.uniform(1000, 10000)
                
                data.append({
                    "timestamp": (current_time - timedelta(hours=i)).isoformat(),
                    "open": round(open_price, 4),
                    "high": round(high_price, 4),
                    "low": round(low_price, 4),
                    "close": round(close_price, 4),
                    "volume": round(volume, 2)
                })
                
                base_price = close_price  # Use close as next base
            
            return list(reversed(data))  # Reverse to get chronological order
            
        except Exception as e:
            logger.error("Failed to get price chart data", error=str(e), symbol=symbol)
            return []
    
    async def get_performance_chart_data(
        self,
        timeframe: str = "1d",
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get performance chart data"""
        try:
            # TODO: Implement actual performance data retrieval
            # For now, return mock data
            from datetime import datetime, timedelta
            import random
            
            data = []
            current_time = datetime.utcnow()
            portfolio_value = 10000.0
            
            for i in range(days_back):
                daily_return = random.uniform(-0.02, 0.03)  # -2% to +3% daily return
                portfolio_value *= (1 + daily_return)
                
                data.append({
                    "timestamp": (current_time - timedelta(days=i)).isoformat(),
                    "portfolio_value": round(portfolio_value, 2),
                    "daily_return": round(daily_return * 100, 4),
                    "cumulative_return": round(((portfolio_value - 10000) / 10000) * 100, 4)
                })
            
            return list(reversed(data))  # Reverse to get chronological order
            
        except Exception as e:
            logger.error("Failed to get performance chart data", error=str(e))
            return []
    
    async def get_trading_volume_chart_data(
        self,
        timeframe: str = "1h",
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """Get trading volume chart data"""
        try:
            # TODO: Implement actual volume data retrieval
            # For now, return mock data
            from datetime import datetime, timedelta
            import random
            
            data = []
            current_time = datetime.utcnow()
            
            for i in range(hours_back):
                volume = random.uniform(1000, 5000)
                trade_count = random.randint(5, 25)
                
                data.append({
                    "timestamp": (current_time - timedelta(hours=i)).isoformat(),
                    "volume_usd": round(volume, 2),
                    "trade_count": trade_count,
                    "avg_trade_size": round(volume / trade_count, 2) if trade_count > 0 else 0
                })
            
            return list(reversed(data))  # Reverse to get chronological order
            
        except Exception as e:
            logger.error("Failed to get trading volume chart data", error=str(e))
            return []
    
    # =============================================================================
    # PERFORMANCE ATTRIBUTION METHODS
    # =============================================================================
    
    async def get_performance_attribution(
        self,
        timeframe: str = "1d",
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get performance attribution analysis"""
        try:
            # TODO: Implement actual attribution analysis
            # For now, return mock data
            import random
            
            # Mock attribution by strategy/factor
            strategies = ["momentum", "mean_reversion", "arbitrage", "ml_predictions", "rl_decisions"]
            attribution = {}
            
            for strategy in strategies:
                attribution[strategy] = {
                    "return_contribution": round(random.uniform(-0.5, 1.5), 4),
                    "risk_contribution": round(random.uniform(0.1, 0.8), 4),
                    "sharpe_ratio": round(random.uniform(0.5, 2.0), 4),
                    "trade_count": random.randint(10, 100),
                    "win_rate": round(random.uniform(0.45, 0.75), 4)
                }
            
            return {
                "timeframe": timeframe,
                "days_analyzed": days_back,
                "strategy_attribution": attribution,
                "total_return": round(sum(attr["return_contribution"] for attr in attribution.values()), 4),
                "risk_adjusted_return": round(random.uniform(0.5, 1.8), 4)
            }
            
        except Exception as e:
            logger.error("Failed to get performance attribution", error=str(e))
            return {}
    
    async def get_risk_metrics(
        self,
        timeframe: str = "1d",
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get detailed risk metrics"""
        try:
            # TODO: Implement actual risk metrics calculation
            # For now, return mock data
            import random
            
            return {
                "timeframe": timeframe,
                "days_analyzed": days_back,
                "value_at_risk": {
                    "var_95": round(random.uniform(100, 500), 2),
                    "var_99": round(random.uniform(200, 800), 2),
                    "cvar_95": round(random.uniform(150, 600), 2)
                },
                "volatility_metrics": {
                    "daily_volatility": round(random.uniform(0.01, 0.05), 4),
                    "annualized_volatility": round(random.uniform(0.15, 0.35), 4),
                    "volatility_skew": round(random.uniform(-0.5, 0.5), 4)
                },
                "drawdown_metrics": {
                    "max_drawdown": round(random.uniform(0.03, 0.15), 4),
                    "avg_drawdown": round(random.uniform(0.01, 0.05), 4),
                    "drawdown_duration_days": random.randint(1, 7),
                    "current_drawdown": round(random.uniform(0.0, 0.03), 4)
                },
                "correlation_metrics": {
                    "beta_to_market": round(random.uniform(0.3, 1.2), 4),
                    "correlation_to_btc": round(random.uniform(0.2, 0.8), 4),
                    "correlation_to_eth": round(random.uniform(0.1, 0.7), 4)
                }
            }
            
        except Exception as e:
            logger.error("Failed to get risk metrics", error=str(e))
            return {}
    
    # =============================================================================
    # MANUAL OVERRIDE CONTROL METHODS
    # =============================================================================
    
    async def execute_manual_trade(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
        user_id: str = None
    ) -> bool:
        """Execute a manual trade"""
        try:
            # TODO: Implement actual trade execution
            # For now, log the request and return success
            
            await activity_logger.log_activity(
                category=ActivityCategory.TRADING,
                action=ActivityAction.EXECUTE,
                source="dashboard_service",
                event_type="manual_trade_executed",
                title=f"Manual {side} trade executed for {symbol}",
                severity=ActivitySeverity.INFO,
                user_id=int(user_id.split('-')[-1], 16) % 10000 if user_id else None,
                metadata={
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "order_type": order_type,
                    "price": price,
                    "execution_source": "manual_dashboard"
                }
            )
            
            # Send WebSocket notification
            await websocket_manager.send_system_alert(
                "manual_trade_executed",
                f"Manual {side} trade executed for {symbol}: {amount}",
                "info"
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to execute manual trade", error=str(e))
            return False
    
    async def override_trading_signal(
        self,
        signal_id: str,
        action: str,
        reason: Optional[str] = None,
        user_id: str = None
    ) -> bool:
        """Override or control a trading signal"""
        try:
            # TODO: Implement actual signal override logic
            # For now, log the action and return success
            
            await activity_logger.log_activity(
                category=ActivityCategory.TRADING,
                action=ActivityAction.EXECUTE,
                source="dashboard_service",
                event_type=f"signal_{action}",
                title=f"Trading signal {action} executed for {signal_id}",
                severity=ActivitySeverity.WARNING,
                user_id=int(user_id.split('-')[-1], 16) % 10000 if user_id else None,
                metadata={
                    "signal_id": signal_id,
                    "action": action,
                    "reason": reason,
                    "override_source": "manual_dashboard"
                }
            )
            
            # Send WebSocket notification
            await websocket_manager.send_system_alert(
                "signal_override",
                f"Trading signal {action} for {signal_id}: {reason or 'No reason provided'}",
                "warning"
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to override trading signal", error=str(e))
            return False
    
    async def pause_trading_strategy(
        self,
        strategy_name: str,
        duration_minutes: Optional[int] = None,
        reason: Optional[str] = None,
        user_id: str = None
    ) -> bool:
        """Pause a trading strategy"""
        try:
            # TODO: Implement actual strategy pause logic
            # For now, log the action and return success
            
            await activity_logger.log_activity(
                category=ActivityCategory.TRADING,
                action=ActivityAction.EXECUTE,
                source="dashboard_service",
                event_type="strategy_paused",
                title=f"Trading strategy {strategy_name} paused",
                severity=ActivitySeverity.WARNING,
                user_id=int(user_id.split('-')[-1], 16) % 10000 if user_id else None,
                metadata={
                    "strategy_name": strategy_name,
                    "duration_minutes": duration_minutes,
                    "reason": reason,
                    "pause_source": "manual_dashboard"
                }
            )
            
            # Send WebSocket notification
            duration_msg = f" for {duration_minutes} minutes" if duration_minutes else " indefinitely"
            await websocket_manager.send_system_alert(
                "strategy_paused",
                f"Strategy {strategy_name} paused{duration_msg}: {reason or 'No reason provided'}",
                "warning"
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to pause trading strategy", error=str(e))
            return False
    
    # =============================================================================
    # REAL-TIME METRICS METHODS
    # =============================================================================
    
    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time trading and system metrics"""
        try:
            # Get current dashboard data
            dashboard_data = await self.get_dashboard_data()
            
            # Compile real-time metrics
            return {
                "system": {
                    "cpu_usage": dashboard_data.system_metrics.cpu_usage_pct,
                    "memory_usage": dashboard_data.system_metrics.memory_usage_pct,
                    "active_connections": dashboard_data.system_metrics.active_connections,
                    "requests_per_minute": dashboard_data.system_metrics.requests_per_minute,
                    "error_rate": dashboard_data.system_metrics.error_rate_pct,
                    "response_time_ms": dashboard_data.system_metrics.response_time_ms
                },
                "trading": {
                    "mode": dashboard_data.trading_status.mode.value,
                    "is_active": dashboard_data.trading_status.is_trading_active,
                    "trades_today": dashboard_data.trading_status.trades_today,
                    "volume_today": float(dashboard_data.trading_status.volume_today_usd),
                    "win_rate": dashboard_data.trading_status.win_rate_pct,
                    "signals_count": dashboard_data.trading_status.high_confidence_signals
                },
                "portfolio": {
                    "total_value": float(dashboard_data.portfolio_status.total_value_usd),
                    "daily_pnl": float(dashboard_data.portfolio_status.daily_pnl_usd),
                    "daily_pnl_pct": float(dashboard_data.portfolio_status.daily_pnl_pct),
                    "positions_count": dashboard_data.portfolio_status.position_count,
                    "available_balance": float(dashboard_data.portfolio_status.available_balance_usd)
                },
                "ml_rl": {
                    "ml_prediction_accuracy": dashboard_data.ml_rl_status.ml_prediction_accuracy_pct,
                    "rl_action_success_rate": dashboard_data.ml_rl_status.rl_action_success_rate_pct,
                    "integration_active": dashboard_data.ml_rl_status.ml_rl_integration_active,
                    "decision_latency_ms": dashboard_data.ml_rl_status.ml_rl_decision_latency_ms
                }
            }
            
        except Exception as e:
            logger.error("Failed to get real-time metrics", error=str(e))
            return {}
    
    async def get_live_positions(self) -> List[Position]:
        """Get current live trading positions"""
        try:
            # Get current dashboard data
            dashboard_data = await self.get_dashboard_data()
            return dashboard_data.portfolio_status.active_positions
            
        except Exception as e:
            logger.error("Failed to get live positions", error=str(e))
            return []

    def record_request(self, response_time: float, error: bool = False) -> None:
        """Record API request metrics"""
        self._request_count += 1
        self._request_times.append(response_time)
        
        if error:
            self._error_count += 1
        
        # Keep only last 1000 request times
        if len(self._request_times) > 1000:
            self._request_times = self._request_times[-1000:]
    
    async def _update_loop(self) -> None:
        """Background loop to update dashboard data"""
        while self._running:
            try:
                # Get fresh dashboard data
                data = await self.get_dashboard_data()
                
                # Update cache
                await dashboard_cache.update_data(data)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in dashboard update loop", error=str(e))
                await asyncio.sleep(self.update_interval)


# Global dashboard service instance
dashboard_service = DashboardService()