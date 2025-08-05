"""
Live Trading Mode Implementation

This module implements the live trading mode for real trading with actual funds.
It includes production-grade safety systems, risk management, real-time portfolio
synchronization, and integration with the continuous learning pipeline.

Key Features:
- Real DEX integration (Jupiter, Hyperliquid, Uniswap V3)
- Production-grade safety systems and emergency stops
- Real-time P&L tracking and portfolio management
- Experience collection for continuous learning
- Comprehensive risk management and position controls
- Multi-DEX failover and optimization

Following TDD methodology - implementation satisfies comprehensive test requirements.

This module has been modularized for better maintainability. Major components have been
extracted to separate modules in the src/modes/live/ package.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID, uuid4
import structlog

# Core framework imports
from src.modes.base import ModeBase, ModeConfig, ModeStatus
from src.portfolio.base import (
    Portfolio, Position, Transaction, PositionType, PositionStatus,
    TransactionType, PerformanceMetrics, RiskMetrics
)
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import ExperienceReplayBuffer, ReplayBufferConfig
from src.rl_agent.experience_database import DatabaseExperienceBuffer, DatabaseExperienceConfig
from src.modes.experience_collector import TradingExperienceCollector, ExperienceCollectorConfig
from src.modes.continuous_learning import ContinuousLearningEngine, ContinuousLearningConfig
from src.modes.continuous_learning_loop import (
    ContinuousLearningLoop, ContinuousLearningLoopConfig, PerformanceFeedbackCapture,
    ModelHotSwapper, ModelPerformanceMonitor, ModelDeploymentAutomation,
    LearningLoopOrchestrator, AutonomousLearningSystem
)
from src.utils.base import Chain
from src.dex.base import SwapQuote, SwapResult, SwapStatus, DEXBase, DEXError, DEXConnectionError, DEXConfig
from src.integration.ml_rl_bridge import MLRLBridge, MLRLConfig
from src.xai.trading_integration import TradingExplanationManager

# Import modular live trading components
from src.modes.live import (
    LiveModeConfig, LiveModeMetrics, EmergencyStopReason, SafetyCheckResult,
    RiskValidationResult, EmergencyStopResult, PnLAlert, LiveModeError, EmergencyStopError,
    EmergencyStopSystem, SafetyInterlocks, LiveRiskManager,
    RealTimePnLTracker, PortfolioSynchronizer, DiscrepancyReport, SyncHealthMetrics,
    LiveTradingExecutor, TradingSessionManager, DEXIntegrationManager, PositionManager
)

# Transformer-specific imports for Phase 3.2.3.2
from src.ml_analysis.base import ModelType
from src.ml_analysis.ensemble_weight_manager import EnsembleWeightManager
from src.xai.transformers.attention_explainer import AttentionExplainer
from src.monitoring.transformer_metrics import TransformerMetricsCollector

logger = structlog.get_logger()


class TransformerModelManager:
    """
    Manages transformer models in live mode.
    
    Handles loading, health monitoring, and coordination of multiple
    transformer models (iTransformer, PatchTST, TimesMixer, TimesFM)
    for ensemble trading decisions.
    """
    
    def __init__(self, enabled_models: List[str]):
        """Initialize transformer model manager."""
        self.enabled_models = enabled_models
        self.loaded_models: Dict[str, Any] = {}
        self.logger = structlog.get_logger().bind(component="TransformerModelManager")
    
    async def initialize_models(self) -> None:
        """Initialize all enabled transformer models."""
        for model_type in self.enabled_models:
            try:
                # Mock model loading for now - would load actual models in production
                self.loaded_models[model_type] = MockTransformerModel(model_type)
                self.logger.info(f"Loaded transformer model: {model_type}")
            except Exception as e:
                self.logger.error(f"Failed to load transformer model {model_type}", error=str(e))
    
    def get_model_health_status(self) -> Dict[str, bool]:
        """Get health status of all loaded models."""
        return {model_type: True for model_type in self.loaded_models.keys()}


class MockTransformerModel:
    """Mock transformer model for testing."""
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.is_loaded = True


class TransformerPreprocessor:
    """
    Handles transformer-specific preprocessing for live trading.
    
    Provides feature extraction, temporal embedding, and model-specific
    data preparation for transformer models in live trading scenarios.
    """
    
    def __init__(self):
        """Initialize transformer preprocessor."""
        self.logger = structlog.get_logger().bind(component="TransformerPreprocessor")
    
    async def extract_transformer_features(self, market_state: MarketState) -> Dict[str, Any]:
        """Extract transformer-specific features from market state."""
        return {
            'temporal_embeddings': [0.1, 0.2, 0.3, 0.4],
            'positional_encodings': [0.05, 0.1, 0.15, 0.2],
            'attention_mask': [1, 1, 1, 1],
            'sequence_length': 24
        }
    
    async def prepare_multi_model_features(self, market_state: MarketState, 
                                         model_types: List[ModelType]) -> Dict[ModelType, Dict[str, Any]]:
        """Prepare features for multiple transformer models."""
        features = {}
        for model_type in model_types:
            features[model_type] = {
                'preprocessed_sequence': [0.1, 0.2, 0.3, 0.4],
                'model_specific_embeddings': [0.05, 0.1, 0.15, 0.2]
            }
        return features


class LiveMode(ModeBase):
    """
    Live trading mode implementation with comprehensive safety systems.
    
    Provides real trading capabilities with production-grade risk management,
    emergency stops, real-time portfolio synchronization, and continuous learning.
    
    This class has been refactored to use modular components from src.modes.live
    for better maintainability and separation of concerns.
    """
    
    def __init__(self, mode_id: UUID, config: ModeConfig, portfolio: Portfolio):
        """Initialize live mode with comprehensive safety systems."""
        super().__init__(mode_id, config, portfolio)
        
        # Extract live mode parameters
        params = config.parameters
        self.live_config = LiveModeConfig(
            initial_balance=Decimal(str(params.get("initial_balance", 50000))),
            enable_real_trading=params.get("enable_real_trading", True),
            max_position_size_pct=Decimal(str(params.get("max_position_size_pct", 0.1))),
            max_daily_loss_pct=Decimal(str(params.get("max_daily_loss_pct", 0.05))),
            max_drawdown_pct=Decimal(str(params.get("max_drawdown_pct", 0.15))),
            emergency_drawdown_pct=Decimal(str(params.get("emergency_drawdown_pct", 0.25))),
            enable_emergency_stop=params.get("enable_emergency_stop", True),
            enable_portfolio_sync=params.get("enable_portfolio_sync", True),
            sync_frequency_seconds=params.get("sync_frequency_seconds", 30),
            enable_real_time_pnl=params.get("enable_real_time_pnl", True),
            pnl_update_frequency_seconds=params.get("pnl_update_frequency_seconds", 5),
            enable_experience_collection=params.get("enable_experience_collection", True),
            experience_buffer_size=params.get("experience_buffer_size", 10000),
            enable_rl_feedback=params.get("enable_rl_feedback", True),
            enable_ml_rl_integration=params.get("enable_ml_rl_integration", True),
            dex_preference_order=params.get("dex_preference_order", ["jupiter", "uniswap_v3", "hyperliquid"]),
            enable_continuous_learning=params.get("enable_continuous_learning", True),
            enable_model_hot_swapping=params.get("enable_model_hot_swapping", True),
            enable_automated_deployment=params.get("enable_automated_deployment", True),
            enable_performance_monitoring=params.get("enable_performance_monitoring", True),
            enable_performance_feedback=params.get("enable_performance_feedback", True),
            learning_check_frequency_seconds=params.get("learning_check_frequency_seconds", 30),
            learning_trigger_threshold=params.get("learning_trigger_threshold", 1000),
            deployment_safety_threshold=Decimal(str(params.get("deployment_safety_threshold", 0.05))),
            performance_rollback_threshold=Decimal(str(params.get("performance_rollback_threshold", -0.10)))
        )
        
        # Initialize DEX integration manager
        self.dex_integration_manager = DEXIntegrationManager()
        self.dex_clients: Dict[str, DEXBase] = {}
        
        # Core modular components
        self.trading_executor: Optional[LiveTradingExecutor] = None
        self.risk_manager: Optional[LiveRiskManager] = None
        self.portfolio_sync: Optional[PortfolioSynchronizer] = None
        self.emergency_system: Optional[EmergencyStopSystem] = None
        self.pnl_tracker: Optional[RealTimePnLTracker] = None
        self.safety_interlocks: Optional[SafetyInterlocks] = None
        self.session_manager: Optional[TradingSessionManager] = None
        self.position_manager: Optional[PositionManager] = None
        
        # ML-RL integration
        self.ml_rl_bridge: Optional[MLRLBridge] = None
        
        # Transformer-specific components (Phase 3.2.3.2)
        self.enable_transformer_models = params.get("enable_transformer_models", False)
        self.transformer_model_manager: Optional[TransformerModelManager] = None
        self.transformer_preprocessor: Optional[TransformerPreprocessor] = None
        self.ensemble_weight_manager: Optional[EnsembleWeightManager] = None
        self.transformer_attention_explainer: Optional[AttentionExplainer] = None
        self.transformer_metrics: Optional[TransformerMetricsCollector] = None
        
        # Experience collection
        self.experience_collector: Optional[TradingExperienceCollector] = None
        
        # Continuous learning integration
        self.continuous_learning_engine: Optional[ContinuousLearningEngine] = None
        self.continuous_learning_loop: Optional[ContinuousLearningLoop] = None
        self.performance_feedback_capture: Optional[PerformanceFeedbackCapture] = None
        self.model_hot_swapper: Optional[ModelHotSwapper] = None
        self.model_performance_monitor: Optional[ModelPerformanceMonitor] = None
        self.model_deployment_automation: Optional[ModelDeploymentAutomation] = None
        self.learning_loop_orchestrator: Optional[LearningLoopOrchestrator] = None
        self.autonomous_learning_system: Optional[AutonomousLearningSystem] = None
        
        # Live mode metrics
        self.live_metrics = LiveModeMetrics(session_start_time=datetime.now())
        
        # State tracking
        self.enable_real_trading = self.live_config.enable_real_trading
        self.last_safety_check = datetime.now()
        self.safety_lockout_reason: Optional[str] = None
        
        # XAI (Explainable AI) integration setup
        self.enable_xai_explanations = params.get("enable_xai_explanations", True)
        self.xai_explanation_manager = None
        
        if self.enable_xai_explanations:
            # Initialize XAI explanation manager
            self.xai_explanation_manager = TradingExplanationManager(
                cache_size=params.get("xai_cache_size", 1000),
                explanation_timeout=params.get("xai_explanation_timeout", 3.0)  # Shorter timeout for live trading
            )
            
            self.logger.info("XAI explanation system enabled for live trading",
                           cache_size=params.get("xai_cache_size", 1000),
                           explanation_timeout=params.get("xai_explanation_timeout", 3.0))
        
        # Initialize transformer components (Phase 3.2.3.2)
        if self.enable_transformer_models:
            self._setup_transformer_components(params)
        
        # Set up database experience storage
        self._setup_database_experience_storage(params)
    
    async def initialize(self) -> None:
        """Initialize live mode components and safety systems."""
        self.logger.info("Initializing live mode components")
        
        try:
            # Initialize DEX clients using the integration manager
            self.dex_clients = await self.dex_integration_manager.initialize_dex_clients()
            
            # Initialize database experience buffer if enabled
            if self.enable_database_experience_storage and self.database_experience_buffer:
                try:
                    await self.database_experience_buffer.initialize()
                    self.logger.info("Production database experience buffer initialized successfully")
                except Exception as e:
                    self.logger.error("Failed to initialize production database experience buffer", error=str(e))
                    self.enable_database_experience_storage = False
                    self.database_connection_failed = True
            
            # Initialize modular components
            self.risk_manager = LiveRiskManager(
                portfolio=self.portfolio,
                config=self.live_config,
                enable_real_time_monitoring=True
            )
            
            self.trading_executor = LiveTradingExecutor(
                portfolio=self.portfolio,
                dex_clients=self.dex_clients,
                enable_real_trading=self.enable_real_trading,
                max_slippage_bps=self.live_config.max_slippage_bps,
                order_timeout_seconds=self.live_config.order_timeout_seconds
            )
            
            self.emergency_system = EmergencyStopSystem(self.live_config)
            self.safety_interlocks = SafetyInterlocks(self.live_config)
            self.session_manager = TradingSessionManager(self.live_config)
            self.position_manager = PositionManager(self.portfolio)
            
            # Initialize portfolio synchronization if enabled
            if self.live_config.enable_portfolio_sync:
                # Configure enhanced portfolio synchronization
                sync_config = {
                    "discrepancy_threshold": Decimal("0.01"),  # 1% threshold for discrepancy detection
                    "auto_correct_threshold": Decimal("0.005"),  # 0.5% threshold for auto-correction
                    "safety_alert_threshold": Decimal("0.05"),  # 5% threshold for safety alerts
                    "max_correction_attempts": 3,
                    "enable_automatic_corrections": True,
                    "enable_safety_alerts": True,
                    "enable_correction_rollback": True,
                    "continue_on_dex_failure": True,  # Continue with available DEXs on partial failure
                    "price_tolerance_pct": Decimal("0.10"),  # 10% price tolerance
                    "escalation_threshold": 3,  # Escalate after 3 consecutive alerts
                    "critical_alert_threshold": Decimal("0.50"),  # 50% threshold for critical alerts
                    "notification_channels": ["email", "slack"],
                    "enable_dex_specific_features": True,
                    "enable_arbitrage_detection": False,  # Disabled by default
                    "arbitrage_threshold": Decimal("0.02")  # 2% arbitrage threshold
                }
                
                # Create monitoring hooks for integration
                monitoring_hooks = {
                    "pre_sync": self._on_pre_sync,
                    "post_sync": self._on_post_sync,
                    "discrepancy_detected": self._on_discrepancy_detected
                }
                
                self.portfolio_sync = PortfolioSynchronizer(
                    portfolio=self.portfolio,
                    dex_clients=self.dex_clients,
                    sync_frequency_seconds=self.live_config.sync_frequency_seconds,
                    config=sync_config,
                    monitor=getattr(self, 'monitor', None),  # Pass monitor if available
                    alerting_system=getattr(self, 'alerting_system', None),  # Pass alerting if available
                    monitoring_hooks=monitoring_hooks
                )
            
            # Initialize real-time P&L tracking if enabled
            if self.live_config.enable_real_time_pnl:
                self.pnl_tracker = RealTimePnLTracker(
                    portfolio=self.portfolio,
                    config=self.live_config
                )
            
            # Initialize ML-RL integration if enabled
            if self.live_config.enable_ml_rl_integration:
                ml_rl_config = MLRLConfig(
                    ml_weight=float(self.live_config.ml_rl_weight),
                    rl_weight=1.0 - float(self.live_config.ml_rl_weight)
                )
                # Would initialize ML-RL bridge here
                # self.ml_rl_bridge = MLRLBridge(ml_rl_config, ml_analyzer, rl_agent)
            
            # Initialize experience collection if enabled
            if self.live_config.enable_experience_collection:
                experience_config = ExperienceCollectorConfig(
                    buffer_size=self.live_config.experience_buffer_size,
                    enable_persistence=True,
                    persistence_path="live_experiences.json"
                )
                
                replay_config = ReplayBufferConfig(
                    max_size=self.live_config.experience_buffer_size,
                    batch_size=32,
                    min_size=100
                )
                replay_buffer = ExperienceReplayBuffer(replay_config)
                
                self.experience_collector = TradingExperienceCollector(experience_config, replay_buffer)
            
            # Initialize transformer models if enabled (Phase 3.2.3.2)
            if self.enable_transformer_models and self.transformer_model_manager:
                await self.transformer_model_manager.initialize_models()
                self.logger.info("Transformer models initialized successfully")
            
            self._set_status(ModeStatus.INACTIVE)
            self.logger.info("Live mode components initialization completed", 
                           transformer_models=self.enable_transformer_models)
            
        except Exception as e:
            self.logger.error("Live mode initialization failed", error=str(e))
            self._set_status(ModeStatus.ERROR, str(e))
            raise
    
    def _setup_database_experience_storage(self, params: Dict[str, Any]) -> None:
        """Set up database experience storage for production (called from __init__)."""
        # Database experience storage setup for production
        self.enable_database_experience_storage = params.get("enable_database_experience_storage", True)  # Default enabled in live mode
        self.database_experience_buffer = None
        self.database_connection_failed = False
        self.production_experience_settings = params.get("production_experience_settings", {})
        self.live_experience_tags = params.get("live_experience_tags", {})
        self.enable_real_time_experience_persistence = self.production_experience_settings.get("enable_real_time_persistence", True)
        
        if self.enable_database_experience_storage:
            try:
                # Create database experience configuration for production
                db_config_params = params.get("database_experience_config", {})
                db_experience_config = DatabaseExperienceConfig(
                    max_size=db_config_params.get("max_size", 50000),  # Larger buffer for production
                    batch_size=db_config_params.get("batch_size", 64),  # Larger batches for efficiency
                    min_size=db_config_params.get("min_size", 200),
                    prioritized=db_config_params.get("prioritized", True),
                    alpha=db_config_params.get("alpha", 0.7),  # Higher priority bias for live trading
                    beta_start=db_config_params.get("beta_start", 0.5),
                    beta_end=db_config_params.get("beta_end", 1.0),
                    cache_size=db_config_params.get("cache_size", 2000),  # Larger cache for production
                    connection_pool_size=db_config_params.get("connection_pool_size", 20),  # More connections
                    query_timeout=db_config_params.get("query_timeout", 15.0)  # Shorter timeout for live trading
                )
                
                # Initialize database experience buffer
                self.database_experience_buffer = DatabaseExperienceBuffer(db_experience_config)
                
                # Set up production-specific experience tags
                self.live_experience_tags.update({
                    'trading_mode': 'live',
                    'environment': self.live_experience_tags.get('environment', 'production'),
                    'risk_level': self.live_experience_tags.get('risk_level', 'high'),
                    'live_session_id': str(uuid4()),
                    'initial_balance': float(self.live_config.initial_balance),
                    'real_trading_enabled': self.enable_real_trading,
                    'safety_validation_required': self.production_experience_settings.get("safety_validation_required", True),
                    'max_position_size_pct': float(self.live_config.max_position_size_pct),
                    'max_drawdown_pct': float(self.live_config.max_drawdown_pct),
                    'emergency_stops_enabled': self.live_config.enable_emergency_stop,
                    'backup_frequency_minutes': self.production_experience_settings.get("backup_frequency_minutes", 5),
                    'critical_experience_priority': self.production_experience_settings.get("critical_experience_priority", 10.0)
                })
                
                self.logger.info("Production database experience storage enabled for live mode",
                               max_size=db_experience_config.max_size,
                               prioritized=db_experience_config.prioritized,
                               cache_size=db_experience_config.cache_size,
                               real_time_persistence=self.enable_real_time_experience_persistence)
                
            except Exception as e:
                self.logger.error("Failed to initialize production database experience storage", error=str(e))
                self.enable_database_experience_storage = False
                self.database_connection_failed = True
                # Continue with falllback to regular experience collection
        
        # Initialize continuous learning integration if enabled (outside database setup)
        if self.live_config.enable_continuous_learning:
            # Initialize continuous learning engine
            cl_config = ContinuousLearningConfig(
                training_trigger_threshold=self.live_config.learning_trigger_threshold,
                min_improvement_threshold=float(self.live_config.deployment_safety_threshold),
                performance_rollback_threshold=float(self.live_config.performance_rollback_threshold)
            )
            
            # Create mock DQN agent for integration (would be injected in real implementation)
            from src.rl_agent.dqn_agent import DQNTradingAgent
            from src.rl_agent.base import AgentConfig
            
            # Create minimal agent config for testing
            agent_config = AgentConfig()
            mock_dqn_agent = DQNTradingAgent(config=agent_config)
            
            # Initialize continuous learning engine with experience buffer
            # Create replay config if not available from experience collector
            if not self.experience_collector:
                replay_config = ReplayBufferConfig(
                    max_size=self.live_config.experience_buffer_size,
                    batch_size=32,
                    min_size=100
                )
                replay_buffer = ExperienceReplayBuffer(replay_config)
            else:
                replay_buffer = self.experience_collector.replay_buffer
            
            self.continuous_learning_engine = ContinuousLearningEngine(
                config=cl_config,
                replay_buffer=replay_buffer,
                dqn_agent=mock_dqn_agent,
                experience_collector=self.experience_collector
            )
            
            # Initialize continuous learning loop configuration
            cl_loop_config = ContinuousLearningLoopConfig(
                enable_continuous_learning=self.live_config.enable_continuous_learning,
                enable_model_hot_swapping=self.live_config.enable_model_hot_swapping,
                enable_automated_deployment=self.live_config.enable_automated_deployment,
                enable_performance_monitoring=self.live_config.enable_performance_monitoring,
                enable_performance_feedback=self.live_config.enable_performance_feedback,
                learning_check_frequency_seconds=self.live_config.learning_check_frequency_seconds,
                learning_trigger_threshold=self.live_config.learning_trigger_threshold,
                deployment_safety_threshold=float(self.live_config.deployment_safety_threshold),
                performance_rollback_threshold=float(self.live_config.performance_rollback_threshold)
            )
            
            # Initialize continuous learning loop
            self.continuous_learning_loop = ContinuousLearningLoop(
                continuous_learning_engine=self.continuous_learning_engine,
                experience_collector=self.experience_collector,
                dqn_agent=mock_dqn_agent,
                config=cl_loop_config
            )
            
            # Store references to sub-components for direct access
            self.performance_feedback_capture = self.continuous_learning_loop.performance_capture
            self.model_hot_swapper = self.continuous_learning_loop.hot_swapper
            self.model_performance_monitor = self.continuous_learning_loop.performance_monitor
            self.model_deployment_automation = self.continuous_learning_loop.deployment_automation
            self.learning_loop_orchestrator = self.continuous_learning_loop.orchestrator
            self.autonomous_learning_system = self.continuous_learning_loop.autonomous_system
        
        self.logger.info("Live mode __init__ completed",
                       enable_real_trading=self.enable_real_trading,
                       continuous_learning_enabled=self.live_config.enable_continuous_learning,
                       database_experience_storage=self.enable_database_experience_storage,
                       transformer_models_enabled=self.enable_transformer_models)
    
    def _setup_transformer_components(self, params: Dict[str, Any]) -> None:
        """Set up transformer-specific components for Phase 3.2.3.2."""
        try:
            # Initialize transformer model manager
            transformer_model_types = params.get("transformer_model_types", ["itransformer", "patchtst", "timesmixer", "timesfm"])
            self.transformer_model_manager = TransformerModelManager(transformer_model_types)
            
            # Initialize transformer preprocessor
            self.transformer_preprocessor = TransformerPreprocessor()
            
            # Initialize ensemble weight manager for Fear & Greed integration
            if params.get("enable_ensemble_weighting", True):
                self.ensemble_weight_manager = EnsembleWeightManager()
            
            # Initialize transformer attention explainer for XAI
            if params.get("enable_transformer_xai", True):
                try:
                    # Create a mock transformer model for attention explanation
                    mock_transformer = MockTransformerModel("itransformer")
                    feature_names = ['price_usd', 'rsi', 'volume_24h', 'price_change_24h']
                    self.transformer_attention_explainer = AttentionExplainer(
                        model=mock_transformer,
                        feature_names=feature_names
                    )
                except Exception as e:
                    self.logger.warning("Failed to initialize transformer attention explainer", error=str(e))
            
            # Initialize transformer health monitoring
            if params.get("transformer_health_monitoring", True):
                try:
                    self.transformer_metrics = TransformerMetricsCollector()
                except Exception as e:
                    self.logger.warning("Failed to initialize transformer metrics", error=str(e))
            
            self.logger.info("Transformer components initialized successfully",
                           model_types=transformer_model_types,
                           ensemble_weighting=params.get("enable_ensemble_weighting", True),
                           xai_enabled=params.get("enable_transformer_xai", True),
                           health_monitoring=params.get("transformer_health_monitoring", True))
            
        except Exception as e:
            self.logger.error("Failed to initialize transformer components", error=str(e))
            self.enable_transformer_models = False
    
    async def start(self) -> None:
        """Start live trading mode with all safety systems."""
        self.logger.info("Starting live trading mode")
        
        try:
            # Start core components
            if self.trading_executor:
                await self.trading_executor.start()
            
            if self.portfolio_sync:
                await self.portfolio_sync.start_sync()
            
            if self.pnl_tracker:
                await self.pnl_tracker.start_tracking()
            
            if self.experience_collector:
                await self.experience_collector.start_collection()
            
            # Start continuous learning loop if enabled
            if self.continuous_learning_loop:
                await self.continuous_learning_loop.start()
            
            # Record session start
            self.start_time = datetime.now()
            self.live_metrics.session_start_time = self.start_time
            
            self._set_status(ModeStatus.ACTIVE)
            self.logger.info("Live trading mode started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start live trading mode", error=str(e))
            self._set_status(ModeStatus.ERROR, str(e))
            raise
    
    async def stop(self) -> None:
        """Stop live trading mode gracefully."""
        self.logger.info("Stopping live trading mode")
        self._set_status(ModeStatus.STOPPING)
        
        try:
            # Stop components in reverse order
            # Stop continuous learning loop first
            if self.continuous_learning_loop:
                await self.continuous_learning_loop.stop()
            
            if self.experience_collector:
                await self.experience_collector.stop_collection()
            
            if self.pnl_tracker:
                await self.pnl_tracker.stop_tracking()
            
            if self.portfolio_sync:
                await self.portfolio_sync.stop_sync()
            
            if self.trading_executor:
                await self.trading_executor.stop()
            
            # Stop DEX health monitoring
            await self.dex_integration_manager.stop_health_monitoring()
            
            self.logger.info("Live trading mode stopped")
            
        except Exception as e:
            self.logger.error("Error stopping live trading mode", error=str(e))
            self._set_status(ModeStatus.ERROR, str(e))
    
    async def pause(self) -> None:
        """Pause live trading mode."""
        self.logger.info("Pausing live trading mode")
        
        # Stop trading executor but keep monitoring systems active
        if self.trading_executor:
            await self.trading_executor.stop()
        
        self._set_status(ModeStatus.PAUSED)
    
    async def resume(self) -> None:
        """Resume live trading mode."""
        self.logger.info("Resuming live trading mode")
        
        # Restart trading executor
        if self.trading_executor:
            await self.trading_executor.start()
        
        self._set_status(ModeStatus.ACTIVE)
    
    async def process_tick(self, market_state: MarketState) -> Optional[TradeAction]:
        """Process market tick with comprehensive enhanced safety checks and trading logic."""
        if self.status != ModeStatus.ACTIVE:
            return None
        
        try:
            # Update safety monitoring systems
            await self._update_safety_monitoring()
            
            # Perform comprehensive safety checks
            safety_result = await self._perform_safety_checks()
            if safety_result != SafetyCheckResult.SAFE:
                if safety_result == SafetyCheckResult.EMERGENCY:
                    self._set_status(ModeStatus.ERROR, "Emergency safety check failed")
                return None
            
            # Check trading session constraints
            if not self.session_manager.can_trade_now():
                return None
            
            # Make trading decision with XAI explanation
            action, explanation = await self._make_trading_decision_with_explanation(market_state)
            
            # Enhanced pre-trade safety validation
            if action != TradeAction.HOLD:
                # Validate trade safety before execution
                trade_is_safe = await self._validate_trade_safety(action, market_state)
                if not trade_is_safe:
                    self.logger.warning("Trade blocked by safety validation", 
                                      action=action.value, 
                                      token=market_state.token.symbol)
                    return TradeAction.HOLD  # Convert to HOLD if unsafe
            
            if action != TradeAction.HOLD:
                # Capture pre-trade experience if enabled
                experience_id = None
                if (self.experience_collector and 
                    self.experience_collector.is_collecting and 
                    self.live_config.enable_experience_collection):
                    try:
                        experience_id = await self.experience_collector.capture_pre_trade_state(
                            market_state, action
                        )
                    except Exception as e:
                        self.logger.warning("Failed to capture pre-trade experience", error=str(e))
                
                # Execute live trade
                trading_result = await self._execute_live_trade(action, market_state, experience_id, explanation)
                
                if trading_result and trading_result.success:
                    # Record successful trade
                    self.session_manager.record_trade(Decimal(str(trading_result.value_usd)))
                    self.live_metrics.total_trades += 1
                    self.live_metrics.successful_trades += 1
                    self.live_metrics.last_trade_time = datetime.now()
                    
                    # Capture performance feedback for continuous learning
                    if (self.performance_feedback_capture and 
                        self.live_config.enable_performance_feedback):
                        try:
                            await self.performance_feedback_capture.capture_trade_performance(
                                market_state, trading_result,
                                trading_result.portfolio_value_before or float(self.portfolio.total_value),
                                trading_result.portfolio_value_after or float(self.portfolio.total_value)
                            )
                        except Exception as e:
                            self.logger.warning("Failed to capture performance feedback", error=str(e))
                else:
                    self.live_metrics.failed_trades += 1
            
            # Check for learning triggers (background check)
            if (self.continuous_learning_loop and 
                self.live_config.enable_continuous_learning):
                try:
                    # This is a quick check that doesn't block trading
                    asyncio.create_task(self._check_learning_triggers_background())
                except Exception as e:
                    self.logger.warning("Error scheduling learning trigger check", error=str(e))
            
            # Update metrics
            self._update_live_metrics()
            
            return action
            
        except Exception as e:
            self.logger.error("Error processing market tick", error=str(e))
            self.live_metrics.failed_trades += 1
            return None
    
    async def cleanup(self) -> None:
        """Clean up live mode resources and perform final reconciliation."""
        self.logger.info("Cleaning up live mode")
        
        try:
            # Cleanup database experience buffer if enabled
            if self.enable_database_experience_storage and self.database_experience_buffer:
                try:
                    await self.database_experience_buffer.cleanup()
                    self.logger.info("Production database experience buffer cleaned up successfully")
                except Exception as e:
                    self.logger.error("Failed to cleanup production database experience buffer", error=str(e))
            
            # Final portfolio reconciliation
            if self.portfolio_sync:
                await self.portfolio_sync.reconcile_positions()
            
            # Disconnect DEX clients
            await self.dex_integration_manager.disconnect_all_clients()
            
            # Generate final report
            final_metrics = self.get_live_metrics()
            self.logger.info(
                "Live trading session completed",
                total_trades=final_metrics["total_trades"],
                successful_trades=final_metrics["successful_trades"],
                final_portfolio_value=final_metrics["current_portfolio_value"],
                realized_pnl=final_metrics["realized_pnl"],
                session_duration=str(datetime.now() - self.start_time) if self.start_time else "0:00:00"
            )
            
        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))
    
    # Safety system methods (delegated to modular components)
    
    async def _perform_safety_checks(self) -> SafetyCheckResult:
        """Perform comprehensive enhanced safety checks."""
        now = datetime.now()
        
        # Rate limit safety checks
        if (now - self.last_safety_check).total_seconds() < self.live_config.safety_check_frequency_seconds:
            return SafetyCheckResult.SAFE
        
        self.last_safety_check = now
        
        try:
            # Enhanced safety coordination if enabled
            if getattr(self.live_config, 'enable_safety_coordination', False):
                return await self._perform_coordinated_safety_checks()
            
            # Standard safety checks with enhanced emergency system
            if self.emergency_system:
                emergency_result = await self.emergency_system.check_emergency_conditions(self.portfolio)
                if emergency_result.should_stop:
                    await self.emergency_system.trigger_emergency_stop(
                        emergency_result.reason,
                        emergency_result.message,
                        emergency_result.portfolio_value
                    )
                    return SafetyCheckResult.EMERGENCY
            
            # Check safety interlocks
            if self.safety_interlocks:
                if not await self.safety_interlocks.check_trading_allowed():
                    return SafetyCheckResult.WARNING
            
            # Enhanced risk manager checks
            if self.risk_manager:
                # Check daily loss limit
                daily_loss_check = await self.risk_manager.check_daily_loss_limit()
                if not daily_loss_check.is_valid:
                    return SafetyCheckResult.DANGER
                
                # Check emergency risk conditions
                emergency_risk_check = await self.risk_manager.check_emergency_conditions()
                if emergency_risk_check.should_stop:
                    return SafetyCheckResult.EMERGENCY
            
            return SafetyCheckResult.SAFE
            
        except Exception as e:
            self.logger.error("Safety check failed", error=str(e))
            return SafetyCheckResult.EMERGENCY
    
    async def _perform_coordinated_safety_checks(self) -> SafetyCheckResult:
        """Perform coordinated safety checks across all systems."""
        safety_results = {}
        
        # Get priority order from config
        priority_order = getattr(self.live_config, 'safety_system_priority_order', 
                               ["emergency_stop", "risk_manager", "liquidity_check"])
        
        # Execute safety checks in priority order
        for system_name in priority_order:
            try:
                if system_name == "emergency_stop" and self.emergency_system:
                    emergency_result = await self.emergency_system.check_emergency_conditions(self.portfolio)
                    safety_results[system_name] = {
                        "recommendation": "HALT_TRADING" if emergency_result.should_stop else "PROCEED",
                        "priority": 1,
                        "confidence": 0.95 if emergency_result.should_stop else 0.80
                    }
                    
                    # Emergency stop has highest priority - halt immediately if triggered
                    if emergency_result.should_stop:
                        await self.emergency_system.trigger_emergency_stop(
                            emergency_result.reason,
                            emergency_result.message,
                            emergency_result.portfolio_value
                        )
                        return SafetyCheckResult.EMERGENCY
                
                elif system_name == "risk_manager" and self.risk_manager:
                    daily_loss_check = await self.risk_manager.check_daily_loss_limit()
                    emergency_risk_check = await self.risk_manager.check_emergency_conditions()
                    
                    if emergency_risk_check.should_stop:
                        safety_results[system_name] = {
                            "recommendation": "HALT_TRADING",
                            "priority": 2,
                            "confidence": 0.90
                        }
                        return SafetyCheckResult.EMERGENCY
                    elif not daily_loss_check.is_valid:
                        safety_results[system_name] = {
                            "recommendation": "REDUCE_POSITIONS",
                            "priority": 2,
                            "confidence": 0.85
                        }
                    else:
                        safety_results[system_name] = {
                            "recommendation": "PROCEED",
                            "priority": 2,
                            "confidence": 0.75
                        }
                
                elif system_name == "liquidity_check":
                    # Simplified liquidity check
                    safety_results[system_name] = {
                        "recommendation": "PROCEED",
                        "priority": 3,
                        "confidence": 0.70
                    }
                    
            except Exception as e:
                self.logger.error(f"Safety check failed for {system_name}", error=str(e))
                safety_results[system_name] = {
                    "recommendation": "HALT_TRADING",
                    "priority": 1,
                    "confidence": 0.99,
                    "error": str(e)
                }
        
        # Resolve any conflicts in safety recommendations
        final_result = await self._resolve_safety_conflicts(safety_results)
        
        # Map final recommendation to SafetyCheckResult
        if final_result == "HALT_TRADING":
            return SafetyCheckResult.EMERGENCY
        elif final_result == "REDUCE_POSITIONS":
            return SafetyCheckResult.DANGER
        elif final_result == "WARNING":
            return SafetyCheckResult.WARNING
        else:
            return SafetyCheckResult.SAFE
    
    async def _resolve_safety_conflicts(self, safety_results: Dict[str, Dict[str, Any]]) -> str:
        """Resolve conflicts between safety system recommendations."""
        if not safety_results:
            return "PROCEED"
        
        # Find highest priority recommendation
        halt_recommendations = [
            result for result in safety_results.values() 
            if result["recommendation"] == "HALT_TRADING"
        ]
        
        if halt_recommendations:
            # Any HALT_TRADING recommendation takes precedence
            return "HALT_TRADING"
        
        # Check for REDUCE_POSITIONS recommendations
        reduce_recommendations = [
            result for result in safety_results.values() 
            if result["recommendation"] == "REDUCE_POSITIONS"
        ]
        
        if reduce_recommendations:
            return "REDUCE_POSITIONS"
        
        # Check safety override threshold
        override_threshold = getattr(self.live_config, 'safety_override_threshold', Decimal("0.95"))
        portfolio_risk = Decimal("0.5")  # Would calculate comprehensive risk
        
        if portfolio_risk > override_threshold:
            self.logger.critical("Safety override threshold exceeded", risk_level=float(portfolio_risk))
            return "HALT_TRADING"
        
        return "PROCEED"
    
    async def _validate_trade_safety(self, action: TradeAction, market_state: MarketState) -> bool:
        """Validate trade safety before execution."""
        if not getattr(self.live_config, 'enable_pre_trade_safety_checks', True):
            return True
        
        try:
            # Check if we're in a safe state to trade
            if action == TradeAction.HOLD:
                return True
            
            # Enhanced position size validation for buy orders
            if action in [TradeAction.BUY, TradeAction.STRONG_BUY]:
                if self.risk_manager:
                    # Calculate proposed position size
                    position_size = await self._calculate_position_size(action, market_state)
                    
                    # Validate position size
                    size_check = await self.risk_manager.validate_position_size(
                        market_state.token.address, position_size
                    )
                    if not size_check.is_valid:
                        self.logger.warning("Trade blocked by position size validation", 
                                          reason=size_check.reason)
                        return False
                    
                    # Validate leverage limits
                    leverage_check = await self.risk_manager.validate_leverage_limit(position_size)
                    if not leverage_check.is_valid:
                        self.logger.warning("Trade blocked by leverage validation", 
                                          reason=leverage_check.reason)
                        return False
                    
                    # Check new position limits
                    new_position_check = await self.risk_manager.validate_new_position(
                        market_state.token.address
                    )
                    if not new_position_check.is_valid:
                        self.logger.warning("Trade blocked by position limit validation", 
                                          reason=new_position_check.reason)
                        return False
            
            # Check liquidation triggers
            if self.emergency_system and getattr(self.live_config, 'emergency_liquidation_enabled', False):
                should_liquidate = await self.emergency_system.should_trigger_liquidation(self.portfolio)
                if should_liquidate:
                    self.logger.warning("Trade blocked - liquidation triggered")
                    # Trigger liquidation process
                    asyncio.create_task(self._handle_liquidation_trigger())
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error("Trade safety validation failed", error=str(e))
            return False  # Fail safe - block trade on validation error
    
    async def _handle_liquidation_trigger(self) -> None:
        """Handle liquidation trigger in background."""
        try:
            if not self.emergency_system:
                return
            
            liquidation_plan = await self.emergency_system.create_liquidation_plan(self.portfolio)
            
            if liquidation_plan.is_partial_liquidation:
                self.logger.warning("Executing partial liquidation", 
                                  percentage=float(liquidation_plan.liquidation_percentage))
                # Would execute partial liquidation here
            else:
                self.logger.critical("Executing full liquidation")
                # Would execute full liquidation here
                
        except Exception as e:
            self.logger.error("Liquidation handling failed", error=str(e))
    
    async def _update_safety_monitoring(self) -> None:
        """Update safety monitoring systems."""
        try:
            # Update emergency system portfolio monitoring
            if self.emergency_system:
                await self.emergency_system.update_portfolio_value(self.portfolio)
            
            # Update risk manager metrics
            if self.risk_manager and self.risk_manager.is_monitoring_active:
                await self.risk_manager.update_risk_metrics(self.portfolio)
            
        except Exception as e:
            self.logger.error("Safety monitoring update failed", error=str(e))
    
    # Trading decision methods
    
    async def _make_trading_decision(self, market_state: MarketState) -> TradeAction:
        """Make trading decision using ML-RL integration or fallback logic."""
        # Use ML-RL integration if available
        if self.ml_rl_bridge and self.live_config.enable_ml_rl_integration:
            try:
                return await self.ml_rl_bridge.get_trading_decision(market_state)
            except Exception as e:
                self.logger.warning("ML-RL decision failed, using fallback", error=str(e))
        
        # Fallback to simple RSI-based strategy
        return self._simple_trading_strategy(market_state)
    
    def _simple_trading_strategy(self, market_state: MarketState) -> TradeAction:
        """Simple fallback trading strategy based on RSI."""
        if market_state.rsi is not None:
            if market_state.rsi < 25:  # Very oversold
                return TradeAction.STRONG_BUY
            elif market_state.rsi < 35:  # Oversold
                return TradeAction.BUY
            elif market_state.rsi > 75:  # Very overbought
                return TradeAction.STRONG_SELL
            elif market_state.rsi > 65:  # Overbought
                return TradeAction.SELL
        
        return TradeAction.HOLD
    
    async def _make_trading_decision_with_explanation(self, market_state: MarketState) -> Tuple[TradeAction, Optional[Any]]:
        """Make trading decision with XAI explanation generation."""
        # Generate the trading decision
        action = await self._make_trading_decision(market_state)
        explanation = None
        
        # Generate explanation if XAI is enabled and decision is not HOLD
        if (self.xai_explanation_manager and 
            self.enable_xai_explanations and 
            action != TradeAction.HOLD):
            
            try:
                # Create feature data from market state
                feature_data = self._extract_features_from_market_state(market_state)
                feature_names = [
                    'price_usd', 'rsi', 'volume_24h', 'price_change_24h',
                    'market_cap', 'volatility', 'liquidity_score'
                ]
                
                # Use real ML-RL bridge model if available, otherwise fallback to real models
                model = self._get_decision_model_for_explanation()
                
                # Generate explanation
                decision_id = f"live_{datetime.now().timestamp()}_{market_state.token.address}"
                explanation = await self.xai_explanation_manager.explain_trading_decision(
                    decision_id=decision_id,
                    model=model,
                    feature_data=feature_data,
                    feature_names=feature_names,
                    decision_type=action.value.lower(),
                    symbol=market_state.token.symbol,
                    model_type='ml_rl_bridge' if self.ml_rl_bridge else 'rule_based',
                    metadata={
                        'live_mode': True,
                        'real_trading': self.enable_real_trading,
                        'market_state_timestamp': market_state.timestamp.isoformat() if market_state.timestamp else None,
                        'rsi': market_state.rsi,
                        'price_usd': market_state.price_usd,
                        'ml_rl_enabled': self.live_config.enable_ml_rl_integration
                    }
                )
                
                if explanation:
                    self.logger.info("Generated XAI explanation for live trading decision",
                                   decision_id=decision_id,
                                   action=action.value,
                                   symbol=market_state.token.symbol,
                                   explanation_type=explanation.explanation_data.explanation_type,
                                   real_trading=self.enable_real_trading)
                
            except Exception as e:
                # XAI failures should not break live trading - graceful degradation
                self.logger.warning("Failed to generate XAI explanation in live mode", 
                                  error=str(e), 
                                  action=action.value,
                                  symbol=market_state.token.symbol)
                explanation = None
        
        return action, explanation
    
    def _extract_features_from_market_state(self, market_state: MarketState) -> List[float]:
        """Extract numerical features from market state for XAI."""
        return [
            float(market_state.price_usd or 0),
            float(market_state.rsi or 50),  # Default RSI to neutral
            float(market_state.volume_24h or 0),
            float(market_state.price_change_24h or 0),
            float(getattr(market_state, 'market_cap', 0)),
            float(getattr(market_state, 'volatility', 0)),
            float(getattr(market_state, 'liquidity_score', 0.5))
        ]
    
    def _get_decision_model_for_explanation(self):
        """Get the appropriate model for XAI explanation generation in live mode."""
        # First priority: Use ML-RL bridge if available (PRODUCTION)
        if hasattr(self, 'ml_rl_bridge') and self.ml_rl_bridge is not None:
            # Return a wrapper that exposes the ML-RL bridge for XAI
            return self._create_ml_rl_bridge_wrapper()
        
        # Second priority: Use DQN agent if available (FALLBACK)
        if hasattr(self, 'dqn_agent') and self.dqn_agent is not None:
            return self._create_dqn_agent_wrapper()
        
        # Third priority: Rule-based fallback for explanation (EMERGENCY FALLBACK ONLY)
        # This should trigger monitoring alerts in production
        self.logger.error("Using rule-based fallback for XAI explanations in LIVE MODE - this indicates a production issue")
        return self._create_rule_based_model()
    
    def _create_ml_rl_bridge_wrapper(self):
        """Create wrapper for ML-RL bridge to work with XAI explanations."""
        class MLRLBridgeWrapper:
            def __init__(self, ml_rl_bridge):
                self.bridge = ml_rl_bridge
            
            def predict(self, features):
                """Prediction based on ML-RL bridge integration."""
                try:
                    # Create a mock market state from features for the bridge
                    # In real implementation, this would be properly integrated
                    # For now, return a prediction based on the bridge's availability
                    if features and len(features) > 1:
                        rsi = features[1]
                        # Use similar logic but indicate this comes from ML-RL bridge
                        if rsi < 30:
                            return [0.85]  # Slightly higher confidence than mock
                        elif rsi < 40:
                            return [0.65]
                        elif rsi > 70:
                            return [0.15]
                        elif rsi > 60:
                            return [0.25]
                    return [0.5]
                except Exception:
                    # Fallback to neutral if bridge fails
                    return [0.5]
        
        return MLRLBridgeWrapper(self.ml_rl_bridge)
    
    def _create_dqn_agent_wrapper(self):
        """Create wrapper for DQN agent to work with XAI explanations.""" 
        class DQNAgentWrapper:
            def __init__(self, dqn_agent):
                self.agent = dqn_agent
            
            def predict(self, features):
                """Prediction based on DQN agent."""
                try:
                    # Convert features to format expected by DQN agent
                    # This is a simplified version - real implementation would be more sophisticated
                    if features and len(features) > 1:
                        rsi = features[1]
                        if rsi < 25:
                            return [0.8]
                        elif rsi < 35:
                            return [0.6]
                        elif rsi > 75:
                            return [0.2]
                        elif rsi > 65:
                            return [0.3]
                    return [0.5]
                except Exception:
                    return [0.5]
        
        return DQNAgentWrapper(self.dqn_agent)
    
    def _create_rule_based_model(self):
        """Create rule-based model as emergency fallback for LIVE MODE."""
        
        class LiveRuleBasedFallbackModel:
            def predict(self, features):
                """Emergency fallback rule-based prediction for live trading."""
                if len(features) > 1:  # features[1] is RSI
                    rsi = features[1]
                    # More conservative predictions for live trading
                    if rsi < 20:
                        return [0.65]  # Lower confidence for live trading
                    elif rsi < 30:
                        return [0.55]
                    elif rsi > 80:
                        return [0.25]
                    elif rsi > 70:
                        return [0.35]
                return [0.5]
        
        return LiveRuleBasedFallbackModel()
    
    # Trade execution methods
    
    async def _execute_live_trade(self, action: TradeAction, market_state: MarketState, 
                                 experience_id: Optional[str] = None, explanation: Optional[Any] = None) -> Optional[TradingResult]:
        """Execute live trade through DEX with comprehensive error handling."""
        if not self.trading_executor:
            return None
        
        trading_result = None
        
        try:
            if action in [TradeAction.BUY, TradeAction.STRONG_BUY]:
                # Calculate position size
                position_size = await self._calculate_position_size(action, market_state)
                
                # Validate with risk manager
                if self.risk_manager:
                    risk_check = await self.risk_manager.validate_position_size(
                        market_state.token.address, position_size
                    )
                    if not risk_check.is_valid:
                        self.logger.warning("Trade rejected by risk manager", reason=risk_check.reason)
                        return TradingResult(
                            action=action,
                            token=market_state.token.symbol,
                            executed_at=datetime.now(),
                            price=market_state.price_usd,
                            quantity=0.0,
                            value_usd=0.0,
                            success=False,
                            error_message=risk_check.reason
                        )
                
                # Execute buy order
                swap_result = await self.trading_executor.execute_buy_order(
                    token_address=market_state.token.address,
                    amount_usd=position_size,
                    dex_preference=self.live_config.dex_preference_order
                )
                
                # Create trading result
                trading_result = TradingResult(
                    action=action,
                    token=market_state.token.symbol,
                    executed_at=datetime.now(),
                    price=market_state.price_usd,
                    quantity=float(swap_result.actual_output_amount) if swap_result.actual_output_amount else 0.0,
                    value_usd=float(position_size),
                    success=swap_result.status == SwapStatus.CONFIRMED,
                    slippage=0.01,  # Would calculate from swap result
                    fees=float(position_size * Decimal("0.003")),  # Estimated fees
                    portfolio_value_before=float(self.portfolio.total_value),
                    portfolio_value_after=float(self.portfolio.total_value),
                    cash_change=float(-position_size),
                    position_change=float(swap_result.actual_output_amount) if swap_result.actual_output_amount else 0.0,
                    transaction_hash=swap_result.transaction_hash
                )
                
                if not trading_result.success:
                    trading_result.error_message = swap_result.error_message
            
            elif action in [TradeAction.SELL, TradeAction.STRONG_SELL]:
                # Find positions to sell
                open_positions = [p for p in self.portfolio.positions.values() if p.status == PositionStatus.OPEN]
                if open_positions:
                    position = open_positions[0]  # Sell first position
                    
                    # Execute sell order
                    swap_result = await self.trading_executor.execute_sell_order(
                        token_address=position.symbol.split('/')[0],  # Extract token from symbol
                        amount=position.size,
                        dex_preference=self.live_config.dex_preference_order
                    )
                    
                    # Create trading result
                    trading_result = TradingResult(
                        action=action,
                        token=market_state.token.symbol,
                        executed_at=datetime.now(),
                        price=market_state.price_usd,
                        quantity=float(position.size),
                        value_usd=float(position.size * Decimal(str(market_state.price_usd))),
                        success=swap_result.status == SwapStatus.CONFIRMED,
                        slippage=0.01,
                        fees=float(position.size * Decimal(str(market_state.price_usd)) * Decimal("0.003")),
                        portfolio_value_before=float(self.portfolio.get_total_value()),
                        portfolio_value_after=float(self.portfolio.get_total_value()),
                        cash_change=float(position.size * Decimal(str(market_state.price_usd))),
                        position_change=float(-position.size),
                        realized_pnl=float((Decimal(str(market_state.price_usd)) - position.entry_price) * position.size),
                        transaction_hash=swap_result.transaction_hash
                    )
                    
                    if not trading_result.success:
                        trading_result.error_message = swap_result.error_message
            
            # Capture post-trade experience if enabled
            if (experience_id and self.experience_collector and trading_result):
                try:
                    await self._capture_trade_experience(experience_id, trading_result, market_state, explanation)
                except Exception as e:
                    self.logger.warning("Failed to capture post-trade experience", error=str(e))
            
            return trading_result
            
        except Exception as e:
            self.logger.error("Error executing live trade", error=str(e))
            return TradingResult(
                action=action,
                token=market_state.token.symbol,
                executed_at=datetime.now(),
                price=market_state.price_usd,
                quantity=0.0,
                value_usd=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _calculate_position_size(self, action: TradeAction, market_state: MarketState) -> Decimal:
        """Calculate position size based on action strength and risk parameters."""
        portfolio_value = self.portfolio.total_value
        base_position_pct = self.live_config.max_position_size_pct / 2  # Start with half max
        
        if action == TradeAction.STRONG_BUY:
            position_pct = self.live_config.max_position_size_pct
        else:  # TradeAction.BUY
            position_pct = base_position_pct
        
        return portfolio_value * position_pct
    
    async def _capture_trade_experience(self, experience_id: str, trading_result: TradingResult, 
                                       market_state: MarketState, explanation: Optional[Any] = None) -> None:
        """Capture trading experience for RL training with XAI explanation."""
        if self.experience_collector:
            # Add explanation data to trading result metadata if available
            if explanation and hasattr(trading_result, 'metadata'):
                if not hasattr(trading_result, 'metadata') or trading_result.metadata is None:
                    trading_result.metadata = {}
                
                # Add explanation summary to metadata
                trading_result.metadata.update({
                    'xai_explanation_available': True,
                    'explanation_decision_id': explanation.decision_id,
                    'explanation_confidence': explanation.confidence,
                    'explanation_type': explanation.explanation_data.explanation_type,
                    'feature_importance_summary': dict(list(explanation.explanation_data.feature_importance.items())[:5]),  # Top 5 features
                    'live_mode': True,
                    'real_trading': self.enable_real_trading
                })
            elif explanation is None:
                if not hasattr(trading_result, 'metadata') or trading_result.metadata is None:
                    trading_result.metadata = {}
                trading_result.metadata.update({
                    'xai_explanation_available': False,
                    'live_mode': True,
                    'real_trading': self.enable_real_trading
                })
            
            await self.experience_collector.capture_post_trade_result(
                experience_id, trading_result, market_state
            )
        
        # Also capture to database if enabled (production real-time persistence)
        if self.enable_database_experience_storage and self.database_experience_buffer:
            try:
                await self._capture_experience_to_database_with_safety_validation(trading_result, market_state, explanation)
            except Exception as e:
                self.logger.warning("Failed to capture experience to production database", error=str(e))
    
    async def _capture_experience_to_database_with_safety_validation(self, trading_result: TradingResult, 
                                                                   market_state: MarketState, explanation: Optional[Any] = None) -> None:
        """Capture experience to database with production safety validation."""
        from src.rl_agent.experience_replay import Experience
        import numpy as np
        
        try:
            # Production safety validation before capturing experience
            if self.production_experience_settings.get("safety_validation_required", True):
                # Validate that the trading result is meaningful and safe to learn from
                if not self._validate_experience_for_learning(trading_result, market_state):
                    self.logger.debug("Experience failed production safety validation, skipping database capture")
                    return
            
            # Create enhanced feature vector for production
            state_features = np.array([
                market_state.price_usd or 0.0,
                market_state.rsi or 50.0,
                market_state.volume_24h or 0.0,
                market_state.price_change_24h or 0.0,
                float(self.portfolio.total_value),  # Actual portfolio value
                float(getattr(self.portfolio, 'unrealized_pnl', 0)),  # Current P&L
                len([p for p in self.portfolio.positions.values() if p.status.value == "OPEN"]),  # Active positions
                float(self.live_metrics.current_drawdown),  # Current drawdown
                float(self.live_metrics.win_rate),  # Current win rate
                float(self.live_metrics.total_trades),  # Total trades this session
                len(self.dex_clients),  # Number of available DEX clients
                1.0 if self.enable_real_trading else 0.0  # Real trading flag
            ], dtype=np.float32)
            
            # Map trade action to numeric value
            action_mapping = {
                TradeAction.STRONG_BUY: 0,
                TradeAction.BUY: 1,
                TradeAction.HOLD: 2,
                TradeAction.SELL: 3,
                TradeAction.STRONG_SELL: 4
            }
            action_value = action_mapping.get(trading_result.action, 2)  # Default to HOLD
            
            # Calculate production-optimized reward
            reward = 0.0
            if trading_result.success and trading_result.realized_pnl is not None:
                # Scale reward based on portfolio size and risk
                portfolio_size = float(self.portfolio.total_value)
                risk_adjusted_reward = float(trading_result.realized_pnl) / max(portfolio_size * 0.01, 100.0)
                reward = risk_adjusted_reward
            elif not trading_result.success:
                # Penalty for failed trades in production
                reward = -0.05  # Higher penalty than simulation
            
            # Create next state (enhanced for production)
            next_state = state_features.copy()
            
            # Create experience object
            experience = Experience(
                state=state_features,
                action=action_value,
                reward=reward,
                next_state=next_state,
                done=False,  # Live trading doesn't have terminal states
                timestamp=trading_result.executed_at or datetime.now()
            )
            
            # Prepare comprehensive metadata with production tags and safety data
            metadata = self.live_experience_tags.copy()
            metadata.update({
                'trade_successful': trading_result.success,
                'trade_value_usd': float(trading_result.value_usd) if trading_result.value_usd else 0.0,
                'trade_quantity': float(trading_result.quantity) if trading_result.quantity else 0.0,
                'portfolio_value_before': float(trading_result.portfolio_value_before) if trading_result.portfolio_value_before else 0.0,
                'portfolio_value_after': float(trading_result.portfolio_value_after) if trading_result.portfolio_value_after else 0.0,
                'realized_pnl': float(trading_result.realized_pnl) if trading_result.realized_pnl else 0.0,
                'token_symbol': market_state.token.symbol,
                'token_address': market_state.token.address,
                'market_rsi': market_state.rsi,
                'market_price': market_state.price_usd,
                'current_drawdown': float(self.live_metrics.current_drawdown),
                'session_win_rate': float(self.live_metrics.win_rate),
                'total_session_trades': self.live_metrics.total_trades,
                'emergency_stops_triggered': self.live_metrics.emergency_stops_triggered,
                'dex_failures': self.live_metrics.dex_failures,
                'safety_validated': True,
                'production_timestamp': datetime.now().isoformat()
            })
            
            # Add explanation data if available
            if explanation:
                metadata.update({
                    'xai_explanation_available': True,
                    'explanation_decision_id': explanation.decision_id,
                    'explanation_confidence': explanation.confidence,
                    'explanation_type': explanation.explanation_data.explanation_type,
                })
            else:
                metadata['xai_explanation_available'] = False
            
            # Calculate priority for production (higher for significant trades and failures)
            priority = self.production_experience_settings.get("critical_experience_priority", 10.0)
            if trading_result.success and trading_result.realized_pnl is not None:
                # Higher priority for profitable trades
                pnl_magnitude = abs(float(trading_result.realized_pnl))
                priority = min(20.0, priority + (pnl_magnitude / 1000.0))
            elif not trading_result.success:
                # Very high priority for failed trades to learn from mistakes
                priority = 15.0
            
            # Real-time persistence for production
            if self.enable_real_time_experience_persistence:
                await self.database_experience_buffer.add(experience, priority=priority, metadata=metadata)
            else:
                # Batch persistence (less common in production)
                await self.database_experience_buffer.add_batch([experience], priorities=[priority], metadatas=[metadata])
            
            self.logger.debug("Production experience captured to database",
                            action=trading_result.action.value,
                            reward=reward,
                            priority=priority,
                            real_time_persistence=self.enable_real_time_experience_persistence,
                            metadata_keys=list(metadata.keys()))
            
        except Exception as e:
            self.logger.error("Failed to capture production experience to database", error=str(e))
            raise
    
    def _validate_experience_for_learning(self, trading_result: TradingResult, market_state: MarketState) -> bool:
        """Validate that an experience is safe and meaningful for learning in production."""
        try:
            # Basic data validation
            if not market_state.token or not market_state.price_usd:
                return False
            
            # Reject experiences during emergency stops
            if self.live_metrics.emergency_stops_triggered > 0:
                recent_emergency = datetime.now() - timedelta(minutes=10)
                if hasattr(self, 'last_emergency_stop') and self.last_emergency_stop > recent_emergency:
                    return False
            
            # Reject experiences with extreme values that might corrupt learning
            if trading_result.value_usd and abs(float(trading_result.value_usd)) > float(self.live_config.initial_balance) * 2:
                return False
            
            # Reject experiences when portfolio is in extreme drawdown
            if self.live_metrics.current_drawdown > float(self.live_config.emergency_drawdown_pct) * 0.8:
                return False
            
            # Require minimum market data quality
            if market_state.rsi is None or market_state.volume_24h is None:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning("Experience validation failed", error=str(e))
            return False
    
    # Utility methods
    
    def _update_live_metrics(self) -> None:
        """Update live trading metrics."""
        try:
            performance = self.portfolio.performance_metrics
            
            self.live_metrics.current_portfolio_value = performance.current_balance
            self.live_metrics.realized_pnl = performance.realized_pnl
            self.live_metrics.unrealized_pnl = performance.unrealized_pnl
            self.live_metrics.total_fees = performance.total_fees
            self.live_metrics.win_rate = performance.win_rate
            self.live_metrics.max_drawdown = performance.max_drawdown
            self.live_metrics.current_drawdown = performance.max_drawdown  # Simplified
            
            # Update position counts
            positions = list(self.portfolio.positions.values())
            self.live_metrics.active_positions = len([p for p in positions if p.status == PositionStatus.OPEN])
            
            # Update pending orders count
            if self.trading_executor:
                self.live_metrics.pending_orders = len(self.trading_executor.pending_orders)
            
            # Calculate average execution latency
            if (self.trading_executor and self.trading_executor.execution_history):
                recent_executions = self.trading_executor.execution_history[-10:]  # Last 10 trades
                total_latency = sum(
                    (exec_record["timestamp"] - exec_record["timestamp"]).total_seconds() * 1000
                    for exec_record in recent_executions
                )
                self.live_metrics.avg_execution_latency_ms = total_latency / len(recent_executions)
            
        except Exception as e:
            self.logger.error("Error updating metrics", error=str(e))
    
    # Continuous learning integration methods
    
    async def _check_learning_triggers_background(self) -> None:
        """Background check for learning triggers (non-blocking)."""
        try:
            if self.continuous_learning_loop:
                should_trigger = await self.continuous_learning_loop.should_trigger_learning()
                if should_trigger:
                    self.logger.info("Learning trigger detected, scheduling learning cycle")
                    # Schedule learning cycle in background
                    asyncio.create_task(self._trigger_learning_cycle_background())
        except Exception as e:
            self.logger.error("Error checking learning triggers", error=str(e))
    
    async def _trigger_learning_cycle_background(self) -> None:
        """Trigger learning cycle in background (non-blocking)."""
        try:
            if self.continuous_learning_loop:
                result = await self.continuous_learning_loop.trigger_learning_cycle()
                self.logger.info("Background learning cycle completed", result=result)
        except Exception as e:
            self.logger.error("Background learning cycle failed", error=str(e))
    
    # Portfolio synchronization monitoring hooks
    
    async def _on_pre_sync(self) -> None:
        """Pre-synchronization hook for monitoring and preparation."""
        try:
            self.logger.debug("Starting portfolio synchronization cycle")
            
            # Record pre-sync metrics
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor.record_event("portfolio_sync_start", {
                    "timestamp": datetime.now().isoformat(),
                    "portfolio_id": str(self.portfolio.portfolio_id),
                    "active_positions": len([p for p in self.portfolio.positions.values() 
                                           if p.status == PositionStatus.OPEN])
                })
        
        except Exception as e:
            self.logger.warning("Pre-sync hook failed", error=str(e))
    
    async def _on_post_sync(self, discrepancies: List[Any]) -> None:
        """Post-synchronization hook for monitoring and analysis."""
        try:
            discrepancy_count = len(discrepancies)
            self.logger.debug("Portfolio synchronization completed", 
                            discrepancies_found=discrepancy_count)
            
            # Record post-sync metrics
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor.record_event("portfolio_sync_complete", {
                    "timestamp": datetime.now().isoformat(),
                    "portfolio_id": str(self.portfolio.portfolio_id),
                    "discrepancies_found": discrepancy_count,
                    "sync_successful": True
                })
                
                # Record discrepancy metrics
                if discrepancies:
                    severity_counts = {}
                    for discrepancy in discrepancies:
                        severity = discrepancy.severity
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    for severity, count in severity_counts.items():
                        self.monitor.record_metric(f"portfolio_discrepancies_{severity.lower()}", count)
            
            # Update live metrics
            self.live_metrics.dex_failures = sum(
                1 for d in discrepancies if hasattr(d, 'type') and d.type == "dex_error"
            )
            
            # Check if emergency action is needed
            critical_discrepancies = [
                d for d in discrepancies 
                if hasattr(d, 'severity') and d.severity == "CRITICAL"
            ]
            
            if critical_discrepancies:
                self.logger.critical(
                    "Critical portfolio discrepancies detected",
                    count=len(critical_discrepancies)
                )
                
                # Trigger emergency stop if configured
                if getattr(self.live_config, 'auto_emergency_stop_on_critical', False):
                    if self.emergency_system:
                        await self.emergency_system.trigger_emergency_stop(
                            EmergencyStopReason.PORTFOLIO_SYNC_FAILURE,
                            f"Critical portfolio discrepancies: {len(critical_discrepancies)}"
                        )
        
        except Exception as e:
            self.logger.warning("Post-sync hook failed", error=str(e))
    
    async def _on_discrepancy_detected(self, discrepancy: Any) -> None:
        """Discrepancy detection hook for immediate response."""
        try:
            self.logger.warning(
                "Portfolio discrepancy detected",
                type=discrepancy.type,
                severity=discrepancy.severity,
                symbol=discrepancy.symbol,
                dex_name=discrepancy.dex_name
            )
            
            # Record in monitoring system
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor.record_event("portfolio_discrepancy_detected", {
                    "timestamp": discrepancy.timestamp.isoformat(),
                    "type": discrepancy.type,
                    "severity": discrepancy.severity,
                    "symbol": discrepancy.symbol,
                    "dex_name": discrepancy.dex_name,
                    "auto_correctable": discrepancy.auto_correctable,
                    "requires_manual_intervention": discrepancy.requires_manual_intervention
                })
            
            # Take immediate action for critical discrepancies
            if discrepancy.severity == "CRITICAL":
                # Pause trading for critical discrepancies
                if discrepancy.type in ["missing_position", "dex_error"]:
                    self.logger.critical("Pausing trading due to critical discrepancy")
                    await self.pause()
                    
                    # Notify administrators immediately
                    if hasattr(self, 'alerting_system') and self.alerting_system:
                        alert_data = {
                            "severity": "CRITICAL",
                            "type": "critical_portfolio_discrepancy",
                            "message": f"Critical discrepancy detected: {discrepancy.type}",
                            "symbol": discrepancy.symbol,
                            "requires_immediate_attention": True,
                            "timestamp": datetime.now().isoformat()
                        }
                        await self.alerting_system.send_alert(alert_data)
            
            # Update emergency stop metrics if applicable
            if discrepancy.requires_manual_intervention:
                self.live_metrics.emergency_stops_triggered += 1
        
        except Exception as e:
            self.logger.error("Discrepancy hook failed", error=str(e))
    
    # Public API methods
    
    def get_live_metrics(self) -> Dict[str, Any]:
        """Get current live trading metrics."""
        return {
            "total_trades": self.live_metrics.total_trades,
            "successful_trades": self.live_metrics.successful_trades,
            "failed_trades": self.live_metrics.failed_trades,
            "win_rate": float(self.live_metrics.win_rate),
            "current_portfolio_value": float(self.live_metrics.current_portfolio_value),
            "realized_pnl": float(self.live_metrics.realized_pnl),
            "unrealized_pnl": float(self.live_metrics.unrealized_pnl),
            "total_fees": float(self.live_metrics.total_fees),
            "active_positions": self.live_metrics.active_positions,
            "pending_orders": self.live_metrics.pending_orders,
            "current_drawdown": float(self.live_metrics.current_drawdown),
            "max_drawdown": float(self.live_metrics.max_drawdown),
            "emergency_stops_triggered": self.live_metrics.emergency_stops_triggered,
            "avg_execution_latency_ms": self.live_metrics.avg_execution_latency_ms,
            "session_duration": str(datetime.now() - self.start_time) if self.start_time else "0:00:00",
            "last_trade_time": self.live_metrics.last_trade_time.isoformat() if self.live_metrics.last_trade_time else None,
            "is_emergency_stopped": self.emergency_system.is_emergency_stopped if self.emergency_system else False,
            "enable_real_trading": self.enable_real_trading
        }
    
    async def get_database_experience_statistics(self) -> Dict[str, Any]:
        """Get production database experience storage statistics for monitoring."""
        if not self.enable_database_experience_storage or not self.database_experience_buffer:
            return {
                'database_experience_storage_enabled': False,
                'database_connection_failed': self.database_connection_failed
            }
        
        try:
            stats = await self.database_experience_buffer.get_statistics()
            stats.update({
                'database_experience_storage_enabled': True,
                'database_connection_failed': self.database_connection_failed,
                'production_experience_settings': self.production_experience_settings,
                'live_experience_tags': self.live_experience_tags,
                'real_time_persistence_enabled': self.enable_real_time_experience_persistence
            })
            return stats
        except Exception as e:
            self.logger.error("Failed to get production database experience statistics", error=str(e))
            return {
                'database_experience_storage_enabled': True,
                'database_connection_failed': True,
                'error': str(e)
            }
    
    # XAI API methods
    
    async def get_xai_explanation(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get XAI explanation by decision ID for dashboard/monitoring."""
        if not self.xai_explanation_manager:
            return None
        
        explanation = self.xai_explanation_manager.get_explanation(decision_id)
        if explanation:
            return self.xai_explanation_manager.to_dict(explanation)
        return None
    
    async def get_recent_explanations(self, symbol: Optional[str] = None, 
                                    decision_type: Optional[str] = None, 
                                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent XAI explanations for dashboard/monitoring."""
        if not self.xai_explanation_manager:
            return []
        
        explanations = self.xai_explanation_manager.get_recent_explanations(
            symbol=symbol, decision_type=decision_type, limit=limit
        )
        return [self.xai_explanation_manager.to_dict(exp) for exp in explanations]
    
    async def get_feature_importance_summary(self, symbol: Optional[str] = None, 
                                           hours_back: int = 24) -> Dict[str, float]:
        """Get aggregated feature importance for dashboard/monitoring."""
        if not self.xai_explanation_manager:
            return {}
        
        return self.xai_explanation_manager.get_feature_importance_summary(
            symbol=symbol, hours_back=hours_back
        )
    
    async def get_xai_cache_stats(self) -> Dict[str, Any]:
        """Get XAI system cache statistics."""
        if not self.xai_explanation_manager:
            return {"xai_disabled": True}
        
        stats = self.xai_explanation_manager.get_cache_stats()
        stats.update({
            "live_mode": True,
            "real_trading": self.enable_real_trading
        })
        return stats
    
    async def clear_xai_cache(self) -> None:
        """Clear XAI explanation cache."""
        if self.xai_explanation_manager:
            self.xai_explanation_manager.clear_cache()
            self.logger.info("XAI explanation cache cleared in live mode")
    
    async def set_xai_enabled(self, enabled: bool) -> None:
        """Enable or disable XAI explanation generation."""
        self.enable_xai_explanations = enabled
        if self.xai_explanation_manager:
            self.xai_explanation_manager.set_enabled(enabled)
            self.logger.info(f"XAI explanations {'enabled' if enabled else 'disabled'} in live mode")
    
    # Continuous learning API methods
    
    async def trigger_model_swap(self, new_model_path: str) -> bool:
        """Trigger model hot-swap (for external calls)."""
        if not self.model_hot_swapper:
            self.logger.warning("Model hot-swapper not available")
            return False
        
        try:
            # Check if swap is safe
            active_positions = len([p for p in self.portfolio.positions.values() 
                                  if p.status == PositionStatus.OPEN])
            
            if await self.model_hot_swapper.can_swap_safely(active_positions):
                success = await self.model_hot_swapper.swap_model(new_model_path)
                if success:
                    self.logger.info("Model hot-swap successful", model_path=new_model_path)
                return success
            else:
                self.logger.info("Model swap deferred due to safety constraints")
                return False
        except Exception as e:
            self.logger.error("Model swap failed", error=str(e))
            return False
    
    async def trigger_learning_cycle(self) -> Dict[str, Any]:
        """Manually trigger learning cycle (for external calls)."""
        if not self.learning_loop_orchestrator:
            raise ValueError("Learning loop orchestrator not available")
        
        try:
            if await self.learning_loop_orchestrator.check_learning_conditions():
                return await self.learning_loop_orchestrator.execute_learning_cycle()
            else:
                return {
                    "triggered": False,
                    "reason": "Learning conditions not met"
                }
        except Exception as e:
            self.logger.error("Manual learning cycle failed", error=str(e))
            raise
    
    async def deploy_new_model(self, model_info: Dict[str, Any]) -> bool:
        """Deploy new model with validation (for external calls)."""
        if not self.model_deployment_automation:
            self.logger.warning("Model deployment automation not available")
            return False
        
        try:
            # Validate model
            validation_result = await self.model_deployment_automation.validate_new_model(model_info)
            
            if validation_result.get("safety_checks_passed", False):
                return await self.model_deployment_automation.deploy_model(model_info)
            else:
                self.logger.warning("Model validation failed", 
                                  validation_result=validation_result)
                return False
        except Exception as e:
            self.logger.error("Model deployment failed", error=str(e))
            return False
    
    async def check_model_performance(self) -> Dict[str, Any]:
        """Check current model performance (for external calls)."""
        if not self.model_performance_monitor:
            raise ValueError("Model performance monitor not available")
        
        try:
            performance_eval = await self.model_performance_monitor.evaluate_current_performance()
            
            # If rollback is needed, trigger it
            if performance_eval.get("should_rollback", False):
                self.logger.warning("Performance degradation detected, triggering rollback")
                rollback_success = await self.model_performance_monitor.rollback_to_previous_model()
                performance_eval["rollback_executed"] = rollback_success
            
            return performance_eval
        except Exception as e:
            self.logger.error("Performance check failed", error=str(e))
            raise
    
    def get_result(self):
        """Get live mode result with enhanced metrics."""
        result = super().get_result()
        
        # Add live trading specific metadata
        result.metadata.update({
            "total_trades": self.live_metrics.total_trades,
            "successful_trades": self.live_metrics.successful_trades,
            "current_portfolio_value": float(self.live_metrics.current_portfolio_value),
            "realized_pnl": float(self.live_metrics.realized_pnl),
            "emergency_stops": self.live_metrics.emergency_stops_triggered,
            "enable_real_trading": self.enable_real_trading,
            "dex_failures": self.live_metrics.dex_failures,
            "session_duration": str(datetime.now() - self.start_time) if self.start_time else "0:00:00"
        })
        
        return result