"""
Test Different Activity Types and Categories
Following TDD methodology - comprehensive testing of all activity categories
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.activity_logging.activity_logger import (
    ActivityLogger, ActivityLogEntry, ActivityCategory, ActivityAction, 
    ActivitySeverity, TradingMode, ChainType
)


class TestTradingActivities:
    """Test trading-related activity logging"""
    
    @pytest.mark.asyncio
    async def test_trade_execution_success(self, activity_logger_instance, sample_trading_activities):
        """Test successful trade execution logging"""
        logger = activity_logger_instance
        
        trade_data = sample_trading_activities[0]  # Buy order
        activity_id = await logger.log_trading_activity(**trade_data)
        
        # Should create trading activity
        assert len(logger._buffer) == 1
        entry = logger._buffer[0]
        
        assert entry.category == ActivityCategory.TRADING
        assert entry.action == ActivityAction.EXECUTE
        assert entry.source == "trading_engine"
        assert entry.event_type == "buy_order"
        assert entry.trading_mode == TradingMode.SIMULATION
        assert entry.token_address == "0x6982508145454Ce325dDbE47a25d4ec3d2311933"
        assert entry.chain == ChainType.ETHEREUM
        assert entry.amount_usd == Decimal("100.50")
    
    @pytest.mark.asyncio
    async def test_trade_execution_failure(self, activity_logger_instance, sample_trading_activities):
        """Test failed trade execution logging"""
        logger = activity_logger_instance
        
        trade_data = sample_trading_activities[2]  # Failed swap
        activity_id = await logger.log_trading_activity(**trade_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.TRADING
        assert entry.action == ActivityAction.FAILURE
        assert entry.severity == ActivitySeverity.ERROR
        assert entry.error_code == "INSUFFICIENT_BALANCE"
        assert entry.error_message == "Insufficient token balance for swap"
    
    @pytest.mark.asyncio
    async def test_trading_mode_variations(self, activity_logger_instance):
        """Test logging activities in different trading modes"""
        logger = activity_logger_instance
        
        modes = [TradingMode.ANALYSIS, TradingMode.SIMULATION, TradingMode.LIVE]
        
        for mode in modes:
            activity_id = await logger.log_trading_activity(
                action=ActivityAction.EXECUTE,
                title=f"Trade in {mode.value} mode",
                trading_mode=mode,
                token_address="0x123",
                chain=ChainType.ETHEREUM,
                amount_usd=Decimal("100.00")
            )
        
        # Should have 3 entries, one for each mode
        assert len(logger._buffer) == 3
        for i, mode in enumerate(modes):
            assert logger._buffer[i].trading_mode == mode
    
    @pytest.mark.asyncio
    async def test_multi_chain_trading(self, activity_logger_instance):
        """Test trading activities across different chains"""
        logger = activity_logger_instance
        
        chains = [ChainType.ETHEREUM, ChainType.SOLANA, ChainType.BASE]
        tokens = ["0x123", "So11111111111111111111111111111111111111112", "0x456"]
        
        for chain, token in zip(chains, tokens):
            activity_id = await logger.log_trading_activity(
                action=ActivityAction.EXECUTE,
                title=f"Trade on {chain.value}",
                trading_mode=TradingMode.LIVE,
                token_address=token,
                chain=chain,
                amount_usd=Decimal("250.75")
            )
        
        # Should log trades for each chain
        assert len(logger._buffer) == 3
        for i, (chain, token) in enumerate(zip(chains, tokens)):
            assert logger._buffer[i].chain == chain
            assert logger._buffer[i].token_address == token
    
    @pytest.mark.asyncio
    async def test_trading_performance_metrics(self, activity_logger_instance):
        """Test trading activities with performance metrics"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_trading_activity(
            action=ActivityAction.EXECUTE,
            title="High-performance trade",
            trading_mode=TradingMode.LIVE,
            token_address="0x123",
            chain=ChainType.ETHEREUM,
            amount_usd=Decimal("1000.00"),
            execution_time_ms=850,
            response_time_ms=200,
            metadata={
                "slippage": 0.05,
                "gas_fee": 15.50,
                "dex": "uniswap_v3"
            }
        )
        
        entry = logger._buffer[0]
        assert entry.execution_time_ms == 850
        assert entry.response_time_ms == 200
        assert entry.metadata["slippage"] == 0.05
        assert entry.metadata["gas_fee"] == 15.50


class TestSystemActivities:
    """Test system-related activity logging"""
    
    @pytest.mark.asyncio
    async def test_system_startup(self, activity_logger_instance, sample_system_activities):
        """Test system startup logging"""
        logger = activity_logger_instance
        
        system_data = sample_system_activities[0]  # Mode switch
        activity_id = await logger.log_activity(**system_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.SYSTEM
        assert entry.action == ActivityAction.START
        assert entry.source == "mode_manager"
        assert entry.event_type == "mode_switch"
        assert entry.metadata["previous_mode"] == "simulation"
        assert entry.metadata["new_mode"] == "live"
    
    @pytest.mark.asyncio
    async def test_system_error(self, activity_logger_instance, sample_system_activities):
        """Test system error logging"""
        logger = activity_logger_instance
        
        error_data = sample_system_activities[1]  # Database error
        activity_id = await logger.log_activity(**error_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.SYSTEM
        assert entry.action == ActivityAction.ERROR
        assert entry.severity == ActivitySeverity.CRITICAL
        assert entry.error_code == "DB_CONNECTION_TIMEOUT"
        assert entry.error_message == "Failed to connect to database after 5 attempts"
    
    @pytest.mark.asyncio
    async def test_system_lifecycle_events(self, activity_logger_instance):
        """Test complete system lifecycle logging"""
        logger = activity_logger_instance
        
        lifecycle_events = [
            (ActivityAction.START, "System startup", "System started successfully"),
            (ActivityAction.PAUSE, "System pause", "System paused for maintenance"),
            (ActivityAction.RESUME, "System resume", "System resumed after maintenance"),
            (ActivityAction.STOP, "System shutdown", "System shutdown initiated")
        ]
        
        for action, title, description in lifecycle_events:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=action,
                source="system_manager",
                event_type="lifecycle",
                title=title,
                description=description,
                metadata={"component": "core_system"}
            )
        
        # Should log all lifecycle events
        assert len(logger._buffer) == 4
        for i, (action, title, description) in enumerate(lifecycle_events):
            assert logger._buffer[i].action == action
            assert logger._buffer[i].title == title
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, activity_logger_instance):
        """Test system health monitoring activities"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.SYSTEM,
            action=ActivityAction.EXECUTE,
            source="health_monitor",
            event_type="health_check",
            title="System health check",
            execution_time_ms=125,
            memory_usage_mb=256,
            cpu_usage_pct=Decimal("45.7"),
            metadata={
                "disk_usage_pct": 68.5,
                "network_latency_ms": 12,
                "active_connections": 15
            }
        )
        
        entry = logger._buffer[0]
        assert entry.execution_time_ms == 125
        assert entry.memory_usage_mb == 256
        assert entry.cpu_usage_pct == Decimal("45.7")
        assert entry.metadata["disk_usage_pct"] == 68.5


class TestUserActivities:
    """Test user-related activity logging"""
    
    @pytest.mark.asyncio
    async def test_user_login(self, activity_logger_instance, sample_user_activities):
        """Test user login logging"""
        logger = activity_logger_instance
        
        login_data = sample_user_activities[0]  # User login
        activity_id = await logger.log_activity(**login_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.USER
        assert entry.action == ActivityAction.LOGIN
        assert entry.source == "dashboard"
        assert entry.user_id == 123456789
        assert entry.ip_address == "192.168.1.100"
        assert "Mozilla/5.0" in entry.user_agent
    
    @pytest.mark.asyncio
    async def test_user_dashboard_navigation(self, activity_logger_instance, sample_user_activities):
        """Test user dashboard navigation logging"""
        logger = activity_logger_instance
        
        nav_data = sample_user_activities[1]  # Page access
        activity_id = await logger.log_activity(**nav_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.USER
        assert entry.action == ActivityAction.ACCESS
        assert entry.dashboard_component == "trading"
        assert entry.dashboard_action == "view_portfolio"
        assert entry.user_id == 123456789
    
    @pytest.mark.asyncio
    async def test_user_session_tracking(self, activity_logger_instance):
        """Test user session activity tracking"""
        logger = activity_logger_instance
        
        session_id = str(uuid.uuid4())
        user_id = 123456789
        
        # Multiple activities in same session
        activities = [
            ("login", "User logged in"),
            ("access", "Accessed trading page"),
            ("access", "Viewed portfolio"),
            ("execute", "Placed trade order"),
            ("logout", "User logged out")
        ]
        
        for action_str, title in activities:
            action = getattr(ActivityAction, action_str.upper())
            activity_id = await logger.log_user_action(
                user_id=user_id,
                action=action,
                component="dashboard",
                title=title,
                session_id=session_id
            )
        
        # Should track all activities in session
        assert len(logger._buffer) == 5
        for entry in logger._buffer:
            assert entry.session_id == session_id
            assert entry.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_user_preferences_changes(self, activity_logger_instance):
        """Test user preference change logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_user_action(
            user_id=123456789,
            action=ActivityAction.UPDATE,
            component="settings",
            title="Updated trading preferences",
            metadata={
                "changed_settings": ["risk_tolerance", "auto_trading"],
                "old_values": {"risk_tolerance": "medium", "auto_trading": False},
                "new_values": {"risk_tolerance": "high", "auto_trading": True}
            }
        )
        
        entry = logger._buffer[0]
        assert entry.action == ActivityAction.UPDATE
        assert entry.dashboard_component == "settings"
        assert "changed_settings" in entry.metadata


class TestMLRLActivities:
    """Test ML/RL-related activity logging"""
    
    @pytest.mark.asyncio
    async def test_ml_prediction(self, activity_logger_instance, sample_ml_rl_activities):
        """Test ML model prediction logging"""
        logger = activity_logger_instance
        
        prediction_data = sample_ml_rl_activities[0]  # Price prediction
        activity_id = await logger.log_activity(**prediction_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.ML_RL
        assert entry.action == ActivityAction.EXECUTE
        assert entry.source == "lstm_model"
        assert entry.event_type == "price_prediction"
        assert entry.execution_time_ms == 850
        assert entry.metadata["confidence"] == 0.87
        assert entry.metadata["predicted_price"] == 0.00145
    
    @pytest.mark.asyncio
    async def test_rl_training(self, activity_logger_instance, sample_ml_rl_activities):
        """Test RL model training logging"""
        logger = activity_logger_instance
        
        training_data = sample_ml_rl_activities[1]  # Model training
        activity_id = await logger.log_activity(**training_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.ML_RL
        assert entry.action == ActivityAction.UPDATE
        assert entry.source == "dqn_agent"
        assert entry.event_type == "model_training"
        assert entry.execution_time_ms == 2500
        assert entry.metadata["episode"] == 1500
        assert entry.metadata["reward"] == 15.7
        assert entry.metadata["epsilon"] == 0.15
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, activity_logger_instance):
        """Test model evaluation logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.ML_RL,
            action=ActivityAction.EXECUTE,
            source="model_evaluator",
            event_type="model_evaluation",
            title="Model performance evaluation",
            execution_time_ms=3500,
            metadata={
                "model_type": "lstm",
                "accuracy": 0.87,
                "precision": 0.84,
                "recall": 0.91,
                "f1_score": 0.87,
                "test_samples": 1000
            }
        )
        
        entry = logger._buffer[0]
        assert entry.metadata["accuracy"] == 0.87
        assert entry.metadata["test_samples"] == 1000
    
    @pytest.mark.asyncio
    async def test_feature_engineering(self, activity_logger_instance):
        """Test feature engineering activity logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_activity(
            category=ActivityCategory.ML_RL,
            action=ActivityAction.CREATE,
            source="feature_engineer",
            event_type="feature_extraction",
            title="Generated trading features",
            execution_time_ms=1200,
            metadata={
                "input_features": 15,
                "output_features": 45,
                "feature_types": ["technical", "social", "on_chain"],
                "data_points": 5000
            }
        )
        
        entry = logger._buffer[0]
        assert entry.action == ActivityAction.CREATE
        assert entry.metadata["input_features"] == 15
        assert entry.metadata["output_features"] == 45


class TestSecurityActivities:
    """Test security-related activity logging"""
    
    @pytest.mark.asyncio
    async def test_security_violation(self, activity_logger_instance, sample_security_activities):
        """Test security violation logging"""
        logger = activity_logger_instance
        
        violation_data = sample_security_activities[0]  # Unauthorized access
        activity_id = await logger.log_activity(**violation_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.SECURITY
        assert entry.action == ActivityAction.VIOLATION
        assert entry.severity == ActivitySeverity.WARNING
        assert entry.source == "auth_manager"
        assert entry.risk_score == 75
        assert entry.security_level == "high"
        assert entry.ip_address == "192.168.1.200"
    
    @pytest.mark.asyncio
    async def test_security_alert(self, activity_logger_instance, sample_security_activities):
        """Test security alert logging"""
        logger = activity_logger_instance
        
        alert_data = sample_security_activities[1]  # Suspicious trade
        activity_id = await logger.log_activity(**alert_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.SECURITY
        assert entry.action == ActivityAction.ALERT
        assert entry.severity == ActivitySeverity.ALERT
        assert entry.risk_score == 90
        assert entry.metadata["pattern"] == "large_volume_spike"
        assert entry.metadata["deviation"] == 3.5
    
    @pytest.mark.asyncio
    async def test_authentication_activities(self, activity_logger_instance):
        """Test authentication-related logging"""
        logger = activity_logger_instance
        
        auth_events = [
            (ActivityAction.LOGIN, "Successful login", ActivitySeverity.INFO, 10),
            (ActivityAction.FAILURE, "Failed login attempt", ActivitySeverity.WARNING, 50),
            (ActivityAction.VIOLATION, "Multiple failed attempts", ActivitySeverity.ERROR, 85),
            (ActivityAction.LOGOUT, "User logout", ActivitySeverity.INFO, 5)
        ]
        
        for action, title, severity, risk_score in auth_events:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SECURITY,
                action=action,
                source="auth_system",
                event_type="authentication",
                title=title,
                severity=severity,
                risk_score=risk_score,
                ip_address="192.168.1.100",
                user_agent="Test Browser"
            )
        
        # Should log all auth events
        assert len(logger._buffer) == 4
        for i, (action, title, severity, risk_score) in enumerate(auth_events):
            assert logger._buffer[i].action == action
            assert logger._buffer[i].severity == severity
            assert logger._buffer[i].risk_score == risk_score


class TestAPIActivities:
    """Test API-related activity logging"""
    
    @pytest.mark.asyncio
    async def test_api_success(self, activity_logger_instance, sample_api_activities):
        """Test successful API call logging"""
        logger = activity_logger_instance
        
        api_data = sample_api_activities[0]  # Successful call
        activity_id = await logger.log_activity(**api_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.API
        assert entry.action == ActivityAction.SUCCESS
        assert entry.source == "jupiter_client"
        assert entry.api_endpoint == "/quote"
        assert entry.http_method == "GET"
        assert entry.http_status == 200
        assert entry.response_time_ms == 450
    
    @pytest.mark.asyncio
    async def test_api_failure(self, activity_logger_instance, sample_api_activities):
        """Test failed API call logging"""
        logger = activity_logger_instance
        
        api_data = sample_api_activities[1]  # Failed call
        activity_id = await logger.log_activity(**api_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.API
        assert entry.action == ActivityAction.FAILURE
        assert entry.severity == ActivitySeverity.WARNING
        assert entry.http_status == 429
        assert entry.error_code == "RATE_LIMIT_EXCEEDED"
    
    @pytest.mark.asyncio
    async def test_api_rate_limiting(self, activity_logger_instance):
        """Test API rate limiting activity logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_api_call(
            api_name="external_service",
            endpoint="/api/data",
            method="POST",
            status_code=429,
            response_time_ms=100,
            success=False,
            error_code="RATE_LIMITED",
            metadata={
                "rate_limit_reset": 3600,
                "remaining_requests": 0,
                "retry_after": 60
            }
        )
        
        entry = logger._buffer[0]
        assert entry.http_status == 429
        assert entry.error_code == "RATE_LIMITED"
        assert entry.metadata["retry_after"] == 60


class TestPerformanceActivities:
    """Test performance-related activity logging"""
    
    @pytest.mark.asyncio
    async def test_performance_success(self, activity_logger_instance, sample_performance_activities):
        """Test successful performance logging"""
        logger = activity_logger_instance
        
        perf_data = sample_performance_activities[0]  # Performance calculation
        activity_id = await logger.log_activity(**perf_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.PERFORMANCE
        assert entry.action == ActivityAction.SUCCESS
        assert entry.source == "portfolio_manager"
        assert entry.execution_time_ms == 125
        assert entry.memory_usage_mb == 45
        assert entry.cpu_usage_pct == Decimal("12.5")
    
    @pytest.mark.asyncio
    async def test_performance_timeout(self, activity_logger_instance, sample_performance_activities):
        """Test performance timeout logging"""
        logger = activity_logger_instance
        
        timeout_data = sample_performance_activities[1]  # Timeout
        activity_id = await logger.log_activity(**timeout_data)
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.PERFORMANCE
        assert entry.action == ActivityAction.FAILURE
        assert entry.severity == ActivitySeverity.WARNING
        assert entry.execution_time_ms == 30000
        assert entry.error_code == "CALCULATION_TIMEOUT"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, activity_logger_instance):
        """Test performance benchmarking logging"""
        logger = activity_logger_instance
        
        activity_id = await logger.log_performance(
            source="benchmark_suite",
            operation="full_system_benchmark",
            execution_time_ms=45000,
            success=True,
            metadata={
                "benchmark_type": "comprehensive",
                "test_cases": 150,
                "passed": 148,
                "failed": 2,
                "performance_score": 92.5
            }
        )
        
        entry = logger._buffer[0]
        assert entry.category == ActivityCategory.PERFORMANCE
        assert entry.execution_time_ms == 45000
        assert entry.metadata["performance_score"] == 92.5


class TestActivityCategoryCompleteness:
    """Test that all activity categories are properly supported"""
    
    @pytest.mark.asyncio
    async def test_all_categories_supported(self, activity_logger_instance):
        """Test that all ActivityCategory enum values can be logged"""
        logger = activity_logger_instance
        
        all_categories = list(ActivityCategory)
        
        for category in all_categories:
            activity_id = await logger.log_activity(
                category=category,
                action=ActivityAction.EXECUTE,
                source=f"test_{category.value}",
                event_type=f"test_{category.value}_event",
                title=f"Test {category.value} activity"
            )
        
        # Should log activity for each category
        assert len(logger._buffer) == len(all_categories)
        for i, category in enumerate(all_categories):
            assert logger._buffer[i].category == category
    
    @pytest.mark.asyncio
    async def test_all_actions_supported(self, activity_logger_instance):
        """Test that all ActivityAction enum values can be logged"""
        logger = activity_logger_instance
        
        all_actions = list(ActivityAction)
        
        for action in all_actions:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=action,
                source=f"test_{action.value}",
                event_type=f"test_{action.value}_event",
                title=f"Test {action.value} action"
            )
        
        # Should log activity for each action
        assert len(logger._buffer) == len(all_actions)
        for i, action in enumerate(all_actions):
            assert logger._buffer[i].action == action
    
    @pytest.mark.asyncio
    async def test_all_severities_supported(self, activity_logger_instance):
        """Test that all ActivitySeverity enum values can be logged"""
        logger = activity_logger_instance
        
        all_severities = list(ActivitySeverity)
        
        for severity in all_severities:
            activity_id = await logger.log_activity(
                category=ActivityCategory.SYSTEM,
                action=ActivityAction.EXECUTE,
                source=f"test_{severity.value}",
                event_type=f"test_{severity.value}_event",
                title=f"Test {severity.value} severity",
                severity=severity
            )
        
        # Should log activity for each severity
        assert len(logger._buffer) == len(all_severities)
        for i, severity in enumerate(all_severities):
            assert logger._buffer[i].severity == severity


# NOTE: All these tests should FAIL initially since the comprehensive implementation
# may not cover all activity types and categories yet. This follows TDD methodology.