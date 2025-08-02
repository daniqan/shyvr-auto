"""
TDD Test Suite for TradingCircuitBreaker

Test-driven development tests for comprehensive trading circuit breaker implementation
covering price movement detection, volume spike detection, volatility-based halts,
market-wide circuit breakers, cooldown management, and graduated response levels.

Following TDD methodology - these tests will initially fail and guide implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock

from src.safety.trading_circuit_breaker import (
    TradingCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerType,
    CircuitBreakerLevel,
    CircuitBreakerReason,
    CircuitBreakerResult,
    MarketCondition,
    PriceMovementData,
    VolumeData,
    VolatilityData,
    MarketData,
    CircuitBreakerStatus
)
from src.safety.emergency_stop_controller import EmergencyStopController
from src.portfolio.base import Portfolio


@pytest.fixture
def circuit_breaker_config():
    """Create test configuration for circuit breaker."""
    return CircuitBreakerConfig(
        # Price movement thresholds
        price_drop_warning_pct=Decimal("0.05"),  # 5% warning
        price_drop_critical_pct=Decimal("0.10"),  # 10% critical
        price_drop_emergency_pct=Decimal("0.15"),  # 15% emergency
        price_spike_warning_pct=Decimal("0.10"),  # 10% spike warning
        price_spike_critical_pct=Decimal("0.20"),  # 20% spike critical
        
        # Volume thresholds
        volume_spike_warning_multiplier=Decimal("3.0"),  # 3x normal volume
        volume_spike_critical_multiplier=Decimal("5.0"),  # 5x normal volume
        volume_spike_emergency_multiplier=Decimal("10.0"),  # 10x normal volume
        
        # Volatility thresholds
        volatility_warning_pct=Decimal("0.15"),  # 15% volatility
        volatility_critical_pct=Decimal("0.25"),  # 25% volatility
        volatility_emergency_pct=Decimal("0.40"),  # 40% volatility
        
        # Market-wide thresholds
        market_crash_threshold_pct=Decimal("0.20"),  # 20% market drop
        flash_crash_threshold_pct=Decimal("0.10"),  # 10% flash crash
        flash_crash_time_window_minutes=5,
        
        # Cooldown periods
        warning_cooldown_minutes=5,
        critical_cooldown_minutes=15,
        emergency_cooldown_minutes=30,
        
        # Detection windows
        price_movement_window_minutes=15,
        volume_analysis_window_minutes=60,
        volatility_calculation_window_minutes=30,
        
        # System settings
        max_concurrent_breakers=5,
        enable_graduated_response=True,
        require_manual_reset_for_emergency=True
    )


@pytest.fixture
def mock_emergency_stop_controller():
    """Create mock emergency stop controller."""
    controller = MagicMock(spec=EmergencyStopController)
    controller.activate_global_emergency_stop = AsyncMock()
    controller.activate_mode_emergency_stop = AsyncMock()
    controller.is_global_emergency_stopped = False
    return controller


@pytest.fixture
def mock_portfolio():
    """Create mock portfolio for testing."""
    portfolio = MagicMock(spec=Portfolio)
    portfolio.total_value = Decimal("100000")
    portfolio.get_performance_metrics.return_value = MagicMock(
        max_drawdown=Decimal("0.05"),
        daily_pnl=Decimal("-1000"),
        initial_balance=Decimal("100000")
    )
    return portfolio


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return MarketData(
        symbol="BTC-USD",
        price=Decimal("50000"),
        volume=Decimal("1000000"),
        timestamp=datetime.now(),
        bid_ask_spread=Decimal("10"),
        market_cap=Decimal("1000000000"),
        volatility_24h=Decimal("0.05")
    )


class TestTradingCircuitBreakerConfig:
    """Test TradingCircuitBreaker configuration validation."""
    
    def test_config_creation_with_valid_parameters(self, circuit_breaker_config):
        """Test creating configuration with valid parameters."""
        assert circuit_breaker_config.price_drop_warning_pct == Decimal("0.05")
        assert circuit_breaker_config.volume_spike_warning_multiplier == Decimal("3.0")
        assert circuit_breaker_config.volatility_warning_pct == Decimal("0.15")
        assert circuit_breaker_config.warning_cooldown_minutes == 5
    
    def test_config_validation_with_invalid_thresholds(self):
        """Test configuration validation with invalid thresholds."""
        with pytest.raises(ValueError, match="Price drop warning threshold must be positive"):
            CircuitBreakerConfig(price_drop_warning_pct=Decimal("-0.05"))
        
        with pytest.raises(ValueError, match="Volume spike multiplier must be greater than 1"):
            CircuitBreakerConfig(volume_spike_warning_multiplier=Decimal("0.5"))
        
        with pytest.raises(ValueError, match="Cooldown period must be positive"):
            CircuitBreakerConfig(warning_cooldown_minutes=-5)
    
    def test_config_threshold_ordering_validation(self):
        """Test that threshold levels are properly ordered."""
        with pytest.raises(ValueError, match="Critical threshold must be greater than warning"):
            CircuitBreakerConfig(
                price_drop_warning_pct=Decimal("0.10"),
                price_drop_critical_pct=Decimal("0.05")  # Less than warning
            )
    
    def test_production_config_creation(self):
        """Test creating production-safe configuration."""
        prod_config = CircuitBreakerConfig.production_config()
        
        # Production should have stricter thresholds
        assert prod_config.price_drop_emergency_pct <= Decimal("0.10")
        assert prod_config.require_manual_reset_for_emergency is True
        assert prod_config.enable_graduated_response is True


class TestTradingCircuitBreakerBasicFunctionality:
    """Test basic TradingCircuitBreaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test circuit breaker initialization."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        assert breaker.config == circuit_breaker_config
        assert breaker.emergency_stop_controller == mock_emergency_stop_controller
        assert len(breaker.active_breakers) == 0
        assert len(breaker.breaker_history) == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_requires_emergency_stop_controller(self, circuit_breaker_config):
        """Test that circuit breaker requires emergency stop controller."""
        with pytest.raises(ValueError, match="EmergencyStopController is required"):
            TradingCircuitBreaker(config=circuit_breaker_config, emergency_stop_controller=None)
    
    @pytest.mark.asyncio
    async def test_check_trading_allowed_with_no_active_breakers(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test trading allowed when no circuit breakers are active."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        result = await breaker.can_execute_trade("BTC-USD", Decimal("1000"))
        assert result.is_allowed is True
        assert result.restrictions == []


class TestPriceMovementCircuitBreakers:
    """Test price movement detection and circuit breakers."""
    
    @pytest.mark.asyncio
    async def test_price_drop_warning_threshold_detection(self, circuit_breaker_config, mock_emergency_stop_controller, sample_market_data):
        """Test detection of price drop warning threshold."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Simulate 5% price drop (warning threshold)
        previous_data = sample_market_data
        current_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("47500"),  # 5% drop from 50000
            volume=sample_market_data.volume,
            timestamp=datetime.now(),
            bid_ask_spread=sample_market_data.bid_ask_spread,
            market_cap=sample_market_data.market_cap,
            volatility_24h=sample_market_data.volatility_24h
        )
        
        # Add price history to enable movement detection
        breaker.price_history["BTC-USD"] = [(previous_data.timestamp, previous_data.price)]
        
        result = await breaker.check_price_movement_triggers(current_data)
        
        assert result.should_trigger is True
        assert result.level == CircuitBreakerLevel.WARNING
        assert result.reason == CircuitBreakerReason.PRICE_DROP_WARNING
        assert "5." in result.message and "%" in result.message
    
    @pytest.mark.asyncio
    async def test_price_drop_critical_threshold_detection(self, circuit_breaker_config, mock_emergency_stop_controller, sample_market_data):
        """Test detection of price drop critical threshold."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Simulate 10% price drop (critical threshold)
        previous_data = sample_market_data
        current_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("45000"),  # 10% drop from 50000
            volume=sample_market_data.volume,
            timestamp=datetime.now(),
            bid_ask_spread=sample_market_data.bid_ask_spread,
            market_cap=sample_market_data.market_cap,
            volatility_24h=sample_market_data.volatility_24h
        )
        
        breaker.price_history["BTC-USD"] = [(previous_data.timestamp, previous_data.price)]
        
        result = await breaker.check_price_movement_triggers(current_data)
        
        assert result.should_trigger is True
        assert result.level == CircuitBreakerLevel.CRITICAL
        assert result.reason == CircuitBreakerReason.PRICE_DROP_CRITICAL
    
    @pytest.mark.asyncio
    async def test_price_spike_detection(self, circuit_breaker_config, mock_emergency_stop_controller, sample_market_data):
        """Test detection of price spike."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Simulate 20% price spike (critical threshold)
        previous_data = sample_market_data
        current_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("60000"),  # 20% spike from 50000
            volume=sample_market_data.volume,
            timestamp=datetime.now(),
            bid_ask_spread=sample_market_data.bid_ask_spread,
            market_cap=sample_market_data.market_cap,
            volatility_24h=sample_market_data.volatility_24h
        )
        
        breaker.price_history["BTC-USD"] = [(previous_data.timestamp, previous_data.price)]
        
        result = await breaker.check_price_movement_triggers(current_data)
        
        assert result.should_trigger is True
        assert result.level == CircuitBreakerLevel.CRITICAL
        assert result.reason == CircuitBreakerReason.PRICE_SPIKE_CRITICAL


class TestVolumeSpikeCircuitBreakers:
    """Test volume spike detection and circuit breakers."""
    
    @pytest.mark.asyncio
    async def test_volume_spike_warning_detection(self, circuit_breaker_config, mock_emergency_stop_controller, sample_market_data):
        """Test detection of volume spike warning."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Set up normal volume history
        normal_volume = Decimal("1000000")
        breaker.volume_history["BTC-USD"] = [
            (datetime.now() - timedelta(minutes=10), normal_volume),
            (datetime.now() - timedelta(minutes=20), normal_volume),
            (datetime.now() - timedelta(minutes=30), normal_volume)
        ]
        
        # Create data with 3x volume spike (warning threshold)
        spike_data = MarketData(
            symbol="BTC-USD",
            price=sample_market_data.price,
            volume=normal_volume * 3,  # 3x normal volume
            timestamp=datetime.now(),
            bid_ask_spread=sample_market_data.bid_ask_spread,
            market_cap=sample_market_data.market_cap,
            volatility_24h=sample_market_data.volatility_24h
        )
        
        result = await breaker.check_volume_spike_triggers(spike_data)
        
        assert result.should_trigger is True
        assert result.level == CircuitBreakerLevel.WARNING
        assert result.reason == CircuitBreakerReason.VOLUME_SPIKE_WARNING
    
    @pytest.mark.asyncio
    async def test_volume_spike_critical_detection(self, circuit_breaker_config, mock_emergency_stop_controller, sample_market_data):
        """Test detection of volume spike critical threshold."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Set up normal volume history
        normal_volume = Decimal("1000000")
        breaker.volume_history["BTC-USD"] = [
            (datetime.now() - timedelta(minutes=10), normal_volume),
            (datetime.now() - timedelta(minutes=20), normal_volume)
        ]
        
        # Create data with 5x volume spike (critical threshold)
        spike_data = MarketData(
            symbol="BTC-USD",
            price=sample_market_data.price,
            volume=normal_volume * 5,  # 5x normal volume
            timestamp=datetime.now(),
            bid_ask_spread=sample_market_data.bid_ask_spread,
            market_cap=sample_market_data.market_cap,
            volatility_24h=sample_market_data.volatility_24h
        )
        
        result = await breaker.check_volume_spike_triggers(spike_data)
        
        assert result.should_trigger is True
        assert result.level == CircuitBreakerLevel.CRITICAL
        assert result.reason == CircuitBreakerReason.VOLUME_SPIKE_CRITICAL


class TestVolatilityCircuitBreakers:
    """Test volatility-based circuit breakers."""
    
    @pytest.mark.asyncio
    async def test_volatility_warning_threshold(self, circuit_breaker_config, mock_emergency_stop_controller, sample_market_data):
        """Test volatility warning threshold detection."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Create data with 15% volatility (warning threshold)
        volatile_data = MarketData(
            symbol="BTC-USD",
            price=sample_market_data.price,
            volume=sample_market_data.volume,
            timestamp=datetime.now(),
            bid_ask_spread=sample_market_data.bid_ask_spread,
            market_cap=sample_market_data.market_cap,
            volatility_24h=Decimal("0.15")  # 15% volatility
        )
        
        result = await breaker.check_volatility_triggers(volatile_data)
        
        assert result.should_trigger is True
        assert result.level == CircuitBreakerLevel.WARNING
        assert result.reason == CircuitBreakerReason.VOLATILITY_WARNING
    
    @pytest.mark.asyncio
    async def test_volatility_emergency_threshold(self, circuit_breaker_config, mock_emergency_stop_controller, sample_market_data):
        """Test volatility emergency threshold detection."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Create data with 40% volatility (emergency threshold)
        volatile_data = MarketData(
            symbol="BTC-USD",
            price=sample_market_data.price,
            volume=sample_market_data.volume,
            timestamp=datetime.now(),
            bid_ask_spread=sample_market_data.bid_ask_spread,
            market_cap=sample_market_data.market_cap,
            volatility_24h=Decimal("0.40")  # 40% volatility
        )
        
        result = await breaker.check_volatility_triggers(volatile_data)
        
        assert result.should_trigger is True
        assert result.level == CircuitBreakerLevel.EMERGENCY
        assert result.reason == CircuitBreakerReason.VOLATILITY_EMERGENCY


class TestMarketWideCircuitBreakers:
    """Test market-wide circuit breakers for extreme conditions."""
    
    @pytest.mark.asyncio
    async def test_market_crash_detection(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test market crash detection across multiple assets."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Simulate market crash - multiple assets dropping 20%
        market_data = [
            MarketData("BTC-USD", Decimal("40000"), Decimal("1000000"), datetime.now(), Decimal("10"), Decimal("800000000"), Decimal("0.05")),
            MarketData("ETH-USD", Decimal("1600"), Decimal("500000"), datetime.now(), Decimal("5"), Decimal("200000000"), Decimal("0.06")),
            MarketData("SOL-USD", Decimal("80"), Decimal("100000"), datetime.now(), Decimal("1"), Decimal("40000000"), Decimal("0.08"))
        ]
        
        # Set up price history for 20% drops
        breaker.price_history["BTC-USD"] = [(datetime.now() - timedelta(minutes=10), Decimal("50000"))]
        breaker.price_history["ETH-USD"] = [(datetime.now() - timedelta(minutes=10), Decimal("2000"))]
        breaker.price_history["SOL-USD"] = [(datetime.now() - timedelta(minutes=10), Decimal("100"))]
        
        result = await breaker.check_market_wide_triggers(market_data)
        
        assert result.should_trigger is True
        assert result.level == CircuitBreakerLevel.EMERGENCY
        assert result.reason == CircuitBreakerReason.MARKET_CRASH
    
    @pytest.mark.asyncio
    async def test_flash_crash_detection(self, circuit_breaker_config, mock_emergency_stop_controller, sample_market_data):
        """Test flash crash detection within time window."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Simulate flash crash - 10% drop within 5 minutes
        current_time = datetime.now()
        flash_crash_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("45000"),  # 10% drop from 50000
            volume=sample_market_data.volume,
            timestamp=current_time,
            bid_ask_spread=sample_market_data.bid_ask_spread,
            market_cap=sample_market_data.market_cap,
            volatility_24h=sample_market_data.volatility_24h
        )
        
        # Set price history within flash crash time window
        breaker.price_history["BTC-USD"] = [
            (current_time - timedelta(minutes=3), Decimal("50000"))
        ]
        
        result = await breaker.check_flash_crash_triggers(flash_crash_data)
        
        assert result.should_trigger is True
        assert result.level == CircuitBreakerLevel.EMERGENCY
        assert result.reason == CircuitBreakerReason.FLASH_CRASH


class TestCooldownPeriodManagement:
    """Test cooldown period management for circuit breakers."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_cooldown_prevents_duplicate_triggers(self, circuit_breaker_config, mock_emergency_stop_controller, sample_market_data):
        """Test that cooldown period prevents duplicate triggers."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Trigger initial circuit breaker
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.WARNING,
            reason=CircuitBreakerReason.PRICE_DROP_WARNING,
            symbol="BTC-USD",
            message="Price drop warning test"
        )
        
        # Attempt to trigger same breaker type during cooldown
        result = await breaker.can_trigger_circuit_breaker(
            CircuitBreakerType.PRICE_MOVEMENT,
            CircuitBreakerLevel.WARNING,
            "BTC-USD"
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cooldown_period_expiration_allows_new_triggers(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test that expired cooldown allows new triggers."""
        # Set very short cooldown for testing
        config = circuit_breaker_config
        config.warning_cooldown_minutes = 0  # No cooldown for testing
        
        breaker = TradingCircuitBreaker(
            config=config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Trigger initial circuit breaker
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.WARNING,
            reason=CircuitBreakerReason.PRICE_DROP_WARNING,
            symbol="BTC-USD",
            message="Price drop warning test"
        )
        
        # Check if can trigger after cooldown expires
        result = await breaker.can_trigger_circuit_breaker(
            CircuitBreakerType.PRICE_MOVEMENT,
            CircuitBreakerLevel.WARNING,
            "BTC-USD"
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_different_circuit_breaker_types_independent_cooldowns(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test that different circuit breaker types have independent cooldowns."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Trigger price movement circuit breaker
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.WARNING,
            reason=CircuitBreakerReason.PRICE_DROP_WARNING,
            symbol="BTC-USD",
            message="Price drop warning test"
        )
        
        # Check if volume circuit breaker can still be triggered
        result = await breaker.can_trigger_circuit_breaker(
            CircuitBreakerType.VOLUME_SPIKE,
            CircuitBreakerLevel.WARNING,
            "BTC-USD"
        )
        
        assert result is True


class TestGraduatedResponseLevels:
    """Test graduated response levels (warning, critical, emergency)."""
    
    @pytest.mark.asyncio
    async def test_warning_level_allows_trading_with_restrictions(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test that warning level allows trading but with restrictions."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Activate warning level circuit breaker
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.WARNING,
            reason=CircuitBreakerReason.PRICE_DROP_WARNING,
            symbol="BTC-USD",
            message="Price drop warning"
        )
        
        result = await breaker.can_execute_trade("BTC-USD", Decimal("1000"))
        
        assert result.is_allowed is True  # Trading allowed but with restrictions
        assert "price_movement_warning" in result.restrictions
        assert result.max_trade_size is not None
        assert result.max_trade_size < Decimal("1000")  # Reduced trade size
    
    @pytest.mark.asyncio
    async def test_critical_level_severely_restricts_trading(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test that critical level severely restricts trading."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Activate critical level circuit breaker
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.CRITICAL,
            reason=CircuitBreakerReason.PRICE_DROP_CRITICAL,
            symbol="BTC-USD",
            message="Price drop critical"
        )
        
        result = await breaker.can_execute_trade("BTC-USD", Decimal("1000"))
        
        assert result.is_allowed is True  # Still allowed but heavily restricted
        assert "price_movement_critical" in result.restrictions
        assert result.max_trade_size < Decimal("200")  # Severely reduced trade size
    
    @pytest.mark.asyncio
    async def test_emergency_level_halts_trading(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test that emergency level halts trading."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Activate emergency level circuit breaker
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.EMERGENCY,
            reason=CircuitBreakerReason.PRICE_DROP_EMERGENCY,
            symbol="BTC-USD",
            message="Price drop emergency"
        )
        
        result = await breaker.can_execute_trade("BTC-USD", Decimal("1000"))
        
        assert result.is_allowed is False  # Trading halted
        assert "emergency_circuit_breaker" in result.restrictions
        
        # Should also trigger emergency stop controller
        mock_emergency_stop_controller.activate_global_emergency_stop.assert_called_once()


class TestEmergencyStopIntegration:
    """Test integration with EmergencyStopController."""
    
    @pytest.mark.asyncio
    async def test_emergency_circuit_breaker_triggers_emergency_stop(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test that emergency level circuit breakers trigger emergency stop."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.MARKET_WIDE,
            level=CircuitBreakerLevel.EMERGENCY,
            reason=CircuitBreakerReason.MARKET_CRASH,
            symbol="MARKET",
            message="Market crash detected"
        )
        
        # Verify emergency stop was triggered
        mock_emergency_stop_controller.activate_global_emergency_stop.assert_called_once()
        args, kwargs = mock_emergency_stop_controller.activate_global_emergency_stop.call_args
        assert "market crash" in kwargs["message"].lower()
    
    @pytest.mark.asyncio
    async def test_critical_circuit_breakers_do_not_trigger_emergency_stop(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test that critical level circuit breakers do not trigger emergency stop."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.CRITICAL,
            reason=CircuitBreakerReason.PRICE_DROP_CRITICAL,
            symbol="BTC-USD",
            message="Price drop critical"
        )
        
        # Verify emergency stop was NOT triggered
        mock_emergency_stop_controller.activate_global_emergency_stop.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_respects_existing_emergency_stop(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test that circuit breaker respects existing emergency stop state."""
        mock_emergency_stop_controller.is_global_emergency_stopped = True
        
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        result = await breaker.can_execute_trade("BTC-USD", Decimal("1000"))
        
        assert result.is_allowed is False
        assert "global_emergency_stop" in result.restrictions


class TestCircuitBreakerManualOperations:
    """Test manual circuit breaker operations."""
    
    @pytest.mark.asyncio
    async def test_manual_circuit_breaker_activation(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test manual activation of circuit breaker."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        result = await breaker.activate_manual_circuit_breaker(
            symbol="BTC-USD",
            level=CircuitBreakerLevel.CRITICAL,
            reason="Manual intervention due to suspicious activity",
            user_id="admin_user"
        )
        
        assert result.was_activated is True
        assert result.level == CircuitBreakerLevel.CRITICAL
        assert "BTC-USD" in breaker.active_breakers
    
    @pytest.mark.asyncio
    async def test_manual_circuit_breaker_deactivation(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test manual deactivation of circuit breaker."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # First activate a circuit breaker
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.WARNING,
            reason=CircuitBreakerReason.PRICE_DROP_WARNING,
            symbol="BTC-USD",
            message="Test activation"
        )
        
        # Then manually deactivate it
        result = await breaker.deactivate_circuit_breaker(
            symbol="BTC-USD",
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            user_id="admin_user",
            reason="Manual reset"
        )
        
        assert result.was_deactivated is True
        assert "BTC-USD" not in breaker.active_breakers
    
    @pytest.mark.asyncio
    async def test_emergency_manual_reset_requires_authentication(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test that emergency manual reset requires proper authentication."""
        config = circuit_breaker_config
        config.require_manual_reset_for_emergency = True
        
        breaker = TradingCircuitBreaker(
            config=config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Activate emergency circuit breaker
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.MARKET_WIDE,
            level=CircuitBreakerLevel.EMERGENCY,
            reason=CircuitBreakerReason.MARKET_CRASH,
            symbol="MARKET",
            message="Market crash"
        )
        
        # Attempt reset without proper authentication should fail
        with pytest.raises(ValueError, match="Authentication required for emergency reset"):
            await breaker.reset_emergency_circuit_breaker(
                user_id="",
                auth_token=""
            )


class TestCircuitBreakerMonitoringAndReporting:
    """Test circuit breaker monitoring and reporting functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_status_reporting(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test circuit breaker status reporting."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Activate multiple circuit breakers
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.WARNING,
            reason=CircuitBreakerReason.PRICE_DROP_WARNING,
            symbol="BTC-USD",
            message="Price warning"
        )
        
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.VOLUME_SPIKE,
            level=CircuitBreakerLevel.CRITICAL,
            reason=CircuitBreakerReason.VOLUME_SPIKE_CRITICAL,
            symbol="ETH-USD",
            message="Volume spike"
        )
        
        status = breaker.get_circuit_breaker_status()
        
        assert status.total_active_breakers == 2
        assert len(status.active_breakers) == 2
        assert status.active_breakers["BTC-USD"].level == CircuitBreakerLevel.WARNING
        assert status.active_breakers["ETH-USD"].level == CircuitBreakerLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_collection(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test collection of circuit breaker metrics."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Simulate some circuit breaker activity
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.WARNING,
            reason=CircuitBreakerReason.PRICE_DROP_WARNING,
            symbol="BTC-USD",
            message="Test"
        )
        
        await breaker.deactivate_circuit_breaker(
            symbol="BTC-USD",
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            user_id="system",
            reason="Auto-recovery"
        )
        
        metrics = breaker.get_circuit_breaker_metrics()
        
        assert metrics["total_activations"] >= 1
        assert metrics["total_deactivations"] >= 1
        assert "activations_by_type" in metrics
        assert "activations_by_level" in metrics
        assert "average_duration_seconds" in metrics
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_audit_trail(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test circuit breaker audit trail generation."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Perform various operations
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.CRITICAL,
            reason=CircuitBreakerReason.PRICE_DROP_CRITICAL,
            symbol="BTC-USD",
            message="Critical price drop"
        )
        
        audit_trail = breaker.get_audit_trail()
        
        assert len(audit_trail) >= 1
        assert audit_trail[0]["action"] == "circuit_breaker_activated"
        assert audit_trail[0]["symbol"] == "BTC-USD"
        assert audit_trail[0]["level"] == "CRITICAL"
        assert "timestamp" in audit_trail[0]


class TestCircuitBreakerIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_market_stress_scenario(self, circuit_breaker_config, mock_emergency_stop_controller, mock_portfolio):
        """Test comprehensive market stress scenario with multiple triggers."""
        breaker = TradingCircuitBreaker(
            config=circuit_breaker_config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Simulate extreme market conditions
        stress_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("40000"),  # 20% drop
            volume=Decimal("10000000"),  # 10x volume spike
            timestamp=datetime.now(),
            bid_ask_spread=Decimal("100"),  # Wide spread
            market_cap=Decimal("800000000"),  # Reduced market cap
            volatility_24h=Decimal("0.45")  # 45% volatility
        )
        
        # Set up price history for comparison
        breaker.price_history["BTC-USD"] = [
            (datetime.now() - timedelta(minutes=10), Decimal("50000"))
        ]
        
        # Set up volume history
        breaker.volume_history["BTC-USD"] = [
            (datetime.now() - timedelta(minutes=30), Decimal("1000000"))
        ]
        
        # Process market data through all circuit breaker checks
        results = await breaker.process_market_data(stress_data)
        
        # Should trigger multiple circuit breakers
        assert len(results.triggered_breakers) >= 2
        assert any(r.reason == CircuitBreakerReason.PRICE_DROP_EMERGENCY for r in results.triggered_breakers)
        assert any(r.reason == CircuitBreakerReason.VOLUME_SPIKE_EMERGENCY for r in results.triggered_breakers)
        assert any(r.reason == CircuitBreakerReason.VOLATILITY_EMERGENCY for r in results.triggered_breakers)
        
        # Should halt trading
        trade_result = await breaker.can_execute_trade("BTC-USD", Decimal("1000"))
        assert trade_result.is_allowed is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_scenario(self, circuit_breaker_config, mock_emergency_stop_controller):
        """Test circuit breaker recovery scenario."""
        # Set short cooldowns for testing
        config = circuit_breaker_config
        config.warning_cooldown_minutes = 0
        config.critical_cooldown_minutes = 0
        
        breaker = TradingCircuitBreaker(
            config=config,
            emergency_stop_controller=mock_emergency_stop_controller
        )
        
        # Trigger circuit breaker
        await breaker.activate_circuit_breaker(
            breaker_type=CircuitBreakerType.PRICE_MOVEMENT,
            level=CircuitBreakerLevel.CRITICAL,
            reason=CircuitBreakerReason.PRICE_DROP_CRITICAL,
            symbol="BTC-USD",
            message="Critical drop"
        )
        
        # Verify trading is restricted
        result1 = await breaker.can_execute_trade("BTC-USD", Decimal("1000"))
        assert result1.is_allowed is True  # Critical allows trading with restrictions
        assert result1.max_trade_size < Decimal("1000")
        
        # Simulate recovery conditions
        recovery_data = MarketData(
            symbol="BTC-USD",
            price=Decimal("48000"),  # Price recovering
            volume=Decimal("500000"),  # Normal volume
            timestamp=datetime.now(),
            bid_ask_spread=Decimal("10"),
            market_cap=Decimal("950000000"),
            volatility_24h=Decimal("0.08")  # Lower volatility
        )
        
        # Process recovery data
        await breaker.check_recovery_conditions(recovery_data)
        
        # Should allow auto-recovery for non-emergency levels
        recovery_result = await breaker.attempt_auto_recovery("BTC-USD")
        assert recovery_result.recovery_attempted is True
        
        # Verify trading restrictions are lifted
        result2 = await breaker.can_execute_trade("BTC-USD", Decimal("1000"))
        assert result2.is_allowed is True
        assert len(result2.restrictions) == 0