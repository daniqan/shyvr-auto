"""
TDD Tests for TradingSafetyManager

Following TDD methodology:
1. Write failing tests first
2. Implement minimal code to make tests pass
3. Refactor for production readiness

Tests cover:
- Pre-trade validation
- Position size limits
- Maximum order value checks
- Rate limiting per trading pair
- Balance verification
- Risk exposure validation
- Integration with trading modes
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4

# Importing modules that should exist but will initially fail
try:
    from src.safety.trading_safety_manager import (
        TradingSafetyManager,
        TradingSafetyConfig,
        PreTradeValidationResult,
        ValidationReason,
        ValidationStatus,
        RateLimitState,
        PositionSizeValidation,
        RiskExposureValidation,
        TradeOrder
    )
except ImportError:
    # This is expected initially in TDD - we'll implement these after tests fail
    pytest.skip("TradingSafetyManager not implemented yet - TDD first phase", allow_module_level=True)

from src.portfolio.base import Portfolio, Position, PositionType, PositionStatus
from src.rl_agent.base import MarketState
from src.utils.base import Chain


class TestTradingSafetyConfig:
    """Test TradingSafetyConfig configuration class."""
    
    def test_default_configuration(self):
        """Test default safety configuration values."""
        config = TradingSafetyConfig()
        
        # Position size limits
        assert config.max_position_size_pct == Decimal("0.1")  # 10%
        assert config.max_single_order_value == Decimal("10000")  # $10k
        assert config.min_order_value == Decimal("10")  # $10
        
        # Rate limiting
        assert config.max_orders_per_minute == 10
        assert config.max_orders_per_hour == 100
        assert config.rate_limit_cooldown_seconds == 60
        
        # Risk exposure
        assert config.max_total_exposure_pct == Decimal("0.8")  # 80%
        assert config.max_sector_concentration_pct == Decimal("0.3")  # 30%
        assert config.max_correlation_threshold == Decimal("0.85")
        
        # Balance requirements
        assert config.min_balance_reserve_pct == Decimal("0.05")  # 5%
        assert config.max_portfolio_utilization_pct == Decimal("0.95")  # 95%
    
    def test_production_configuration(self):
        """Test production-safe configuration."""
        config = TradingSafetyConfig.production_config()
        
        # Production should be more conservative
        assert config.max_position_size_pct <= Decimal("0.05")  # 5% max
        assert config.max_single_order_value <= Decimal("5000")  # $5k max
        assert config.max_orders_per_minute <= 5
        assert config.min_balance_reserve_pct >= Decimal("0.1")  # 10% reserve
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Valid configuration should not raise
        config = TradingSafetyConfig(
            max_position_size_pct=Decimal("0.05"),
            max_single_order_value=Decimal("1000")
        )
        assert config.max_position_size_pct == Decimal("0.05")
        
        # Invalid configuration should raise
        with pytest.raises(ValueError, match="Position size percentage must be between 0 and 1"):
            TradingSafetyConfig(max_position_size_pct=Decimal("1.5"))
        
        with pytest.raises(ValueError, match="Order value must be positive"):
            TradingSafetyConfig(max_single_order_value=Decimal("-100"))


class TestPreTradeValidationResult:
    """Test PreTradeValidationResult data structure."""
    
    def test_validation_result_creation(self):
        """Test creating validation results."""
        result = PreTradeValidationResult(
            is_valid=True,
            status=ValidationStatus.APPROVED,
            reasons=[],
            recommendations=["Trade approved for execution"]
        )
        
        assert result.is_valid is True
        assert result.status == ValidationStatus.APPROVED
        assert len(result.reasons) == 0
        assert "approved" in result.recommendations[0].lower()
    
    def test_validation_failure_result(self):
        """Test validation failure scenarios."""
        result = PreTradeValidationResult(
            is_valid=False,
            status=ValidationStatus.REJECTED,
            reasons=[ValidationReason.POSITION_SIZE_EXCEEDED, ValidationReason.INSUFFICIENT_BALANCE],
            recommendations=["Reduce position size", "Wait for balance increase"]
        )
        
        assert result.is_valid is False
        assert result.status == ValidationStatus.REJECTED
        assert ValidationReason.POSITION_SIZE_EXCEEDED in result.reasons
        assert ValidationReason.INSUFFICIENT_BALANCE in result.reasons
        assert len(result.recommendations) == 2


class TestTradingSafetyManager:
    """Test TradingSafetyManager core functionality."""
    
    @pytest.fixture
    def safety_config(self):
        """Create test safety configuration."""
        return TradingSafetyConfig(
            max_position_size_pct=Decimal("0.1"),
            max_single_order_value=Decimal("5000"),
            max_orders_per_minute=5,
            min_balance_reserve_pct=Decimal("0.05")
        )
    
    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio for testing."""
        portfolio = MagicMock(spec=Portfolio)
        portfolio.total_value = Decimal("100000")
        portfolio.available_balance = Decimal("20000")
        portfolio.positions = {}
        return portfolio
    
    @pytest.fixture
    def safety_manager(self, safety_config, mock_portfolio):
        """Create TradingSafetyManager instance."""
        return TradingSafetyManager(
            config=safety_config,
            portfolio=mock_portfolio
        )
    
    def test_safety_manager_initialization(self, safety_manager, safety_config, mock_portfolio):
        """Test safety manager initialization."""
        assert safety_manager.config == safety_config
        assert safety_manager.portfolio == mock_portfolio
        assert isinstance(safety_manager.rate_limit_state, dict)
        assert safety_manager.is_active is True
    
    @pytest.mark.asyncio
    async def test_position_size_validation_success(self, safety_manager):
        """Test successful position size validation."""
        # Test trade within position size limits
        trade_action = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("5000"),  # 5% of portfolio
            confidence=0.8
        )
        
        result = await safety_manager.validate_position_size(trade_action)
        
        assert result.is_valid is True
        assert result.status == ValidationStatus.APPROVED
        assert len(result.reasons) == 0
    
    @pytest.mark.asyncio
    async def test_position_size_validation_failure(self, safety_manager):
        """Test position size validation failure."""
        # Test trade exceeding position size limits
        trade_action = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("15000"),  # 15% of portfolio, exceeds 10% limit
            confidence=0.8
        )
        
        result = await safety_manager.validate_position_size(trade_action)
        
        assert result.is_valid is False
        assert result.status == ValidationStatus.REJECTED
        assert ValidationReason.POSITION_SIZE_EXCEEDED in result.reasons
        assert any("position size" in rec.lower() for rec in result.recommendations)
    
    @pytest.mark.asyncio
    async def test_order_value_validation(self, safety_manager):
        """Test order value validation."""
        # Test order within value limits
        valid_trade = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("3000"),  # Within $5000 limit
            confidence=0.8
        )
        
        result = await safety_manager.validate_order_value(valid_trade)
        assert result.is_valid is True
        
        # Test order exceeding value limits
        invalid_trade = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("7000"),  # Exceeds $5000 limit
            confidence=0.8
        )
        
        result = await safety_manager.validate_order_value(invalid_trade)
        assert result.is_valid is False
        assert ValidationReason.ORDER_VALUE_EXCEEDED in result.reasons
    
    @pytest.mark.asyncio
    async def test_rate_limiting_validation(self, safety_manager):
        """Test rate limiting per trading pair."""
        token_address = "0x123"
        trade_action = TradeOrder(
            token_address=token_address,
            action_type="buy",
            amount=Decimal("1000"),
            confidence=0.8
        )
        
        # First trades should pass
        for i in range(5):  # Max orders per minute is 5
            result = await safety_manager.validate_rate_limits(trade_action)
            assert result.is_valid is True
            # Simulate trade execution
            await safety_manager.record_trade_execution(trade_action)
        
        # Next trade should fail due to rate limiting
        result = await safety_manager.validate_rate_limits(trade_action)
        assert result.is_valid is False
        assert ValidationReason.RATE_LIMIT_EXCEEDED in result.reasons
    
    @pytest.mark.asyncio
    async def test_balance_verification(self, safety_manager, mock_portfolio):
        """Test balance verification."""
        # Set portfolio balance
        mock_portfolio.available_balance = Decimal("10000")
        mock_portfolio.total_value = Decimal("50000")  # 5% reserve = $2500
        
        # Test trade within available balance
        valid_trade = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("7000"),  # Leaves $3000, above $2500 reserve
            confidence=0.8
        )
        
        result = await safety_manager.validate_balance(valid_trade)
        assert result.is_valid is True
        
        # Test trade exceeding available balance
        invalid_trade = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("9000"),  # Leaves $1000, below $2500 reserve
            confidence=0.8
        )
        
        result = await safety_manager.validate_balance(invalid_trade)
        assert result.is_valid is False
        assert ValidationReason.INSUFFICIENT_BALANCE in result.reasons
    
    @pytest.mark.asyncio
    async def test_risk_exposure_validation(self, safety_manager, mock_portfolio):
        """Test risk exposure validation."""
        # Mock existing positions
        existing_positions = {
            "ETH/USDC": Position(
                position_id=uuid4(),
                symbol="ETH/USDC",
                chain=Chain.ETHEREUM,
                position_type=PositionType.SPOT,
                dex_name="uniswap_v3",
                size=Decimal("30000"),  # 30% of portfolio
                entry_price=Decimal("100"),
                current_price=Decimal("105"),
                status=PositionStatus.OPEN
            )
        }
        mock_portfolio.positions = existing_positions
        mock_portfolio.total_value = Decimal("100000")
        
        # Test trade that would exceed total exposure limit (80%)
        trade_action = TradeOrder(
            token_address="0x222",
            action_type="buy",
            amount=Decimal("55000"),  # Would create 55% + 30% = 85% exposure
            confidence=0.8
        )
        
        result = await safety_manager.validate_risk_exposure(trade_action)
        assert result.is_valid is False
        assert ValidationReason.RISK_EXPOSURE_EXCEEDED in result.reasons
    
    @pytest.mark.asyncio
    async def test_comprehensive_pre_trade_validation(self, safety_manager):
        """Test comprehensive pre-trade validation pipeline."""
        from src.discovery.base import DiscoveredToken
        from src.utils.base import Chain
        
        trade_action = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("3000"),
            confidence=0.8
        )
        
        # Create proper MarketState
        token = DiscoveredToken(
            address="0x123",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.utcnow(),
            discovery_source="test",
            price_usd=50.0,
            volume_24h=1000000.0,
            price_change_24h=-5.0,
            market_cap=50000000.0
        )
        market_state = MarketState(
            token=token,
            price_usd=50.0,
            rsi=30.0,
            volume_24h=1000000.0,
            price_change_24h=-5.0,
            timestamp=datetime.utcnow()
        )
        
        result = await safety_manager.validate_pre_trade(trade_action, market_state)
        
        # Should validate all aspects
        assert isinstance(result, PreTradeValidationResult)
        if result.is_valid:
            assert result.status == ValidationStatus.APPROVED
        else:
            assert result.status in [ValidationStatus.REJECTED, ValidationStatus.CONDITIONAL]
            assert len(result.reasons) > 0
            assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_emergency_stop_functionality(self, safety_manager):
        """Test emergency stop mechanism."""
        # Activate emergency stop
        await safety_manager.activate_emergency_stop("Testing emergency stop")
        
        assert safety_manager.is_emergency_stopped is True
        
        # All trade validations should fail during emergency stop
        trade_action = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("1000"),
            confidence=0.8
        )
        
        result = await safety_manager.validate_pre_trade(trade_action)
        assert result.is_valid is False
        assert ValidationReason.EMERGENCY_STOP_ACTIVE in result.reasons
        
        # Deactivate emergency stop
        await safety_manager.deactivate_emergency_stop()
        assert safety_manager.is_emergency_stopped is False
    
    @pytest.mark.asyncio
    async def test_safety_metrics_tracking(self, safety_manager):
        """Test safety metrics tracking and reporting."""
        # Execute several validation scenarios
        trades = [
            TradeOrder("0x123", "buy", Decimal("1000"), 0.8),  # Should pass
            TradeOrder("0x124", "buy", Decimal("15000"), 0.8),  # Should fail - size
            TradeOrder("0x125", "buy", Decimal("7000"), 0.8),   # Should fail - value
        ]
        
        for trade in trades:
            await safety_manager.validate_pre_trade(trade)
        
        metrics = safety_manager.get_safety_metrics()
        
        assert "total_validations" in metrics
        assert "approvals" in metrics
        assert "rejections" in metrics
        assert "rejection_reasons" in metrics
        assert metrics["total_validations"] == 3
        assert metrics["rejections"] >= 2  # At least 2 should have failed
    
    def test_rate_limit_state_management(self, safety_manager):
        """Test rate limit state tracking."""
        token_address = "0x123"
        
        # Initially no rate limit state
        assert token_address not in safety_manager.rate_limit_state
        
        # Record trades and check state
        now = datetime.utcnow()
        safety_manager._update_rate_limit_state(token_address, now)
        
        assert token_address in safety_manager.rate_limit_state
        state = safety_manager.rate_limit_state[token_address]
        assert isinstance(state, RateLimitState)
        assert len(state.recent_trades) >= 0
    
    @pytest.mark.asyncio
    async def test_integration_with_portfolio_manager(self, safety_manager, mock_portfolio):
        """Test integration with portfolio management system."""
        # Mock portfolio methods that safety manager might call
        mock_portfolio.get_position = AsyncMock(return_value=None)
        mock_portfolio.calculate_total_exposure = AsyncMock(return_value=Decimal("0.3"))
        
        trade_action = TradeOrder(
            token_address="0x123",
            action_type="buy",
            amount=Decimal("5000"),
            confidence=0.8
        )
        
        result = await safety_manager.validate_pre_trade(trade_action)
        
        # Should have called portfolio methods for comprehensive validation
        assert isinstance(result, PreTradeValidationResult)
        # Exact validation depends on implementation, but should complete without error


class TestValidationEnums:
    """Test validation enum types."""
    
    def test_validation_status_enum(self):
        """Test ValidationStatus enum."""
        assert ValidationStatus.APPROVED.value == "approved"
        assert ValidationStatus.REJECTED.value == "rejected"
        assert ValidationStatus.CONDITIONAL.value == "conditional"
        assert ValidationStatus.PENDING.value == "pending"
    
    def test_validation_reason_enum(self):
        """Test ValidationReason enum."""
        reasons = [
            ValidationReason.POSITION_SIZE_EXCEEDED,
            ValidationReason.ORDER_VALUE_EXCEEDED,
            ValidationReason.RATE_LIMIT_EXCEEDED,
            ValidationReason.INSUFFICIENT_BALANCE,
            ValidationReason.RISK_EXPOSURE_EXCEEDED,
            ValidationReason.EMERGENCY_STOP_ACTIVE,
            ValidationReason.MARKET_CONDITIONS,
            ValidationReason.SYSTEM_ERROR
        ]
        
        for reason in reasons:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0


class TestPositionSizeValidation:
    """Test position size validation utilities."""
    
    def test_position_size_calculator(self):
        """Test position size calculation."""
        portfolio_value = Decimal("100000")
        max_position_pct = Decimal("0.1")
        
        validation = PositionSizeValidation(
            portfolio_value=portfolio_value,
            max_position_pct=max_position_pct
        )
        
        max_size = validation.calculate_max_position_size()
        assert max_size == Decimal("10000")  # 10% of $100k
        
        # Test validation
        assert validation.is_size_valid(Decimal("5000")) is True
        assert validation.is_size_valid(Decimal("15000")) is False


class TestRiskExposureValidation:
    """Test risk exposure validation utilities."""
    
    def test_risk_exposure_calculator(self):
        """Test risk exposure calculation."""
        existing_exposure = Decimal("0.3")  # 30%
        max_total_exposure = Decimal("0.8")  # 80%
        
        validation = RiskExposureValidation(
            current_exposure=existing_exposure,
            max_total_exposure=max_total_exposure
        )
        
        remaining_capacity = validation.calculate_remaining_capacity()
        assert remaining_capacity == Decimal("0.5")  # 50% remaining
        
        # Test validation
        assert validation.is_exposure_valid(Decimal("0.4")) is True   # 30% + 40% = 70%
        assert validation.is_exposure_valid(Decimal("0.6")) is False  # 30% + 60% = 90%


@pytest.mark.integration
class TestTradingSafetyIntegration:
    """Integration tests for trading safety manager."""
    
    @pytest.mark.asyncio
    async def test_integration_with_live_mode(self):
        """Test integration with live trading mode."""
        # This test will be implemented once LiveMode integration is complete
        pytest.skip("Integration test requires LiveMode implementation")
    
    @pytest.mark.asyncio 
    async def test_integration_with_simulation_mode(self):
        """Test integration with simulation trading mode."""
        # This test will be implemented once SimulationMode integration is complete
        pytest.skip("Integration test requires SimulationMode implementation")
    
    @pytest.mark.asyncio
    async def test_concurrent_validation_performance(self):
        """Test performance under concurrent validation requests."""
        # This test will validate performance under load
        pytest.skip("Performance test requires full implementation")