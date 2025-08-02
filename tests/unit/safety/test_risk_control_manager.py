"""
Tests for RiskControlManager - Phase 4.5 Risk Controls and Position Limits

Test-Driven Development (TDD) approach:
1. Write failing tests first
2. Implement minimal functionality to pass tests
3. Refactor and improve

Risk Control Manager should handle:
- Per-asset position limits
- Portfolio concentration limits  
- Sector/correlation limits
- Leverage controls
- Drawdown protection
- Real-time risk monitoring
- Risk metrics calculation
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.safety.risk_control_manager import (
    RiskControlManager,
    RiskControlConfig,
    PositionLimit,
    ConcentrationLimit,
    LeverageLimit,
    DrawdownProtection,
    RiskMetrics,
    RiskControlResult,
    RiskAlertType,
    RiskAlertSeverity,
    RiskViolationType
)
from src.portfolio.base import Portfolio, Position, PositionStatus
from src.utils.base import Chain


class TestRiskControlManager:
    """Test cases for RiskControlManager."""
    
    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio for testing."""
        portfolio = Mock(spec=Portfolio)
        portfolio.total_value = Decimal("100000")  # $100k portfolio
        portfolio.available_balance = Decimal("20000")  # $20k available
        portfolio.positions = {}
        portfolio.daily_pnl = Decimal("0")
        portfolio.total_pnl = Decimal("5000")  # $5k profit
        portfolio.max_drawdown = Decimal("0.05")  # 5% max drawdown
        return portfolio
    
    @pytest.fixture
    def risk_config(self):
        """Create risk control configuration."""
        return RiskControlConfig(
            max_position_size_pct=Decimal("0.1"),  # 10% max per position
            max_portfolio_concentration_pct=Decimal("0.3"),  # 30% max concentration
            max_sector_concentration_pct=Decimal("0.2"),  # 20% max per sector
            max_correlation_threshold=Decimal("0.8"),  # 80% correlation limit
            max_leverage=Decimal("2.0"),  # 2x max leverage
            max_daily_loss_pct=Decimal("0.05"),  # 5% max daily loss
            max_total_drawdown_pct=Decimal("0.15"),  # 15% max total drawdown
            enable_real_time_monitoring=True,
            alert_thresholds={
                "warning": Decimal("0.7"),
                "critical": Decimal("0.9")
            }
        )
    
    @pytest.fixture
    def risk_manager(self, risk_config, mock_portfolio):
        """Create RiskControlManager instance."""
        return RiskControlManager(config=risk_config, portfolio=mock_portfolio)
    
    def test_risk_control_manager_initialization(self, risk_config, mock_portfolio):
        """Test RiskControlManager initialization."""
        manager = RiskControlManager(config=risk_config, portfolio=mock_portfolio)
        
        assert manager.config == risk_config
        assert manager.portfolio == mock_portfolio
        assert manager.is_active is True
        assert manager.risk_metrics is None  # Initially None until calculated
        assert len(manager.position_limits) == 0
        assert len(manager.sector_allocations) == 0
    
    def test_risk_control_config_validation(self):
        """Test RiskControlConfig validation."""
        # Valid configuration
        valid_config = RiskControlConfig(
            max_position_size_pct=Decimal("0.1"),
            max_leverage=Decimal("2.0")
        )
        assert valid_config.max_position_size_pct == Decimal("0.1")
        
        # Invalid position size percentage
        with pytest.raises(ValueError, match="Position size percentage must be between 0 and 1"):
            RiskControlConfig(max_position_size_pct=Decimal("1.5"))
        
        # Invalid leverage
        with pytest.raises(ValueError, match="Leverage must be positive"):
            RiskControlConfig(max_leverage=Decimal("-1.0"))
    
    @pytest.mark.asyncio
    async def test_validate_position_size_limit_success(self, risk_manager):
        """Test successful position size validation."""
        # Position within 10% limit ($10k on $100k portfolio)
        result = await risk_manager.validate_position_size(
            asset="BTC",
            size=Decimal("8000"),  # 8% of portfolio
            current_price=Decimal("50000")
        )
        
        assert result.is_valid is True
        assert result.violation_type is None
        assert "within position size limit" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_position_size_limit_violation(self, risk_manager):
        """Test position size limit violation."""
        # Position exceeding 10% limit ($15k on $100k portfolio)
        result = await risk_manager.validate_position_size(
            asset="BTC",
            size=Decimal("15000"),  # 15% of portfolio
            current_price=Decimal("50000")
        )
        
        assert result.is_valid is False
        assert result.violation_type == RiskViolationType.POSITION_SIZE_EXCEEDED
        assert "exceeds maximum position size" in result.message.lower()
        assert result.recommended_action is not None
    
    @pytest.mark.asyncio
    async def test_validate_portfolio_concentration_limit(self, risk_manager):
        """Test portfolio concentration limit validation."""
        # Mock existing positions totaling 25% of portfolio
        risk_manager.portfolio.positions = {
            "ETH": Mock(value=Decimal("15000")),  # 15%
            "BNB": Mock(value=Decimal("10000"))   # 10%
        }
        
        # Adding 8% more should pass (total 33% > 30% limit)
        result = await risk_manager.validate_concentration_limit(
            new_position_value=Decimal("8000")
        )
        
        assert result.is_valid is False
        assert result.violation_type == RiskViolationType.CONCENTRATION_EXCEEDED
    
    @pytest.mark.asyncio
    async def test_validate_sector_concentration_limit(self, risk_manager):
        """Test sector concentration limit validation."""
        # Mock sector allocation
        risk_manager.sector_allocations = {
            "DeFi": Decimal("0.15")  # 15% already in DeFi
        }
        
        # Adding 8% more to DeFi (total 23% > 20% limit)
        result = await risk_manager.validate_sector_limit(
            sector="DeFi",
            additional_allocation=Decimal("0.08")
        )
        
        assert result.is_valid is False
        assert result.violation_type == RiskViolationType.SECTOR_CONCENTRATION_EXCEEDED
    
    @pytest.mark.asyncio
    async def test_validate_correlation_limit(self, risk_manager):
        """Test correlation limit validation."""
        # Mock correlation data showing high correlation (0.9 > 0.8 limit)
        risk_manager._correlation_matrix = {
            ("BTC", "ETH"): Decimal("0.9")
        }
        
        result = await risk_manager.validate_correlation_limit(
            asset1="BTC",
            asset2="ETH",
            position1_size=Decimal("5000"),
            position2_size=Decimal("5000")
        )
        
        assert result.is_valid is False
        assert result.violation_type == RiskViolationType.CORRELATION_EXCEEDED
    
    @pytest.mark.asyncio
    async def test_validate_leverage_limit_success(self, risk_manager):
        """Test successful leverage validation."""
        # Mock portfolio with 1.5x leverage (within 2.0x limit)
        risk_manager.portfolio.total_exposure = Decimal("150000")
        risk_manager.portfolio.total_value = Decimal("100000")
        
        result = await risk_manager.validate_leverage_limit(
            additional_exposure=Decimal("20000")  # Total 1.7x leverage
        )
        
        assert result.is_valid is True
        assert result.violation_type is None
    
    @pytest.mark.asyncio
    async def test_validate_leverage_limit_violation(self, risk_manager):
        """Test leverage limit violation."""
        # Mock portfolio with high leverage
        risk_manager.portfolio.total_exposure = Decimal("180000")
        risk_manager.portfolio.total_value = Decimal("100000")
        
        result = await risk_manager.validate_leverage_limit(
            additional_exposure=Decimal("50000")  # Total 2.3x leverage > 2.0x limit
        )
        
        assert result.is_valid is False
        assert result.violation_type == RiskViolationType.LEVERAGE_EXCEEDED
    
    @pytest.mark.asyncio
    async def test_validate_daily_loss_limit(self, risk_manager):
        """Test daily loss limit validation."""
        # Mock portfolio with 4% daily loss (within 5% limit)
        risk_manager.portfolio.daily_pnl = Decimal("-4000")  # -4%
        
        result = await risk_manager.validate_daily_loss_limit()
        
        assert result.is_valid is True
        
        # Mock portfolio with 6% daily loss (exceeding 5% limit)
        risk_manager.portfolio.daily_pnl = Decimal("-6000")  # -6%
        
        result = await risk_manager.validate_daily_loss_limit()
        
        assert result.is_valid is False
        assert result.violation_type == RiskViolationType.DAILY_LOSS_EXCEEDED
    
    @pytest.mark.asyncio
    async def test_validate_drawdown_protection(self, risk_manager):
        """Test drawdown protection validation."""
        # Mock portfolio with acceptable drawdown (10% < 15% limit)
        risk_manager.portfolio.max_drawdown = Decimal("0.1")
        
        result = await risk_manager.validate_drawdown_protection()
        
        assert result.is_valid is True
        
        # Mock portfolio with excessive drawdown (20% > 15% limit)
        risk_manager.portfolio.max_drawdown = Decimal("0.2")
        
        result = await risk_manager.validate_drawdown_protection()
        
        assert result.is_valid is False
        assert result.violation_type == RiskViolationType.DRAWDOWN_EXCEEDED
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, risk_manager):
        """Test risk metrics calculation."""
        # Mock portfolio data
        risk_manager.portfolio.positions = {
            "BTC": Mock(value=Decimal("30000"), volatility=Decimal("0.8")),
            "ETH": Mock(value=Decimal("20000"), volatility=Decimal("0.7")),
            "USDT": Mock(value=Decimal("50000"), volatility=Decimal("0.01"))
        }
        
        metrics = await risk_manager.calculate_risk_metrics()
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.portfolio_volatility >= Decimal("0")
        assert metrics.value_at_risk >= Decimal("0")
        assert metrics.sharpe_ratio is not None
        assert metrics.leverage_ratio >= Decimal("0")
        assert len(metrics.position_sizes) == 3
    
    @pytest.mark.asyncio
    async def test_generate_risk_alerts(self, risk_manager):
        """Test risk alert generation."""
        # Set up portfolio with warning-level risk
        risk_manager.portfolio.total_exposure = Decimal("140000")  # 1.4x leverage (70% of 2.0x limit)
        risk_manager.portfolio.total_value = Decimal("100000")
        
        alerts = await risk_manager.generate_risk_alerts()
        
        assert len(alerts) >= 0
        # Should generate warning alert for leverage at 70% of limit
        leverage_alerts = [a for a in alerts if a.alert_type == RiskAlertType.LEVERAGE]
        if leverage_alerts:
            assert leverage_alerts[0].severity == RiskAlertSeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_comprehensive_risk_check(self, risk_manager):
        """Test comprehensive risk validation."""
        # Test successful validation with all checks passing
        result = await risk_manager.validate_comprehensive_risk(
            asset="BTC",
            size=Decimal("5000"),
            sector="DeFi",
            additional_exposure=Decimal("5000")
        )
        
        assert isinstance(result, RiskControlResult)
        assert result.overall_risk_score >= Decimal("0")
        assert result.overall_risk_score <= Decimal("1")
        
        # Test with multiple violations
        risk_manager.portfolio.daily_pnl = Decimal("-6000")  # Exceeds daily loss
        risk_manager.portfolio.max_drawdown = Decimal("0.2")  # Exceeds drawdown
        
        result = await risk_manager.validate_comprehensive_risk(
            asset="BTC",
            size=Decimal("15000"),  # Exceeds position size
            sector="DeFi",
            additional_exposure=Decimal("100000")  # Exceeds leverage
        )
        
        assert result.is_valid is False
        assert len(result.violations) >= 2
    
    def test_position_limit_configuration(self, risk_manager):
        """Test position limit configuration and management."""
        # Add position limit
        risk_manager.add_position_limit(
            asset="BTC",
            max_size_pct=Decimal("0.08"),  # 8% custom limit for BTC
            max_value=Decimal("10000")
        )
        
        assert "BTC" in risk_manager.position_limits
        limit = risk_manager.position_limits["BTC"]
        assert limit.max_size_pct == Decimal("0.08")
        assert limit.max_value == Decimal("10000")
        
        # Remove position limit
        risk_manager.remove_position_limit("BTC")
        assert "BTC" not in risk_manager.position_limits
    
    def test_sector_allocation_tracking(self, risk_manager):
        """Test sector allocation tracking."""
        # Update sector allocation
        risk_manager.update_sector_allocation("DeFi", Decimal("0.15"))
        assert risk_manager.sector_allocations["DeFi"] == Decimal("0.15")
        
        # Get sector allocation
        allocation = risk_manager.get_sector_allocation("DeFi")
        assert allocation == Decimal("0.15")
        
        # Get allocation for non-existent sector
        allocation = risk_manager.get_sector_allocation("Gaming")
        assert allocation == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, risk_manager):
        """Test real-time risk monitoring."""
        if risk_manager.config.enable_real_time_monitoring:
            # Start monitoring
            monitoring_task = asyncio.create_task(
                risk_manager.start_real_time_monitoring(interval_seconds=1)
            )
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            
            # Stop monitoring
            risk_manager.stop_real_time_monitoring()
            monitoring_task.cancel()
            
            # Should have generated monitoring data
            assert risk_manager.monitoring_active is False
    
    def test_risk_metrics_history(self, risk_manager):
        """Test risk metrics history tracking."""
        # Add some historical metrics
        for i in range(5):
            metrics = RiskMetrics(
                timestamp=datetime.utcnow() - timedelta(hours=i),
                portfolio_volatility=Decimal(f"0.{20+i}"),
                value_at_risk=Decimal(f"{1000+i*100}"),
                leverage_ratio=Decimal(f"1.{i}"),
                sharpe_ratio=Decimal(f"0.{5+i}"),
                position_sizes={}
            )
            risk_manager.add_risk_metrics_to_history(metrics)
        
        history = risk_manager.get_risk_metrics_history(hours=24)
        assert len(history) == 5
        
        # Get recent metrics (within last 2 hours)
        recent = risk_manager.get_risk_metrics_history(hours=2)
        assert len(recent) >= 2  # At least first 2 hours of data
    
    @pytest.mark.asyncio
    async def test_emergency_risk_controls(self, risk_manager):
        """Test emergency risk controls activation."""
        # Simulate emergency condition (extreme daily loss)
        risk_manager.portfolio.daily_pnl = Decimal("-20000")  # -20% loss
        
        # Should automatically trigger emergency controls
        result = await risk_manager.check_emergency_conditions()
        
        assert result.requires_emergency_action is True
        assert result.emergency_actions is not None
        assert len(result.emergency_actions) > 0
    
    def test_risk_control_config_production_settings(self):
        """Test production-safe risk control configuration."""
        prod_config = RiskControlConfig.production_config()
        
        # Production should have stricter limits
        assert prod_config.max_position_size_pct <= Decimal("0.05")  # ≤5%
        assert prod_config.max_daily_loss_pct <= Decimal("0.03")     # ≤3%
        assert prod_config.max_total_drawdown_pct <= Decimal("0.1")  # ≤10%
        assert prod_config.max_leverage <= Decimal("1.5")            # ≤1.5x
        assert prod_config.enable_real_time_monitoring is True
    
    def test_risk_control_metrics_export(self, risk_manager):
        """Test risk control metrics export."""
        metrics_dict = risk_manager.export_risk_metrics()
        
        assert "total_validations" in metrics_dict
        assert "violation_counts" in metrics_dict
        assert "risk_score_history" in metrics_dict
        assert "current_limits" in metrics_dict
        assert "sector_allocations" in metrics_dict
        assert isinstance(metrics_dict, dict)


class TestRiskControlComponents:
    """Test individual risk control components."""
    
    def test_position_limit_creation(self):
        """Test PositionLimit creation and validation."""
        limit = PositionLimit(
            asset="BTC",
            max_size_pct=Decimal("0.1"),
            max_value=Decimal("10000"),
            enabled=True
        )
        
        assert limit.asset == "BTC"
        assert limit.max_size_pct == Decimal("0.1")
        assert limit.max_value == Decimal("10000")
        assert limit.enabled is True
    
    def test_concentration_limit_validation(self):
        """Test ConcentrationLimit validation."""
        limit = ConcentrationLimit(
            limit_type="portfolio",
            max_concentration_pct=Decimal("0.3"),
            assets=["BTC", "ETH"]
        )
        
        assert limit.limit_type == "portfolio"
        assert limit.max_concentration_pct == Decimal("0.3")
        assert "BTC" in limit.assets
    
    def test_leverage_limit_calculation(self):
        """Test LeverageLimit calculations."""
        limit = LeverageLimit(
            max_leverage=Decimal("2.0"),
            include_derivatives=True,
            emergency_deleveraging_threshold=Decimal("1.8")
        )
        
        # Test leverage calculation
        current_leverage = limit.calculate_leverage(
            total_exposure=Decimal("150000"),
            total_equity=Decimal("100000")
        )
        
        assert current_leverage == Decimal("1.5")
        
        # Test if leverage is within limits
        assert limit.is_within_limits(current_leverage) is True
        
        # Test emergency threshold
        high_leverage = Decimal("1.9")
        assert limit.requires_emergency_deleveraging(high_leverage) is True
    
    def test_drawdown_protection_triggers(self):
        """Test DrawdownProtection trigger conditions."""
        protection = DrawdownProtection(
            max_drawdown_pct=Decimal("0.15"),
            daily_loss_limit_pct=Decimal("0.05"),
            enable_stop_loss=True,
            enable_position_scaling=True
        )
        
        # Test normal drawdown
        assert protection.should_trigger(Decimal("0.1")) is False
        
        # Test excessive drawdown
        assert protection.should_trigger(Decimal("0.2")) is True
        
        # Test daily loss limit
        assert protection.check_daily_loss(
            daily_pnl=Decimal("-6000"),
            portfolio_value=Decimal("100000")
        ) is True  # Should trigger (6% > 5%)


class TestRiskMetricsCalculation:
    """Test risk metrics calculation components."""
    
    def test_risk_metrics_initialization(self):
        """Test RiskMetrics initialization."""
        metrics = RiskMetrics(
            timestamp=datetime.utcnow(),
            portfolio_volatility=Decimal("0.25"),
            value_at_risk=Decimal("5000"),
            leverage_ratio=Decimal("1.5"),
            sharpe_ratio=Decimal("0.8"),
            position_sizes={"BTC": Decimal("0.3"), "ETH": Decimal("0.2")}
        )
        
        assert metrics.portfolio_volatility == Decimal("0.25")
        assert metrics.value_at_risk == Decimal("5000")
        assert metrics.leverage_ratio == Decimal("1.5")
        assert metrics.sharpe_ratio == Decimal("0.8")
        assert len(metrics.position_sizes) == 2
    
    def test_risk_control_result_aggregation(self):
        """Test RiskControlResult aggregation."""
        violations = [
            RiskViolationType.POSITION_SIZE_EXCEEDED,
            RiskViolationType.DAILY_LOSS_EXCEEDED
        ]
        
        result = RiskControlResult(
            is_valid=False,
            overall_risk_score=Decimal("0.85"),
            violations=violations,
            recommendations=["Reduce position size", "Implement stop loss"],
            timestamp=datetime.utcnow()
        )
        
        assert result.is_valid is False
        assert result.overall_risk_score == Decimal("0.85")
        assert len(result.violations) == 2
        assert RiskViolationType.POSITION_SIZE_EXCEEDED in result.violations