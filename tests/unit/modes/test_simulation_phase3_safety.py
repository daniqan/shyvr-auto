"""
Test suite for SimulationMode Phase 3 - Safety Systems Integration.

Tests the integration of simulation-specific safety systems with virtual
portfolio protection mechanisms and risk monitoring adapted for simulation.

Following TDD methodology - tests written before implementation.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.modes.base import ModeType, ModeStatus, ModeConfig
from src.portfolio.base import Portfolio, PortfolioConfig, Position, PositionType, PositionStatus
from src.rl_agent.base import MarketState, TradeAction
from src.utils.base import Chain
from src.modes.simulation_mode import SimulationMode


def create_mock_token():
    """Helper function to create mock DiscoveredToken."""
    from src.discovery.base import DiscoveredToken, TokenStatus
    from src.utils.base import Chain
    
    return DiscoveredToken(
        address="TEST_TOKEN_123",
        name="Test Token",
        symbol="TEST",
        chain=Chain.SOLANA,
        discovered_at=datetime.now(),
        discovery_source="test",
        status=TokenStatus.VALIDATED
    )


class TestSimulationSafetyIntegration:
    """Test suite for simulation-specific safety systems."""
    
    @pytest.fixture
    def portfolio(self):
        """Portfolio instance for testing."""
        config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC"
        )
        return Portfolio(
            portfolio_id=uuid4(),
            name="Test Portfolio",
            config=config,
            cash_balance=Decimal("10000"),
            total_value=Decimal("10000")
        )
    
    @pytest.fixture
    def simulation_config(self):
        """Simulation mode configuration with safety features."""
        return ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "initial_balance": 10000,
                "enable_fees": True,
                "slippage_bps": 10,
                "enable_simulation_safety": True,
                "max_simulation_drawdown_pct": 0.20,  # More lenient than live
                "max_simulation_position_size_pct": 0.15,  # Slightly higher than live
                "enable_virtual_portfolio_protection": True,
                "simulation_risk_monitoring_interval": 2.0,
                "enable_simulation_circuit_breakers": True
            }
        )
    
    @pytest.fixture
    def simulation_mode_with_safety(self, simulation_config, portfolio):
        """SimulationMode instance with safety systems enabled."""
        return SimulationMode(
            mode_id=uuid4(),
            config=simulation_config,
            portfolio=portfolio
        )
    
    async def test_simulation_safety_initialization(self, simulation_mode_with_safety):
        """Test initialization of simulation-specific safety systems."""
        mode = simulation_mode_with_safety
        await mode.initialize()
        
        # Should initialize simulation safety manager
        assert hasattr(mode, 'simulation_safety_manager')
        assert mode.simulation_safety_manager is not None
        
        # Should have simulation-specific risk thresholds
        safety_config = mode.simulation_safety_config
        assert safety_config['max_drawdown_pct'] == 0.20  # More lenient than live trading
        assert safety_config['max_position_size_pct'] == 0.15
        assert safety_config['enable_circuit_breakers'] is True
        
        # Should have virtual portfolio protection
        assert hasattr(mode, 'virtual_portfolio_protector')
        assert mode.virtual_portfolio_protector is not None
    
    async def test_virtual_portfolio_protection_mechanisms(self, simulation_mode_with_safety):
        """Test virtual portfolio protection mechanisms."""
        mode = simulation_mode_with_safety
        await mode.initialize()
        await mode.start()
        
        # Mock virtual portfolio protector
        mode.virtual_portfolio_protector = Mock()
        mode.virtual_portfolio_protector.validate_trade_safety = AsyncMock(
            return_value={'is_safe': True, 'risk_level': 'low'}
        )
        mode.virtual_portfolio_protector.check_portfolio_health = AsyncMock(
            return_value={'health_status': 'good', 'protection_active': False}
        )
        
        # Execute a trade
        mock_token = create_mock_token()
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0,
            rsi=25.0
        )
        
        action = await mode.process_tick(market_state)
        
        # Should validate trade safety
        mode.virtual_portfolio_protector.validate_trade_safety.assert_called_once()
        
        # Should monitor portfolio health
        mode.virtual_portfolio_protector.check_portfolio_health.assert_called_once()
    
    async def test_simulation_specific_risk_limits(self, simulation_mode_with_safety):
        """Test simulation-specific risk limits (more lenient than live trading)."""
        mode = simulation_mode_with_safety
        await mode.initialize()
        
        # Test position size limit (15% for simulation vs 10% for live)
        large_position_size = Decimal("1400")  # 14% of portfolio
        
        risk_check = await mode.simulation_safety_manager.validate_position_size(
            "BTC_ADDRESS", large_position_size
        )
        
        # Should allow larger positions in simulation
        assert risk_check.is_valid is True
        assert risk_check.risk_level in ['low', 'medium']
        
        # Test very large position (exceeds even simulation limits)
        oversized_position = Decimal("1800")  # 18% of portfolio
        
        risk_check = await mode.simulation_safety_manager.validate_position_size(
            "BTC_ADDRESS", oversized_position
        )
        
        # Should reject positions exceeding simulation limits
        assert risk_check.is_valid is False
        assert "position size exceeds" in risk_check.reason.lower()
    
    async def test_simulation_drawdown_monitoring(self, simulation_mode_with_safety):
        """Test simulation-specific drawdown monitoring."""
        mode = simulation_mode_with_safety
        await mode.initialize()
        
        # Simulate portfolio drawdown
        mode.virtual_portfolio.peak_value = Decimal("12000")
        mode.virtual_portfolio.current_value = Decimal("9800")  # 18.3% drawdown
        
        # Should allow higher drawdown in simulation (20% vs 15% for live)
        drawdown_check = await mode.simulation_safety_manager.check_max_drawdown()
        assert drawdown_check.is_valid is True
        
        # Exceed simulation drawdown limit
        mode.virtual_portfolio.current_value = Decimal("9000")  # 25% drawdown
        
        drawdown_check = await mode.simulation_safety_manager.check_max_drawdown()
        assert drawdown_check.is_valid is False
        assert "drawdown" in drawdown_check.reason.lower()
    
    async def test_simulation_circuit_breakers(self, simulation_mode_with_safety):
        """Test simulation-specific circuit breakers."""
        mode = simulation_mode_with_safety
        await mode.initialize()
        await mode.start()
        
        # Mock circuit breaker system
        mode.simulation_circuit_breaker = Mock()
        mode.simulation_circuit_breaker.check_trading_halt_conditions = AsyncMock(
            return_value={'halt_trading': False, 'reason': None}
        )
        
        # Simulate extreme market conditions
        mock_token = create_mock_token()
        extreme_market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=-25.0,  # Extreme volatility
            volume_24h=50000000.0,   # High volume
            rsi=10.0                 # Extremely oversold
        )
        
        action = await mode.process_tick(extreme_market_state)
        
        # Should check circuit breaker conditions
        mode.simulation_circuit_breaker.check_trading_halt_conditions.assert_called_once()
    
    async def test_virtual_stop_loss_protection(self, simulation_mode_with_safety):
        """Test virtual stop-loss protection mechanisms."""
        mode = simulation_mode_with_safety
        await mode.initialize()
        
        # Add position with loss
        position = Position(
            position_id=uuid4(),
            symbol="BTC/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("0.01"),
            entry_price=Decimal("100000"),
            current_price=Decimal("90000"),  # 10% loss
            status=PositionStatus.OPEN
        )
        
        mode.virtual_portfolio.add_position(position)
        
        # Check stop loss triggers
        stop_loss_positions = await mode.simulation_safety_manager.check_stop_losses()
        
        # Should trigger stop loss (assuming 8% threshold)
        assert len(stop_loss_positions) == 1
        assert stop_loss_positions[0].position_id == position.position_id
    
    async def test_simulation_emergency_stop(self, simulation_mode_with_safety):
        """Test simulation-specific emergency stop procedures."""
        mode = simulation_mode_with_safety
        await mode.initialize()
        await mode.start()
        
        # Mock emergency stop system
        mode.simulation_emergency_stop = Mock()
        mode.simulation_emergency_stop.trigger_emergency_stop = AsyncMock(return_value=True)
        mode.simulation_emergency_stop.is_emergency_stop_active = Mock(return_value=False)
        
        # Trigger emergency condition (extreme drawdown)
        mode.virtual_portfolio.peak_value = Decimal("12000")
        mode.virtual_portfolio.current_value = Decimal("8000")  # 33% drawdown
        
        # Should trigger emergency stop
        emergency_triggered = await mode.simulation_safety_manager.check_emergency_conditions()
        
        assert emergency_triggered['emergency_stop_required'] is True
        assert 'extreme_drawdown' in emergency_triggered['reasons']
    
    async def test_safety_metrics_collection(self, simulation_mode_with_safety):
        """Test collection of simulation safety metrics."""
        mode = simulation_mode_with_safety
        await mode.initialize()
        await mode.start()
        
        # Generate some trading activity
        mock_token = create_mock_token()
        
        for i in range(5):
            market_state = MarketState(
                token=mock_token,
                price_usd=100.0 + i,
                price_change_24h=5.0,
                volume_24h=1000000.0,
                rsi=30.0 + i * 5
            )
            await mode.process_tick(market_state)
        
        # Get safety metrics
        safety_metrics = await mode.get_simulation_safety_metrics()
        
        # Should track safety-related metrics
        assert 'risk_violations_count' in safety_metrics
        assert 'emergency_stops_triggered' in safety_metrics
        assert 'circuit_breaker_activations' in safety_metrics
        assert 'virtual_portfolio_health_score' in safety_metrics
        assert 'max_observed_drawdown' in safety_metrics
        assert 'position_risk_distribution' in safety_metrics
    
    async def test_safety_alert_system(self, simulation_mode_with_safety):
        """Test simulation safety alert system."""
        mode = simulation_mode_with_safety
        await mode.initialize()
        
        # Mock alert system
        mode.simulation_alert_system = Mock()
        mode.simulation_alert_system.generate_risk_alert = AsyncMock()
        mode.simulation_alert_system.get_active_alerts = Mock(return_value=[])
        
        # Create risk condition
        large_position_size = Decimal("1600")  # 16% of portfolio
        
        risk_check = await mode.simulation_safety_manager.validate_position_size(
            "BTC_ADDRESS", large_position_size
        )
        
        # Should generate appropriate alerts for risk conditions
        if not risk_check.is_valid:
            mode.simulation_alert_system.generate_risk_alert.assert_called_once()
    
    async def test_safety_configuration_validation(self, portfolio):
        """Test validation of simulation safety configuration."""
        # Invalid safety configuration
        invalid_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "enable_simulation_safety": True,
                "max_simulation_drawdown_pct": -0.1,  # Invalid negative
                "max_simulation_position_size_pct": 1.5,  # Invalid > 100%
                "simulation_risk_monitoring_interval": 0  # Invalid zero
            }
        )
        
        # Should handle invalid configuration
        mode = SimulationMode(
            mode_id=uuid4(),
            config=invalid_config,
            portfolio=portfolio
        )
        
        # Should either apply defaults or raise validation error
        try:
            await mode.initialize()
            # If successful, check defaults were applied
            assert mode.simulation_safety_config['max_drawdown_pct'] > 0
            assert mode.simulation_safety_config['max_position_size_pct'] <= 1.0
        except ValueError:
            # Validation error is acceptable
            pass


class TestVirtualPortfolioProtection:
    """Test suite for virtual portfolio protection mechanisms."""
    
    @pytest.fixture
    def virtual_portfolio_protector(self):
        """Mock virtual portfolio protector."""
        protector = Mock()
        protector.validate_trade_safety = AsyncMock(
            return_value={'is_safe': True, 'risk_level': 'low', 'recommendations': []}
        )
        protector.check_portfolio_health = AsyncMock(
            return_value={'health_status': 'good', 'risk_score': 0.3}
        )
        protector.apply_protective_measures = AsyncMock(return_value=True)
        return protector
    
    async def test_virtual_portfolio_health_monitoring(self, virtual_portfolio_protector):
        """Test virtual portfolio health monitoring."""
        # Simulate portfolio with various conditions
        portfolio_state = {
            'total_value': Decimal('9500'),
            'unrealized_pnl': Decimal('-500'),
            'position_count': 5,
            'concentration_risk': 0.25,
            'volatility': 0.15
        }
        
        health_check = await virtual_portfolio_protector.check_portfolio_health(portfolio_state)
        
        # Should assess portfolio health
        virtual_portfolio_protector.check_portfolio_health.assert_called_once_with(portfolio_state)
        assert 'health_status' in health_check
        assert 'risk_score' in health_check
    
    async def test_trade_safety_validation(self, virtual_portfolio_protector):
        """Test trade safety validation for virtual trades."""
        trade_request = {
            'action': 'BUY',
            'token_address': 'BTC_ADDRESS',
            'amount_usd': Decimal('1200'),
            'current_portfolio_value': Decimal('10000')
        }
        
        safety_check = await virtual_portfolio_protector.validate_trade_safety(trade_request)
        
        # Should validate trade safety
        virtual_portfolio_protector.validate_trade_safety.assert_called_once_with(trade_request)
        assert 'is_safe' in safety_check
        assert 'risk_level' in safety_check
    
    async def test_protective_measures_application(self, virtual_portfolio_protector):
        """Test application of protective measures."""
        risk_scenario = {
            'type': 'high_concentration_risk',
            'severity': 'medium',
            'affected_positions': ['BTC_ADDRESS', 'ETH_ADDRESS']
        }
        
        protection_applied = await virtual_portfolio_protector.apply_protective_measures(risk_scenario)
        
        # Should apply protective measures
        virtual_portfolio_protector.apply_protective_measures.assert_called_once_with(risk_scenario)
        assert isinstance(protection_applied, bool)


class TestSimulationRiskAdaptation:
    """Test suite for risk monitoring adapted for simulation environments."""
    
    @pytest.fixture
    def simulation_mode_setup(self):
        """Setup simulation mode for risk adaptation testing."""
        portfolio_config = PortfolioConfig(
            initial_balance=Decimal("10000"),
            base_currency="USDC"
        )
        portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="Risk Test Portfolio",
            config=portfolio_config,
            cash_balance=Decimal("10000"),
            total_value=Decimal("10000")
        )
        
        mode_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "initial_balance": 10000,
                "enable_adaptive_risk_monitoring": True,
                "simulation_risk_tolerance": "medium",
                "enable_experimental_strategies": True
            }
        )
        
        return SimulationMode(
            mode_id=uuid4(),
            config=mode_config,
            portfolio=portfolio
        )
    
    async def test_adaptive_risk_thresholds(self, simulation_mode_setup):
        """Test adaptive risk thresholds for simulation environment."""
        mode = simulation_mode_setup
        await mode.initialize()
        
        # Mock adaptive risk manager
        mode.adaptive_risk_manager = Mock()
        mode.adaptive_risk_manager.adjust_thresholds_for_simulation = Mock(
            return_value={
                'position_size_multiplier': 1.5,  # 50% higher than live
                'drawdown_tolerance_multiplier': 1.33,  # 33% higher than live
                'volatility_tolerance_multiplier': 1.2   # 20% higher than live
            }
        )
        
        # Should adapt risk thresholds for simulation
        adaptations = mode.adaptive_risk_manager.adjust_thresholds_for_simulation()
        
        assert adaptations['position_size_multiplier'] > 1.0
        assert adaptations['drawdown_tolerance_multiplier'] > 1.0
        assert adaptations['volatility_tolerance_multiplier'] > 1.0
    
    async def test_experimental_strategy_safety(self, simulation_mode_setup):
        """Test safety measures for experimental strategies in simulation."""
        mode = simulation_mode_setup
        await mode.initialize()
        
        # Mock experimental strategy manager
        mode.experimental_strategy_manager = Mock()
        mode.experimental_strategy_manager.validate_experimental_trade = AsyncMock(
            return_value={'approved': True, 'risk_category': 'experimental_high'}
        )
        
        # Test experimental trade validation
        experimental_trade = {
            'strategy_type': 'experimental_momentum',
            'confidence_level': 0.6,
            'position_size': Decimal('800')
        }
        
        validation = await mode.experimental_strategy_manager.validate_experimental_trade(
            experimental_trade
        )
        
        # Should validate experimental strategies
        mode.experimental_strategy_manager.validate_experimental_trade.assert_called_once()
        assert 'approved' in validation
        assert 'risk_category' in validation
    
    async def test_simulation_specific_risk_metrics(self, simulation_mode_setup):
        """Test simulation-specific risk metrics collection."""
        mode = simulation_mode_setup
        await mode.initialize()
        await mode.start()
        
        # Generate trading activity
        mock_token = create_mock_token()
        market_state = MarketState(
            token=mock_token,
            price_usd=100.0,
            price_change_24h=5.0,
            volume_24h=1000000.0
        )
        
        await mode.process_tick(market_state)
        
        # Get simulation-specific risk metrics
        risk_metrics = await mode.get_simulation_risk_metrics()
        
        # Should include simulation-specific metrics
        assert 'virtual_var_95' in risk_metrics
        assert 'simulation_sharpe_ratio' in risk_metrics
        assert 'experimental_strategy_performance' in risk_metrics
        assert 'virtual_portfolio_stability' in risk_metrics
        assert 'simulation_accuracy_score' in risk_metrics