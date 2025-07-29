"""
Comprehensive TDD test suite for live trading mode.

This module contains extensive failing tests that define the requirements for 
the live trading mode implementation. Following strict TDD methodology - 
these tests are written first and will fail until the implementation is complete.

Key Requirements Tested:
- Real DEX integration (Jupiter, Hyperliquid, Uniswap V3)
- Production-grade safety systems and risk management
- Real-time P&L tracking and portfolio management
- Emergency stop and safety interlocks
- Experience collection for continuous learning
- Integration with RL feedback loop
- Error handling and recovery mechanisms
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

from src.modes.base import ModeType, ModeStatus, ModeConfig, ModeResult
from src.portfolio.base import (
    Portfolio, PortfolioConfig, Position, Transaction, PositionType,
    PositionStatus, TransactionType, PerformanceMetrics, RiskMetrics
)
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import ExperienceReplayBuffer, ReplayBufferConfig
from src.modes.experience_collector import TradingExperienceCollector, ExperienceCollectorConfig
from src.utils.base import Chain
from src.dex.base import SwapQuote, SwapResult, SwapStatus, DEXBase, DEXError

# Import classes that don't exist yet - these will fail until implemented
try:
    from src.modes.live_mode import (
        LiveMode,
        LiveTradingExecutor,
        LiveRiskManager,
        PortfolioSynchronizer,
        EmergencyStopSystem,
        LiveModeConfig,
        LiveModeMetrics,
        LiveModeError,
        PositionManager,
        RealTimePnLTracker,
        SafetyInterlocks,
        TradingSessionManager
    )
except ImportError:
    # These classes don't exist yet - will be implemented after tests
    LiveMode = None
    LiveTradingExecutor = None
    LiveRiskManager = None
    PortfolioSynchronizer = None
    EmergencyStopSystem = None
    LiveModeConfig = None
    LiveModeMetrics = None
    LiveModeError = None
    PositionManager = None
    RealTimePnLTracker = None
    SafetyInterlocks = None
    TradingSessionManager = None


def create_mock_token():
    """Helper function to create mock DiscoveredToken."""
    from src.discovery.base import DiscoveredToken, TokenStatus
    from datetime import datetime
    
    return DiscoveredToken(
        address="TEST_TOKEN_123",
        name="Test Token",
        symbol="TEST",
        chain=Chain.SOLANA,
        discovered_at=datetime.now(),
        discovery_source="test",
        status=TokenStatus.VALIDATED
    )


def create_mock_market_state(price_usd: float = 1.5, token_address: str = "TEST_TOKEN") -> MarketState:
    """Helper function to create MarketState for testing."""
    token = create_mock_token()
    token.address = token_address
    
    return MarketState(
        token=token,
        price_usd=price_usd,
        volume_24h=1000000.0,
        price_change_24h=5.2,
        market_cap=50000000,
        rsi=45.0,
        macd=0.05,
        sma_20=1.45,
        ema_12=1.48,
        bollinger_upper=1.6,
        bollinger_lower=1.4
    )


def create_mock_dex_client():
    """Create mock DEX client for testing."""
    mock_client = AsyncMock(spec=DEXBase)
    mock_client.get_quote.return_value = SwapQuote(
        input_token="USDC",
        output_token="TEST_TOKEN",
        input_amount=Decimal("1000"),
        output_amount=Decimal("666.67"),
        price=Decimal("1.5"),
        price_impact_bps=50,
        slippage_bps=10,
        dex_name="jupiter",
        quote_id="test_quote_123"
    )
    mock_client.execute_swap.return_value = SwapResult(
        transaction_hash="test_tx_hash",
        status=SwapStatus.CONFIRMED,
        input_token="USDC",
        output_token="TEST_TOKEN",
        input_amount=Decimal("1000"),
        actual_output_amount=Decimal("666.0"),
        timestamp=datetime.now(),
        dex_name="jupiter"
    )
    return mock_client


@pytest.fixture
def mock_portfolio():
    """Create mock portfolio for testing."""
    portfolio = Mock(spec=Portfolio)
    portfolio.cash_balance = Decimal("10000")
    portfolio.total_value = Decimal("15000")
    portfolio.positions = {}
    
    # Mock methods
    portfolio.get_positions_by_chain.return_value = {}
    portfolio.get_positions_by_dex.return_value = {}
    
    # Create mock performance metrics
    performance_metrics = PerformanceMetrics(
        total_pnl=Decimal("1000"),
        realized_pnl=Decimal("800"),
        unrealized_pnl=Decimal("200"),
        total_fees=Decimal("50"),
        win_rate=Decimal("0.65"),
        avg_win=Decimal("150"),
        avg_loss=Decimal("80"),
        profit_factor=Decimal("1.8"),
        sharpe_ratio=Decimal("1.2"),
        max_drawdown=Decimal("0.08"),
        total_trades=20,
        initial_balance=Decimal("10000"),
        current_balance=Decimal("11000")
    )
    portfolio.performance_metrics = performance_metrics
    
    # Add properties that live mode expects
    portfolio.open_positions = {}
    portfolio.total_unrealized_pnl = Decimal("200")
    
    return portfolio


@pytest.fixture
def mock_dex_clients():
    """Create mock DEX clients for testing."""
    jupiter_client = create_mock_dex_client()
    jupiter_client.chain = Chain.SOLANA
    jupiter_client.name = "jupiter"
    
    uniswap_client = create_mock_dex_client()
    uniswap_client.chain = Chain.ETHEREUM
    uniswap_client.name = "uniswap_v3"
    
    hyperliquid_client = create_mock_dex_client()
    hyperliquid_client.chain = Chain.ETHEREUM
    hyperliquid_client.name = "hyperliquid"
    
    return {
        "jupiter": jupiter_client,
        "uniswap_v3": uniswap_client,
        "hyperliquid": hyperliquid_client
    }


@pytest.fixture
def live_mode_config():
    """Create live mode configuration for testing."""
    return ModeConfig(
        mode_type=ModeType.LIVE_TRADING,
        enabled=True,
        auto_start=False,
        max_runtime_minutes=None,
        stop_on_error=True,
        log_level="INFO",
        parameters={
            "initial_balance": 50000,
            "enable_real_trading": True,
            "max_position_size_pct": 0.1,
            "max_daily_loss_pct": 0.05,
            "max_drawdown_pct": 0.15,
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.4,
            "enable_emergency_stop": True,
            "emergency_drawdown_pct": 0.25,
            "max_slippage_bps": 100,
            "enable_portfolio_sync": True,
            "sync_frequency_seconds": 30,
            "enable_experience_collection": True,
            "experience_buffer_size": 10000,
            "enable_rl_feedback": True,
            "safety_check_frequency_seconds": 10,
            "max_open_positions": 15,
            "min_trade_interval_seconds": 5,
            "enable_risk_monitoring": True,
            "position_timeout_minutes": 60,
            "enable_real_time_pnl": True,
            "pnl_update_frequency_seconds": 5,
            "trading_hours_start": 0,
            "trading_hours_end": 24,
            "enable_weekends": True,
            "dex_preference_order": ["jupiter", "uniswap_v3", "hyperliquid"],
            "enable_cross_dex_arbitrage": False,
            "max_concurrent_orders": 5,
            "order_timeout_seconds": 30
        }
    )


class TestLiveModeConfig:
    """Test live mode configuration and validation."""
    
    def test_live_mode_config_creation(self):
        """Test LiveModeConfig creation and validation."""
        config = LiveModeConfig(
            initial_balance=Decimal("50000"),
            enable_real_trading=True,
            max_position_size_pct=Decimal("0.1"),
            max_daily_loss_pct=Decimal("0.05"),
            emergency_drawdown_pct=Decimal("0.25"),
            dex_preference_order=["jupiter", "uniswap_v3", "hyperliquid"]
        )
        
        assert config.initial_balance == Decimal("50000")
        assert config.enable_real_trading is True
        assert config.max_position_size_pct == Decimal("0.1")
        assert len(config.dex_preference_order) == 3
    
    def test_live_mode_config_validation_errors(self):
        """Test validation errors in live mode configuration."""
        # Test invalid position size
        with pytest.raises(ValueError, match="max_position_size_pct must be between 0 and 1"):
            LiveModeConfig(max_position_size_pct=Decimal("1.5"))
        
        # Test invalid drawdown percentage
        with pytest.raises(ValueError, match="emergency_drawdown_pct must be between 0 and 1"):
            LiveModeConfig(emergency_drawdown_pct=Decimal("2.0"))
        
        # Test empty DEX list
        with pytest.raises(ValueError, match="At least one DEX must be configured"):
            LiveModeConfig(dex_preference_order=[])


class TestLiveTradingExecutor:
    """Test live trading executor with real DEX integration."""
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_executor_initialization(self, mock_dex_clients, mock_portfolio):
        """Test live trading executor initialization."""
        executor = LiveTradingExecutor(
            portfolio=mock_portfolio,
            dex_clients=mock_dex_clients,
            enable_real_trading=True,
            max_slippage_bps=100
        )
        
        assert executor.portfolio == mock_portfolio
        assert len(executor.dex_clients) == 3
        assert executor.enable_real_trading is True
        assert executor.max_slippage_bps == 100
        assert executor.is_active is False
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_execute_real_buy_order(self, mock_dex_clients, mock_portfolio):
        """Test execution of real buy order through DEX."""
        executor = LiveTradingExecutor(
            portfolio=mock_portfolio,
            dex_clients=mock_dex_clients,
            enable_real_trading=True
        )
        await executor.start()
        
        result = await executor.execute_buy_order(
            token_address="TEST_TOKEN_123",
            amount_usd=Decimal("1000"),
            dex_preference=["jupiter"]
        )
        
        assert result.success is True
        assert result.status == SwapStatus.CONFIRMED
        assert result.input_amount == Decimal("1000")
        mock_dex_clients["jupiter"].execute_swap.assert_called_once()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_execute_real_sell_order(self, mock_dex_clients, mock_portfolio):
        """Test execution of real sell order through DEX."""
        executor = LiveTradingExecutor(
            portfolio=mock_portfolio,
            dex_clients=mock_dex_clients,
            enable_real_trading=True
        )
        await executor.start()
        
        result = await executor.execute_sell_order(
            token_address="TEST_TOKEN_123",
            amount=Decimal("500"),
            dex_preference=["jupiter"]
        )
        
        assert result.success is True
        assert result.status == SwapStatus.CONFIRMED
        mock_dex_clients["jupiter"].execute_swap.assert_called_once()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_dex_failover_mechanism(self, mock_dex_clients, mock_portfolio):
        """Test DEX failover when primary DEX fails."""
        # Configure first DEX to fail
        mock_dex_clients["jupiter"].execute_swap.side_effect = DEXError("Jupiter unavailable")
        
        executor = LiveTradingExecutor(
            portfolio=mock_portfolio,
            dex_clients=mock_dex_clients,
            enable_real_trading=True
        )
        await executor.start()
        
        result = await executor.execute_buy_order(
            token_address="TEST_TOKEN_123",
            amount_usd=Decimal("1000"),
            dex_preference=["jupiter", "uniswap_v3", "hyperliquid"]
        )
        
        # Should fallback to uniswap_v3
        assert result.success is True
        assert result.dex_name == "uniswap_v3"
        mock_dex_clients["uniswap_v3"].execute_swap.assert_called_once()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_order_timeout_handling(self, mock_dex_clients, mock_portfolio):
        """Test handling of order timeouts."""
        # Configure DEX to timeout
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate slow execution
            return create_mock_dex_client().execute_swap.return_value
        
        mock_dex_clients["jupiter"].execute_swap.side_effect = slow_execute
        
        executor = LiveTradingExecutor(
            portfolio=mock_portfolio,
            dex_clients=mock_dex_clients,
            enable_real_trading=True,
            order_timeout_seconds=1
        )
        await executor.start()
        
        result = await executor.execute_buy_order(
            token_address="TEST_TOKEN_123",
            amount_usd=Decimal("1000")
        )
        
        assert result.success is False
        assert "timeout" in result.error_message.lower()


class TestLiveRiskManager:
    """Test live risk management with production safety systems."""
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_risk_manager_initialization(self, mock_portfolio):
        """Test live risk manager initialization."""
        risk_config = {
            "max_position_size_pct": 0.1,
            "max_daily_loss_pct": 0.05,
            "max_drawdown_pct": 0.15,
            "stop_loss_pct": 0.08,
            "emergency_drawdown_pct": 0.25
        }
        
        risk_manager = LiveRiskManager(
            portfolio=mock_portfolio,
            config=risk_config,
            enable_real_time_monitoring=True
        )
        
        assert risk_manager.portfolio == mock_portfolio
        assert risk_manager.max_position_size_pct == Decimal("0.1")
        assert risk_manager.enable_real_time_monitoring is True
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_position_size_validation(self, mock_portfolio):
        """Test position size validation against risk limits."""
        mock_portfolio.get_total_value.return_value = Decimal("50000")
        
        risk_manager = LiveRiskManager(
            portfolio=mock_portfolio,
            config={"max_position_size_pct": 0.1}
        )
        
        # Valid position size (5% of portfolio)
        validation = await risk_manager.validate_position_size("TEST_TOKEN", Decimal("2500"))
        assert validation.is_valid is True
        
        # Invalid position size (15% of portfolio)
        validation = await risk_manager.validate_position_size("TEST_TOKEN", Decimal("7500"))
        assert validation.is_valid is False
        assert "exceeds maximum" in validation.reason.lower()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_emergency_stop_trigger(self, mock_portfolio):
        """Test emergency stop trigger on excessive drawdown."""
        # Mock portfolio with high drawdown
        performance = PerformanceMetrics(
            total_pnl=Decimal("-15000"),  # 30% loss
            max_drawdown=Decimal("0.30"),
            initial_balance=Decimal("50000"),
            current_balance=Decimal("35000")
        )
        mock_portfolio.get_performance_metrics.return_value = performance
        
        risk_manager = LiveRiskManager(
            portfolio=mock_portfolio,
            config={"emergency_drawdown_pct": 0.25}  # 25% emergency threshold
        )
        
        emergency_check = await risk_manager.check_emergency_conditions()
        assert emergency_check.should_stop is True
        assert emergency_check.reason == "EMERGENCY_DRAWDOWN_EXCEEDED"
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_daily_loss_limit_check(self, mock_portfolio):
        """Test daily loss limit enforcement."""
        risk_manager = LiveRiskManager(
            portfolio=mock_portfolio,
            config={"max_daily_loss_pct": 0.05}
        )
        
        # Simulate 8% daily loss
        validation = await risk_manager.check_daily_loss_limit(daily_pnl=Decimal("-4000"))
        assert validation.is_valid is False
        assert "daily loss limit" in validation.reason.lower()


class TestPortfolioSynchronizer:
    """Test real-time portfolio synchronization."""
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_portfolio_sync_initialization(self, mock_portfolio, mock_dex_clients):
        """Test portfolio synchronizer initialization."""
        sync = PortfolioSynchronizer(
            portfolio=mock_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        assert sync.portfolio == mock_portfolio
        assert sync.sync_frequency_seconds == 30
        assert sync.is_syncing is False
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_real_time_portfolio_sync(self, mock_portfolio, mock_dex_clients):
        """Test real-time portfolio synchronization with DEX data."""
        sync = PortfolioSynchronizer(
            portfolio=mock_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=1
        )
        
        await sync.start_sync()
        assert sync.is_syncing is True
        
        # Let it sync once
        await asyncio.sleep(1.1)
        
        await sync.stop_sync()
        assert sync.is_syncing is False
        
        # Should have called portfolio update methods
        assert sync.last_sync_time is not None
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_position_reconciliation(self, mock_portfolio, mock_dex_clients):
        """Test position reconciliation between portfolio and DEX."""
        sync = PortfolioSynchronizer(
            portfolio=mock_portfolio,
            dex_clients=mock_dex_clients
        )
        
        # Mock portfolio position
        portfolio_position = Position(
            position_id=uuid4(),
            symbol="TEST_TOKEN/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("1000"),
            entry_price=Decimal("1.5"),
            current_price=Decimal("1.6"),
            status=PositionStatus.OPEN
        )
        mock_portfolio.get_positions.return_value = [portfolio_position]
        
        discrepancies = await sync.reconcile_positions()
        assert isinstance(discrepancies, list)


class TestEmergencyStopSystem:
    """Test emergency stop and safety interlocks."""
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_emergency_stop_initialization(self):
        """Test emergency stop system initialization."""
        emergency_system = EmergencyStopSystem(
            max_drawdown_pct=Decimal("0.25"),
            max_daily_loss_pct=Decimal("0.10"),
            enable_kill_switch=True
        )
        
        assert emergency_system.max_drawdown_pct == Decimal("0.25")
        assert emergency_system.enable_kill_switch is True
        assert emergency_system.is_emergency_stopped is False
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_emergency_stop_trigger(self, mock_portfolio):
        """Test emergency stop trigger mechanisms."""
        emergency_system = EmergencyStopSystem(
            max_drawdown_pct=Decimal("0.20"),
            max_daily_loss_pct=Decimal("0.08")
        )
        
        # Trigger emergency stop due to drawdown
        await emergency_system.trigger_emergency_stop(
            reason="MAX_DRAWDOWN_EXCEEDED",
            portfolio_value=Decimal("40000"),
            initial_value=Decimal("50000")
        )
        
        assert emergency_system.is_emergency_stopped is True
        assert emergency_system.stop_reason == "MAX_DRAWDOWN_EXCEEDED"
        assert emergency_system.stop_timestamp is not None
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_safety_interlocks(self):
        """Test safety interlock mechanisms."""
        safety = SafetyInterlocks(
            enable_trading_hours=True,
            trading_hours_start=9,  # 9 AM
            trading_hours_end=17,   # 5 PM
            enable_position_limits=True,
            max_open_positions=10
        )
        
        # Test trading hours check (assuming test runs during non-trading hours)
        is_allowed = await safety.check_trading_allowed()
        # Result depends on when test is run
        
        # Test position limit check
        is_allowed = await safety.check_position_limits(current_positions=15)
        assert is_allowed is False  # Exceeds max 10 positions


class TestRealTimePnLTracker:
    """Test real-time P&L tracking system."""
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_pnl_tracker_initialization(self, mock_portfolio):
        """Test real-time P&L tracker initialization."""
        pnl_tracker = RealTimePnLTracker(
            portfolio=mock_portfolio,
            update_frequency_seconds=5,
            enable_detailed_tracking=True
        )
        
        assert pnl_tracker.portfolio == mock_portfolio
        assert pnl_tracker.update_frequency_seconds == 5
        assert pnl_tracker.is_tracking is False
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_real_time_pnl_updates(self, mock_portfolio):
        """Test real-time P&L calculation and updates."""
        pnl_tracker = RealTimePnLTracker(
            portfolio=mock_portfolio,
            update_frequency_seconds=1
        )
        
        await pnl_tracker.start_tracking()
        assert pnl_tracker.is_tracking is True
        
        # Let it track for a short period
        await asyncio.sleep(1.1)
        
        await pnl_tracker.stop_tracking()
        assert pnl_tracker.is_tracking is False
        
        # Should have captured P&L snapshots
        assert len(pnl_tracker.pnl_history) > 0
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_pnl_alert_triggers(self, mock_portfolio):
        """Test P&L alert triggering mechanisms."""
        pnl_tracker = RealTimePnLTracker(
            portfolio=mock_portfolio,
            loss_alert_threshold=Decimal("0.05")  # 5% loss alert
        )
        
        # Simulate significant loss
        await pnl_tracker.update_pnl(
            current_value=Decimal("47500"),  # 5% loss from 50000
            unrealized_pnl=Decimal("-2500")
        )
        
        alerts = pnl_tracker.get_active_alerts()
        assert len(alerts) > 0
        assert any("loss threshold" in alert.message.lower() for alert in alerts)


class TestLiveMode:
    """Test main live trading mode class."""
    
    async def test_live_mode_initialization(self, live_mode_config, mock_portfolio, mock_dex_clients):
        """Test live mode initialization with all components."""
        with patch('src.modes.live_mode.LiveTradingExecutor') as mock_executor, \
             patch('src.modes.live_mode.LiveRiskManager') as mock_risk_manager, \
             patch('src.modes.live_mode.PortfolioSynchronizer') as mock_sync, \
             patch('src.modes.live_mode.EmergencyStopSystem') as mock_emergency:
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            assert live_mode.config.mode_type == ModeType.LIVE_TRADING
            assert live_mode.status == ModeStatus.INACTIVE
            assert live_mode.enable_real_trading is True

    async def test_dex_clients_initialization_success(self, live_mode_config, mock_portfolio):
        """Test successful DEX clients initialization with all three exchanges."""
        with patch('src.modes.live_mode.JupiterDEXClient') as mock_jupiter, \
             patch('src.modes.live_mode.UniswapV3Client') as mock_uniswap, \
             patch('src.modes.live_mode.HyperliquidDEXClient') as mock_hyperliquid, \
             patch('src.modes.live_mode.SolanaWallet') as mock_solana_wallet, \
             patch('src.modes.live_mode.EthereumWallet') as mock_eth_wallet, \
             patch('src.modes.live_mode.os.getenv') as mock_getenv:
            
            # Mock environment variables
            mock_getenv.side_effect = lambda key, default=None: {
                'SOLANA_PRIVATE_KEY': 'test_solana_key',
                'SOLANA_NETWORK': 'devnet',
                'ETHEREUM_PRIVATE_KEY': 'test_eth_key',
                'HYPERLIQUID_PRIVATE_KEY': 'test_hl_key',
                'HYPERLIQUID_WALLET_ADDRESS': 'test_hl_address',
                'HYPERLIQUID_API_KEY': 'test_hl_api_key',
                'INFURA_PROJECT_ID': 'test_infura_id'
            }.get(key, default)
            
            # Configure successful initialization
            mock_jupiter_instance = AsyncMock()
            mock_jupiter_instance.connect.return_value = True
            mock_jupiter_instance.is_connected = True
            mock_jupiter.return_value = mock_jupiter_instance
            
            mock_uniswap_instance = AsyncMock()
            mock_uniswap_instance.connect.return_value = True
            mock_uniswap_instance.is_connected = True
            mock_uniswap.return_value = mock_uniswap_instance
            
            mock_hyperliquid_instance = AsyncMock()
            mock_hyperliquid_instance.connect.return_value = True
            mock_hyperliquid_instance.is_connected = True
            mock_hyperliquid.return_value = mock_hyperliquid_instance
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode._initialize_dex_clients()
            
            # Verify all DEX clients were initialized
            mock_jupiter.assert_called_once()
            mock_uniswap.assert_called_once()
            mock_hyperliquid.assert_called_once()
            
            # Verify connections were performed
            mock_jupiter_instance.connect.assert_called_once()
            mock_uniswap_instance.connect.assert_called_once()
            mock_hyperliquid_instance.connect.assert_called_once()
            
            # Verify clients are stored
            assert hasattr(live_mode, 'dex_clients')
            assert 'jupiter' in live_mode.dex_clients
            assert 'uniswap_v3' in live_mode.dex_clients
            assert 'hyperliquid' in live_mode.dex_clients

    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_dex_clients_initialization_with_wallet_integration(self, live_mode_config, mock_portfolio):
        """Test DEX clients initialization includes proper wallet integration."""
        with patch('src.dex.JupiterDEXClient') as mock_jupiter, \
             patch('src.dex.UniswapV3Client') as mock_uniswap, \
             patch('src.dex.HyperliquidDEXClient') as mock_hyperliquid, \
             patch('src.wallet.SolanaWallet') as mock_solana_wallet, \
             patch('src.wallet.EthereumWallet') as mock_eth_wallet, \
             patch('os.getenv') as mock_getenv:
            
            # Mock environment variables
            mock_getenv.side_effect = lambda key, default=None: {
                'SOLANA_PRIVATE_KEY': 'test_solana_key',
                'ETHEREUM_PRIVATE_KEY': 'test_eth_key',
                'HYPERLIQUID_PRIVATE_KEY': 'test_hl_key',
                'HYPERLIQUID_WALLET_ADDRESS': 'test_hl_address',
                'INFURA_PROJECT_ID': 'test_infura_id'
            }.get(key, default)
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode._initialize_dex_clients()
            
            # Verify Jupiter was initialized with Solana wallet
            mock_jupiter.assert_called_once()
            jupiter_call_args = mock_jupiter.call_args
            assert 'wallet' in jupiter_call_args.kwargs
            
            # Verify Uniswap was initialized with Ethereum wallet
            mock_uniswap.assert_called_once()
            uniswap_call_args = mock_uniswap.call_args
            assert 'wallet' in uniswap_call_args.kwargs
            
            # Verify Hyperliquid was initialized with wallet address
            mock_hyperliquid.assert_called_once()
            hyperliquid_call_args = mock_hyperliquid.call_args
            assert 'wallet_address' in hyperliquid_call_args.kwargs

    @pytest.mark.skip(reason="Implementation pending - TDD")  
    async def test_dex_clients_connection_failure_handling(self, live_mode_config, mock_portfolio):
        """Test graceful handling of DEX connection failures."""
        with patch('src.dex.JupiterDEXClient') as mock_jupiter, \
             patch('src.dex.UniswapV3Client') as mock_uniswap, \
             patch('src.dex.HyperliquidDEXClient') as mock_hyperliquid:
            
            # Configure Jupiter to fail connection
            mock_jupiter_instance = AsyncMock()
            mock_jupiter_instance.initialize.side_effect = DEXError("Jupiter connection failed")
            mock_jupiter.return_value = mock_jupiter_instance
            
            # Configure others to succeed
            mock_uniswap_instance = AsyncMock()
            mock_uniswap_instance.initialize.return_value = None
            mock_uniswap_instance.health_check.return_value = True
            mock_uniswap.return_value = mock_uniswap_instance
            
            mock_hyperliquid_instance = AsyncMock()
            mock_hyperliquid_instance.initialize.return_value = None
            mock_hyperliquid_instance.health_check.return_value = True
            mock_hyperliquid.return_value = mock_hyperliquid_instance
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode._initialize_dex_clients()
            
            # Should have working clients for successful connections
            assert 'uniswap_v3' in live_mode.dex_clients
            assert 'hyperliquid' in live_mode.dex_clients
            
            # Jupiter should be marked as unavailable but not crash the system
            assert 'jupiter' not in live_mode.dex_clients or live_mode.dex_clients['jupiter'] is None
            
            # Should log the failure but continue
            assert hasattr(live_mode, 'dex_connection_errors')
            assert 'jupiter' in live_mode.dex_connection_errors

    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_dex_clients_safety_checks_disable_trading(self, live_mode_config, mock_portfolio):
        """Test safety checks disable live trading when critical DEX clients fail."""
        with patch('src.dex.JupiterDEXClient') as mock_jupiter, \
             patch('src.dex.UniswapV3Client') as mock_uniswap, \
             patch('src.dex.HyperliquidDEXClient') as mock_hyperliquid:
            
            # Configure all DEX clients to fail
            for mock_client in [mock_jupiter, mock_uniswap, mock_hyperliquid]:
                mock_instance = AsyncMock()
                mock_instance.initialize.side_effect = DEXError("Connection failed")
                mock_client.return_value = mock_instance
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode._initialize_dex_clients()
            
            # Should disable live trading when no DEX clients are available
            assert live_mode.enable_real_trading is False
            assert hasattr(live_mode, 'safety_lockout_reason')
            assert 'DEX' in live_mode.safety_lockout_reason

    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_dex_clients_health_check_monitoring(self, live_mode_config, mock_portfolio):
        """Test ongoing health check monitoring of DEX clients."""
        with patch('src.dex.JupiterDEXClient') as mock_jupiter, \
             patch('src.dex.UniswapV3Client') as mock_uniswap, \
             patch('src.dex.HyperliquidDEXClient') as mock_hyperliquid:
            
            # Configure successful initialization
            mock_jupiter_instance = AsyncMock()
            mock_jupiter_instance.initialize.return_value = None
            mock_jupiter_instance.health_check.return_value = True
            mock_jupiter.return_value = mock_jupiter_instance
            
            mock_uniswap_instance = AsyncMock()
            mock_uniswap_instance.initialize.return_value = None
            mock_uniswap_instance.health_check.return_value = True
            mock_uniswap.return_value = mock_uniswap_instance
            
            mock_hyperliquid_instance = AsyncMock()
            mock_hyperliquid_instance.initialize.return_value = None
            mock_hyperliquid_instance.health_check.return_value = True
            mock_hyperliquid.return_value = mock_hyperliquid_instance
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode._initialize_dex_clients()
            
            # Should have health check monitoring enabled
            assert hasattr(live_mode, 'dex_health_monitor')
            assert live_mode.dex_health_monitor is not None
            
            # Should periodically check DEX health
            await live_mode._check_dex_health()
            
            # All clients should have been health checked
            mock_jupiter_instance.health_check.assert_called()
            mock_uniswap_instance.health_check.assert_called()
            mock_hyperliquid_instance.health_check.assert_called()

    async def test_dex_clients_missing_environment_variables(self, live_mode_config, mock_portfolio):
        """Test handling of missing required environment variables."""
        with patch('src.modes.live_mode.os.getenv') as mock_getenv:
            # Mock missing environment variables
            mock_getenv.return_value = None
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            with pytest.raises(ValueError, match="Missing required environment variable"):
                await live_mode._initialize_dex_clients()

    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_dex_clients_configuration_validation(self, live_mode_config, mock_portfolio):
        """Test DEX client configuration validation."""
        with patch('src.dex.JupiterDEXClient') as mock_jupiter, \
             patch('os.getenv') as mock_getenv:
            
            # Mock environment variables with invalid values
            mock_getenv.side_effect = lambda key, default=None: {
                'SOLANA_PRIVATE_KEY': 'invalid_key_format',
                'SOLANA_NETWORK': 'invalid_network'
            }.get(key, default)
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            # Should validate configuration and handle invalid values
            await live_mode._initialize_dex_clients()
            
            # Should log configuration errors
            assert hasattr(live_mode, 'dex_config_errors')
            assert len(live_mode.dex_config_errors) > 0
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_live_mode_startup_sequence(self, live_mode_config, mock_portfolio):
        """Test live mode startup and component initialization."""
        with patch('src.modes.live_mode.LiveTradingExecutor') as mock_executor, \
             patch('src.modes.live_mode.LiveRiskManager') as mock_risk_manager:
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            assert live_mode.status == ModeStatus.INACTIVE
            
            await live_mode.start()
            assert live_mode.status == ModeStatus.ACTIVE
            
            # Should have initialized all components
            mock_executor.assert_called_once()
            mock_risk_manager.assert_called_once()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_live_mode_trading_decision(self, live_mode_config, mock_portfolio):
        """Test live mode trading decision process."""
        with patch('src.modes.live_mode.LiveTradingExecutor') as mock_executor, \
             patch('src.modes.live_mode.LiveRiskManager') as mock_risk_manager:
            
            # Configure mocks
            mock_risk_manager.return_value.validate_new_position.return_value = Mock(is_valid=True)
            mock_executor.return_value.execute_buy_order.return_value = SwapResult(
                transaction_hash="test_tx",
                status=SwapStatus.CONFIRMED,
                input_token="USDC",
                output_token="TEST_TOKEN",
                input_amount=Decimal("1000")
            )
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Process market tick with buy signal
            market_state = create_mock_market_state(price_usd=1.0)  # Low RSI for buy signal
            market_state.rsi = 25.0  # Oversold condition
            
            action = await live_mode.process_tick(market_state)
            assert action in [TradeAction.BUY, TradeAction.STRONG_BUY]
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_live_mode_emergency_stop_integration(self, live_mode_config, mock_portfolio):
        """Test emergency stop integration with live mode."""
        with patch('src.modes.live_mode.EmergencyStopSystem') as mock_emergency:
            
            # Configure emergency system to trigger stop
            emergency_instance = mock_emergency.return_value
            emergency_instance.is_emergency_stopped = True
            emergency_instance.stop_reason = "MAX_DRAWDOWN_EXCEEDED"
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Process tick should return None when emergency stopped
            market_state = create_mock_market_state()
            action = await live_mode.process_tick(market_state)
            assert action is None
            
            # Mode should transition to ERROR status
            assert live_mode.status == ModeStatus.ERROR
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_live_mode_experience_collection_integration(self, live_mode_config, mock_portfolio):
        """Test experience collection integration for continuous learning."""
        with patch('src.modes.live_mode.TradingExperienceCollector') as mock_collector:
            
            # Enable experience collection in config
            live_mode_config.parameters["enable_experience_collection"] = True
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Should have created experience collector
            mock_collector.assert_called_once()
            
            # Process trading action - should capture experience
            market_state = create_mock_market_state()
            market_state.rsi = 25.0  # Buy signal
            
            with patch.object(live_mode, '_execute_live_trade') as mock_execute:
                mock_execute.return_value = TradingResult(
                    action=TradeAction.BUY,
                    token="TEST_TOKEN",
                    executed_at=datetime.now(),
                    price=1.0,
                    quantity=1000.0,
                    value_usd=1000.0,
                    success=True
                )
                
                action = await live_mode.process_tick(market_state)
                
                # Should have called experience collection methods
                collector_instance = mock_collector.return_value
                collector_instance.capture_pre_trade_state.assert_called_once()


class TestLiveModeErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_dex_connection_error_handling(self, live_mode_config, mock_portfolio):
        """Test handling of DEX connection errors."""
        with patch('src.modes.live_mode.LiveTradingExecutor') as mock_executor:
            
            # Configure executor to raise connection error
            mock_executor.return_value.execute_buy_order.side_effect = DEXError("Connection failed")
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            market_state = create_mock_market_state()
            market_state.rsi = 25.0  # Buy signal
            
            # Should handle error gracefully and not crash
            action = await live_mode.process_tick(market_state)
            
            # Mode should remain active but log error
            assert live_mode.status == ModeStatus.ACTIVE
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_portfolio_sync_error_recovery(self, live_mode_config, mock_portfolio):
        """Test portfolio synchronization error recovery."""
        with patch('src.modes.live_mode.PortfolioSynchronizer') as mock_sync:
            
            # Configure sync to fail initially then succeed
            sync_instance = mock_sync.return_value
            sync_instance.reconcile_positions.side_effect = [
                Exception("Sync failed"),  # First call fails
                []  # Second call succeeds
            ]
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Should recover from sync error
            assert live_mode.status == ModeStatus.ACTIVE


class TestLiveModePerformance:
    """Test live mode performance and metrics."""
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_live_mode_metrics_collection(self, live_mode_config, mock_portfolio):
        """Test collection of live trading metrics."""
        live_mode = LiveMode(
            mode_id=uuid4(),
            config=live_mode_config,
            portfolio=mock_portfolio
        )
        
        await live_mode.initialize()
        await live_mode.start()
        
        # Simulate some trading activity
        market_state = create_mock_market_state()
        await live_mode.process_tick(market_state)
        
        # Get metrics
        metrics = live_mode.get_live_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_trades" in metrics
        assert "current_portfolio_value" in metrics
        assert "realized_pnl" in metrics
        assert "unrealized_pnl" in metrics
        assert "active_positions" in metrics
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_trading_latency_measurement(self, live_mode_config, mock_portfolio):
        """Test measurement of trading execution latency."""
        live_mode = LiveMode(
            mode_id=uuid4(),
            config=live_mode_config,
            portfolio=mock_portfolio
        )
        
        await live_mode.initialize()
        await live_mode.start()
        
        # Execute trade and measure latency
        start_time = datetime.now()
        market_state = create_mock_market_state()
        market_state.rsi = 25.0  # Buy signal
        
        action = await live_mode.process_tick(market_state)
        
        # Should track execution latency
        metrics = live_mode.get_live_metrics()
        if "execution_latency_ms" in metrics:
            assert metrics["execution_latency_ms"] > 0


class TestLiveModeCleanup:
    """Test live mode cleanup and shutdown procedures."""
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_graceful_shutdown(self, live_mode_config, mock_portfolio):
        """Test graceful shutdown of live mode."""
        with patch('src.modes.live_mode.LiveTradingExecutor') as mock_executor, \
             patch('src.modes.live_mode.PortfolioSynchronizer') as mock_sync, \
             patch('src.modes.live_mode.EmergencyStopSystem') as mock_emergency:
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            assert live_mode.status == ModeStatus.ACTIVE
            
            await live_mode.stop()
            assert live_mode.status == ModeStatus.STOPPING
            
            await live_mode.cleanup()
            
            # Should have cleaned up all components
            mock_executor.return_value.stop.assert_called_once()
            mock_sync.return_value.stop_sync.assert_called_once()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_final_portfolio_reconciliation(self, live_mode_config, mock_portfolio):
        """Test final portfolio reconciliation on shutdown."""
        with patch('src.modes.live_mode.PortfolioSynchronizer') as mock_sync:
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            await live_mode.stop()
            
            # Should perform final reconciliation
            sync_instance = mock_sync.return_value
            await live_mode.cleanup()
            
            # Verify final sync was called
            sync_instance.reconcile_positions.assert_called()


# Integration Tests
class TestLiveModeIntegration:
    """Integration tests for live mode with other system components."""
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_ml_rl_integration_with_live_trading(self, live_mode_config, mock_portfolio):
        """Test ML-RL integration with live trading decisions."""
        # This test verifies that ML predictions and RL actions 
        # properly integrate with live trading execution
        
        with patch('src.integration.ml_rl_bridge.MLRLBridge') as mock_bridge:
            
            # Configure ML-RL bridge to return strong buy signal
            mock_bridge.return_value.get_trading_decision.return_value = TradeAction.STRONG_BUY
            
            live_mode_config.parameters["enable_ml_rl_integration"] = True
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            market_state = create_mock_market_state()
            action = await live_mode.process_tick(market_state)
            
            # Should have used ML-RL decision
            assert action == TradeAction.STRONG_BUY
            mock_bridge.return_value.get_trading_decision.assert_called_once()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_continuous_learning_feedback_loop(self, live_mode_config, mock_portfolio):
        """Test continuous learning feedback loop in live trading."""
        # This test verifies that live trading results feed back into
        # the RL training pipeline for continuous improvement
        
        live_mode_config.parameters["enable_rl_feedback"] = True
        live_mode_config.parameters["enable_experience_collection"] = True
        
        with patch('src.modes.experience_collector.TradingExperienceCollector') as mock_collector:
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Simulate successful trade
            market_state = create_mock_market_state()
            market_state.rsi = 25.0
            
            with patch.object(live_mode, '_execute_live_trade') as mock_execute:
                mock_execute.return_value = TradingResult(
                    action=TradeAction.BUY,
                    token="TEST_TOKEN",
                    executed_at=datetime.now(),
                    price=1.0,
                    quantity=1000.0,
                    value_usd=1000.0,
                    success=True,
                    realized_pnl=50.0  # Profitable trade
                )
                
                await live_mode.process_tick(market_state)
                
                # Should capture experience for RL training
                collector_instance = mock_collector.return_value
                collector_instance.capture_pre_trade_state.assert_called_once()
                collector_instance.capture_post_trade_result.assert_called_once()


# Continuous Learning Loop Integration Tests
class TestContinuousLearningLoopIntegration:
    """Test suite for continuous learning loop integration in LiveMode."""
    
    async def test_continuous_learning_loop_initialization(self, live_mode_config, mock_portfolio):
        """Test that continuous learning loop is properly initialized in LiveMode."""
        live_mode_config.parameters["enable_continuous_learning"] = True
        live_mode_config.parameters["learning_check_frequency_seconds"] = 30
        
        with patch('src.modes.live_mode.ContinuousLearningLoop') as mock_loop:
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            
            # Should initialize continuous learning loop
            mock_loop.assert_called_once()
            loop_instance = mock_loop.return_value
            assert live_mode.continuous_learning_loop is loop_instance
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_background_learning_check_triggered(self, live_mode_config, mock_portfolio):
        """Test that background learning checks are triggered during live trading."""
        live_mode_config.parameters["enable_continuous_learning"] = True
        
        with patch('src.modes.live_mode.ContinuousLearningLoop') as mock_loop:
            mock_loop_instance = mock_loop.return_value
            mock_loop_instance.should_trigger_learning.return_value = True
            mock_loop_instance.trigger_learning_cycle.return_value = {
                "training_triggered": True,
                "new_model_version": "v1.2.0",
                "performance_improved": True,
                "model_activated": True
            }
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Simulate multiple trading ticks to trigger learning check
            market_state = create_mock_market_state()
            
            for _ in range(3):  # Simulate multiple ticks
                await live_mode.process_tick(market_state)
                await asyncio.sleep(0.1)  # Small delay to simulate time passage
            
            # Should check for learning triggers
            mock_loop_instance.should_trigger_learning.assert_called()
            mock_loop_instance.trigger_learning_cycle.assert_called()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_model_hot_swapping_during_live_trading(self, live_mode_config, mock_portfolio):
        """Test that models can be hot-swapped during live trading without interruption."""
        live_mode_config.parameters["enable_continuous_learning"] = True
        live_mode_config.parameters["enable_model_hot_swapping"] = True
        
        with patch('src.modes.live_mode.ContinuousLearningLoop') as mock_loop, \
             patch('src.modes.live_mode.ModelHotSwapper') as mock_swapper:
            
            mock_swapper_instance = mock_swapper.return_value
            mock_swapper_instance.can_swap_safely.return_value = True
            mock_swapper_instance.swap_model.return_value = True
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Trigger model swap
            new_model_path = "/models/v1.2.0/model.pth"
            await live_mode.trigger_model_swap(new_model_path)
            
            # Should perform safety checks and swap model
            mock_swapper_instance.can_swap_safely.assert_called_once()
            mock_swapper_instance.swap_model.assert_called_once_with(new_model_path)
            
            # Trading should continue uninterrupted
            market_state = create_mock_market_state()
            action = await live_mode.process_tick(market_state)
            assert action is not None  # Still able to make trading decisions
    
    async def test_performance_feedback_capture(self, live_mode_config, mock_portfolio):
        """Test that trading performance feedback is captured for learning evaluation."""
        live_mode_config.parameters["enable_continuous_learning"] = True
        live_mode_config.parameters["enable_performance_feedback"] = True
        
        with patch('src.modes.live_mode.PerformanceFeedbackCapture') as mock_feedback:
            mock_feedback_instance = mock_feedback.return_value
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Simulate profitable trade
            market_state = create_mock_market_state()
            
            with patch.object(live_mode, '_execute_live_trade') as mock_execute:
                trading_result = TradingResult(
                    action=TradeAction.BUY,
                    token="TEST_TOKEN",
                    executed_at=datetime.now(),
                    price=1.0,
                    quantity=1000.0,
                    value_usd=1000.0,
                    success=True,
                    realized_pnl=50.0,
                    portfolio_value_before=50000.0,
                    portfolio_value_after=50050.0
                )
                mock_execute.return_value = trading_result
                
                await live_mode.process_tick(market_state)
                
                # Should capture performance feedback
                mock_feedback_instance.capture_trade_performance.assert_called_once()
                args = mock_feedback_instance.capture_trade_performance.call_args[0]
                assert args[0] == market_state  # market state
                assert args[1] == trading_result  # trading result
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_learning_loop_orchestrator_manages_complete_cycle(self, live_mode_config, mock_portfolio):
        """Test that learning loop orchestrator manages the complete learning cycle."""
        live_mode_config.parameters["enable_continuous_learning"] = True
        
        with patch('src.modes.live_mode.LearningLoopOrchestrator') as mock_orchestrator:
            mock_orchestrator_instance = mock_orchestrator.return_value
            mock_orchestrator_instance.check_learning_conditions.return_value = True
            mock_orchestrator_instance.execute_learning_cycle.return_value = {
                "data_preparation": {"status": "completed", "experiences_used": 1500},
                "training": {"status": "completed", "episodes": 100, "final_reward": 0.85},
                "validation": {"status": "completed", "improvement": 0.12},
                "deployment": {"status": "completed", "model_version": "v1.3.0"},
                "feedback_integration": {"status": "completed", "feedback_samples": 50}
            }
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Trigger learning cycle
            await live_mode.trigger_learning_cycle()
            
            # Should orchestrate complete cycle
            mock_orchestrator_instance.check_learning_conditions.assert_called_once()
            mock_orchestrator_instance.execute_learning_cycle.assert_called_once()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_model_deployment_automation_with_safety_checks(self, live_mode_config, mock_portfolio):
        """Test automated model deployment with comprehensive safety checks."""
        live_mode_config.parameters["enable_continuous_learning"] = True
        live_mode_config.parameters["enable_automated_deployment"] = True
        live_mode_config.parameters["deployment_safety_threshold"] = 0.05  # 5% improvement required
        
        with patch('src.modes.live_mode.ModelDeploymentAutomation') as mock_deployment:
            mock_deployment_instance = mock_deployment.return_value
            mock_deployment_instance.validate_new_model.return_value = {
                "performance_improvement": 0.08,  # 8% improvement
                "safety_checks_passed": True,
                "rollback_safety": True,
                "validation_score": 0.92
            }
            mock_deployment_instance.deploy_model.return_value = True
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Simulate new model ready for deployment
            new_model_info = {
                "model_path": "/models/v1.4.0/model.pth",
                "performance_metrics": {"reward": 0.88, "win_rate": 0.68},
                "training_episodes": 150
            }
            
            await live_mode.deploy_new_model(new_model_info)
            
            # Should validate and deploy model
            mock_deployment_instance.validate_new_model.assert_called_once()
            mock_deployment_instance.deploy_model.assert_called_once()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_continuous_learning_integration_with_experience_collection(self, live_mode_config, mock_portfolio):
        """Test integration between continuous learning and experience collection."""
        live_mode_config.parameters["enable_continuous_learning"] = True
        live_mode_config.parameters["enable_experience_collection"] = True
        live_mode_config.parameters["learning_trigger_threshold"] = 1000
        
        with patch('src.modes.live_mode.ContinuousLearningLoop') as mock_loop, \
             patch('src.modes.experience_collector.TradingExperienceCollector') as mock_collector:
            
            mock_collector_instance = mock_collector.return_value
            mock_collector_instance.get_buffer_size.return_value = 1200  # Above threshold
            mock_collector_instance.should_trigger_training.return_value = True
            
            mock_loop_instance = mock_loop.return_value
            mock_loop_instance.check_experience_buffer.return_value = True
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Simulate trading that adds experiences
            market_state = create_mock_market_state()
            
            with patch.object(live_mode, '_execute_live_trade') as mock_execute:
                mock_execute.return_value = TradingResult(
                    action=TradeAction.BUY,
                    token="TEST_TOKEN",
                    executed_at=datetime.now(),
                    price=1.0,
                    quantity=1000.0,
                    value_usd=1000.0,
                    success=True
                )
                
                await live_mode.process_tick(market_state)
                
                # Should check experience buffer for learning trigger
                mock_loop_instance.check_experience_buffer.assert_called()
    
    async def test_learning_cycle_error_handling_and_recovery(self, live_mode_config, mock_portfolio):
        """Test error handling and recovery during learning cycles."""
        live_mode_config.parameters["enable_continuous_learning"] = True
        
        with patch('src.modes.live_mode.ContinuousLearningLoop') as mock_loop:
            mock_loop_instance = mock_loop.return_value
            
            # Simulate learning cycle failure
            mock_loop_instance.trigger_learning_cycle.side_effect = Exception("Training failed")
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Should handle learning cycle failure gracefully
            try:
                await live_mode.trigger_learning_cycle()
            except Exception:
                pass  # Expected to catch and handle
            
            # Trading should continue despite learning failure
            market_state = create_mock_market_state()
            action = await live_mode.process_tick(market_state)
            assert action is not None  # Still able to trade
            
            # Should log error and continue
            assert live_mode.status == ModeStatus.ACTIVE
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_learning_performance_monitoring_and_rollback(self, live_mode_config, mock_portfolio):
        """Test performance monitoring and automatic rollback of poorly performing models."""
        live_mode_config.parameters["enable_continuous_learning"] = True
        live_mode_config.parameters["enable_performance_monitoring"] = True
        live_mode_config.parameters["performance_rollback_threshold"] = -0.10  # 10% degradation
        
        with patch('src.modes.live_mode.ModelPerformanceMonitor') as mock_monitor:
            mock_monitor_instance = mock_monitor.return_value
            mock_monitor_instance.evaluate_current_performance.return_value = {
                "performance_change": -0.15,  # 15% degradation
                "should_rollback": True,
                "previous_model_available": True
            }
            mock_monitor_instance.rollback_to_previous_model.return_value = True
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Simulate performance check triggering rollback
            await live_mode.check_model_performance()
            
            # Should detect degradation and rollback
            mock_monitor_instance.evaluate_current_performance.assert_called_once()
            mock_monitor_instance.rollback_to_previous_model.assert_called_once()
    
    @pytest.mark.skip(reason="Implementation pending - TDD")
    async def test_autonomous_learning_system_complete_integration(self, live_mode_config, mock_portfolio):
        """Test complete autonomous learning system integration."""
        live_mode_config.parameters.update({
            "enable_continuous_learning": True,
            "enable_experience_collection": True,
            "enable_model_hot_swapping": True,
            "enable_automated_deployment": True,
            "enable_performance_monitoring": True,
            "learning_check_frequency_seconds": 10,
            "learning_trigger_threshold": 500,
            "deployment_safety_threshold": 0.03
        })
        
        with patch('src.modes.live_mode.AutonomousLearningSystem') as mock_system:
            mock_system_instance = mock_system.return_value
            mock_system_instance.run_complete_cycle.return_value = {
                "cycle_id": "cycle_001",
                "phases": {
                    "data_collection": {"status": "completed", "experiences": 650},
                    "trigger_evaluation": {"status": "completed", "triggered": True},
                    "training": {"status": "completed", "episodes": 75},
                    "validation": {"status": "completed", "passed": True},
                    "deployment": {"status": "completed", "model_id": "v1.5.0"},
                    "monitoring": {"status": "active", "baseline_established": True}
                },
                "performance_improvement": 0.06,
                "cycle_duration_minutes": 12.5
            }
            
            live_mode = LiveMode(
                mode_id=uuid4(),
                config=live_mode_config,
                portfolio=mock_portfolio
            )
            
            await live_mode.initialize()
            await live_mode.start()
            
            # Simulate extended trading session that triggers multiple learning cycles
            market_state = create_mock_market_state()
            
            for i in range(10):  # Simulate 10 trading ticks
                market_state.price_usd = 1.0 + (i * 0.1)  # Varying price
                await live_mode.process_tick(market_state)
                await asyncio.sleep(0.01)  # Small delay
            
            # Should have initiated autonomous learning cycle
            mock_system_instance.run_complete_cycle.assert_called()
            
            # Verify system remains active and continues trading
            assert live_mode.status == ModeStatus.ACTIVE
            
            # Final tick should still work
            final_action = await live_mode.process_tick(market_state)
            assert final_action is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])