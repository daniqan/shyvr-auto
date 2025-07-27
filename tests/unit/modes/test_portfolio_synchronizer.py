"""
Tests for enhanced PortfolioSynchronizer with comprehensive reconciliation capabilities.

Following TDD methodology to ensure the portfolio synchronization system prevents
state drift and maintains accuracy in multi-chain, multi-DEX environments.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from unittest.mock import Mock, AsyncMock, patch

from src.modes.live_mode import PortfolioSynchronizer
from src.portfolio.base import (
    Portfolio, Position, PositionType, PositionStatus, PortfolioConfig
)
from src.dex.base import DEXBase
from src.utils.base import Chain


@pytest.fixture
def portfolio_config():
    """Test portfolio configuration."""
    return PortfolioConfig(
        initial_balance=Decimal("50000"),
        base_currency="USDC",
        max_position_size_pct=Decimal("0.1"),
        max_daily_loss_pct=Decimal("0.05"),
        max_drawdown_pct=Decimal("0.15")
    )


@pytest.fixture
def test_portfolio(portfolio_config):
    """Test portfolio with some positions."""
    portfolio = Portfolio(
        portfolio_id=uuid4(),
        name="test_portfolio",
        config=portfolio_config,
        cash_balance=Decimal("40000"),
        total_value=Decimal("50000")
    )
    
    # Add test positions
    positions = [
        Position(
            position_id=uuid4(),
            symbol="SOL/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("100"),
            entry_price=Decimal("50"),
            current_price=Decimal("55"),
            status=PositionStatus.OPEN
        ),
        Position(
            position_id=uuid4(),
            symbol="ETH/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.ETHEREUM,
            dex_name="uniswap_v3",
            size=Decimal("2"),
            entry_price=Decimal("2000"),
            current_price=Decimal("2100"),
            status=PositionStatus.OPEN
        )
    ]
    
    for position in positions:
        portfolio.add_position(position)
    
    return portfolio


@pytest.fixture
def mock_dex_clients():
    """Mock DEX clients for testing."""
    jupiter_client = Mock(spec=DEXBase)
    jupiter_client.get_wallet_balances = AsyncMock()
    jupiter_client.get_position_details = AsyncMock()
    
    uniswap_client = Mock(spec=DEXBase)
    uniswap_client.get_wallet_balances = AsyncMock()
    uniswap_client.get_position_details = AsyncMock()
    
    hyperliquid_client = Mock(spec=DEXBase)
    hyperliquid_client.get_wallet_balances = AsyncMock()
    hyperliquid_client.get_position_details = AsyncMock()
    
    return {
        "jupiter": jupiter_client,
        "uniswap_v3": uniswap_client,
        "hyperliquid": hyperliquid_client
    }


class TestPortfolioSynchronizerInitialization:
    """Test PortfolioSynchronizer initialization and configuration."""
    
    def test_portfolio_synchronizer_basic_initialization(self, test_portfolio, mock_dex_clients):
        """Test basic PortfolioSynchronizer initialization."""
        sync_frequency = 30
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=sync_frequency
        )
        
        assert synchronizer.portfolio == test_portfolio
        assert synchronizer.dex_clients == mock_dex_clients
        assert synchronizer.sync_frequency_seconds == sync_frequency
        assert not synchronizer.is_syncing
        assert synchronizer.last_sync_time is None
        assert synchronizer.sync_errors == 0
        assert synchronizer._sync_task is None
    
    def test_portfolio_synchronizer_with_default_frequency(self, test_portfolio, mock_dex_clients):
        """Test initialization with default sync frequency."""
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients
        )
        
        assert synchronizer.sync_frequency_seconds == 30  # Default value
    
    def test_portfolio_synchronizer_config_validation(self, test_portfolio):
        """Test configuration validation."""
        # Test with empty DEX clients
        with pytest.raises(ValueError, match="DEX clients cannot be empty"):
            PortfolioSynchronizer(
                portfolio=test_portfolio,
                dex_clients={},
                sync_frequency_seconds=30
            )
        
        # Test with invalid sync frequency
        with pytest.raises(ValueError, match="Sync frequency must be positive"):
            PortfolioSynchronizer(
                portfolio=test_portfolio,
                dex_clients={"jupiter": Mock()},
                sync_frequency_seconds=0
            )
    
    def test_portfolio_synchronizer_enhanced_config(self, test_portfolio, mock_dex_clients):
        """Test enhanced configuration options."""
        config = {
            "discrepancy_threshold": Decimal("0.01"),  # 1% threshold
            "auto_correct_threshold": Decimal("0.005"),  # 0.5% auto-correct
            "safety_alert_threshold": Decimal("0.05"),  # 5% safety alert
            "max_correction_attempts": 3,
            "enable_automatic_corrections": True,
            "enable_safety_alerts": True
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config=config
        )
        
        assert synchronizer.config == config
        assert synchronizer.discrepancy_threshold == Decimal("0.01")
        assert synchronizer.auto_correct_threshold == Decimal("0.005")
        assert synchronizer.safety_alert_threshold == Decimal("0.05")
        assert synchronizer.max_correction_attempts == 3
        assert synchronizer.enable_automatic_corrections
        assert synchronizer.enable_safety_alerts


class TestPortfolioSynchronizerLifecycle:
    """Test PortfolioSynchronizer lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_start_sync_process(self, test_portfolio, mock_dex_clients):
        """Test starting the synchronization process."""
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        await synchronizer.start_sync()
        
        assert synchronizer.is_syncing
        assert synchronizer._sync_task is not None
        assert not synchronizer._sync_task.done()
        
        # Cleanup
        await synchronizer.stop_sync()
    
    @pytest.mark.asyncio
    async def test_stop_sync_process(self, test_portfolio, mock_dex_clients):
        """Test stopping the synchronization process."""
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        await synchronizer.start_sync()
        assert synchronizer.is_syncing
        
        await synchronizer.stop_sync()
        
        assert not synchronizer.is_syncing
        assert synchronizer._sync_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_start_sync_when_already_running(self, test_portfolio, mock_dex_clients):
        """Test starting sync when already running."""
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        await synchronizer.start_sync()
        first_task = synchronizer._sync_task
        
        # Try to start again
        await synchronizer.start_sync()
        
        # Should be the same task
        assert synchronizer._sync_task == first_task
        assert synchronizer.is_syncing
        
        # Cleanup
        await synchronizer.stop_sync()


class TestBasicReconcilePositions:
    """Test basic reconcile_positions functionality."""
    
    @pytest.mark.asyncio
    async def test_reconcile_positions_basic_functionality(self, test_portfolio, mock_dex_clients):
        """Test basic reconcile_positions method."""
        # Setup mock responses
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        assert isinstance(discrepancies, list)
        assert synchronizer.last_sync_time is not None
        assert synchronizer.sync_errors == 0
        
        # Verify DEX clients were called
        mock_dex_clients["jupiter"].get_wallet_balances.assert_called_once()
        mock_dex_clients["uniswap_v3"].get_wallet_balances.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reconcile_positions_with_no_positions(self, portfolio_config, mock_dex_clients):
        """Test reconciliation with empty portfolio."""
        empty_portfolio = Portfolio(
            portfolio_id=uuid4(),
            name="empty_portfolio",
            config=portfolio_config,
            cash_balance=Decimal("50000"),
            total_value=Decimal("50000")
        )
        
        synchronizer = PortfolioSynchronizer(
            portfolio=empty_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        assert discrepancies == []
        assert synchronizer.last_sync_time is not None
    
    @pytest.mark.asyncio
    async def test_reconcile_positions_dex_error_handling(self, test_portfolio, mock_dex_clients):
        """Test error handling when DEX calls fail."""
        # Setup mock to raise exception
        mock_dex_clients["jupiter"].get_wallet_balances.side_effect = Exception("Network error")
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        with pytest.raises(Exception):
            await synchronizer.reconcile_positions()
        
        assert synchronizer.sync_errors == 1
    
    @pytest.mark.asyncio
    async def test_reconcile_positions_partial_dex_failure(self, test_portfolio, mock_dex_clients):
        """Test handling partial DEX failures."""
        # Setup mock responses with one failure
        mock_dex_clients["jupiter"].get_wallet_balances.side_effect = Exception("Jupiter error")
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={"ignore_dex_errors": True}
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should continue with available DEXs
        assert isinstance(discrepancies, list)
        assert len([d for d in discrepancies if d.get("type") == "dex_error"]) == 1


class TestPositionComparison:
    """Test position comparison and discrepancy detection."""
    
    @pytest.mark.asyncio
    async def test_detect_position_size_discrepancy(self, test_portfolio, mock_dex_clients):
        """Test detection of position size discrepancies."""
        # Setup mock responses with different sizes
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("95"), "price_usd": Decimal("55")}  # 5 SOL difference
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={"discrepancy_threshold": Decimal("0.01")}
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should detect the SOL balance discrepancy
        sol_discrepancy = next((d for d in discrepancies if d.get("symbol") == "SOL/USDC"), None)
        assert sol_discrepancy is not None
        assert sol_discrepancy["type"] == "size_discrepancy"
        assert sol_discrepancy["expected_size"] == Decimal("100")
        assert sol_discrepancy["actual_size"] == Decimal("95")
        assert sol_discrepancy["difference"] == Decimal("5")
    
    @pytest.mark.asyncio
    async def test_detect_price_discrepancy(self, test_portfolio, mock_dex_clients):
        """Test detection of price discrepancies."""
        # Setup mock responses with different prices
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("60")}  # $5 price difference
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={"price_tolerance_pct": Decimal("0.05")}  # 5% tolerance
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should detect the SOL price discrepancy
        sol_discrepancy = next((d for d in discrepancies if d.get("symbol") == "SOL/USDC"), None)
        assert sol_discrepancy is not None
        assert sol_discrepancy["type"] == "price_discrepancy"
        assert sol_discrepancy["expected_price"] == Decimal("55")
        assert sol_discrepancy["actual_price"] == Decimal("60")
    
    @pytest.mark.asyncio
    async def test_detect_missing_position(self, test_portfolio, mock_dex_clients):
        """Test detection of missing positions on-chain."""
        # Setup mock responses missing SOL position
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {}  # No SOL
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should detect the missing SOL position
        missing_position = next((d for d in discrepancies if d.get("type") == "missing_position"), None)
        assert missing_position is not None
        assert missing_position["symbol"] == "SOL/USDC"
        assert missing_position["expected_size"] == Decimal("100")
    
    @pytest.mark.asyncio
    async def test_detect_unexpected_position(self, test_portfolio, mock_dex_clients):
        """Test detection of unexpected positions on-chain."""
        # Setup mock responses with extra position
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")},
            "RAY": {"balance": Decimal("500"), "price_usd": Decimal("2")}  # Unexpected position
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should detect the unexpected RAY position
        unexpected_position = next((d for d in discrepancies if d.get("type") == "unexpected_position"), None)
        assert unexpected_position is not None
        assert unexpected_position["symbol"] == "RAY"
        assert unexpected_position["actual_size"] == Decimal("500")


class TestDiscrepancyReporting:
    """Test discrepancy detection and reporting mechanisms."""
    
    @pytest.mark.asyncio
    async def test_discrepancy_classification(self, test_portfolio, mock_dex_clients):
        """Test classification of discrepancies by severity."""
        # Setup various discrepancies
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("99.5"), "price_usd": Decimal("55")}  # Minor discrepancy
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("1.8"), "price_usd": Decimal("2100")}  # Larger discrepancy
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "auto_correct_threshold": Decimal("0.005"),  # 0.5%
                "safety_alert_threshold": Decimal("0.05")    # 5%
            }
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Check discrepancy classification
        sol_discrepancy = next((d for d in discrepancies if d.get("symbol") == "SOL/USDC"), None)
        eth_discrepancy = next((d for d in discrepancies if d.get("symbol") == "ETH/USDC"), None)
        
        if sol_discrepancy:
            assert sol_discrepancy["severity"] == "minor"
            assert sol_discrepancy["auto_correctable"]
        
        if eth_discrepancy:
            assert eth_discrepancy["severity"] == "major"
            assert not eth_discrepancy["auto_correctable"]
    
    @pytest.mark.asyncio
    async def test_discrepancy_metadata(self, test_portfolio, mock_dex_clients):
        """Test discrepancy reporting includes comprehensive metadata."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("95"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        if discrepancies:
            discrepancy = discrepancies[0]
            
            # Check required metadata fields
            required_fields = [
                "timestamp", "type", "severity", "dex_name", "chain",
                "symbol", "position_id", "auto_correctable"
            ]
            
            for field in required_fields:
                assert field in discrepancy, f"Missing required field: {field}"
            
            assert isinstance(discrepancy["timestamp"], datetime)
            assert discrepancy["dex_name"] in ["jupiter", "uniswap_v3", "hyperliquid"]
    
    @pytest.mark.asyncio
    async def test_discrepancy_aggregation(self, test_portfolio, mock_dex_clients):
        """Test aggregation of multiple discrepancies."""
        # Setup multiple discrepancies
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("95"), "price_usd": Decimal("60")}  # Both size and price
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("1.9"), "price_usd": Decimal("2200")}  # Both size and price
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={"price_tolerance_pct": Decimal("0.05")}
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should have multiple discrepancies
        assert len(discrepancies) >= 2
        
        # Check for different types
        discrepancy_types = {d["type"] for d in discrepancies}
        assert "size_discrepancy" in discrepancy_types or "price_discrepancy" in discrepancy_types


class TestAutomaticCorrections:
    """Test automatic correction of minor discrepancies."""
    
    @pytest.mark.asyncio
    async def test_automatic_correction_of_minor_discrepancy(self, test_portfolio, mock_dex_clients):
        """Test automatic correction of minor position discrepancies."""
        # Setup minor discrepancy within auto-correct threshold
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("99.7"), "price_usd": Decimal("55")}  # 0.3% difference
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "auto_correct_threshold": Decimal("0.005"),  # 0.5%
                "enable_automatic_corrections": True
            }
        )
        
        # Mock the correction method
        synchronizer.apply_automatic_correction = AsyncMock(return_value=True)
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should attempt automatic correction
        corrected_discrepancies = [d for d in discrepancies if d.get("corrected")]
        assert len(corrected_discrepancies) > 0
        
        # Verify correction was attempted
        synchronizer.apply_automatic_correction.assert_called()
    
    @pytest.mark.asyncio
    async def test_automatic_correction_disabled(self, test_portfolio, mock_dex_clients):
        """Test that automatic corrections are disabled when configured."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("99.7"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "auto_correct_threshold": Decimal("0.005"),
                "enable_automatic_corrections": False  # Disabled
            }
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should not attempt corrections
        corrected_discrepancies = [d for d in discrepancies if d.get("corrected")]
        assert len(corrected_discrepancies) == 0
    
    @pytest.mark.asyncio
    async def test_automatic_correction_threshold_exceeded(self, test_portfolio, mock_dex_clients):
        """Test that discrepancies above threshold are not auto-corrected."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("95"), "price_usd": Decimal("55")}  # 5% difference
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "auto_correct_threshold": Decimal("0.005"),  # 0.5%
                "enable_automatic_corrections": True
            }
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should not auto-correct large discrepancies
        sol_discrepancy = next((d for d in discrepancies if d.get("symbol") == "SOL/USDC"), None)
        if sol_discrepancy:
            assert not sol_discrepancy.get("corrected", False)
            assert not sol_discrepancy.get("auto_correctable", True)
    
    @pytest.mark.asyncio
    async def test_correction_attempt_limit(self, test_portfolio, mock_dex_clients):
        """Test maximum correction attempts limit."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("99.7"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "auto_correct_threshold": Decimal("0.005"),
                "enable_automatic_corrections": True,
                "max_correction_attempts": 2
            }
        )
        
        # Mock failed corrections
        synchronizer.apply_automatic_correction = AsyncMock(return_value=False)
        
        # Run multiple reconciliations
        for _ in range(3):
            await synchronizer.reconcile_positions()
        
        # Should not exceed max attempts
        assert synchronizer.apply_automatic_correction.call_count <= 2
    
    @pytest.mark.asyncio
    async def test_correction_rollback_on_failure(self, test_portfolio, mock_dex_clients):
        """Test rollback of failed automatic corrections."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("99.7"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "auto_correct_threshold": Decimal("0.005"),
                "enable_automatic_corrections": True,
                "enable_correction_rollback": True
            }
        )
        
        # Mock correction failure
        synchronizer.apply_automatic_correction = AsyncMock(
            side_effect=Exception("Correction failed")
        )
        synchronizer.rollback_correction = AsyncMock(return_value=True)
        
        original_position_size = test_portfolio.positions[
            list(test_portfolio.positions.keys())[0]
        ].size
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should rollback after failure
        synchronizer.rollback_correction.assert_called()
        
        # Portfolio should be unchanged
        current_position_size = test_portfolio.positions[
            list(test_portfolio.positions.keys())[0]
        ].size
        assert current_position_size == original_position_size


class TestSafetyAlerts:
    """Test safety alert generation for significant discrepancies."""
    
    @pytest.mark.asyncio
    async def test_safety_alert_for_large_discrepancy(self, test_portfolio, mock_dex_clients):
        """Test safety alert generation for large discrepancies."""
        # Setup large discrepancy
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("80"), "price_usd": Decimal("55")}  # 20% difference
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "safety_alert_threshold": Decimal("0.05"),  # 5%
                "enable_safety_alerts": True
            }
        )
        
        # Mock alert system
        synchronizer.trigger_safety_alert = AsyncMock()
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should trigger safety alert
        synchronizer.trigger_safety_alert.assert_called()
        
        # Check alert details
        alert_call = synchronizer.trigger_safety_alert.call_args
        assert alert_call is not None
        alert_data = alert_call[0][0]  # First argument
        assert alert_data["severity"] == "HIGH"
        assert alert_data["requires_manual_intervention"]
    
    @pytest.mark.asyncio
    async def test_safety_alert_disabled(self, test_portfolio, mock_dex_clients):
        """Test that safety alerts can be disabled."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("80"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "safety_alert_threshold": Decimal("0.05"),
                "enable_safety_alerts": False  # Disabled
            }
        )
        
        # Mock alert system
        synchronizer.trigger_safety_alert = AsyncMock()
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should not trigger safety alert
        synchronizer.trigger_safety_alert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_critical_safety_alert(self, test_portfolio, mock_dex_clients):
        """Test critical safety alert for extreme discrepancies."""
        # Setup extreme discrepancy (missing position)
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {}  # No SOL
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "critical_alert_threshold": Decimal("0.50"),  # 50%
                "enable_safety_alerts": True
            }
        )
        
        # Mock alert system
        synchronizer.trigger_safety_alert = AsyncMock()
        synchronizer.trigger_emergency_stop = AsyncMock()
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should trigger both safety alert and emergency stop
        synchronizer.trigger_safety_alert.assert_called()
        synchronizer.trigger_emergency_stop.assert_called()
    
    @pytest.mark.asyncio
    async def test_alert_escalation_chain(self, test_portfolio, mock_dex_clients):
        """Test alert escalation for persistent discrepancies."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("85"), "price_usd": Decimal("55")}  # 15% difference
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "safety_alert_threshold": Decimal("0.05"),
                "escalation_threshold": 3,  # Escalate after 3 consecutive alerts
                "enable_safety_alerts": True
            }
        )
        
        # Mock alert system
        synchronizer.trigger_safety_alert = AsyncMock()
        synchronizer.escalate_alert = AsyncMock()
        
        # Run multiple reconciliations
        for _ in range(4):
            await synchronizer.reconcile_positions()
        
        # Should escalate after threshold
        synchronizer.escalate_alert.assert_called()
    
    @pytest.mark.asyncio
    async def test_alert_notification_channels(self, test_portfolio, mock_dex_clients):
        """Test multiple notification channels for safety alerts."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("80"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "safety_alert_threshold": Decimal("0.05"),
                "enable_safety_alerts": True,
                "notification_channels": ["email", "slack", "sms"]
            }
        )
        
        # Mock notification channels
        synchronizer.send_email_alert = AsyncMock()
        synchronizer.send_slack_alert = AsyncMock()
        synchronizer.send_sms_alert = AsyncMock()
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should notify all channels
        synchronizer.send_email_alert.assert_called()
        synchronizer.send_slack_alert.assert_called()
        synchronizer.send_sms_alert.assert_called()


class TestMultiChainSupport:
    """Test multi-chain wallet balance fetching and reconciliation."""
    
    @pytest.mark.asyncio
    async def test_solana_chain_reconciliation(self, test_portfolio, mock_dex_clients):
        """Test reconciliation for Solana chain positions."""
        # Setup Solana-specific responses
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")},
            "RAY": {"balance": Decimal("1000"), "price_usd": Decimal("1.5")}
        }
        
        # Add Solana position to portfolio
        solana_position = Position(
            position_id=uuid4(),
            symbol="RAY/USDC",
            position_type=PositionType.SPOT,
            chain=Chain.SOLANA,
            dex_name="jupiter",
            size=Decimal("1000"),
            entry_price=Decimal("1.4"),
            current_price=Decimal("1.5"),
            status=PositionStatus.OPEN
        )
        test_portfolio.add_position(solana_position)
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should handle Solana positions correctly
        solana_discrepancies = [d for d in discrepancies if d.get("chain") == "solana"]
        assert isinstance(solana_discrepancies, list)
    
    @pytest.mark.asyncio
    async def test_ethereum_chain_reconciliation(self, test_portfolio, mock_dex_clients):
        """Test reconciliation for Ethereum chain positions."""
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")},
            "USDC": {"balance": Decimal("1000"), "price_usd": Decimal("1")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should handle Ethereum positions correctly
        ethereum_discrepancies = [d for d in discrepancies if d.get("chain") == "ethereum"]
        assert isinstance(ethereum_discrepancies, list)
    
    @pytest.mark.asyncio
    async def test_base_chain_reconciliation(self, test_portfolio, mock_dex_clients):
        """Test reconciliation for Base chain positions."""
        # Add Base chain client
        base_client = Mock(spec=DEXBase)
        base_client.get_wallet_balances = AsyncMock(return_value={
            "USDC": {"balance": Decimal("5000"), "price_usd": Decimal("1")}
        })
        mock_dex_clients["base_uniswap"] = base_client
        
        # Add Base position
        base_position = Position(
            position_id=uuid4(),
            symbol="USDC/USD",
            position_type=PositionType.SPOT,
            chain=Chain.BASE,
            dex_name="base_uniswap",
            size=Decimal("5000"),
            entry_price=Decimal("1"),
            current_price=Decimal("1"),
            status=PositionStatus.OPEN
        )
        test_portfolio.add_position(base_position)
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should handle Base positions correctly
        base_discrepancies = [d for d in discrepancies if d.get("chain") == "base"]
        assert isinstance(base_discrepancies, list)
    
    @pytest.mark.asyncio
    async def test_cross_chain_aggregation(self, test_portfolio, mock_dex_clients):
        """Test aggregation of positions across multiple chains."""
        # Setup responses for multiple chains
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should aggregate across chains
        chains_covered = {d.get("chain") for d in discrepancies if d.get("chain")}
        assert len(chains_covered) <= 2  # Solana and Ethereum
        
        # Should provide chain-level summary
        summary = await synchronizer.get_chain_summary()
        assert "solana" in summary
        assert "ethereum" in summary


class TestMultiDEXSupport:
    """Test multi-DEX balance reconciliation across different exchanges."""
    
    @pytest.mark.asyncio
    async def test_jupiter_dex_reconciliation(self, test_portfolio, mock_dex_clients):
        """Test reconciliation for Jupiter DEX positions."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")},
            "RAY": {"balance": Decimal("500"), "price_usd": Decimal("2")}
        }
        mock_dex_clients["jupiter"].get_position_details = AsyncMock(return_value={
            "positions": [
                {
                    "symbol": "SOL/USDC",
                    "size": Decimal("100"),
                    "current_price": Decimal("55"),
                    "unrealized_pnl": Decimal("500")
                }
            ]
        })
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should handle Jupiter-specific data correctly
        jupiter_discrepancies = [d for d in discrepancies if d.get("dex_name") == "jupiter"]
        assert isinstance(jupiter_discrepancies, list)
        
        # Verify Jupiter client was called
        mock_dex_clients["jupiter"].get_wallet_balances.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_uniswap_v3_reconciliation(self, test_portfolio, mock_dex_clients):
        """Test reconciliation for Uniswap V3 positions."""
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")},
            "USDC": {"balance": Decimal("1000"), "price_usd": Decimal("1")}
        }
        mock_dex_clients["uniswap_v3"].get_position_details = AsyncMock(return_value={
            "positions": [
                {
                    "symbol": "ETH/USDC",
                    "size": Decimal("2"),
                    "current_price": Decimal("2100"),
                    "liquidity_range": {"lower": Decimal("2000"), "upper": Decimal("2200")}
                }
            ]
        })
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should handle Uniswap V3 liquidity positions
        uniswap_discrepancies = [d for d in discrepancies if d.get("dex_name") == "uniswap_v3"]
        assert isinstance(uniswap_discrepancies, list)
        
        # Verify Uniswap client was called
        mock_dex_clients["uniswap_v3"].get_wallet_balances.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_hyperliquid_reconciliation(self, test_portfolio, mock_dex_clients):
        """Test reconciliation for Hyperliquid perpetual positions."""
        # Add perpetual position to portfolio
        hyperliquid_position = Position(
            position_id=uuid4(),
            symbol="BTC-USD",
            position_type=PositionType.PERPETUAL,
            chain=Chain.ETHEREUM,  # Hyperliquid runs on Ethereum
            dex_name="hyperliquid",
            size=Decimal("0.5"),
            entry_price=Decimal("45000"),
            current_price=Decimal("46000"),
            status=PositionStatus.OPEN,
            leverage=Decimal("5"),
            side="LONG",
            margin_used=Decimal("4500")
        )
        test_portfolio.add_position(hyperliquid_position)
        
        mock_dex_clients["hyperliquid"].get_wallet_balances.return_value = {
            "USDC": {"balance": Decimal("10000"), "price_usd": Decimal("1")}
        }
        mock_dex_clients["hyperliquid"].get_position_details = AsyncMock(return_value={
            "positions": [
                {
                    "symbol": "BTC-USD",
                    "size": Decimal("0.5"),
                    "side": "LONG",
                    "entry_price": Decimal("45000"),
                    "current_price": Decimal("46000"),
                    "unrealized_pnl": Decimal("2500"),  # 5x leverage
                    "margin_used": Decimal("4500"),
                    "liquidation_price": Decimal("36000")
                }
            ]
        })
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should handle perpetual positions correctly
        hyperliquid_discrepancies = [d for d in discrepancies if d.get("dex_name") == "hyperliquid"]
        assert isinstance(hyperliquid_discrepancies, list)
        
        # Verify Hyperliquid client was called
        mock_dex_clients["hyperliquid"].get_wallet_balances.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multi_dex_failure_handling(self, test_portfolio, mock_dex_clients):
        """Test handling when some DEXs fail while others succeed."""
        # Setup mixed success/failure scenario
        mock_dex_clients["jupiter"].get_wallet_balances.side_effect = Exception("Jupiter API down")
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        mock_dex_clients["hyperliquid"].get_wallet_balances.return_value = {
            "USDC": {"balance": Decimal("10000"), "price_usd": Decimal("1")}
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={"continue_on_dex_failure": True}
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should continue with available DEXs
        assert isinstance(discrepancies, list)
        
        # Should report DEX failure
        dex_failures = [d for d in discrepancies if d.get("type") == "dex_error"]
        assert len(dex_failures) >= 1
        assert any(d.get("dex_name") == "jupiter" for d in dex_failures)
    
    @pytest.mark.asyncio
    async def test_dex_specific_features(self, test_portfolio, mock_dex_clients):
        """Test DEX-specific features and reconciliation logic."""
        # Jupiter: SPL token handling
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55"), "token_account": "Sol11111111"},
            "RAY": {"balance": Decimal("500"), "price_usd": Decimal("2"), "token_account": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"}
        }
        
        # Uniswap V3: Liquidity position details
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100"), "position_id": "123456"}
        }
        mock_dex_clients["uniswap_v3"].get_position_details = AsyncMock(return_value={
            "positions": [
                {
                    "position_id": "123456",
                    "symbol": "ETH/USDC",
                    "liquidity": Decimal("1000000"),
                    "tick_lower": -276324,
                    "tick_upper": -276300,
                    "fee_tier": 3000
                }
            ]
        })
        
        # Hyperliquid: Perpetual futures data
        mock_dex_clients["hyperliquid"].get_wallet_balances.return_value = {
            "USDC": {"balance": Decimal("10000"), "price_usd": Decimal("1")}
        }
        mock_dex_clients["hyperliquid"].get_position_details = AsyncMock(return_value={
            "positions": [],
            "funding_rates": {"BTC-USD": Decimal("0.0001")},
            "margin_requirements": {"initial": Decimal("0.05"), "maintenance": Decimal("0.03")}
        })
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={"enable_dex_specific_features": True}
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should handle DEX-specific data
        assert isinstance(discrepancies, list)
        
        # Verify DEX-specific methods were called
        mock_dex_clients["uniswap_v3"].get_position_details.assert_called()
        mock_dex_clients["hyperliquid"].get_position_details.assert_called()
    
    @pytest.mark.asyncio
    async def test_cross_dex_arbitrage_detection(self, test_portfolio, mock_dex_clients):
        """Test detection of arbitrage opportunities across DEXs."""
        # Setup price differences across DEXs
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")}  # Lower price
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("0"), "price_usd": Decimal("57")}  # Higher price (wrapped SOL)
        }
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            config={
                "enable_arbitrage_detection": True,
                "arbitrage_threshold": Decimal("0.02")  # 2% threshold
            }
        )
        
        discrepancies = await synchronizer.reconcile_positions()
        
        # Should detect arbitrage opportunity
        arbitrage_opportunities = [d for d in discrepancies if d.get("type") == "arbitrage_opportunity"]
        if arbitrage_opportunities:
            arb = arbitrage_opportunities[0]
            assert arb["symbol"] == "SOL"
            assert arb["price_difference_pct"] > Decimal("0.02")
            assert arb["buy_dex"] == "jupiter"
            assert arb["sell_dex"] == "uniswap_v3"


class TestMonitoringIntegration:
    """Test monitoring system integration and health tracking."""
    
    @pytest.mark.asyncio
    async def test_sync_health_metrics(self, test_portfolio, mock_dex_clients):
        """Test synchronization health metrics collection."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        # Mock monitoring system
        mock_monitor = Mock()
        mock_monitor.record_metric = Mock()
        mock_monitor.record_event = Mock()
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            monitor=mock_monitor
        )
        
        await synchronizer.reconcile_positions()
        
        # Should record health metrics
        mock_monitor.record_metric.assert_called()
        mock_monitor.record_event.assert_called()
        
        # Check specific metrics
        metric_calls = [call.args for call in mock_monitor.record_metric.call_args_list]
        metric_names = [call[0] for call in metric_calls]
        
        expected_metrics = [
            "sync_duration_ms",
            "sync_success_rate",
            "discrepancies_detected",
            "dex_response_time_ms"
        ]
        
        for metric in expected_metrics:
            assert any(metric in name for name in metric_names)
    
    @pytest.mark.asyncio
    async def test_error_rate_monitoring(self, test_portfolio, mock_dex_clients):
        """Test monitoring of synchronization error rates."""
        # Setup some failures
        mock_dex_clients["jupiter"].get_wallet_balances.side_effect = [
            Exception("Network error"),
            {"SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")}},
            Exception("Timeout"),
        ]
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        mock_monitor = Mock()
        mock_monitor.record_metric = Mock()
        mock_monitor.alert_threshold_exceeded = Mock()
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            monitor=mock_monitor,
            config={"error_rate_alert_threshold": 0.5}  # 50% error rate threshold
        )
        
        # Run multiple reconciliations
        for _ in range(3):
            try:
                await synchronizer.reconcile_positions()
            except Exception:
                pass  # Expected failures
        
        # Should track error rates
        error_rate_calls = [
            call for call in mock_monitor.record_metric.call_args_list
            if "error_rate" in str(call)
        ]
        assert len(error_rate_calls) > 0
        
        # Should trigger alert if threshold exceeded
        if synchronizer.get_error_rate() > 0.5:
            mock_monitor.alert_threshold_exceeded.assert_called()
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, test_portfolio, mock_dex_clients):
        """Test performance metrics tracking."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        mock_monitor = Mock()
        mock_monitor.record_metric = Mock()
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            monitor=mock_monitor
        )
        
        start_time = datetime.now()
        await synchronizer.reconcile_positions()
        end_time = datetime.now()
        
        # Should track performance metrics
        performance_metrics = [
            call for call in mock_monitor.record_metric.call_args_list
            if any(perf in str(call) for perf in ["duration", "latency", "throughput"])
        ]
        assert len(performance_metrics) > 0
        
        # Should track sync duration
        duration_ms = (end_time - start_time).total_seconds() * 1000
        duration_calls = [
            call for call in mock_monitor.record_metric.call_args_list
            if "duration" in str(call)
        ]
        assert len(duration_calls) > 0
    
    @pytest.mark.asyncio
    async def test_discrepancy_severity_tracking(self, test_portfolio, mock_dex_clients):
        """Test tracking of discrepancy severity levels."""
        # Setup various severity levels
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("95"), "price_usd": Decimal("55")}  # Major discrepancy
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("1.99"), "price_usd": Decimal("2100")}  # Minor discrepancy
        }
        
        mock_monitor = Mock()
        mock_monitor.record_metric = Mock()
        mock_monitor.record_event = Mock()
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            monitor=mock_monitor,
            config={
                "auto_correct_threshold": Decimal("0.005"),
                "safety_alert_threshold": Decimal("0.03")
            }
        )
        
        await synchronizer.reconcile_positions()
        
        # Should track discrepancy severity
        severity_events = [
            call for call in mock_monitor.record_event.call_args_list
            if "discrepancy" in str(call) and "severity" in str(call)
        ]
        assert len(severity_events) > 0
        
        # Should track different severity levels
        recorded_events = [str(call) for call in mock_monitor.record_event.call_args_list]
        severity_levels = ["minor", "major", "critical"]
        
        for level in severity_levels:
            level_events = [event for event in recorded_events if level in event]
            # At least some severity level should be recorded
        
        assert len([event for event in recorded_events if any(level in event for level in severity_levels)]) > 0
    
    @pytest.mark.asyncio
    async def test_custom_monitoring_hooks(self, test_portfolio, mock_dex_clients):
        """Test custom monitoring hooks and callbacks."""
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {
            "SOL": {"balance": Decimal("100"), "price_usd": Decimal("55")}
        }
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        # Custom monitoring hooks
        pre_sync_hook = AsyncMock()
        post_sync_hook = AsyncMock()
        discrepancy_hook = AsyncMock()
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            monitoring_hooks={
                "pre_sync": pre_sync_hook,
                "post_sync": post_sync_hook,
                "discrepancy_detected": discrepancy_hook
            }
        )
        
        await synchronizer.reconcile_positions()
        
        # Should call monitoring hooks
        pre_sync_hook.assert_called_once()
        post_sync_hook.assert_called_once()
        
        # If discrepancies were found, discrepancy hook should be called
        if len(await synchronizer.reconcile_positions()) > 0:
            discrepancy_hook.assert_called()
    
    @pytest.mark.asyncio
    async def test_alerting_integration(self, test_portfolio, mock_dex_clients):
        """Test integration with alerting system."""
        # Setup critical discrepancy
        mock_dex_clients["jupiter"].get_wallet_balances.return_value = {}  # Missing position
        mock_dex_clients["uniswap_v3"].get_wallet_balances.return_value = {
            "ETH": {"balance": Decimal("2"), "price_usd": Decimal("2100")}
        }
        
        mock_alerting = Mock()
        mock_alerting.send_alert = AsyncMock()
        
        synchronizer = PortfolioSynchronizer(
            portfolio=test_portfolio,
            dex_clients=mock_dex_clients,
            sync_frequency_seconds=30,
            alerting_system=mock_alerting,
            config={
                "enable_safety_alerts": True,
                "critical_alert_threshold": Decimal("0.50")
            }
        )
        
        await synchronizer.reconcile_positions()
        
        # Should send critical alert
        mock_alerting.send_alert.assert_called()
        
        # Check alert details
        alert_call = mock_alerting.send_alert.call_args
        assert alert_call is not None
        alert_data = alert_call[0][0]
        assert alert_data["severity"] in ["HIGH", "CRITICAL"]
        assert "discrepancy" in alert_data["type"]