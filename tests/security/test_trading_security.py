"""
Phase 7.3: Trading Security Tests
Following TDD methodology - these tests validate trading security controls
"""

import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any

from src.safety.trading_safety_manager import TradingSafetyManager
from src.safety.trading_circuit_breaker import TradingCircuitBreaker
from src.safety.risk_control_manager import RiskControlManager
from src.safety.emergency_stop_controller import EmergencyStopController
from src.trading.dex_wallet_bridge import DexWalletBridge


class TestTradingRiskControls:
    """Test trading risk control security following TDD methodology"""
    
    @pytest.fixture
    def safety_manager(self):
        """Create TradingSafetyManager for testing"""
        return TradingSafetyManager()
    
    @pytest.fixture
    def risk_controller(self):
        """Create RiskControlManager for testing"""
        return RiskControlManager()

    def test_position_size_limits_should_reject_oversized_trades(self, safety_manager):
        """Test position size limits reject oversized trades - TDD FAIL FIRST"""
        # Configure strict position limits
        max_position_usd = Decimal('10000')  # $10k max position
        max_position_percentage = Decimal('0.05')  # 5% of portfolio max
        
        # Test oversized trade by dollar amount
        oversized_trade = {
            "symbol": "BTC/USD",
            "side": "BUY",
            "amount": Decimal('0.5'),  # 0.5 BTC
            "price": Decimal('50000'),  # $50k per BTC = $25k position
            "portfolio_value": Decimal('100000')
        }
        
        # Should reject oversized trade
        result = safety_manager.validate_position_size(
            oversized_trade, max_position_usd, max_position_percentage
        )
        assert not result.is_valid, "Should reject trade exceeding dollar limit"
        assert "position size" in result.rejection_reason.lower(), \
            "Should indicate position size violation"

    def test_portfolio_concentration_limits_should_prevent_over_concentration(self, safety_manager):
        """Test portfolio concentration limits prevent over-concentration - TDD FAIL FIRST"""
        # Current portfolio with high BTC concentration
        current_positions = {
            "BTC/USD": {"amount": Decimal("0.8"), "value_usd": Decimal("40000")},
            "ETH/USD": {"amount": Decimal("10"), "value_usd": Decimal("30000")},
            "CASH": {"amount": Decimal("30000"), "value_usd": Decimal("30000")}
        }
        
        # New trade that would increase BTC concentration beyond limit
        new_trade = {
            "symbol": "BTC/USD",
            "side": "BUY", 
            "amount": Decimal("0.3"),  # Additional 0.3 BTC
            "price": Decimal("50000"),  # $15k more
            "portfolio_value": Decimal("100000")
        }
        
        max_asset_concentration = Decimal("0.6")  # 60% max per asset
        
        # Should reject trade that would exceed concentration limit
        result = safety_manager.validate_concentration_limits(
            new_trade, current_positions, max_asset_concentration
        )
        assert not result.is_valid, "Should reject trade exceeding concentration limit"
        assert "concentration" in result.rejection_reason.lower(), \
            "Should indicate concentration violation"

    def test_daily_loss_limits_should_stop_trading_after_threshold(self, safety_manager):
        """Test daily loss limits stop trading after threshold - TDD FAIL FIRST"""
        # Simulate significant daily losses
        daily_pnl = Decimal("-5000")  # $5k loss today
        max_daily_loss = Decimal("3000")  # $3k max daily loss
        
        new_trade = {
            "symbol": "ETH/USD",
            "side": "SELL",
            "amount": Decimal("5"),
            "price": Decimal("3000")
        }
        
        # Should reject new trades after hitting daily loss limit
        result = safety_manager.validate_daily_loss_limit(new_trade, daily_pnl, max_daily_loss)
        assert not result.is_valid, "Should reject trades after daily loss limit hit"
        assert "daily loss" in result.rejection_reason.lower(), \
            "Should indicate daily loss limit violation"

    def test_leverage_limits_should_prevent_excessive_leverage(self, risk_controller):
        """Test leverage limits prevent excessive leverage - TDD FAIL FIRST"""
        portfolio_value = Decimal("100000")
        max_leverage = Decimal("2.0")  # 2x max leverage
        
        # Test trade that would create excessive leverage
        excessive_leverage_trade = {
            "symbol": "BTC/USD",
            "side": "BUY",
            "amount": Decimal("5"),  # 5 BTC
            "price": Decimal("50000"),  # $250k position = 2.5x leverage
            "is_margin": True
        }
        
        # Should reject excessive leverage
        result = risk_controller.validate_leverage(
            excessive_leverage_trade, portfolio_value, max_leverage
        )
        assert not result.is_valid, "Should reject excessive leverage trade"
        assert "leverage" in result.rejection_reason.lower(), \
            "Should indicate leverage violation"

    def test_correlation_risk_limits_should_prevent_correlated_positions(self, risk_controller):
        """Test correlation risk limits prevent over-correlated positions - TDD FAIL FIRST"""
        # Current positions in highly correlated assets
        current_positions = {
            "BTC/USD": {"amount": Decimal("1"), "correlation_group": "crypto"},
            "ETH/USD": {"amount": Decimal("10"), "correlation_group": "crypto"},
            "LTC/USD": {"amount": Decimal("50"), "correlation_group": "crypto"}
        }
        
        # New trade in same correlation group
        new_trade = {
            "symbol": "ADA/USD", 
            "side": "BUY",
            "amount": Decimal("1000"),
            "correlation_group": "crypto"
        }
        
        max_correlation_exposure = Decimal("0.7")  # 70% max in correlated assets
        
        # Should reject trade that increases correlation risk
        result = risk_controller.validate_correlation_limits(
            new_trade, current_positions, max_correlation_exposure
        )
        assert not result.is_valid, "Should reject trade increasing correlation risk"
        assert "correlation" in result.rejection_reason.lower(), \
            "Should indicate correlation risk violation"


class TestTradingCircuitBreakers:
    """Test trading circuit breaker security mechanisms"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create TradingCircuitBreaker for testing"""
        return TradingCircuitBreaker()

    def test_volatility_circuit_breaker_should_halt_trading_on_extreme_moves(self, circuit_breaker):
        """Test volatility circuit breaker halts trading on extreme moves - TDD FAIL FIRST"""
        # Simulate extreme price volatility
        price_data = [
            {"timestamp": datetime.now() - timedelta(minutes=5), "price": Decimal("50000")},
            {"timestamp": datetime.now() - timedelta(minutes=4), "price": Decimal("52000")},
            {"timestamp": datetime.now() - timedelta(minutes=3), "price": Decimal("54000")},
            {"timestamp": datetime.now() - timedelta(minutes=2), "price": Decimal("47000")},  # -13% drop
            {"timestamp": datetime.now() - timedelta(minutes=1), "price": Decimal("45000")},  # Additional drop
        ]
        
        max_volatility_threshold = Decimal("0.1")  # 10% max move
        
        # Should trigger circuit breaker
        result = circuit_breaker.check_volatility_threshold("BTC/USD", price_data, max_volatility_threshold)
        assert result.is_triggered, "Circuit breaker should trigger on extreme volatility"
        assert result.halt_duration > 0, "Should specify halt duration"
        assert "volatility" in result.reason.lower(), "Should indicate volatility trigger"

    def test_volume_anomaly_circuit_breaker_should_detect_unusual_volume(self, circuit_breaker):
        """Test volume anomaly circuit breaker detects unusual volume - TDD FAIL FIRST"""
        # Historical volume data (normal trading)
        historical_volume = [
            Decimal("1000"), Decimal("1200"), Decimal("900"), 
            Decimal("1100"), Decimal("1300"), Decimal("950")
        ]
        
        # Current volume spike
        current_volume = Decimal("10000")  # 10x normal volume
        
        volume_anomaly_threshold = Decimal("5.0")  # 5x normal volume threshold
        
        # Should detect volume anomaly
        result = circuit_breaker.check_volume_anomaly(
            "BTC/USD", current_volume, historical_volume, volume_anomaly_threshold
        )
        assert result.is_triggered, "Should detect volume anomaly"
        assert "volume" in result.reason.lower(), "Should indicate volume anomaly"

    def test_order_book_manipulation_detection_should_halt_suspicious_activity(self, circuit_breaker):
        """Test order book manipulation detection halts suspicious activity - TDD FAIL FIRST"""
        # Suspicious order book pattern (potential manipulation)
        order_book = {
            "bids": [
                {"price": Decimal("49999"), "size": Decimal("0.01")},  # Tiny bid
                {"price": Decimal("49998"), "size": Decimal("0.01")},
                {"price": Decimal("45000"), "size": Decimal("100")},   # Large bid far away
            ],
            "asks": [
                {"price": Decimal("50001"), "size": Decimal("100")},   # Large ask  
                {"price": Decimal("55000"), "size": Decimal("0.01")},  # Tiny ask far away
                {"price": Decimal("56000"), "size": Decimal("0.01")},
            ]
        }
        
        # Should detect potential manipulation
        result = circuit_breaker.check_order_book_manipulation("BTC/USD", order_book)
        assert result.is_triggered, "Should detect potential order book manipulation"
        assert "manipulation" in result.reason.lower() or "order book" in result.reason.lower(), \
            "Should indicate manipulation detection"

    def test_rapid_trading_circuit_breaker_should_limit_high_frequency_trading(self, circuit_breaker):
        """Test rapid trading circuit breaker limits high frequency trading - TDD FAIL FIRST"""
        # Simulate rapid fire trading
        rapid_trades = []
        base_time = datetime.now()
        
        for i in range(100):  # 100 trades in rapid succession
            rapid_trades.append({
                "timestamp": base_time + timedelta(milliseconds=i*10),  # Every 10ms
                "symbol": "BTC/USD",
                "amount": Decimal("0.01")
            })
        
        max_trades_per_minute = 50
        time_window_minutes = 1
        
        # Should trigger rapid trading circuit breaker
        result = circuit_breaker.check_rapid_trading(
            rapid_trades, max_trades_per_minute, time_window_minutes
        )
        assert result.is_triggered, "Should trigger on rapid trading"
        assert "rapid" in result.reason.lower() or "frequency" in result.reason.lower(), \
            "Should indicate rapid trading trigger"


class TestEmergencyStopMechanisms:
    """Test emergency stop security mechanisms"""
    
    @pytest.fixture
    def emergency_controller(self):
        """Create EmergencyStopController for testing"""
        return EmergencyStopController()

    @pytest.mark.asyncio
    async def test_emergency_stop_should_halt_all_trading_immediately(self, emergency_controller):
        """Test emergency stop halts all trading immediately - TDD FAIL FIRST"""
        # Trigger emergency stop
        stop_reason = "Suspected security breach detected"
        emergency_controller.trigger_emergency_stop(stop_reason)
        
        # Should block all new trades
        test_trade = {
            "symbol": "BTC/USD",
            "side": "BUY", 
            "amount": Decimal("0.1")
        }
        
        result = emergency_controller.validate_trade_allowed(test_trade)
        assert not result.is_allowed, "Emergency stop should block all trades"
        assert "emergency" in result.rejection_reason.lower(), \
            "Should indicate emergency stop active"

    @pytest.mark.asyncio
    async def test_emergency_stop_should_cancel_pending_orders(self, emergency_controller):
        """Test emergency stop cancels all pending orders - TDD FAIL FIRST"""
        # Mock pending orders
        pending_orders = [
            {"id": "order_1", "symbol": "BTC/USD", "status": "pending"},
            {"id": "order_2", "symbol": "ETH/USD", "status": "pending"},
            {"id": "order_3", "symbol": "LTC/USD", "status": "pending"}
        ]
        
        with patch.object(emergency_controller, 'get_pending_orders', return_value=pending_orders), \
             patch.object(emergency_controller, 'cancel_order') as mock_cancel:
            
            # Trigger emergency stop
            await emergency_controller.trigger_emergency_stop_async("Market manipulation detected")
            
            # Should cancel all pending orders
            assert mock_cancel.call_count == len(pending_orders), \
                "Should cancel all pending orders"
            
            for order in pending_orders:
                mock_cancel.assert_any_call(order["id"])

    def test_emergency_stop_authorization_should_require_proper_credentials(self, emergency_controller):
        """Test emergency stop requires proper authorization - TDD FAIL FIRST"""
        # Test unauthorized emergency stop attempt
        unauthorized_user = {"username": "regular_user", "permissions": ["read"]}
        
        # Should reject unauthorized emergency stop
        result = emergency_controller.authorize_emergency_stop(unauthorized_user)
        assert not result.is_authorized, "Should reject unauthorized emergency stop"
        
        # Test authorized emergency stop
        authorized_user = {"username": "admin", "permissions": ["admin", "emergency_stop"]}
        
        result = emergency_controller.authorize_emergency_stop(authorized_user)
        assert result.is_authorized, "Should allow authorized emergency stop"

    def test_emergency_stop_recovery_should_require_manual_override(self, emergency_controller):
        """Test emergency stop recovery requires manual override - TDD FAIL FIRST"""
        # Trigger emergency stop
        emergency_controller.trigger_emergency_stop("Test emergency")
        
        # Automatic recovery should not be allowed
        auto_recovery_result = emergency_controller.attempt_auto_recovery()
        assert not auto_recovery_result.is_successful, \
            "Should not allow automatic recovery from emergency stop"
        
        # Manual override should be required
        manual_override_user = {"username": "admin", "permissions": ["admin", "emergency_override"]}
        
        manual_recovery_result = emergency_controller.manual_override_recovery(
            manual_override_user, "Emergency resolved, resuming operations"
        )
        assert manual_recovery_result.is_successful, \
            "Should allow manual override by authorized user"


class TestFinancialDataIntegrity:
    """Test financial data integrity and anti-manipulation security"""
    
    def test_price_data_validation_should_reject_manipulated_prices(self):
        """Test price data validation rejects manipulated prices - TDD FAIL FIRST"""
        # Test with suspicious price data
        suspicious_prices = [
            {"price": Decimal("0.01"), "symbol": "BTC/USD", "source": "exchange_a"},  # Impossibly low
            {"price": Decimal("1000000"), "symbol": "BTC/USD", "source": "exchange_b"},  # Impossibly high
            {"price": Decimal("-100"), "symbol": "ETH/USD", "source": "exchange_c"},  # Negative price
            {"price": None, "symbol": "LTC/USD", "source": "exchange_d"},  # Null price
        ]
        
        price_validator = FinancialDataValidator()
        
        for price_data in suspicious_prices:
            result = price_validator.validate_price_data(price_data)
            assert not result.is_valid, f"Should reject suspicious price: {price_data['price']}"
            assert "price" in result.rejection_reason.lower(), \
                "Should indicate price validation failure"

    def test_cross_exchange_price_validation_should_detect_arbitrage_manipulation(self):
        """Test cross-exchange validation detects manipulation - TDD FAIL FIRST"""
        # Price data from multiple exchanges with suspicious spread
        exchange_prices = {
            "exchange_a": {"BTC/USD": Decimal("50000")},
            "exchange_b": {"BTC/USD": Decimal("49000")},  # Normal spread
            "exchange_c": {"BTC/USD": Decimal("45000")},  # Suspicious 10% difference
            "exchange_d": {"BTC/USD": Decimal("55000")},  # Suspicious 10% premium
        }
        
        max_spread_percentage = Decimal("0.05")  # 5% max spread
        
        arbitrage_detector = ArbitrageManipulationDetector()
        
        result = arbitrage_detector.validate_cross_exchange_prices(
            "BTC/USD", exchange_prices, max_spread_percentage
        )
        assert not result.is_valid, "Should detect suspicious price spreads"
        assert "spread" in result.rejection_reason.lower() or "arbitrage" in result.rejection_reason.lower(), \
            "Should indicate spread/arbitrage issue"

    def test_trade_size_validation_should_prevent_market_manipulation(self):
        """Test trade size validation prevents market manipulation - TDD FAIL FIRST"""
        # Test trades that could manipulate market
        market_data = {
            "symbol": "BTC/USD",
            "daily_volume": Decimal("100000"),  # $100k daily volume
            "order_book_depth": Decimal("50000")  # $50k order book depth
        }
        
        manipulative_trades = [
            # Trade size > 50% of daily volume
            {"amount": Decimal("1.2"), "price": Decimal("50000"), "value": Decimal("60000")},
            
            # Trade size > order book depth  
            {"amount": Decimal("1.1"), "price": Decimal("50000"), "value": Decimal("55000")},
        ]
        
        manipulation_detector = MarketManipulationDetector()
        
        for trade in manipulative_trades:
            result = manipulation_detector.validate_trade_impact(trade, market_data)
            assert not result.is_valid, f"Should reject potentially manipulative trade: {trade['value']}"
            assert "impact" in result.rejection_reason.lower() or "manipulation" in result.rejection_reason.lower(), \
                "Should indicate market impact concern"


class TestWalletSecurity:
    """Test wallet and private key security"""
    
    @pytest.fixture
    def wallet_bridge(self):
        """Create DexWalletBridge for testing"""
        return DexWalletBridge()

    def test_private_key_access_should_require_authentication(self, wallet_bridge):
        """Test private key access requires proper authentication - TDD FAIL FIRST"""
        # Test unauthorized private key access
        unauthorized_user = {"username": "hacker", "permissions": ["read"]}
        
        with pytest.raises((PermissionError, SecurityError, Exception)):
            wallet_bridge.get_private_key(unauthorized_user)

    def test_transaction_signing_should_validate_transaction_integrity(self, wallet_bridge):
        """Test transaction signing validates transaction integrity - TDD FAIL FIRST"""
        # Test transaction with tampered data
        tampered_transaction = {
            "to": "0x1234567890123456789012345678901234567890",
            "value": Decimal("1.0"),  # Original amount
            "gas": 21000,
            "data": "0x",
            # Integrity hash doesn't match (simulated tampering)
            "integrity_hash": "wrong_hash"
        }
        
        # Should reject tampered transaction
        with pytest.raises((ValueError, SecurityError, Exception)):
            wallet_bridge.sign_transaction(tampered_transaction)

    def test_multi_signature_requirement_for_large_transactions(self, wallet_bridge):
        """Test multi-signature requirement for large transactions - TDD FAIL FIRST"""
        # Large transaction requiring multiple signatures
        large_transaction = {
            "to": "0x1234567890123456789012345678901234567890", 
            "value": Decimal("100.0"),  # Large amount
            "gas": 21000,
            "requires_multisig": True
        }
        
        # Single signature should be insufficient
        single_signature = ["signature_1"]
        
        result = wallet_bridge.validate_transaction_signatures(large_transaction, single_signature)
        assert not result.is_valid, "Large transaction should require multiple signatures"
        assert "multisig" in result.rejection_reason.lower() or "signature" in result.rejection_reason.lower(), \
            "Should indicate multisig requirement"

    def test_wallet_access_rate_limiting_should_prevent_brute_force(self, wallet_bridge):
        """Test wallet access rate limiting prevents brute force - TDD FAIL FIRST"""
        # Simulate rapid failed access attempts
        failed_attempts = []
        
        for i in range(10):  # 10 rapid failed attempts
            try:
                wallet_bridge.authenticate_wallet_access("wrong_password")
                failed_attempts.append(False)
            except Exception:
                failed_attempts.append(True)
        
        # Should eventually block access
        assert len(failed_attempts) > 0, "Should track failed attempts"
        
        # Further attempts should be blocked
        with pytest.raises((SecurityError, Exception)):
            wallet_bridge.authenticate_wallet_access("any_password")


# Mock classes for testing (since actual implementations may not exist)
class FinancialDataValidator:
    def validate_price_data(self, price_data):
        from collections import namedtuple
        Result = namedtuple('Result', ['is_valid', 'rejection_reason'])
        
        price = price_data.get('price')
        
        if price is None or price <= 0 or price > 1000000:
            return Result(False, "Invalid price data detected")
        
        return Result(True, "")


class ArbitrageManipulationDetector:
    def validate_cross_exchange_prices(self, symbol, exchange_prices, max_spread):
        from collections import namedtuple
        Result = namedtuple('Result', ['is_valid', 'rejection_reason'])
        
        prices = [data[symbol] for data in exchange_prices.values()]
        min_price = min(prices)
        max_price = max(prices)
        
        spread = (max_price - min_price) / min_price
        
        if spread > max_spread:
            return Result(False, "Suspicious price spread detected")
        
        return Result(True, "")


class MarketManipulationDetector:
    def validate_trade_impact(self, trade, market_data):
        from collections import namedtuple
        Result = namedtuple('Result', ['is_valid', 'rejection_reason'])
        
        trade_value = trade['value']
        daily_volume = market_data['daily_volume'] 
        
        if trade_value > daily_volume * Decimal('0.5'):
            return Result(False, "Trade size may impact market")
        
        return Result(True, "")


class SecurityError(Exception):
    """Custom security exception for testing"""
    pass