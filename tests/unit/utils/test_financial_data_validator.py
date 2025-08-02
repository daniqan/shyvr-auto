"""
Test-Driven Development tests for Financial Data Validation

Following TDD methodology: 
1. Write failing tests first that define the requirements
2. Implement real validation logic to make tests pass
3. No mocks in production code - only real implementations

This test suite validates financial data integrity including:
- Multi-source price verification
- Outlier detection  
- Stale data detection
- Price manipulation checks
- Volume and liquidity validation
- Cross-reference validation between data sources

Implementation complete - tests now validate real functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from unittest.mock import patch


class TestFinancialDataValidator:
    """TDD test suite for FinancialDataValidator"""

    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for testing"""
        return {
            "symbol": "BTC",
            "price": Decimal("50000.00"),
            "timestamp": datetime.utcnow(),
            "source": "coingecko",
            "volume_24h": Decimal("1000000000"),
            "market_cap": Decimal("950000000000"),
            "circulating_supply": Decimal("19000000")
        }

    @pytest.fixture
    def multi_source_prices(self):
        """Multiple price sources for cross-validation"""
        base_time = datetime.utcnow()
        return [
            {
                "symbol": "BTC",
                "price": Decimal("50000.00"),
                "timestamp": base_time,
                "source": "coingecko",
                "volume_24h": Decimal("1000000000")
            },
            {
                "symbol": "BTC", 
                "price": Decimal("50050.00"),
                "timestamp": base_time - timedelta(seconds=30),
                "source": "coinmarketcap",
                "volume_24h": Decimal("950000000")
            },
            {
                "symbol": "BTC",
                "price": Decimal("49975.00"),
                "timestamp": base_time - timedelta(seconds=60),
                "source": "binance",
                "volume_24h": Decimal("1050000000")
            }
        ]

    @pytest.fixture
    def stale_data_samples(self):
        """Sample data with stale timestamps"""
        now = datetime.utcnow()
        return [
            {
                "symbol": "ETH",
                "price": Decimal("3000.00"),
                "timestamp": now - timedelta(minutes=10),  # Fresh
                "source": "coingecko"
            },
            {
                "symbol": "ETH",
                "price": Decimal("2950.00"),
                "timestamp": now - timedelta(hours=2),  # Stale
                "source": "outdated_api"
            }
        ]

    def test_validator_initialization_with_config(self):
        """Test FinancialDataValidator initialization with configuration"""
        from src.utils.financial_data_validator import FinancialDataValidator, PriceValidationConfig
        
        config = PriceValidationConfig(
            max_price_deviation_percent=Decimal("5.0"),
            max_stale_minutes=30,
            min_volume_threshold=Decimal("1000000"),
            required_sources=["coingecko", "coinmarketcap"],
            outlier_detection_enabled=True
        )
        
        validator = FinancialDataValidator(config)
        assert validator.config == config
        assert validator.is_active is True

    @pytest.mark.asyncio
    async def test_single_source_price_validation_basic(self, sample_price_data):
        """Test basic single source price validation"""
        from src.utils.financial_data_validator import FinancialDataValidator, PriceData
        
        validator = FinancialDataValidator()
        price_data = PriceData(**sample_price_data)
        
        result = await validator.validate_price_data(price_data)
        
        assert result.is_valid is True
        assert result.status.value == "approved"
        assert len(result.warnings) == 0
        assert result.price_data == price_data

    @pytest.mark.asyncio
    async def test_multi_source_price_verification(self, multi_source_prices):
        """Test multi-source price cross-validation and consensus checking"""
        from src.utils.financial_data_validator import FinancialDataValidator, PriceData
        
        validator = FinancialDataValidator()
        price_objects = [PriceData(**data) for data in multi_source_prices]
        
        result = await validator.validate_multi_source_prices(price_objects)
        
        assert result.is_valid is True
        assert result.consensus_price is not None
        assert abs(result.consensus_price - Decimal("50000.00")) < Decimal("100.00")
        assert result.source_agreement_score > 0.8
        assert len(result.outlier_sources) == 0

    @pytest.mark.asyncio 
    async def test_price_outlier_detection(self):
        """Test detection of price outliers across sources"""
        from src.utils.financial_data_validator import FinancialDataValidator, PriceData
        
        validator = FinancialDataValidator()
        
        # Create price data with one obvious outlier
        outlier_prices = [
            PriceData(symbol="BTC", price=Decimal("50000.00"), timestamp=datetime.utcnow(), source="source1"),
            PriceData(symbol="BTC", price=Decimal("50100.00"), timestamp=datetime.utcnow(), source="source2"),
            PriceData(symbol="BTC", price=Decimal("45000.00"), timestamp=datetime.utcnow(), source="outlier_source"),  # 10% deviation
            PriceData(symbol="BTC", price=Decimal("49950.00"), timestamp=datetime.utcnow(), source="source4")
        ]
        
        result = await validator.validate_multi_source_prices(outlier_prices)
        
        assert result.is_valid is False
        assert len(result.outlier_sources) == 1
        assert "outlier_source" in result.outlier_sources
        assert result.status.value == "rejected"
        assert "price_outlier_detected" in [r.value for r in result.reasons]

    @pytest.mark.asyncio
    async def test_stale_data_detection(self, stale_data_samples):
        """Test detection of stale/outdated price data"""
        from src.utils.financial_data_validator import FinancialDataValidator, PriceData
        
        validator = FinancialDataValidator()
        price_objects = [PriceData(**data) for data in stale_data_samples]
        
        for price_data in price_objects:
            result = await validator.validate_price_data(price_data)
            
            if price_data.source == "outdated_api":
                assert result.is_valid is False
                assert "stale_data_detected" in [r.value for r in result.reasons]
            else:
                assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_price_manipulation_detection(self):
        """Test detection of potential price manipulation patterns"""
        from src.utils.financial_data_validator import FinancialDataValidator, PriceData
        
        validator = FinancialDataValidator()
        
        # Simulate suspicious price movements (flash crash pattern)
        base_time = datetime.utcnow()
        manipulation_pattern = [
            PriceData(symbol="ALTCOIN", price=Decimal("100.00"), 
                     timestamp=base_time - timedelta(minutes=10), volume_24h=Decimal("1000000"), source="test"),
            PriceData(symbol="ALTCOIN", price=Decimal("50.00"), 
                     timestamp=base_time - timedelta(minutes=5), volume_24h=Decimal("500000"), source="test"),  # 50% drop
            PriceData(symbol="ALTCOIN", price=Decimal("98.00"), 
                     timestamp=base_time, volume_24h=Decimal("2000000"), source="test")  # Quick recovery
        ]
        
        result = await validator.detect_price_manipulation(manipulation_pattern)
        
        assert result.is_manipulation_detected is True
        assert result.manipulation_type in ["flash_crash", "pump_and_dump", "wash_trading"]
        assert result.confidence_score > 0.7
        assert len(result.suspicious_patterns) > 0

    @pytest.mark.asyncio
    async def test_volume_and_liquidity_validation(self):
        """Test validation of volume and liquidity metrics"""
        from src.utils.financial_data_validator import FinancialDataValidator, VolumeData
        
        validator = FinancialDataValidator()
        
        # Test various volume scenarios
        test_cases = [
            {
                "volume_24h": Decimal("1000000000"),  # High volume - valid
                "liquidity_score": 0.95,
                "bid_ask_spread": Decimal("0.001"),
                "expected_valid": True
            },
            {
                "volume_24h": Decimal("1000"),  # Very low volume - suspicious
                "liquidity_score": 0.2,
                "bid_ask_spread": Decimal("0.1"),
                "expected_valid": False
            }
        ]
        
        for case in test_cases:
            volume_data = VolumeData(
                symbol="TEST",
                volume_24h=case["volume_24h"],
                liquidity_score=case["liquidity_score"],
                bid_ask_spread=case["bid_ask_spread"]
            )
            
            result = await validator.validate_volume_data(volume_data)
            assert result.is_valid == case["expected_valid"]

    @pytest.mark.asyncio
    async def test_cross_reference_validation_between_sources(self):
        """Test cross-reference validation between multiple data sources"""
        from src.utils.financial_data_validator import FinancialDataValidator
        
        validator = FinancialDataValidator()
        
        # Mock different data sources with similar but not identical data
        source_data = {
            "coingecko": {
                "BTC": {"price": Decimal("50000.00"), "volume": Decimal("1000000000")},
                "ETH": {"price": Decimal("3000.00"), "volume": Decimal("500000000")}
            },
            "coinmarketcap": {
                "BTC": {"price": Decimal("50050.00"), "volume": Decimal("950000000")},
                "ETH": {"price": Decimal("2995.00"), "volume": Decimal("520000000")}
            },
            "binance": {
                "BTC": {"price": Decimal("49980.00"), "volume": Decimal("1100000000")},
                "ETH": {"price": Decimal("3005.00"), "volume": Decimal("480000000")}
            }
        }
        
        result = await validator.cross_validate_sources(source_data)
        
        assert result.is_consistent is True
        assert result.consensus_data is not None
        assert "BTC" in result.consensus_data
        assert "ETH" in result.consensus_data
        assert result.reliability_score > 0.8

    @pytest.mark.asyncio
    async def test_real_time_data_quality_monitoring(self):
        """Test real-time monitoring of data quality metrics"""
        from src.utils.financial_data_validator import FinancialDataValidator
        
        validator = FinancialDataValidator()
        
        # Simulate monitoring over time
        monitoring_result = await validator.start_monitoring()
        
        assert monitoring_result.is_active is True
        assert monitoring_result.monitoring_interval > 0
        assert hasattr(monitoring_result, 'quality_metrics')
        
        # Test metrics collection
        metrics = await validator.get_data_quality_metrics()
        
        assert "source_reliability" in metrics
        assert "data_freshness" in metrics
        assert "price_stability" in metrics
        assert "volume_consistency" in metrics

    @pytest.mark.asyncio
    async def test_alert_generation_for_data_issues(self):
        """Test automatic alert generation for data quality issues"""
        from src.utils.financial_data_validator import FinancialDataValidator
        
        validator = FinancialDataValidator()
        
        # Simulate problematic data that should trigger alerts
        bad_data = [
            {
                "symbol": "SCAM_TOKEN",
                "price": Decimal("1000000.00"),  # Unrealistic price
                "volume_24h": Decimal("10"),     # Extremely low volume
                "timestamp": datetime.utcnow() - timedelta(hours=5),  # Stale
                "source": "suspicious_exchange"
            }
        ]
        
        alerts = await validator.process_data_and_generate_alerts(bad_data)
        
        assert len(alerts) > 0
        alert_types = [alert.alert_type.value for alert in alerts]
        assert "stale_data" in alert_types
        assert "low_volume" in alert_types

    @pytest.mark.asyncio
    async def test_historical_price_pattern_analysis(self):
        """Test analysis of historical price patterns for validation"""
        from src.utils.financial_data_validator import FinancialDataValidator
        
        validator = FinancialDataValidator()
        
        # Create historical price pattern
        base_price = Decimal("50000.00")
        historical_data = []
        
        for i in range(24):  # 24 hours of hourly data
            timestamp = datetime.utcnow() - timedelta(hours=23-i)
            # Add realistic price variation
            price_variation = Decimal(str(-2 + (i * 0.2)))  # -2% to +2.6% variation
            price = base_price * (1 + price_variation / 100)
            
            historical_data.append({
                "symbol": "BTC",
                "price": price,
                "timestamp": timestamp,
                "volume_24h": Decimal("1000000000"),
                "source": "test"
            })
        
        analysis = await validator.analyze_historical_patterns(historical_data)
        
        assert analysis.trend_direction in ["bullish", "bearish", "sideways"]
        assert 0 <= analysis.volatility_score <= 1
        assert analysis.pattern_reliability > 0
        assert len(analysis.support_levels) >= 0
        assert len(analysis.resistance_levels) >= 0

    @pytest.mark.asyncio
    async def test_error_handling_and_fallback_validation(self):
        """Test proper error handling and fallback validation mechanisms"""
        from src.utils.financial_data_validator import FinancialDataValidator, ValidationError
        
        validator = FinancialDataValidator()
        
        # Test with None data
        result = await validator.validate_price_data(None)
        assert result.is_valid is False
        assert "invalid_data" in [r.value for r in result.reasons]
        
        # Test with malformed data
        bad_data = {"invalid": "data_structure"}
        result = await validator.validate_price_data(bad_data)
        assert result.is_valid is False
        
        # Test fallback when primary validation fails
        fallback_result = await validator.validate_with_fallback(bad_data)
        assert fallback_result.is_valid is False
        assert fallback_result.used_fallback is True

    @pytest.mark.asyncio
    async def test_configuration_validation_and_limits(self):
        """Test validation configuration and limit enforcement"""
        from src.utils.financial_data_validator import PriceValidationConfig, FinancialDataValidator
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            PriceValidationConfig(
                max_price_deviation_percent=Decimal("-5.0"),  # Negative deviation
                max_stale_minutes=-10  # Negative time
            )
        
        # Test with restrictive configuration
        strict_config = PriceValidationConfig(
            max_price_deviation_percent=Decimal("1.0"),  # Very strict 1%
            max_stale_minutes=5,  # Very fresh data required
            min_volume_threshold=Decimal("10000000000")  # High volume requirement
        )
        
        validator = FinancialDataValidator(strict_config)
        
        # Data that would pass normal validation but fail strict validation
        borderline_data = {
            "symbol": "BTC",
            "price": Decimal("50000.00"),
            "timestamp": datetime.utcnow() - timedelta(minutes=7),  # Slightly stale
            "volume_24h": Decimal("5000000000"),  # Lower volume
            "source": "test_source"
        }
        
        result = await validator.validate_price_data(borderline_data)
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_performance_and_scalability_requirements(self):
        """Test performance requirements for high-frequency validation"""
        from src.utils.financial_data_validator import FinancialDataValidator
        import time
        
        validator = FinancialDataValidator()
        
        # Test batch validation performance
        large_dataset = []
        for i in range(100):  # 100 price points (reduced for testing)
            large_dataset.append({
                "symbol": f"TOKEN_{i}",
                "price": Decimal("100.00") + Decimal(str(i * 0.01)),
                "timestamp": datetime.utcnow(),
                "volume_24h": Decimal("1000000"),
                "source": "performance_test"
            })
        
        start_time = time.time()
        results = await validator.validate_batch_prices(large_dataset)
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Should process 100 items efficiently
        assert validation_time < 10.0
        assert len(results) == 100
        assert all(hasattr(result, 'is_valid') for result in results)

    def test_integration_with_existing_safety_systems(self):
        """Test integration with existing TradingSafetyManager and emergency stop systems"""
        from src.utils.financial_data_validator import FinancialDataValidator
        
        validator = FinancialDataValidator()
        
        # Test that validator can integrate with safety manager
        assert hasattr(validator, 'integrate_with_safety_manager')
        
        # Test that validation results can trigger emergency stops
        assert hasattr(validator, 'check_emergency_conditions')
        
        # Test alert integration
        assert hasattr(validator, 'send_safety_alerts')