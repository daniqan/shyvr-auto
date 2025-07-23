"""
Unit tests for DetectHoneypot API client
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch
import aiohttp

from src.utils.base import Chain
from src.discovery.base import DiscoveredToken, TokenStatus
from src.evaluation.honeypot_detector import HoneypotDetector
from src.evaluation.base import (
    SecurityFlags,
    EvaluationResult,
    RiskLevel,
    EvaluationStatus,
    SecurityEvaluationError,
)


class TestHoneypotDetector:
    """Test DetectHoneypot API client"""

    @pytest.fixture
    def honeypot_detector(self):
        """Create honeypot detector for testing"""
        return HoneypotDetector(rate_limit=20)

    @pytest.fixture
    def sample_token(self):
        """Create sample token for testing"""
        return DiscoveredToken(
            address="0x123456789abcdef",
            chain=Chain.ETHEREUM,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
        )

    @pytest.fixture
    def safe_token_data(self):
        """Mock response for safe token"""
        return {
            "IsHoneypot": False,
            "BuyTax": "5.0",
            "SellTax": "5.0",
            "OwnershipRenounced": True,
            "CanMint": False,
            "CanPause": False,
            "CanBlacklist": False,
            "IsVerified": True,
        }

    @pytest.fixture
    def honeypot_data(self):
        """Mock response for honeypot token"""
        return {
            "IsHoneypot": True,
            "BuyTax": "1.0",
            "SellTax": "99.0",  # Very high sell tax
            "OwnershipRenounced": False,
            "CanMint": True,
            "CanPause": True,
            "CanBlacklist": True,
            "IsVerified": False,
        }

    @pytest.fixture
    def risky_token_data(self):
        """Mock response for risky but not honeypot token"""
        return {
            "IsHoneypot": False,
            "BuyTax": "10.0",
            "SellTax": "25.0",  # High sell tax
            "OwnershipRenounced": False,  # Risk factor
            "CanMint": True,   # Risk factor
            "CanPause": False,
            "CanBlacklist": False,
            "IsVerified": True,
        }

    def test_initialization(self, honeypot_detector):
        """Test detector initialization"""
        assert honeypot_detector.rate_limit == 20
        assert honeypot_detector.session is None
        assert honeypot_detector._detection_cache == {}

    def test_supported_chains(self, honeypot_detector):
        """Test supported chains"""
        chains = honeypot_detector.get_supported_chains()
        expected_chains = [
            Chain.ETHEREUM, Chain.BSC, Chain.POLYGON, 
            Chain.ARBITRUM, Chain.AVALANCHE, Chain.FANTOM, Chain.SOLANA
        ]
        assert all(chain in chains for chain in expected_chains)

    def test_get_chain_param(self, honeypot_detector):
        """Test chain parameter mapping"""
        assert honeypot_detector._get_chain_param(Chain.ETHEREUM) == "eth"
        assert honeypot_detector._get_chain_param(Chain.BSC) == "bsc"
        assert honeypot_detector._get_chain_param(Chain.POLYGON) == "polygon"
        assert honeypot_detector._get_chain_param(Chain.ARBITRUM) == "arbitrum"
        assert honeypot_detector._get_chain_param(Chain.AVALANCHE) == "avax"
        assert honeypot_detector._get_chain_param(Chain.FANTOM) == "ftm"
        assert honeypot_detector._get_chain_param(Chain.SOLANA) == "solana"
        
        # Test unsupported chain
        assert honeypot_detector._get_chain_param(Chain.BASE) is None

    def test_parse_safe_token_response(self, honeypot_detector, sample_token, safe_token_data):
        """Test parsing safe token response"""
        flags = honeypot_detector._parse_honeypot_response(safe_token_data, sample_token)

        assert not flags.is_honeypot
        assert not flags.is_rugpull_risk  # Ownership renounced and no dangerous functions
        assert not flags.has_mint_function
        assert flags.ownership_renounced
        assert not flags.has_pause_function
        assert not flags.has_blacklist_function
        assert flags.contract_verified
        assert flags.honeypot_probability == 0.0
        assert flags.rugpull_probability == 0.0
        assert flags.security_score > 80.0  # Should be high for safe token
        assert flags.is_safe_to_trade()

    def test_parse_honeypot_response(self, honeypot_detector, sample_token, honeypot_data):
        """Test parsing honeypot token response"""
        flags = honeypot_detector._parse_honeypot_response(honeypot_data, sample_token)

        assert flags.is_honeypot
        assert flags.is_rugpull_risk
        assert flags.has_mint_function
        assert not flags.ownership_renounced
        assert flags.has_pause_function
        assert flags.has_blacklist_function
        assert not flags.contract_verified
        assert flags.honeypot_probability == 0.9  # Should be very high
        assert flags.rugpull_probability > 0.5   # Should be high
        assert flags.security_score < 50.0       # Should be low
        assert not flags.is_safe_to_trade()

    def test_parse_risky_token_response(self, honeypot_detector, sample_token, risky_token_data):
        """Test parsing risky but not honeypot token response"""
        flags = honeypot_detector._parse_honeypot_response(risky_token_data, sample_token)

        assert not flags.is_honeypot
        assert flags.is_rugpull_risk  # Due to ownership not renounced + mint function
        assert flags.has_mint_function
        assert not flags.ownership_renounced
        assert flags.contract_verified
        assert flags.honeypot_probability < 0.9  # Not a honeypot but some risk
        assert flags.rugpull_probability > 0.5   # High due to risk factors
        assert 30.0 < flags.security_score < 70.0  # Medium security score

    @pytest.mark.asyncio
    async def test_make_request_success(self, honeypot_detector, safe_token_data):
        """Test successful API request"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=safe_token_data)
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await honeypot_detector._make_request("/test")
            assert result == safe_token_data

    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self, honeypot_detector):
        """Test API rate limit handling"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(SecurityEvaluationError, match="rate limit exceeded"):
                await honeypot_detector._make_request("/test")

    @pytest.mark.asyncio
    async def test_make_request_error(self, honeypot_detector):
        """Test API error handling"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(SecurityEvaluationError, match="DetectHoneypot API error 500"):
                await honeypot_detector._make_request("/test")

    @pytest.mark.asyncio
    async def test_detect_honeypot_success(self, honeypot_detector, safe_token_data):
        """Test successful honeypot detection"""
        with patch.object(honeypot_detector, "_make_request", return_value=safe_token_data):
            result = await honeypot_detector.detect_honeypot("0x123", Chain.ETHEREUM)
            
            assert result == safe_token_data
            # Should be cached
            assert "ethereum:0x123" in honeypot_detector._detection_cache

    @pytest.mark.asyncio
    async def test_detect_honeypot_cached(self, honeypot_detector, safe_token_data):
        """Test cached honeypot detection"""
        # Pre-cache the result
        honeypot_detector._detection_cache["ethereum:0x123"] = safe_token_data
        
        with patch.object(honeypot_detector, "_make_request") as mock_request:
            result = await honeypot_detector.detect_honeypot("0x123", Chain.ETHEREUM)
            
            # Should return cached result without API call
            mock_request.assert_not_called()
            assert result == safe_token_data

    @pytest.mark.asyncio
    async def test_detect_honeypot_unsupported_chain(self, honeypot_detector):
        """Test honeypot detection for unsupported chain"""
        with pytest.raises(SecurityEvaluationError, match="not supported by DetectHoneypot"):
            await honeypot_detector.detect_honeypot("0x123", Chain.BASE)

    @pytest.mark.asyncio
    async def test_evaluate_token_safe(self, honeypot_detector, sample_token, safe_token_data):
        """Test evaluating a safe token"""
        with patch.object(honeypot_detector, "detect_honeypot", return_value=safe_token_data):
            result = await honeypot_detector.evaluate_token(sample_token)

            assert result.status == EvaluationStatus.COMPLETED
            assert result.overall_risk == RiskLevel.LOW
            assert result.is_approved
            assert result.recommended_action == "MONITOR"
            assert result.confidence_level == 90.0
            assert result.security_risk < 40.0
            assert result.security_flags.is_safe_to_trade()
            assert len(result.warnings) == 0  # Safe token should have no warnings

    @pytest.mark.asyncio
    async def test_evaluate_token_honeypot(self, honeypot_detector, sample_token, honeypot_data):
        """Test evaluating a honeypot token"""
        with patch.object(honeypot_detector, "detect_honeypot", return_value=honeypot_data):
            result = await honeypot_detector.evaluate_token(sample_token)

            assert result.status == EvaluationStatus.COMPLETED
            assert result.overall_risk == RiskLevel.VERY_HIGH
            assert not result.is_approved
            assert result.recommended_action == "AVOID"
            assert result.security_risk >= 80.0
            assert not result.security_flags.is_safe_to_trade()
            assert "TOKEN IS A HONEYPOT" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_evaluate_token_risky(self, honeypot_detector, sample_token, risky_token_data):
        """Test evaluating a risky token"""
        with patch.object(honeypot_detector, "detect_honeypot", return_value=risky_token_data):
            result = await honeypot_detector.evaluate_token(sample_token)

            assert result.status == EvaluationStatus.COMPLETED
            assert result.overall_risk in [RiskLevel.HIGH, RiskLevel.MEDIUM]
            assert "rug pull risk" in " ".join(result.warnings).lower()
            assert "ownership not renounced" in " ".join(result.warnings).lower()

    @pytest.mark.asyncio
    async def test_evaluate_token_unsupported_chain(self, honeypot_detector):
        """Test evaluating token for unsupported chain"""
        unsupported_token = DiscoveredToken(
            address="0x123",
            chain=Chain.BASE,  # Not supported
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
        )

        result = await honeypot_detector.evaluate_token(unsupported_token)

        assert result.status == EvaluationStatus.COMPLETED
        assert result.overall_risk == RiskLevel.MEDIUM
        assert result.overall_score == 50.0
        assert "not available for base" in result.warnings[0].lower()
        assert "Chain not supported by DetectHoneypot API" in result.notes[0]

    @pytest.mark.asyncio
    async def test_evaluate_token_api_error(self, honeypot_detector, sample_token):
        """Test token evaluation with API error"""
        with patch.object(honeypot_detector, "detect_honeypot", side_effect=Exception("API Error")):
            result = await honeypot_detector.evaluate_token(sample_token)

            assert result.status == EvaluationStatus.FAILED
            assert result.overall_risk == RiskLevel.VERY_HIGH
            assert result.overall_score == 0.0
            assert not result.is_approved
            assert result.recommended_action == "AVOID"
            assert result.confidence_level == 0.0
            assert result.security_risk == 100.0
            assert "Security evaluation failed" in result.warnings[0]

    def test_tax_analysis_in_parsing(self, honeypot_detector, sample_token):
        """Test tax analysis in response parsing"""
        high_tax_data = {
            "IsHoneypot": False,
            "BuyTax": "15.0",   # High buy tax
            "SellTax": "30.0",  # Very high sell tax
            "OwnershipRenounced": True,
            "CanMint": False,
            "CanPause": False,
            "CanBlacklist": False,
            "IsVerified": True,
        }

        flags = honeypot_detector._parse_honeypot_response(high_tax_data, sample_token)

        # High taxes should increase honeypot probability
        assert flags.honeypot_probability > 0.3
        # Security score should be penalized for high taxes
        assert flags.security_score < 70.0

    def test_asymmetric_tax_detection(self, honeypot_detector, sample_token):
        """Test detection of asymmetric buy/sell taxes"""
        asymmetric_data = {
            "IsHoneypot": False,
            "BuyTax": "5.0",    # Normal buy tax
            "SellTax": "50.0",  # Much higher sell tax (suspicious)
            "OwnershipRenounced": True,
            "CanMint": False,
            "CanPause": False,
            "CanBlacklist": False,
            "IsVerified": True,
        }

        flags = honeypot_detector._parse_honeypot_response(asymmetric_data, sample_token)

        # Should detect the suspicious sell tax pattern
        assert flags.honeypot_probability > 0.5
        assert flags.security_score < 60.0

    @pytest.mark.asyncio
    async def test_session_management(self, honeypot_detector):
        """Test HTTP session management"""
        # Session should be None initially
        assert honeypot_detector.session is None

        # Getting session should create it with longer timeout
        session = await honeypot_detector._get_session()
        assert isinstance(session, aiohttp.ClientSession)
        assert session.timeout.total == 60  # Should have 60 second timeout
        assert honeypot_detector.session is session

        # Closing should set to None
        await honeypot_detector.close()
        assert honeypot_detector.session is None

    @pytest.mark.asyncio
    async def test_context_manager(self, honeypot_detector):
        """Test using detector as context manager"""
        async with honeypot_detector as detector:
            assert detector is honeypot_detector
            # Session should be available during context
            session = await detector._get_session()
            assert session is not None

        # Session should be closed after context
        assert honeypot_detector.session is None