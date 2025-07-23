"""
DetectHoneypot API Client for Security Evaluation
Provides honeypot detection and rug pull analysis across multiple chains
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any
import structlog

from src.utils.base import Chain
from src.discovery.base import DiscoveredToken
from .base import (
    TokenEvaluatorBase,
    EvaluationResult,
    SecurityFlags,
    RiskLevel,
    EvaluationStatus,
    SecurityEvaluationError
)


logger = structlog.get_logger()


class HoneypotDetector(TokenEvaluatorBase):
    """DetectHoneypot.com API client for token security analysis"""
    
    BASE_URL = "https://api.detecthoneypot.com"
    
    # Chain mapping for DetectHoneypot API
    CHAIN_MAPPING = {
        Chain.ETHEREUM: "eth",
        Chain.BSC: "bsc", 
        Chain.POLYGON: "polygon",
        Chain.ARBITRUM: "arbitrum",
        Chain.AVALANCHE: "avax",
        Chain.FANTOM: "ftm",
        Chain.SOLANA: "solana",
    }
    
    def __init__(self, rate_limit: int = 20):
        super().__init__(rate_limit=rate_limit)
        self.session: Optional[aiohttp.ClientSession] = None
        self._detection_cache: Dict[str, Dict[str, Any]] = {}
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60)  # Longer timeout for simulation
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make HTTP request to DetectHoneypot API"""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    raise SecurityEvaluationError("DetectHoneypot API rate limit exceeded")
                elif response.status != 200:
                    text = await response.text()
                    raise SecurityEvaluationError(f"DetectHoneypot API error {response.status}: {text}")
                
                data = await response.json()
                return data
                
        except aiohttp.ClientError as e:
            raise SecurityEvaluationError(f"DetectHoneypot API request failed: {str(e)}")
    
    def _get_chain_param(self, chain: Chain) -> Optional[str]:
        """Convert Chain enum to DetectHoneypot chain parameter"""
        return self.CHAIN_MAPPING.get(chain)
    
    def _parse_honeypot_response(self, data: Dict, token: DiscoveredToken) -> SecurityFlags:
        """Parse DetectHoneypot API response into SecurityFlags"""
        
        # Extract key security information
        is_honeypot = data.get("IsHoneypot", False)
        
        # Calculate honeypot probability based on multiple factors
        honeypot_probability = 0.0
        if is_honeypot:
            honeypot_probability = 0.9
        else:
            # Check for suspicious patterns
            buy_tax = float(data.get("BuyTax", 0))
            sell_tax = float(data.get("SellTax", 0))
            
            if sell_tax > 50:  # >50% sell tax is very suspicious
                honeypot_probability += 0.6
            elif sell_tax > 20:  # >20% sell tax is suspicious
                honeypot_probability += 0.3
            
            if buy_tax > 20:  # High buy tax
                honeypot_probability += 0.2
            
            if sell_tax > buy_tax * 3:  # Sell tax much higher than buy tax
                honeypot_probability += 0.3
        
        # Calculate rug pull probability
        rugpull_probability = 0.0
        
        # Check ownership and functions
        ownership_renounced = data.get("OwnershipRenounced", False)
        has_mint_function = data.get("CanMint", False)
        has_pause_function = data.get("CanPause", False)
        has_blacklist_function = data.get("CanBlacklist", False)
        
        if not ownership_renounced:
            rugpull_probability += 0.4
        if has_mint_function:
            rugpull_probability += 0.3
        if has_pause_function:
            rugpull_probability += 0.2
        if has_blacklist_function:
            rugpull_probability += 0.2
        
        # Calculate overall security score (0-100)
        security_score = 100.0
        security_score -= honeypot_probability * 50  # Honeypot risk penalty
        security_score -= rugpull_probability * 30   # Rug pull risk penalty
        security_score -= max(buy_tax - 5, 0) * 2    # High tax penalty
        security_score -= max(sell_tax - 5, 0) * 2   # High tax penalty
        security_score = max(security_score, 0.0)
        
        # Add bonuses for good practices
        if ownership_renounced:
            security_score += 10
        if data.get("IsVerified", False):
            security_score += 15
        
        security_score = min(security_score, 100.0)
        
        return SecurityFlags(
            is_honeypot=is_honeypot,
            is_rugpull_risk=rugpull_probability > 0.5,
            has_mint_function=has_mint_function,
            ownership_renounced=ownership_renounced,
            has_pause_function=has_pause_function,
            has_blacklist_function=has_blacklist_function,
            contract_verified=data.get("IsVerified", False),
            honeypot_probability=min(honeypot_probability, 1.0),
            rugpull_probability=min(rugpull_probability, 1.0),
            security_score=security_score,
        )
    
    async def detect_honeypot(self, token_address: str, chain: Chain) -> Dict[str, Any]:
        """Detect if a token is a honeypot"""
        # Check cache first
        cache_key = f"{chain.value}:{token_address}"
        if cache_key in self._detection_cache:
            return self._detection_cache[cache_key]
        
        chain_param = self._get_chain_param(chain)
        if not chain_param:
            raise SecurityEvaluationError(f"Chain {chain.value} not supported by DetectHoneypot")
        
        try:
            endpoint = "/api/v1/honeypot"
            params = {
                "address": token_address,
                "chain": chain_param
            }
            
            data = await self._make_request(endpoint, params=params)
            
            # Cache the result
            self._detection_cache[cache_key] = data
            
            self.logger.info("Honeypot detection completed", 
                           token=token_address, 
                           chain=chain.value,
                           is_honeypot=data.get("IsHoneypot", False))
            
            return data
            
        except Exception as e:
            self.logger.error("Honeypot detection failed", 
                            token=token_address, 
                            chain=chain.value, 
                            error=str(e))
            raise
    
    async def evaluate_token(self, token: DiscoveredToken) -> EvaluationResult:
        """Evaluate a token for security risks"""
        start_time = datetime.now()
        
        try:
            # Skip evaluation for unsupported chains
            if token.chain not in self.CHAIN_MAPPING:
                self.logger.warning("Chain not supported for security evaluation", 
                                  chain=token.chain.value)
                return EvaluationResult(
                    token=token,
                    evaluated_at=start_time,
                    status=EvaluationStatus.COMPLETED,
                    overall_risk=RiskLevel.MEDIUM,
                    overall_score=50.0,
                    warnings=[f"Security evaluation not available for {token.chain.value}"],
                    notes=["Chain not supported by DetectHoneypot API"]
                )
            
            # Perform honeypot detection
            detection_data = await self.detect_honeypot(token.address, token.chain)
            security_flags = self._parse_honeypot_response(detection_data, token)
            
            # Calculate risks
            security_risk = (security_flags.honeypot_probability + security_flags.rugpull_probability) * 50
            
            # Determine overall risk level
            if security_flags.is_honeypot or security_risk >= 80:
                overall_risk = RiskLevel.VERY_HIGH
                recommended_action = "AVOID"
                is_approved = False
            elif security_risk >= 60:
                overall_risk = RiskLevel.HIGH
                recommended_action = "AVOID"
                is_approved = False
            elif security_risk >= 40:
                overall_risk = RiskLevel.MEDIUM
                recommended_action = "CAUTION"
                is_approved = security_flags.is_safe_to_trade()
            else:
                overall_risk = RiskLevel.LOW
                recommended_action = "MONITOR"
                is_approved = security_flags.is_safe_to_trade()
            
            # Create warnings
            warnings = []
            if security_flags.is_honeypot:
                warnings.append("⚠️ TOKEN IS A HONEYPOT - DO NOT TRADE")
            if security_flags.is_rugpull_risk:
                warnings.append("⚠️ High rug pull risk detected")
            if security_flags.honeypot_probability > 0.5:
                warnings.append(f"High honeypot probability: {security_flags.honeypot_probability:.1%}")
            if not security_flags.ownership_renounced:
                warnings.append("Ownership not renounced - centralization risk")
            if security_flags.has_mint_function:
                warnings.append("Token has mint function - inflation risk")
            
            # Create notes
            notes = []
            buy_tax = detection_data.get("BuyTax", 0)
            sell_tax = detection_data.get("SellTax", 0)
            if buy_tax > 0 or sell_tax > 0:
                notes.append(f"Trading taxes: {buy_tax}% buy, {sell_tax}% sell")
            if security_flags.contract_verified:
                notes.append("✅ Contract is verified")
            
            # Calculate confidence level
            confidence_level = 90.0  # High confidence in security detection
            
            evaluation_duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return EvaluationResult(
                token=token,
                evaluated_at=start_time,
                evaluation_duration_ms=evaluation_duration,
                status=EvaluationStatus.COMPLETED,
                overall_risk=overall_risk,
                overall_score=security_flags.security_score,
                security_flags=security_flags,
                is_approved=is_approved,
                recommended_action=recommended_action,
                confidence_level=confidence_level,
                security_risk=security_risk,
                liquidity_risk=0.0,  # Not evaluated by this detector
                volatility_risk=0.0,  # Not evaluated by this detector
                social_risk=0.0,      # Not evaluated by this detector
                warnings=warnings,
                notes=notes,
                metadata={
                    "honeypot_data": detection_data,
                    "detection_source": "detecthoneypot.com"
                }
            )
            
        except Exception as e:
            evaluation_duration = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error("Security evaluation failed", 
                            token=token.address, 
                            chain=token.chain.value, 
                            error=str(e))
            
            return EvaluationResult(
                token=token,
                evaluated_at=start_time,
                evaluation_duration_ms=evaluation_duration,
                status=EvaluationStatus.FAILED,
                overall_risk=RiskLevel.VERY_HIGH,
                overall_score=0.0,
                is_approved=False,
                recommended_action="AVOID",
                confidence_level=0.0,
                security_risk=100.0,
                warnings=[f"Security evaluation failed: {str(e)}"],
                metadata={"error": str(e)}
            )
    
    def get_supported_chains(self) -> List[Chain]:
        """Get list of supported chains"""
        return list(self.CHAIN_MAPPING.keys())
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()