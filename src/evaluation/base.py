"""
Base classes for token evaluation system
Defines interfaces and data structures for token fundamental analysis and security evaluation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import structlog

from src.utils.base import Chain
from src.discovery.base import DiscoveredToken


logger = structlog.get_logger()


class RiskLevel(Enum):
    """Risk levels for token evaluation"""
    VERY_LOW = "very_low"      # 0-20% risk
    LOW = "low"                # 20-40% risk  
    MEDIUM = "medium"          # 40-60% risk
    HIGH = "high"              # 60-80% risk
    VERY_HIGH = "very_high"    # 80-100% risk


class EvaluationStatus(Enum):
    """Status of token evaluation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class SecurityFlags:
    """Security-related flags for a token"""
    is_honeypot: bool = False
    is_rugpull_risk: bool = False
    has_mint_function: bool = False
    ownership_renounced: bool = False
    liquidity_locked: bool = False
    has_pause_function: bool = False
    has_blacklist_function: bool = False
    contract_verified: bool = False
    
    # Additional security metrics
    honeypot_probability: float = 0.0
    rugpull_probability: float = 0.0
    security_score: float = 0.0  # 0-100 scale
    
    def is_safe_to_trade(self) -> bool:
        """Determine if token is safe enough to trade"""
        return (
            not self.is_honeypot and
            not self.is_rugpull_risk and
            self.security_score >= 50.0
        )


@dataclass
class FundamentalMetrics:
    """Fundamental analysis metrics for a token"""
    # Liquidity metrics
    liquidity_usd: Optional[float] = None
    liquidity_locked_pct: Optional[float] = None
    
    # Holder metrics
    holder_count: Optional[int] = None
    top_10_holders_pct: Optional[float] = None
    dev_wallet_pct: Optional[float] = None
    
    # Volume metrics
    volume_24h: Optional[float] = None
    volume_change_pct: Optional[float] = None
    volume_to_liquidity_ratio: Optional[float] = None
    
    # Price metrics
    price_usd: Optional[float] = None
    price_change_1h: Optional[float] = None
    price_change_24h: Optional[float] = None
    market_cap: Optional[float] = None
    fully_diluted_valuation: Optional[float] = None
    
    # Social metrics
    social_score: Optional[float] = None
    twitter_followers: Optional[int] = None
    telegram_members: Optional[int] = None
    discord_members: Optional[int] = None
    
    def calculate_fundamental_score(self) -> float:
        """Calculate overall fundamental score (0-100)"""
        score = 0.0
        factors = 0
        
        # Liquidity scoring (25% weight)
        if self.liquidity_usd is not None:
            if self.liquidity_usd > 1000000:  # >$1M
                score += 25
            elif self.liquidity_usd > 100000:  # >$100K
                score += 20
            elif self.liquidity_usd > 10000:  # >$10K
                score += 15
            else:
                score += 5
            factors += 1
        
        # Holder distribution (25% weight)
        if self.holder_count is not None:
            if self.holder_count > 1000:
                score += 25
            elif self.holder_count > 100:
                score += 20
            elif self.holder_count > 50:
                score += 15
            else:
                score += 5
            factors += 1
        
        # Volume (25% weight) 
        if self.volume_24h is not None and self.liquidity_usd is not None:
            volume_ratio = self.volume_24h / max(self.liquidity_usd, 1)
            if volume_ratio > 0.5:  # High volume/liquidity ratio
                score += 25
            elif volume_ratio > 0.1:
                score += 20
            elif volume_ratio > 0.05:
                score += 15
            else:
                score += 10
            factors += 1
        
        # Social presence (25% weight)
        if self.social_score is not None:
            score += min(self.social_score, 25)
            factors += 1
        
        return score if factors > 0 else 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result for a token"""
    # Basic info
    token: DiscoveredToken
    evaluated_at: datetime
    evaluation_duration_ms: Optional[float] = None
    
    # Evaluation status
    status: EvaluationStatus = EvaluationStatus.PENDING
    overall_risk: RiskLevel = RiskLevel.MEDIUM
    overall_score: float = 0.0  # 0-100 combined score
    
    # Detailed analysis
    security_flags: Optional[SecurityFlags] = None
    fundamental_metrics: Optional[FundamentalMetrics] = None
    
    # Recommendations
    is_approved: bool = False
    recommended_action: str = "HOLD"  # BUY, SELL, HOLD, AVOID
    confidence_level: float = 0.0  # 0-100
    
    # Risk breakdown
    security_risk: float = 0.0      # 0-100
    liquidity_risk: float = 0.0     # 0-100
    volatility_risk: float = 0.0    # 0-100
    social_risk: float = 0.0        # 0-100
    
    # Additional data
    warnings: List[str] = None
    notes: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.notes is None:
            self.notes = []
        if self.metadata is None:
            self.metadata = {}
    
    def calculate_overall_risk(self) -> RiskLevel:
        """Calculate overall risk level based on individual risk factors"""
        avg_risk = (
            self.security_risk + 
            self.liquidity_risk + 
            self.volatility_risk + 
            self.social_risk
        ) / 4
        
        if avg_risk >= 80:
            return RiskLevel.VERY_HIGH
        elif avg_risk >= 60:
            return RiskLevel.HIGH
        elif avg_risk >= 40:
            return RiskLevel.MEDIUM
        elif avg_risk >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def should_approve(self) -> bool:
        """Determine if token should be approved for trading"""
        return (
            self.security_flags is not None and
            self.security_flags.is_safe_to_trade() and
            self.overall_risk in [RiskLevel.VERY_LOW, RiskLevel.LOW, RiskLevel.MEDIUM] and
            self.overall_score >= 40.0 and
            self.confidence_level >= 60.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "token_address": self.token.address,
            "token_chain": self.token.chain.value,
            "evaluated_at": self.evaluated_at.isoformat(),
            "evaluation_duration_ms": self.evaluation_duration_ms,
            "status": self.status.value,
            "overall_risk": self.overall_risk.value,
            "overall_score": self.overall_score,
            "is_approved": self.is_approved,
            "recommended_action": self.recommended_action,
            "confidence_level": self.confidence_level,
            "security_risk": self.security_risk,
            "liquidity_risk": self.liquidity_risk,
            "volatility_risk": self.volatility_risk,
            "social_risk": self.social_risk,
            "warnings": self.warnings,
            "notes": self.notes,
            "metadata": self.metadata,
        }


class TokenEvaluatorBase(ABC):
    """Abstract base class for token evaluators"""
    
    def __init__(self, rate_limit: int = 30):
        self.rate_limit = rate_limit
        self.logger = structlog.get_logger().bind(evaluator=self.__class__.__name__)
    
    @abstractmethod
    async def evaluate_token(self, token: DiscoveredToken) -> EvaluationResult:
        """Evaluate a single token and return detailed results"""
        pass
    
    async def evaluate_tokens(self, tokens: List[DiscoveredToken]) -> List[EvaluationResult]:
        """Evaluate multiple tokens"""
        results = []
        for token in tokens:
            try:
                result = await self.evaluate_token(token)
                results.append(result)
            except Exception as e:
                self.logger.error("Token evaluation failed", 
                                token=token.address, 
                                chain=token.chain.value, 
                                error=str(e))
                # Create failed evaluation result
                failed_result = EvaluationResult(
                    token=token,
                    evaluated_at=datetime.now(),
                    status=EvaluationStatus.FAILED,
                    overall_risk=RiskLevel.VERY_HIGH,
                    warnings=[f"Evaluation failed: {str(e)}"]
                )
                results.append(failed_result)
        
        return results
    
    async def health_check(self) -> bool:
        """Check if the evaluator is healthy"""
        try:
            # Override in subclasses with specific health check logic
            return True
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False


class EvaluationError(Exception):
    """Base exception for token evaluation errors"""
    pass


class SecurityEvaluationError(EvaluationError):
    """Raised when security evaluation fails"""
    pass


class FundamentalEvaluationError(EvaluationError):
    """Raised when fundamental analysis fails"""
    pass