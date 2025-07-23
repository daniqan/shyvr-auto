"""
Base classes and interfaces for the RLTE system
Provides abstract base classes that define the architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio


class Chain(Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    SOLANA = "solana"
    BASE = "base"


class TradingMode(Enum):
    """Trading operation modes"""
    ANALYSIS = "analysis"
    SIMULATION = "simulation"
    LIVE = "live"


@dataclass
class TokenInfo:
    """Basic token information"""
    address: str
    symbol: str
    name: str
    chain: Chain
    decimals: int


@dataclass
class AgentRule:
    """Agent-generated rule"""
    id: str
    prompt_text: str
    rule_type: str  # 'filter', 'dca', 'risk', 'custom'
    parsed_conditions: Dict[str, Any]
    active: bool = True
    priority: int = 0


class Module(ABC):
    """Abstract base class for all RLTE modules"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_rules: List[AgentRule] = []
    
    @abstractmethod
    async def process(self, data: Any, agent_rules: Optional[List[AgentRule]] = None) -> Any:
        """Process data with optional agent rule augmentation"""
        pass
    
    def apply_agent_rules(self, data: Any, rules: List[AgentRule]) -> Any:
        """Apply agent rules to modify processing behavior"""
        # Default implementation - override in subclasses
        return data
    
    def validate_config(self) -> bool:
        """Validate module configuration"""
        return True


class Discoverer(Module):
    """Abstract base class for token discovery"""
    
    @abstractmethod
    async def scan_chain(self, chain: Chain) -> List[TokenInfo]:
        """Scan a specific chain for new tokens"""
        pass
    
    @abstractmethod
    async def scan_all_chains(self) -> List[TokenInfo]:
        """Scan all supported chains"""
        pass


class Evaluator(Module):
    """Abstract base class for token evaluation"""
    
    @abstractmethod
    async def evaluate_fundamentals(self, token: TokenInfo) -> Dict[str, Any]:
        """Evaluate token fundamentals"""
        pass
    
    @abstractmethod
    async def filter_candidates(self, tokens: List[TokenInfo]) -> List[TokenInfo]:
        """Filter tokens based on criteria"""
        pass


class MLAnalyzer(Module):
    """Abstract base class for ML analysis"""
    
    @abstractmethod
    async def predict_price(self, token: TokenInfo, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate price predictions"""
        pass
    
    @abstractmethod
    async def calculate_features(self, token: TokenInfo) -> Dict[str, Any]:
        """Calculate ML features"""
        pass


class RLAgent(Module):
    """Abstract base class for RL trading agent"""
    
    @abstractmethod
    async def act(self, state: Dict[str, Any], training: bool = False) -> int:
        """Choose action based on state"""
        pass
    
    @abstractmethod
    async def update(self, experience: Dict[str, Any]) -> None:
        """Update agent based on experience"""
        pass


class TradingEngine(Module):
    """Abstract base class for trading execution"""
    
    @abstractmethod
    async def execute_trade(self, action: int, token: TokenInfo, amount: float) -> Dict[str, Any]:
        """Execute a trade"""
        pass
    
    @abstractmethod
    async def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        pass


class AgentFramework(Module):
    """Abstract base class for natural language agent"""
    
    @abstractmethod
    async def parse_prompt(self, prompt: str) -> AgentRule:
        """Parse natural language prompt into actionable rule"""
        pass
    
    @abstractmethod
    async def validate_rule(self, rule: AgentRule) -> bool:
        """Validate that rule is safe and reasonable"""
        pass
    
    @abstractmethod
    async def apply_rules_to_module(self, module: Module, rules: List[AgentRule]) -> None:
        """Apply rules to modify module behavior"""
        pass


# Exception classes
class RLTEException(Exception):
    """Base exception for RLTE system"""
    pass


class ConfigurationError(RLTEException):
    """Configuration-related errors"""
    pass


class APIError(RLTEException):
    """API-related errors"""
    pass


class TradingError(RLTEException):
    """Trading-related errors"""
    pass


class AgentError(RLTEException):
    """Agent-related errors"""
    pass