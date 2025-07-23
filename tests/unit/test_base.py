"""
Unit tests for base classes and interfaces
Tests the abstract base classes and core data structures
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.utils.base import (
    Chain, TradingMode, TokenInfo, AgentRule, Module,
    Discoverer, Evaluator, MLAnalyzer, RLAgent, TradingEngine, AgentFramework,
    RLTEException, ConfigurationError, APIError, TradingError, AgentError
)


class TestEnums:
    """Test enum definitions"""

    def test_chain_enum(self):
        """Test Chain enum values"""
        assert Chain.ETHEREUM.value == "ethereum"
        assert Chain.SOLANA.value == "solana"
        assert Chain.BASE.value == "base"
        
        # Test enum membership
        assert "ethereum" in [chain.value for chain in Chain]
        assert len(Chain) == 3

    def test_trading_mode_enum(self):
        """Test TradingMode enum values"""
        assert TradingMode.ANALYSIS.value == "analysis"
        assert TradingMode.SIMULATION.value == "simulation"
        assert TradingMode.LIVE.value == "live"
        
        assert len(TradingMode) == 3


class TestDataClasses:
    """Test data class structures"""

    def test_token_info_creation(self):
        """Test TokenInfo data class"""
        token = TokenInfo(
            address="0x123",
            symbol="TEST",
            name="Test Token",
            chain=Chain.ETHEREUM,
            decimals=18
        )
        
        assert token.address == "0x123"
        assert token.symbol == "TEST"
        assert token.name == "Test Token"
        assert token.chain == Chain.ETHEREUM
        assert token.decimals == 18

    def test_token_info_equality(self):
        """Test TokenInfo equality comparison"""
        token1 = TokenInfo("0x123", "TEST", "Test", Chain.ETHEREUM, 18)
        token2 = TokenInfo("0x123", "TEST", "Test", Chain.ETHEREUM, 18)
        token3 = TokenInfo("0x456", "TEST", "Test", Chain.ETHEREUM, 18)
        
        assert token1 == token2
        assert token1 != token3

    def test_agent_rule_creation(self):
        """Test AgentRule data class"""
        rule = AgentRule(
            id="rule-1",
            prompt_text="Buy when price drops 10%",
            rule_type="dca",
            parsed_conditions={"threshold": 0.1, "action": "buy"},
            active=True,
            priority=5
        )
        
        assert rule.id == "rule-1"
        assert rule.prompt_text == "Buy when price drops 10%"
        assert rule.rule_type == "dca"
        assert rule.parsed_conditions["threshold"] == 0.1
        assert rule.active is True
        assert rule.priority == 5

    def test_agent_rule_defaults(self):
        """Test AgentRule default values"""
        rule = AgentRule(
            id="rule-2",
            prompt_text="Test rule",
            rule_type="filter",
            parsed_conditions={}
        )
        
        assert rule.active is True  # Default
        assert rule.priority == 0   # Default


class TestExceptions:
    """Test custom exception classes"""

    def test_rlte_exception(self):
        """Test base RLTEException"""
        with pytest.raises(RLTEException) as exc_info:
            raise RLTEException("Test error")
        
        assert str(exc_info.value) == "Test error"

    def test_configuration_error(self):
        """Test ConfigurationError inheritance"""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Config error")
        
        assert str(exc_info.value) == "Config error"
        assert isinstance(exc_info.value, RLTEException)

    def test_api_error(self):
        """Test APIError inheritance"""
        with pytest.raises(APIError) as exc_info:
            raise APIError("API error")
        
        assert isinstance(exc_info.value, RLTEException)

    def test_trading_error(self):
        """Test TradingError inheritance"""
        with pytest.raises(TradingError) as exc_info:
            raise TradingError("Trading error")
        
        assert isinstance(exc_info.value, RLTEException)

    def test_agent_error(self):
        """Test AgentError inheritance"""
        with pytest.raises(AgentError) as exc_info:
            raise AgentError("Agent error")
        
        assert isinstance(exc_info.value, RLTEException)


class TestModule:
    """Test Module base class"""

    def test_module_initialization(self):
        """Test Module initialization"""
        config = {"test_key": "test_value"}
        
        # Create a concrete implementation for testing
        class TestModule(Module):
            async def process(self, data, agent_rules=None):
                return data
        
        module = TestModule(config)
        assert module.config == config
        assert module.agent_rules == []

    def test_module_apply_agent_rules(self):
        """Test apply_agent_rules default implementation"""
        class TestModule(Module):
            async def process(self, data, agent_rules=None):
                return self.apply_agent_rules(data, agent_rules or [])
        
        module = TestModule({})
        test_data = {"test": "data"}
        test_rules = [AgentRule("1", "test", "filter", {}, True, 0)]
        
        # Default implementation should return data unchanged
        result = module.apply_agent_rules(test_data, test_rules)
        assert result == test_data

    def test_module_validate_config(self):
        """Test validate_config default implementation"""
        class TestModule(Module):
            async def process(self, data, agent_rules=None):
                return data
        
        module = TestModule({})
        assert module.validate_config() is True


class TestAbstractClasses:
    """Test abstract base classes cannot be instantiated"""

    def test_module_abstract(self):
        """Test Module is abstract"""
        with pytest.raises(TypeError):
            Module({})

    def test_discoverer_abstract(self):
        """Test Discoverer is abstract"""
        with pytest.raises(TypeError):
            Discoverer({})

    def test_evaluator_abstract(self):
        """Test Evaluator is abstract"""
        with pytest.raises(TypeError):
            Evaluator({})

    def test_ml_analyzer_abstract(self):
        """Test MLAnalyzer is abstract"""
        with pytest.raises(TypeError):
            MLAnalyzer({})

    def test_rl_agent_abstract(self):
        """Test RLAgent is abstract"""
        with pytest.raises(TypeError):
            RLAgent({})

    def test_trading_engine_abstract(self):
        """Test TradingEngine is abstract"""
        with pytest.raises(TypeError):
            TradingEngine({})

    def test_agent_framework_abstract(self):
        """Test AgentFramework is abstract"""
        with pytest.raises(TypeError):
            AgentFramework({})


class TestConcreteImplementations:
    """Test concrete implementations of abstract classes"""

    def test_discoverer_implementation(self):
        """Test concrete Discoverer implementation"""
        class TestDiscoverer(Discoverer):
            async def scan_chain(self, chain: Chain):
                return [TokenInfo("0x123", "TEST", "Test", chain, 18)]
            
            async def scan_all_chains(self):
                results = []
                for chain in Chain:
                    results.extend(await self.scan_chain(chain))
                return results
            
            async def process(self, data, agent_rules=None):
                return await self.scan_all_chains()
        
        discoverer = TestDiscoverer({})
        assert discoverer.config == {}

    def test_evaluator_implementation(self):
        """Test concrete Evaluator implementation"""
        class TestEvaluator(Evaluator):
            async def evaluate_fundamentals(self, token: TokenInfo):
                return {
                    "liquidity": 10000,
                    "holders": 100,
                    "risk_score": 0.3
                }
            
            async def filter_candidates(self, tokens):
                # Simple filter: only tokens with >50 holders
                filtered = []
                for token in tokens:
                    fundamentals = await self.evaluate_fundamentals(token)
                    if fundamentals["holders"] > 50:
                        filtered.append(token)
                return filtered
            
            async def process(self, data, agent_rules=None):
                if isinstance(data, list):
                    return await self.filter_candidates(data)
                return await self.evaluate_fundamentals(data)
        
        evaluator = TestEvaluator({})
        assert hasattr(evaluator, 'evaluate_fundamentals')
        assert hasattr(evaluator, 'filter_candidates')

    def test_ml_analyzer_implementation(self):
        """Test concrete MLAnalyzer implementation"""
        class TestMLAnalyzer(MLAnalyzer):
            async def predict_price(self, token: TokenInfo, features):
                return {
                    "predicted_price": features.get("current_price", 0) * 1.1,
                    "confidence": 0.85,
                    "direction": "up"
                }
            
            async def calculate_features(self, token: TokenInfo):
                return {
                    "current_price": 0.00123,
                    "volume_24h": 50000,
                    "market_cap": 1000000
                }
            
            async def process(self, data, agent_rules=None):
                features = await self.calculate_features(data)
                return await self.predict_price(data, features)
        
        analyzer = TestMLAnalyzer({})
        assert hasattr(analyzer, 'predict_price')
        assert hasattr(analyzer, 'calculate_features')

    def test_rl_agent_implementation(self):
        """Test concrete RLAgent implementation"""
        class TestRLAgent(RLAgent):
            def __init__(self, config):
                super().__init__(config)
                self.q_values = {}
            
            async def act(self, state, training=False):
                # Simple action selection
                return 1  # Always buy
            
            async def update(self, experience):
                # Simple Q-value update
                state_key = str(experience.get("state", ""))
                self.q_values[state_key] = experience.get("reward", 0)
            
            async def process(self, data, agent_rules=None):
                return await self.act(data)
        
        agent = TestRLAgent({})
        assert hasattr(agent, 'act')
        assert hasattr(agent, 'update')
        assert agent.q_values == {}

    @pytest.mark.asyncio
    async def test_async_method_calls(self):
        """Test async method calls on concrete implementations"""
        class TestModule(Module):
            async def process(self, data, agent_rules=None):
                return {"processed": data}
        
        module = TestModule({"test": True})
        result = await module.process("test_data")
        
        assert result == {"processed": "test_data"}


class TestModuleIntegration:
    """Test module integration patterns"""

    @pytest.mark.asyncio
    async def test_module_chaining(self):
        """Test chaining modules together"""
        class StepOneModule(Module):
            async def process(self, data, agent_rules=None):
                return {"step1": data, "value": 1}
        
        class StepTwoModule(Module):
            async def process(self, data, agent_rules=None):
                return {"step2": data, "value": data.get("value", 0) * 2}
        
        step1 = StepOneModule({})
        step2 = StepTwoModule({})
        
        # Chain the modules
        result1 = await step1.process("input")
        result2 = await step2.process(result1)
        
        assert result2["step2"]["step1"] == "input"
        assert result2["value"] == 2

    @pytest.mark.asyncio
    async def test_agent_rules_integration(self):
        """Test agent rules integration with modules"""
        class RuleAwareModule(Module):
            async def process(self, data, agent_rules=None):
                if agent_rules:
                    # Apply rules to modify behavior
                    multiplier = 1
                    for rule in agent_rules:
                        if rule.rule_type == "boost":
                            multiplier *= rule.parsed_conditions.get("factor", 1)
                    
                    return {"data": data, "multiplier": multiplier}
                return {"data": data, "multiplier": 1}
        
        module = RuleAwareModule({})
        boost_rule = AgentRule(
            id="boost-1",
            prompt_text="Boost by 2x",
            rule_type="boost",
            parsed_conditions={"factor": 2},
            active=True
        )
        
        # Test without rules
        result1 = await module.process("test")
        assert result1["multiplier"] == 1
        
        # Test with rules
        result2 = await module.process("test", [boost_rule])
        assert result2["multiplier"] == 2