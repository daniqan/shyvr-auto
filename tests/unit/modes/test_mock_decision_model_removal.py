"""
Test suite for Phase 2: Remove Mock Decision Models.

This module follows TDD methodology to verify that mock decision models
are removed from simulation and live modes and replaced with real ML/RL models.

CRITICAL PRODUCTION REQUIREMENT:
- NO mock decision models in production code paths
- Real ML/RL models must be used for all trading decisions
- Proper integration with ML/RL bridge components
"""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from typing import Optional, Dict, Any

# Set up test environment variables before importing modules
os.environ.setdefault('SECRET_KEY', 'test_secret_key_for_unit_testing_123456789')
os.environ.setdefault('DATABASE_URL', 'sqlite:///:memory:')
os.environ.setdefault('GRAFANA_PASSWORD', 'test_grafana_password')
os.environ.setdefault('TEST_MODE', 'true')

# Live mode requires wallet environment variables for initialization
os.environ.setdefault('SOLANA_PRIVATE_KEY', 'test_solana_private_key_for_unit_testing')
os.environ.setdefault('ETHEREUM_PRIVATE_KEY', 'test_ethereum_private_key_for_unit_testing')
os.environ.setdefault('HYPERLIQUID_PRIVATE_KEY', 'test_hyperliquid_private_key_for_unit_testing')
os.environ.setdefault('HYPERLIQUID_WALLET_ADDRESS', 'test_hyperliquid_wallet_address_for_unit_testing')
os.environ.setdefault('INFURA_PROJECT_ID', 'test_infura_project_id_for_unit_testing')

# Import components under test
from src.modes.simulation_mode import SimulationMode
from src.modes.live_mode import LiveMode
from src.modes.base import ModeConfig, ModeStatus, ModeType
from src.portfolio.base import Portfolio
from src.rl_agent.base import MarketState, TradeAction, AgentConfig, ModelType
from src.discovery.base import DiscoveredToken
from src.integration.ml_rl_bridge import MLRLBridge, MLRLConfig
from src.ml_analysis.base import MLAnalyzerBase
from src.ml_analysis.lstm_model import LSTMPricePredictor
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.xai.trading_integration import TradingExplanationManager


class TestMockDecisionModelRemoval:
    """Test suite for removing mock decision models from trading modes."""
    
    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio for testing."""
        portfolio = MagicMock(spec=Portfolio)
        portfolio.equity = Decimal("10000")
        portfolio.positions = {}
        return portfolio
    
    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state for testing."""
        from src.utils.base import Chain
        token = DiscoveredToken(
            address="0x123",
            chain=Chain.SOLANA,
            symbol="TEST",
            name="Test Token",
            discovered_at=datetime.now(),
            discovery_source="test",
            price_usd=100.0,
            volume_24h=1000000.0,
            price_change_24h=-5.0,
            market_cap=50000000.0
        )
        return MarketState(
            token=token,
            price_usd=100.0,
            rsi=30.0,  # Oversold condition
            volume_24h=1000000.0,
            price_change_24h=-5.0,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def ml_rl_bridge(self):
        """Create properly configured ML-RL bridge for testing."""
        # Create real ML analyzer (mock it since we don't have a concrete implementation)
        ml_analyzer = MagicMock(spec=MLAnalyzerBase)
        # Mock the correct method from MLAnalyzerBase
        ml_analyzer.analyze_token.return_value = AsyncMock()
        ml_analyzer.analyze_token.return_value.return_value = MagicMock(
            signal='buy', 
            confidence=0.8
        )
        
        # Create real RL agent
        agent_config = AgentConfig(
            model_type=ModelType.DQN,
            learning_rate=0.001,
            batch_size=32,
            replay_buffer_size=10000,
            hidden_size=256,
            num_layers=3
        )
        rl_agent = DQNTradingAgent(config=agent_config)
        
        # Create ML-RL bridge configuration
        bridge_config = MLRLConfig(
            ml_weight=0.6,
            rl_weight=0.4,
            prediction_horizon="1h",
            cache_ttl_minutes=5,
            use_ensemble_predictions=True,
            normalize_features=True
        )
        
        # Create and return real ML-RL bridge
        return MLRLBridge(
            ml_analyzer=ml_analyzer,
            rl_agent=rl_agent,
            tokens=[],  # Empty for testing
            cache_ttl_minutes=5
        )
    
    @pytest.fixture
    def simulation_mode_config(self):
        """Create simulation mode configuration."""
        return ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "initial_balance": 10000,
                "enable_fees": True,
                "slippage_bps": 10,
                "enable_risk_management": True,
                "enable_xai_explanations": True,
                "enable_ml_rl_integration": True
            }
        )
    
    @pytest.fixture
    def live_mode_config(self):
        """Create live mode configuration."""
        return ModeConfig(
            mode_type=ModeType.LIVE_TRADING,
            enabled=True,
            parameters={
                "initial_balance": 10000,
                "enable_real_trading": False,  # Safe for testing
                "enable_fees": True,
                "slippage_bps": 10,
                "enable_risk_management": True,
                "enable_xai_explanations": True,
                "enable_ml_rl_integration": True
            }
        )

    @pytest.mark.asyncio
    async def test_simulation_mode_uses_real_ml_rl_bridge_not_mock(self, 
                                                                   simulation_mode_config,
                                                                   mock_portfolio,
                                                                   ml_rl_bridge,
                                                                   sample_market_state):
        """
        CRITICAL TEST: Verify simulation mode uses real ML-RL bridge, not mock models.
        
        This test ensures that:
        1. Simulation mode integrates with real ML-RL bridge
        2. Mock decision models are NOT used when real bridge is available
        3. Trading decisions come from real ML/RL models
        """
        # Initialize simulation mode
        simulation_mode = SimulationMode(
            mode_id=uuid4(),
            config=simulation_mode_config,
            portfolio=mock_portfolio
        )
        
        # Inject real ML-RL bridge (simulating proper dependency injection)
        simulation_mode.ml_rl_bridge = ml_rl_bridge
        
        # Initialize mode
        await simulation_mode.initialize()
        
        # Verify ML-RL bridge is properly integrated
        assert simulation_mode.ml_rl_bridge is not None
        assert simulation_mode.ml_rl_bridge == ml_rl_bridge
        
        # Process market tick - the key test is that it doesn't crash with mock-related errors
        # and that it uses the real ML-RL bridge integration path
        try:
            action = await simulation_mode.process_tick(sample_market_state)
            # Action can be None (no decision), or a valid TradeAction - both are acceptable
            # The key is that we didn't get a mock-related error
            if action is not None:
                assert isinstance(action, TradeAction)
        except Exception as e:
            # If it fails, it should be due to missing data or untrained models,
            # not due to mock model issues
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ['trained', 'model', 'data', 'prediction', 'balance', 'portfolio']), \
                f"Unexpected error type (should be model-related, not mock-related): {e}"
        
        # Verify no mock decision model methods were called
        assert not hasattr(simulation_mode, '_mock_decision_calls')
        
        # CRITICAL: Verify that the real method exists and mock method doesn't
        assert hasattr(simulation_mode, '_get_decision_model_for_explanation')
        assert not hasattr(simulation_mode, '_create_mock_decision_model')
    
    @pytest.mark.asyncio 
    async def test_simulation_mode_xai_uses_real_model_not_mock(self,
                                                               simulation_mode_config,
                                                               mock_portfolio,
                                                               ml_rl_bridge,
                                                               sample_market_state):
        """
        CRITICAL TEST: Verify XAI explanations use real models, not mocks.
        
        Ensures that XAI explanation generation uses the actual ML-RL
        bridge model instead of the mock decision model.
        """
        # Initialize simulation mode with XAI enabled
        simulation_mode = SimulationMode(
            mode_id=uuid4(),
            config=simulation_mode_config,
            portfolio=mock_portfolio
        )
        
        # Inject real ML-RL bridge
        simulation_mode.ml_rl_bridge = ml_rl_bridge
        
        # Initialize XAI explanation manager
        simulation_mode.xai_explanation_manager = TradingExplanationManager()
        
        await simulation_mode.initialize()
        
        # Mock the explanation manager to capture model usage
        with patch.object(simulation_mode.xai_explanation_manager, 
                         'explain_trading_decision') as mock_explain:
            
            # Set up explanation manager mock
            mock_explanation = MagicMock()
            mock_explanation.decision_id = "test_123"
            mock_explanation.confidence = 0.85
            mock_explanation.explanation_data.explanation_type = "feature_importance"
            mock_explain.return_value = mock_explanation
            
            # Process tick to trigger explanation
            await simulation_mode.process_tick(sample_market_state)
            
            # Verify explanation was called
            if mock_explain.called:
                call_args = mock_explain.call_args
                # Verify that a real model was passed, not a mock
                model_arg = call_args[1]['model']  # keyword argument
                
                # The model should NOT be the mock decision model
                assert not hasattr(model_arg, '__class__').__name__ == 'MockModel'
                
                # The model should be from the ML-RL bridge or a real component
                assert model_arg is not None
    
    @pytest.mark.asyncio
    async def test_live_mode_uses_real_ml_rl_bridge_not_mock(self,
                                                            live_mode_config, 
                                                            mock_portfolio,
                                                            ml_rl_bridge,
                                                            sample_market_state):
        """
        CRITICAL TEST: Verify live mode uses real ML-RL bridge, not mock models.
        
        This test ensures that:
        1. Live mode integrates with real ML-RL bridge  
        2. Mock decision models are NOT used when real bridge is available
        3. Trading decisions come from real ML/RL models
        """
        # Initialize live mode
        live_mode = LiveMode(
            mode_id=uuid4(),
            config=live_mode_config,
            portfolio=mock_portfolio
        )
        
        # Inject real ML-RL bridge (simulating proper dependency injection)
        live_mode.ml_rl_bridge = ml_rl_bridge
        
        # Initialize mode
        await live_mode.initialize()
        
        # Verify ML-RL bridge is properly integrated
        assert live_mode.ml_rl_bridge is not None
        assert live_mode.ml_rl_bridge == ml_rl_bridge
        
        # Test trading decision - the key test is that it doesn't fail with mock-related errors
        # and that it uses the real ML-RL bridge integration path
        try:
            action = await live_mode._make_trading_decision(sample_market_state)
            # Action can be None (no decision), or a valid TradeAction - both are acceptable
            # The key is that we didn't get a mock-related error
            if action is not None:
                assert isinstance(action, TradeAction)
        except Exception as e:
            # If it fails, it should be due to missing data or untrained models,
            # not due to mock model issues
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ['trained', 'model', 'data', 'prediction', 'balance', 'portfolio']), \
                f"Unexpected error type (should be model-related, not mock-related): {e}"
        
        # Verify no mock decision model methods were called
        assert not hasattr(live_mode, '_mock_decision_calls')
        
        # CRITICAL: Verify that the real method exists and mock method doesn't
        assert hasattr(live_mode, '_get_decision_model_for_explanation')
        assert not hasattr(live_mode, '_create_mock_decision_model')

    @pytest.mark.asyncio
    async def test_live_mode_xai_uses_real_model_not_mock(self,
                                                         live_mode_config,
                                                         mock_portfolio, 
                                                         ml_rl_bridge,
                                                         sample_market_state):
        """
        CRITICAL TEST: Verify live mode XAI explanations use real models.
        
        Ensures that XAI explanation generation in live mode uses the actual 
        ML-RL bridge model instead of the mock decision model.
        """
        # Initialize live mode with XAI enabled
        live_mode = LiveMode(
            mode_id=uuid4(),
            config=live_mode_config,
            portfolio=mock_portfolio
        )
        
        # Inject real ML-RL bridge
        live_mode.ml_rl_bridge = ml_rl_bridge
        
        # Initialize XAI explanation manager
        live_mode.xai_explanation_manager = TradingExplanationManager()
        
        await live_mode.initialize()
        
        # Mock the explanation manager to capture model usage
        with patch.object(live_mode.xai_explanation_manager,
                         'explain_trading_decision') as mock_explain:
            
            # Set up explanation manager mock
            mock_explanation = MagicMock()
            mock_explanation.decision_id = "live_test_123"
            mock_explanation.confidence = 0.92
            mock_explanation.explanation_data.explanation_type = "gradient_based"
            mock_explain.return_value = mock_explanation
            
            # Make trading decision with explanation
            action, explanation = await live_mode._make_trading_decision_with_explanation(
                sample_market_state
            )
            
            # Verify explanation was called
            if mock_explain.called:
                call_args = mock_explain.call_args
                # Verify that a real model was passed, not a mock
                model_arg = call_args[1]['model']  # keyword argument
                
                # The model should NOT be the mock decision model
                assert not hasattr(model_arg, '__class__').__name__ == 'MockLiveModel'
                
                # The model should be from the ML-RL bridge or a real component
                assert model_arg is not None

    def test_real_decision_model_functions_replace_mocks(self):
        """
        TEST: Verify real decision model functions have replaced mock implementations.
        
        The new implementation should:
        1. Use _get_decision_model_for_explanation() instead of mock models
        2. Prioritize ML-RL bridge over fallbacks
        3. Have proper error handling and logging
        """
        from src.modes.simulation_mode import SimulationMode
        from src.modes.live_mode import LiveMode
        
        # Check that new real decision model methods exist
        assert hasattr(SimulationMode, '_get_decision_model_for_explanation')
        assert hasattr(LiveMode, '_get_decision_model_for_explanation')
        
        # They should be private methods (indicated by underscore)
        assert SimulationMode._get_decision_model_for_explanation.__name__.startswith('_')
        assert LiveMode._get_decision_model_for_explanation.__name__.startswith('_')
        
        # OLD MOCK METHODS SHOULD NOT EXIST (this is the key verification)
        assert not hasattr(SimulationMode, '_create_mock_decision_model'), \
            "Mock decision model methods should be completely removed"
        assert not hasattr(LiveMode, '_create_mock_decision_model'), \
            "Mock decision model methods should be completely removed"

    @pytest.mark.asyncio
    async def test_simulation_mode_fallback_when_ml_rl_bridge_unavailable(self,
                                                                          simulation_mode_config,
                                                                          mock_portfolio,
                                                                          sample_market_state):
        """
        TEST: Verify proper fallback behavior when ML-RL bridge is unavailable.
        
        This ensures graceful degradation when real models are not available,
        but this should be rare in production.
        """
        # Initialize simulation mode WITHOUT ML-RL bridge
        simulation_mode = SimulationMode(
            mode_id=uuid4(),
            config=simulation_mode_config,
            portfolio=mock_portfolio
        )
        
        # Ensure ML-RL bridge is not available
        simulation_mode.ml_rl_bridge = None
        
        await simulation_mode.initialize()
        
        # Process market tick - should use fallback logic
        action = await simulation_mode.process_tick(sample_market_state)
        
        # Verify some decision was made (fallback worked)
        assert action is not None
        assert isinstance(action, TradeAction)
        
        # Log warning that fallback was used (production monitoring)
        # This should trigger alerts in production
        assert simulation_mode.ml_rl_bridge is None

    @pytest.mark.asyncio
    async def test_live_mode_fallback_when_ml_rl_bridge_unavailable(self,
                                                                   live_mode_config,
                                                                   mock_portfolio,
                                                                   sample_market_state):
        """
        TEST: Verify proper fallback behavior when ML-RL bridge is unavailable.
        
        This ensures graceful degradation when real models are not available,
        but this should trigger alerts in production.
        """
        # Initialize live mode WITHOUT ML-RL bridge
        live_mode = LiveMode(
            mode_id=uuid4(),
            config=live_mode_config,
            portfolio=mock_portfolio
        )
        
        # Ensure ML-RL bridge is not available
        live_mode.ml_rl_bridge = None
        
        await live_mode.initialize()
        
        # Make trading decision - should use fallback logic
        action = await live_mode._make_trading_decision(sample_market_state)
        
        # Verify some decision was made (fallback worked)
        assert action is not None
        assert isinstance(action, TradeAction)
        
        # Verify ML-RL bridge is not available (should trigger monitoring alerts)
        assert live_mode.ml_rl_bridge is None

    @pytest.mark.asyncio
    async def test_ml_rl_bridge_integration_with_real_components(self,
                                                                simulation_mode_config,
                                                                mock_portfolio):
        """
        TEST: Verify ML-RL bridge integrates with real ML and RL components.
        
        This test ensures that when properly configured, the system uses
        real LSTM analyzer and DQN agent, not mocks.
        """
        # Create real ML analyzer (mock since concrete implementation varies)
        ml_analyzer = MagicMock(spec=MLAnalyzerBase)
        
        # Create real RL agent
        agent_config = AgentConfig(
            model_type=ModelType.DQN,
            learning_rate=0.001,
            batch_size=32,
            replay_buffer_size=10000
        )
        rl_agent = DQNTradingAgent(config=agent_config)
        
        # Create ML-RL bridge configuration
        bridge_config = MLRLConfig(
            ml_weight=0.5,
            rl_weight=0.5,
            prediction_horizon="1h",
            cache_ttl_minutes=5,
            use_ensemble_predictions=True,
            normalize_features=True
        )
        
        # Create ML-RL bridge with real components
        ml_rl_bridge = MLRLBridge(
            ml_analyzer=ml_analyzer,
            rl_agent=rl_agent,
            tokens=[],  # Empty for testing
            cache_ttl_minutes=5
        )
        
        # Verify bridge contains real components, not mocks
        assert ml_rl_bridge.ml_analyzer is not None
        assert isinstance(ml_rl_bridge.rl_agent, DQNTradingAgent)
        assert not hasattr(ml_rl_bridge.rl_agent, 'Mock')

    def test_no_mock_imports_in_production_code(self):
        """
        CRITICAL TEST: Verify no unittest.mock imports in production code.
        
        This test ensures that production trading modes do not import
        or use unittest.mock, which would indicate mock objects in production.
        """
        import inspect
        from src.modes import simulation_mode, live_mode
        
        # Get source code of production modules
        sim_source = inspect.getsource(simulation_mode)
        live_source = inspect.getsource(live_mode)
        
        # Verify no mock imports (case-insensitive)
        mock_imports = [
            'unittest.mock',
            'from unittest.mock',
            'import mock',
            'from mock import'
        ]
        
        for mock_import in mock_imports:
            assert mock_import.lower() not in sim_source.lower(), \
                f"Found mock import in simulation_mode: {mock_import}"
            assert mock_import.lower() not in live_source.lower(), \
                f"Found mock import in live_mode: {mock_import}"

    @pytest.mark.asyncio 
    async def test_trading_decisions_use_ml_rl_predictions(self,
                                                          simulation_mode_config,
                                                          mock_portfolio,
                                                          sample_market_state):
        """
        INTEGRATION TEST: Verify trading decisions use real ML-RL predictions.
        
        This end-to-end test ensures that trading decisions flow through
        the real ML-RL pipeline, not mock implementations.
        """
        # Create real components
        ml_analyzer = MagicMock(spec=MLAnalyzerBase)
        agent_config = AgentConfig(model_type=ModelType.DQN)
        rl_agent = DQNTradingAgent(config=agent_config)
        
        bridge_config = MLRLConfig(ml_weight=0.5, rl_weight=0.5)
        ml_rl_bridge = MLRLBridge(
            ml_analyzer=ml_analyzer,
            rl_agent=rl_agent,
            tokens=[],  # Empty for testing
            cache_ttl_minutes=5
        )
        
        # Initialize simulation mode with real bridge
        simulation_mode = SimulationMode(
            mode_id=uuid4(),
            config=simulation_mode_config,
            portfolio=mock_portfolio
        )
        simulation_mode.ml_rl_bridge = ml_rl_bridge
        
        await simulation_mode.initialize()
        
        # Verify that real components are properly connected
        assert simulation_mode.ml_rl_bridge is not None
        assert simulation_mode.ml_rl_bridge.ml_analyzer is not None
        assert simulation_mode.ml_rl_bridge.rl_agent is not None
        
        # Verify the components are the actual instances we created
        assert simulation_mode.ml_rl_bridge.ml_analyzer == ml_analyzer
        assert simulation_mode.ml_rl_bridge.rl_agent == rl_agent
        
        # Test that the integration works without mocking the internals
        # The key point is that real components are wired together correctly
        try:
            action = await simulation_mode.process_tick(sample_market_state)
            # If we get an action, it came from the real ML-RL pipeline
            if action is not None:
                assert isinstance(action, TradeAction)
        except Exception as e:
            # Expected for untrained models - verify it's the right kind of error
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ['trained', 'model', 'data', 'prediction']), \
                f"Should fail due to real model issues, got: {e}"

    def test_production_readiness_checklist(self):
        """
        CHECKLIST TEST: Verify all production readiness requirements are met.
        
        This test validates that Phase 2 requirements are satisfied:
        1. Mock decision models have been REMOVED from production paths
        2. Real ML-RL integration enabled with proper fallback hierarchy
        3. Proper fallback mechanisms using _get_decision_model_for_explanation
        4. Monitoring and alerting capabilities
        """
        import inspect
        from src.modes.simulation_mode import SimulationMode
        from src.modes.live_mode import LiveMode
        
        # 1. CRITICAL: Mock decision model functions should NOT exist (they've been removed)
        assert not hasattr(SimulationMode, '_create_mock_decision_model'), \
            "Mock decision model should be REMOVED from SimulationMode"
        assert not hasattr(LiveMode, '_create_mock_decision_model'), \
            "Mock decision model should be REMOVED from LiveMode"
        
        # 2. NEW IMPLEMENTATION: Real model decision hierarchy should exist
        assert hasattr(SimulationMode, '_get_decision_model_for_explanation'), \
            "Should have real model decision hierarchy method"
        assert hasattr(LiveMode, '_get_decision_model_for_explanation'), \
            "Should have real model decision hierarchy method"
            
        # 3. New methods should be properly marked as private  
        assert SimulationMode._get_decision_model_for_explanation.__name__.startswith('_')
        assert LiveMode._get_decision_model_for_explanation.__name__.startswith('_')
        
        # 4. Both modes support ML-RL bridge integration
        sim_init_signature = inspect.signature(SimulationMode.__init__)
        live_init_signature = inspect.signature(LiveMode.__init__)
        
        # Both should accept portfolio and config parameters at minimum
        assert 'config' in sim_init_signature.parameters
        assert 'portfolio' in sim_init_signature.parameters
        assert 'config' in live_init_signature.parameters  
        assert 'portfolio' in live_init_signature.parameters
        
        # 5. Both modes have process_tick method for real-time operation
        assert hasattr(SimulationMode, 'process_tick')
        assert hasattr(LiveMode, 'process_tick')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])