"""
End-to-End Pipeline Validation Tests

Tests the complete ML→RL→Trading pipeline integration to ensure
all components work together seamlessly from token discovery 
through trading execution.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
import pandas as pd

from src.discovery.base import DiscoveredToken, TokenStatus
from src.utils.base import Chain
from src.ml_analysis.base import PredictionResult, ModelType, TechnicalIndicators
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.integration.ml_rl_bridge import MLEnhancedMarketState, MLRLBridge
from src.rl_agent.trading_environment import TradingEnvironment, EnvironmentConfig
from src.rl_agent.dqn_agent import DQNTradingAgent, AgentConfig


@pytest.fixture
def sample_discovered_tokens():
    """Create sample discovered tokens for testing"""
    return [
        DiscoveredToken(
            address="0x1234567890abcdef",
            symbol="TEST1",
            name="Test Token 1",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="testing",
            status=TokenStatus.DISCOVERED,
            price_usd=0.001234,
            volume_24h=50000.0,
            market_cap=1000000.0,
            price_change_24h=15.2,
            decimals=18,
            total_supply=1000000000.0,
            trending_score=0.75
        ),
        DiscoveredToken(
            address="0xabcdef1234567890",
            symbol="TEST2", 
            name="Test Token 2",
            chain=Chain.SOLANA,
            discovered_at=datetime.now(),
            discovery_source="testing",
            status=TokenStatus.DISCOVERED,
            price_usd=0.00892,
            volume_24h=75000.0,
            market_cap=2500000.0,
            price_change_24h=8.7,
            decimals=6,
            total_supply=500000000.0,
            trending_score=0.65
        )
    ]


@pytest.fixture
def sample_ml_predictions(sample_discovered_tokens):
    """Create sample ML predictions for testing"""
    predictions = []
    for token in sample_discovered_tokens:
        # Create technical indicators
        tech_indicators = TechnicalIndicators(
            rsi=65.2,
            macd=0.0012,
            sma_20=token.price_usd * 0.98,
            ema_12=token.price_usd * 1.01,
            bollinger_upper=token.price_usd * 1.05,
            bollinger_lower=token.price_usd * 0.95
        )
        
        prediction = PredictionResult(
            token=token,
            analyzed_at=datetime.now(),
            model_type=ModelType.LSTM,
            price_prediction_1h=token.price_usd * 1.02,
            price_prediction_4h=token.price_usd * 1.05,
            price_prediction_24h=token.price_usd * 1.08,
            confidence=0.75,
            technical_indicators=tech_indicators,
            volatility_forecast=0.15
        )
        predictions.append(prediction)
    
    return predictions


@pytest.fixture
def mock_historical_data():
    """Create mock historical price data"""
    dates = pd.date_range(start='2025-01-01', periods=100, freq='1H')
    return {
        'timestamps': dates,
        'prices': np.random.uniform(0.001, 0.01, 100),
        'volumes': np.random.uniform(1000, 100000, 100)
    }


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality"""
    
    def test_ml_to_rl_data_flow(self, sample_discovered_tokens, sample_ml_predictions):
        """Test data flows correctly from ML predictions to RL agent"""
        token = sample_discovered_tokens[0]
        prediction = sample_ml_predictions[0]
        
        # Create enhanced market state from ML prediction
        enhanced_state = MLEnhancedMarketState.from_prediction(
            prediction=prediction,
            current_portfolio_value=10000.0,
            position_size=0.0
        )
        
        # Verify ML features are properly integrated
        assert enhanced_state.ml_prediction_1h == prediction.price_prediction_1h
        assert enhanced_state.ml_prediction_4h == prediction.price_prediction_4h
        assert enhanced_state.ml_prediction_24h == prediction.price_prediction_24h
        assert enhanced_state.ml_confidence == prediction.confidence
        assert enhanced_state.volatility_forecast == prediction.volatility_forecast
        
        # Verify feature vector has correct dimensions (25 features: 19 base + 6 ML)
        feature_vector = enhanced_state.to_feature_vector()
        assert len(feature_vector) == 25
        assert feature_vector.dtype == np.float32
        
        # Verify all features are properly normalized
        assert np.all(np.isfinite(feature_vector))
        assert not np.any(np.isnan(feature_vector))

    def test_rl_agent_decision_making(self, sample_discovered_tokens, sample_ml_predictions):
        """Test RL agent makes decisions based on ML-enhanced market state"""
        # Create mock RL agent
        agent_config = AgentConfig(
            learning_rate=0.001,
            epsilon_start=0.1,  # Low exploration for testing
            batch_size=32
        )
        
        with patch('src.rl_agent.dqn_agent.DQNNetwork') as mock_network:
            # Mock neural network to return deterministic action
            mock_model = MagicMock()
            mock_model.forward.return_value = [2.0, 1.0, 0.5, 1.5, 0.8]  # Action values
            mock_network.return_value = mock_model
            
            agent = DQNTradingAgent(agent_config)
            
            # Create enhanced market state
            prediction = sample_ml_predictions[0]
            enhanced_state = MLEnhancedMarketState.from_prediction(
                prediction=prediction,
                current_portfolio_value=10000.0,
                position_size=0.0
            )
            
            # Get RL action
            action = agent.predict_action(enhanced_state)
            
            # Verify action is valid
            assert isinstance(action, TradeAction)
            assert action in [TradeAction.HOLD, TradeAction.BUY, TradeAction.SELL, 
                            TradeAction.STRONG_BUY, TradeAction.STRONG_SELL]

    def test_trading_environment_execution(self, sample_discovered_tokens, mock_historical_data):
        """Test trading environment executes trades correctly"""
        env_config = EnvironmentConfig(
            max_episode_steps=50,
            transaction_fee=0.001,
            slippage_factor=0.002
        )
        
        environment = TradingEnvironment(
            config=env_config,
            tokens=sample_discovered_tokens,
            historical_data=mock_historical_data
        )
        
        # Reset environment
        initial_state = environment.reset()
        assert isinstance(initial_state, MarketState)
        
        # Execute a buy action
        next_state, reward, done, info = environment.step(TradeAction.BUY)
        
        # Verify trade execution
        assert isinstance(next_state, MarketState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Verify portfolio has been updated
        assert environment.portfolio.cash_balance < environment.config.initial_balance
        assert len(environment.portfolio.positions) > 0

    def test_complete_pipeline_integration(self, sample_discovered_tokens, 
                                         sample_ml_predictions, mock_historical_data):
        """Test complete pipeline from ML predictions to trading execution"""
        # Step 1: Create ML-RL bridge
        mock_ml_analyzer = MagicMock()
        mock_ml_analyzer.analyze_batch.return_value = sample_ml_predictions
        
        agent_config = AgentConfig(epsilon_start=0.1)
        
        with patch('src.rl_agent.dqn_agent.DQNNetwork') as mock_network:
            mock_model = MagicMock()
            mock_model.forward.return_value = [2.0, 1.0, 0.5, 1.5, 0.8]
            mock_network.return_value = mock_model
            
            mock_rl_agent = DQNTradingAgent(agent_config)
            
            bridge = MLRLBridge(
                ml_analyzer=mock_ml_analyzer,
                rl_agent=mock_rl_agent,
                tokens=sample_discovered_tokens
            )
            
            # Step 2: Get integrated ML-RL decisions
            portfolio_value = 10000.0
            positions = {}
            results = bridge.predict_and_act(portfolio_value, positions)
            
            # Verify integrated results
            assert len(results) == len(sample_discovered_tokens)
            for result in results:
                assert 'token' in result
                assert 'ml_prediction' in result
                assert 'rl_action' in result
                assert 'enhanced_state' in result
                assert 'confidence' in result
                
                # Verify data types
                assert isinstance(result['ml_prediction'], PredictionResult)
                assert isinstance(result['rl_action'], TradeAction)
                assert isinstance(result['enhanced_state'], MLEnhancedMarketState)
                assert isinstance(result['confidence'], float)
        
        # Step 3: Execute trades in environment
        env_config = EnvironmentConfig(max_episode_steps=10)
        environment = TradingEnvironment(
            config=env_config,
            tokens=sample_discovered_tokens,
            historical_data=mock_historical_data
        )
        
        environment.reset()
        
        # Execute each recommended action
        total_reward = 0.0
        for result in results:
            next_state, reward, done, info = environment.step(result['rl_action'])
            total_reward += reward
            
            if done:
                break
        
        # Verify pipeline execution completed successfully
        assert isinstance(total_reward, float)
        assert environment.current_step > 0

    def test_pipeline_performance_metrics(self, sample_discovered_tokens, 
                                        sample_ml_predictions, mock_historical_data):
        """Test pipeline generates proper performance metrics"""
        # Create ML-RL bridge with performance tracking
        mock_ml_analyzer = MagicMock()
        mock_ml_analyzer.analyze_batch.return_value = sample_ml_predictions
        
        with patch('src.rl_agent.dqn_agent.DQNNetwork') as mock_network:
            mock_model = MagicMock()
            mock_model.forward.return_value = [1.5, 2.0, 0.5, 1.0, 0.8]
            mock_network.return_value = mock_model
            
            agent_config = AgentConfig(epsilon_start=0.0)  # No exploration
            mock_rl_agent = DQNTradingAgent(agent_config)
            
            bridge = MLRLBridge(
                ml_analyzer=mock_ml_analyzer,
                rl_agent=mock_rl_agent,
                tokens=sample_discovered_tokens
            )
            
            # Measure prediction and decision times
            start_time = datetime.now()
            results = bridge.predict_and_act(10000.0, {})
            end_time = datetime.now()
            
            # Verify performance targets
            execution_time = (end_time - start_time).total_seconds()
            assert execution_time < 1.0  # Should complete in <1 second
            
            # Verify prediction quality
            for result in results:
                assert result['confidence'] > 0.0
                assert result['ml_prediction'].confidence > 0.0
                
                # Verify ML predictions are reasonable
                ml_pred = result['ml_prediction']
                current_price = ml_pred.token.price_usd
                assert ml_pred.price_prediction_1h > current_price * 0.5
                assert ml_pred.price_prediction_1h < current_price * 2.0

    def test_pipeline_error_handling(self, sample_discovered_tokens):
        """Test pipeline handles errors gracefully"""
        # Test with failing ML analyzer
        mock_ml_analyzer = MagicMock()
        mock_ml_analyzer.analyze_batch.side_effect = Exception("ML model failed")
        
        with patch('src.rl_agent.dqn_agent.DQNNetwork') as mock_network:
            mock_network.return_value = MagicMock()
            
            agent_config = AgentConfig()
            mock_rl_agent = DQNTradingAgent(agent_config)
            
            bridge = MLRLBridge(
                ml_analyzer=mock_ml_analyzer,
                rl_agent=mock_rl_agent,
                tokens=sample_discovered_tokens
            )
            
            # Should handle ML failure gracefully
            with pytest.raises(Exception):
                bridge.predict_and_act(10000.0, {})

    def test_pipeline_caching_behavior(self, sample_discovered_tokens, sample_ml_predictions):
        """Test ML prediction caching works correctly in pipeline"""
        mock_ml_analyzer = MagicMock()
        mock_ml_analyzer.analyze_batch.return_value = sample_ml_predictions
        
        with patch('src.rl_agent.dqn_agent.DQNNetwork') as mock_network:
            mock_network.return_value = MagicMock()
            
            agent_config = AgentConfig()
            mock_rl_agent = DQNTradingAgent(agent_config)
            
            bridge = MLRLBridge(
                ml_analyzer=mock_ml_analyzer,
                rl_agent=mock_rl_agent,
                tokens=sample_discovered_tokens,
                cache_ttl_minutes=5
            )
            
            # First call should trigger ML analysis
            results1 = bridge.predict_and_act(10000.0, {})
            assert mock_ml_analyzer.analyze_batch.call_count == 1
            
            # Second call within cache TTL should use cache
            results2 = bridge.predict_and_act(10000.0, {})
            assert mock_ml_analyzer.analyze_batch.call_count == 1  # No additional calls
            
            # Results should be identical due to caching
            assert len(results1) == len(results2)

    @pytest.mark.asyncio
    async def test_pipeline_scalability(self, mock_historical_data):
        """Test pipeline can handle multiple tokens efficiently"""
        # Create larger token set
        large_token_set = []
        for i in range(10):  # Test with 10 tokens
            token = DiscoveredToken(
                address=f"0x{i:040x}",
                symbol=f"TEST{i}",
                name=f"Test Token {i}",
                chain=Chain.ETHEREUM,
                discovered_at=datetime.now(),
                discovery_source="testing",
                price_usd=0.001 * (i + 1),
                volume_24h=10000.0 * (i + 1)
            )
            large_token_set.append(token)
        
        # Create corresponding predictions
        predictions = []
        for token in large_token_set:
            prediction = PredictionResult(
                token=token,
                analyzed_at=datetime.now(),
                model_type=ModelType.LSTM,
                price_prediction_1h=token.price_usd * 1.02,
                confidence=0.7
            )
            predictions.append(prediction)
        
        # Test batch processing
        mock_ml_analyzer = MagicMock()
        mock_ml_analyzer.analyze_batch.return_value = predictions
        
        with patch('src.rl_agent.dqn_agent.DQNNetwork') as mock_network:
            mock_network.return_value = MagicMock()
            
            agent_config = AgentConfig()
            mock_rl_agent = DQNTradingAgent(agent_config)
            
            bridge = MLRLBridge(
                ml_analyzer=mock_ml_analyzer,
                rl_agent=mock_rl_agent,
                tokens=large_token_set
            )
            
            # Measure batch processing time
            start_time = datetime.now()
            results = bridge.predict_and_act(10000.0, {})
            end_time = datetime.now()
            
            # Verify scalability
            execution_time = (end_time - start_time).total_seconds()
            assert execution_time < 2.0  # Should handle 10 tokens in <2 seconds
            assert len(results) == 10


class TestPipelineValidation:
    """Test pipeline validation and health checks"""
    
    def test_pipeline_data_validation(self, sample_discovered_tokens):
        """Test pipeline validates input data properly"""
        # Test with invalid token data
        invalid_token = DiscoveredToken(
            address="",  # Invalid empty address
            symbol="",   # Invalid empty symbol
            name="Test",
            chain=Chain.ETHEREUM,
            discovered_at=datetime.now(),
            discovery_source="testing",
            price_usd=-1.0  # Invalid negative price
        )
        
        # Pipeline should handle invalid data gracefully
        mock_ml_analyzer = MagicMock()
        mock_ml_analyzer.analyze_batch.return_value = []
        
        with patch('src.rl_agent.dqn_agent.DQNNetwork'):
            agent_config = AgentConfig()
            mock_rl_agent = DQNTradingAgent(agent_config)
            
            bridge = MLRLBridge(
                ml_analyzer=mock_ml_analyzer,
                rl_agent=mock_rl_agent,
                tokens=[invalid_token]
            )
            
            # Should not crash with invalid data
            results = bridge.predict_and_act(10000.0, {})
            assert isinstance(results, list)

    def test_pipeline_consistency(self, sample_discovered_tokens, sample_ml_predictions):
        """Test pipeline produces consistent results"""
        mock_ml_analyzer = MagicMock()
        mock_ml_analyzer.analyze_batch.return_value = sample_ml_predictions
        
        with patch('src.rl_agent.dqn_agent.DQNNetwork') as mock_network:
            # Mock deterministic neural network
            mock_model = MagicMock()
            mock_model.forward.return_value = [2.0, 1.0, 0.5, 1.5, 0.8]
            mock_network.return_value = mock_model
            
            agent_config = AgentConfig(epsilon_start=0.0)  # No randomness
            mock_rl_agent = DQNTradingAgent(agent_config)
            
            bridge = MLRLBridge(
                ml_analyzer=mock_ml_analyzer,
                rl_agent=mock_rl_agent,
                tokens=sample_discovered_tokens
            )
            
            # Run multiple times with same inputs
            results1 = bridge.predict_and_act(10000.0, {})
            results2 = bridge.predict_and_act(10000.0, {})
            
            # Results should be identical (due to caching and no randomness)
            assert len(results1) == len(results2)
            for r1, r2 in zip(results1, results2):
                assert r1['rl_action'] == r2['rl_action']
                assert r1['confidence'] == r2['confidence']