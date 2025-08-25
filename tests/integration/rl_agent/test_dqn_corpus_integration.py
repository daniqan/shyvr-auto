"""
DQN Corpus Integration Tests

Tests the integration between DQN training pipeline and GCS corpus data.
Following TDD methodology - these tests should FAIL initially and guide implementation.

Tests actual GCS corpus data integration with DQN training pipeline:
- Real GCS corpus loading (no mocks)
- Corpus feature integration into RL state space
- Historical data simulation using corpus
- End-to-end DQN training with corpus data
"""

import asyncio
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# DQN and RL components
from src.rl_agent.training_pipeline import DQNTrainingPipeline, TrainingConfig
from src.rl_agent.trading_environment import TradingEnvironment, EnvironmentConfig
from src.rl_agent.dqn_agent import DQNTradingAgent
from src.rl_agent.base import AgentConfig, TradeAction, MarketState

# Corpus data components  
from src.data_pipeline.gcs_corpus_loader import GCSCorpusLoader
from src.discovery.base import DiscoveredToken

# Test utilities
from tests.utils.test_data_utils import create_test_token


class TestDQNCorpusIntegration:
    """Test DQN training pipeline integration with GCS corpus data"""
    
    @pytest.fixture(scope="class")
    async def corpus_loader(self):
        """Create GCS corpus loader instance"""
        # Use temporary cache directory for tests
        cache_dir = tempfile.mkdtemp(prefix="test_corpus_cache_")
        loader = GCSCorpusLoader(
            bucket_name="shyvr-models-prod",
            cache_dir=cache_dir,
            cache_ttl_hours=1  # Short TTL for tests
        )
        yield loader
        
        # Cleanup cache directory
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
    
    @pytest.fixture
    async def corpus_data(self, corpus_loader):
        """Load actual corpus data for testing"""
        try:
            # Load daily corpus data
            df = await corpus_loader.load_corpus_from_gcs(
                gcs_prefix=None,  # Use latest version
                timeframe="daily",
                token=None  # Load all tokens
            )
            
            # Verify we have expected corpus structure
            assert len(df) > 0, "Corpus data should not be empty"
            
            # Check for expected columns (features from corpus)
            expected_cols = ['timestamp', 'close', 'volume', 'market_cap']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            assert not missing_cols, f"Missing expected columns: {missing_cols}"
            
            return df
            
        except Exception as e:
            pytest.skip(f"Could not load corpus data from GCS: {e}")
    
    @pytest.fixture
    def test_tokens(self):
        """Create test tokens for DQN training"""
        return [
            create_test_token(
                address="0x1234",
                symbol="BTC",
                name="Bitcoin",
                price_usd=45000.0,
                volume_24h=1000000000,
                market_cap=850000000000
            ),
            create_test_token(
                address="0x5678", 
                symbol="ETH",
                name="Ethereum",
                price_usd=3000.0,
                volume_24h=500000000,
                market_cap=360000000000
            )
        ]
    
    @pytest.fixture
    def training_config(self):
        """Create training configuration for DQN"""
        return TrainingConfig(
            num_episodes=5,  # Small number for tests
            max_steps_per_episode=10,
            learning_rate=0.001,
            batch_size=8,
            save_frequency=5,
            early_stopping_patience=10,
            use_ml_features=True,  # Enable ML features from corpus
            use_database_experiences=False  # Disable for corpus integration tests
        )
    
    async def test_corpus_data_loading(self, corpus_loader):
        """Test basic corpus data loading functionality"""
        # Test listing available versions
        versions = await corpus_loader.list_available_corpus_versions()
        assert len(versions) > 0, "Should find at least one corpus version"
        
        # Test getting latest version
        latest_version = await corpus_loader.get_latest_corpus_version()
        assert latest_version is not None
        assert latest_version in versions
        
        # Test loading corpus data
        df = await corpus_loader.load_corpus_from_gcs(
            timeframe="daily",
            token="BTC"  # Filter for specific token
        )
        
        assert len(df) > 0, "Should load corpus data"
        assert 'timestamp' in df.columns, "Should have timestamp column"
        
        # Verify data quality
        assert not df['timestamp'].isna().any(), "Timestamps should not be null"
        assert df['timestamp'].is_monotonic_increasing, "Timestamps should be ordered"
    
    async def test_dqn_pipeline_corpus_integration(self, corpus_loader, corpus_data, test_tokens, training_config):
        """Test DQN training pipeline integration with corpus data"""
        
        # This test should FAIL initially - it's testing functionality we need to implement
        
        # Create enhanced DQN training pipeline that can load corpus data
        pipeline = DQNTrainingPipeline(training_config, test_tokens, {})
        
        # Test: Pipeline should have corpus loader integration
        assert hasattr(pipeline, 'corpus_loader'), "Pipeline should have corpus_loader attribute"
        assert hasattr(pipeline, 'load_corpus_for_rl'), "Pipeline should have load_corpus_for_rl method"
        
        # Test: Load corpus data for RL training
        rl_corpus_data = await pipeline.load_corpus_for_rl(
            tokens=test_tokens,
            timeframe="daily",
            lookback_days=30
        )
        
        assert isinstance(rl_corpus_data, dict), "Should return dictionary of corpus data"
        assert len(rl_corpus_data) > 0, "Should load corpus data for tokens"
        
        # Test: Corpus data should be properly formatted for RL
        for token_address, token_data in rl_corpus_data.items():
            assert 'prices' in token_data, "Should have price data"
            assert 'features' in token_data, "Should have feature data" 
            assert 'timestamps' in token_data, "Should have timestamp data"
            
            # Verify feature integration
            features = token_data['features']
            assert isinstance(features, np.ndarray), "Features should be numpy array"
            assert features.shape[1] > 10, "Should have multiple feature columns"
    
    async def test_rl_state_from_corpus_features(self, corpus_loader, test_tokens, training_config):
        """Test conversion of corpus features to RL state representation"""
        
        # This test should FAIL initially - testing new functionality
        
        pipeline = DQNTrainingPipeline(training_config, test_tokens, {})
        
        # Load sample corpus data
        corpus_df = await corpus_loader.load_corpus_from_gcs(
            timeframe="daily",
            token="BTC"
        )
        
        # Test: Create RL states from corpus features
        rl_states = await pipeline.create_rl_state_from_corpus(
            corpus_data=corpus_df,
            token_symbol="BTC", 
            lookback_window=10
        )
        
        assert len(rl_states) > 0, "Should create RL states from corpus"
        
        # Test: Each RL state should be a proper MarketState
        for state in rl_states:
            assert isinstance(state, MarketState), "Should create MarketState objects"
            assert state.token is not None, "State should have token"
            assert state.price_usd > 0, "State should have valid price"
            assert state.timestamp is not None, "State should have timestamp"
            
            # Test: Enhanced features from corpus
            state_vector = state.to_vector()
            assert len(state_vector) > 15, "State vector should include corpus features"
            assert not np.isnan(state_vector).any(), "State vector should not contain NaN values"
    
    async def test_trading_environment_corpus_integration(self, corpus_loader, test_tokens, training_config):
        """Test TradingEnvironment integration with corpus historical data"""
        
        # This test should FAIL initially - testing enhanced environment
        
        # Load corpus data for environment
        corpus_data = {}
        for token in test_tokens:
            df = await corpus_loader.load_corpus_from_gcs(
                timeframe="daily",
                token=token.symbol
            )
            
            if len(df) > 0:
                corpus_data[token.address] = {
                    'prices': df['close'].tolist(),
                    'volumes': df['volume'].tolist() if 'volume' in df.columns else [1000000] * len(df),
                    'timestamps': df['timestamp'].tolist(),
                    'features': df.select_dtypes(include=[np.number]).values  # All numeric features
                }
        
        # Create environment with corpus data
        env_config = EnvironmentConfig(max_episode_steps=20)
        environment = TradingEnvironment(env_config, test_tokens, corpus_data)
        
        # Test: Environment should use corpus data for simulation
        assert hasattr(environment, 'corpus_features'), "Environment should have corpus_features"
        assert hasattr(environment, 'get_enhanced_state'), "Environment should have get_enhanced_state method"
        
        # Test: Reset and get initial state
        initial_state = environment.reset()
        assert isinstance(initial_state, MarketState)
        
        # Test: Enhanced state should include corpus features
        enhanced_state = environment.get_enhanced_state(initial_state)
        assert hasattr(enhanced_state, 'corpus_features'), "Should have corpus features"
        assert len(enhanced_state.corpus_features) > 0, "Should include corpus feature values"
        
        # Test: Step through environment with corpus data
        action = TradeAction.BUY
        token = test_tokens[0]
        next_state, reward, done, info = environment.step(action, token, 0.1)
        
        assert isinstance(next_state, MarketState)
        assert 'corpus_features_used' in info, "Should indicate corpus features were used"
    
    async def test_end_to_end_dqn_corpus_training(self, corpus_loader, test_tokens, training_config):
        """Test complete DQN training pipeline with corpus data integration"""
        
        # This is the main integration test - should FAIL initially
        
        # Load corpus data for all test tokens
        corpus_data = {}
        for token in test_tokens:
            try:
                df = await corpus_loader.load_corpus_from_gcs(
                    timeframe="daily", 
                    token=token.symbol
                )
                if len(df) > 0:
                    corpus_data[token.address] = {
                        'prices': df['close'].tolist()[:50],  # Limit for test
                        'volumes': df['volume'].tolist()[:50] if 'volume' in df.columns else [1000000] * 50,
                        'timestamps': df['timestamp'].tolist()[:50],
                        'market_caps': df['market_cap'].tolist()[:50] if 'market_cap' in df.columns else [1000000000] * 50
                    }
            except Exception as e:
                # Use synthetic data if corpus not available for token
                corpus_data[token.address] = {
                    'prices': [float(token.price_usd * (1 + np.random.normal(0, 0.02))) for _ in range(50)],
                    'volumes': [float(token.volume_24h * (1 + np.random.normal(0, 0.1))) for _ in range(50)],
                    'timestamps': [(datetime.now() - timedelta(days=50-i)).isoformat() for i in range(50)],
                    'market_caps': [float(token.market_cap * (1 + np.random.normal(0, 0.02))) for _ in range(50)]
                }
        
        # Create DQN pipeline with corpus integration  
        pipeline = DQNTrainingPipeline(training_config, test_tokens, corpus_data)
        
        # Test: Pipeline initialization with corpus
        assert pipeline.corpus_loader is not None, "Pipeline should initialize corpus loader"
        assert hasattr(pipeline.environment, 'corpus_features'), "Environment should have corpus features"
        
        # Test: Agent should have enhanced input size for corpus features
        expected_enhanced_size = MarketState.get_feature_size() + 20  # Base + corpus features
        actual_input_size = pipeline.agent.input_size
        assert actual_input_size >= expected_enhanced_size, f"Agent input size {actual_input_size} should include corpus features (expected >= {expected_enhanced_size})"
        
        # Test: Run training episode with corpus data
        episode_result = pipeline.run_episode(0)
        
        assert 'total_reward' in episode_result
        assert 'corpus_features_used' in episode_result, "Episode should indicate corpus features were used"
        assert episode_result['num_steps'] > 0
        
        # Test: Complete training with corpus
        training_result = pipeline.train()
        
        assert training_result['episodes_completed'] > 0
        assert 'final_metrics' in training_result
        assert training_result['final_metrics']['mean_reward'] is not None
        
        # Test: Verify corpus features contributed to training
        assert training_result.get('corpus_integration_successful', False), "Training should confirm corpus integration"
    
    async def test_corpus_feature_quality(self, corpus_data):
        """Test quality and completeness of corpus features for RL"""
        
        # Verify corpus has sufficient data quality for RL training
        assert len(corpus_data) >= 100, "Should have sufficient historical data points"
        
        # Check for required OHLCV data
        required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        available_ohlcv = [col for col in required_ohlcv if col in corpus_data.columns]
        assert len(available_ohlcv) >= 3, f"Should have at least 3 OHLCV columns, found: {available_ohlcv}"
        
        # Check for additional features that enhance RL state
        feature_columns = [col for col in corpus_data.columns if col not in ['timestamp', 'symbol']]
        assert len(feature_columns) >= 10, f"Should have at least 10 feature columns for RL, found {len(feature_columns)}"
        
        # Verify numeric data quality
        numeric_columns = corpus_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            non_null_pct = corpus_data[col].notna().mean()
            assert non_null_pct >= 0.8, f"Column {col} should have at least 80% non-null values, has {non_null_pct:.1%}"
            
            if col in ['close', 'volume', 'market_cap']:
                assert (corpus_data[col] > 0).all(), f"Column {col} should have all positive values"
    
    async def test_corpus_to_market_state_conversion(self, corpus_loader, test_tokens):
        """Test conversion from corpus data to MarketState objects"""
        
        # This test should FAIL initially - testing new conversion functionality
        
        # Load corpus data for a specific token
        corpus_df = await corpus_loader.load_corpus_from_gcs(
            timeframe="daily",
            token="BTC"
        )
        
        if len(corpus_df) == 0:
            pytest.skip("No corpus data available for BTC")
        
        # Test: Convert corpus row to MarketState
        from src.rl_agent.training_pipeline import CorpusToMarketStateConverter
        converter = CorpusToMarketStateConverter()
        
        sample_row = corpus_df.iloc[0]
        token = test_tokens[0]  # BTC token
        
        market_state = converter.convert_corpus_to_market_state(
            corpus_row=sample_row,
            token=token
        )
        
        # Verify MarketState properties
        assert isinstance(market_state, MarketState)
        assert market_state.token == token
        assert market_state.price_usd > 0
        assert market_state.timestamp is not None
        
        # Test: Enhanced features from corpus
        if hasattr(market_state, 'corpus_features'):
            assert len(market_state.corpus_features) > 0
            assert not np.isnan(market_state.corpus_features).any()
        
        # Test: State vector includes corpus features
        state_vector = market_state.to_vector()
        base_feature_size = 15  # Approximate base MarketState feature size
        assert len(state_vector) > base_feature_size, "State vector should include additional corpus features"


class TestCorpusIntegrationError:
    """Test error handling and edge cases for corpus integration"""
    
    async def test_missing_corpus_data_handling(self, test_tokens, training_config):
        """Test DQN pipeline behavior when corpus data is unavailable"""
        
        # Test with empty corpus data
        pipeline = DQNTrainingPipeline(training_config, test_tokens, {})
        
        # Should handle gracefully and not crash
        try:
            episode_result = pipeline.run_episode(0)
            # Should succeed with fallback to synthetic data
            assert 'total_reward' in episode_result
        except Exception as e:
            # Should have informative error message
            assert "corpus" in str(e).lower() or "data" in str(e).lower()
    
    async def test_malformed_corpus_data_handling(self, test_tokens, training_config):
        """Test handling of malformed or incomplete corpus data"""
        
        # Create malformed corpus data
        malformed_data = {
            test_tokens[0].address: {
                'prices': [1, 2, 3],  # Too few data points
                'volumes': [100, 200],  # Mismatched length
                'timestamps': ['invalid-timestamp'] * 3
            }
        }
        
        pipeline = DQNTrainingPipeline(training_config, test_tokens, malformed_data)
        
        # Should handle malformed data gracefully
        try:
            episode_result = pipeline.run_episode(0)
            # Should either succeed with data cleaning or fail with clear error
            assert 'total_reward' in episode_result
        except ValueError as e:
            assert "data" in str(e).lower()  # Should have clear error about data issues


@pytest.mark.asyncio
async def test_run_all_corpus_integration_tests():
    """Run all corpus integration tests in sequence"""
    
    # This is a meta-test to run the full integration suite
    # Useful for CI/CD and comprehensive testing
    
    test_instance = TestDQNCorpusIntegration()
    
    # Initialize fixtures
    corpus_loader = await test_instance.corpus_loader()
    
    try:
        corpus_data = await test_instance.corpus_data(corpus_loader)
        test_tokens = test_instance.test_tokens()
        training_config = test_instance.training_config()
        
        # Run core integration tests
        await test_instance.test_corpus_data_loading(corpus_loader)
        await test_instance.test_dqn_pipeline_corpus_integration(corpus_loader, corpus_data, test_tokens, training_config)
        await test_instance.test_end_to_end_dqn_corpus_training(corpus_loader, test_tokens, training_config)
        
        print("✅ All DQN corpus integration tests completed")
        
    except Exception as e:
        print(f"❌ DQN corpus integration tests failed: {e}")
        raise
    
    finally:
        # Cleanup
        if hasattr(corpus_loader, 'clear_old_cache'):
            await corpus_loader.clear_old_cache(ttl_hours=0)  # Clear all cache


if __name__ == "__main__":
    """Run integration tests directly"""
    asyncio.run(test_run_all_corpus_integration_tests())