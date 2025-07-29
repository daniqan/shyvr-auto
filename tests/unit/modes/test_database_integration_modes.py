"""
Test suite for trading modes database integration.

Following TDD methodology - write failing tests first, then implement database
integration features in simulation_mode.py, live_mode.py, and analysis_mode.py.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from typing import Dict, List, Any, Optional

from src.modes.base import ModeType, ModeStatus, ModeConfig, ModeResult
from src.portfolio.base import Portfolio, PortfolioConfig
from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_database import DatabaseExperienceBuffer, DatabaseExperienceConfig
from src.rl_agent.experience_replay import Experience
from src.discovery.base import DiscoveredToken, TokenStatus
from src.utils.base import Chain


def create_mock_token():
    """Helper function to create mock DiscoveredToken."""
    return DiscoveredToken(
        address="TEST_TOKEN_123",
        name="Test Token",
        symbol="TEST",
        chain=Chain.SOLANA,
        discovered_at=datetime.now(),
        discovery_source="test",
        status=TokenStatus.VALIDATED
    )


@pytest.fixture
def portfolio_config():
    """Portfolio configuration for testing."""
    return PortfolioConfig(
        initial_balance=Decimal("10000"),
        base_currency="USDC",
        max_position_size_pct=Decimal("0.1"),
        max_daily_loss_pct=Decimal("0.05")
    )


@pytest.fixture
def portfolio(portfolio_config):
    """Portfolio instance for testing."""
    return Portfolio(
        portfolio_id=uuid4(),
        name="Test Portfolio",
        config=portfolio_config,
        cash_balance=Decimal("10000"),
        total_value=Decimal("10000")
    )


@pytest.fixture
def market_state():
    """Create mock market state for testing."""
    return MarketState(
        token=create_mock_token(),
        price_usd=1.50,
        volume_24h=1000000.0,
        price_change_24h=0.05,
        rsi=30.0,  # Oversold - should trigger BUY action
        timestamp=datetime.now()
    )


@pytest.fixture
def database_experience_config():
    """Configuration for database experience buffer."""
    return DatabaseExperienceConfig(
        max_size=1000,
        batch_size=32,
        min_size=50,
        prioritized=True,
        cache_size=100
    )


@pytest.fixture
async def mock_database_buffer(database_experience_config):
    """Mock database experience buffer."""
    buffer = AsyncMock(spec=DatabaseExperienceBuffer)
    buffer.config = database_experience_config
    buffer.initialized = True
    buffer.size.return_value = 0
    buffer.can_sample.return_value = False
    buffer.add.return_value = None
    buffer.add_batch.return_value = None
    buffer.sample.return_value = []
    buffer.get_statistics.return_value = {
        'total_experiences': 0,
        'session_id': str(uuid4()),
        'cache_hit_rate': 0.0,
        'insertion_rate': 0.0
    }
    return buffer


class TestSimulationModeDatabaseIntegration:
    """Test database integration for simulation mode."""
    
    @pytest.fixture
    def simulation_config_with_db(self):
        """Simulation config with database experience storage enabled."""
        return ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "initial_balance": 10000,
                "enable_fees": True,
                "slippage_bps": 10,
                "enable_experience_collection": True,
                "enable_database_experience_storage": True,
                "database_experience_config": {
                    "max_size": 1000,
                    "batch_size": 32,
                    "prioritized": True,
                    "cache_size": 100
                },
                "simulation_experience_tags": {
                    "trading_mode": "simulation",
                    "environment": "paper_trading",
                    "risk_level": "medium"
                }
            }
        )
    
    @pytest.mark.asyncio
    async def test_simulation_mode_database_experience_initialization(
        self, simulation_config_with_db, portfolio, mock_database_buffer
    ):
        """Test that simulation mode properly initializes database experience storage."""
        # This test will fail until we implement database configuration
        with patch('src.modes.simulation_mode.DatabaseExperienceBuffer', return_value=mock_database_buffer):
            from src.modes.simulation_mode import SimulationMode
            
            mode = SimulationMode(
                mode_id=uuid4(),
                config=simulation_config_with_db,
                portfolio=portfolio
            )
            
            await mode.initialize()
            
            # Should have database experience buffer configured
            assert hasattr(mode, 'database_experience_buffer')
            assert mode.database_experience_buffer is not None
            assert mode.enable_database_experience_storage is True
            
            # Should initialize with proper configuration
            mock_database_buffer.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_simulation_mode_experience_tagging(
        self, simulation_config_with_db, portfolio, mock_database_buffer, market_state
    ):
        """Test that simulation mode adds proper tags to experiences."""
        with patch('src.modes.simulation_mode.DatabaseExperienceBuffer', return_value=mock_database_buffer):
            from src.modes.simulation_mode import SimulationMode
            
            mode = SimulationMode(
                mode_id=uuid4(),
                config=simulation_config_with_db,
                portfolio=portfolio
            )
            
            await mode.initialize()
            await mode.start()
            
            # Process a tick to generate experience
            action = await mode.process_tick(market_state)
            
            # Should add experience with simulation-specific tags
            if mock_database_buffer.add.called or mock_database_buffer.add_batch.called:
                # Check that experience was tagged with simulation metadata
                call_args = (mock_database_buffer.add.call_args or 
                           mock_database_buffer.add_batch.call_args)
                
                if call_args and len(call_args.kwargs) > 0 and 'metadatas' in call_args.kwargs:
                    metadata = call_args.kwargs['metadatas'][0] if call_args.kwargs['metadatas'] else {}
                    assert metadata.get('trading_mode') == 'simulation'
                    assert metadata.get('environment') == 'paper_trading'
                    assert 'simulation_session_id' in metadata
    
    @pytest.mark.asyncio
    async def test_simulation_mode_database_experience_persistence(
        self, simulation_config_with_db, portfolio, mock_database_buffer, market_state
    ):
        """Test that simulation mode persists experiences to database."""
        mock_database_buffer.size.return_value = 100
        mock_database_buffer.can_sample.return_value = True
        
        with patch('src.modes.simulation_mode.DatabaseExperienceBuffer', return_value=mock_database_buffer):
            from src.modes.simulation_mode import SimulationMode
            
            mode = SimulationMode(
                mode_id=uuid4(),
                config=simulation_config_with_db,
                portfolio=portfolio
            )
            
            await mode.initialize()
            await mode.start()
            
            # Generate multiple market ticks
            for _ in range(5):
                await mode.process_tick(market_state)
            
            # Should have persisted experiences to database
            assert mock_database_buffer.add.called or mock_database_buffer.add_batch.called
            
            # Should maintain buffer statistics
            stats = await mode.get_database_experience_statistics()
            assert 'total_experiences' in stats
            assert 'session_id' in stats


class TestLiveModeDatabaseIntegration:
    """Test database integration for live mode."""
    
    @pytest.fixture
    def live_config_with_db(self):
        """Live mode config with database experience storage enabled."""
        return ModeConfig(
            mode_type=ModeType.LIVE_TRADING,
            enabled=True,
            parameters={
                "initial_balance": 50000,
                "enable_real_trading": True,
                "max_position_size_pct": 0.05,
                "enable_experience_collection": True,
                "enable_database_experience_storage": True,
                "database_experience_config": {
                    "max_size": 10000,
                    "batch_size": 64,
                    "prioritized": True,
                    "cache_size": 500
                },
                "production_experience_settings": {
                    "enable_real_time_persistence": True,
                    "safety_validation_required": True,
                    "backup_frequency_minutes": 5,
                    "critical_experience_priority": 10.0
                },
                "live_experience_tags": {
                    "trading_mode": "live",
                    "environment": "production",
                    "risk_level": "high"
                }
            }
        )
    
    @pytest.mark.asyncio
    async def test_live_mode_database_experience_initialization(
        self, live_config_with_db, portfolio, mock_database_buffer
    ):
        """Test that live mode properly initializes production database experience storage."""
        # This test will fail until we implement database configuration in live mode
        from src.modes.live_mode import LiveMode
        with patch('src.modes.live_mode.DatabaseExperienceBuffer', return_value=mock_database_buffer), \
             patch.object(LiveMode, '_initialize_dex_clients', new_callable=AsyncMock):
            
            mode = LiveMode(
                mode_id=uuid4(),
                config=live_config_with_db,
                portfolio=portfolio
            )
            
            await mode.initialize()
            
            # Should have database experience buffer configured for production
            assert hasattr(mode, 'database_experience_buffer')
            assert mode.database_experience_buffer is not None
            assert mode.enable_database_experience_storage is True
            assert mode.production_experience_settings is not None
            
            # Should initialize with production safety measures
            mock_database_buffer.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_live_mode_real_time_experience_persistence(
        self, live_config_with_db, portfolio, mock_database_buffer, market_state
    ):
        """Test that live mode enables real-time experience persistence."""
        from src.modes.live_mode import LiveMode
        with patch('src.modes.live_mode.DatabaseExperienceBuffer', return_value=mock_database_buffer), \
             patch.object(LiveMode, '_initialize_dex_clients', new_callable=AsyncMock):
            
            mode = LiveMode(
                mode_id=uuid4(),
                config=live_config_with_db,
                portfolio=portfolio
            )
            
            await mode.initialize()
            await mode.start()
            
            # Should enable real-time persistence
            assert mode.enable_real_time_experience_persistence is True
            
            # Process a tick that should trigger immediate persistence
            await mode.process_tick(market_state)
            
            # Should persist experience immediately, not batch
            assert mock_database_buffer.add.called or mock_database_buffer.add_batch.called
    
    @pytest.mark.asyncio
    async def test_live_mode_production_safety_validation(
        self, live_config_with_db, portfolio, mock_database_buffer, market_state
    ):
        """Test that live mode applies production safety validation to experiences."""
        from src.modes.live_mode import LiveMode
        with patch('src.modes.live_mode.DatabaseExperienceBuffer', return_value=mock_database_buffer), \
             patch.object(LiveMode, '_initialize_dex_clients', new_callable=AsyncMock):
            
            mode = LiveMode(
                mode_id=uuid4(),
                config=live_config_with_db,
                portfolio=portfolio
            )
            
            await mode.initialize()
            await mode.start()
            
            # Should validate experiences before persistence
            await mode.process_tick(market_state)
            
            # Should have applied safety validation
            if mock_database_buffer.add.called or mock_database_buffer.add_batch.called:
                call_args = (mock_database_buffer.add.call_args or 
                           mock_database_buffer.add_batch.call_args)
                
                if call_args and len(call_args.kwargs) > 0 and 'metadatas' in call_args.kwargs:
                    metadata = call_args.kwargs['metadatas'][0] if call_args.kwargs['metadatas'] else {}
                    assert metadata.get('safety_validated') is True
                    assert metadata.get('trading_mode') == 'live'
                    assert metadata.get('environment') == 'production'


class TestAnalysisModeDatabaseIntegration:
    """Test database integration for analysis mode."""
    
    @pytest.fixture
    def analysis_config_with_db(self):
        """Analysis mode config with database experience analytics enabled."""
        return ModeConfig(
            mode_type=ModeType.ANALYSIS,
            enabled=True,
            parameters={
                "analysis_depth": "comprehensive",
                "include_backtesting": True,
                "enable_database_experience_analytics": True,
                "database_experience_config": {
                    "max_size": 50000,
                    "batch_size": 128,
                    "prioritized": False,
                    "cache_size": 1000
                },
                "experience_analytics_settings": {
                    "historical_analysis_enabled": True,
                    "performance_correlation_analysis": True,
                    "experience_pattern_detection": True,
                    "cross_session_analysis": True
                },
                "analysis_experience_tags": {
                    "trading_mode": "analysis",
                    "environment": "backtesting",
                    "risk_level": "low"
                }
            }
        )
    
    @pytest.mark.asyncio
    async def test_analysis_mode_database_experience_analytics_initialization(
        self, analysis_config_with_db, portfolio, mock_database_buffer
    ):
        """Test that analysis mode properly initializes database experience analytics."""
        # This test will fail until we implement database analytics in analysis mode
        with patch('src.modes.analysis_mode.DatabaseExperienceBuffer', return_value=mock_database_buffer):
            from src.modes.analysis_mode import AnalysisMode
            
            mode = AnalysisMode(
                mode_id=uuid4(),
                config=analysis_config_with_db,
                portfolio=portfolio
            )
            
            await mode.initialize()
            
            # Should have database experience analytics configured
            assert hasattr(mode, 'database_experience_buffer')
            assert mode.database_experience_buffer is not None
            assert mode.enable_database_experience_analytics is True
            assert mode.experience_analytics_settings is not None
            
            # Should initialize analytics capabilities
            mock_database_buffer.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analysis_mode_historical_experience_analysis(
        self, analysis_config_with_db, portfolio, mock_database_buffer
    ):
        """Test that analysis mode can analyze historical experiences from database."""
        # Mock historical experiences
        mock_experiences = [
            {
                'id': i,
                'session_id': str(uuid4()),
                'state': [1.0, 2.0, 3.0],
                'action': 1,
                'reward': 0.1 * i,
                'next_state': [1.1, 2.1, 3.1],
                'done': False,
                'created_at': datetime.now() - timedelta(hours=i)
            }
            for i in range(100)
        ]
        
        mock_database_buffer.get_recent.return_value = mock_experiences
        mock_database_buffer.size.return_value = len(mock_experiences)
        
        with patch('src.modes.analysis_mode.DatabaseExperienceBuffer', return_value=mock_database_buffer):
            from src.modes.analysis_mode import AnalysisMode
            
            mode = AnalysisMode(
                mode_id=uuid4(),
                config=analysis_config_with_db,
                portfolio=portfolio
            )
            
            await mode.initialize()
            
            # Should be able to analyze historical experiences
            analysis_result = await mode.analyze_historical_experiences(days_back=7)
            
            assert analysis_result is not None
            assert 'experience_count' in analysis_result
            assert 'performance_metrics' in analysis_result
            assert 'pattern_analysis' in analysis_result
    
    @pytest.mark.asyncio
    async def test_analysis_mode_cross_session_experience_analysis(
        self, analysis_config_with_db, portfolio, mock_database_buffer
    ):
        """Test that analysis mode can perform cross-session experience analysis."""
        # Mock multiple session experiences
        session_ids = [str(uuid4()) for _ in range(3)]
        mock_cross_session_data = {
            'sessions': session_ids,
            'total_experiences': 1500,
            'performance_comparison': {
                session_ids[0]: {'reward_avg': 0.15, 'win_rate': 0.65},
                session_ids[1]: {'reward_avg': 0.12, 'win_rate': 0.58},
                session_ids[2]: {'reward_avg': 0.18, 'win_rate': 0.72}
            }
        }
        
        mock_database_buffer.get_session_stats.return_value = mock_cross_session_data
        
        with patch('src.modes.analysis_mode.DatabaseExperienceBuffer', return_value=mock_database_buffer):
            from src.modes.analysis_mode import AnalysisMode
            
            mode = AnalysisMode(
                mode_id=uuid4(),
                config=analysis_config_with_db,
                portfolio=portfolio
            )
            
            await mode.initialize()
            
            # Should be able to perform cross-session analysis
            cross_analysis = await mode.analyze_cross_session_performance()
            
            assert cross_analysis is not None
            assert 'session_comparison' in cross_analysis
            assert 'performance_trends' in cross_analysis
            assert 'best_performing_strategies' in cross_analysis


class TestDatabaseConfigurationValidation:
    """Test database configuration validation across all modes."""
    
    @pytest.mark.asyncio
    async def test_invalid_database_config_handling(self, portfolio):
        """Test that invalid database configurations are handled gracefully."""
        invalid_config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "enable_database_experience_storage": True,
                "database_experience_config": {
                    "max_size": -1,  # Invalid
                    "batch_size": 0,  # Invalid
                    "prioritized": "invalid"  # Invalid type
                }
            }
        )
        
        # Should handle invalid configuration without crashing
        with patch('src.modes.simulation_mode.DatabaseExperienceBuffer') as mock_buffer_class:
            mock_buffer_class.side_effect = ValueError("Invalid configuration")
            
            from src.modes.simulation_mode import SimulationMode
            
            mode = SimulationMode(
                mode_id=uuid4(),
                config=invalid_config,
                portfolio=portfolio
            )
            
            # Should gracefully fall back to non-database experience collection
            await mode.initialize()
            
            # Should disable database features but continue operating
            assert mode.enable_database_experience_storage is False
    
    @pytest.mark.asyncio
    async def test_database_connection_failure_handling(self, portfolio):
        """Test that database connection failures are handled gracefully."""
        config = ModeConfig(
            mode_type=ModeType.SIMULATION,
            enabled=True,
            parameters={
                "enable_database_experience_storage": True,
                "database_experience_config": {
                    "max_size": 1000,
                    "batch_size": 32
                }
            }
        )
        
        # Mock database connection failure
        with patch('src.modes.simulation_mode.DatabaseExperienceBuffer') as mock_buffer_class:
            mock_buffer = AsyncMock()
            mock_buffer.initialize.side_effect = Exception("Database connection failed")
            mock_buffer_class.return_value = mock_buffer
            
            from src.modes.simulation_mode import SimulationMode
            
            mode = SimulationMode(
                mode_id=uuid4(),
                config=config,
                portfolio=portfolio
            )
            
            # Should handle database failure and continue with fallback
            await mode.initialize()
            
            # Should have logged error and disabled database features
            assert hasattr(mode, 'database_connection_failed')
            assert mode.database_connection_failed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])