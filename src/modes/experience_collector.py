"""
Real Experience Collection Pipeline

Captures real trading experiences from all modes (simulation, live trading)
and converts them to RL training experiences. This is the critical component
that allows the RL agent to learn from actual trading results.

Following TDD methodology - implementation after comprehensive tests.
"""

import asyncio
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
import numpy as np
import structlog

from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import Experience, ExperienceReplayBuffer
from src.ml_analysis.base import PredictionResult
from src.utils.database import (
    insert_experience_batch, query_experiences_by_session, 
    sample_prioritized_experiences, update_experience_priorities,
    get_session_statistics, check_rl_database_health, 
    stream_experiences_with_memory_optimization, execute_concurrent_batch_operations,
    atomic_experience_batch_operation, get_database_connection
)


logger = structlog.get_logger()


@dataclass
class ExperienceCollectorConfig:
    """Configuration for trading experience collector"""
    
    # Buffer management
    buffer_size: int = 10000
    min_experience_gap_seconds: int = 1  # Minimum time between experiences
    reward_calculation_window: int = 5   # Look-ahead window for reward calculation
    
    # Persistence
    enable_persistence: bool = True
    persistence_path: str = "experiences.json"
    
    # Database storage options
    enable_database_storage: bool = False  # Enable database storage instead of file
    database_batch_size: int = 50  # Batch size for database operations
    database_connection_timeout: int = 30  # Connection timeout in seconds
    database_retry_attempts: int = 3  # Number of retry attempts for database operations
    database_retry_delay: float = 1.0  # Initial retry delay in seconds
    
    # Session limits
    max_experiences_per_session: int = 1000
    experience_timeout_hours: int = 24  # Timeout for incomplete experiences
    
    # Reward weights
    pnl_weight: float = 0.4
    execution_quality_weight: float = 0.2
    risk_weight: float = 0.2
    market_timing_weight: float = 0.2


@dataclass
class TradingExperienceData:
    """Complete trading experience data for RL training"""
    
    experience_id: str
    timestamp: datetime
    
    # Trading action and states
    action: TradeAction
    pre_trade_state: np.ndarray  # Market state before trade
    post_trade_state: np.ndarray  # Market state after trade
    
    # RL training data
    reward: float
    done: bool = False
    
    # Trade execution details
    success: bool = True
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    execution_time_ms: float = 0.0
    
    # Reward components
    pnl_reward: float = 0.0
    execution_quality_penalty: float = 0.0
    risk_penalty: float = 0.0
    market_timing_reward: float = 0.0
    
    # Portfolio context
    portfolio_value_before: float = 0.0
    portfolio_value_after: float = 0.0
    position_size_change: float = 0.0
    
    # Market context
    market_volatility: float = 0.0
    market_direction: float = 0.0  # Price change during experience
    
    # Quality metrics
    confidence: float = 0.0  # ML prediction confidence if available
    total_fees: float = 0.0
    slippage_penalty: float = 0.0
    
    # Database-specific fields
    database_session_id: Optional[str] = None  # Database session ID
    checksum: Optional[str] = None  # Data integrity checksum
    version: int = 1  # Experience data version
    storage_metadata: Optional[Dict[str, Any]] = None  # Database storage metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience data to dictionary for serialization"""
        return {
            'experience_id': self.experience_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action.value,
            'pre_trade_state': self.pre_trade_state.tolist(),
            'post_trade_state': self.post_trade_state.tolist(),
            'reward': self.reward,
            'done': self.done,
            'success': self.success,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'fees': self.fees,
            'slippage': self.slippage,
            'execution_time_ms': self.execution_time_ms,
            'pnl_reward': self.pnl_reward,
            'execution_quality_penalty': self.execution_quality_penalty,
            'risk_penalty': self.risk_penalty,
            'market_timing_reward': self.market_timing_reward,
            'portfolio_value_before': self.portfolio_value_before,
            'portfolio_value_after': self.portfolio_value_after,
            'position_size_change': self.position_size_change,
            'market_volatility': self.market_volatility,
            'market_direction': self.market_direction,
            'confidence': self.confidence,
            'total_fees': self.total_fees,
            'slippage_penalty': self.slippage_penalty,
            'database_session_id': self.database_session_id,
            'checksum': self.checksum,
            'version': self.version,
            'storage_metadata': self.storage_metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingExperienceData':
        """Create experience data from dictionary"""
        return cls(
            experience_id=data['experience_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            action=TradeAction(data['action']),
            pre_trade_state=np.array(data['pre_trade_state'], dtype=np.float32),
            post_trade_state=np.array(data['post_trade_state'], dtype=np.float32),
            reward=data['reward'],
            done=data['done'],
            success=data['success'],
            realized_pnl=data['realized_pnl'],
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
            fees=data['fees'],
            slippage=data['slippage'],
            execution_time_ms=data.get('execution_time_ms', 0.0),
            pnl_reward=data.get('pnl_reward', 0.0),
            execution_quality_penalty=data.get('execution_quality_penalty', 0.0),
            risk_penalty=data.get('risk_penalty', 0.0),
            market_timing_reward=data.get('market_timing_reward', 0.0),
            portfolio_value_before=data.get('portfolio_value_before', 0.0),
            portfolio_value_after=data.get('portfolio_value_after', 0.0),
            position_size_change=data.get('position_size_change', 0.0),
            market_volatility=data.get('market_volatility', 0.0),
            market_direction=data.get('market_direction', 0.0),
            confidence=data.get('confidence', 0.0),
            total_fees=data.get('total_fees', 0.0),
            slippage_penalty=data.get('slippage_penalty', 0.0),
            database_session_id=data.get('database_session_id'),
            checksum=data.get('checksum'),
            version=data.get('version', 1),
            storage_metadata=data.get('storage_metadata')
        )


class TradingExperienceCollector:
    """
    Real Experience Collection Pipeline
    
    Captures trading experiences from all modes and converts them to RL training data.
    This is the critical component that enables the RL agent to learn from actual
    trading results rather than just simulated data.
    """
    
    def __init__(self, config: ExperienceCollectorConfig, replay_buffer: ExperienceReplayBuffer):
        self.config = config
        self.replay_buffer = replay_buffer
        self.logger = structlog.get_logger().bind(component="TradingExperienceCollector")
        
        # Experience tracking
        self.active_experiences: Dict[str, Dict[str, Any]] = {}
        self.completed_experiences: List[TradingExperienceData] = []
        
        # Collection state
        self.is_collecting = False
        self.session_start_time: Optional[datetime] = None
        self.last_experience_time: Optional[datetime] = None
        self.experiences_this_session = 0
        
        # Statistics
        self.total_experiences_captured = 0
        self.total_experiences_added_to_buffer = 0
        self.session_statistics = {}
        
        # Database-specific attributes
        self.database_session_id = str(uuid.uuid4()) if config.enable_database_storage else None
        self.database_connection = None
        self.pending_database_operations = []
        
        self.logger.info("Trading Experience Collector initialized",
                        buffer_size=config.buffer_size,
                        enable_persistence=config.enable_persistence,
                        enable_database_storage=config.enable_database_storage,
                        max_per_session=config.max_experiences_per_session)
    
    async def start_collection(self) -> None:
        """Start collecting trading experiences"""
        if self.is_collecting:
            self.logger.warning("Experience collection already started")
            return
        
        self.is_collecting = True
        self.session_start_time = datetime.now()
        self.experiences_this_session = 0
        self.last_experience_time = None
        
        # Initialize database connection if enabled
        if self.config.enable_database_storage:
            try:
                # Test database connectivity
                async with get_database_connection() as conn:
                    await conn.fetchval("SELECT 1")
                self.logger.info("Database connection initialized for experience storage")
            except Exception as e:
                self.logger.error("Failed to initialize database connection", error=str(e))
                raise ExperienceCollectionError(f"Database connection failed: {e}")
        
        # Load previous experiences if persistence enabled
        if self.config.enable_persistence:
            try:
                if self.config.enable_database_storage:
                    # Load from database if available
                    loaded_count = await self.load_experiences_from_database(self.database_session_id)
                else:
                    # Load from file
                    loaded_count = await self.load_experiences()
                self.logger.info("Loaded previous experiences", count=loaded_count)
            except Exception as e:
                self.logger.warning("Could not load previous experiences", error=str(e))
        
        self.logger.info("Experience collection started")
    
    async def stop_collection(self) -> None:
        """Stop collecting experiences and finalize session"""
        if not self.is_collecting:
            self.logger.warning("Experience collection not started")
            return
        
        self.is_collecting = False
        
        # Add any remaining completed experiences to buffer
        if self.completed_experiences:
            await self.add_experiences_to_buffer()
        
        # Persist experiences if enabled
        if self.config.enable_persistence:
            try:
                if self.config.enable_database_storage:
                    await self.persist_experiences_to_database()
                    self.logger.info("Experiences persisted to database")
                else:
                    await self.persist_experiences()
                    self.logger.info("Experiences persisted to file")
            except Exception as e:
                self.logger.error("Failed to persist experiences", error=str(e))
        
        # Clean up any timed-out active experiences
        await self.cleanup_timed_out_experiences()
        
        session_duration = datetime.now() - self.session_start_time
        self.logger.info("Experience collection stopped",
                        session_duration=str(session_duration),
                        experiences_captured=self.experiences_this_session,
                        active_remaining=len(self.active_experiences))
    
    async def capture_pre_trade_state(self, market_state: MarketState, action: TradeAction) -> Optional[str]:
        """
        Capture the market state before executing a trade
        
        Args:
            market_state: Current market state
            action: Intended trading action
            
        Returns:
            Experience ID for tracking, or None if rejected
        """
        if not self.is_collecting:
            raise ExperienceCollectionError("Experience collection not started")
        
        # Check session limits
        if self.experiences_this_session >= self.config.max_experiences_per_session:
            self.logger.warning("Session experience limit reached", 
                              limit=self.config.max_experiences_per_session)
            return None
        
        # Check minimum time gap
        now = datetime.now()
        if (self.last_experience_time and 
            (now - self.last_experience_time).total_seconds() < self.config.min_experience_gap_seconds):
            self.logger.debug("Experience rejected due to minimum time gap")
            return None
        
        # Generate unique experience ID
        experience_id = str(uuid.uuid4())
        
        # Store initial experience data
        self.active_experiences[experience_id] = {
            'id': experience_id,
            'timestamp': now,
            'action': action,
            'pre_trade_state': market_state.to_vector(),
            'market_state': market_state
        }
        
        self.last_experience_time = now
        self.experiences_this_session += 1
        
        self.logger.debug("Pre-trade state captured",
                         experience_id=experience_id,
                         action=action.value,
                         token=market_state.token.symbol)
        
        return experience_id
    
    async def capture_post_trade_result(self, experience_id: str, trading_result: TradingResult, 
                                        post_market_state: MarketState) -> None:
        """
        Capture the result of a trade execution and complete the experience
        
        Args:
            experience_id: ID from capture_pre_trade_state
            trading_result: Result of trade execution
            post_market_state: Market state after trade
        """
        if not self.is_collecting:
            raise ExperienceCollectionError("Experience collection not started")
        
        if experience_id not in self.active_experiences:
            raise ExperienceCollectionError(f"Experience ID {experience_id} not found")
        
        active_exp = self.active_experiences[experience_id]
        
        # Calculate reward from trading result
        reward = await self._calculate_reward(active_exp, trading_result, post_market_state)
        
        # Determine if episode is done (failed trade or major loss)
        done = not trading_result.success or trading_result.realized_pnl < -1000
        
        # Create completed experience
        experience_data = TradingExperienceData(
            experience_id=experience_id,
            timestamp=active_exp['timestamp'],
            action=active_exp['action'],
            pre_trade_state=active_exp['pre_trade_state'],
            post_trade_state=post_market_state.to_vector(),
            reward=reward,
            done=done,
            success=trading_result.success,
            realized_pnl=trading_result.realized_pnl,
            unrealized_pnl=trading_result.unrealized_pnl,
            fees=trading_result.fees,
            slippage=trading_result.slippage,
            execution_time_ms=(trading_result.executed_at - active_exp['timestamp']).total_seconds() * 1000,
            portfolio_value_before=trading_result.portfolio_value_before,
            portfolio_value_after=trading_result.portfolio_value_after,
            position_size_change=trading_result.position_change,
            market_volatility=post_market_state.market_volatility,
            market_direction=post_market_state.price_change_24h,
            confidence=post_market_state.prediction_confidence or 0.0,
            total_fees=trading_result.fees,
            slippage_penalty=abs(trading_result.slippage) * 10.0  # Penalty for high slippage
        )
        
        # Calculate reward components
        await self._calculate_reward_components(experience_data, active_exp, trading_result, post_market_state)
        
        # Move from active to completed
        self.completed_experiences.append(experience_data)
        del self.active_experiences[experience_id]
        
        self.total_experiences_captured += 1
        
        self.logger.info("Trading experience completed",
                        experience_id=experience_id,
                        action=experience_data.action.value,
                        reward=experience_data.reward,
                        pnl=experience_data.realized_pnl,
                        success=experience_data.success)
    
    async def add_experiences_to_buffer(self) -> int:
        """
        Add completed experiences to the replay buffer
        
        Returns:
            Number of experiences added
        """
        if not self.completed_experiences:
            return 0
        
        added_count = 0
        
        for exp_data in self.completed_experiences:
            try:
                # Convert to RL experience format
                rl_experience = self._convert_to_rl_experience(exp_data)
                
                # Add to replay buffer
                self.replay_buffer.add(rl_experience)
                added_count += 1
                
            except Exception as e:
                self.logger.error("Failed to add experience to buffer",
                                experience_id=exp_data.experience_id,
                                error=str(e))
        
        # Clear completed experiences after adding to buffer
        self.completed_experiences.clear()
        self.total_experiences_added_to_buffer += added_count
        
        self.logger.info("Experiences added to replay buffer", count=added_count)
        return added_count
    
    async def persist_experiences(self) -> None:
        """Persist completed experiences to file"""
        if not self.config.enable_persistence:
            return
        
        try:
            # Prepare data for serialization
            experiences_data = [exp.to_dict() for exp in self.completed_experiences]
            
            # Write to file
            with open(self.config.persistence_path, 'w') as f:
                json.dump(experiences_data, f, indent=2)
            
            self.logger.info("Experiences persisted",
                           count=len(experiences_data),
                           path=self.config.persistence_path)
        
        except Exception as e:
            self.logger.error("Failed to persist experiences", error=str(e))
            raise
    
    async def load_experiences(self) -> int:
        """
        Load experiences from file
        
        Returns:
            Number of experiences loaded
        """
        if not self.config.enable_persistence:
            return 0
        
        try:
            with open(self.config.persistence_path, 'r') as f:
                experiences_data = json.load(f)
            
            loaded_count = 0
            for exp_dict in experiences_data:
                try:
                    exp_data = TradingExperienceData.from_dict(exp_dict)
                    
                    # Convert to RL experience and add to buffer
                    rl_experience = self._convert_to_rl_experience(exp_data)
                    self.replay_buffer.add(rl_experience)
                    loaded_count += 1
                    
                except Exception as e:
                    self.logger.warning("Failed to load experience",
                                      experience_id=exp_dict.get('experience_id'),
                                      error=str(e))
            
            self.logger.info("Experiences loaded from file",
                           count=loaded_count,
                           path=self.config.persistence_path)
            return loaded_count
        
        except FileNotFoundError:
            self.logger.info("No previous experiences file found")
            return 0
        except Exception as e:
            self.logger.error("Failed to load experiences", error=str(e))
            raise
    
    async def cleanup_timed_out_experiences(self) -> int:
        """
        Clean up experiences that have timed out
        
        Returns:
            Number of experiences cleaned up
        """
        if not self.active_experiences:
            return 0
        
        timeout_delta = timedelta(hours=self.config.experience_timeout_hours)
        now = datetime.now()
        timed_out_ids = []
        
        for exp_id, exp_data in self.active_experiences.items():
            if now - exp_data['timestamp'] > timeout_delta:
                timed_out_ids.append(exp_id)
        
        # Remove timed out experiences
        for exp_id in timed_out_ids:
            del self.active_experiences[exp_id]
            self.logger.warning("Experience timed out", experience_id=exp_id)
        
        return len(timed_out_ids)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        session_duration = 0
        if self.session_start_time:
            session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        # Calculate average reward
        avg_reward = 0.0
        if self.completed_experiences:
            avg_reward = sum(exp.reward for exp in self.completed_experiences) / len(self.completed_experiences)
        
        # Calculate success rate
        success_rate = 0.0
        if self.completed_experiences:
            successful = sum(1 for exp in self.completed_experiences if exp.success)
            success_rate = successful / len(self.completed_experiences)
        
        return {
            'is_collecting': self.is_collecting,
            'session_duration': session_duration,
            'total_experiences': self.total_experiences_captured,
            'active_experiences': len(self.active_experiences),
            'completed_experiences': len(self.completed_experiences),
            'experiences_this_session': self.experiences_this_session,
            'experiences_added_to_buffer': self.total_experiences_added_to_buffer,
            'average_reward': avg_reward,
            'success_rate': success_rate,
            'buffer_can_sample': self.replay_buffer.can_sample() if self.replay_buffer else False,
            'buffer_size': len(self.replay_buffer) if self.replay_buffer else 0
        }
    
    def get_buffer_size(self) -> int:
        """Get current replay buffer size"""
        return len(self.replay_buffer) if self.replay_buffer else 0
    
    def should_trigger_training(self, threshold: int) -> bool:
        """Check if training should be triggered based on buffer size"""
        return self.get_buffer_size() >= threshold
    
    def _convert_to_rl_experience(self, exp_data: TradingExperienceData) -> Experience:
        """Convert trading experience to RL experience format"""
        return Experience(
            state=exp_data.pre_trade_state,
            action=self._encode_action_for_rl(exp_data.action),
            reward=exp_data.reward,
            next_state=exp_data.post_trade_state,
            done=exp_data.done,
            timestamp=exp_data.timestamp
        )
    
    def _encode_action_for_rl(self, action: TradeAction) -> int:
        """Encode trading action for RL training"""
        action_mapping = {
            TradeAction.HOLD: 0,
            TradeAction.BUY: 1,
            TradeAction.SELL: 2,
            TradeAction.STRONG_BUY: 3,
            TradeAction.STRONG_SELL: 4
        }
        return action_mapping.get(action, 0)
    
    async def _calculate_reward(self, active_exp: Dict[str, Any], trading_result: TradingResult, 
                               post_market_state: MarketState) -> float:
        """Calculate reward for the trading experience"""
        if not trading_result.success:
            return -10.0  # Penalty for failed trades
        
        # Base reward from P&L
        pnl_reward = trading_result.realized_pnl * self.config.pnl_weight
        
        # Execution quality penalty (fees + slippage)
        execution_penalty = (trading_result.fees + abs(trading_result.slippage) * 100) * self.config.execution_quality_weight
        
        # Risk penalty based on portfolio drawdown
        risk_penalty = 0.0
        if trading_result.portfolio_value_after < trading_result.portfolio_value_before:
            portfolio_loss_pct = (trading_result.portfolio_value_before - trading_result.portfolio_value_after) / trading_result.portfolio_value_before
            risk_penalty = portfolio_loss_pct * 100 * self.config.risk_weight
        
        # Market timing reward based on price movement alignment
        market_timing_reward = 0.0
        pre_state = active_exp['market_state']
        action = active_exp['action']
        price_change = (post_market_state.price_usd - pre_state.price_usd) / pre_state.price_usd
        
        if action in [TradeAction.BUY, TradeAction.STRONG_BUY] and price_change > 0:
            market_timing_reward = price_change * 50 * self.config.market_timing_weight
        elif action in [TradeAction.SELL, TradeAction.STRONG_SELL] and price_change < 0:
            market_timing_reward = abs(price_change) * 50 * self.config.market_timing_weight
        
        total_reward = pnl_reward - execution_penalty - risk_penalty + market_timing_reward
        
        return float(total_reward)
    
    async def _calculate_reward_components(self, exp_data: TradingExperienceData, 
                                          active_exp: Dict[str, Any], trading_result: TradingResult,
                                          post_market_state: MarketState) -> None:
        """Calculate detailed reward components for analysis"""
        # P&L reward component
        exp_data.pnl_reward = trading_result.realized_pnl * self.config.pnl_weight
        
        # Execution quality penalty
        exp_data.execution_quality_penalty = (trading_result.fees + abs(trading_result.slippage) * 100) * self.config.execution_quality_weight
        
        # Risk penalty
        if trading_result.portfolio_value_after < trading_result.portfolio_value_before:
            portfolio_loss_pct = (trading_result.portfolio_value_before - trading_result.portfolio_value_after) / trading_result.portfolio_value_before
            exp_data.risk_penalty = portfolio_loss_pct * 100 * self.config.risk_weight
        
        # Market timing reward
        pre_state = active_exp['market_state']
        price_change = (post_market_state.price_usd - pre_state.price_usd) / pre_state.price_usd
        action = active_exp['action']
        
        if action in [TradeAction.BUY, TradeAction.STRONG_BUY] and price_change > 0:
            exp_data.market_timing_reward = price_change * 50 * self.config.market_timing_weight
        elif action in [TradeAction.SELL, TradeAction.STRONG_SELL] and price_change < 0:
            exp_data.market_timing_reward = abs(price_change) * 50 * self.config.market_timing_weight

    # ========================================
    # DATABASE INTEGRATION METHODS
    # ========================================

    async def persist_experiences_to_database(self) -> int:
        """Persist completed experiences to database"""
        if not self.config.enable_database_storage or not self.completed_experiences:
            return 0
        
        try:
            # Prepare experiences for database insertion
            db_experiences = []
            for exp_data in self.completed_experiences:
                # Enrich experience with database-specific metadata
                enriched_exp = self._enrich_experience_for_database(exp_data)
                
                # Convert to database format
                db_exp = {
                    'session_id': self.database_session_id,
                    'step_number': len(db_experiences) + 1,
                    'state': enriched_exp.pre_trade_state.tolist(),
                    'action': self._encode_action_for_rl(enriched_exp.action),
                    'reward': enriched_exp.reward,
                    'next_state': enriched_exp.post_trade_state.tolist(),
                    'done': enriched_exp.done,
                    'priority': 0.5,  # Default priority
                    'metadata': {
                        'experience_id': enriched_exp.experience_id,
                        'timestamp': enriched_exp.timestamp.isoformat(),
                        'success': enriched_exp.success,
                        'realized_pnl': enriched_exp.realized_pnl,
                        'fees': enriched_exp.fees,
                        'slippage': enriched_exp.slippage,
                        'checksum': enriched_exp.checksum,
                        'version': enriched_exp.version,
                        'storage_metadata': enriched_exp.storage_metadata
                    }
                }
                db_experiences.append(db_exp)
            
            # Insert batch to database
            inserted_count = await insert_experience_batch(db_experiences)
            
            # Clear completed experiences after successful insertion
            self.completed_experiences.clear()
            
            self.logger.info("Experiences persisted to database", count=inserted_count)
            return inserted_count
            
        except Exception as e:
            self.logger.error("Failed to persist experiences to database", error=str(e))
            # Fallback to file persistence if available
            if self.config.persistence_path:
                await self._fallback_to_file_persistence()
            raise ExperienceCollectionError(f"Database persistence failed: {e}")

    async def load_experiences_from_database(self, session_id: str, limit: int = 1000, offset: int = 0) -> int:
        """Load experiences from database by session ID"""
        if not self.config.enable_database_storage:
            return 0
        
        try:
            experiences = await query_experiences_by_session(session_id, limit, offset)
            
            loaded_count = 0
            for exp_dict in experiences:
                try:
                    # Convert database format to RL experience
                    rl_experience = self._convert_database_experience_to_rl(exp_dict)
                    self.replay_buffer.add(rl_experience)
                    loaded_count += 1
                    
                except Exception as e:
                    self.logger.warning("Failed to load experience from database",
                                      experience_id=exp_dict.get('id'),
                                      error=str(e))
            
            self.logger.info("Experiences loaded from database",
                           count=loaded_count,
                           session_id=session_id)
            return loaded_count
            
        except Exception as e:
            self.logger.error("Failed to load experiences from database", 
                            session_id=session_id, error=str(e))
            return 0

    async def sample_prioritized_experiences_from_database(self, batch_size: int = 32, 
                                                         alpha: float = 0.6,
                                                         session_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Sample prioritized experiences from database"""
        if not self.config.enable_database_storage:
            return []
        
        try:
            return await sample_prioritized_experiences(batch_size, alpha, session_ids)
        except Exception as e:
            self.logger.error("Failed to sample prioritized experiences from database", error=str(e))
            return []

    async def update_experience_priorities_in_database(self, priority_updates: List[Dict[str, Any]]) -> int:
        """Update experience priorities in database after training"""
        if not self.config.enable_database_storage or not priority_updates:
            return 0
        
        try:
            return await update_experience_priorities(priority_updates)
        except Exception as e:
            self.logger.error("Failed to update experience priorities in database", error=str(e))
            return 0

    async def get_session_statistics_from_database(self, session_id: str) -> Dict[str, Any]:
        """Get session statistics from database"""
        if not self.config.enable_database_storage:
            return {}
        
        try:
            return await get_session_statistics(session_id)
        except Exception as e:
            self.logger.error("Failed to get session statistics from database", 
                            session_id=session_id, error=str(e))
            return {}

    async def check_database_health(self) -> Dict[str, Any]:
        """Check database health for experience storage"""
        if not self.config.enable_database_storage:
            return {'status': 'disabled'}
        
        try:
            return await check_rl_database_health()
        except Exception as e:
            self.logger.error("Database health check failed", error=str(e))
            return {'status': 'unhealthy', 'error': str(e)}

    async def stream_experiences_from_database(self, session_id: str, 
                                             batch_size: int = 1000) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Stream experiences from database with memory optimization"""
        if not self.config.enable_database_storage:
            return
        
        try:
            async for batch in stream_experiences_with_memory_optimization(session_id, batch_size):
                yield batch
        except Exception as e:
            self.logger.error("Failed to stream experiences from database", 
                            session_id=session_id, error=str(e))

    async def process_experiences_in_batches(self) -> int:
        """Process experiences in batches for database efficiency"""
        if not self.config.enable_database_storage or not self.completed_experiences:
            return 0
        
        batch_size = self.config.database_batch_size
        total_processed = 0
        
        try:
            # Process experiences in batches
            for i in range(0, len(self.completed_experiences), batch_size):
                batch = self.completed_experiences[i:i + batch_size]
                
                # Prepare batch for database
                db_experiences = []
                for exp_data in batch:
                    enriched_exp = self._enrich_experience_for_database(exp_data)
                    db_exp = self._convert_experience_to_database_format(enriched_exp)
                    db_experiences.append(db_exp)
                
                # Insert batch
                inserted_count = await insert_experience_batch(db_experiences)
                total_processed += inserted_count
                
                self.logger.debug("Processed experience batch", 
                                batch_size=len(batch), 
                                inserted=inserted_count)
            
            # Clear processed experiences
            self.completed_experiences.clear()
            
            self.logger.info("Batch processing completed", total_processed=total_processed)
            return total_processed
            
        except Exception as e:
            self.logger.error("Failed to process experiences in batches", error=str(e))
            return total_processed

    async def persist_experiences_to_database_with_retry(self) -> int:
        """Persist experiences to database with exponential backoff retry"""
        if not self.config.enable_database_storage:
            return 0
        
        retry_attempts = self.config.database_retry_attempts
        retry_delay = self.config.database_retry_delay
        
        for attempt in range(retry_attempts):
            try:
                return await self.persist_experiences_to_database()
                
            except Exception as e:
                if attempt == retry_attempts - 1:
                    # Last attempt failed
                    self.logger.error("All retry attempts failed for database persistence", 
                                    attempts=retry_attempts, error=str(e))
                    raise
                
                # Wait before retry with exponential backoff
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                
                self.logger.warning("Database persistence attempt failed, retrying",
                                  attempt=attempt + 1,
                                  retry_delay=retry_delay)
        
        return 0

    async def execute_concurrent_database_operations(self, operations: List[List[tuple]]) -> List[Dict[str, Any]]:
        """Execute concurrent database operations"""
        if not self.config.enable_database_storage:
            return []
        
        try:
            return await execute_concurrent_batch_operations(operations)
        except Exception as e:
            self.logger.error("Failed to execute concurrent database operations", error=str(e))
            return []

    async def persist_experiences_atomically(self) -> Dict[str, Any]:
        """Persist experiences atomically with session metrics"""
        if not self.config.enable_database_storage or not self.completed_experiences:
            return {'experiences_inserted': 0, 'session_updated': False}
        
        try:
            # Prepare experiences for atomic operation
            db_experiences = []
            for exp_data in self.completed_experiences:
                enriched_exp = self._enrich_experience_for_database(exp_data)
                db_exp = self._convert_experience_to_database_format(enriched_exp)
                db_experiences.append(db_exp)
            
            # Calculate session metrics
            session_metrics = {
                'session_id': self.database_session_id,
                'total_reward': sum(exp.reward for exp in self.completed_experiences),
                'episode_length': len(self.completed_experiences),
                'avg_reward': sum(exp.reward for exp in self.completed_experiences) / len(self.completed_experiences),
                'max_reward': max(exp.reward for exp in self.completed_experiences),
                'min_reward': min(exp.reward for exp in self.completed_experiences)
            }
            
            # Execute atomic operation
            result = await atomic_experience_batch_operation(db_experiences, session_metrics)
            
            # Clear experiences on success
            if result.get('experiences_inserted', 0) > 0:
                self.completed_experiences.clear()
            
            return result
            
        except Exception as e:
            self.logger.error("Atomic database operation failed", error=str(e))
            raise ExperienceCollectionError(f"Database transaction failed: {e}")

    def _enrich_experience_for_database(self, exp_data: TradingExperienceData) -> TradingExperienceData:
        """Enrich experience with database-specific metadata"""
        # Generate checksum for data integrity
        checksum = self._generate_experience_checksum(exp_data)
        
        # Create enriched copy
        enriched_exp = TradingExperienceData(
            experience_id=exp_data.experience_id,
            timestamp=exp_data.timestamp,
            action=exp_data.action,
            pre_trade_state=exp_data.pre_trade_state,
            post_trade_state=exp_data.post_trade_state,
            reward=exp_data.reward,
            done=exp_data.done,
            success=exp_data.success,
            realized_pnl=exp_data.realized_pnl,
            unrealized_pnl=exp_data.unrealized_pnl,
            fees=exp_data.fees,
            slippage=exp_data.slippage,
            execution_time_ms=exp_data.execution_time_ms,
            pnl_reward=exp_data.pnl_reward,
            execution_quality_penalty=exp_data.execution_quality_penalty,
            risk_penalty=exp_data.risk_penalty,
            market_timing_reward=exp_data.market_timing_reward,
            portfolio_value_before=exp_data.portfolio_value_before,
            portfolio_value_after=exp_data.portfolio_value_after,
            position_size_change=exp_data.position_size_change,
            market_volatility=exp_data.market_volatility,
            market_direction=exp_data.market_direction,
            confidence=exp_data.confidence,
            total_fees=exp_data.total_fees,
            slippage_penalty=exp_data.slippage_penalty,
            database_session_id=self.database_session_id,
            checksum=checksum,
            version=1,
            storage_metadata={
                'collector_id': id(self),
                'enriched_at': datetime.now().isoformat(),
                'batch_size': self.config.database_batch_size
            }
        )
        
        return enriched_exp

    def _generate_experience_checksum(self, exp_data: TradingExperienceData) -> str:
        """Generate SHA-256 checksum for experience data integrity"""
        # Create string representation of core experience data
        checksum_data = f"{exp_data.experience_id}|{exp_data.timestamp.isoformat()}|{exp_data.action.value}|{exp_data.reward}|{exp_data.success}"
        
        # Generate SHA-256 hash
        return hashlib.sha256(checksum_data.encode('utf-8')).hexdigest()

    def _validate_experience_checksum(self, exp_data: TradingExperienceData, checksum: str) -> bool:
        """Validate experience checksum for data integrity"""
        expected_checksum = self._generate_experience_checksum(exp_data)
        return expected_checksum == checksum

    def _convert_experience_to_database_format(self, exp_data: TradingExperienceData) -> Dict[str, Any]:
        """Convert experience data to database format"""
        return {
            'session_id': self.database_session_id,
            'step_number': self.total_experiences_captured + 1,
            'state': exp_data.pre_trade_state.tolist(),
            'action': self._encode_action_for_rl(exp_data.action),
            'reward': exp_data.reward,
            'next_state': exp_data.post_trade_state.tolist(),
            'done': exp_data.done,
            'priority': 0.5,
            'metadata': exp_data.to_dict()
        }

    def _convert_database_experience_to_rl(self, db_exp: Dict[str, Any]) -> Experience:
        """Convert database experience to RL experience format"""
        return Experience(
            state=np.array(db_exp['state'], dtype=np.float32),
            action=db_exp['action'],
            reward=db_exp['reward'],
            next_state=np.array(db_exp['next_state'], dtype=np.float32) if db_exp['next_state'] else None,
            done=db_exp['done'],
            timestamp=db_exp['created_at']
        )

    async def _fallback_to_file_persistence(self) -> None:
        """Fallback to file persistence when database fails"""
        try:
            await self.persist_experiences()
            self.logger.info("Fallback to file persistence successful")
        except Exception as e:
            self.logger.error("Fallback file persistence also failed", error=str(e))


class ExperienceCollectionError(Exception):
    """Raised when experience collection operations fail"""
    pass