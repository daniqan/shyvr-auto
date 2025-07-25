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
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import structlog

from src.rl_agent.base import MarketState, TradeAction, TradingResult
from src.rl_agent.experience_replay import Experience, ExperienceReplayBuffer
from src.ml_analysis.base import PredictionResult


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
            'slippage_penalty': self.slippage_penalty
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
            slippage_penalty=data.get('slippage_penalty', 0.0)
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
        
        self.logger.info("Trading Experience Collector initialized",
                        buffer_size=config.buffer_size,
                        enable_persistence=config.enable_persistence,
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
        
        # Load previous experiences if persistence enabled
        if self.config.enable_persistence:
            try:
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


class ExperienceCollectionError(Exception):
    """Raised when experience collection operations fail"""
    pass