# Claude Development Context - Shyvr AI RLTE

## 🎯 Project Overview

**Shyvr AI Reinforcement Learning Trading Engine** - An AI-augmented cryptocurrency trading bot with machine learning, reinforcement learning, and natural language agent capabilities for multi-chain token analysis and automated trading.

## 📊 Current Status (2025-07-23)

- **Current Phase**: Phase 4 - RL Trading Agent (Completed)
- **Overall Progress**: 85% Complete
- **Total Tests**: 315+ passing tests  
- **Test Coverage**: 62% overall
- **Architecture**: Production-ready microservices with Cloud Run deployment

## ✅ Completed Phases

### Phase 1: Foundation & Setup ✅
- Docker containerization with Google Cloud Run deployment
- Comprehensive configuration management (YAML + environment variables)
- Structured logging, health monitoring, CI/CD pipeline
- 88% test coverage, production-ready infrastructure

### Phase 2: Discovery & Evaluation ✅  
- **101 passing tests** for token discovery and evaluation
- Multi-chain support: Solana, Ethereum, Base blockchains
- BirdEye and Jupiter API clients with rate limiting
- Security evaluation including honeypot detection
- <30 second token evaluation pipeline

### Phase 3: ML Analysis & Feature Engineering ✅
- **89 passing ML tests** with comprehensive coverage
- **LSTM Neural Networks**: PyTorch implementation with attention mechanism
- **17 Technical Indicators**: RSI, MACD, SMA, EMA, Bollinger Bands, ATR, OBV
- **Ensemble Model System**: Multi-model predictions with dynamic weighting  
- **ML-Enhanced Evaluator**: Integrates ML predictions with fundamental analysis
- **Multi-timeframe Predictions**: 1h, 4h, 24h price forecasts
- **<1 second inference time** achieved

### Phase 4: RL Trading Agent ✅
- **125 passing RL tests** with 91-97% coverage per component
- **DQN Neural Network**: Deep Q-Network with PyTorch implementation
- **Trading Environment**: Realistic portfolio simulation with costs and slippage
- **Experience Replay**: Both standard and prioritized replay buffers
- **Advanced Reward Engineering**: Risk-adjusted returns with Sharpe ratio, VaR, drawdown penalties
- **Comprehensive Architecture**: Abstract base classes for extensibility

## 🏗️ Architecture Overview

### Core Modules
```
src/
├── discovery/        # Token discovery (BirdEye, Jupiter APIs) - 101 tests ✅
├── evaluation/       # Fundamental analysis + ML-enhanced evaluation  
├── ml_analysis/      # LSTM models, technical indicators - 89 tests ✅
├── rl_agent/         # Reinforcement learning (Phase 4 - 125 tests ✅)
├── agent/            # Natural language agent (future)
├── modes/            # Trading modes (analysis, simulation, live)
└── utils/            # Shared utilities and configuration
```

### Key Technologies
- **Backend**: Python 3.12, FastAPI, asyncio
- **ML/AI**: PyTorch, pandas, numpy, technical analysis
- **Database**: PostgreSQL with SQLAlchemy
- **Deployment**: Docker, Google Cloud Run, GitHub Actions
- **APIs**: BirdEye, Jupiter, Telegram Bot
- **Testing**: pytest, asyncio testing, comprehensive mocking

## 🧠 Phase 3: ML Analysis Deep Dive

### ML Architecture Components

#### 1. Base ML Framework (`src/ml_analysis/base.py`)
- **MLAnalyzerBase**: Abstract base class for all ML analyzers
- **PredictionResult**: Comprehensive prediction data structure
- **TechnicalIndicators**: 17 technical analysis indicators
- **MarketFeatures**: Market context and sentiment data
- **ModelType**: Enum for LSTM, Transformer, Ensemble models

#### 2. LSTM Neural Networks (`src/ml_analysis/lstm_model.py`)
- **LSTMNetwork**: PyTorch neural network with attention mechanism
- **LSTMPricePredictor**: Complete prediction pipeline
- **Multi-timeframe**: 1h, 4h, 24h price predictions
- **Trading Signals**: Stop-loss, take-profit, position sizing
- **Model Persistence**: Save/load trained models

#### 3. Feature Engineering (`src/ml_analysis/feature_engineer.py`)
- **Technical Indicators**: RSI, MACD, SMA, EMA, Bollinger Bands, ATR, OBV
- **Feature Matrix**: Normalized vectors for ML training
- **Caching System**: 30-minute TTL for performance
- **Market Context**: Fear/Greed index, volatility regime
- **Token Features**: Chain encoding, age, quality scoring

#### 4. Model Management (`src/ml_analysis/model_manager.py`)
- **Ensemble Coordination**: Multi-model weighted predictions
- **Performance Tracking**: Dynamic model weight adjustment
- **Health Monitoring**: Model status and performance metrics
- **Batch Processing**: Efficient multi-token analysis
- **Caching Strategy**: Intelligent prediction caching

#### 5. ML-Enhanced Evaluation (`src/evaluation/ml_evaluator.py`)
- **Hybrid Analysis**: ML + fundamental analysis combination
- **Risk Assessment**: ML-driven volatility and uncertainty calculation
- **Dynamic Weighting**: Configurable ML vs fundamental weights (40/60 default)
- **Trading Recommendations**: Buy/sell/hold/avoid with confidence scoring
- **Batch Evaluation**: Scalable multi-token evaluation

### Technical Achievements
- **95% Coverage** on base ML classes and feature engineering
- **84% Coverage** on LSTM neural network implementation  
- **78% Coverage** on model management and ensemble system
- **Sub-second Inference**: <1s ML prediction generation
- **Production Ready**: Robust error handling and performance optimization

## 🤖 Phase 4: RL Trading Agent Deep Dive

### RL Architecture Components

#### 1. Base RL Framework (`src/rl_agent/base.py`)
- **RLAgentBase**: Abstract base class for all RL trading agents
- **MarketState**: 19-dimensional feature vector for neural network input
- **TradeAction**: Enum for trading actions (BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL)
- **TradingResult**: Comprehensive trading execution tracking
- **RewardMetrics**: Portfolio performance and risk metrics

#### 2. DQN Neural Network (`src/rl_agent/dqn_agent.py`)
- **DQNNetwork**: PyTorch neural network with configurable architecture
- **DQNTradingAgent**: Complete DQN implementation with epsilon-greedy exploration
- **Experience Replay**: Integration with replay buffer systems
- **Target Network Updates**: Stable Q-learning with periodic target updates
- **Model Persistence**: Save/load trained DQN models

#### 3. Trading Environment (`src/rl_agent/trading_environment.py`)
- **Portfolio Management**: Realistic position tracking with P&L calculation
- **Transaction Costs**: Configurable fees and slippage simulation
- **Market Simulation**: Price updates with volatility and trend modeling
- **Technical Indicators**: RSI, MACD calculation for market state
- **Episode Management**: Configurable episode length and termination conditions

#### 4. Experience Replay (`src/rl_agent/experience_replay.py`)
- **Standard Replay Buffer**: Uniform sampling with configurable capacity
- **Prioritized Replay**: Priority-based sampling with importance weights
- **Memory Management**: Efficient deque-based storage with overflow handling
- **TD Error Updates**: Priority updates based on temporal difference errors
- **Beta Annealing**: Importance sampling weight annealing

#### 5. Advanced Reward Engineering (`src/rl_agent/reward_engineering.py`)
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, VaR
- **Market Adjustments**: Volatility regime and sentiment-based rewards
- **Consistency Rewards**: Rolling Sharpe stability and win rate optimization
- **Efficiency Metrics**: Transaction cost and execution quality rewards
- **Configurable Weights**: Customizable reward component weighting

### Technical Achievements
- **97% Coverage** on base RL classes and DQN implementation
- **93% Coverage** on experience replay buffer systems
- **91% Coverage** on trading environment simulation
- **96% Coverage** on advanced reward engineering
- **Sub-second Decisions**: <1s action prediction and execution
- **Production Ready**: Comprehensive error handling and logging

## 🧪 Testing Strategy

### Test Distribution
- **Discovery Tests**: 56 tests covering API clients and token scanning
- **Evaluation Tests**: 45 tests for fundamental analysis and security
- **ML Analysis Tests**: 89 tests across all ML components
- **ML Integration Tests**: 21 tests for evaluation pipeline integration
- **RL Agent Tests**: 125 tests across all RL components
  - Base framework: 28 tests (97% coverage)
  - DQN implementation: 25 tests (97% coverage)
  - Trading environment: 25 tests (91% coverage)
  - Experience replay: 24 tests (93% coverage)
  - Reward engineering: 21 tests (96% coverage)
- **Total**: 315+ comprehensive tests

### Testing Patterns
- **Test-Driven Development**: Write tests first, then implementation
- **Comprehensive Mocking**: External API and model mocking
- **Edge Case Coverage**: Error conditions and boundary testing
- **Integration Testing**: End-to-end pipeline validation
- **Performance Testing**: Model training and inference benchmarks

## 🔧 Development Guidelines

### Code Standards
- **Type Hints**: Full type annotation throughout codebase
- **Async/Await**: Non-blocking I/O for all operations
- **Error Handling**: Comprehensive exception handling with logging
- **Documentation**: Docstrings for all classes and methods
- **Testing**: 90%+ coverage requirement for new code

### ML Model Guidelines
- **Modular Design**: Abstract base classes for extensibility
- **Feature Engineering**: Standardized feature vector creation
- **Model Persistence**: Consistent save/load implementations
- **Performance Tracking**: Built-in accuracy and performance monitoring
- **Ensemble Ready**: Design for multi-model coordination

### Commit Patterns
- **Micro-commits**: Small, focused commits with clear messages
- **Feature Branches**: Separate branches for major features
- **Test Coverage**: Include tests in the same commit as implementation
- **Documentation**: Update docs with significant changes

## 🚀 Phase 5: Integration & Training Pipeline (Next Steps)

### Planned Implementation
- **RL Training Pipeline**: Automated DQN training with hyperparameter optimization
- **ML-RL Integration**: Connect RL agent with ML prediction pipeline
- **Backtesting Framework**: Historical performance evaluation system
- **Live Trading Interface**: Real-time trading execution with risk management
- **Performance Monitoring**: Real-time metrics and alerting

### Success Criteria
- 60% win rate after training convergence
- Sharpe ratio >1.5 in backtesting
- <15% maximum drawdown constraint
- Full ML-RL integration with <2s decision latency
- Production-ready trading pipeline

## 💡 Key Learnings & Patterns

### ML Pipeline Design
- **Async Architecture**: All ML operations are async for scalability
- **Caching Strategy**: Intelligent caching with TTL for performance
- **Error Resilience**: Graceful degradation when ML models fail
- **Ensemble Benefits**: Multiple models improve prediction accuracy
- **Feature Engineering**: Proper normalization critical for model performance

### Integration Patterns  
- **Hybrid Analysis**: Combining ML with fundamental analysis improves decisions
- **Dynamic Weighting**: Configurable weights allow fine-tuning
- **Confidence Scoring**: Model agreement boosts recommendation confidence
- **Risk Adjustment**: ML uncertainty feeds into risk calculations

### Testing Patterns
- **Mock External APIs**: Isolate unit tests from external dependencies
- **Test ML Edge Cases**: Handle insufficient data and model failures
- **Integration Testing**: Validate end-to-end pipeline behavior
- **Performance Testing**: Ensure sub-second response times

### RL Training Patterns
- **Experience Replay**: Efficient sample utilization with prioritized sampling
- **Reward Engineering**: Risk-adjusted returns prevent over-optimization
- **Environment Simulation**: Realistic trading costs and market dynamics
- **Model Persistence**: Consistent save/load for continuous training
- **Hyperparameter Optimization**: Systematic grid search for optimal performance

## 🔍 Common Issues & Solutions

### ML Model Issues
- **Data Insufficiency**: Fallback to default indicators when data is limited
- **Model Loading**: Save/load model architecture along with weights
- **Memory Management**: Efficient tensor operations and cleanup
- **Async Compatibility**: Ensure all ML operations are properly async

### Integration Challenges
- **Type Mismatches**: Careful handling of None values in calculations
- **Error Propagation**: Graceful error handling without breaking pipeline
- **Performance Bottlenecks**: Caching and batch processing optimization
- **Config Management**: Centralized configuration with validation

## 📚 Additional Resources

### Key Files
- `src/ml_analysis/base.py` - Core ML data structures and interfaces
- `src/ml_analysis/lstm_model.py` - Neural network implementation
- `src/ml_analysis/feature_engineer.py` - Technical indicators and features
- `src/ml_analysis/model_manager.py` - Ensemble coordination
- `src/evaluation/ml_evaluator.py` - ML-enhanced evaluation pipeline
- `src/rl_agent/base.py` - Core RL data structures and interfaces
- `src/rl_agent/dqn_agent.py` - Deep Q-Network implementation
- `src/rl_agent/trading_environment.py` - Trading simulation environment
- `src/rl_agent/experience_replay.py` - Experience replay buffers
- `src/rl_agent/reward_engineering.py` - Advanced reward calculation

### Documentation
- `README.md` - Project overview and setup instructions
- `PROGRESS.md` - Detailed development progress and metrics
- `config/config.yaml` - Main configuration with ML parameters
- `tests/unit/ml_analysis/` - Comprehensive ML test suite
- `tests/unit/rl_agent/` - Comprehensive RL test suite

### Performance Benchmarks
- Token discovery: <5 minutes for new tokens
- Token evaluation: <30 seconds per token
- ML inference: <1 second per prediction  
- Batch processing: 100+ tokens per minute
- Model training: Convergence within 1000 epochs
- RL action prediction: <1 second per decision
- Trading environment step: <100ms per action
- Experience replay sampling: <50ms per batch

---

*Last Updated: 2025-07-23*  
*Current Focus: Phase 5 Integration & Training Pipeline*  
*Next Milestone: ML-RL Integration and Training Pipeline*