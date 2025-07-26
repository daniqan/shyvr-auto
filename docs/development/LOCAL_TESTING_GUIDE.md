# Local Testing Setup Guide

## Overview

This guide provides step-by-step instructions for setting up a complete local testing environment for the Shyvr AI RLTE trading engine. You'll be able to run Analysis Mode and Simulation Mode with real market data while using safe testnet networks for blockchain interactions.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [API Keys Setup](#api-keys-setup)
3. [Wallet Configuration](#wallet-configuration)
4. [RPC Endpoint Configuration](#rpc-endpoint-configuration)
5. [Environment Setup](#environment-setup)
6. [Docker Setup](#docker-setup)
7. [Running Tests](#running-tests)
8. [Verification Steps](#verification-steps)
9. [Mode-Specific Testing](#mode-specific-testing)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software
- **Docker** (version 20.0+) and **Docker Compose** (version 2.0+)
- **Python** 3.12+ with **uv** package manager
- **Git** for version control
- **curl** for API testing

### System Requirements
- 4GB+ RAM available for Docker containers
- 2GB+ free disk space
- Stable internet connection for API calls

### Install Dependencies

```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <your-repo-url>
cd shyvrai-rlte

# Install Python dependencies
uv sync
```

## API Keys Setup

### Required API Keys

#### 1. BaseScan API Key (Required)
**Purpose**: Blockchain data for Base network analysis

**Setup Steps**:
1. Visit [BaseScan](https://basescan.org/apis)
2. Create a free account
3. Generate an API key
4. Free tier provides 100,000 calls/day (sufficient for testing)

**Rate Limits**: 5 requests/second (handled automatically by the system)

#### 2. BirdEye API Key (Essential)
**Purpose**: Real-time cryptocurrency price data and market metrics

**Setup Steps**:
1. Visit [BirdEye API](https://birdeye.so/developers)
2. Sign up for developer access
3. Generate API key
4. Free tier provides comprehensive market data

**Rate Limits**: 60 requests/minute

#### 3. Helius API Key (For Solana)
**Purpose**: Solana blockchain data and RPC services

**Setup Steps**:
1. Visit [Helius](https://helius.xyz/)
2. Create developer account
3. Generate API key
4. Free tier provides sufficient RPC calls for testing

**Rate Limits**: 100 requests/minute

#### 4. Etherscan API Key (For Ethereum)
**Purpose**: Ethereum blockchain data and contract verification

**Setup Steps**:
1. Visit [Etherscan API](https://etherscan.io/apis)
2. Create free account
3. Generate API key
4. Free tier provides 100,000 calls/day

**Rate Limits**: 5 requests/second

### Optional API Keys

#### X (Twitter) API (For Social Sentiment)
**Setup Steps**:
1. Visit [X Developer Platform](https://developer.x.com/)
2. Apply for developer account
3. Create app and generate bearer token
4. Note: Enhanced features require paid tier

#### Telegram Bot Token (For Notifications)
**Setup Steps**:
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Create new bot with `/newbot`
3. Save the bot token provided
4. Add bot to your testing channel

## Wallet Configuration

### Testnet Wallet Setup

**Important**: Use dedicated testnet wallets with NO real funds

#### Solana Testnet Wallet

```bash
# Install Solana CLI (if not installed)
sh -c "$(curl -sSfL https://release.solana.com/v1.18.4/install)"

# Create new testnet wallet
solana-keygen new --outfile ~/solana-testnet-keypair.json

# Set to devnet
solana config set --url https://api.devnet.solana.com

# Get wallet address
solana address

# Request testnet SOL (devnet)
solana airdrop 5 <your-wallet-address>
```

#### Ethereum Testnet Wallet (Sepolia)

```bash
# Option 1: Use MetaMask
# 1. Install MetaMask browser extension
# 2. Create new wallet or import existing
# 3. Add Sepolia testnet network
# 4. Get testnet ETH from Sepolia faucet

# Option 2: Create with Python (programmatic)
python3 -c "
from eth_account import Account
account = Account.create()
print(f'Address: {account.address}')
print(f'Private Key: {account.key.hex()}')
"
```

**Get Testnet Funds**:
- Sepolia ETH: [Sepolia Faucet](https://sepoliafaucet.com/)
- Base Sepolia: [Base Faucet](https://bridge.base.org/deposit)

#### Base Testnet Wallet

Base uses the same Ethereum-compatible wallets. Simply add Base Sepolia network:
- **Network Name**: Base Sepolia
- **RPC URL**: https://sepolia.base.org
- **Chain ID**: 84532
- **Currency Symbol**: ETH
- **Block Explorer**: https://sepolia.basescan.org

## RPC Endpoint Configuration

### Free RPC Providers

#### Ethereum Mainnet/Sepolia
```bash
# Option 1: Infura (Recommended)
# 1. Visit https://infura.io/
# 2. Create free account
# 3. Create new project
# 4. Use project ID in RPC URL

ETH_RPC_URL="https://sepolia.infura.io/v3/YOUR_PROJECT_ID"

# Option 2: Alchemy
ETH_RPC_URL="https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY"

# Option 3: Public RPC (rate limited)
ETH_RPC_URL="https://rpc.sepolia.org"
```

#### Solana
```bash
# Devnet (recommended for testing)
SOLANA_RPC_URL="https://api.devnet.solana.com"

# With Helius (better performance)
SOLANA_RPC_URL="https://devnet.helius-rpc.com/?api-key=YOUR_HELIUS_KEY"
```

#### Base
```bash
# Base Sepolia testnet
BASE_RPC_URL="https://sepolia.base.org"

# Base mainnet (for price data only)
BASE_RPC_URL="https://mainnet.base.org"
```

## Environment Setup

### Create Environment File

```bash
# Copy example environment file
cp .env.example .env.local

# Edit with your API keys and configuration
nano .env.local  # or use your preferred editor
```

### Local Testing Configuration

```bash
# .env.local - Local Testing Configuration

# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Database (will be created by Docker)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=shyvr_rlte
DB_USER=rlte_user
DB_PASSWORD=local_dev_password

# Required API Keys
BASESCAN_API_KEY=your_basescan_api_key_here
BIRDEYE_API_KEY=your_birdeye_api_key_here
HELIUS_API_KEY=your_helius_api_key_here
ETHERSCAN_API_KEY=your_etherscan_api_key_here

# Optional API Keys (can be left empty for basic testing)
X_BEARER_TOKEN=your_x_bearer_token_here
TELEGRAM_TOKEN=your_telegram_bot_token_here

# Wallet Configuration (TESTNET ONLY)
# Solana Testnet
SOLANA_NETWORK=devnet
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_PRIVATE_KEY=your_solana_testnet_private_key_base58
SOLANA_WALLET_ADDRESS=your_solana_testnet_address

# Ethereum Testnet
ETH_NETWORK=sepolia
ETH_RPC_URL=https://sepolia.infura.io/v3/YOUR_PROJECT_ID
ETH_PRIVATE_KEY=your_ethereum_testnet_private_key_hex
ETH_WALLET_ADDRESS=your_ethereum_testnet_address

# Base Testnet
BASE_NETWORK=base-sepolia
BASE_RPC_URL=https://sepolia.base.org
BASE_PRIVATE_KEY=your_base_testnet_private_key_hex
BASE_WALLET_ADDRESS=your_base_testnet_address

# Trading Configuration (Safe defaults for testing)
TRADING_MODES_ANALYSIS=true
TRADING_MODES_SIMULATION=true
TRADING_MODES_LIVE=false

# Risk Management (Conservative for testing)
RISK_MAX_POSITION_SIZE_PCT=0.1  # 0.1% max position
RISK_MAX_DAILY_LOSS_PCT=1.0     # 1% daily loss limit
RISK_STOP_LOSS_PCT=5.0          # 5% stop loss
SIMULATION_INITIAL_BALANCE=1000  # $1000 virtual balance
```

## Docker Setup

### Build and Start Services

```bash
# Build the application
docker-compose -f docker/docker-compose.yml build

# Start PostgreSQL database
docker-compose -f docker/docker-compose.yml up postgres -d

# Wait for database to be ready (check logs)
docker-compose -f docker/docker-compose.yml logs postgres

# Start the main application
docker-compose -f docker/docker-compose.yml up rlte -d

# View application logs
docker-compose -f docker/docker-compose.yml logs -f rlte
```

### Alternative: Start All Services

```bash
# Start all services at once
docker-compose -f docker/docker-compose.yml up -d

# Monitor startup
docker-compose -f docker/docker-compose.yml logs -f
```

### Environment Variables for Docker

Create a `.env` file in the project root for Docker Compose:

```bash
# .env file for Docker Compose
DB_PASSWORD=local_dev_password
HELIUS_API_KEY=your_helius_api_key
ETHERSCAN_API_KEY=your_etherscan_api_key
BIRDEYE_API_KEY=your_birdeye_api_key
BASESCAN_API_KEY=your_basescan_api_key
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
uv run python scripts/run_tests.py --unit

# Run specific module tests
uv run python scripts/run_tests.py --unit --module ml_analysis
uv run python scripts/run_tests.py --unit --module rl_agent
uv run python scripts/run_tests.py --unit --module modes

# Run with coverage
uv run python scripts/run_tests.py --unit --coverage
```

### Integration Tests

```bash
# Run integration tests (requires API keys)
uv run python scripts/run_tests.py --integration

# Run ML-RL integration tests
uv run python scripts/run_tests.py --integration --module integration

# Run cross-module tests
uv run python scripts/run_tests.py --integration --module cross_module
```

### Performance Tests

```bash
# Run performance benchmarks
uv run python scripts/run_tests.py --performance

# Specific performance test
uv run python scripts/run_performance_tests.py
```

### End-to-End Tests

```bash
# Run complete pipeline tests (requires all APIs)
uv run python scripts/run_tests.py --e2e

# Test specific modes
uv run pytest tests/integration/test_mode_integration_e2e.py -v
```

## Verification Steps

### 1. Health Check

```bash
# Check application health
curl -f http://localhost:8080/health

# Expected response:
# {"status":"healthy","timestamp":"2025-01-XX","version":"0.1.0"}
```

### 2. Database Connection

```bash
# Check database connection
docker-compose -f docker/docker-compose.yml exec postgres psql -U rlte_user -d shyvr_rlte -c "SELECT version();"
```

### 3. API Connectivity

```bash
# Test API endpoints (if application exposes test endpoints)
curl -f http://localhost:8080/api/test/birdeye
curl -f http://localhost:8080/api/test/helius
curl -f http://localhost:8080/api/test/basescan
```

### 4. Wallet Connectivity

```bash
# Test wallet connections (use application endpoints or logs)
# Check application logs for wallet initialization:
docker-compose -f docker/docker-compose.yml logs rlte | grep -i "wallet.*connected"
```

### 5. Run System Health Check

```bash
# Run built-in health checks
uv run python -m src.modes.system_health_monitor

# Or via pytest
uv run pytest tests/unit/modes/test_system_health_monitor.py -v
```

## Mode-Specific Testing

### Analysis Mode Testing

Analysis Mode performs market analysis without trading.

#### Start Analysis Mode

```bash
# Via Python script
uv run python -c "
import asyncio
from src.modes.analysis_mode import AnalysisMode
from src.modes.base import ModeConfig

async def test_analysis():
    config = ModeConfig(mode_name='analysis_test')
    analysis = AnalysisMode(config)
    await analysis.initialize()
    
    # Analyze a specific token (example: SOL)
    result = await analysis.analyze_token('So11111111111111111111111111111111111111112')
    print(f'Analysis Result: {result}')
    
    await analysis.cleanup()

asyncio.run(test_analysis())
"
```

#### Analysis Mode Features to Test

1. **Token Discovery**: Scans for new tokens on configured chains
2. **Fundamental Analysis**: Evaluates token metrics (liquidity, holders, etc.)
3. **Technical Analysis**: 17 technical indicators (RSI, MACD, Bollinger Bands)
4. **ML Predictions**: LSTM neural network price predictions
5. **Risk Assessment**: Honeypot detection and security evaluation
6. **Reporting**: Generates comprehensive analysis reports

#### Test Commands

```bash
# Test token discovery
uv run pytest tests/unit/discovery/ -v

# Test evaluation pipeline
uv run pytest tests/unit/evaluation/ -v

# Test ML analysis
uv run pytest tests/unit/ml_analysis/ -v

# Test complete analysis mode
uv run pytest tests/unit/modes/test_analysis_mode.py -v
```

### Simulation Mode Testing

Simulation Mode performs paper trading with virtual funds.

#### Start Simulation Mode

```bash
# Via Python script
uv run python -c "
import asyncio
from src.modes.simulation_mode import SimulationMode
from src.modes.base import ModeConfig

async def test_simulation():
    config = ModeConfig(mode_name='simulation_test')
    simulation = SimulationMode(config)
    await simulation.initialize()
    
    # Start simulation with virtual portfolio
    await simulation.start_simulation()
    
    # Run for a few iterations
    for i in range(10):
        await simulation.step()
        await asyncio.sleep(1)
    
    # Get performance metrics
    metrics = await simulation.get_performance_metrics()
    print(f'Simulation Metrics: {metrics}')
    
    await simulation.cleanup()

asyncio.run(test_simulation())
"
```

#### Simulation Mode Features to Test

1. **Virtual Portfolio**: Manages virtual funds and positions
2. **Paper Trading**: Executes trades without real money
3. **RL Agent Training**: Trains DQN agent on market data
4. **Risk Management**: Enforces stop-loss and position limits
5. **Performance Tracking**: Tracks P&L, Sharpe ratio, drawdown
6. **Experience Collection**: Gathers data for RL training

#### Test Commands

```bash
# Test simulation mode
uv run pytest tests/unit/modes/test_simulation_mode.py -v

# Test RL agent
uv run pytest tests/unit/rl_agent/ -v

# Test portfolio management
uv run pytest tests/unit/portfolio/ -v

# Test simulation experience integration
uv run pytest tests/unit/modes/test_simulation_experience_integration.py -v
```

### Integration Testing

#### ML-RL Integration

```bash
# Test ML-RL integration bridge
uv run pytest tests/unit/integration/test_ml_rl_integration.py -v

# Test cross-module integration
uv run pytest tests/integration/test_cross_module_integration.py -v

# Test accuracy validation
uv run pytest tests/validation/test_ml_rl_accuracy_validation.py -v
```

#### Real Data Integration

```bash
# Test with real APIs (requires API keys)
uv run pytest tests/integration/test_jupiter_solana_integration.py -v

# Test real ML models
uv run pytest tests/integration/test_real_ml_models.py -v

# Test real RL integration
uv run pytest tests/integration/test_real_rl_integration.py -v
```

## Troubleshooting

### Common Issues

#### 1. Docker Container Fails to Start

**Symptoms**: Container exits immediately or fails health checks

**Solutions**:
```bash
# Check container logs
docker-compose -f docker/docker-compose.yml logs rlte

# Check environment variables
docker-compose -f docker/docker-compose.yml exec rlte env | grep -E "(API_KEY|RPC_URL)"

# Restart with clean state
docker-compose -f docker/docker-compose.yml down -v
docker-compose -f docker/docker-compose.yml up --build
```

#### 2. Database Connection Issues

**Symptoms**: Database connection errors in logs

**Solutions**:
```bash
# Check PostgreSQL status
docker-compose -f docker/docker-compose.yml exec postgres pg_isready -U rlte_user

# Reset database
docker-compose -f docker/docker-compose.yml down postgres
docker volume rm docker_postgres_data
docker-compose -f docker/docker-compose.yml up postgres -d
```

#### 3. API Rate Limit Errors

**Symptoms**: 429 errors in logs, API calls failing

**Solutions**:
- Verify API keys are correct and active
- Check rate limits in `config/config.yaml`
- Reduce request frequency for testing
- Use mock data for development

```bash
# Test API keys manually
curl -H "X-API-KEY: your_birdeye_key" "https://public-api.birdeye.so/defi/price?address=So11111111111111111111111111111111111111112"
```

#### 4. Wallet Connection Issues

**Symptoms**: Wallet initialization failures, RPC errors

**Solutions**:
```bash
# Test RPC endpoints
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' \
  https://api.devnet.solana.com

# Verify wallet addresses
# Solana
solana balance YOUR_WALLET_ADDRESS --url devnet

# Ethereum (using web3)
python3 -c "
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('https://sepolia.infura.io/v3/YOUR_PROJECT_ID'))
print(f'Connected: {w3.is_connected()}')
print(f'Balance: {w3.eth.get_balance(\"YOUR_WALLET_ADDRESS\")}')
"
```

#### 5. Missing Dependencies

**Symptoms**: Import errors, missing packages

**Solutions**:
```bash
# Reinstall dependencies
uv sync --force

# Check Python version
python --version  # Should be 3.12+

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

#### 6. ML Model Loading Issues

**Symptoms**: PyTorch errors, model loading failures

**Solutions**:
```bash
# Test PyTorch installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Clear model cache
rm -rf models/*.pt models/*.pkl

# Run ML tests
uv run pytest tests/unit/ml_analysis/test_lstm_model.py -v
```

#### 7. Permission Errors

**Symptoms**: File permission errors, Docker socket issues

**Solutions**:
```bash
# Fix Docker permissions (Linux)
sudo usermod -aG docker $USER
# Logout and login again

# Fix file permissions
chmod +x scripts/*.sh
chmod 600 .env*
```

### Debug Mode

#### Enable Verbose Logging

```bash
# Set debug logging in .env
LOG_LEVEL=DEBUG

# Or export environment variable
export LOG_LEVEL=DEBUG

# Restart services
docker-compose -f docker/docker-compose.yml restart
```

#### Debug Specific Modules

```bash
# Run specific tests with debug output
uv run pytest tests/unit/modes/test_analysis_mode.py -v -s --log-cli-level=DEBUG

# Python debug session
uv run python -m pdb -c "
from src.modes.analysis_mode import AnalysisMode
# Set breakpoints and debug
"
```

### Performance Issues

#### Check Resource Usage

```bash
# Monitor Docker containers
docker stats

# Check system resources
htop  # or top
df -h  # disk usage
free -m  # memory usage
```

#### Optimize for Development

```bash
# Reduce ML model complexity (in config/config.yaml)
ml:
  models:
    lstm:
      hidden_size: 32  # Reduced from 64
      num_layers: 1    # Reduced from 2
      
# Reduce batch sizes
rl:
  batch_size: 32      # Reduced from 64
  memory_size: 10000  # Reduced from 50000
```

### Getting Help

#### Logs and Diagnostics

```bash
# Collect all logs
docker-compose -f docker/docker-compose.yml logs > debug_logs.txt

# System health report
uv run python -m src.modes.system_health_monitor > health_report.txt

# Test results
uv run python scripts/run_tests.py --all --coverage > test_report.txt
```

#### Check Configuration

```bash
# Validate configuration
uv run python -c "
from src.utils.config import Config
config = Config()
print('Configuration loaded successfully')
print(f'Trading modes enabled: {config.trading.modes}')
print(f'APIs configured: {list(config.apis.keys())}')
"
```

## Next Steps

Once your local testing environment is working:

1. **Run Analysis Mode** on real tokens to see ML predictions
2. **Start Simulation Mode** to test trading strategies
3. **Monitor Performance** using the built-in metrics
4. **Experiment with Configuration** to optimize for your use case
5. **Review Logs** to understand system behavior
6. **Contribute** improvements back to the project

## Security Reminders

- ⚠️ **NEVER** use mainnet private keys in testing
- ⚠️ **ALWAYS** use testnet/devnet networks
- ⚠️ **NEVER** commit private keys to version control
- ⚠️ **USE** dedicated testing wallets with minimal funds
- ⚠️ **VERIFY** all configuration before running live trades

---

**Happy Testing!** 🚀

For additional support, check the project documentation or create an issue in the repository.