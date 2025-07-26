# Shyvr RLTE Dashboard - Local Testing Guide

This guide provides complete setup instructions for testing the Shyvr RLTE dashboard in a local development environment.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+ with pip
- Git

### 1. Environment Setup

Copy the local development environment file:
```bash
cp .env.local .env
```

### 2. Database Setup

Start the local PostgreSQL database:
```bash
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml up -d postgres
```

Wait for the database to be healthy:
```bash
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml ps
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Application

```bash
python main.py
```

### 5. Access the Dashboard

- **Dashboard UI**: http://localhost:8080/
- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health

## 📋 Authentication & API Testing

### Default Admin Credentials

The application creates a default admin user on startup:
- **Username**: admin
- **API Key**: Check the console output for the generated API key

Example output:
```
2025-07-26 17:11:28 [info] Default admin user created api_key=6T3XFDjzrFnBXlktjqqJrRMR2Eg-ApGOuv96NlgKKC4 username=admin
```

### API Testing with curl

Replace `YOUR_API_KEY` with the generated API key from the console:

#### Health Check
```bash
curl http://localhost:8080/health
```

#### Dashboard Data
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/data
```

#### Trading Status
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/trading/status
```

#### Switch to Mode 1 (Analysis)
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"mode": "analysis"}' \
     http://localhost:8080/dashboard/trading/mode
```

#### Switch to Mode 2 (Simulation)
```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"mode": "simulation"}' \
     http://localhost:8080/dashboard/trading/mode
```

#### System Health
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/system/health
```

## 🎯 Testing Modes 1 & 2

### Mode 1: Analysis Mode
- **Purpose**: Token discovery and analysis without trading
- **Features**:
  - Real-time token scanning
  - ML-powered analysis
  - Risk assessment
  - Portfolio simulation
- **Safety**: No actual trading, read-only operations

### Mode 2: Simulation Mode  
- **Purpose**: Paper trading with virtual portfolio
- **Features**:
  - Virtual trading with mock funds ($10,000 starting balance)
  - Real market data for analysis
  - ML/RL training and testing
  - Performance tracking
- **Safety**: No real money involved, all trades are simulated

### Mode Switching via Dashboard UI

1. Open http://localhost:8080/
2. Navigate to the **Trading** tab
3. Select desired mode (Analysis or Simulation)
4. Click **Switch Mode**
5. Verify the mode change in the **Overview** tab

### Mode Switching via API

```bash
# Switch to Analysis Mode (Mode 1)
curl -X POST \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"mode": "analysis"}' \
     http://localhost:8080/dashboard/trading/mode

# Switch to Simulation Mode (Mode 2)
curl -X POST \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"mode": "simulation"}' \
     http://localhost:8080/dashboard/trading/mode
```

## 🔧 Configuration

### Local Development Settings (.env.local)

Key configuration highlights:
- **Environment**: development
- **Database**: Local PostgreSQL via Docker
- **Trading Modes**: Analysis and Simulation enabled, Live disabled
- **Security**: Development-only keys (NOT for production)
- **Wallets**: Testnet/Devnet only for safety

### Database Configuration

- **Host**: localhost:5432
- **Database**: shyvr_rlte
- **User**: rlte_user
- **Password**: dev_password_change_in_prod

### API Keys

For full functionality, add your API keys to `.env.local`:
```env
BIRDEYE_API_KEY=your_birdeye_api_key
HELIUS_API_KEY=your_helius_api_key
ETHERSCAN_API_KEY=your_etherscan_api_key
```

## 🧪 Testing Features

### 1. Dashboard UI Components

- **Overview Tab**: System status, portfolio summary, trading activity
- **Trading Tab**: Mode selection, controls, recent trades
- **Portfolio Tab**: Detailed portfolio view, positions, risk metrics
- **ML & RL Tab**: Model status, training progress, integration metrics
- **System Tab**: Health monitoring, performance metrics, logs
- **Settings Tab**: Risk management, dashboard preferences

### 2. WebSocket Real-time Updates

The dashboard uses WebSocket for real-time updates:
- **URL**: ws://localhost:8080/dashboard/ws/{connection_id}
- **Features**: Live data updates, system alerts, trading notifications

### 3. Mock Data for Testing

The application includes comprehensive mock data:
- Portfolio value: $10,000
- Daily P&L: $75.25 (0.75%)
- Active positions: 3
- Trading performance: 65.5% win rate
- ML accuracy: 78%
- RL success rate: 64.2%

## 🐛 Troubleshooting

### Application Won't Start

1. **Check Docker**: Ensure Docker Desktop is running
2. **Database**: Verify PostgreSQL container is healthy
3. **Dependencies**: Run `pip install -r requirements.txt`
4. **Environment**: Ensure `.env.local` exists and is configured

### Dashboard Not Loading

1. **Check Server**: Verify application is running on port 8080
2. **Browser Cache**: Clear browser cache and reload
3. **Console Errors**: Check browser developer console for JavaScript errors

### API Authentication Errors

1. **API Key**: Ensure you're using the correct API key from console output
2. **Headers**: Verify `Authorization: Bearer YOUR_API_KEY` header format
3. **Permissions**: Check that the user has required permissions

### Database Connection Issues

1. **Container Status**: Check PostgreSQL container with `docker ps`
2. **Health Check**: Verify database health with Docker Compose
3. **Port Conflicts**: Ensure port 5432 is not in use by other applications

### Performance Issues

1. **Resource Usage**: Check Docker resource limits
2. **Log Level**: Reduce log level from DEBUG to INFO in production
3. **Mock Data**: Enable mock data for faster testing

## 📊 Expected Test Results

### Successful Health Check Response
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "components": {
    "database": "not_implemented",
    "ml_models": "not_implemented", 
    "rl_agent": "not_implemented"
  }
}
```

### Successful Mode Switch Response
```json
{
  "success": true,
  "message": "Trading mode switched to simulation",
  "timestamp": "2025-07-26T21:19:20.366558"
}
```

### Successful Dashboard Data Response
Contains complete system metrics, portfolio status, trading status, and ML/RL status with mock data.

## 🔐 Security Notes

### Local Development Security

- Default admin user with known credentials
- Development-only secret keys
- No real wallet private keys
- Testnet/devnet configurations only
- Live trading disabled by default

### Production Considerations

- Change all default passwords and secret keys
- Use secure, random API keys
- Enable proper authentication and authorization
- Configure real wallet keys securely
- Enable monitoring and alerting

## 📚 Next Steps

After successful local testing:

1. **Development**: Start building custom features
2. **Integration**: Connect real API keys for live data
3. **Testing**: Run comprehensive test suite
4. **Deployment**: Follow production deployment guide
5. **Monitoring**: Set up logging and alerting

## 🆘 Support

For issues:
1. Check logs in `logs/rlte.log`
2. Review Docker container logs
3. Verify configuration in `.env.local`
4. Run the test script: `python test_dashboard_local.py`

The dashboard is now fully configured for local development and testing of modes 1 (Analysis) and 2 (Simulation)!