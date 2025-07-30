# Shyvr RLTE Dashboard Access and Operation Guide

This comprehensive guide covers accessing, navigating, and operating all dashboards in the Shyvr RLTE trading system for operators and administrators.

## Table of Contents

1. [Dashboard Overview](#dashboard-overview)
2. [Web Dashboard Access](#web-dashboard-access)
3. [Grafana Dashboard](#grafana-dashboard)
4. [Google Cloud Console](#google-cloud-console)
5. [Dashboard Navigation and Features](#dashboard-navigation-and-features)
6. [Monitoring Trading Performance](#monitoring-trading-performance)
7. [ML/RL Metrics Monitoring](#mlrl-metrics-monitoring)
8. [Charts and Alerts Interpretation](#charts-and-alerts-interpretation)
9. [User Management and Access Controls](#user-management-and-access-controls)
10. [Dashboard Troubleshooting](#dashboard-troubleshooting)

## Dashboard Overview

The Shyvr RLTE system provides three primary monitoring interfaces:

### Dashboard Architecture
```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Web Dashboard     │    │   Grafana        │    │ Google Cloud Console│
│   - Real-time UI    │    │   - Metrics      │    │ - Infrastructure    │
│   - Trading Control │    │   - Visualization│    │ - Native Monitoring │
│   - XAI Integration │    │   - Alerting     │    │ - Alert Policies    │
│   - WebSocket Data  │    │   - History      │    │ - Custom Metrics    │
└─────────────────────┘    └──────────────────┘    └─────────────────────┘
         │                            │                        │
         └────────────────────────────┼────────────────────────┘
                                      │
                         ┌──────────────────┐
                         │ PostgreSQL       │
                         │ - RL Experience  │
                         │ - Trading Data   │
                         │ - System Logs    │
                         └──────────────────┘
```

## Web Dashboard Access

### 1. Local Development Access

**Default URL**: `http://localhost:8080`

**Quick Start:**
```bash
# Start the application
python main.py

# Alternative port if 8080 is in use
PORT=8081 python main.py
```

**Environment Configuration:**
```bash
# Set custom host and port
export HOST=0.0.0.0
export PORT=8080
```

### 2. Production Access

**HTTPS URL**: `https://your-domain.com`

**Load Balancer Configuration:**
- Traffic distributed across multiple instances
- SSL/TLS termination at load balancer
- Health checks on `/health` endpoint

### 3. Authentication

#### Default Admin Credentials (Development)
- **Username**: `admin`
- **API Key**: Generated on startup (check console output)

```bash
# Example console output
2025-07-26 17:11:28 [info] Default admin user created api_key=6T3XFDjzrFnBXlktjqqJrRMR2Eg-ApGOuv96NlgKKC4 username=admin
```

#### API Authentication
```bash
# Using Bearer token
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/data
```

#### Production Security
- Change default passwords immediately
- Use strong, randomly generated API keys
- Enable multi-factor authentication
- Configure proper SSL certificates
- Set up VPN access for sensitive environments

## Grafana Dashboard

### 1. Access Methods

#### Local Development
```bash
# Start Grafana with Docker Compose
docker-compose up -d grafana

# Access Grafana
URL: http://localhost:3000
Username: admin
Password: admin (change on first login)
```

#### Production Access
```bash
# HTTPS URL
https://grafana.your-domain.com

# Or subdirectory
https://your-domain.com/grafana
```

### 2. Dashboard Import

**Import Pre-configured Dashboards:**
1. Navigate to Grafana → Import
2. Upload JSON files from `monitoring/grafana/`:
   - `dashboard.json` - Main trading dashboard
   - `experience_storage_dashboard.json` - RL experience storage
   - `rl_production_dashboard.json` - RL production metrics

**Dashboard Configuration:**
```bash
# Setup Grafana with GCP integration
./scripts/setup_grafana_gcp.py --project-id YOUR_PROJECT_ID --grafana-url http://localhost:3000
```

### 3. Data Sources

**Prometheus Configuration:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'shyvr-rlte'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

**Google Cloud Monitoring:**
- Native GCP metrics integration
- Custom Shyvr RLTE metrics
- Automated alert policy sync

## Google Cloud Console

### 1. Access and Setup

**Console URL**: `https://console.cloud.google.com`

**Project Setup:**
```bash
# Set up GCP monitoring
./scripts/setup_gcp_monitoring.py --project-id YOUR_PROJECT_ID

# Validate setup
./scripts/validate_monitoring.py --project-id YOUR_PROJECT_ID
```

### 2. Monitoring Navigation

**Key Sections:**
- **Monitoring → Metrics Explorer**: Custom Shyvr RLTE metrics
- **Monitoring → Alerting**: Alert policies and notification channels
- **Monitoring → Dashboards**: Cloud monitoring dashboards
- **Operations → Logging**: Application and system logs

**Quick Access URLs:**
```
Metrics Explorer: https://console.cloud.google.com/monitoring/metrics-explorer
Alerting: https://console.cloud.google.com/monitoring/alerting
Dashboards: https://console.cloud.google.com/monitoring/dashboards
```

### 3. Custom Metrics Access

**Trading Metrics:**
```
custom.googleapis.com/shyvr_rlte/trading_total_pnl{currency, strategy}
custom.googleapis.com/shyvr_rlte/trading_success_rate{symbol, timeframe}
custom.googleapis.com/shyvr_rlte/trading_volume{symbol, side, dex}
```

**RL Experience Storage:**
```
custom.googleapis.com/shyvr_rlte/rl_experience_storage_rate{session_id, trading_mode}
custom.googleapis.com/shyvr_rlte/rl_experience_query_latency{operation_type}
custom.googleapis.com/shyvr_rlte/rl_database_connection_count
```

## Dashboard Navigation and Features

### 1. Web Dashboard Layout

#### Main Navigation Tabs
- **Overview**: System status, portfolio summary, key metrics
- **Trading**: Mode selection, controls, recent trades, manual trading
- **Portfolio**: Detailed positions, risk metrics, balance history
- **ML & RL**: Model status, training progress, experience storage
- **System**: Health monitoring, performance metrics, logs
- **Settings**: Risk management, dashboard preferences, user settings

#### Real-time Features
- **WebSocket Integration**: Live data updates every 5 seconds
- **Alert Notifications**: Real-time system alerts and warnings
- **Trading Activity**: Live trade execution and status updates

### 2. Key Dashboard Endpoints

#### Core Data Endpoints
```bash
# Dashboard overview data
GET /dashboard/data

# Trading status and controls
GET /dashboard/trading/status
POST /dashboard/trading/mode

# Portfolio information
GET /dashboard/portfolio/overview
GET /dashboard/portfolio/positions

# System health
GET /dashboard/system/health
GET /dashboard/system/metrics
```

#### XAI (Explainable AI) Endpoints
```bash
# XAI explanations with pagination
GET /xai/explanations?limit=10&offset=0

# Specific explanation details
GET /xai/explanations/{explanation_id}

# Feature importance analysis
GET /xai/feature-importance

# XAI cache performance
GET /xai/cache-stats
```

#### Experience Storage Endpoints
```bash
# RL experience data
GET /dashboard/experience/sessions
GET /dashboard/experience/metrics
GET /dashboard/experience/storage-stats
```

### 3. WebSocket Real-time Updates

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/dashboard/ws/your-connection-id');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    // Handle real-time updates
};
```

**Data Types:**
- Trading updates (new positions, P&L changes)
- System alerts (risk warnings, errors)
- ML/RL model updates (training progress, accuracy)
- Market data updates (price changes, volume)

## Monitoring Trading Performance

### 1. Key Performance Indicators

#### Trading Metrics Dashboard
- **Total P&L**: Current profit/loss in USD
- **Daily P&L**: Today's trading performance
- **Success Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Mean time per trade
- **Volume Metrics**: Total trading volume by symbol/DEX

#### Performance Visualization
```bash
# Access trading performance data
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/trading/performance

# Get detailed trade history
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/trading/history?limit=100
```

### 2. Risk Monitoring

#### Risk Level Indicators
- **Overall Risk Level**: 0-1 scale (0.8+ triggers warnings)
- **Position Risk**: Risk per active position
- **Portfolio Risk**: Overall portfolio volatility
- **Margin Ratio**: Available margin vs used margin
- **Correlation Risk**: Position correlation analysis

#### Risk Alerts
- **Critical (Red)**: Risk level > 0.9, immediate action required
- **Warning (Yellow)**: Risk level > 0.8, attention needed
- **Normal (Green)**: Risk level ≤ 0.8, normal operations

### 3. Trading Mode Management

#### Mode Operations
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

# Switch to Live Mode (Mode 3) - Production only
curl -X POST \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"mode": "live"}' \
     http://localhost:8080/dashboard/trading/mode
```

#### Mode Features
- **Analysis Mode**: Token discovery, ML analysis, no trading
- **Simulation Mode**: Paper trading with $10,000 virtual portfolio
- **Live Mode**: Real trading with actual funds (production only)

## ML/RL Metrics Monitoring

### 1. Machine Learning Metrics

#### Model Performance Indicators
- **Model Accuracy**: Prediction accuracy percentage
- **Confidence Scores**: Model prediction confidence
- **Feature Importance**: Top contributing features
- **Model Drift**: Performance degradation over time
- **Training Status**: Current training state and progress

#### Accessing ML Metrics
```bash
# ML model status
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/ml/status

# Feature importance data
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/xai/feature-importance

# Model training history
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/ml/training/history
```

### 2. Reinforcement Learning Metrics

#### RL Performance Metrics
- **Agent Reward**: Cumulative reward per episode
- **Exploration Rate**: Current epsilon value
- **Training Progress**: Episodes completed vs target
- **Success Rate**: Percentage of successful trading decisions
- **Experience Storage Rate**: Experiences stored per second
- **Database Performance**: Query latency and connection health

#### RL Database Monitoring
```bash
# Experience storage statistics
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/experience/storage-stats

# Training session metrics
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/experience/sessions

# Database connection health
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/system/database/health
```

### 3. XAI (Explainable AI) Integration

#### XAI Explanation Types
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **Permutation Importance**: Feature importance through permutation
- **Gradient-based**: Gradient attribution explanations

#### XAI Dashboard Features
```bash
# Recent explanations with pagination
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8080/xai/explanations?limit=10"

# Explanation details
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/xai/explanations/explanation-id-123

# XAI system performance
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/xai/cache-stats
```

## Charts and Alerts Interpretation

### 1. Chart Types and Reading

#### Performance Charts
- **Line Charts**: P&L over time, portfolio value trends
- **Bar Charts**: Trading volume, success rates by time period
- **Heatmaps**: Correlation matrices, risk distribution
- **Candlestick Charts**: Price action and trading signals
- **Histograms**: Trade size distribution, duration analysis

#### Metric Interpretation
```
P&L Chart Colors:
├── Green: Positive performance, profitable periods
├── Red: Negative performance, loss periods
└── Gray: Breakeven or no trading activity

Risk Level Gauge:
├── 0.0-0.6: Green (Safe)
├── 0.6-0.8: Yellow (Moderate)
├── 0.8-0.9: Orange (High)
└── 0.9-1.0: Red (Critical)
```

### 2. Alert System

#### Alert Levels and Response
- **INFO**: Informational messages, no action required
- **WARNING**: Attention needed, monitor closely
- **ERROR**: Problem detected, investigation required
- **CRITICAL**: Immediate action required, system at risk

#### Alert Channels
- **Dashboard**: Real-time alerts in web interface
- **Slack**: Rich formatted messages with context
- **Email**: HTML notifications with detailed information
- **Webhooks**: Custom integrations for external systems
- **Telegram**: Legacy support for mobile notifications

#### Critical Alert Examples
```
Emergency Stop Triggered:
├── Reason: High risk level detected
├── Action: All positions closed immediately
├── Response: Investigate root cause, adjust risk parameters
└── Recovery: Manual review before resuming trading

Database Connection Lost:
├── Reason: PostgreSQL connection timeout
├── Action: Switch to backup connection or degraded mode
├── Response: Check database health, restart if needed
└── Recovery: Verify data integrity after reconnection
```

### 3. Performance Thresholds

#### Trading Performance Benchmarks
```
Success Rate Thresholds:
├── Excellent: > 70%
├── Good: 60-70%
├── Average: 50-60%
├── Poor: 40-50%
└── Critical: < 40%

Daily Drawdown Limits:
├── Normal: < 5%
├── Warning: 5-15%
├── High: 15-25%
└── Critical: > 25%
```

#### System Performance Targets
```
Response Time Targets:
├── Dashboard API: < 100ms
├── XAI Endpoints: < 200ms
├── Database Queries: < 50ms
└── WebSocket Updates: < 25ms

Resource Utilization:
├── CPU Usage: < 70%
├── Memory Usage: < 80%
├── Database Connections: < 80% of pool
└── Network Bandwidth: < 50% of capacity
```

## User Management and Access Controls

### 1. User Roles and Permissions

#### Role Hierarchy
```
Admin:
├── Full system access
├── User management
├── Trading controls
├── System configuration
└── Emergency controls

Trader:
├── Trading dashboard access
├── Position management
├── Risk parameter adjustment
├── Limited system information
└── No user management

Observer:
├── Read-only dashboard access
├── Performance metrics viewing
├── Chart and alert viewing
├── No trading controls
└── No configuration changes

API User:
├── Programmatic access
├── Data export capabilities
├── Limited control endpoints
├── Rate limiting applied
└── Audit logging enabled
```

#### Permission System
```python
# Permission decorators in API
@require_read      # Basic read access
@require_write     # Modify trading parameters
@require_trading   # Execute trades
@require_admin     # Administrative functions
```

### 2. Authentication Methods

#### API Key Authentication
```bash
# Generate new API key (admin only)
curl -X POST \
     -H "Authorization: Bearer ADMIN_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"username": "new_user", "role": "trader"}' \
     http://localhost:8080/dashboard/admin/users

# Revoke API key
curl -X DELETE \
     -H "Authorization: Bearer ADMIN_API_KEY" \
     http://localhost:8080/dashboard/admin/users/api-key-to-revoke
```

#### Session Management
- **Session Timeout**: Configurable (default 8 hours)
- **Concurrent Sessions**: Limited per user
- **Session Monitoring**: Active session tracking
- **Automatic Logout**: On suspicious activity

### 3. Access Control Configuration

#### Environment Variables
```bash
# Authentication settings
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_SESSION_TIMEOUT=28800  # 8 hours
DASHBOARD_MAX_CONCURRENT_SESSIONS=3

# Role-based permissions
DASHBOARD_ADMIN_USERS=admin,supervisor
DASHBOARD_TRADER_USERS=trader1,trader2
DASHBOARD_OBSERVER_USERS=monitor,analyst
```

#### Production Security Best Practices
- Use strong, unique API keys (32+ characters)
- Enable rate limiting (100 requests/minute per user)
- Implement IP whitelisting for sensitive operations
- Regular audit of user access and permissions
- Multi-factor authentication for admin accounts
- SSL/TLS encryption for all communications

## Dashboard Troubleshooting

### 1. Common Issues and Solutions

#### Dashboard Won't Load
```bash
# Check application status
curl http://localhost:8080/health

# Check logs for errors
tail -f logs/rlte.log

# Verify dependencies
pip install -r requirements.txt

# Check port availability
lsof -i :8080
```

#### Authentication Issues
```bash
# Verify API key format
echo "YOUR_API_KEY" | wc -c  # Should be 32+ characters

# Test authentication
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/data

# Check user permissions
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/user/permissions
```

#### Performance Issues
```bash
# Check system resources
top -p $(pgrep -f "python main.py")

# Monitor database connections
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/system/database/stats

# Check WebSocket connections
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/websocket/stats
```

### 2. Database Connectivity Issues

#### PostgreSQL Connection Problems
```bash
# Test database connectivity
python -c "
from src.database.connection import get_database_connection
try:
    conn = get_database_connection()
    print('Database connection successful')
    conn.close()
except Exception as e:
    print(f'Database connection failed: {e}')
"

# Check database container (Docker)
docker ps | grep postgres
docker logs container_name

# Verify database configuration
grep -E "DATABASE_|DB_" .env.local
```

#### RL Experience Storage Issues
```bash
# Test experience storage
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/experience/test-connection

# Check storage performance
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/dashboard/experience/storage-stats

# Verify database schema
python scripts/validate_database_schema.py
```

### 3. XAI System Troubleshooting

#### XAI Explanation Issues
```bash
# Check XAI system status
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/xai/cache-stats

# Test explainer availability
python -c "
from src.xai.factory import ExplainerFactory
factory = ExplainerFactory()
print(f'Available explainers: {factory.get_available_explainers()}')
"

# Clear XAI cache if needed
curl -X POST \
     -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/xai/cache/clear
```

### 4. Monitoring System Issues

#### Grafana Connection Problems
```bash
# Test Grafana connectivity
curl http://localhost:3000/api/health

# Check Prometheus data source
curl http://localhost:3000/api/datasources

# Verify metric collection
curl http://localhost:8080/metrics | grep trading_
```

#### Google Cloud Monitoring Issues
```bash
# Test GCP credentials
gcloud auth application-default print-access-token

# Verify custom metrics
gcloud logging read "resource.type=gce_instance" --limit=10

# Check alert policies
gcloud alpha monitoring policies list
```

### 5. Performance Optimization

#### Dashboard Response Time Optimization
- Enable caching for static data
- Use pagination for large datasets
- Implement connection pooling
- Optimize database queries
- Use WebSocket for real-time data instead of polling

#### Memory Usage Optimization
- Monitor Python memory usage with profiling
- Clear unused WebSocket connections
- Implement proper cleanup for background processes
- Use appropriate cache TTL values
- Regular garbage collection monitoring

### 6. Emergency Procedures

#### System Emergency Stops
```bash
# Emergency stop all trading
curl -X POST \
     -H "Authorization: Bearer ADMIN_API_KEY" \
     http://localhost:8080/dashboard/emergency/stop-all-trading

# Check emergency stop status
curl -H "Authorization: Bearer ADMIN_API_KEY" \
     http://localhost:8080/dashboard/emergency/status
```

#### Data Backup and Recovery
```bash
# Backup critical data
python scripts/backup_critical_data.py

# Export trading history
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8080/dashboard/export/trading-history?format=csv"

# Export RL experience data
curl -H "Authorization: Bearer YOUR_API_KEY" \
     "http://localhost:8080/dashboard/export/experience-data?format=json"
```

## Best Practices

### 1. Daily Operations

#### Morning Checklist
- [ ] Check overnight system health and alerts
- [ ] Verify all dashboards are accessible
- [ ] Review trading performance from previous day
- [ ] Check database connection and performance
- [ ] Validate ML/RL model status
- [ ] Confirm risk parameters are appropriate

#### Continuous Monitoring
- Monitor real-time alerts and notifications
- Track key performance indicators
- Watch for unusual trading patterns
- Monitor system resource usage
- Check XAI explanation quality and availability

### 2. Maintenance Procedures

#### Weekly Maintenance
- Review and archive old log files
- Check database performance and optimization
- Update API keys and rotate secrets
- Review user access and permissions
- Analyze trading performance trends
- Update risk management parameters

#### Monthly Reviews
- Comprehensive system health audit
- Performance benchmark comparisons
- User access audit and cleanup
- Dashboard feature usage analysis
- Security review and updates
- Documentation updates

### 3. Security Considerations

#### Access Control
- Regular audit of user permissions
- Monitoring of API key usage
- Review of failed authentication attempts
- Network access control validation
- SSL certificate renewal tracking

#### Data Protection
- Encrypted storage of sensitive data
- Secure transmission of all communications
- Regular backup validation
- Data retention policy compliance
- Privacy regulation adherence

This comprehensive guide provides operators and administrators with everything needed to effectively access, navigate, and operate the Shyvr RLTE dashboard system across all its components and interfaces.