# Shyvr RLTE Monitoring and Alerting System

This document provides a comprehensive overview of the monitoring and alerting system implemented for the Shyvr RLTE trading bot.

## Overview

The monitoring system provides real-time visibility into trading performance, risk management, and system health through multiple integrated platforms:

- **Google Cloud Monitoring**: Native GCP monitoring with custom metrics and alerting
- **Prometheus Metrics Collection**: Comprehensive metrics exposed via `/metrics` endpoint
- **Grafana Dashboards**: Visual monitoring with pre-built dashboards integrated with GCP
- **Multi-Channel Alerting**: Slack, email, webhook, and Telegram notifications
- **Safety Monitoring**: Risk level tracking and emergency detection
- **Performance Tracking**: Trading success rates, P&L, and volume monitoring
- **ML/RL Model Monitoring**: Machine learning and reinforcement learning performance tracking

## Architecture

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Trading Bot       │───▶│ Google Cloud     │───▶│    Alert Policies   │
│   - Trading Engine  │    │ Monitoring       │    │    - Critical       │
│   - ML/RL Models    │    │ - Custom Metrics │    │    - Warnings       │
│   - Risk Manager    │    │ - Time Series    │    │    - Thresholds     │
└─────────────────────┘    └──────────────────┘    └─────────────────────┘
         │                            │                        │
         ▼                            ▼                        ▼
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Prometheus Metrics  │    │ Grafana          │    │ Notification        │
│ - Local Collection  │───▶│ Dashboards       │    │ Channels            │
│ - /metrics endpoint │    │ - GCP Integration│    │ - Slack             │
│ - Health Checks     │    │ - Real-time      │    │ - Email             │
└─────────────────────┘    └──────────────────┘    │ - Webhooks          │
                                                    │ - Telegram          │
                                                    └─────────────────────┘
```

## Components

### 1. Google Cloud Monitoring Integration

#### Cloud Monitoring Features
The system integrates natively with Google Cloud Monitoring to provide:

- **Custom Metrics**: Trading-specific metrics sent directly to Cloud Monitoring
- **Automated Alerting**: GCP-native alert policies with sophisticated conditions
- **Scalable Infrastructure**: Cloud-based monitoring that scales with trading volume
- **Integration with GCP Services**: Native integration with other Google Cloud services

#### Custom Metric Categories

**Trading Performance Metrics:**
```
custom.googleapis.com/shyvr_rlte/trading_total_pnl{currency, strategy}
custom.googleapis.com/shyvr_rlte/trading_daily_pnl{currency, date}
custom.googleapis.com/shyvr_rlte/trading_volume{symbol, side, dex}
custom.googleapis.com/shyvr_rlte/trading_success_rate{symbol, timeframe}
custom.googleapis.com/shyvr_rlte/trading_position_count{symbol, side}
```

**Risk Management Metrics:**
```
custom.googleapis.com/shyvr_rlte/safety_risk_level{risk_type, timeframe}
custom.googleapis.com/shyvr_rlte/safety_emergency_stops{reason, trigger}
custom.googleapis.com/shyvr_rlte/safety_liquidations{symbol, side, reason}
custom.googleapis.com/shyvr_rlte/safety_drawdown_percentage{timeframe}
custom.googleapis.com/shyvr_rlte/safety_margin_ratio{account}
```

**System Health Metrics:**
```
custom.googleapis.com/shyvr_rlte/system_uptime{component}
custom.googleapis.com/shyvr_rlte/system_response_time{operation, endpoint}
custom.googleapis.com/shyvr_rlte/system_error_rate{error_type, component}
```

**ML/RL Performance Metrics:**
```
custom.googleapis.com/shyvr_rlte/ml_model_accuracy{model_type, timeframe}
custom.googleapis.com/shyvr_rlte/ml_prediction_confidence{model_type, symbol}
custom.googleapis.com/shyvr_rlte/rl_agent_reward{agent_id, episode}
custom.googleapis.com/shyvr_rlte/rl_exploration_rate{agent_id}
```

**RL Experience Storage Metrics:**
```
custom.googleapis.com/shyvr_rlte/rl_experience_storage_rate{session_id, trading_mode}
custom.googleapis.com/shyvr_rlte/rl_experience_total_count{session_id}
custom.googleapis.com/shyvr_rlte/rl_experience_query_latency{operation_type}
custom.googleapis.com/shyvr_rlte/rl_training_session_duration{session_id, status}
custom.googleapis.com/shyvr_rlte/rl_database_connection_count
custom.googleapis.com/shyvr_rlte/rl_experience_batch_size{operation}
```

**DEX and Wallet Metrics:**
```
custom.googleapis.com/shyvr_rlte/dex_connection_status{dex, chain}
custom.googleapis.com/shyvr_rlte/wallet_balance{token, chain, wallet_type}
```

#### Alert Policies

**Critical Alerts (Immediate Response Required):**
- Trading system down (uptime < 1 for >60s)
- Critical risk level (risk_level > 0.9 for >120s)
- Emergency stop triggered (any emergency stop event)
- Large daily loss (daily_pnl < -$1000 immediately)
- Low margin ratio (margin_ratio < 1.5 for >120s)

**Warning Alerts (Attention Required):**
- High risk level (risk_level > 0.8 for >300s)
- High drawdown (drawdown > 15% for >300s)
- Low trading success rate (success_rate < 40% for >600s)
- ML model accuracy degradation (accuracy < 65% for >900s)
- RL experience storage slow queries (query_latency > 100ms for >300s)
- High database connection usage (connection_count > 15 for >180s)
- RL training session failures (session failure rate > 20% for >600s)
- DEX connection issues (connection_status < 1 for >180s)
- High system error rate (error_rate > 10/min for >300s)

#### Cloud Monitoring Setup

**Automated Setup:**
```bash
# Run the complete monitoring setup
./deploy/setup_monitoring.sh -p YOUR_PROJECT_ID -a alerts@yourcompany.com

# Validate the setup
./scripts/validate_monitoring.py --project-id YOUR_PROJECT_ID
```

**Manual Configuration:**
```bash
# Set up GCP monitoring only
uv run scripts/setup_gcp_monitoring.py --project-id YOUR_PROJECT_ID

# Set up Grafana integration
uv run scripts/setup_grafana_gcp.py --project-id YOUR_PROJECT_ID --grafana-url http://localhost:3000
```

### 2. Multi-Channel Notification System

#### Supported Notification Channels

**Slack Integration:**
- Rich message formatting with colors and attachments
- Channel-specific routing
- Rate limiting to prevent spam
- Emoji and formatting support

**Email Notifications:**
- HTML and plain text formats
- Priority-based recipient lists
- Template-based message formatting
- SMTP configuration support

**Webhook Integration:**
- Custom webhook endpoints
- Flexible payload formatting (JSON, form data)
- Authentication support (Bearer token, API key)
- Retry logic with exponential backoff

**Telegram Bot (Legacy):**
- Maintained for backward compatibility
- HTML formatted messages
- Chat-based notifications

#### Notification Features

**Deduplication:**
- Prevents duplicate alerts within configurable time windows
- Hash-based message identification
- Per-channel deduplication tracking

**Rate Limiting:**
- Configurable limits per alert level
- Hour-based rate limiting windows
- Automatic backoff when limits exceeded

**Delivery Status Tracking:**
- Success/failure tracking per channel
- Retry mechanisms for failed deliveries
- Health monitoring of notification channels

### 3. Prometheus Metrics Collection System

#### Base Components
- **MetricsRegistry**: Custom Prometheus registry for isolated metrics
- **MetricsCollector**: Abstract base class for all collectors
- **Recording Decorators**: Automatic execution time tracking

#### Collectors

##### Trading Metrics Collector
Tracks key trading performance indicators:

- **P&L Metrics**: Total and daily profit/loss tracking
- **Volume Metrics**: Trading volume by symbol and side
- **Success Rate**: Percentage of successful trades
- **Position Count**: Number of active positions
- **Trade Distributions**: Size and duration histograms

**Key Metrics:**
```
trading_total_pnl{currency}
trading_daily_pnl{currency,date}
trading_volume_total{symbol,side}
trading_success_rate{symbol,timeframe}
trading_position_count{symbol}
trading_trade_size_seconds{symbol,side}
trading_trade_duration_seconds{symbol}
```

##### Safety Metrics Collector
Monitors risk and safety systems:

- **Risk Level**: Overall system risk (0-1 scale)
- **Emergency Stops**: Critical system shutdown events
- **Liquidations**: Position liquidation tracking
- **Drawdown**: Portfolio drawdown monitoring
- **Risk Violations**: Threshold violation counts

**Key Metrics:**
```
safety_risk_level{risk_type,timeframe}
safety_emergency_stops_total{reason,trigger}
safety_liquidations_total{symbol,side,reason}
safety_drawdown_percentage{timeframe}
safety_margin_ratio{account}
safety_volatility_score{timeframe}
```

##### RL Experience Storage Metrics Collector
Monitors database-backed RL experience storage performance:

- **Storage Rate**: Experiences stored per second by session and mode
- **Query Performance**: Database query latency for different operations
- **Connection Health**: Database connection pool usage and health
- **Session Tracking**: Training session duration and completion rates
- **Data Volume**: Total experience count and growth trends
- **Batch Operations**: Batch processing performance and throughput

**Key Metrics:**
```
rl_experience_storage_rate_per_second{session_id,trading_mode}
rl_experience_total_count{session_id}
rl_experience_query_latency_seconds{operation_type}
rl_training_session_duration_seconds{session_id,status}
rl_database_connection_count
rl_experience_batch_size{operation}
rl_database_query_error_rate{error_type}
rl_experience_retrieval_success_rate{batch_size}
```

### 2. Alerting System

#### Alert Levels
- **INFO**: Informational messages
- **WARNING**: Issues requiring attention
- **ERROR**: Error conditions
- **CRITICAL**: Immediate action required

#### Telegram Integration
- Queue-based alert processing
- HTML formatted messages
- Configurable alert levels
- Emergency stop notifications
- Risk threshold alerts

#### Alert Types
- **Trading Alerts**: Performance degradation, losses
- **Risk Alerts**: High risk levels, margin calls
- **Safety Alerts**: Emergency stops, liquidations
- **System Alerts**: Application health, connectivity

### 3. Dashboard System

#### Grafana Dashboard
Pre-configured dashboard with panels for:

**Trading Performance**
- Total and daily P&L displays
- Success rate gauges
- Volume and trade count metrics
- Position tracking

**Risk Management**
- Risk level gauges with color-coded thresholds
- Drawdown tracking
- Safety event timelines
- Margin ratio monitoring

**System Health**
- Application uptime
- Response times
- Error rates
- Database performance

**RL Experience Storage**
- Experience storage rate trends
- Database query latency histograms
- Connection pool utilization
- Training session completion rates
- Experience batch processing metrics
- Database error rate tracking

#### Prometheus Configuration
- 30-second scrape intervals
- Alert rule definitions
- Data retention policies
- Target configuration

## Configuration

### Monitoring Settings

```yaml
monitoring:
  enabled: true
  prometheus:
    enabled: true
    metrics_path: "/metrics"
    collection_interval: 30
    
  alerting:
    telegram:
      enabled: true
      bot_token: "${TELEGRAM_MONITORING_BOT_TOKEN}"
      chat_id: "${TELEGRAM_MONITORING_CHAT_ID}"
      
  thresholds:
    trading:
      max_daily_loss_pct: 20.0
      min_success_rate: 0.3
    safety:
      critical_risk_level: 0.9
      high_risk_level: 0.8
```

### Environment Variables

Required for comprehensive monitoring functionality:

```bash
# Google Cloud Platform
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# Alert Email Addresses
ALERT_EMAIL_CRITICAL=critical-alerts@yourcompany.com
ALERT_EMAIL_WARNING=warning-alerts@yourcompany.com

# Slack Integration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CHANNEL=#trading-alerts

# Email SMTP Configuration
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USER=alerts@yourcompany.com
ALERT_SMTP_PASSWORD=your-app-password
ALERT_FROM_EMAIL=shyvr-alerts@yourcompany.com

# Webhook Integration
ALERT_WEBHOOK_URL=https://your-webhook-endpoint.com/alerts
WEBHOOK_AUTH_TOKEN=your-webhook-token

# Grafana Integration
GRAFANA_URL=http://localhost:3000
GRAFANA_API_KEY=your_grafana_api_key
GRAFANA_ORG=Main Org.

# Telegram Alerting (Legacy)
TELEGRAM_MONITORING_BOT_TOKEN=your_bot_token
TELEGRAM_MONITORING_CHAT_ID=your_chat_id

# Prometheus Push Gateway (optional)
PROMETHEUS_PUSH_GATEWAY=http://localhost:9091
```

## Usage

### Starting the Monitoring System

The monitoring system is automatically initialized when the application starts:

```python
from src.monitoring import (
    MetricsRegistry, 
    TradingMetricsCollector, 
    SafetyMetricsCollector,
    AlertingService
)

# Initialize components
metrics_registry = MetricsRegistry()
trading_metrics = TradingMetricsCollector(metrics_registry)
safety_metrics = SafetyMetricsCollector(metrics_registry)
alerting_service = AlertingService(telegram_config)
```

### Recording Metrics

#### Trading Events
```python
# Record successful trade
trading_metrics.record_trade_success({
    'symbol': 'BTC/USD',
    'side': 'buy',
    'size': Decimal('1000.00'),
    'duration': 300.5
})

# Update P&L
trading_metrics.record_pnl_update(
    total_pnl=Decimal('2500.00'),
    daily_pnl=Decimal('150.00')
)
```

#### Safety Events
```python
# Record emergency stop
safety_metrics.record_emergency_stop({
    'reason': 'high_risk',
    'trigger': 'drawdown_limit',
    'positions_closed': 3
})

# Update risk level
safety_metrics.update_risk_level(0.85, 'overall')
```

### Sending Alerts

#### Direct Alerts
```python
# Critical alert
await alerting_service.send_critical_alert(
    "Emergency stop triggered",
    "risk_manager",
    metadata={"risk_level": 0.95}
)

# Warning alert
await alerting_service.send_warning_alert(
    "High drawdown detected",
    "portfolio_manager"
)
```

#### Alert Objects
```python
from src.monitoring import Alert, AlertLevel

alert = Alert(
    level=AlertLevel.ERROR,
    message="API connection failed",
    source="dex_client",
    metadata={"endpoint": "/api/v1/quote"}
)

await alerting_service.send_alert(alert)
```

## Setup Instructions

### 1. Install Dependencies

```bash
uv add prometheus_client
```

### 2. Configure Telegram Bot

1. Create a new bot with [@BotFather](https://t.me/botfather)
2. Get the bot token
3. Add bot to your monitoring channel
4. Get the chat ID
5. Set environment variables

### 3. Set Up Grafana and Prometheus

#### Using Docker Compose

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/grafana/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/grafana/alert_rules.yml:/etc/prometheus/alert_rules.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

Start the stack:
```bash
docker-compose up -d
```

### 4. Import Dashboard

1. Open Grafana at http://localhost:3000
2. Login with admin/admin
3. Import `monitoring/grafana/dashboard.json`
4. Configure Prometheus data source

### 5. Test the System

```bash
# Check metrics endpoint
curl http://localhost:8080/metrics

# Send test alert
python -c "
import asyncio
from src.monitoring import AlertingService, AlertLevel

async def test():
    config = {'enabled': True, 'bot_token': 'your_token', 'chat_id': 'your_chat'}
    service = AlertingService(telegram_config=config)
    await service.send_alert(AlertLevel.INFO, 'Test alert', 'test_system')

asyncio.run(test())
"
```

## Alert Rules

### Critical Alerts (Immediate Response)
- Emergency stop triggered
- Risk level > 90%
- Daily drawdown > 25%
- Trading bot down
- Margin ratio < 1.2

### Warning Alerts (Attention Required)
- Risk level > 80%
- Daily drawdown > 15%
- Success rate < 40%
- High rate of liquidations
- No trading activity for 30+ minutes

### Info Alerts (Informational)
- System startup/shutdown
- Configuration changes
- Periodic health reports

## Troubleshooting

### Common Issues

#### No Metrics Data
- Check `/metrics` endpoint accessibility
- Verify Prometheus scraping configuration
- Ensure trading bot is running

#### Alerts Not Sending
- Verify Telegram bot token and chat ID
- Check network connectivity
- Review alerting service logs

#### Dashboard Not Loading
- Confirm Grafana-Prometheus connection
- Check dashboard JSON import
- Verify time range settings

### Debug Commands

```bash
# Check metrics collection
curl -s http://localhost:8080/metrics | grep trading_

# Test Prometheus connectivity
curl -s http://localhost:9090/api/v1/targets

# Verify Grafana data source
curl -s http://admin:admin@localhost:3000/api/datasources
```

## Development

### Adding New Metrics

1. Create metric in collector class:
```python
self.new_metric = registry.get_counter(
    "new_metric_total",
    "Description of new metric",
    ["label1", "label2"]
)
```

2. Update metric in collect_metrics():
```python
def collect_metrics(self):
    # Collect data
    value = self._get_metric_value()
    self.new_metric.labels(label1="value1", label2="value2").inc(value)
```

3. Add to metric definitions:
```python
def get_metric_definitions(self):
    return {
        "new_metric_total": "Description of new metric"
    }
```

4. Add tests:
```python
def test_new_metric_collection(self):
    collector = MetricsCollector(registry)
    collector.collect_metrics()
    # Assert metric was updated
```

### Creating Custom Collectors

```python
from src.monitoring.base import MetricsCollector

class CustomMetricsCollector(MetricsCollector):
    def __init__(self, registry):
        super().__init__(registry)
        self.custom_gauge = registry.get_gauge(
            "custom_metric",
            "Custom metric description",
            ["type"]
        )
    
    def collect_metrics(self):
        # Implement collection logic
        value = self._get_custom_value()
        self.custom_gauge.labels(type="example").set(value)
    
    def get_metric_definitions(self):
        return {
            "custom_metric": "Custom metric description"
        }
```

## Best Practices

### Metric Design
- Use consistent naming conventions
- Include appropriate labels for filtering
- Choose correct metric types (counter, gauge, histogram)
- Document all metrics clearly

### Alerting
- Set appropriate thresholds for your use case
- Avoid alert fatigue with too many notifications
- Use different severity levels appropriately
- Test alert delivery regularly

### Performance
- Monitor metrics collection overhead
- Use appropriate collection intervals
- Implement proper error handling
- Cache expensive computations

### Security
- Secure Grafana access
- Protect Telegram bot tokens
- Use HTTPS for external access
- Implement proper authentication

## Monitoring the Monitors

The monitoring system includes self-monitoring capabilities:

- Metrics collection performance tracking
- Alert delivery success monitoring
- Dashboard availability checking
- Automatic health status reporting

This ensures the monitoring system itself remains reliable and provides visibility into its own operation.