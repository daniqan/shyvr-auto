# Shyvr RLTE Monitoring and Alerting System

This document provides a comprehensive overview of the monitoring and alerting system implemented for the Shyvr RLTE trading bot.

## Overview

The monitoring system provides real-time visibility into trading performance, risk management, and system health through:

- **Prometheus Metrics Collection**: Comprehensive metrics exposed via `/metrics` endpoint
- **Grafana Dashboards**: Visual monitoring with pre-built dashboards
- **Telegram Alerting**: Real-time notifications for critical events
- **Safety Monitoring**: Risk level tracking and emergency detection
- **Performance Tracking**: Trading success rates, P&L, and volume monitoring

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Trading Bot   │───▶│  Prometheus  │───▶│    Grafana      │
│                 │    │   Metrics    │    │   Dashboard     │
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────┐                        ┌─────────────────┐
│ Alerting System │                        │   Monitoring    │
│  (Telegram Bot) │                        │   Operators     │
└─────────────────┘                        └─────────────────┘
```

## Components

### 1. Metrics Collection System

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

Required for full functionality:

```bash
# Telegram Alerting
TELEGRAM_MONITORING_BOT_TOKEN=your_bot_token
TELEGRAM_MONITORING_CHAT_ID=your_chat_id

# Grafana Integration (optional)
GRAFANA_DASHBOARD_URL=http://localhost:3000
GRAFANA_API_KEY=your_api_key

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