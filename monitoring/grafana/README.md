# Grafana Dashboard Configuration for Shyvr RLTE

This directory contains monitoring and alerting configurations for the Shyvr RLTE trading bot.

## Files

- `dashboard.json` - Grafana dashboard configuration with comprehensive trading and safety metrics
- `prometheus.yml` - Prometheus scraping configuration
- `alert_rules.yml` - Prometheus alerting rules for critical events
- `README.md` - This documentation file

## Setup Instructions

### 1. Prerequisites

- Docker and Docker Compose
- Grafana v9.0+
- Prometheus v2.30+

### 2. Quick Start with Docker Compose

Create a `docker-compose.monitoring.yml` file in your project root:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: shyvr-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/grafana/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/grafana/alert_rules.yml:/etc/prometheus/alert_rules.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: shyvr-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### 3. Start Monitoring Stack

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### 4. Import Dashboard

1. Open Grafana at http://localhost:3000
2. Login with admin/admin
3. Go to Dashboards > Import
4. Upload the `dashboard.json` file
5. Configure Prometheus data source (http://prometheus:9090)

### 5. Configure Alerting

The alert rules are automatically loaded by Prometheus. To receive notifications:

1. Set up Alertmanager (optional)
2. Configure notification channels in Grafana
3. Enable Telegram alerts in your bot configuration

## Dashboard Panels

### Trading Performance
- **Total P&L**: Current profit/loss in USD
- **Daily P&L**: Today's profit/loss
- **Success Rate**: Percentage of successful trades
- **Trading Volume**: Volume over time
- **Active Positions**: Number of open positions

### Risk Management
- **Risk Level**: Current overall risk level (0-1)
- **Drawdown**: Current drawdown percentage
- **Safety Events**: Emergency stops, liquidations, violations
- **Margin Ratio**: Available vs used margin

### System Health
- **Trade Distributions**: Size and duration histograms
- **Performance Metrics**: Success rates and volumes over time

## Alerting Rules

### Critical Alerts
- **Emergency Stop Triggered**: Immediate notification
- **Critical Risk Level**: Risk > 90%
- **Trading Bot Down**: Service unavailable
- **Excessive Drawdown**: Daily drawdown > 25%
- **Low Margin Ratio**: Margin ratio < 1.2

### Warning Alerts
- **High Risk Level**: Risk > 80%
- **High Drawdown**: Daily drawdown > 15%
- **Low Success Rate**: Success rate < 40%
- **No Trading Activity**: No trades for 30 minutes
- **Liquidation Occurred**: Position liquidated

## Customization

### Adding New Metrics

1. Add metrics to your collectors in `src/monitoring/`
2. Update dashboard panels to display new metrics
3. Add relevant alert rules if needed

### Modifying Alert Thresholds

Edit the `alert_rules.yml` file to adjust:
- Alert thresholds (`expr` values)
- Alert duration (`for` values)
- Severity levels and components

### Dashboard Customization

The dashboard JSON can be modified to:
- Add new panels
- Change visualization types
- Adjust time ranges and refresh intervals
- Modify color schemes and thresholds

## Metrics Reference

### Trading Metrics
- `trading_total_pnl{currency}` - Total profit/loss
- `trading_daily_pnl{currency,date}` - Daily profit/loss
- `trading_volume_total{symbol,side}` - Trading volume
- `trading_success_rate{symbol,timeframe}` - Success rate
- `trading_position_count{symbol}` - Active positions

### Safety Metrics
- `safety_risk_level{risk_type,timeframe}` - Risk level (0-1)
- `safety_emergency_stops_total{reason,trigger}` - Emergency stops
- `safety_liquidations_total{symbol,side,reason}` - Liquidations
- `safety_drawdown_percentage{timeframe}` - Drawdown percentage
- `safety_margin_ratio{account}` - Margin ratio

## Troubleshooting

### Dashboard Not Loading
- Check Prometheus data source configuration
- Verify metrics endpoint is accessible at `/metrics`
- Ensure trading bot is running and exposing metrics

### No Data in Panels
- Check Prometheus scraping configuration
- Verify metric names match dashboard queries
- Check time range settings

### Alerts Not Firing
- Verify alert rule syntax in Prometheus
- Check alert evaluation intervals
- Ensure thresholds are appropriate for your setup

## Production Considerations

### Security
- Change default Grafana admin password
- Enable authentication and authorization
- Use HTTPS for external access
- Secure Prometheus with authentication

### Scaling
- Consider using external storage for metrics retention
- Set up high availability for Prometheus and Grafana
- Implement proper backup strategies

### Monitoring the Monitors
- Monitor Prometheus and Grafana health
- Set up external health checks
- Implement redundant alerting channels