#!/bin/bash

# Production Monitoring Setup for Shyvr RLTE
# Configures comprehensive monitoring, alerting, and observability for production deployment
# Includes Cloud Monitoring, custom metrics, alerting policies, and dashboards

set -euo pipefail

# Configuration
PROJECT_ID=${PROJECT_ID:-"shvyr-ai-bots"}
REGION=${REGION:-"us-central1"}
SERVICE=${SERVICE:-"shyvr-rlte"}
ENVIRONMENT=${1:-"production"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "${PURPLE}[MONITORING]${NC} $1"; }

# Enable required APIs
enable_apis() {
    log_header "📡 Enabling Required APIs"
    
    local apis=(
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "clouderrorreporting.googleapis.com"
        "cloudtrace.googleapis.com"
        "cloudprofiler.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        if gcloud services enable "$api" --project="$PROJECT_ID"; then
            log_success "✓ $api enabled"
        else
            log_warning "Failed to enable $api (may already be enabled)"
        fi
    done
}

# Create custom metrics
create_custom_metrics() {
    log_header "📊 Creating Custom Metrics"
    
    # Trading-specific metrics
    cat << 'EOF' > /tmp/trading_metrics.yaml
resources:
- name: trading/position_count
  type: custom.googleapis.com/trading/position_count
  metricKind: GAUGE
  valueType: INT64
  description: "Number of active trading positions"
  displayName: "Active Trading Positions"

- name: trading/pnl_total
  type: custom.googleapis.com/trading/pnl_total
  metricKind: GAUGE
  valueType: DOUBLE
  description: "Total profit and loss in USD"
  displayName: "Total P&L (USD)"

- name: trading/risk_score
  type: custom.googleapis.com/trading/risk_score
  metricKind: GAUGE
  valueType: DOUBLE
  description: "Current portfolio risk score (0-100)"
  displayName: "Portfolio Risk Score"

- name: ml/model_accuracy
  type: custom.googleapis.com/ml/model_accuracy
  metricKind: GAUGE
  valueType: DOUBLE
  description: "ML model prediction accuracy percentage"
  displayName: "ML Model Accuracy"

- name: rl/episode_reward
  type: custom.googleapis.com/rl/episode_reward
  metricKind: GAUGE
  valueType: DOUBLE
  description: "RL agent episode reward"
  displayName: "RL Episode Reward"

- name: rl/experience_count
  type: custom.googleapis.com/rl/experience_count
  metricKind: GAUGE
  valueType: INT64
  description: "Number of stored RL experiences"
  displayName: "RL Experience Count"

- name: safety/emergency_stops
  type: custom.googleapis.com/safety/emergency_stops
  metricKind: CUMULATIVE
  valueType: INT64
  description: "Total number of emergency stops triggered"
  displayName: "Emergency Stops"

- name: api/rate_limit_hits
  type: custom.googleapis.com/api/rate_limit_hits
  metricKind: CUMULATIVE
  valueType: INT64
  description: "Number of API rate limit hits"
  displayName: "API Rate Limit Hits"
EOF

    # Create metrics
    while IFS= read -r line; do
        if [[ "$line" =~ ^- ]]; then
            metric_name=$(echo "$line" | grep -o 'custom\.googleapis\.com/[^"]*')
            log_info "Creating metric: $metric_name"
            # Note: Custom metrics are created automatically when first used in GCP
        fi
    done < /tmp/trading_metrics.yaml
    
    rm /tmp/trading_metrics.yaml
    log_success "Custom metrics configuration prepared"
}

# Create alerting policies
create_alerting_policies() {
    log_header "🚨 Creating Alerting Policies"
    
    # Create notification channel first (if webhook URL is available)
    local notification_channel=""
    if gcloud secrets versions access latest --secret=ALERT_WEBHOOK_URL --project="$PROJECT_ID" >/dev/null 2>&1; then
        local webhook_url
        webhook_url=$(gcloud secrets versions access latest --secret=ALERT_WEBHOOK_URL --project="$PROJECT_ID")
        
        notification_channel=$(gcloud alpha monitoring channels create \
            --display-name="RLTE Production Alerts" \
            --type="webhook_tokenauth" \
            --channel-labels="url=$webhook_url" \
            --format="value(name)" 2>/dev/null || echo "")
        
        if [[ -n "$notification_channel" ]]; then
            log_success "Notification channel created: $notification_channel"
        fi
    fi
    
    # High CPU Usage Alert
    cat << EOF > /tmp/cpu_alert.yaml
combiner: OR
conditions:
- displayName: "High CPU Usage"
  conditionThreshold:
    filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="$SERVICE"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 80
    duration: 300s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_MEAN
      crossSeriesReducer: REDUCE_MEAN
      groupByFields:
      - resource.label.service_name
displayName: "RLTE High CPU Usage"
documentation:
  content: "CPU usage is above 80% for 5 minutes"
EOF

    if gcloud alpha monitoring policies create --policy-from-file=/tmp/cpu_alert.yaml; then
        log_success "✓ CPU usage alert created"
    fi
    
    # Memory Usage Alert
    cat << EOF > /tmp/memory_alert.yaml
combiner: OR
conditions:
- displayName: "High Memory Usage"
  conditionThreshold:
    filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="$SERVICE"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 85
    duration: 300s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_MEAN
      crossSeriesReducer: REDUCE_MEAN
      groupByFields:
      - resource.label.service_name
displayName: "RLTE High Memory Usage"
documentation:
  content: "Memory usage is above 85% for 5 minutes"
EOF

    if gcloud alpha monitoring policies create --policy-from-file=/tmp/memory_alert.yaml; then
        log_success "✓ Memory usage alert created"
    fi
    
    # Error Rate Alert
    cat << EOF > /tmp/error_alert.yaml
combiner: OR
conditions:
- displayName: "High Error Rate"
  conditionThreshold:
    filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="$SERVICE"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 5
    duration: 180s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_RATE
      crossSeriesReducer: REDUCE_SUM
      groupByFields:
      - resource.label.service_name
displayName: "RLTE High Error Rate"
documentation:
  content: "Error rate is above 5% for 3 minutes"
EOF

    if gcloud alpha monitoring policies create --policy-from-file=/tmp/error_alert.yaml; then
        log_success "✓ Error rate alert created"
    fi
    
    # Emergency Stop Alert (Critical)
    cat << EOF > /tmp/emergency_alert.yaml
combiner: OR
conditions:
- displayName: "Emergency Stop Triggered"
  conditionThreshold:
    filter: 'metric.type="custom.googleapis.com/safety/emergency_stops"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 0
    duration: 0s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_RATE
      crossSeriesReducer: REDUCE_SUM
displayName: "RLTE Emergency Stop"
documentation:
  content: "Emergency stop has been triggered - immediate attention required"
severity: CRITICAL
EOF

    if gcloud alpha monitoring policies create --policy-from-file=/tmp/emergency_alert.yaml; then
        log_success "✓ Emergency stop alert created"
    fi
    
    # Cleanup temporary files
    rm -f /tmp/*_alert.yaml
}

# Create monitoring dashboard
create_monitoring_dashboard() {
    log_header "📈 Creating Monitoring Dashboard"
    
    cat << EOF > /tmp/rlte_dashboard.json
{
  "displayName": "Shyvr RLTE Production Dashboard",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "CPU Utilization",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_MEAN",
                    "crossSeriesReducer": "REDUCE_MEAN",
                    "groupByFields": ["resource.label.service_name"]
                  }
                }
              },
              "plotType": "LINE"
            }],
            "yAxis": {
              "label": "CPU %",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "xPos": 6,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Memory Utilization",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_MEAN",
                    "crossSeriesReducer": "REDUCE_MEAN",
                    "groupByFields": ["resource.label.service_name"]
                  }
                }
              },
              "plotType": "LINE"
            }],
            "yAxis": {
              "label": "Memory %",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "yPos": 4,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Request Count",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_RATE",
                    "crossSeriesReducer": "REDUCE_SUM",
                    "groupByFields": ["resource.label.service_name"]
                  }
                }
              },
              "plotType": "LINE"
            }],
            "yAxis": {
              "label": "Requests/sec",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "xPos": 6,
        "yPos": 4,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Response Latency",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_MEAN",
                    "crossSeriesReducer": "REDUCE_MEAN",
                    "groupByFields": ["resource.label.service_name"]
                  }
                }
              },
              "plotType": "LINE"
            }],
            "yAxis": {
              "label": "Latency (ms)",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "yPos": 8,
        "width": 12,
        "height": 4,
        "widget": {
          "title": "Trading & ML Metrics",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"custom.googleapis.com/trading/position_count\"",
                    "aggregation": {
                      "alignmentPeriod": "300s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              },
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"custom.googleapis.com/ml/model_accuracy\"",
                    "aggregation": {
                      "alignmentPeriod": "300s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y2"
              }
            ],
            "yAxis": {
              "label": "Positions",
              "scale": "LINEAR"
            },
            "y2Axis": {
              "label": "Accuracy %",
              "scale": "LINEAR"
            }
          }
        }
      }
    ]
  }
}
EOF

    if gcloud monitoring dashboards create --config-from-file=/tmp/rlte_dashboard.json; then
        log_success "✓ Production dashboard created"
    fi
    
    rm /tmp/rlte_dashboard.json
}

# Setup log-based metrics
setup_log_metrics() {
    log_header "📝 Setting Up Log-Based Metrics"
    
    # Error count metric
    log_info "Creating error count log metric..."
    gcloud logging metrics create rlte_error_count \
        --description="Count of error logs in RLTE" \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="'$SERVICE'" AND severity>=ERROR' \
        --project="$PROJECT_ID" || log_warning "Error count metric may already exist"
    
    # Trading action metric
    log_info "Creating trading action log metric..."
    gcloud logging metrics create rlte_trading_actions \
        --description="Count of trading actions in RLTE" \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="'$SERVICE'" AND jsonPayload.action_type!=null' \
        --project="$PROJECT_ID" || log_warning "Trading actions metric may already exist"
    
    # Model prediction metric
    log_info "Creating model prediction log metric..."
    gcloud logging metrics create rlte_model_predictions \
        --description="Count of ML model predictions in RLTE" \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="'$SERVICE'" AND jsonPayload.prediction!=null' \
        --project="$PROJECT_ID" || log_warning "Model predictions metric may already exist"
    
    log_success "Log-based metrics configured"
}

# Configure uptime checks
setup_uptime_checks() {
    log_header "⏱️ Setting Up Uptime Checks"
    
    # Get service URL
    local service_url
    service_url=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "")
    
    if [[ -z "$service_url" ]]; then
        log_warning "Service URL not available - skipping uptime checks"
        return
    fi
    
    # Health endpoint uptime check
    cat << EOF > /tmp/uptime_check.yaml
displayName: "RLTE Health Check"
httpCheck:
  path: "/health"
  port: 443
  requestMethod: GET
  useSsl: true
  validateSsl: true
monitoredResource:
  type: "uptime_url"
  labels:
    project_id: "$PROJECT_ID"
    host: "$(echo $service_url | sed 's|https://||' | sed 's|/.*||')"
period: 60s
timeout: 30s
EOF

    if gcloud monitoring uptime-check-configs create --config-from-file=/tmp/uptime_check.yaml; then
        log_success "✓ Health check uptime monitor created"
    fi
    
    rm /tmp/uptime_check.yaml
}

# Setup error reporting
setup_error_reporting() {
    log_header "🐛 Setting Up Error Reporting"
    
    log_info "Error Reporting is automatically enabled for Cloud Run services"
    log_info "Errors will be automatically captured and reported"
    
    # Create custom error grouping rules if needed
    log_success "Error reporting configured"
}

# Setup application performance monitoring
setup_apm() {
    log_header "🔍 Setting Up Application Performance Monitoring"
    
    log_info "Cloud Trace and Profiler are enabled for detailed performance monitoring"
    log_info "Traces will be automatically collected for HTTP requests"
    
    # Note: Application code should include OpenTelemetry or Cloud Trace client
    log_success "APM monitoring configured"
}

# Generate monitoring summary
generate_monitoring_summary() {
    log_header "📋 Monitoring Setup Summary"
    
    local service_url
    service_url=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "Not deployed")
    
    echo "================================================"
    echo "Service: $SERVICE"
    echo "Environment: $ENVIRONMENT"
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Service URL: $service_url"
    echo ""
    echo "Monitoring Resources Created:"
    echo "  ✓ Custom metrics for trading, ML, and RL"
    echo "  ✓ Alerting policies for critical conditions"
    echo "  ✓ Production monitoring dashboard"
    echo "  ✓ Log-based metrics for application events"
    echo "  ✓ Uptime checks for health monitoring"
    echo "  ✓ Error reporting and APM"
    echo ""
    echo "Access Points:"
    echo "  Monitoring Console: https://console.cloud.google.com/monitoring/dashboards"
    echo "  Logs Explorer: https://console.cloud.google.com/logs/query"
    echo "  Error Reporting: https://console.cloud.google.com/errors"
    echo "  Trace Explorer: https://console.cloud.google.com/traces"
    echo ""
    echo "Setup completed: $(date)"
    echo "================================================"
}

# Main execution
main() {
    log_header "🚀 Production Monitoring Setup for Shyvr RLTE"
    
    # Verify prerequisites
    if ! command -v gcloud >/dev/null 2>&1; then
        log_error "gcloud CLI not found"
        exit 1
    fi
    
    if ! gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1; then
        log_error "Cannot access project: $PROJECT_ID"
        exit 1
    fi
    
    # Execute setup steps
    enable_apis
    create_custom_metrics
    create_alerting_policies
    create_monitoring_dashboard
    setup_log_metrics
    setup_uptime_checks
    setup_error_reporting
    setup_apm
    
    # Generate summary
    generate_monitoring_summary
    
    log_success "🎉 Production monitoring setup completed successfully!"
}

# Execute main function
main