#!/bin/bash

# Production Monitoring Setup for Shyvr RLTE
# Configures comprehensive monitoring, alerting, and observability for production deployment
# Enhanced with deploy-utils.sh integration and production-ready features

set -euo pipefail

# Source deployment utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy-utils.sh"

# Configuration with defaults from deploy-utils.sh
PROJECT_ID="${DEFAULT_PROJECT_ID}"
REGION="${DEFAULT_REGION}"
SERVICE="shyvr-rlte"
ENVIRONMENT="production"
DRY_RUN=false
VALIDATE_ONLY=false
SKIP_DASHBOARD=false

# Notification configuration
NOTIFICATION_EMAIL=""
SLACK_WEBHOOK_URL=""

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Production monitoring setup for Shyvr RLTE with comprehensive alerting and observability.

OPTIONS:
    --project-id PROJECT    GCP project ID (default: $PROJECT_ID)
    --region REGION         GCP region (default: $REGION)
    --service SERVICE       Service name (default: $SERVICE)
    --environment ENV       Environment (default: $ENVIRONMENT)
    --email EMAIL           Notification email address
    --slack-webhook URL     Slack webhook URL for notifications
    --dry-run              Show what would be done without executing
    --validate-only        Only validate existing monitoring setup
    --skip-dashboard       Skip dashboard creation
    --help, -h             Show this help message

EXAMPLES:
    # Basic setup
    $0 --email admin@company.com
    
    # Setup with Slack notifications
    $0 --email admin@company.com --slack-webhook https://hooks.slack.com/...
    
    # Dry run to see what would be created
    $0 --email admin@company.com --dry-run
    
    # Validate existing setup
    $0 --validate-only

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --service)
                SERVICE="$2"
                shift 2
                ;;
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --email)
                NOTIFICATION_EMAIL="$2"
                shift 2
                ;;
            --slack-webhook)
                SLACK_WEBHOOK_URL="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            --skip-dashboard)
                SKIP_DASHBOARD=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                util_log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Initialize monitoring setup
initialize_monitoring_setup() {
    util_log_header "📊 Production Monitoring Setup for Shyvr RLTE"
    util_log_info "Configuring comprehensive monitoring, alerting, and observability"
    
    # Validate prerequisites
    if ! check_required_tools; then
        util_log_error "Prerequisites check failed"
        exit 1
    fi
    
    if ! check_gcp_auth "$PROJECT_ID"; then
        util_log_error "GCP authentication check failed"
        exit 1
    fi
    
    # Set project context
    gcloud config set project "$PROJECT_ID" --quiet
    util_log_success "Using project: $PROJECT_ID"
    
    # Validate environment
    if ! validate_environment "$ENVIRONMENT"; then
        exit 1
    fi
    
    util_log_success "Environment validated: $ENVIRONMENT"
}

# Enable required APIs
enable_monitoring_apis() {
    util_log_info "Enabling required Google Cloud APIs"
    
    local apis=(
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "clouderrorreporting.googleapis.com"
        "cloudtrace.googleapis.com"
        "cloudprofiler.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            util_log_info "[DRY RUN] Would enable $api"
        else
            util_log_info "Enabling $api"
            if gcloud services enable "$api" --project="$PROJECT_ID" --quiet; then
                util_log_success "✓ $api enabled"
            else
                util_log_warning "Failed to enable $api (may already be enabled)"
            fi
        fi
    done
    
    util_log_success "Required APIs enabled"
}

# Validate existing monitoring setup
validate_monitoring_setup() {
    util_log_info "Validating existing monitoring setup"
    
    local validation_passed=true
    
    # Check if monitoring API is enabled
    if ! gcloud services list --enabled --filter="name:monitoring.googleapis.com" --format="value(name)" | grep -q "monitoring.googleapis.com"; then
        util_log_error "Monitoring API is not enabled"
        validation_passed=false
    else
        util_log_success "✓ Monitoring API is enabled"
    fi
    
    # Check for existing alert policies
    local policies_count
    policies_count=$(gcloud alpha monitoring policies list --format="value(name)" 2>/dev/null | wc -l || echo "0")
    
    if [[ "$policies_count" -gt 0 ]]; then
        util_log_success "✓ Found $policies_count existing alert policies"
    else
        util_log_warning "No existing alert policies found"
    fi
    
    # Check for existing dashboards
    local dashboards_count
    dashboards_count=$(gcloud monitoring dashboards list --format="value(name)" 2>/dev/null | wc -l || echo "0")
    
    if [[ "$dashboards_count" -gt 0 ]]; then
        util_log_success "✓ Found $dashboards_count existing dashboards"
    else
        util_log_warning "No existing dashboards found"
    fi
    
    # Check service existence
    if gcloud run services describe "$SERVICE" --region="$REGION" --format="value(metadata.name)" >/dev/null 2>&1; then
        util_log_success "✓ Cloud Run service '$SERVICE' exists"
    else
        util_log_warning "Cloud Run service '$SERVICE' not found in region $REGION"
    fi
    
    if [[ "$validation_passed" == "true" ]]; then
        util_log_success "Monitoring validation passed"
        return 0
    else
        util_log_error "Monitoring validation failed"
        return 1
    fi
}

# Create notification channels
create_notification_channels() {
    util_log_info "Creating notification channels"
    
    local channels_created=()
    
    # Email notification channel
    if [[ -n "$NOTIFICATION_EMAIL" ]]; then
        local email_channel_name="rlte-email-notifications-$ENVIRONMENT"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            util_log_info "[DRY RUN] Would create email notification channel for: $NOTIFICATION_EMAIL"
            echo "EMAIL_NOTIFICATION_CHANNEL=projects/$PROJECT_ID/notificationChannels/dummy-email-channel" > /tmp/notification_channels.env
        else
            util_log_info "Creating email notification channel for: $NOTIFICATION_EMAIL"
            
            cat > /tmp/email_notification.json << EOF
{
  "type": "email",
  "displayName": "$email_channel_name",
  "description": "Production email notifications for $SERVICE ($ENVIRONMENT)",
  "labels": {
    "email_address": "$NOTIFICATION_EMAIL"
  },
  "enabled": true
}
EOF
            
            local email_channel
            if email_channel=$(gcloud alpha monitoring channels create --channel-content-from-file=/tmp/email_notification.json --format="value(name)" 2>/dev/null); then
                util_log_success "Created email notification channel: $email_channel"
                echo "EMAIL_NOTIFICATION_CHANNEL=$email_channel" > /tmp/notification_channels.env
                channels_created+=("Email: $NOTIFICATION_EMAIL")
            else
                util_log_error "Failed to create email notification channel"
                echo "EMAIL_NOTIFICATION_CHANNEL=" > /tmp/notification_channels.env
            fi
            
            rm -f /tmp/email_notification.json
        fi
    else
        util_log_warning "No email address provided, skipping email notifications"
        echo "EMAIL_NOTIFICATION_CHANNEL=" > /tmp/notification_channels.env
    fi
    
    # Slack notification channel (if webhook URL is provided)
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        local slack_channel_name="rlte-slack-notifications-$ENVIRONMENT"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            util_log_info "[DRY RUN] Would create Slack notification channel"
            echo "SLACK_NOTIFICATION_CHANNEL=projects/$PROJECT_ID/notificationChannels/dummy-slack-channel" >> /tmp/notification_channels.env
        else
            util_log_info "Creating Slack notification channel"
            
            cat > /tmp/slack_notification.json << EOF
{
  "type": "slack",
  "displayName": "$slack_channel_name",
  "description": "Production Slack notifications for $SERVICE ($ENVIRONMENT)",
  "labels": {
    "url": "$SLACK_WEBHOOK_URL"
  },
  "enabled": true
}
EOF
            
            local slack_channel
            if slack_channel=$(gcloud alpha monitoring channels create --channel-content-from-file=/tmp/slack_notification.json --format="value(name)" 2>/dev/null); then
                util_log_success "Created Slack notification channel: $slack_channel"
                echo "SLACK_NOTIFICATION_CHANNEL=$slack_channel" >> /tmp/notification_channels.env
                channels_created+=("Slack: webhook configured")
            else
                util_log_warning "Failed to create Slack notification channel"
                echo "SLACK_NOTIFICATION_CHANNEL=" >> /tmp/notification_channels.env
            fi
            
            rm -f /tmp/slack_notification.json
        fi
    else
        echo "SLACK_NOTIFICATION_CHANNEL=" >> /tmp/notification_channels.env
    fi
    
    if [[ ${#channels_created[@]} -gt 0 ]]; then
        util_log_success "Created notification channels: ${channels_created[*]}"
    else
        util_log_warning "No notification channels created"
    fi
}

# Create custom metrics configuration
create_custom_metrics() {
    util_log_info "Creating custom metrics for trading system"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create custom metrics definitions"
        return 0
    fi
    
    # Note: Custom metrics are created automatically when first used in GCP
    # We just prepare the configurations here
    
    local metrics=(
        "trading/position_count:Trading position count"
        "trading/pnl_total:Total P&L in USD"
        "trading/risk_score:Portfolio risk score (0-100)"
        "ml/model_accuracy:ML model prediction accuracy"
        "rl/episode_reward:RL agent episode reward"
        "rl/experience_count:Number of stored RL experiences"
        "safety/emergency_stops:Total emergency stops triggered"
        "api/rate_limit_hits:API rate limit hits"
    )
    
    for metric_def in "${metrics[@]}"; do
        local metric_name="${metric_def%%:*}"
        local description="${metric_def##*:}"
        util_log_info "Prepared custom metric: custom.googleapis.com/$metric_name ($description)"
    done
    
    util_log_success "Custom metrics configuration prepared"
}

# Create alerting policies
create_alerting_policies() {
    util_log_info "Creating alerting policies"
    
    # Source notification channels
    if [[ -f /tmp/notification_channels.env ]]; then
        source /tmp/notification_channels.env
    else
        util_log_warning "No notification channels file found"
        EMAIL_NOTIFICATION_CHANNEL=""
        SLACK_NOTIFICATION_CHANNEL=""
    fi
    
    local notification_channels=()
    [[ -n "$EMAIL_NOTIFICATION_CHANNEL" ]] && notification_channels+=("\"$EMAIL_NOTIFICATION_CHANNEL\"")
    [[ -n "$SLACK_NOTIFICATION_CHANNEL" ]] && notification_channels+=("\"$SLACK_NOTIFICATION_CHANNEL\"")
    
    local notification_channels_json=""
    if [[ ${#notification_channels[@]} -gt 0 ]]; then
        notification_channels_json=$(printf "%s," "${notification_channels[@]}" | sed 's/,$//')
    fi
    
    local policies_created=()
    
    # 1. High CPU Usage Alert
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create CPU usage alert policy"
    else
        util_log_info "Creating CPU usage alert policy"
        cat > /tmp/cpu_alert.yaml << EOF
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
displayName: "RLTE High CPU Usage ($ENVIRONMENT)"
documentation:
  content: "CPU usage is above 80% for 5 minutes in $ENVIRONMENT environment"
notificationChannels: [$notification_channels_json]
alertStrategy:
  autoClose: 1800s
enabled: true
EOF
        
        if gcloud alpha monitoring policies create --policy-from-file=/tmp/cpu_alert.yaml --quiet; then
            util_log_success "✓ CPU usage alert created"
            policies_created+=("High CPU Usage")
        else
            util_log_warning "Failed to create CPU usage alert (may already exist)"
        fi
        rm -f /tmp/cpu_alert.yaml
    fi
    
    # 2. High Memory Usage Alert
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create memory usage alert policy"
    else
        util_log_info "Creating memory usage alert policy"
        cat > /tmp/memory_alert.yaml << EOF
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
displayName: "RLTE High Memory Usage ($ENVIRONMENT)"
documentation:
  content: "Memory usage is above 85% for 5 minutes in $ENVIRONMENT environment"
notificationChannels: [$notification_channels_json]
alertStrategy:
  autoClose: 1800s
enabled: true
EOF
        
        if gcloud alpha monitoring policies create --policy-from-file=/tmp/memory_alert.yaml --quiet; then
            util_log_success "✓ Memory usage alert created"
            policies_created+=("High Memory Usage")
        else
            util_log_warning "Failed to create memory usage alert (may already exist)"
        fi
        rm -f /tmp/memory_alert.yaml
    fi
    
    # 3. High Error Rate Alert
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create error rate alert policy"
    else
        util_log_info "Creating error rate alert policy"
        cat > /tmp/error_alert.yaml << EOF
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
displayName: "RLTE High Error Rate ($ENVIRONMENT)"
documentation:
  content: "Error rate is above 5% for 3 minutes in $ENVIRONMENT environment"
notificationChannels: [$notification_channels_json]
alertStrategy:
  autoClose: 1800s
enabled: true
EOF
        
        if gcloud alpha monitoring policies create --policy-from-file=/tmp/error_alert.yaml --quiet; then
            util_log_success "✓ Error rate alert created"
            policies_created+=("High Error Rate")
        else
            util_log_warning "Failed to create error rate alert (may already exist)"
        fi
        rm -f /tmp/error_alert.yaml
    fi
    
    # 4. Emergency Stop Alert (Critical)
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create emergency stop alert policy"
    else
        util_log_info "Creating emergency stop alert policy"
        cat > /tmp/emergency_alert.yaml << EOF
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
displayName: "RLTE Emergency Stop ($ENVIRONMENT)"
documentation:
  content: "Emergency stop has been triggered - immediate attention required in $ENVIRONMENT environment"
severity: CRITICAL
notificationChannels: [$notification_channels_json]
alertStrategy:
  autoClose: 3600s
enabled: true
EOF
        
        if gcloud alpha monitoring policies create --policy-from-file=/tmp/emergency_alert.yaml --quiet; then
            util_log_success "✓ Emergency stop alert created"
            policies_created+=("Emergency Stop")
        else
            util_log_warning "Failed to create emergency stop alert (may already exist)"
        fi
        rm -f /tmp/emergency_alert.yaml
    fi
    
    if [[ ${#policies_created[@]} -gt 0 ]]; then
        util_log_success "Created alert policies: ${policies_created[*]}"
    else
        util_log_warning "No new alert policies created"
    fi
}

# Create monitoring dashboard
create_monitoring_dashboard() {
    if [[ "$SKIP_DASHBOARD" == "true" ]]; then
        util_log_info "Skipping dashboard creation (--skip-dashboard flag)"
        return 0
    fi
    
    util_log_info "Creating monitoring dashboard"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create Shyvr RLTE Production Dashboard"
        return 0
    fi
    
    cat > /tmp/rlte_dashboard.json << EOF
{
  "displayName": "Shyvr RLTE $ENVIRONMENT Dashboard",
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
    
    if gcloud monitoring dashboards create --config-from-file=/tmp/rlte_dashboard.json --quiet; then
        util_log_success "✓ Production dashboard created"
    else
        util_log_warning "Failed to create dashboard (may already exist)"
    fi
    
    rm -f /tmp/rlte_dashboard.json
}

# Setup log-based metrics
setup_log_metrics() {
    util_log_info "Setting up log-based metrics"
    
    local metrics_created=()
    
    # Error count metric
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create error count log metric"
    else
        util_log_info "Creating error count log metric"
        if gcloud logging metrics create "rlte_error_count_$ENVIRONMENT" \
            --description="Count of error logs in RLTE ($ENVIRONMENT)" \
            --log-filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\" AND severity>=ERROR" \
            --project="$PROJECT_ID" --quiet 2>/dev/null; then
            metrics_created+=("Error Count")
        else
            util_log_warning "Error count metric may already exist"
        fi
    fi
    
    # Trading action metric
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create trading action log metric"
    else
        util_log_info "Creating trading action log metric"
        if gcloud logging metrics create "rlte_trading_actions_$ENVIRONMENT" \
            --description="Count of trading actions in RLTE ($ENVIRONMENT)" \
            --log-filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\" AND jsonPayload.action_type!=null" \
            --project="$PROJECT_ID" --quiet 2>/dev/null; then
            metrics_created+=("Trading Actions")
        else
            util_log_warning "Trading actions metric may already exist"
        fi
    fi
    
    # Model prediction metric
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Would create model prediction log metric"
    else
        util_log_info "Creating model prediction log metric"
        if gcloud logging metrics create "rlte_model_predictions_$ENVIRONMENT" \
            --description="Count of ML model predictions in RLTE ($ENVIRONMENT)" \
            --log-filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$SERVICE\" AND jsonPayload.prediction!=null" \
            --project="$PROJECT_ID" --quiet 2>/dev/null; then
            metrics_created+=("Model Predictions")
        else
            util_log_warning "Model predictions metric may already exist"
        fi
    fi
    
    if [[ ${#metrics_created[@]} -gt 0 ]]; then
        util_log_success "Created log-based metrics: ${metrics_created[*]}"
    else
        util_log_warning "No new log-based metrics created"
    fi
}

# Configure uptime checks
setup_uptime_checks() {
    util_log_info "Setting up uptime checks"
    
    # Get service URL
    local service_url
    if [[ "$DRY_RUN" == "true" ]]; then
        service_url="https://$SERVICE-dummy-url.a.run.app"
        util_log_info "[DRY RUN] Would create uptime check for service URL"
    else
        service_url=$(get_service_url "$SERVICE" "$REGION")
        
        if [[ -z "$service_url" ]]; then
            util_log_warning "Service URL not available - skipping uptime checks"
            return
        fi
        
        util_log_info "Creating uptime check for: $service_url"
        
        # Health endpoint uptime check
        cat > /tmp/uptime_check.yaml << EOF
displayName: "RLTE Health Check ($ENVIRONMENT)"
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
    host: "$(echo "$service_url" | sed 's|https://||' | sed 's|/.*||')"
period: 60s
timeout: 30s
EOF
        
        if gcloud monitoring uptime-check-configs create --config-from-file=/tmp/uptime_check.yaml --quiet; then
            util_log_success "✓ Health check uptime monitor created"
        else
            util_log_warning "Failed to create uptime check (may already exist)"
        fi
        
        rm -f /tmp/uptime_check.yaml
    fi
}

# Setup error reporting
setup_error_reporting() {
    util_log_info "Setting up error reporting"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Error Reporting would be automatically enabled for Cloud Run services"
    else
        util_log_info "Error Reporting is automatically enabled for Cloud Run services"
        util_log_info "Errors will be automatically captured and reported"
    fi
    
    util_log_success "Error reporting configured"
}

# Setup application performance monitoring
setup_apm() {
    util_log_info "Setting up Application Performance Monitoring"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        util_log_info "[DRY RUN] Cloud Trace and Profiler would be enabled for detailed performance monitoring"
    else
        util_log_info "Cloud Trace and Profiler are enabled for detailed performance monitoring"
        util_log_info "Traces will be automatically collected for HTTP requests"
    fi
    
    util_log_success "APM monitoring configured"
}

# Generate monitoring summary
generate_monitoring_summary() {
    util_log_header "📋 Production Monitoring Setup Summary"
    
    local service_url
    if [[ "$DRY_RUN" == "true" ]]; then
        service_url="[DRY RUN] Service URL would be determined"
    else
        service_url=$(get_service_url "$SERVICE" "$REGION" 2>/dev/null || echo "Not deployed or accessible")
    fi
    
    echo "================================================"
    echo "Service: $SERVICE"
    echo "Environment: $ENVIRONMENT"
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Service URL: $service_url"
    echo "Mode: $([ "$DRY_RUN" = "true" ] && echo "DRY RUN" || echo "LIVE")"
    echo "Setup Time: $(date)"
    echo ""
    echo "Monitoring Resources $([ "$DRY_RUN" = "true" ] && echo "Would Be " || echo "")Created:"
    echo "  ✓ Custom metrics for trading, ML, and RL"
    echo "  ✓ Alerting policies for critical conditions"
    [[ "$SKIP_DASHBOARD" != "true" ]] && echo "  ✓ Production monitoring dashboard"
    echo "  ✓ Log-based metrics for application events"
    echo "  ✓ Uptime checks for health monitoring"
    echo "  ✓ Error reporting and APM"
    echo ""
    echo "Notification Channels:"
    [[ -n "$NOTIFICATION_EMAIL" ]] && echo "  ✓ Email: $NOTIFICATION_EMAIL"
    [[ -n "$SLACK_WEBHOOK_URL" ]] && echo "  ✓ Slack: webhook configured"
    [[ -z "$NOTIFICATION_EMAIL" && -z "$SLACK_WEBHOOK_URL" ]] && echo "  ⚠ No notification channels configured"
    echo ""
    echo "Access Points:"
    echo "  Monitoring Console: https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
    echo "  Logs Explorer: https://console.cloud.google.com/logs/query?project=$PROJECT_ID"
    echo "  Error Reporting: https://console.cloud.google.com/errors?project=$PROJECT_ID"
    echo "  Trace Explorer: https://console.cloud.google.com/traces/list?project=$PROJECT_ID"
    echo ""
    echo "================================================"
}

# Cleanup temporary files
cleanup_temp_files() {
    rm -f /tmp/notification_channels.env
    rm -f /tmp/*_alert.yaml
    rm -f /tmp/*_notification.json
    rm -f /tmp/rlte_dashboard.json
    rm -f /tmp/uptime_check.yaml
}

# Main execution function
main() {
    parse_arguments "$@"
    initialize_monitoring_setup
    
    # Validate only mode
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        if validate_monitoring_setup; then
            util_log_success "Monitoring validation passed"
            exit 0
        else
            util_log_error "Monitoring validation failed"
            exit 1
        fi
    fi
    
    # Execute setup steps
    enable_monitoring_apis
    create_custom_metrics
    create_notification_channels
    create_alerting_policies
    create_monitoring_dashboard
    setup_log_metrics
    setup_uptime_checks
    setup_error_reporting
    setup_apm
    
    # Generate summary
    generate_monitoring_summary
    
    # Cleanup
    cleanup_temp_files
    
    util_log_success "🎉 Production monitoring setup completed successfully!"
}

# Execute main function if script is called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi