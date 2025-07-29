#!/bin/bash
set -e

# Shyvr RLTE Monitoring Setup Script
# This script sets up comprehensive monitoring for the trading system in GCP

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/monitoring_config.json"
RESULTS_FILE="$PROJECT_ROOT/monitoring_setup_results.json"

# Default values
PROJECT_ID=""
ENVIRONMENT="production"
ALERT_EMAIL=""
DRY_RUN=false
VALIDATE_ONLY=false
SETUP_GRAFANA=true

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup comprehensive monitoring for Shyvr RLTE trading system in GCP.

OPTIONS:
    -p, --project-id PROJECT_ID     GCP project ID (required)
    -e, --environment ENV           Environment (production, staging, development)
    -a, --alert-email EMAIL         Email address for alerts
    -d, --dry-run                   Show what would be done without executing
    -v, --validate-only             Only validate existing setup
    --no-grafana                    Skip Grafana dashboard setup
    -h, --help                      Show this help message

EXAMPLES:
    $0 -p my-gcp-project -a alerts@company.com
    $0 -p my-gcp-project -e staging --validate-only
    $0 -p my-gcp-project -d --no-grafana

ENVIRONMENT VARIABLES:
    GOOGLE_APPLICATION_CREDENTIALS  Path to GCP service account key
    ALERT_EMAIL_CRITICAL           Email for critical alerts
    ALERT_EMAIL_WARNING           Email for warning alerts
    GRAFANA_API_KEY               Grafana API key (if using existing instance)

EOF
}

# Print colored output
print_status() {
    local level=$1
    local message=$2
    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    print_status "INFO" "Checking prerequisites..."
    
    # Check if gcloud is installed and authenticated
    if ! command -v gcloud &> /dev/null; then
        print_status "ERROR" "gcloud CLI is not installed"
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null 2>&1; then
        print_status "ERROR" "Not authenticated with gcloud. Run 'gcloud auth login'"
        exit 1
    fi
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_status "ERROR" "Python 3 is not installed"
        exit 1
    fi
    
    # Check if uv is available for dependency management
    if ! command -v uv &> /dev/null; then
        print_status "WARNING" "uv is not installed. Falling back to python3"
    fi
    
    # Validate project ID
    if [ -z "$PROJECT_ID" ]; then
        print_status "ERROR" "Project ID is required. Use -p or --project-id"
        exit 1
    fi
    
    # Check if project exists and is accessible
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        print_status "ERROR" "Cannot access project '$PROJECT_ID' or project doesn't exist"
        exit 1
    fi
    
    print_status "SUCCESS" "Prerequisites check passed"
}

# Enable required GCP APIs
enable_apis() {
    print_status "INFO" "Enabling required GCP APIs..."
    
    local apis=(
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "cloudresourcemanager.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        if [ "$DRY_RUN" = true ]; then
            print_status "INFO" "[DRY RUN] Would enable API: $api"
        else
            print_status "INFO" "Enabling API: $api"
            gcloud services enable "$api" --project="$PROJECT_ID"
        fi
    done
    
    print_status "SUCCESS" "APIs enabled successfully"
}

# Set up environment variables for configuration
setup_environment() {
    print_status "INFO" "Setting up environment variables..."
    
    # Export environment variables for the Python script
    export PROJECT_ID="$PROJECT_ID"
    export ENVIRONMENT="$ENVIRONMENT"
    
    # Set alert emails
    if [ -n "$ALERT_EMAIL" ]; then
        export ALERT_EMAIL_CRITICAL="$ALERT_EMAIL"
        export ALERT_EMAIL_WARNING="$ALERT_EMAIL"
    else
        # Use environment variables if set
        if [ -z "$ALERT_EMAIL_CRITICAL" ] && [ -z "$ALERT_EMAIL_WARNING" ]; then
            print_status "WARNING" "No alert email addresses configured"
        fi
    fi
    
    print_status "SUCCESS" "Environment configured"
}

# Run the monitoring setup
run_monitoring_setup() {
    print_status "INFO" "Running monitoring setup..."
    
    local python_cmd="python3"
    if command -v uv &> /dev/null; then
        python_cmd="uv run python"
    fi
    
    local setup_script="$PROJECT_ROOT/scripts/setup_gcp_monitoring.py"
    local cmd_args=(
        "--project-id" "$PROJECT_ID"
        "--config" "$CONFIG_FILE"
        "--output" "$RESULTS_FILE"
    )
    
    if [ "$VALIDATE_ONLY" = true ]; then
        cmd_args+=("--validate-only")
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would run: $python_cmd $setup_script ${cmd_args[*]}"
        return 0
    fi
    
    print_status "INFO" "Executing monitoring setup..."
    if $python_cmd "$setup_script" "${cmd_args[@]}"; then
        print_status "SUCCESS" "Monitoring setup completed successfully"
        
        # Display results if available
        if [ -f "$RESULTS_FILE" ]; then
            print_status "INFO" "Setup results:"
            cat "$RESULTS_FILE" | python3 -m json.tool | head -20
        fi
    else
        print_status "ERROR" "Monitoring setup failed"
        return 1
    fi
}

# Set up Grafana dashboards
setup_grafana() {
    if [ "$SETUP_GRAFANA" = false ]; then
        print_status "INFO" "Grafana setup skipped"
        return 0
    fi
    
    print_status "INFO" "Setting up Grafana dashboards..."
    
    local grafana_script="$PROJECT_ROOT/scripts/setup_grafana_gcp.py"
    
    if [ ! -f "$grafana_script" ]; then
        print_status "WARNING" "Grafana setup script not found, skipping dashboard creation"
        return 0
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would set up Grafana dashboards"
        return 0
    fi
    
    local python_cmd="python3"
    if command -v uv &> /dev/null; then
        python_cmd="uv run python"
    fi
    
    if $python_cmd "$grafana_script" --project-id "$PROJECT_ID"; then
        print_status "SUCCESS" "Grafana dashboards configured"
    else
        print_status "WARNING" "Grafana setup failed, but monitoring is still functional"
    fi
}

# Run validation tests
run_validation() {
    print_status "INFO" "Running monitoring validation..."
    
    local validation_script="$PROJECT_ROOT/scripts/validate_monitoring.py"
    
    if [ ! -f "$validation_script" ]; then
        print_status "WARNING" "Validation script not found, skipping validation"
        return 0
    fi
    
    if [ "$DRY_RUN" = true ]; then
        print_status "INFO" "[DRY RUN] Would run monitoring validation"
        return 0
    fi
    
    local python_cmd="python3"
    if command -v uv &> /dev/null; then
        python_cmd="uv run python"
    fi
    
    if $python_cmd "$validation_script" --project-id "$PROJECT_ID"; then
        print_status "SUCCESS" "Monitoring validation passed"
    else
        print_status "WARNING" "Monitoring validation had issues, check logs"
    fi
}

# Generate summary report
generate_summary() {
    print_status "INFO" "Generating setup summary..."
    
    cat << EOF

================================================================================
SHYVR RLTE MONITORING SETUP SUMMARY
================================================================================

Project ID: $PROJECT_ID
Environment: $ENVIRONMENT
Setup Time: $(date)
Setup Mode: $([ "$DRY_RUN" = true ] && echo "DRY RUN" || echo "LIVE")

COMPONENTS CONFIGURED:
✓ Google Cloud Monitoring custom metrics
✓ Alert policies for trading system monitoring
✓ Notification channels for alerts
$([ "$SETUP_GRAFANA" = true ] && echo "✓ Grafana dashboard integration" || echo "- Grafana setup skipped")

METRICS CATEGORIES:
- Trading Performance (P&L, volume, success rate)
- Risk Management (risk level, drawdown, emergency stops)
- System Health (uptime, response times, error rates)
- ML/RL Performance (model accuracy, agent rewards)
- DEX Connectivity (connection status, wallet balances)

ALERT POLICIES:
- Critical: System down, emergency stops, critical risk levels
- Warning: High drawdown, low success rates, ML model issues

NEXT STEPS:
1. Verify alert email delivery
2. Test alert policies with synthetic data
3. Monitor dashboard for first 24 hours
4. Configure additional notification channels if needed

MONITORING URLS:
- Cloud Monitoring: https://console.cloud.google.com/monitoring?project=$PROJECT_ID
$([ -n "$GRAFANA_DASHBOARD_URL" ] && echo "- Grafana: $GRAFANA_DASHBOARD_URL")

DOCUMENTATION:
- See docs/MONITORING_SYSTEM.md for detailed information
- Setup results: $RESULTS_FILE

================================================================================

EOF
    
    print_status "SUCCESS" "Monitoring setup completed!"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -a|--alert-email)
                ALERT_EMAIL="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            --no-grafana)
                SETUP_GRAFANA=false
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_status "ERROR" "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Main execution flow
main() {
    print_status "INFO" "Starting Shyvr RLTE monitoring setup..."
    
    parse_args "$@"
    check_prerequisites
    setup_environment
    
    if [ "$VALIDATE_ONLY" = false ]; then
        enable_apis
    fi
    
    run_monitoring_setup
    
    if [ "$VALIDATE_ONLY" = false ] && [ "$DRY_RUN" = false ]; then
        setup_grafana
        run_validation
    fi
    
    generate_summary
}

# Run the main function with all arguments
main "$@"