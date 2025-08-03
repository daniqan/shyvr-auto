#!/bin/bash

# Automated Rollback Script for Shyvr RLTE
# Implements quick rollback capabilities for production and staging environments
# Supports automatic rollback triggers and manual rollback operations

set -euo pipefail

# Configuration
PROJECT_ID=${PROJECT_ID:-"shvyr-ai-bots"}
REGION=${REGION:-"us-central1"}
ENVIRONMENT=${1:-""}
ROLLBACK_TARGET=${2:-"previous"}  # previous, specific-revision, or last-known-good

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
log_header() { echo -e "${PURPLE}[ROLLBACK]${NC} $1"; }

# Get service name based on environment
get_service_name() {
    case "$ENVIRONMENT" in
        staging)
            echo "shyvr-rlte-staging"
            ;;
        production)
            echo "shyvr-rlte"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT (must be 'staging' or 'production')"
            exit 1
            ;;
    esac
}

# Get revision history
get_revision_history() {
    local service="$1"
    
    log_info "📋 Getting revision history for $service..."
    
    gcloud run revisions list \
        --service="$service" \
        --region="$REGION" \
        --format="table(metadata.name,metadata.creationTimestamp,status.conditions[0].status,spec.template.spec.containers[0].image)" \
        --sort-by="~metadata.creationTimestamp"
}

# Get current active revision
get_current_revision() {
    local service="$1"
    
    gcloud run services describe "$service" \
        --region="$REGION" \
        --format="value(status.latestReadyRevisionName)" 2>/dev/null || echo ""
}

# Get previous revision (last successfully deployed)
get_previous_revision() {
    local service="$1"
    local current_revision="$2"
    
    # Get all revisions sorted by creation time (newest first)
    local revisions
    revisions=$(gcloud run revisions list \
        --service="$service" \
        --region="$REGION" \
        --format="value(metadata.name)" \
        --sort-by="~metadata.creationTimestamp" \
        --filter="status.conditions[0].status=True")
    
    # Find the revision that comes after the current one (i.e., the previous one)
    local found_current=false
    for revision in $revisions; do
        if [[ "$found_current" == "true" ]]; then
            echo "$revision"
            return 0
        fi
        if [[ "$revision" == "$current_revision" ]]; then
            found_current=true
        fi
    done
    
    return 1
}

# Health check function
check_service_health() {
    local service="$1"
    local max_attempts=${2:-5}
    
    local service_url
    service_url=$(gcloud run services describe "$service" --region="$REGION" --format="value(status.url)" 2>/dev/null)
    
    if [[ -z "$service_url" ]]; then
        log_error "Could not get service URL for $service"
        return 1
    fi
    
    log_info "🔍 Checking health of $service at $service_url"
    
    local attempts=0
    while [[ $attempts -lt $max_attempts ]]; do
        if curl -f -s -m 30 "$service_url/health" >/dev/null 2>&1; then
            log_success "✓ Health check passed for $service"
            return 0
        fi
        
        ((attempts++))
        log_warning "Health check attempt $attempts/$max_attempts failed"
        [[ $attempts -lt $max_attempts ]] && sleep 10
    done
    
    log_error "✗ Health checks failed for $service after $max_attempts attempts"
    return 1
}

# Perform rollback
perform_rollback() {
    local service="$1"
    local target_revision="$2"
    local rollback_reason="${3:-manual}"
    
    log_header "🔄 Performing rollback for $service"
    log_info "Target revision: $target_revision"
    log_info "Rollback reason: $rollback_reason"
    
    # Get current traffic allocation for logging
    local current_traffic
    current_traffic=$(gcloud run services describe "$service" --region="$REGION" --format="json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
traffic = data.get('status', {}).get('traffic', [])
for t in traffic:
    print(f\"{t.get('revisionName', 'N/A')}={t.get('percent', 0)}%\")
" 2>/dev/null || echo "unknown")
    
    log_info "Current traffic allocation: $current_traffic"
    
    # Record rollback start time
    local rollback_start_time=$(date +%s)
    
    # Perform immediate traffic switch to target revision
    log_info "Switching 100% traffic to $target_revision..."
    
    if gcloud run services update-traffic "$service" \
        --region="$REGION" \
        --to-revisions="$target_revision=100" \
        --quiet; then
        
        local rollback_end_time=$(date +%s)
        local rollback_duration=$((rollback_end_time - rollback_start_time))
        
        log_success "Traffic switched to $target_revision in ${rollback_duration} seconds"
        
        # Wait for traffic to stabilize
        log_info "Waiting for traffic to stabilize..."
        sleep 30
        
        # Verify rollback success
        if check_service_health "$service" 5; then
            log_success "🎉 Rollback completed successfully!"
            
            # Log rollback details
            log_rollback_details "$service" "$target_revision" "$rollback_reason" "$rollback_duration"
            
            return 0
        else
            log_error "Rollback completed but service health checks failed"
            return 1
        fi
    else
        log_error "Failed to switch traffic to $target_revision"
        return 1
    fi
}

# Log rollback details for audit trail
log_rollback_details() {
    local service="$1"
    local target_revision="$2"
    local rollback_reason="$3"
    local rollback_duration="$4"
    
    local service_url
    service_url=$(gcloud run services describe "$service" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "unknown")
    
    log_header "📋 Rollback Summary"
    echo "================================================"
    echo "Service: $service"
    echo "Environment: $ENVIRONMENT"
    echo "Target Revision: $target_revision"
    echo "Rollback Reason: $rollback_reason"
    echo "Rollback Duration: ${rollback_duration}s"
    echo "Service URL: $service_url"
    echo "Timestamp: $(date)"
    echo "Initiated by: $(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -1)"
    echo "================================================"
    
    # Save to rollback log file
    local log_file="rollback_log_$(date +%Y%m%d).txt"
    {
        echo "$(date): ROLLBACK - Service: $service, Environment: $ENVIRONMENT, Target: $target_revision, Reason: $rollback_reason, Duration: ${rollback_duration}s"
    } >> "$log_file"
    
    log_info "Rollback details saved to: $log_file"
}

# Emergency rollback function (fastest possible)
emergency_rollback() {
    local service="$1"
    
    log_header "🚨 EMERGENCY ROLLBACK INITIATED"
    log_warning "This is an emergency rollback - minimal validation"
    
    local previous_revision
    local current_revision
    current_revision=$(get_current_revision "$service")
    
    if [[ -z "$current_revision" ]]; then
        log_error "Cannot determine current revision for emergency rollback"
        exit 1
    fi
    
    previous_revision=$(get_previous_revision "$service" "$current_revision")
    
    if [[ -z "$previous_revision" ]]; then
        log_error "Cannot determine previous revision for emergency rollback"
        exit 1
    fi
    
    log_warning "Emergency rollback: $current_revision -> $previous_revision"
    
    # Immediate traffic switch without health checks
    if gcloud run services update-traffic "$service" \
        --region="$REGION" \
        --to-revisions="$previous_revision=100" \
        --quiet; then
        log_success "Emergency rollback traffic switch completed"
        
        # Basic health check
        sleep 15
        if check_service_health "$service" 2; then
            log_success "🎉 Emergency rollback completed successfully!"
        else
            log_warning "Emergency rollback completed but health status unclear"
        fi
    else
        log_error "Emergency rollback failed"
        exit 1
    fi
}

# Automated rollback based on health monitoring
automated_rollback() {
    local service="$1"
    local health_threshold=${2:-3}  # Number of consecutive failures before rollback
    
    log_header "🤖 Automated Health-Based Rollback Monitor"
    log_info "Monitoring $service health (failure threshold: $health_threshold)"
    
    local consecutive_failures=0
    local max_monitoring_time=600  # 10 minutes
    local monitoring_start=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        local elapsed_time=$((current_time - monitoring_start))
        
        if [[ $elapsed_time -gt $max_monitoring_time ]]; then
            log_info "Monitoring period completed (${max_monitoring_time}s)"
            break
        fi
        
        if check_service_health "$service" 1; then
            consecutive_failures=0
            log_info "Health check passed (${elapsed_time}s elapsed)"
        else
            ((consecutive_failures++))
            log_warning "Health check failed (consecutive failures: $consecutive_failures/$health_threshold)"
            
            if [[ $consecutive_failures -ge $health_threshold ]]; then
                log_error "Health threshold exceeded - triggering automated rollback"
                
                local current_revision=$(get_current_revision "$service")
                local previous_revision=$(get_previous_revision "$service" "$current_revision")
                
                if [[ -n "$previous_revision" ]]; then
                    perform_rollback "$service" "$previous_revision" "automated_health_failure"
                    break
                else
                    log_error "Cannot determine previous revision for automated rollback"
                    exit 1
                fi
            fi
        fi
        
        sleep 30  # Check every 30 seconds
    done
}

# Interactive rollback selection
interactive_rollback() {
    local service="$1"
    
    log_header "🎯 Interactive Rollback Selection"
    
    echo "Available revisions for $service:"
    get_revision_history "$service"
    
    echo ""
    read -p "Enter the revision name to rollback to: " selected_revision
    
    if [[ -z "$selected_revision" ]]; then
        log_error "No revision selected"
        exit 1
    fi
    
    # Validate revision exists
    if ! gcloud run revisions describe "$selected_revision" --region="$REGION" >/dev/null 2>&1; then
        log_error "Revision not found: $selected_revision"
        exit 1
    fi
    
    read -p "Rollback reason (optional): " rollback_reason
    rollback_reason=${rollback_reason:-"manual_interactive"}
    
    echo ""
    log_warning "About to rollback $service to $selected_revision"
    read -p "Are you sure? (yes/no): " confirmation
    
    if [[ "$confirmation" == "yes" ]]; then
        perform_rollback "$service" "$selected_revision" "$rollback_reason"
    else
        log_info "Rollback cancelled"
        exit 0
    fi
}

# Main function
main() {
    local service
    service=$(get_service_name)
    
    log_header "🔄 Rollback Manager for Shyvr RLTE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Service: $service"
    log_info "Rollback target: $ROLLBACK_TARGET"
    
    # Verify service exists
    if ! gcloud run services describe "$service" --region="$REGION" >/dev/null 2>&1; then
        log_error "Service not found: $service"
        exit 1
    fi
    
    case "$ROLLBACK_TARGET" in
        previous)
            local current_revision=$(get_current_revision "$service")
            local previous_revision=$(get_previous_revision "$service" "$current_revision")
            
            if [[ -n "$previous_revision" ]]; then
                perform_rollback "$service" "$previous_revision" "rollback_to_previous"
            else
                log_error "Cannot determine previous revision"
                exit 1
            fi
            ;;
            
        emergency)
            emergency_rollback "$service"
            ;;
            
        monitor)
            automated_rollback "$service" 3
            ;;
            
        interactive)
            interactive_rollback "$service"
            ;;
            
        *)
            # Assume it's a specific revision name
            if gcloud run revisions describe "$ROLLBACK_TARGET" --region="$REGION" >/dev/null 2>&1; then
                perform_rollback "$service" "$ROLLBACK_TARGET" "rollback_to_specific"
            else
                log_error "Invalid rollback target: $ROLLBACK_TARGET"
                exit 1
            fi
            ;;
    esac
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <environment> [rollback_target]"
    echo ""
    echo "Environments:"
    echo "  staging     - Rollback staging environment"
    echo "  production  - Rollback production environment"
    echo ""
    echo "Rollback targets:"
    echo "  previous           - Rollback to previous revision (default)"
    echo "  emergency          - Emergency rollback (fastest, minimal validation)"
    echo "  monitor            - Automated health-based rollback monitoring"
    echo "  interactive        - Interactive revision selection"
    echo "  <revision-name>    - Rollback to specific revision"
    echo ""
    echo "Examples:"
    echo "  $0 staging previous"
    echo "  $0 production emergency"
    echo "  $0 production shyvr-rlte-00001-abc"
    echo "  $0 staging monitor"
    echo "  $0 production interactive"
    exit 1
fi

# Execute main function
main