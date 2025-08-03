#!/bin/bash

# Blue-Green Deployment Script for Shyvr RLTE
# Implements zero-downtime deployment with gradual traffic migration and automatic rollback
# Supports both staging and production environments

set -euo pipefail

# Configuration
PROJECT_ID=${PROJECT_ID:-"shvyr-ai-bots"}
REGION=${REGION:-"us-central1"}
REPOSITORY=${REPOSITORY:-"shyvr-ai-prod"}
ENVIRONMENT=${1:-"staging"}  # staging or production
IMAGE_TAG=${2:-"latest"}

# Environment-specific settings
if [[ "$ENVIRONMENT" == "production" ]]; then
    SERVICE="shyvr-rlte"
    INITIAL_TRAFFIC_PERCENT=10
    INTERMEDIATE_TRAFFIC_PERCENT=50
    HEALTH_CHECK_ATTEMPTS=10
    HEALTH_CHECK_INTERVAL=15
    TRAFFIC_MIGRATION_DELAY=120
else
    SERVICE="shyvr-rlte-staging"
    INITIAL_TRAFFIC_PERCENT=50
    INTERMEDIATE_TRAFFIC_PERCENT=100
    HEALTH_CHECK_ATTEMPTS=5
    HEALTH_CHECK_INTERVAL=10
    TRAFFIC_MIGRATION_DELAY=60
fi

IMAGE_NAME="us-central1-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/shyvr-rlte"
DEPLOYMENT_ID=$(date +"%Y%m%d-%H%M%S")

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
log_header() { echo -e "${PURPLE}[BLUE-GREEN]${NC} $1"; }

# Cleanup and rollback functions
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Blue-green deployment failed with exit code $exit_code"
        if [[ -n "${CURRENT_REVISION:-}" ]]; then
            log_warning "Initiating automatic rollback to revision $CURRENT_REVISION"
            rollback_deployment
        fi
    fi
}

trap cleanup EXIT

rollback_deployment() {
    if [[ -n "${CURRENT_REVISION:-}" ]]; then
        log_warning "🔄 Rolling back to revision: $CURRENT_REVISION"
        
        if gcloud run services update-traffic "$SERVICE" \
            --region="$REGION" \
            --to-revisions="$CURRENT_REVISION=100" \
            --quiet; then
            log_success "Rollback completed successfully"
        else
            log_error "Rollback failed - manual intervention required"
        fi
    else
        log_warning "No previous revision available for rollback"
    fi
}

# Health check function
perform_health_checks() {
    local service_url="$1"
    local max_attempts="$2"
    local interval="$3"
    
    log_info "🔍 Performing $max_attempts health checks with ${interval}s interval..."
    
    local failed_attempts=0
    local max_failures=3
    
    for i in $(seq 1 $max_attempts); do
        if curl -f -s -m 30 "$service_url/health" > /dev/null; then
            log_success "✓ Health check $i/$max_attempts passed"
        else
            log_warning "✗ Health check $i/$max_attempts failed"
            ((failed_attempts++))
            
            if [[ $failed_attempts -ge $max_failures ]]; then
                log_error "Too many health check failures ($failed_attempts/$max_failures)"
                return 1
            fi
        fi
        
        [[ $i -lt $max_attempts ]] && sleep $interval
    done
    
    if [[ $failed_attempts -gt 0 ]]; then
        log_warning "Health checks completed with $failed_attempts failures (acceptable)"
    else
        log_success "All health checks passed successfully"
    fi
    
    return 0
}

# Validate deployment configuration
validate_deployment_config() {
    log_info "🔍 Validating deployment configuration..."
    
    # Check if image exists
    if ! gcloud artifacts docker images describe "$IMAGE_NAME:$IMAGE_TAG" --project="$PROJECT_ID" >/dev/null 2>&1; then
        log_error "Image not found: $IMAGE_NAME:$IMAGE_TAG"
        exit 1
    fi
    
    # Check if service exists
    if ! gcloud run services describe "$SERVICE" --region="$REGION" >/dev/null 2>&1; then
        log_warning "Service $SERVICE does not exist - this will be a first deployment"
        CURRENT_REVISION=""
    else
        CURRENT_REVISION=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.latestReadyRevisionName)" 2>/dev/null || echo "")
        log_info "Current revision: ${CURRENT_REVISION:-'none'}"
    fi
    
    log_success "Deployment configuration validated"
}

# Deploy new revision
deploy_new_revision() {
    log_header "🚀 Deploying new revision for $ENVIRONMENT environment"
    
    local env_vars="ENVIRONMENT=$ENVIRONMENT,LOG_LEVEL=INFO,TRADING_MODE=simulation,BUILD_ID=$DEPLOYMENT_ID,COMMIT_SHA=${IMAGE_TAG}"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        env_vars="$env_vars,ML_OPTIMIZED=true,RL_STORAGE_ENABLED=true,MODEL_PRESERVATION_ENABLED=true"
    fi
    
    # Deploy new revision without traffic
    if ! gcloud run deploy "$SERVICE" \
        --image="$IMAGE_NAME:$IMAGE_TAG" \
        --region="$REGION" \
        --platform=managed \
        --port=8080 \
        --memory=6Gi \
        --cpu=4 \
        --concurrency=50 \
        --timeout=1800 \
        --max-instances=10 \
        --min-instances=1 \
        --allow-unauthenticated \
        --cpu-boost \
        --execution-environment=gen2 \
        --session-affinity \
        --set-env-vars="$env_vars" \
        --add-cloudsql-instances="$PROJECT_ID:us-central1:shyvr-rlte-db" \
        --set-secrets="TELEGRAM_TOKEN=TELEGRAM_TOKEN:latest,WEBHOOK_SECRET=WEBHOOK_SECRET:latest,DB_PASSWORD=DB_PASSWORD:latest,DATABASE_URL=DATABASE_URL:latest,SECRET_KEY=SECRET_KEY:latest,JWT_SECRET=JWT_SECRET:latest" \
        --no-traffic \
        --tag="$ENVIRONMENT-$DEPLOYMENT_ID" \
        --quiet; then
        log_error "Failed to deploy new revision"
        exit 1
    fi
    
    # Get new revision name
    NEW_REVISION=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.latestCreatedRevisionName)")
    log_success "New revision deployed: $NEW_REVISION"
}

# Execute blue-green traffic migration
execute_blue_green_migration() {
    log_header "🚦 Executing blue-green traffic migration"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)")
    
    if [[ -z "$SERVICE_URL" ]]; then
        log_error "Failed to get service URL"
        exit 1
    fi
    
    log_info "Service URL: $SERVICE_URL"
    
    if [[ -n "$CURRENT_REVISION" ]]; then
        # Gradual traffic migration
        log_info "Starting gradual traffic migration..."
        
        # Phase 1: Initial traffic to new revision
        log_info "Phase 1: Migrating ${INITIAL_TRAFFIC_PERCENT}% traffic to new revision"
        gcloud run services update-traffic "$SERVICE" \
            --region="$REGION" \
            --to-revisions="$NEW_REVISION=$INITIAL_TRAFFIC_PERCENT,$CURRENT_REVISION=$((100-INITIAL_TRAFFIC_PERCENT))" \
            --quiet
        
        # Wait and monitor
        log_info "Monitoring new revision for $TRAFFIC_MIGRATION_DELAY seconds..."
        sleep $TRAFFIC_MIGRATION_DELAY
        
        # Health checks
        if ! perform_health_checks "$SERVICE_URL" "$HEALTH_CHECK_ATTEMPTS" "$HEALTH_CHECK_INTERVAL"; then
            log_error "Health checks failed during phase 1"
            return 1
        fi
        
        # Phase 2: Increase traffic if not already at 100%
        if [[ $INTERMEDIATE_TRAFFIC_PERCENT -lt 100 && $INTERMEDIATE_TRAFFIC_PERCENT -gt $INITIAL_TRAFFIC_PERCENT ]]; then
            log_info "Phase 2: Migrating ${INTERMEDIATE_TRAFFIC_PERCENT}% traffic to new revision"
            gcloud run services update-traffic "$SERVICE" \
                --region="$REGION" \
                --to-revisions="$NEW_REVISION=$INTERMEDIATE_TRAFFIC_PERCENT,$CURRENT_REVISION=$((100-INTERMEDIATE_TRAFFIC_PERCENT))" \
                --quiet
            
            sleep $TRAFFIC_MIGRATION_DELAY
            
            if ! perform_health_checks "$SERVICE_URL" "$HEALTH_CHECK_ATTEMPTS" "$HEALTH_CHECK_INTERVAL"; then
                log_error "Health checks failed during phase 2"
                return 1
            fi
        fi
        
        # Phase 3: Complete migration
        log_info "Phase 3: Migrating 100% traffic to new revision"
        gcloud run services update-traffic "$SERVICE" \
            --region="$REGION" \
            --to-revisions="$NEW_REVISION=100" \
            --quiet
        
        # Final health checks
        sleep 30
        if ! perform_health_checks "$SERVICE_URL" 5 10; then
            log_error "Final health checks failed"
            return 1
        fi
        
        log_success "Blue-green deployment completed successfully!"
        
    else
        # First deployment - direct 100% traffic
        log_info "First deployment - directing 100% traffic to new revision"
        gcloud run services update-traffic "$SERVICE" \
            --region="$REGION" \
            --to-revisions="$NEW_REVISION=100" \
            --quiet
        
        # Extended validation for first deployment
        sleep 60
        if ! perform_health_checks "$SERVICE_URL" 5 15; then
            log_error "First deployment validation failed"
            return 1
        fi
        
        log_success "First deployment completed successfully!")
    fi
}

# Update external integrations
update_integrations() {
    log_header "🔗 Updating external integrations"
    
    SERVICE_URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)")
    
    # Update Telegram webhook if in production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        if gcloud secrets versions access latest --secret=TELEGRAM_TOKEN --project="$PROJECT_ID" >/dev/null 2>&1; then
            TELEGRAM_TOKEN=$(gcloud secrets versions access latest --secret=TELEGRAM_TOKEN --project="$PROJECT_ID")
            
            if curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook" \
                -d "url=$SERVICE_URL/webhook" \
                -d "drop_pending_updates=true" >/dev/null; then
                log_success "Telegram webhook updated for production"
            else
                log_warning "Failed to update Telegram webhook (non-critical)"
            fi
        fi
    fi
}

# Generate deployment report
generate_deployment_report() {
    log_header "📋 Deployment Report"
    
    echo "================================================"
    echo "Environment: $ENVIRONMENT"
    echo "Deployment ID: $DEPLOYMENT_ID"
    echo "Image: $IMAGE_NAME:$IMAGE_TAG"
    echo "Previous Revision: ${CURRENT_REVISION:-'none'}"
    echo "New Revision: $NEW_REVISION"
    echo "Service URL: $(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)")"
    echo "Health Check: $(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)")/health"
    echo "Deployment Time: $(date)"
    echo "================================================"
    
    log_info "📊 Post-deployment monitoring commands:"
    log_info "   View logs: gcloud run logs tail $SERVICE --region $REGION"
    log_info "   Monitor metrics: gcloud run services describe $SERVICE --region $REGION"
    log_info "   Check health: curl $(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)")/health"
}

# Main execution
main() {
    log_header "🔄 Starting Blue-Green Deployment for $ENVIRONMENT"
    
    # Validation
    validate_deployment_config
    
    # Deploy new revision
    deploy_new_revision
    
    # Execute traffic migration
    if execute_blue_green_migration; then
        # Update integrations
        update_integrations
        
        # Generate report
        generate_deployment_report
        
        log_success "🎉 Blue-green deployment completed successfully!"
        exit 0
    else
        log_error "❌ Blue-green deployment failed"
        exit 1
    fi
}

# Show usage if no arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <environment> [image_tag]"
    echo ""
    echo "Environments:"
    echo "  staging    - Deploy to staging environment with 50% -> 100% traffic migration"
    echo "  production - Deploy to production with 10% -> 50% -> 100% traffic migration"
    echo ""
    echo "Examples:"
    echo "  $0 staging latest"
    echo "  $0 production v1.2.3"
    echo "  $0 production 20250803-142030"
    exit 1
fi

# Validate environment parameter
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    log_error "Invalid environment: $ENVIRONMENT (must be 'staging' or 'production')"
    exit 1
fi

# Execute main function
main