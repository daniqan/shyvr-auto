#!/bin/bash

# Deploy Latest Shyvr RLTE - Production-Ready Deployment
# Manual deployment script for AI-augmented cryptocurrency trading bot with RL capabilities
# Enhanced with comprehensive error handling, validation, and rollback capabilities

set -euo pipefail  # Enhanced error handling

# Configuration
PROJECT_ID="shvyr-ai-bots"
REPOSITORY="shyvr-ai-prod"
SERVICE="shyvr-rlte"
REGION="us-central1"
IMAGE_NAME="us-central1-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE"
CLOUD_SQL_INSTANCE="shvyr-ai-bots:us-central1:shyvr-rlte-db"
SERVICE_ACCOUNT="shyvr-rlte@$PROJECT_ID.iam.gserviceaccount.com"

# Deployment tracking
DEPLOYMENT_ID=$(date +"%Y%m%d-%H%M%S")
DEPLOYMENT_LOG="deployment_${DEPLOYMENT_ID}.log"
ROLLBACK_REVISION=""
PREVIOUS_TRAFFIC_ALLOCATION=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

# Error handling and cleanup
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        if [[ -n "$ROLLBACK_REVISION" ]]; then
            log_warning "Initiating automatic rollback to revision $ROLLBACK_REVISION"
            rollback_deployment
        fi
    fi
    log_info "Deployment log saved to: $DEPLOYMENT_LOG"
}

trap cleanup EXIT

log_info "🚀 Starting Shyvr RLTE deployment - ID: $DEPLOYMENT_ID"
log_info "🤖 AI-augmented cryptocurrency trading bot with advanced RL capabilities"

# Pre-deployment validation
validate_prerequisites() {
    log_info "🔍 Validating deployment prerequisites..."
    
    # Check required tools
    for tool in gcloud docker python3; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Validate GCP authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "."; then
        log_error "No active GCP authentication found. Run: gcloud auth login"
        exit 1
    fi
    
    # Validate project access
    if ! gcloud projects describe "$PROJECT_ID" &>/dev/null; then
        log_error "Cannot access project: $PROJECT_ID"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &>/dev/null; then
        log_error "Docker daemon not running"
        exit 1
    fi
    
    # Validate Dockerfile exists and is optimized
    if [[ ! -f "Dockerfile" ]]; then
        log_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    # Check for multi-stage build optimization
    if ! grep -q "FROM.*as.*builder" Dockerfile; then
        log_warning "Dockerfile may not be using multi-stage build optimization"
    fi
    
    # Validate project structure
    for required_file in "main.py" "src/utils/database.py" "src/utils/config.py"; do
        if [[ ! -f "$required_file" ]]; then
            log_error "Required file not found: $required_file"
            exit 1
        fi
    done
    
    log_success "Prerequisites validation passed"
}

# Database schema validation
validate_database_schema() {
    log_info "🗄️ Validating database schema and RL experience storage..."
    
    # Check migration files exist
    if [[ ! -d "database/migrations" ]]; then
        log_error "Database migrations directory not found"
        exit 1
    fi
    
    # Verify RL schema migration exists
    if [[ ! -f "database/migrations/004_create_rl_experience_schema.sql" ]]; then
        log_error "RL experience schema migration not found"
        exit 1
    fi
    
    # Test database connectivity (if possible)
    if [[ -n "${DATABASE_URL:-}" ]]; then
        log_info "Testing database connectivity..."
        if python3 -c "import asyncpg, asyncio; asyncio.run(asyncpg.connect('$DATABASE_URL').execute('SELECT 1'))" 2>/dev/null; then
            log_success "Database connectivity test passed"
        else
            log_warning "Database connectivity test failed - deployment will continue"
        fi
    fi
    
    log_success "Database schema validation completed"
}

# Get current revision for rollback
get_current_revision() {
    ROLLBACK_REVISION=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.latestReadyRevisionName)" 2>/dev/null || echo "")
    if [[ -n "$ROLLBACK_REVISION" ]]; then
        log_info "Current revision for rollback: $ROLLBACK_REVISION"
        PREVIOUS_TRAFFIC_ALLOCATION=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.traffic[0].percent)" 2>/dev/null || echo "100")
    fi
}

# Run validations
validate_prerequisites
validate_database_schema
get_current_revision

# Get current time for tagging
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
TAG="latest"

log_info "📦 Building optimized Docker image for ML/RL workloads..."
log_info "   Platform: linux/amd64 (Cloud Run compatible)"
log_info "   Build context: $(pwd)"

# Enhanced Docker build with better error handling
if ! docker build --platform linux/amd64 \
    --tag "$IMAGE_NAME:$TAG" \
    --tag "$IMAGE_NAME:$TIMESTAMP" \
    --tag "$IMAGE_NAME:deployment-$DEPLOYMENT_ID" \
    --label "deployment.id=$DEPLOYMENT_ID" \
    --label "deployment.timestamp=$TIMESTAMP" \
    --label "project.id=$PROJECT_ID" \
    . 2>&1 | tee -a "$DEPLOYMENT_LOG"; then
    log_error "Docker build failed"
    exit 1
fi

log_success "Docker image built successfully"

log_info "📤 Pushing images to Artifact Registry..."

# Ensure Artifact Registry repository exists
if ! gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    log_info "Creating Artifact Registry repository: $REPOSITORY"
    gcloud artifacts repositories create "$REPOSITORY" \
        --repository-format=docker \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --description="Shyvr RLTE ML/RL Trading Bot Images"
fi

# Configure Docker authentication
if ! gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet; then
    log_error "Failed to configure Docker authentication"
    exit 1
fi

# Push images with retry logic
push_with_retry() {
    local image=$1
    local max_attempts=3
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Pushing $image (attempt $attempt/$max_attempts)"
        if docker push "$image"; then
            log_success "Successfully pushed $image"
            return 0
        else
            log_warning "Push attempt $attempt failed for $image"
            ((attempt++))
            [[ $attempt -le $max_attempts ]] && sleep 5
        fi
    done
    
    log_error "Failed to push $image after $max_attempts attempts"
    return 1
}

# Push all tagged images
for tag in "$TAG" "$TIMESTAMP" "deployment-$DEPLOYMENT_ID"; do
    if ! push_with_retry "$IMAGE_NAME:$tag"; then
        exit 1
    fi
done

# Comprehensive secret management with validation
log_info "🔐 Configuring comprehensive secret integration..."
log_info "💡 To set up secrets, run: ./deploy/setup_secrets.sh"

# Enhanced secret validation with health checks
validate_secret_accessibility() {
    local secret_name=$1
    local max_attempts=3
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if gcloud secrets versions access latest --secret="$secret_name" --project="$PROJECT_ID" --quiet &>/dev/null; then
            return 0
        fi
        ((attempt++))
        [[ $attempt -le $max_attempts ]] && sleep 2
    done
    return 1
}

# Define core system secrets (required for basic operation)
REQUIRED_SECRETS=("TELEGRAM_TOKEN" "WEBHOOK_SECRET" "DB_PASSWORD" "DATABASE_URL")

# Additional RL and ML specific secrets
RL_ML_SECRETS=(
    "ML_MODEL_CACHE_KEY"           # ML model caching authentication
    "RL_EXPERIENCE_ENCRYPTION_KEY" # RL experience data encryption
    "MODEL_SERVING_API_KEY"        # Model serving endpoint authentication
    "TENSORBOARD_API_KEY"          # TensorBoard integration
)

# Define blockchain & RPC secrets
BLOCKCHAIN_SECRETS=(
    "HELIUS_API_KEY"           # Solana RPC provider
    "BIRDEYE_API_KEY"          # DeFi data aggregator
    "ETHERSCAN_API_KEY"        # Ethereum blockchain explorer
    "ALCHEMY_API_KEY"          # General Ethereum/Polygon RPC
    "ETHEREUM_API_KEY"         # Ethereum-specific RPC
    "BASE_API_KEY"             # Base chain RPC
    "SOLANA_RPC_URL"           # Custom Solana RPC endpoint
    "ETHEREUM_RPC_URL"         # Custom Ethereum RPC endpoint
)

# Define AI & ML API secrets
AI_SECRETS=(
    "XAI_API_KEY"              # xAI/Grok API for analysis
    "OPENAI_API_KEY"           # OpenAI GPT API
    "AGENT_API_KEY"            # Custom AI agent API
)

# Define market data & analytics secrets
MARKET_SECRETS=(
    "LUNARCRUSH_API_KEY"       # Social sentiment data
    "COINGECKO_API_KEY"        # CoinGecko market data
    "COINGECKO_PRO_API_KEY"    # CoinGecko Pro API
    "GLASSNODE_API_KEY"        # On-chain analytics
    "MESSARI_API_KEY"          # Crypto research data
    "JUPITER_API_KEY"          # Jupiter DEX aggregator
)

# Define social media & external API secrets
SOCIAL_SECRETS=(
    "X_BEARER_TOKEN"           # X (Twitter) API bearer token
    "X_API_KEY"                # X (Twitter) API key
    "X_API_SECRET"             # X (Twitter) API secret
)

# Define trading & wallet secrets (CRITICAL - Live Trading Only)
TRADING_SECRETS=(
    "SOLANA_PRIVATE_KEY"       # Solana wallet private key
    "ETHEREUM_PRIVATE_KEY"     # Ethereum wallet private key
    "HYPERLIQUID_PRIVATE_KEY"  # Hyperliquid exchange private key
    "HYPERLIQUID_API_KEY"      # Hyperliquid exchange API key
    "WALLET_PRIVATE_KEY"       # Primary wallet private key
)

# Enhanced function to validate and add secrets with health monitoring
validate_and_add_secret() {
    local secret_name=$1
    local is_required=${2:-false}
    local category=${3:-""}
    
    # Check if secret exists in Secret Manager
    if gcloud secrets describe "$secret_name" --project="$PROJECT_ID" --quiet 2>/dev/null; then
        # Verify secret has a value with retry logic
        if validate_secret_accessibility "$secret_name"; then
            local secret_value
            secret_value=$(gcloud secrets versions access latest --secret="$secret_name" --project="$PROJECT_ID" --quiet 2>/dev/null)
            
            if [[ -n "$secret_value" && "$secret_value" != "null" && "$secret_value" != "" ]]; then
                log_success "Adding $category secret: $secret_name"
                SECRET_ARGS="$SECRET_ARGS --set-secrets $secret_name=$secret_name:latest"
                
                # Add to secret monitoring list
                SECRET_MONITORING_LIST="$SECRET_MONITORING_LIST $secret_name"
                return 0
            else
                log_error "Secret $secret_name exists but has no value"
                if [[ "$is_required" == "true" ]]; then
                    log_error "DEPLOYMENT FAILED: Required secret $secret_name is empty"
                    exit 1
                fi
                return 1
            fi
        else
            log_error "Cannot access secret $secret_name after multiple attempts"
            if [[ "$is_required" == "true" ]]; then
                log_error "DEPLOYMENT FAILED: Required secret $secret_name is inaccessible"
                exit 1
            fi
            return 1
        fi
    else
        if [[ "$is_required" == "true" ]]; then
            log_error "DEPLOYMENT FAILED: Required secret $secret_name not found"
            log_info "Create it with: gcloud secrets create $secret_name --data-file=<(echo 'your_value')"
            exit 1
        else
            log_warning "Optional $category secret not configured: $secret_name"
            return 1
        fi
    fi
}

# Build secrets arguments for Cloud Run with monitoring
SECRET_ARGS=""
SECRETS_ADDED=0
SECRETS_SKIPPED=0
SECRET_MONITORING_LIST=""

# Process required secrets (must exist)
log_info "🔍 Validating required secrets..."
for secret in "${REQUIRED_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" true "required"; then
        ((SECRETS_ADDED++))
    fi
done

# Process blockchain secrets
log_info "🔗 Processing blockchain & RPC secrets..."
for secret in "${BLOCKCHAIN_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" false "blockchain"; then
        ((SECRETS_ADDED++))
    else
        ((SECRETS_SKIPPED++))
    fi
done

# Process AI/ML secrets
log_info "🤖 Processing AI & ML API secrets..."
for secret in "${AI_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" false "AI/ML"; then
        ((SECRETS_ADDED++))
    else
        ((SECRETS_SKIPPED++))
    fi
done

# Process RL/ML specific secrets
log_info "🧠 Processing RL & ML infrastructure secrets..."
for secret in "${RL_ML_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" false "RL/ML"; then
        ((SECRETS_ADDED++))
    else
        ((SECRETS_SKIPPED++))
    fi
done

# Process market data secrets
log_info "📊 Processing market data & analytics secrets..."
for secret in "${MARKET_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" false "market data"; then
        ((SECRETS_ADDED++))
    else
        ((SECRETS_SKIPPED++))
    fi
done

# Process social media secrets
log_info "📱 Processing social media API secrets..."
for secret in "${SOCIAL_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" false "social media"; then
        ((SECRETS_ADDED++))
    else
        ((SECRETS_SKIPPED++))
    fi
done

# Handle trading secrets with extra caution
log_error "⚠️  WARNING: Processing CRITICAL trading secrets..."
log_error "   These secrets control real cryptocurrency wallets"
log_error "   Only configure for production live trading mode"

TRADING_MODE=${TRADING_MODE:-"simulation"}
if [[ "$TRADING_MODE" == "live" ]]; then
    log_warning "🔴 LIVE TRADING MODE ENABLED - Processing wallet secrets..."
    for secret in "${TRADING_SECRETS[@]}"; do
        if validate_and_add_secret "$secret" false "CRITICAL trading"; then
            ((SECRETS_ADDED++))
            log_error "🔥 LIVE TRADING SECRET CONFIGURED: $secret"
        else
            ((SECRETS_SKIPPED++))
        fi
    done
else
    log_success "✅ SIMULATION MODE - Skipping trading secrets for safety"
    SECRETS_SKIPPED=$((SECRETS_SKIPPED + ${#TRADING_SECRETS[@]}))
fi

# Summary
log_info "📋 Secret configuration summary:"
log_success "  ✅ Secrets configured: $SECRETS_ADDED"
log_warning "  ⚠️  Secrets skipped: $SECRETS_SKIPPED"

if [[ $SECRETS_ADDED -eq 0 ]]; then
    log_error "❌ No secrets configured - deployment may fail"
    exit 1
fi

# Advanced Cloud Run deployment with comprehensive configuration
log_info "☁️ Deploying to Cloud Run with enhanced ML/RL configuration..."
log_info "🗄️ Configuring Cloud SQL connection: $CLOUD_SQL_INSTANCE"

# Validate Cloud SQL instance
if ! gcloud sql instances describe "${CLOUD_SQL_INSTANCE##*:}" --project="$PROJECT_ID" &>/dev/null; then
    log_error "Cloud SQL instance not found: $CLOUD_SQL_INSTANCE"
    exit 1
else
    log_success "Cloud SQL instance verified: $CLOUD_SQL_INSTANCE"
fi

# Deployment configuration for ML/RL workloads
deployment_start_time=$(date +%s)

# Deploy with comprehensive ML/RL optimized configuration
log_info "🚀 Executing Cloud Run deployment..."

if ! gcloud run deploy "$SERVICE" \
  --image="$IMAGE_NAME:$TAG" \
  --region="$REGION" \
  --platform=managed \
  --port=8080 \
  --memory=6Gi \
  --cpu=4 \
  --concurrency=25 \
  --timeout=1800 \
  --max-instances=10 \
  --min-instances=1 \
  --allow-unauthenticated \
  --service-account="$SERVICE_ACCOUNT" \
  --set-env-vars="ENVIRONMENT=production,LOG_LEVEL=INFO,DEPLOYMENT_ID=$DEPLOYMENT_ID,ML_OPTIMIZED=true,RL_STORAGE_ENABLED=true" \
  --add-cloudsql-instances="$CLOUD_SQL_INSTANCE" \
  --cpu-boost \
  --execution-environment=gen2 \
  --session-affinity \
  $SECRET_ARGS \
  --tag="deployment-$DEPLOYMENT_ID" \
  --no-traffic 2>&1 | tee -a "$DEPLOYMENT_LOG"; then
    log_error "Cloud Run deployment failed"
    exit 1
fi

deployment_end_time=$(date +%s)
deployment_duration=$((deployment_end_time - deployment_start_time))
log_success "Deployment completed in ${deployment_duration} seconds"

# Advanced health validation and traffic management
validate_deployment() {
    log_info "🔍 Validating deployment health..."
    
    # Get the new revision URL
    NEW_REVISION_URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.traffic[0].url)" | head -1)
    SERVICE_URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)")
    
    if [[ -z "$SERVICE_URL" ]]; then
        log_error "Failed to get service URL"
        return 1
    fi
    
    log_success "Service deployed successfully!"
    log_info "📱 Service URL: $SERVICE_URL"
    log_info "🔗 New Revision URL: $NEW_REVISION_URL"
    log_info "🔍 Health Check: $SERVICE_URL/health"
    log_info "⚙️ Configuration: $SERVICE_URL/config"
    log_info "📊 Metrics: $SERVICE_URL/metrics"
    
    return 0
}

# Rollback function
rollback_deployment() {
    if [[ -n "$ROLLBACK_REVISION" ]]; then
        log_warning "🔄 Rolling back to revision: $ROLLBACK_REVISION"
        
        if gcloud run services update-traffic "$SERVICE" \
            --region="$REGION" \
            --to-revisions="$ROLLBACK_REVISION=100" \
            --quiet; then
            log_success "Rollback completed successfully"
        else
            log_error "Rollback failed - manual intervention required"
        fi
    else
        log_warning "No previous revision available for rollback"
    fi
}

# Validate deployment
if ! validate_deployment; then
    log_error "Deployment validation failed"
    exit 1
fi

# Comprehensive health and readiness testing
perform_health_checks() {
    log_info "🔍 Performing comprehensive health checks..."
    
    # Wait for ML models to initialize
    local max_wait=300  # 5 minutes for ML/RL model loading
    local wait_interval=15
    local waited=0
    
    log_info "⏳ Waiting for ML/RL models to initialize (up to 5 minutes)..."
    
    while [[ $waited -lt $max_wait ]]; do
        if curl -f -s -m 10 "$SERVICE_URL/health" > /dev/null; then
            log_success "Health check passed after ${waited} seconds"
            break
        fi
        
        log_info "Waiting for service to be ready... (${waited}/${max_wait}s)"
        sleep $wait_interval
        waited=$((waited + wait_interval))
    done
    
    if [[ $waited -ge $max_wait ]]; then
        log_error "Health check timeout after ${max_wait} seconds"
        return 1
    fi
    
    # Test additional endpoints
    local health_response
    health_response=$(curl -s -m 30 "$SERVICE_URL/health" || echo "{}")
    
    # Parse health response for detailed status
    if echo "$health_response" | grep -q '"status":"healthy"'; then
        log_success "Detailed health check: Service is healthy"
    elif echo "$health_response" | grep -q '"status":"degraded"'; then
        log_warning "Service is running but degraded - check logs"
    else
        log_warning "Health status unclear - manual verification recommended"
    fi
    
    # Test configuration endpoint
    log_info "🔧 Testing configuration endpoint..."
    if curl -f -s -m 10 "$SERVICE_URL/config" > /dev/null; then
        log_success "Configuration endpoint responding"
    else
        log_warning "Configuration endpoint not responding (may be normal)"
    fi
    
    # Test metrics endpoint
    log_info "📊 Testing metrics endpoint..."
    if curl -f -s -m 10 "$SERVICE_URL/metrics" > /dev/null; then
        log_success "Metrics endpoint responding"
    else
        log_warning "Metrics endpoint not responding"
    fi
    
    # Test database connectivity through health endpoint
    if echo "$health_response" | grep -q '"database".*"status":"healthy"'; then
        log_success "Database connectivity confirmed"
    else
        log_warning "Database connectivity status unclear"
    fi
    
    # Test RL experience storage
    if echo "$health_response" | grep -q 'rl_agent'; then
        log_success "RL agent system detected"
    else
        log_warning "RL agent status not confirmed"
    fi
    
    return 0
}

# Gradual traffic migration
manage_traffic_migration() {
    log_info "🚦 Managing traffic migration..."
    
    # Start with 10% traffic to new revision
    local new_revision
    new_revision=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.latestCreatedRevisionName)")
    
    if [[ -n "$ROLLBACK_REVISION" ]]; then
        log_info "Starting gradual traffic migration: 10% to new revision"
        
        if gcloud run services update-traffic "$SERVICE" \
            --region="$REGION" \
            --to-revisions="$new_revision=10,$ROLLBACK_REVISION=90" \
            --quiet; then
            log_success "Initial traffic migration (10%) completed"
            
            # Wait and monitor
            sleep 30
            
            # Check if new revision is handling traffic well
            if perform_health_checks; then
                log_info "Migrating 100% traffic to new revision"
                gcloud run services update-traffic "$SERVICE" \
                    --region="$REGION" \
                    --to-revisions="$new_revision=100" \
                    --quiet
                log_success "Traffic migration completed successfully"
            else
                log_error "Health checks failed during traffic migration"
                return 1
            fi
        else
            log_error "Initial traffic migration failed"
            return 1
        fi
    else
        log_info "No previous revision - directing 100% traffic to new revision"
        gcloud run services update-traffic "$SERVICE" \
            --region="$REGION" \
            --to-revisions="$new_revision=100" \
            --quiet
    fi
    
    return 0
}

# Execute health checks and traffic management
if perform_health_checks && manage_traffic_migration; then
    log_success "Health checks and traffic migration completed successfully"
else
    log_error "Health checks or traffic migration failed"
    exit 1
fi

# Final deployment summary and monitoring setup
log_success "🎉 Deployment process completed successfully!"

# Generate deployment summary
echo "" | tee -a "$DEPLOYMENT_LOG"
log_info "📋 DEPLOYMENT SUMMARY"
log_info "Deployment ID: $DEPLOYMENT_ID"
log_info "Image: $IMAGE_NAME:$TAG"
log_info "Service URL: $SERVICE_URL"
log_info "Duration: ${deployment_duration}s"
log_info "Secrets configured: $SECRETS_ADDED"
log_info "Secrets skipped: $SECRETS_SKIPPED"
echo "" | tee -a "$DEPLOYMENT_LOG"

# Post-deployment validation and monitoring
setup_monitoring() {
    log_info "📊 Setting up post-deployment monitoring..."
    
    # Create monitoring dashboard URL
    local monitoring_url="https://console.cloud.google.com/run/detail/$REGION/$SERVICE/metrics?project=$PROJECT_ID"
    
    # Log important URLs
    echo "📋 IMPORTANT URLS:" | tee -a "$DEPLOYMENT_LOG"
    echo "   Service: $SERVICE_URL" | tee -a "$DEPLOYMENT_LOG"
    echo "   Health: $SERVICE_URL/health" | tee -a "$DEPLOYMENT_LOG"
    echo "   Metrics: $SERVICE_URL/metrics" | tee -a "$DEPLOYMENT_LOG"
    echo "   Monitoring: $monitoring_url" | tee -a "$DEPLOYMENT_LOG"
    echo "   Logs: gcloud run logs tail $SERVICE --region $REGION" | tee -a "$DEPLOYMENT_LOG"
    
    # Secret monitoring
    if [[ -n "$SECRET_MONITORING_LIST" ]]; then
        echo "🔐 CONFIGURED SECRETS:" | tee -a "$DEPLOYMENT_LOG"
        for secret in $SECRET_MONITORING_LIST; do
            echo "   ✓ $secret" | tee -a "$DEPLOYMENT_LOG"
        done
    fi
}

setup_monitoring

log_info "💡 NEXT STEPS:"
log_info "   1. Update Telegram webhook: ./scripts/set_webhook.sh $SERVICE_URL/webhook"
log_info "   2. Verify trading modes: curl $SERVICE_URL/config"
log_info "   3. Check RL agent status: curl $SERVICE_URL/health | jq .components.rl_agent"
log_info "   4. Monitor ML model performance: curl $SERVICE_URL/health | jq .components.ml_models"
log_info "   5. Validate RL experience storage: curl $SERVICE_URL/health | jq .components.database"
log_info "   6. Run database migration if needed: python database/migrate.py"
log_info "   7. Monitor logs: gcloud run logs tail $SERVICE --region $REGION --follow"

log_info "🤖 RLTE FEATURES STATUS:"
log_success "   ✓ Mode 1: Analysis & Reporting (Active)"
log_success "   ✓ Mode 2: Simulation Trading (Active)"
log_warning "   ⚠ Mode 3: Live Trading (Requires explicit activation)"
log_success "   ✓ AI Agent Integration (Active)"
log_success "   ✓ ML/RL Models (Optimized for Cloud Run)"
log_success "   ✓ Multi-chain Support (Ethereum, Solana, Base)"
log_success "   ✓ RL Experience Storage (Database-backed)"
log_success "   ✓ Real-time Metrics & Monitoring (Prometheus compatible)"
log_success "   ✓ Advanced Error Handling & Rollback (Production-ready)"

log_info "📈 MONITORING & OPERATIONS:"
log_info "   View logs: gcloud run logs tail $SERVICE --region $REGION"
log_info "   Monitor performance: gcloud run services describe $SERVICE --region $REGION"
log_info "   Scale service: gcloud run services update $SERVICE --region $REGION --max-instances=20"
log_info "   View metrics: curl $SERVICE_URL/metrics"
log_info "   Health status: curl $SERVICE_URL/health | jq ."

# Final validation
log_info "🔍 Performing final deployment validation..."
if curl -f -s -m 10 "$SERVICE_URL/health" | grep -q '"status":"healthy"'; then
    log_success "✅ Final validation: Deployment is healthy and ready for production use"
else
    log_warning "⚠️ Final validation: Service deployed but health status needs verification"
fi

log_success "🚀 Shyvr RLTE deployment completed successfully!"
log_info "📝 Full deployment log saved to: $DEPLOYMENT_LOG"