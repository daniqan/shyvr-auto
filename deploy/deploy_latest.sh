#!/bin/bash

# Deploy Latest Shyvr RLTE
# Manual deployment script for AI-augmented cryptocurrency trading bot

set -e

# Configuration
PROJECT_ID="shvyr-ai-bots"
REPOSITORY="shyvr-ai-prod"
SERVICE="shyvr-rlte"
REGION="us-central1"
IMAGE_NAME="us-central1-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Deploying Shyvr RLTE to Cloud Run...${NC}"
echo -e "${PURPLE}🤖 AI-augmented cryptocurrency trading bot with RL capabilities${NC}"

# Get current time for tagging
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
TAG="latest"

echo -e "${YELLOW}📦 Building Docker image...${NC}"
echo -e "${BLUE}   Platform: linux/amd64 (optimized for ML workloads)${NC}"
docker build --platform linux/amd64 -t "$IMAGE_NAME:$TAG" -t "$IMAGE_NAME:$TIMESTAMP" .

echo -e "${YELLOW}📤 Pushing to Artifact Registry...${NC}"
docker push "$IMAGE_NAME:$TAG"
docker push "$IMAGE_NAME:$TIMESTAMP"

# Configure secrets from Secret Manager
echo -e "${YELLOW}🔐 Configuring comprehensive secret integration...${NC}"
echo -e "${BLUE}💡 To set up secrets, run: ./deploy/setup_secrets.sh${NC}"

# Define core system secrets (required for basic operation)
REQUIRED_SECRETS=("TELEGRAM_TOKEN" "WEBHOOK_SECRET" "DB_PASSWORD" "DATABASE_URL")

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

# Combine all optional secrets
OPTIONAL_SECRETS=(
    "${BLOCKCHAIN_SECRETS[@]}"
    "${AI_SECRETS[@]}"
    "${MARKET_SECRETS[@]}"
    "${SOCIAL_SECRETS[@]}"
)

# Function to validate and add secrets
validate_and_add_secret() {
    local secret_name=$1
    local is_required=${2:-false}
    local category=${3:-""}
    
    # Check if secret exists in Secret Manager
    if gcloud secrets describe "$secret_name" --project=$PROJECT_ID --quiet 2>/dev/null; then
        # Verify secret has a value
        local secret_value
        secret_value=$(gcloud secrets versions access latest --secret="$secret_name" --project=$PROJECT_ID --quiet 2>/dev/null)
        
        if [[ -n "$secret_value" && "$secret_value" != "null" ]]; then
            echo -e "${GREEN}✓${NC} Adding $category secret: $secret_name"
            SECRET_ARGS="$SECRET_ARGS --set-secrets $secret_name=$secret_name:latest"
            return 0
        else
            echo -e "${RED}❌${NC} Secret $secret_name exists but has no value"
            if [[ "$is_required" == "true" ]]; then
                echo -e "${RED}💥 DEPLOYMENT FAILED: Required secret $secret_name is empty${NC}"
                exit 1
            fi
            return 1
        fi
    else
        if [[ "$is_required" == "true" ]]; then
            echo -e "${RED}❌ DEPLOYMENT FAILED: Required secret $secret_name not found${NC}"
            echo -e "${BLUE}💡 Create it with: gcloud secrets create $secret_name --data-file=<(echo 'your_value')${NC}"
            exit 1
        else
            echo -e "${YELLOW}⚠${NC} Optional $category secret not configured: $secret_name"
            return 1
        fi
    fi
}

# Build secrets arguments for Cloud Run
SECRET_ARGS=""
SECRETS_ADDED=0
SECRETS_SKIPPED=0

# Process required secrets (must exist)
echo -e "${BLUE}🔍 Validating required secrets...${NC}"
for secret in "${REQUIRED_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" true "required"; then
        ((SECRETS_ADDED++))
    fi
done

# Process blockchain secrets
echo -e "${BLUE}🔗 Processing blockchain & RPC secrets...${NC}"
for secret in "${BLOCKCHAIN_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" false "blockchain"; then
        ((SECRETS_ADDED++))
    else
        ((SECRETS_SKIPPED++))
    fi
done

# Process AI/ML secrets
echo -e "${BLUE}🤖 Processing AI & ML API secrets...${NC}"
for secret in "${AI_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" false "AI/ML"; then
        ((SECRETS_ADDED++))
    else
        ((SECRETS_SKIPPED++))
    fi
done

# Process market data secrets
echo -e "${BLUE}📊 Processing market data & analytics secrets...${NC}"
for secret in "${MARKET_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" false "market data"; then
        ((SECRETS_ADDED++))
    else
        ((SECRETS_SKIPPED++))
    fi
done

# Process social media secrets
echo -e "${BLUE}📱 Processing social media API secrets...${NC}"
for secret in "${SOCIAL_SECRETS[@]}"; do
    if validate_and_add_secret "$secret" false "social media"; then
        ((SECRETS_ADDED++))
    else
        ((SECRETS_SKIPPED++))
    fi
done

# Handle trading secrets with extra caution
echo -e "${RED}⚠️  WARNING: Processing CRITICAL trading secrets...${NC}"
echo -e "${RED}   These secrets control real cryptocurrency wallets${NC}"
echo -e "${RED}   Only configure for production live trading mode${NC}"

TRADING_MODE=${TRADING_MODE:-"simulation"}
if [[ "$TRADING_MODE" == "live" ]]; then
    echo -e "${YELLOW}🔴 LIVE TRADING MODE ENABLED - Processing wallet secrets...${NC}"
    for secret in "${TRADING_SECRETS[@]}"; do
        if validate_and_add_secret "$secret" false "CRITICAL trading"; then
            ((SECRETS_ADDED++))
            echo -e "${RED}🔥 LIVE TRADING SECRET CONFIGURED: $secret${NC}"
        else
            ((SECRETS_SKIPPED++))
        fi
    done
else
    echo -e "${GREEN}✅ SIMULATION MODE - Skipping trading secrets for safety${NC}"
    SECRETS_SKIPPED=$((SECRETS_SKIPPED + ${#TRADING_SECRETS[@]}))
fi

# Summary
echo -e "${BLUE}📋 Secret configuration summary:${NC}"
echo -e "${GREEN}  ✅ Secrets configured: $SECRETS_ADDED${NC}"
echo -e "${YELLOW}  ⚠️  Secrets skipped: $SECRETS_SKIPPED${NC}"

if [[ $SECRETS_ADDED -eq 0 ]]; then
    echo -e "${RED}❌ No secrets configured - deployment may fail${NC}"
    exit 1
fi

# Configure Cloud SQL connection
CLOUD_SQL_INSTANCE="shvyr-ai-bots:us-central1:shyvr-rlte-db"

echo -e "${YELLOW}☁️  Deploying to Cloud Run with enhanced ML/RL configuration...${NC}"
echo -e "${BLUE}🗄️  Configuring Cloud SQL connection: $CLOUD_SQL_INSTANCE${NC}"
gcloud run deploy $SERVICE \
  --image "$IMAGE_NAME:$TAG" \
  --region $REGION \
  --platform managed \
  --port 8080 \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 50 \
  --timeout 900 \
  --max-instances 5 \
  --min-instances 0 \
  --allow-unauthenticated \
  --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO" \
  --add-cloudsql-instances "$CLOUD_SQL_INSTANCE" \
  $SECRET_ARGS \
  --quiet

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE --region $REGION --format="value(status.url)")

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${BLUE}📱 Service URL: $SERVICE_URL${NC}"
echo -e "${BLUE}🔍 Health Check: $SERVICE_URL/health${NC}"
echo -e "${BLUE}⚙️  Configuration: $SERVICE_URL/config${NC}"

# Test health endpoint
echo -e "${YELLOW}🔍 Testing health endpoint...${NC}"
sleep 10  # Extra time for ML models to load

if curl -f -s "$SERVICE_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
    
    # Test configuration endpoint
    echo -e "${YELLOW}🔧 Testing configuration endpoint...${NC}"
    if curl -f -s "$SERVICE_URL/config" > /dev/null; then
        echo -e "${GREEN}✅ Configuration endpoint responding!${NC}"
    else
        echo -e "${YELLOW}⚠${NC} Configuration endpoint not responding (may be normal)"
    fi
    
else
    echo -e "${RED}❌ Health check failed. Service may still be starting...${NC}"
    echo -e "${YELLOW}💡 ML models may take additional time to load (up to 2 minutes)${NC}"
fi

echo -e "${BLUE}🎉 Deployment process completed!${NC}"
echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "   1. Update Telegram webhook: ./scripts/set_webhook.sh $SERVICE_URL/webhook"
echo -e "   2. Verify trading modes are properly configured"
echo -e "   3. Check agent rules are loaded correctly"
echo -e "   4. Monitor system performance and ML model accuracy"

echo -e "${PURPLE}🤖 RLTE Features:${NC}"
echo -e "${GREEN}   ✓ Mode 1: Analysis & Reporting${NC}"
echo -e "${GREEN}   ✓ Mode 2: Simulation Trading${NC}"
echo -e "${YELLOW}   ⚠ Mode 3: Live Trading (disabled by default)${NC}"
echo -e "${GREEN}   ✓ AI Agent Integration${NC}"
echo -e "${GREEN}   ✓ ML/RL Models${NC}"
echo -e "${GREEN}   ✓ Multi-chain Support${NC}"

echo -e "${BLUE}📊 View logs: gcloud run logs tail $SERVICE --region $REGION${NC}"