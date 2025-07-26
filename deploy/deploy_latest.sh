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
echo -e "${YELLOW}🔐 Configuring secrets from Google Cloud Secret Manager...${NC}"
echo -e "${BLUE}💡 Make sure secrets are created in Secret Manager first:${NC}"
echo -e "   gcloud secrets create TELEGRAM_TOKEN --data-file=<(echo 'your_token')"

# Define required secrets for RLTE
REQUIRED_SECRETS=("TELEGRAM_TOKEN" "WEBHOOK_SECRET" "DB_PASSWORD")

# Define optional secrets for RLTE (APIs, ML models, etc.)
OPTIONAL_SECRETS=(
    # Social/Market APIs
    "X_BEARER_TOKEN" 
    "X_API_KEY" 
    "X_API_SECRET"
    
    # Blockchain APIs (Note: Base now uses Etherscan API v2, same key as Ethereum)
    "ETHERSCAN_API_KEY" 
    "HELIUS_API_KEY" 
    "BIRDEYE_API_KEY"
    
    # AI/ML APIs
    "XAI_API_KEY" 
    "OPENAI_API_KEY"
    "AGENT_API_KEY"
    
    # Database and Infrastructure
    "DATABASE_URL"
    
    # Trading (for live mode - disabled by default)
    "SOLANA_RPC_URL"
    "ETHEREUM_RPC_URL"
    "WALLET_PRIVATE_KEY"
)

# Build secrets arguments for Cloud Run
SECRET_ARGS=""

# Add required secrets
for secret in "${REQUIRED_SECRETS[@]}"; do
    echo -e "${GREEN}✓${NC} Adding required secret: $secret"
    SECRET_ARGS="$SECRET_ARGS --set-secrets $secret=$secret:latest"
done

# Add optional secrets (check if they exist in Secret Manager)
for secret in "${OPTIONAL_SECRETS[@]}"; do
    # Check if secret exists in Secret Manager
    if gcloud secrets describe "$secret" --quiet 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Adding optional secret: $secret"
        SECRET_ARGS="$SECRET_ARGS --set-secrets $secret=$secret:latest"
    else
        echo -e "${YELLOW}⚠${NC} Optional secret not found in Secret Manager: $secret"
    fi
done

echo -e "${YELLOW}☁️  Deploying to Cloud Run with enhanced ML/RL configuration...${NC}"
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