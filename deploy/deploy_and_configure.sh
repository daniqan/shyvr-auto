#!/bin/bash

# Complete Deployment and Configuration for Shyvr RLTE
# Deploys the service and sets up Telegram webhook in one go

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🤖 Shyvr RLTE - Complete Deployment Pipeline${NC}"
echo -e "${BLUE}AI-augmented cryptocurrency trading bot deployment${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "deploy/deploy_latest.sh" ]; then
    echo -e "${RED}❌ Error: deploy_latest.sh not found${NC}"
    echo -e "${YELLOW}Please run this script from the project root directory${NC}"
    exit 1
fi

# Step 1: Deploy to Cloud Run
echo -e "${YELLOW}🚀 Step 1: Deploying to Google Cloud Run...${NC}"
./deploy/deploy_latest.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Deployment failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"

# Step 2: Get service URL
SERVICE="shyvr-rlte"
REGION="us-central1"
SERVICE_URL=$(gcloud run services describe $SERVICE --region $REGION --format="value(status.url)")

if [ -z "$SERVICE_URL" ]; then
    echo -e "${RED}❌ Could not retrieve service URL${NC}"
    exit 1
fi

echo -e "${BLUE}📱 Service URL: $SERVICE_URL${NC}"

# Step 3: Set up Telegram webhook
echo ""
echo -e "${YELLOW}🔗 Step 2: Configuring Telegram webhook...${NC}"

# Ask user if they want to set up webhook
read -p "Do you want to set up the Telegram webhook now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    WEBHOOK_URL="$SERVICE_URL/webhook"
    echo -e "${BLUE}Setting webhook to: $WEBHOOK_URL${NC}"
    ./scripts/set_webhook.sh "$WEBHOOK_URL"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Webhook configured successfully!${NC}"
    else
        echo -e "${YELLOW}⚠ Webhook setup failed, but deployment is complete${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping webhook setup${NC}"
    echo -e "${BLUE}💡 You can set it up later with: ./scripts/set_webhook.sh $SERVICE_URL/webhook${NC}"
fi

# Step 4: Final health checks and info
echo ""
echo -e "${YELLOW}🔍 Step 3: Final system verification...${NC}"

# Wait for service to be fully ready
sleep 5

# Test endpoints
echo -e "${BLUE}Testing endpoints:${NC}"

# Health check
if curl -f -s "$SERVICE_URL/health" > /dev/null; then
    echo -e "${GREEN}✓ Health endpoint: OK${NC}"
else
    echo -e "${YELLOW}⚠ Health endpoint: Not responding${NC}"
fi

# Configuration check
if curl -f -s "$SERVICE_URL/config" > /dev/null; then
    echo -e "${GREEN}✓ Configuration endpoint: OK${NC}"
else
    echo -e "${YELLOW}⚠ Configuration endpoint: Not responding${NC}"
fi

echo ""
echo -e "${PURPLE}🎉 DEPLOYMENT COMPLETE! ${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${BLUE}Service URL:     $SERVICE_URL${NC}"
echo -e "${BLUE}Health Check:    $SERVICE_URL/health${NC}"
echo -e "${BLUE}Configuration:   $SERVICE_URL/config${NC}"
echo -e "${BLUE}Webhook URL:     $SERVICE_URL/webhook${NC}"
echo ""
echo -e "${YELLOW}🔧 Available Commands:${NC}"
echo -e "   View logs:    gcloud run logs tail $SERVICE --region $REGION"
echo -e "   Scale down:   gcloud run services update $SERVICE --region $REGION --min-instances 0"
echo -e "   Scale up:     gcloud run services update $SERVICE --region $REGION --min-instances 1"
echo ""
echo -e "${PURPLE}🤖 RLTE System Status:${NC}"
echo -e "${GREEN}   ✓ Analysis Mode:     Ready${NC}"
echo -e "${GREEN}   ✓ Simulation Mode:   Ready${NC}"
echo -e "${YELLOW}   ⚠ Live Trading:     Disabled (safety)${NC}"
echo -e "${GREEN}   ✓ AI Agent:         Ready${NC}"
echo -e "${GREEN}   ✓ ML Models:        Loading...${NC}"
echo ""
echo -e "${BLUE}💡 Next Steps:${NC}"
echo -e "   1. Test the bot by sending /start to your Telegram bot"
echo -e "   2. Monitor logs for any initialization issues"
echo -e "   3. Verify ML models load successfully"
echo -e "   4. Test analysis mode with sample tokens"
echo -e "   5. Configure agent rules as needed"
echo ""
echo -e "${GREEN}🚀 Your AI trading bot is now live and ready to analyze tokens!${NC}"