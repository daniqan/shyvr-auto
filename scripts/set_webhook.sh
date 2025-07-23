#!/bin/bash

# Set Telegram Webhook for Shyvr RLTE
# Configure Telegram bot to send updates to the deployed service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if URL argument is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Error: No webhook URL provided${NC}"
    echo -e "${YELLOW}Usage: $0 <webhook_url>${NC}"
    echo -e "${BLUE}Example: $0 https://your-service-url.run.app/webhook${NC}"
    exit 1
fi

WEBHOOK_URL="$1"
TELEGRAM_TOKEN_SECRET="TELEGRAM_TOKEN"

echo -e "${BLUE}🔗 Setting up Telegram webhook for Shyvr RLTE...${NC}"
echo -e "${YELLOW}Webhook URL: $WEBHOOK_URL${NC}"

# Get Telegram token from Google Cloud Secret Manager
echo -e "${YELLOW}🔐 Retrieving Telegram token from Secret Manager...${NC}"
TELEGRAM_TOKEN=$(gcloud secrets versions access latest --secret="$TELEGRAM_TOKEN_SECRET" 2>/dev/null)

if [ -z "$TELEGRAM_TOKEN" ]; then
    echo -e "${RED}❌ Error: Could not retrieve Telegram token from Secret Manager${NC}"
    echo -e "${YELLOW}💡 Make sure the secret exists: gcloud secrets create $TELEGRAM_TOKEN_SECRET --data-file=<(echo 'your_token')${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Telegram token retrieved successfully${NC}"

# Set the webhook
echo -e "${YELLOW}📡 Setting Telegram webhook...${NC}"
RESPONSE=$(curl -s "https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook?url=$WEBHOOK_URL")

# Check if webhook was set successfully
if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo -e "${GREEN}✅ Webhook set successfully!${NC}"
    
    # Get webhook info to confirm
    echo -e "${YELLOW}🔍 Verifying webhook configuration...${NC}"
    WEBHOOK_INFO=$(curl -s "https://api.telegram.org/bot$TELEGRAM_TOKEN/getWebhookInfo")
    
    if echo "$WEBHOOK_INFO" | grep -q '"ok":true'; then
        echo -e "${GREEN}✓ Webhook verification successful${NC}"
        echo -e "${BLUE}📋 Webhook Details:${NC}"
        echo "$WEBHOOK_INFO" | python3 -m json.tool | grep -E "(url|pending_update_count|last_error)"
    else
        echo -e "${YELLOW}⚠ Could not verify webhook (but it may still be working)${NC}"
    fi
    
else
    echo -e "${RED}❌ Failed to set webhook${NC}"
    echo -e "${YELLOW}Response: $RESPONSE${NC}"
    exit 1
fi

echo -e "${BLUE}🎉 Telegram webhook configuration completed!${NC}"
echo -e "${YELLOW}💡 Test your bot by sending a message to it on Telegram${NC}"
echo -e "${BLUE}📊 Monitor webhook status with: curl -s \"https://api.telegram.org/bot\$TELEGRAM_TOKEN/getWebhookInfo\"${NC}"