#!/bin/bash
# Investigate corpus features issues

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 Investigating Corpus Features${NC}"
echo "=========================================="
echo

# Check if Cloud SQL Proxy is running
if ! pgrep -f "cloud-sql-proxy" > /dev/null; then
    echo -e "${YELLOW}Starting Cloud SQL Proxy...${NC}"
    ./cloud-sql-proxy --port 5433 \
        --credentials-file=/Users/kendo/.config/gcloud/application_default_credentials.json \
        shvyr-ai-bots:us-central1:shyvr-rlte-prod &
    PROXY_PID=$!
    sleep 3
    echo -e "${GREEN}✅ Cloud SQL Proxy started (PID: $PROXY_PID)${NC}"
else
    echo -e "${GREEN}✅ Cloud SQL Proxy already running${NC}"
fi

# Set environment variables
export DATABASE_URL="postgresql://rlte_user:DeFi2024_Secure@localhost:5433/shyvr_rlte_prod"
export SECRET_KEY="test_secret_key_for_development_only"
export COINGECKO_API_KEY="CG-2XJEjcZFTxrhGcQnKU9E7vJe"
export LUNARCRUSH_API_KEY="lwgnz4efvpp65dqvbj4xdnfhw36ebcrqr3bymqai"
export HELIUS_API_KEY="4e5f5e0d-0b1a-4b5e-8e5a-5c5d5e5f5e5d"
export ENVIRONMENT="development"
export DB_NAME="shyvr_rlte_prod"

echo -e "${GREEN}✅ Environment configured${NC}"
echo

# Run the investigation script
echo -e "${YELLOW}Running feature investigation...${NC}"
echo "=========================================="
uv run python scripts/data_collection/check_corpus_features.py

echo
echo -e "${GREEN}Investigation complete!${NC}"