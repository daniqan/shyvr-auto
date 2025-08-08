#!/bin/bash
#
# Initial Corpus Collection Runner
# Sets up environment and runs the corpus collection script
#
# Usage:
#   ./scripts/data_collection/collect_initial_corpus.sh [test|production|custom]
#
# Examples:
#   # Test run (2 tokens, 7 days)
#   ./scripts/data_collection/collect_initial_corpus.sh test
#   
#   # Production run (10 tokens, 180 days) 
#   ./scripts/data_collection/collect_initial_corpus.sh production
#   
#   # Custom run
#   ./scripts/data_collection/collect_initial_corpus.sh custom --tokens bitcoin ethereum --days 30

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 Initial Corpus Collection Setup${NC}"
echo ""

# Determine environment
MODE=${1:-test}
shift || true  # Remove first argument to pass remaining to Python script

# Check if Cloud SQL proxy is running
if ! lsof -i:5433 > /dev/null 2>&1; then
    echo -e "${YELLOW}📡 Starting Cloud SQL Proxy...${NC}"
    
    # Check if proxy binary exists
    if [ -f ~/cloud-sql-proxy ]; then
        ~/cloud-sql-proxy --port=5433 shvyr-ai-bots:us-central1:shyvr-rlte-db-prod &
        PROXY_PID=$!
        sleep 5
        echo -e "${GREEN}✅ Cloud SQL Proxy started (PID: $PROXY_PID)${NC}"
    else
        echo -e "${RED}❌ Cloud SQL Proxy not found at ~/cloud-sql-proxy${NC}"
        echo "Please install it first:"
        echo "  curl -o ~/cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.11.0/cloud-sql-proxy.darwin.amd64"
        echo "  chmod +x ~/cloud-sql-proxy"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Cloud SQL Proxy already running${NC}"
fi

# Get production database password from Secret Manager
echo -e "${YELLOW}🔑 Fetching secrets from GCP...${NC}"
DB_PASS=$(gcloud secrets versions access latest --secret="DB_PASSWORD" --project=shvyr-ai-bots 2>/dev/null || echo "")

if [ -z "$DB_PASS" ]; then
    echo -e "${RED}❌ Failed to fetch database password from Secret Manager${NC}"
    echo "Please ensure you're authenticated with gcloud and have access to the project"
    exit 1
fi

# Set up environment variables
export DATABASE_URL="postgresql://rlte_prod_user:${DB_PASS}@localhost:5433/shyvr_rlte_prod"
export DB_HOST="localhost"
export DB_PORT="5433"
export DB_USER="rlte_prod_user"
export DB_PASSWORD="${DB_PASS}"
export DB_NAME="shyvr_rlte_prod"

# Use development environment to allow localhost connection
export ENVIRONMENT="development"
export SECRET_KEY="production_secret_key_for_corpus_collection_32ch"

# Fetch API keys from Secret Manager
export COINGECKO_API_KEY=$(gcloud secrets versions access latest --secret="COINGECKO_API_KEY" --project=shvyr-ai-bots 2>/dev/null)
export LUNARCRUSH_API_KEY=$(gcloud secrets versions access latest --secret="LUNARCRUSH_API_KEY" --project=shvyr-ai-bots 2>/dev/null)
export HELIUS_API_KEY=$(gcloud secrets versions access latest --secret="HELIUS_API_KEY" --project=shvyr-ai-bots 2>/dev/null)

# Set dummy values for unused but required keys
export TELEGRAM_TOKEN="dummy_telegram_token"
export ETHERSCAN_API_KEY="dummy_etherscan_key"
export BIRDEYE_API_KEY="dummy_birdeye_key"

echo -e "${GREEN}✅ Environment configured${NC}"
echo ""

# Run the collection script
echo -e "${GREEN}🚀 Starting corpus collection (mode: $MODE)${NC}"
echo "============================================================"
echo ""

# Determine Python command arguments based on mode
case "$MODE" in
    test)
        PYTHON_ARGS="--test"
        ;;
    production)
        PYTHON_ARGS="--production"
        echo -e "${YELLOW}⚠️  This will collect 180 days of data for 10 tokens${NC}"
        echo -e "${YELLOW}   Estimated time: 30-60 minutes${NC}"
        echo ""
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
        ;;
    custom)
        PYTHON_ARGS="$@"
        ;;
    *)
        echo -e "${RED}❌ Invalid mode: $MODE${NC}"
        echo "Usage: $0 [test|production|custom] [additional args]"
        exit 1
        ;;
esac

# Run with uv
uv run python scripts/data_collection/run_initial_corpus_collection.py $PYTHON_ARGS

EXIT_CODE=$?

# Cleanup
if [ ! -z "$PROXY_PID" ]; then
    echo ""
    echo -e "${YELLOW}🔧 Stopping Cloud SQL Proxy...${NC}"
    kill $PROXY_PID 2>/dev/null || true
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Corpus collection completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ Corpus collection failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE