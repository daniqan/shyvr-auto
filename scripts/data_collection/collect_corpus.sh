#!/bin/bash
#
# Unified Corpus Collection Runner
# Supports both single-granularity and multi-granularity collection
#
# Usage:
#   ./scripts/data_collection/collect_corpus.sh [mode] [options]
#
# Modes:
#   clean         - Clean ALL existing corpus data
#   clean-multi   - Clean multi-granularity data only
#   test          - Test run (2 tokens, 7 days, single granularity)
#   test-multi    - Test multi-granularity (2 tokens, multiple timeframes)
#   production    - Production run (10 tokens, 365 days, single granularity)
#   multi         - Full multi-granularity production run (7 tokens, multiple timeframes)
#   multi-dry     - Dry run for multi-granularity collection
#   custom        - Custom configuration (pass additional arguments)
#
# Examples:
#   # Clean existing data
#   ./scripts/data_collection/collect_corpus.sh clean
#   
#   # Test single granularity
#   ./scripts/data_collection/collect_corpus.sh test
#
#   # Test multi-granularity
#   ./scripts/data_collection/collect_corpus.sh test-multi
#   
#   # Production multi-granularity
#   ./scripts/data_collection/collect_corpus.sh multi
#   
#   # Custom multi-granularity with specific tokens
#   ./scripts/data_collection/collect_corpus.sh custom --multi --tokens WETH USDC --timeframes daily hourly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${GREEN}🔧 Corpus Collection Setup${NC}"
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

# Set minimal environment variables
export GOOGLE_CLOUD_PROJECT="shvyr-ai-bots"
export ENVIRONMENT="development"  
export SECRET_KEY="corpus_collection_secret_key_min_32_characters_long"
export DB_PORT="5433"
export DB_HOST="localhost"
export DB_USER="rlte_prod_user"
export DB_NAME="shyvr_rlte_prod"

echo -e "${GREEN}✅ Environment configured${NC}"
echo ""

# Function to run single-granularity collection
run_single_granularity() {
    local PYTHON_ARGS="$@"
    echo -e "${BLUE}📊 Running single-granularity collection${NC}"
    uv run python scripts/data_collection/run_initial_corpus_collection.py $PYTHON_ARGS
    return $?
}

# Function to run multi-granularity collection
run_multi_granularity() {
    local PYTHON_ARGS="$@"
    echo -e "${BLUE}📊 Running multi-granularity collection${NC}"
    uv run python scripts/collect_multi_granularity_corpus.py $PYTHON_ARGS
    return $?
}

# Determine Python command arguments based on mode
echo -e "${GREEN}🚀 Starting corpus collection (mode: $MODE)${NC}"
echo "============================================================"
echo ""

case "$MODE" in
    clean)
        echo -e "${YELLOW}🧹 Cleaning ALL corpus data...${NC}"
        echo ""
        uv run python scripts/data_collection/clean_corpus_data.py --mode all
        EXIT_CODE=$?
        ;;
        
    clean-multi)
        echo -e "${YELLOW}🧹 Cleaning multi-granularity corpus data...${NC}"
        echo ""
        uv run python scripts/data_collection/clean_corpus_data.py --mode multi
        EXIT_CODE=$?
        ;;
        
    test)
        echo -e "${BLUE}📋 Test mode: 2 tokens, 7 days, single granularity${NC}"
        run_single_granularity --test
        EXIT_CODE=$?
        ;;
        
    test-multi)
        echo -e "${BLUE}📋 Test multi-granularity: 2 tokens, multiple timeframes${NC}"
        echo "   Tokens: WETH, USDC"
        echo "   Timeframes: daily (30 days), hourly (24 hours)"
        echo ""
        
        # Use the test config file from the config directory
        CONFIG_FILE="$PROJECT_ROOT/config/corpus_collection_test.yaml"
        
        # Run directly without confirmation in test mode
        echo -e "${BLUE}📊 Running multi-granularity collection${NC}"
        run_multi_granularity --config "$CONFIG_FILE"
        EXIT_CODE=$?
        ;;
        
    production)
        echo -e "${YELLOW}⚠️  Production mode: 10 tokens, 365 days, single granularity${NC}"
        echo -e "${YELLOW}   Estimated time: 60-90 minutes${NC}"
        echo ""
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
        run_single_granularity --production
        EXIT_CODE=$?
        ;;
        
    multi)
        echo -e "${YELLOW}⚠️  Multi-granularity production mode${NC}"
        echo "   Tokens: WETH, WBTC, USDC, LINK, UNI, SOL, PEPE"
        echo "   Timeframes:"
        echo "     • Daily: 365 days"
        echo "     • 4-hour: 180 days"
        echo "     • Hourly: 90 days"
        echo -e "${YELLOW}   Estimated time: 2-3 hours${NC}"
        echo -e "${YELLOW}   Expected data: ~30,000 candles${NC}"
        echo ""
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
        
        # Run with production config
        CONFIG_FILE="$PROJECT_ROOT/config/corpus_collection.yaml"
        run_multi_granularity --config "$CONFIG_FILE"
        EXIT_CODE=$?
        ;;
        
    multi-dry)
        echo -e "${BLUE}📋 Multi-granularity dry run${NC}"
        CONFIG_FILE="$PROJECT_ROOT/config/corpus_collection.yaml"
        run_multi_granularity --config "$CONFIG_FILE" --dry-run
        EXIT_CODE=$?
        ;;
        
    custom)
        # Check if --multi flag is present
        if [[ "$@" == *"--multi"* ]]; then
            # Remove --multi flag and run multi-granularity
            ARGS="${@//--multi/}"
            run_multi_granularity $ARGS
        else
            # Run single-granularity
            run_single_granularity "$@"
        fi
        EXIT_CODE=$?
        ;;
        
    *)
        echo -e "${RED}❌ Invalid mode: $MODE${NC}"
        echo ""
        echo "Available modes:"
        echo "  clean       - Clean ALL existing corpus data"
        echo "  clean-multi - Clean multi-granularity data only"
        echo "  test        - Test run (single granularity)"
        echo "  test-multi  - Test multi-granularity"
        echo "  production  - Production run (single granularity)"
        echo "  multi       - Multi-granularity production"
        echo "  multi-dry   - Multi-granularity dry run"
        echo "  custom      - Custom configuration"
        echo ""
        echo "Examples:"
        echo "  ./collect_corpus.sh clean"
        echo "  ./collect_corpus.sh clean-multi"
        echo "  ./collect_corpus.sh test-multi"
        echo "  ./collect_corpus.sh multi-dry"
        echo "  ./collect_corpus.sh multi"
        exit 1
        ;;
esac

# Cleanup
if [ ! -z "$PROXY_PID" ]; then
    echo ""
    echo -e "${YELLOW}🔧 Stopping Cloud SQL Proxy...${NC}"
    kill $PROXY_PID 2>/dev/null || true
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Corpus collection completed successfully!${NC}"
    
    # Show summary for multi-granularity runs
    if [[ "$MODE" == "multi"* ]] || [[ "$MODE" == "test-multi" ]]; then
        echo ""
        echo -e "${BLUE}📊 Collection Summary:${NC}"
        echo "   Check the logs above for detailed statistics"
        echo "   Data stored in database with appropriate granularity tags"
    fi
else
    echo ""
    echo -e "${RED}❌ Corpus collection failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE