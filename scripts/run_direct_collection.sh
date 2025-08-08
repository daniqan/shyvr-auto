#!/bin/bash

# Direct Initial Corpus Collection Runner
# 
# This script sets up the environment and runs the direct collection script
# without the overhead of container builds. Designed for production execution.

set -euo pipefail

# Configuration
PROJECT_ID="shvyr-ai-bots"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    log_success "Python 3 found"
    
    # Check if we're in a virtual environment (recommended)
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_warning "Not running in a virtual environment. Consider using 'source venv/bin/activate'"
    else
        log_success "Running in virtual environment: $VIRTUAL_ENV"
    fi
    
    # Check if we're authenticated with GCP
    if command -v gcloud &> /dev/null; then
        if gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 &> /dev/null; then
            ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1)
            log_success "Authenticated with GCP as: $ACTIVE_ACCOUNT"
        else
            log_warning "Not authenticated with GCP. Run 'gcloud auth login' for optimal secret loading"
        fi
    else
        log_warning "gcloud CLI not found. Script will use environment variables for secrets"
    fi
    
    # Check for key environment variables if no GCP
    if [[ -z "${DATABASE_URL:-}" ]] && ! command -v gcloud &> /dev/null; then
        log_error "DATABASE_URL environment variable is required when gcloud is not available"
        log_info "Either install gcloud CLI and authenticate, or set DATABASE_URL manually"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Install only the packages we need for corpus collection
    pip install --quiet \
        asyncpg \
        pandas \
        numpy \
        structlog \
        pydantic \
        aiohttp \
        google-cloud-secret-manager \
        google-cloud-storage \
        pytz \
        python-dateutil
    
    log_success "Dependencies installed"
}

# Function to set environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Set project ID
    export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
    
    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
    
    # Production environment
    export ENVIRONMENT="production"
    
    log_success "Environment configured"
}

# Function to run the collection
run_collection() {
    log_info "Starting initial corpus collection..."
    
    cd "$PROJECT_ROOT"
    
    # Default arguments
    BATCH_SIZE=3
    RESUME_FROM=0
    EXTRA_ARGS=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --resume-from)
                RESUME_FROM="$2"
                shift 2
                ;;
            --dry-run)
                EXTRA_ARGS="$EXTRA_ARGS --dry-run"
                shift
                ;;
            --skip-export)
                EXTRA_ARGS="$EXTRA_ARGS --skip-export"
                shift
                ;;
            *)
                log_error "Unknown argument: $1"
                exit 1
                ;;
        esac
    done
    
    # Show configuration
    log_info "Collection configuration:"
    echo "  Project ID: $PROJECT_ID"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Resume from batch: $RESUME_FROM"
    echo "  Extra args: $EXTRA_ARGS"
    echo ""
    
    # Run the collection script
    python3 "$SCRIPT_DIR/direct_corpus_collection.py" \
        --project-id "$PROJECT_ID" \
        --batch-size "$BATCH_SIZE" \
        --resume-from "$RESUME_FROM" \
        $EXTRA_ARGS
}

# Function to show help
show_help() {
    cat << EOF
Direct Initial Corpus Collection Runner

This script runs the initial corpus collection directly without container overhead.

Usage: $0 [OPTIONS]

OPTIONS:
    --batch-size N      Process N tokens per batch (default: 3)
    --resume-from N     Resume from batch number N (default: 0)
    --dry-run           Check configuration without collecting data
    --skip-export       Skip GCS export after collection
    --help              Show this help message

EXAMPLES:
    # Run full collection
    $0
    
    # Run with smaller batches (safer, slower)
    $0 --batch-size 2
    
    # Resume from batch 3 (if previous run was interrupted)
    $0 --resume-from 3
    
    # Dry run to test configuration
    $0 --dry-run
    
    # Collect but skip GCS export
    $0 --skip-export

REQUIREMENTS:
    - Python 3.7+
    - GCP authentication (recommended) OR DATABASE_URL environment variable
    - Network access to CoinGecko, DeFiLlama, Alternative.me APIs
    - CloudSQL instance accessible

ENVIRONMENT VARIABLES:
    DATABASE_URL          - CloudSQL connection string (if not using GCP secrets)
    COINGECKO_API_KEY    - CoinGecko API key (if not using GCP secrets)
    LUNARCRUSH_API_KEY   - LunarCrush API key (optional)
    HELIUS_API_KEY       - Helius API key (optional)

EOF
}

# Main execution
main() {
    if [[ "${1:-}" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    log_info "Starting direct initial corpus collection"
    log_info "Project: $PROJECT_ID"
    log_info "Time: $(date)"
    echo ""
    
    # Run checks and setup
    check_prerequisites
    echo ""
    
    setup_environment
    echo ""
    
    # Ask for confirmation unless it's a dry run
    if [[ "$*" != *"--dry-run"* ]]; then
        echo "This will collect 6 months of historical data for 10 crypto tokens."
        echo "The process may take 30-60 minutes and will make real API calls."
        echo ""
        read -p "Continue? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Collection cancelled by user"
            exit 0
        fi
        echo ""
    fi
    
    # Install dependencies
    install_dependencies
    echo ""
    
    # Run collection
    run_collection "$@"
    
    log_success "Script completed"
}

# Execute main function with all arguments
main "$@"