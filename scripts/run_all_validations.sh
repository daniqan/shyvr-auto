#!/bin/bash

# Shyvr RLTE - Run All Validation Scripts
# Comprehensive validation orchestration script

set -euo pipefail

# Configuration
PROJECT_ID="${1:-shvyr-ai-bots}"
BASE_URL="${2:-http://localhost:8080}"
SKIP_API_VALIDATION="${3:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Banner
echo "=========================================="
echo "  Shyvr RLTE Validation Orchestration"
echo "=========================================="
echo ""
log_info "Project ID: $PROJECT_ID"
log_info "Base URL: $BASE_URL"
log_info "Skip API Validation: $SKIP_API_VALIDATION"
echo ""

# Change to project root directory
cd "$ROOT_DIR"

# Ensure reports directory exists
mkdir -p reports

# Check if Python and required modules are available
log_info "🐍 Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we can import required modules
if ! python3 -c "import asyncio, aiohttp, asyncpg" &> /dev/null; then
    log_warning "Some Python dependencies may be missing. Validation scripts will report specific issues."
fi

log_success "Python environment ready"
echo ""

# Run the orchestration script
log_info "🚀 Starting validation orchestration..."
echo ""

# Prepare arguments
PYTHON_ARGS=(
    "--project-id" "$PROJECT_ID"
    "--base-url" "$BASE_URL"
)

if [ "$SKIP_API_VALIDATION" = "true" ]; then
    PYTHON_ARGS+=("--skip-api-validation")
fi

# Execute the validation orchestration
if python3 -m scripts.run_all_validations "${PYTHON_ARGS[@]}"; then
    echo ""
    log_success "🎉 All validations completed successfully!"
    log_success "✅ System is ready for deployment"
    echo ""
    log_info "📋 Next steps:"
    log_info "  1. Review validation reports in reports/ directory"
    log_info "  2. Complete manual deployment checklist items"
    log_info "  3. Proceed with deployment"
    log_info "  4. Run post-deployment smoke tests"
    echo ""
    exit 0
else
    echo ""
    log_error "💥 Validation orchestration failed!"
    log_error "❌ System is NOT ready for deployment"
    echo ""
    log_info "📋 Next steps:"
    log_info "  1. Review validation reports in reports/ directory"
    log_info "  2. Fix identified issues"
    log_info "  3. Re-run validations"
    echo ""
    exit 1
fi