#!/bin/bash
# =============================================================================
# Container Build Script for Shyvr RLTE
# =============================================================================
set -euo pipefail

# Configuration
IMAGE_NAME="shyvr-rlte"
REGISTRY="${REGISTRY:-}"
VERSION="${VERSION:-latest}"
BUILD_TYPE="${BUILD_TYPE:-optimized}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Help function
show_help() {
    cat << EOF
Container Build Script for Shyvr RLTE

Usage: $0 [OPTIONS] COMMAND

Commands:
  build       Build the Docker image
  push        Push the image to registry
  run         Run the container locally
  test        Test the built image
  clean       Clean up build artifacts
  all         Build, test, and optionally push

Options:
  -t, --tag TAG           Set image tag (default: latest)
  -r, --registry REG      Set container registry
  -f, --dockerfile FILE   Dockerfile to use (default: Dockerfile.optimized)
  --no-cache             Build without cache
  --push                 Push after successful build
  --test                 Run tests after build
  -h, --help             Show this help

Examples:
  $0 build                           # Build with default settings
  $0 build -t v1.0.0                # Build with specific tag
  $0 build --no-cache --test         # Build without cache and test
  $0 all -t v1.2.0 --push           # Build, test, and push
  
Environment Variables:
  REGISTRY        Container registry URL
  VERSION         Default image version/tag
  BUILD_TYPE      Build type (optimized, debug)
  DOCKER_BUILDKIT Enable BuildKit (recommended)

EOF
}

# Parse command line arguments
DOCKERFILE="Dockerfile.optimized"
NO_CACHE=""
PUSH=false
TEST=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -f|--dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --test)
            TEST=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        build|push|run|test|clean|all)
            COMMAND="$1"
            shift
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate command
if [[ -z "$COMMAND" ]]; then
    error "No command specified. Use -h for help."
fi

# Set full image name
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${VERSION}"
fi

# Pre-build checks
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    # Check Dockerfile exists
    if [[ ! -f "$DOCKERFILE" ]]; then
        error "Dockerfile not found: $DOCKERFILE"
    fi
    
    # Check pyproject.toml
    if [[ ! -f "pyproject.toml" ]]; then
        error "pyproject.toml not found"
    fi
    
    # Check uv.lock
    if [[ ! -f "uv.lock" ]]; then
        warn "uv.lock not found - dependencies may not be locked"
    fi
    
    info "Prerequisites check passed"
}

# Build function
build_image() {
    log "Building Docker image: $FULL_IMAGE_NAME"
    log "Using Dockerfile: $DOCKERFILE"
    
    # Enable BuildKit for better performance
    export DOCKER_BUILDKIT=1
    
    # Build arguments
    BUILD_ARGS=(
        "--file" "$DOCKERFILE"
        "--tag" "$FULL_IMAGE_NAME"
        "--label" "org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        "--label" "org.opencontainers.image.version=${VERSION}"
        "--label" "org.opencontainers.image.revision=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    )
    
    if [[ -n "$NO_CACHE" ]]; then
        BUILD_ARGS+=("$NO_CACHE")
    fi
    
    # Build the image
    if docker build "${BUILD_ARGS[@]}" .; then
        log "Build completed successfully"
        
        # Show image info
        docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    else
        error "Build failed"
    fi
}

# Test function
test_image() {
    log "Testing Docker image: $FULL_IMAGE_NAME"
    
    # Test 1: Image exists
    if ! docker image inspect "$FULL_IMAGE_NAME" &> /dev/null; then
        error "Image not found: $FULL_IMAGE_NAME"
    fi
    
    # Test 2: Container can start
    log "Testing container startup..."
    CONTAINER_ID=$(docker run -d --rm \
        -e ENVIRONMENT=test \
        -e DATABASE_URL=postgresql://test:test@localhost:5432/test \
        "$FULL_IMAGE_NAME" \
        python -c "import time; time.sleep(30)")
    
    # Wait for startup
    sleep 10
    
    # Test 3: Health check
    if docker exec "$CONTAINER_ID" python -c "
import sys
try:
    import src.utils.config
    import src.dashboard.api
    import torch
    import fastapi
    import asyncpg
    print('All imports successful')
    sys.exit(0)
except Exception as e:
    print(f'Import failed: {e}')
    sys.exit(1)
"; then
        log "Import tests passed"
    else
        docker stop "$CONTAINER_ID" 2>/dev/null || true
        error "Import tests failed"
    fi
    
    # Test 4: Security check (non-root user)
    USER_CHECK=$(docker exec "$CONTAINER_ID" whoami)
    if [[ "$USER_CHECK" == "rlte" ]]; then
        log "Security check passed: running as non-root user"
    else
        docker stop "$CONTAINER_ID" 2>/dev/null || true
        error "Security check failed: not running as expected user (got: $USER_CHECK)"
    fi
    
    # Cleanup
    docker stop "$CONTAINER_ID" 2>/dev/null || true
    
    log "All tests passed"
}

# Push function  
push_image() {
    if [[ -z "$REGISTRY" ]]; then
        error "Registry not specified. Set REGISTRY environment variable or use -r option."
    fi
    
    log "Pushing image to registry: $FULL_IMAGE_NAME"
    
    if docker push "$FULL_IMAGE_NAME"; then
        log "Push completed successfully"
    else
        error "Push failed"
    fi
}

# Run function
run_image() {
    log "Running container locally: $FULL_IMAGE_NAME"
    
    # Check for .env file
    ENV_FILE=""
    if [[ -f ".env.production" ]]; then
        ENV_FILE="--env-file .env.production"
        info "Using .env.production"
    elif [[ -f ".env" ]]; then
        ENV_FILE="--env-file .env"
        info "Using .env"
    else
        warn "No environment file found"
    fi
    
    # Run with proper configuration
    docker run -it --rm \
        --name "shyvr-rlte-local" \
        -p 8080:8080 \
        $ENV_FILE \
        -e ENVIRONMENT=local \
        "$FULL_IMAGE_NAME"
}

# Clean function
clean_artifacts() {
    log "Cleaning up build artifacts..."
    
    # Remove dangling images
    if docker images -f "dangling=true" -q | grep -q .; then
        docker rmi $(docker images -f "dangling=true" -q) 2>/dev/null || true
        log "Removed dangling images"
    fi
    
    # Clean build cache
    docker builder prune -f || true
    
    log "Cleanup completed"
}

# Main execution
main() {
    log "Starting container build process..."
    log "Command: $COMMAND"
    log "Image: $FULL_IMAGE_NAME"
    log "Dockerfile: $DOCKERFILE"
    
    case "$COMMAND" in
        build)
            check_prerequisites
            build_image
            if [[ "$TEST" == true ]]; then
                test_image
            fi
            if [[ "$PUSH" == true ]]; then
                push_image
            fi
            ;;
        test)
            test_image
            ;;
        push)
            push_image
            ;;
        run)
            run_image
            ;;
        clean)
            clean_artifacts
            ;;
        all)
            check_prerequisites
            build_image
            test_image
            if [[ "$PUSH" == true ]]; then
                push_image
            fi
            ;;
        *)
            error "Unknown command: $COMMAND"
            ;;
    esac
    
    log "Process completed successfully"
}

# Execute main function
main