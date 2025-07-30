# Multi-stage Docker build for Shyvr RLTE
# Optimized for ML/RL workloads with CPU-only PyTorch

# =============================================================================
# Build stage - Dependency installation and compilation
# =============================================================================
FROM python:3.12-slim as builder

# Set build arguments for better control
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building (optimized layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core build tools
    build-essential=12.9* \
    gcc=4:12.2.0-3* \
    g++=4:12.2.0-3* \
    # Development headers
    libpq-dev=15.8-* \
    libssl-dev=3.0.* \
    libffi-dev=3.4.* \
    # Network tools
    curl=7.88.* \
    wget=1.21.* \
    # Version control
    git=1:2.39.* \
    # Package config
    pkg-config=1.8.* \
    # TA-Lib build dependencies
    libtool=2.4.* \
    autoconf=2.71-* \
    automake=1:1.16.* \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install TA-Lib from source (required for trading analysis)
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/ta-lib*

# Install uv for faster dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies with optimized settings
# Using CPU-only PyTorch for container optimization
RUN uv sync \
    --frozen \
    --no-dev \
    --no-install-project \
    --extra production \
    --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --index-url https://download.pytorch.org/whl/cpu torch>=2.7.1+cpu \
    && find /app/.venv -name "*.pyc" -delete \
    && find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + \
    && find /app/.venv -name "*.so" -exec strip {} \;

# =============================================================================
# Production stage - Optimized runtime environment
# =============================================================================
FROM python:3.12-slim as production

# Set production environment variables (security and performance optimized)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    ENVIRONMENT=production \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Security hardening
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    # Performance optimization
    MALLOC_ARENA_MAX=2 \
    # FastAPI optimization
    FORWARDED_ALLOW_IPS="*" \
    # Database connection pooling
    ASYNCPG_CONNECTION_TIMEOUT=10 \
    ASYNCPG_COMMAND_TIMEOUT=30

# Install runtime system dependencies (minimal and pinned versions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PostgreSQL client libraries (required for asyncpg)
    libpq5=15.8-* \
    # SSL/TLS support
    libssl3=3.0.* \
    ca-certificates=* \
    # TA-Lib runtime library
    libta-lib0=0.4.* \
    # Process management
    dumb-init=1.2.* \
    # Security updates
    && apt-get upgrade -y \
    # Cleanup to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create dedicated user with proper security constraints
RUN groupadd --gid 1000 rlte \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash rlte \
    # Create application directories with proper permissions
    && mkdir -p /app/data /app/logs /app/models /app/cache /app/tmp \
    && chown -R rlte:rlte /app \
    # Set secure permissions
    && chmod 750 /app \
    && chmod 770 /app/data /app/logs /app/models /app/cache /app/tmp

# Copy Python environment from builder (optimized transfer)
COPY --from=builder --chown=rlte:rlte /app/.venv /app/.venv

# Copy TA-Lib from builder
COPY --from=builder /usr/local/lib/libta_lib.* /usr/local/lib/
COPY --from=builder /usr/local/include/ta-lib/ /usr/local/include/ta-lib/
RUN ldconfig

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory and switch to non-root user early
WORKDIR /app
USER rlte

# Copy application code with proper ownership and permissions
COPY --chown=rlte:rlte src/ src/
COPY --chown=rlte:rlte config/ config/
COPY --chown=rlte:rlte main.py ./
COPY --chown=rlte:rlte static/ static/

# Create .dockerignore-aware structure
# (Note: actual files filtered by .dockerignore)

# Security: Validate Python environment
RUN python -c "import sys; print(f'Python version: {sys.version}')" \
    && python -c "import torch; print(f'PyTorch version: {torch.__version__}')" \
    && python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')" \
    && python -c "import asyncpg; print('PostgreSQL client: OK')" \
    && python -c "import talib; print('TA-Lib: OK')"

# Advanced health check with timeout and comprehensive checks
HEALTHCHECK --interval=30s \
            --timeout=15s \
            --start-period=60s \
            --retries=3 \
    CMD python -c "\
import sys, requests, time; \
start = time.time(); \
try: \
    response = requests.get('http://localhost:8080/health', timeout=10); \
    assert response.status_code == 200; \
    health = response.json(); \
    assert health.get('status') in ['healthy', 'degraded']; \
    print(f'Health check passed in {time.time()-start:.2f}s'); \
    sys.exit(0); \
except Exception as e: \
    print(f'Health check failed: {e}'); \
    sys.exit(1)"

# Expose port with documentation
EXPOSE 8080/tcp

# Add labels for better container management
LABEL maintainer="Shyvr RLTE Team" \
      version="0.1.0" \
      description="AI-augmented cryptocurrency trading bot with reinforcement learning" \
      python.version="3.12" \
      pytorch.version="2.7.1+cpu" \
      fastapi.version="latest" \
      org.opencontainers.image.source="https://github.com/your-org/shyvrai-rlte" \
      org.opencontainers.image.documentation="https://github.com/your-org/shyvrai-rlte/docs" \
      security.non-root="true" \
      optimization.multi-stage="true"

# Use dumb-init for proper signal handling and process management
ENTRYPOINT ["dumb-init", "--"]

# Production-optimized command with proper worker configuration
# Single worker for ML/RL workloads to avoid model conflicts
CMD ["python", "-m", "uvicorn", \
     "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log", \
     "--use-colors", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]