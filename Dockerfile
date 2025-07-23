# Multi-stage Docker build for Shyvr RLTE
# Optimized for ML/RL workloads with CPU-only PyTorch

# Build stage
FROM python:3.12-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv (faster than pip)
RUN uv sync --frozen --no-dev --no-install-project

# Production stage
FROM python:3.12-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    ENVIRONMENT=production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash rlte && \
    mkdir -p /app && \
    chown -R rlte:rlte /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Required for asyncpg
    libpq5 \
    # Required for some crypto libraries
    libssl3 \
    # Required for TA-Lib
    libta-lib0 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory and user
WORKDIR /app
USER rlte

# Copy application code
COPY --chown=rlte:rlte src/ src/
COPY --chown=rlte:rlte config/ config/
COPY --chown=rlte:rlte main.py .

# Create directories for data and logs
RUN mkdir -p data logs models cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]