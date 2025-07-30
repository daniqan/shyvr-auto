# Docker Configuration

This directory contains Docker-related configuration files for the Shyvr RLTE system.

## Files

- `Dockerfile.optimized` - Production-optimized Docker build configuration
- `docker-compose.production.yml` - Production Docker Compose stack with PostgreSQL, Redis, and monitoring

## Usage

```bash
# Build optimized container
docker build -f docker/Dockerfile.optimized -t shyvr-rlte:optimized .

# Run production stack
docker-compose -f docker/docker-compose.production.yml up -d
```

For complete deployment instructions, see [../docs/deployment/](../docs/deployment/).