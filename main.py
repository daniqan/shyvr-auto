"""
Shyvr AI Reinforcement Learning Trading Engine (RLTE)
Main application entry point
"""

import asyncio
import logging
import sys
from pathlib import Path

import structlog
from fastapi import FastAPI
import uvicorn

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import init_config, get_config
from src.utils.base import ConfigurationError

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# FastAPI application
app = FastAPI(
    title="Shyvr RLTE",
    description="AI-augmented cryptocurrency trading bot with reinforcement learning",
    version="0.1.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        # Initialize configuration
        config = init_config()
        logger.info("Configuration loaded", environment=config.app.environment)
        
        # Set up logging level
        logging.getLogger().setLevel(getattr(logging, config.app.log_level))
        
        logger.info("Shyvr RLTE starting up", version=config.app.version)
        
    except ConfigurationError as e:
        logger.error("Configuration error", error=str(e))
        raise
    except Exception as e:
        logger.error("Startup error", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shyvr RLTE shutting down")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Shyvr RLTE API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        config = get_config()
        return {
            "status": "healthy",
            "version": config.app.version,
            "environment": config.app.environment,
            "components": {
                "database": "not_implemented",
                "redis": "not_implemented", 
                "ml_models": "not_implemented",
                "rl_agent": "not_implemented"
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/config")
async def get_configuration():
    """Get current configuration (excluding secrets)"""
    try:
        config = get_config()
        
        # Return safe config info (no secrets)
        return {
            "app": {
                "name": config.app.name,
                "version": config.app.version,
                "environment": config.app.environment,
                "debug": config.app.debug
            },
            "trading": {
                "modes": config.trading.modes,
                "risk_management": {
                    "max_position_size_pct": config.trading.risk_management.max_position_size_pct,
                    "max_daily_loss_pct": config.trading.risk_management.max_daily_loss_pct,
                    "max_drawdown_pct": config.trading.risk_management.max_drawdown_pct
                }
            },
            "agent": {
                "model_type": config.agent.model_type,
                "max_active_rules": config.agent.max_active_rules
            }
        }
    except Exception as e:
        logger.error("Failed to get configuration", error=str(e))
        return {"error": str(e)}


def main():
    """Main application entry point"""
    try:
        # Initialize basic logging
        logging.basicConfig(level=logging.INFO)
        
        # Load configuration to get settings
        config = init_config()
        
        # Run the FastAPI app
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8080,
            log_level=config.app.log_level.lower(),
            reload=config.app.debug,
            workers=1  # Single worker for now
        )
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
