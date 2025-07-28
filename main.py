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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import init_config, get_config
from src.utils.base import ConfigurationError
from src.dashboard.api import dashboard_api
from src.dashboard.service import dashboard_service
from src.monitoring.base import MetricsRegistry
from src.monitoring.trading_metrics import TradingMetricsCollector
from src.monitoring.safety_metrics import SafetyMetricsCollector
from src.monitoring.analysis_metrics import AnalysisMetricsCollector

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

# Initialize monitoring system
metrics_registry = MetricsRegistry()
trading_metrics = TradingMetricsCollector(metrics_registry)
safety_metrics = SafetyMetricsCollector(metrics_registry)
analysis_metrics = AnalysisMetricsCollector(metrics_registry)

# Include dashboard routes
app.include_router(dashboard_api.router)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


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
        
        # Start dashboard service
        await dashboard_service.start()
        logger.info("Dashboard service started")
        
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
    
    # Stop dashboard service
    await dashboard_service.stop()
    logger.info("Dashboard service stopped")


@app.get("/")
async def root():
    """Root endpoint - serve dashboard"""
    return FileResponse("static/index.html")


@app.get("/api/auth/key")
async def get_api_key():
    """Get the admin API key for dashboard authentication"""
    try:
        from src.dashboard.auth import dashboard_auth
        
        # Check if we have an admin user and API key
        admin_user = None
        admin_api_key = None
        
        # Find admin user first
        for user_id, user in dashboard_auth._users.items():
            if user.username == "admin" and "admin" in user.permissions:
                admin_user = user
                break
        
        if not admin_user:
            logger.error("Admin user not found in authentication system")
            return {"error": "Admin user not found"}
        
        # Find corresponding API key
        for api_key, user_id in dashboard_auth._api_keys.items():
            if user_id == admin_user.user_id:
                admin_api_key = api_key
                break
        
        if not admin_api_key:
            logger.error("Admin API key not found")
            return {"error": "Admin API key not found"}
        
        logger.info("API key requested for dashboard authentication", user_id=admin_user.user_id)
        return {"api_key": admin_api_key}
        
    except Exception as e:
        logger.error("Failed to get API key", error=str(e))
        return {"error": str(e)}


@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {"message": "Shyvr RLTE API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint with comprehensive component status"""
    try:
        config = get_config()
        
        # Check database health
        from src.utils.database import check_database_health
        database_health = await check_database_health()
        
        # Determine overall status based on component health
        overall_status = "healthy"
        if database_health.get("status") == "unhealthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "version": config.app.version,
            "environment": config.app.environment,
            "timestamp": database_health.get("timestamp", "NOW()"),
            "components": {
                "database": {
                    "status": database_health.get("status", "unknown"),
                    "connectivity": database_health.get("connectivity", False),
                    "database_size": database_health.get("database_size", "unknown"),
                    "active_connections": database_health.get("active_connections", 0),
                    "tables_exist": database_health.get("tables_exist", False),
                    "details": database_health
                },
                "ml_models": {
                    "status": "not_implemented",
                    "details": "ML model health checks not yet implemented"
                },
                "rl_agent": {
                    "status": "not_implemented", 
                    "details": "RL agent health checks not yet implemented"
                }
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "NOW()",
            "components": {
                "database": {"status": "error", "error": str(e)},
                "ml_models": {"status": "unknown"},
                "rl_agent": {"status": "unknown"}
            }
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


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping by monitoring systems.
    """
    try:
        # Collect fresh metrics from all collectors
        trading_metrics.collect_metrics()
        safety_metrics.collect_metrics()
        analysis_metrics.collect_metrics()
        
        # Generate Prometheus format output
        metrics_output = metrics_registry.generate_output()
        
        # Return with proper content type for Prometheus
        from fastapi import Response
        return Response(
            content=metrics_output,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        # Return empty metrics in case of error to avoid breaking monitoring
        return Response(
            content=b"# Error generating metrics\n",
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )


def main():
    """Main application entry point"""
    try:
        # Initialize basic logging
        logging.basicConfig(level=logging.INFO)
        
        # Load configuration to get settings
        config = init_config()
        
        # Get host and port from config with fallbacks
        host = getattr(config.app, 'host', '0.0.0.0')
        port = getattr(config.app, 'port', 8080)
        
        print(f"Starting Shyvr RLTE on {host}:{port}")
        print(f"Environment: {config.app.environment}")
        print(f"Dashboard URL: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        
        # Run the FastAPI app
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            log_level=config.app.log_level.lower(),
            reload=config.app.debug,
            workers=1  # Single worker for now
        )
        
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Error: Port {port if 'port' in locals() else 8080} is already in use.")
            print("Solutions:")
            print("1. Kill the existing process using the port")
            print("2. Use a different port: PORT=8081 python main.py")
            print("3. Find the process: lsof -i :8080")
        else:
            print(f"Network error: {e}")
        sys.exit(1)
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
