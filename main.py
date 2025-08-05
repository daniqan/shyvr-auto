"""
Shyvr AI Reinforcement Learning Trading Engine (RLTE)
Main application entry point
"""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.dashboard.api import dashboard_api
from src.dashboard.service import dashboard_service
from src.monitoring.analysis_metrics import AnalysisMetricsCollector
from src.monitoring.base import MetricsRegistry
from src.monitoring.safety_metrics import SafetyMetricsCollector
from src.monitoring.trading_metrics import TradingMetricsCollector
from src.utils.base import ConfigurationError
from src.utils.config import get_config, init_config
from src.model_preservation import initialize_preservation_system, shutdown_preservation_system
from src.model_preservation.integration import setup_preservation_hooks

# Import enhanced logging system
from src.enhanced_logging.enhanced_logging import initialize_logging, LogLevel

# Configure enhanced structured logging for Phase 6.2
initialize_logging(
    log_level=LogLevel.INFO,
    enable_file_logging=True,
    enable_console_logging=True,
    enable_aggregation=True
)

logger = structlog.get_logger()


async def get_ml_model_health() -> dict[str, Any]:
    """Get ML model health status"""
    try:
        from src.ml_analysis.model_manager import ModelManager
        from src.utils.config import get_config

        config = get_config()
        model_manager = ModelManager(config.dict() if hasattr(config, 'dict') else None)
        health_status = await model_manager.health_check()

        return {
            "status": "healthy" if health_status.get("overall_healthy", False) else "unhealthy",
            "models_loaded": len(health_status.get("models", {})),
            "ensemble_available": health_status.get("ensemble_available", False),
            "cache_size": health_status.get("cache_size", 0),
            "details": health_status
        }
    except Exception as e:
        logger.error("ML model health check failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "details": "Failed to check ML model health"
        }


async def get_rl_agent_health() -> dict[str, Any]:
    """Get RL agent health status"""
    try:
        from src.rl_agent.base import AgentConfig, ModelType
        from src.rl_agent.dqn_agent import DQNTradingAgent

        agent_config = AgentConfig(
            model_type=ModelType.DQN,
            learning_rate=0.001,
            batch_size=32,
            replay_buffer_size=10000,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=1000,
            target_update_frequency=100,
            max_position_size=0.1,
            max_daily_loss=0.05
        )

        agent = DQNTradingAgent(agent_config)
        health_status = await agent.health_check()

        return {
            "status": "healthy" if health_status.get("meets_targets", False) else "training",
            "is_trained": health_status.get("is_trained", False),
            "training_episodes": health_status.get("training_episodes", 0),
            "meets_targets": health_status.get("meets_targets", False),
            "performance": health_status.get("performance_metrics", {}),
            "details": health_status
        }
    except Exception as e:
        logger.error("RL agent health check failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "details": "Failed to check RL agent health"
        }


async def get_transformer_health() -> dict[str, Any]:
    """Get transformer models health status"""
    try:
        from src.ml_analysis.base import ModelType
        import asyncio
        
        # Dictionary to store transformer health results
        transformer_models = {}
        total_memory_usage = 0.0
        models_loaded = 0
        models_healthy = 0
        
        # Define transformer types to check
        transformer_types = {
            'itransformer': {
                'module': 'src.ml_analysis.transformers.itransformer',
                'class': 'iTransformerPredictor',
                'model_type': ModelType.ITRANSFORMER,
                'config': {'d_model': 512, 'n_heads': 8, 'n_layers': 6}
            },
            'patchtst': {
                'module': 'src.ml_analysis.transformers.patchtst',
                'class': 'PatchTSTPredictor', 
                'model_type': ModelType.PATCHTST,
                'config': {'d_model': 512, 'n_heads': 8, 'patch_size': 16}
            },
            'timesmixer': {
                'module': 'src.ml_analysis.transformers.timesmixer',
                'class': 'TimesMixerPredictor',
                'model_type': ModelType.TIMESMIXER,
                'config': {'d_model': 512, 'seq_len': 336, 'pred_len': 96}
            },
            'timesfm': {
                'module': 'src.ml_analysis.transformers.timesfm_wrapper',
                'class': 'TimesFMWrapper',
                'model_type': ModelType.TIMESFM,
                'config': {'model_size': 'small', 'horizon_length': 128}
            }
        }
        
        # Check each transformer type
        for transformer_name, transformer_info in transformer_types.items():
            try:
                # Import the transformer class dynamically
                module = __import__(transformer_info['module'], fromlist=[transformer_info['class']])
                transformer_class = getattr(module, transformer_info['class'])
                
                # Create transformer instance
                transformer = transformer_class(
                    transformer_info['model_type'], 
                    transformer_info['config']
                )
                
                # Perform health check
                is_healthy = await transformer.health_check()
                
                # Get health metrics
                memory_usage = getattr(transformer, 'get_memory_usage', lambda: 0.0)()
                avg_latency = getattr(transformer, 'get_avg_inference_latency', lambda: 0.0)()
                cache_hit_rate = getattr(transformer, 'get_cache_hit_rate', lambda: 0.0)()
                cache_status = getattr(transformer, 'get_model_cache_status', lambda: {
                    'size': 0, 'max_size': 100, 'hit_rate': 0.0
                })()
                
                transformer_models[transformer_name] = {
                    'loaded': True,
                    'healthy': is_healthy,
                    'memory_usage_mb': memory_usage,
                    'avg_inference_latency_ms': avg_latency,
                    'cache_hit_rate': cache_hit_rate,
                    'cache_status': cache_status
                }
                
                total_memory_usage += memory_usage
                models_loaded += 1
                if is_healthy:
                    models_healthy += 1
                    
            except Exception as e:
                logger.warning(f"Failed to check {transformer_name} health", error=str(e))
                transformer_models[transformer_name] = {
                    'loaded': False,
                    'healthy': False,
                    'error': str(e),
                    'memory_usage_mb': 0.0,
                    'avg_inference_latency_ms': 0.0,
                    'cache_hit_rate': 0.0
                }
        
        # Determine overall status
        if models_loaded == 0:
            status = "error"
        elif models_healthy == models_loaded:
            status = "healthy"
        elif models_healthy > 0:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "models": transformer_models,
            "total_memory_usage_mb": total_memory_usage,
            "models_loaded": models_loaded,
            "models_healthy": models_healthy,
            "timestamp": "NOW()"
        }
        
    except Exception as e:
        logger.error("Transformer health check failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "models": {},
            "total_memory_usage_mb": 0.0,
            "models_loaded": 0,
            "models_healthy": 0,
            "timestamp": "NOW()"
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
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
        
        # Initialize model preservation system
        await initialize_preservation_system()
        logger.info("Model preservation system initialized")
        
        # Setup preservation hooks
        await setup_preservation_hooks(app)
        logger.info("Model preservation hooks configured")

    except ConfigurationError as e:
        logger.error("Configuration error", error=str(e))
        raise
    except Exception as e:
        logger.error("Startup error", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shyvr RLTE shutting down")
    
    # Shutdown model preservation system (performs emergency backup)
    await shutdown_preservation_system()
    logger.info("Model preservation system shut down")

    # Stop dashboard service
    await dashboard_service.stop()
    logger.info("Dashboard service stopped")
    
    # Shutdown enhanced logging system
    from src.enhanced_logging.enhanced_logging import shutdown_logging
    await shutdown_logging()
    logger.info("Enhanced logging system shut down")


# FastAPI application
app = FastAPI(
    title="Shyvr RLTE",
    description="AI-augmented cryptocurrency trading bot with reinforcement learning",
    version="0.1.0",
    lifespan=lifespan
)

# Initialize monitoring system
metrics_registry = MetricsRegistry()
trading_metrics = TradingMetricsCollector(metrics_registry)
safety_metrics = SafetyMetricsCollector(metrics_registry)
analysis_metrics = AnalysisMetricsCollector(metrics_registry)

# Include dashboard routes
app.include_router(dashboard_api.router)

# Include preservation API routes
from src.model_preservation.api import preservation_api
app.include_router(preservation_api.router)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")




@app.get("/")
async def root() -> FileResponse:
    """Root endpoint - serve dashboard"""
    return FileResponse("static/index.html")


@app.get("/api/auth/key")
async def get_api_key() -> dict[str, str]:
    """Get the admin API key for dashboard authentication"""
    try:
        from src.dashboard.auth import dashboard_auth

        # Check if we have an admin user and API key
        admin_user = None
        admin_api_key = None

        # Find admin user first
        for _user_id, user in dashboard_auth._users.items():
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
async def api_root() -> dict[str, str]:
    """API root endpoint"""
    return {"message": "Shyvr RLTE API", "status": "running"}


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint with comprehensive component status"""
    try:
        config = get_config()

        # Check database health
        from src.utils.database import check_database_health
        database_health = await check_database_health()

        # Get transformer health
        transformer_health = await get_transformer_health()
        
        # Determine overall status based on component health
        overall_status = "healthy"
        if database_health.get("status") == "unhealthy":
            overall_status = "degraded"
        if transformer_health.get("status") in ["error", "unhealthy"]:
            overall_status = "degraded"
        elif transformer_health.get("status") == "degraded" and overall_status == "healthy":
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
                "ml_models": await get_ml_model_health(),
                "rl_agent": await get_rl_agent_health(),
                "transformers": transformer_health
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
                "ml_models": {"status": "error", "error": "Health check failed"},
                "rl_agent": {"status": "error", "error": "Health check failed"},
                "transformers": {"status": "error", "error": "Health check failed"}
            }
        }


@app.get("/config")
async def get_configuration() -> dict[str, Any]:
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
async def get_metrics() -> Response:
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


def main() -> None:
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
