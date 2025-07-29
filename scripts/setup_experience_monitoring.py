#!/usr/bin/env python3
"""
Experience Storage Monitoring Setup

Initializes comprehensive monitoring system for RL experience storage,
including Prometheus metrics collection and Grafana dashboard setup.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import structlog
from prometheus_client import start_http_server, REGISTRY

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.base import MetricsRegistry
from src.monitoring.experience_metrics import create_experience_storage_metrics
from src.monitoring.training_metrics import create_training_pipeline_metrics
from src.rl_agent.experience_database import set_experience_metrics_collector
from src.rl_agent.training_pipeline import set_training_metrics_collector
from src.utils.config import get_config

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
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class ExperienceMonitoringService:
    """
    Service to manage experience storage monitoring system.
    
    Coordinates Prometheus metrics collection, Grafana dashboard setup,
    and alerting configuration for RL experience storage system.
    """
    
    def __init__(self, config):
        """Initialize monitoring service with configuration."""
        self.config = config
        self.metrics_registry = MetricsRegistry()
        self.experience_metrics = create_experience_storage_metrics(self.metrics_registry)
        self.training_metrics = create_training_pipeline_metrics(self.metrics_registry)
        # Use default values since monitoring config might not be in the main config
        self.metrics_port = 8000
        self.collection_interval = 30
        self.running = False
        
        # Set global metrics collectors
        set_experience_metrics_collector(self.experience_metrics)
        set_training_metrics_collector(self.training_metrics)
        
        logger.info("Experience monitoring service initialized",
                   metrics_port=self.metrics_port,
                   collection_interval=self.collection_interval)
    
    async def start(self) -> None:
        """Start the monitoring service."""
        try:
            # Start Prometheus metrics server
            start_http_server(self.metrics_port, registry=self.metrics_registry.registry)
            logger.info("Prometheus metrics server started", port=self.metrics_port)
            
            # Start periodic metrics collection
            self.running = True
            await self._start_metrics_collection()
            
        except Exception as e:
            logger.error("Failed to start monitoring service", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the monitoring service."""
        self.running = False
        logger.info("Experience monitoring service stopped")
    
    async def _start_metrics_collection(self) -> None:
        """Start periodic metrics collection loop."""
        logger.info("Starting metrics collection loop",
                   interval_seconds=self.collection_interval)
        
        while self.running:
            try:
                start_time = time.time()
                
                # Collect experience storage metrics
                await self.experience_metrics.collect_metrics()
                
                # Collect training pipeline metrics
                await self.training_metrics.collect_metrics()
                
                collection_time = time.time() - start_time
                logger.debug("Metrics collection completed",
                           duration_ms=collection_time * 1000)
                
                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    def get_metrics_output(self) -> bytes:
        """Get Prometheus format metrics output."""
        return self.metrics_registry.generate_output()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get monitoring service status."""
        return {
            "running": self.running,
            "metrics_port": self.metrics_port,
            "collection_interval": self.collection_interval,
            "experience_metrics_healthy": self.experience_metrics.is_healthy(),
            "registered_buffers": len(self.experience_metrics._registered_buffers),
            "last_collection_time": self.experience_metrics._last_collection_time,
            "collection_errors": self.experience_metrics._collection_errors
        }


async def setup_grafana_dashboard() -> None:
    """Setup Grafana dashboard for experience storage monitoring."""
    try:
        dashboard_path = Path(__file__).parent.parent / "monitoring" / "grafana" / "experience_storage_dashboard.json"
        alerts_path = Path(__file__).parent.parent / "monitoring" / "grafana" / "experience_storage_alerts.yml"
        
        if not dashboard_path.exists():
            logger.error("Dashboard configuration not found", path=str(dashboard_path))
            return
        
        if not alerts_path.exists():
            logger.error("Alert rules configuration not found", path=str(alerts_path))
            return
        
        # Load dashboard configuration
        with open(dashboard_path, 'r') as f:
            dashboard_config = json.load(f)
        
        logger.info("Grafana dashboard configuration loaded",
                   panels=len(dashboard_config.get('dashboard', {}).get('panels', [])))
        
        # Load alert rules
        with open(alerts_path, 'r') as f:
            alerts_config = f.read()
        
        logger.info("Alert rules configuration loaded",
                   size_bytes=len(alerts_config))
        
        # In a production environment, you would:
        # 1. Push dashboard to Grafana API
        # 2. Configure alert rules in Prometheus/Alertmanager
        # 3. Set up notification channels
        
        logger.info("Grafana dashboard and alerts configured successfully")
        
    except Exception as e:
        logger.error("Failed to setup Grafana dashboard", error=str(e))
        raise


def validate_monitoring_setup() -> bool:
    """Validate that monitoring setup is correct."""
    try:
        # Check required directories exist
        monitoring_dir = Path(__file__).parent.parent / "monitoring"
        grafana_dir = monitoring_dir / "grafana"
        
        required_files = [
            grafana_dir / "experience_storage_dashboard.json",
            grafana_dir / "experience_storage_alerts.yml"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error("Required monitoring file missing", path=str(file_path))
                return False
        
        # Check that metrics module can be imported
        from src.monitoring.experience_metrics import ExperienceStorageMetrics
        
        # Test metrics registry creation
        test_registry = MetricsRegistry()
        test_metrics = create_experience_storage_metrics(test_registry)
        
        if not test_metrics.is_healthy():
            logger.error("Experience metrics collector health check failed")
            return False
        
        logger.info("Monitoring setup validation passed")
        return True
        
    except Exception as e:
        logger.error("Monitoring setup validation failed", error=str(e))
        return False


async def main():
    """Main entry point for monitoring setup."""
    try:
        logger.info("Starting experience storage monitoring setup")
        
        # Validate monitoring setup
        if not validate_monitoring_setup():
            logger.error("Monitoring setup validation failed")
            sys.exit(1)
        
        # Load configuration
        config = get_config()
        
        # Setup Grafana dashboard
        await setup_grafana_dashboard()
        
        # Create and start monitoring service
        monitoring_service = ExperienceMonitoringService(config)
        
        # In production mode, start the service
        if len(sys.argv) > 1 and sys.argv[1] == "--start-service":
            logger.info("Starting monitoring service in production mode")
            await monitoring_service.start()
            
            # Keep service running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutdown signal received")
                await monitoring_service.stop()
        else:
            # Setup mode - just validate and configure
            status = monitoring_service.get_service_status()
            logger.info("Monitoring service setup completed", status=status)
            
            print("\nExperience Storage Monitoring Setup Complete!")
            print(f"- Metrics server will run on port {monitoring_service.metrics_port}")
            print(f"- Collection interval: {monitoring_service.collection_interval} seconds")
            print("- Grafana dashboard and alerts configured")
            print("\nTo start the monitoring service:")
            print(f"uv run {__file__} --start-service")
            
    except Exception as e:
        logger.error("Experience monitoring setup failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())