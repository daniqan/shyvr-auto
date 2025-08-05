#!/usr/bin/env python3

"""
Transformer Monitoring Setup Script

Deployment script for setting up comprehensive transformer monitoring 
dashboards and alerting rules in GCP Cloud Monitoring.

This script follows TDD methodology and integrates with existing monitoring infrastructure.

Key Features:
1. Creates transformer-specific custom metrics
2. Deploys monitoring dashboards to GCP Cloud Monitoring  
3. Sets up alerting rules for transformer models
4. Integrates with existing production monitoring
5. Updates monitoring scripts to include transformer metrics

Usage:
    uv run python deploy/monitoring/setup_transformer_monitoring.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.gcp_transformer_monitoring import (
    GCPTransformerMonitoring,
    TransformerCloudMonitoringClient,
    TransformerAlertPolicyManager,
    TransformerDashboardManager,
    TransformerCustomMetrics
)

logger = logging.getLogger(__name__)


class TransformerMonitoringSetup:
    """
    Comprehensive transformer monitoring setup for production deployment.
    
    Handles the complete setup process including custom metrics creation,
    dashboard deployment, alert policy configuration, and integration
    with existing monitoring infrastructure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize transformer monitoring setup.
        
        Args:
            config: Configuration dictionary containing project details
        """
        self.project_id = config['project_id']
        self.region = config['region']
        self.service_name = config['service_name']
        self.monitoring_config = config.get('monitoring_config', {})
        self.transformer_models = config.get('transformer_models', {})
        
        # Initialize components
        self.gcp_monitoring = GCPTransformerMonitoring(
            project_id=self.project_id,
            region=self.region,
            service_name=self.service_name
        )
        
        self.dashboard_manager = TransformerDashboardManager(self.project_id)
        self.alert_manager = TransformerAlertPolicyManager(self.project_id)
        self.metrics_manager = TransformerCustomMetrics(self.project_id)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def setup_custom_metrics(self) -> Dict[str, Any]:
        """Set up transformer-specific custom metrics."""
        try:
            self.logger.info("Creating transformer custom metrics...")
            
            # Create custom metrics for transformer monitoring
            metrics_result = await self.gcp_monitoring.create_transformer_custom_metrics(
                self.transformer_models
            )
            
            self.logger.info(f"Successfully created {len(metrics_result)} custom metrics")
            
            return {
                'status': 'success',
                'created': len(metrics_result),
                'metrics': [metric['type'] for metric in metrics_result]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create custom metrics: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def setup_dashboards(self) -> Dict[str, Any]:
        """Set up transformer monitoring dashboards."""
        try:
            self.logger.info("Creating transformer monitoring dashboards...")
            
            # Create dashboard configuration
            dashboard_config = await self.gcp_monitoring.create_transformer_dashboard(
                self.transformer_models
            )
            
            # Deploy dashboard to GCP
            deployment_result = await self.gcp_monitoring.deploy_dashboard(dashboard_config)
            
            if deployment_result['status'] == 'success':
                self.logger.info(f"Successfully deployed dashboard: {deployment_result['dashboard_id']}")
                
                return {
                    'status': 'success',
                    'created': 1,
                    'dashboards': [{
                        'name': 'Transformer Model Monitoring Dashboard',
                        'dashboard_id': deployment_result['dashboard_id'],
                        'dashboard_url': deployment_result['dashboard_url'],
                        'panels_count': 8  # Health, Memory, Latency, Cache, Loading, Attention, Gradient, Summary
                    }]
                }
            else:
                return deployment_result
                
        except Exception as e:
            self.logger.error(f"Failed to create dashboards: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def setup_alert_policies(self) -> Dict[str, Any]:
        """Set up transformer alerting policies."""
        try:
            self.logger.info("Creating transformer alert policies...")
            
            # Configure alert settings
            alert_config = {
                'notification_channels': [
                    f'projects/{self.project_id}/notificationChannels/email-alerts',
                    f'projects/{self.project_id}/notificationChannels/slack-alerts'
                ],
                'thresholds': {
                    'memory_usage_mb': 7372.8,  # 90% of 8Gi
                    'inference_latency_ms': 1000.0,
                    'cache_hit_rate': 0.5,  # 50%
                    'loading_failures': 3
                }
            }
            
            # Deploy alert policies
            alerts_result = await self.gcp_monitoring.deploy_all_alert_policies(alert_config)
            
            if alerts_result['status'] == 'success':
                self.logger.info(f"Successfully deployed {alerts_result['deployed_policies']} alert policies")
                
                return {
                    'status': 'success',
                    'created': alerts_result['deployed_policies'],
                    'alert_policies': [
                        {'name': name, 'status': 'deployed'} 
                        for name in alerts_result['policy_names']
                    ]
                }
            else:
                return alerts_result
                
        except Exception as e:
            self.logger.error(f"Failed to create alert policies: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def integrate_with_existing_monitoring(self) -> Dict[str, Any]:
        """Integrate with existing production monitoring."""
        try:
            self.logger.info("Integrating with existing production monitoring...")
            
            # Import and integrate with production monitoring
            from src.monitoring.production_monitoring_dashboard import ProductionMonitoringDashboard
            
            production_monitoring = ProductionMonitoringDashboard(
                project_id=self.project_id,
                region=self.region
            )
            
            # Perform integration
            integration_result = await self.gcp_monitoring.integrate_with_production_monitoring(
                production_monitoring
            )
            
            self.logger.info("Successfully integrated with production monitoring")
            return integration_result
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with existing monitoring: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def update_monitoring_scripts(self) -> Dict[str, Any]:
        """Update existing monitoring scripts with transformer metrics."""
        try:
            self.logger.info("Updating existing monitoring scripts...")
            
            # Update monitoring scripts
            script_updates = await self.gcp_monitoring.update_existing_monitoring_scripts()
            
            self.logger.info(f"Updated {len(script_updates)} monitoring scripts")
            return {
                'status': 'success',
                'updated_scripts': script_updates
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update monitoring scripts: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


async def setup_transformer_monitoring(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to set up complete transformer monitoring.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing setup results
    """
    setup = TransformerMonitoringSetup(config)
    
    results = {
        'timestamp': logger.info("Starting transformer monitoring setup..."),
        'custom_metrics': {},
        'dashboards': {},
        'alert_policies': {},
        'integration': {},
        'script_updates': {},
        'status': 'success'
    }
    
    try:
        # Step 1: Create custom metrics
        logger.info("Step 1: Creating custom metrics...")
        results['custom_metrics'] = await setup.setup_custom_metrics()
        
        if results['custom_metrics']['status'] != 'success':
            results['status'] = 'partial_failure'
        
        # Step 2: Set up dashboards
        logger.info("Step 2: Setting up dashboards...")
        results['dashboards'] = await setup.setup_dashboards()
        
        if results['dashboards']['status'] != 'success':
            results['status'] = 'partial_failure'
        
        # Step 3: Set up alert policies
        logger.info("Step 3: Setting up alert policies...")
        results['alert_policies'] = await setup.setup_alert_policies()
        
        if results['alert_policies']['status'] != 'success':
            results['status'] = 'partial_failure'
        
        # Step 4: Integrate with existing monitoring
        logger.info("Step 4: Integrating with existing monitoring...")
        results['integration'] = await setup.integrate_with_existing_monitoring()
        
        if results['integration']['status'] != 'success':
            results['status'] = 'partial_failure'
        
        # Step 5: Update monitoring scripts
        logger.info("Step 5: Updating monitoring scripts...")
        results['script_updates'] = await setup.update_monitoring_scripts()
        
        if results['script_updates']['status'] != 'success':
            results['status'] = 'partial_failure'
        
        logger.info("Transformer monitoring setup completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Transformer monitoring setup failed: {e}")
        results['status'] = 'failed'
        results['error'] = str(e)
        return results


async def create_transformer_dashboards(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create transformer monitoring dashboards.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dashboard creation results
    """
    setup = TransformerMonitoringSetup(config)
    return await setup.setup_dashboards()


async def deploy_transformer_alerts(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deploy transformer alert policies.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Alert deployment results
    """
    setup = TransformerMonitoringSetup(config)
    return await setup.setup_alert_policies()


async def main():
    """Main function for script execution."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'project_id': 'shvyr-ai-bots',
        'region': 'us-central1',
        'service_name': 'shyvr-rlte',
        'monitoring_config': {
            'dashboards_enabled': True,
            'alerts_enabled': True,
            'custom_metrics_enabled': True
        },
        'transformer_models': {
            'itransformer': {
                'model_type': 'ITRANSFORMER',
                'enabled': True,
                'memory_limit_mb': 2048,
                'latency_threshold_ms': 100
            },
            'patchtst': {
                'model_type': 'PATCHTST',
                'enabled': True,
                'memory_limit_mb': 1536,
                'latency_threshold_ms': 150
            },
            'timesmixer': {
                'model_type': 'TIMESMIXER',
                'enabled': True,
                'memory_limit_mb': 2560,
                'latency_threshold_ms': 120
            },
            'timesfm': {
                'model_type': 'TIMESFM',
                'enabled': True,
                'memory_limit_mb': 7680,  # 7.5Gi for large model
                'latency_threshold_ms': 500
            }
        }
    }
    
    try:
        # Run setup
        results = await setup_transformer_monitoring(config)
        
        # Print results
        if results['status'] == 'success':
            print("\n✅ Transformer monitoring setup completed successfully!")
            print(f"📊 Custom metrics created: {results['custom_metrics'].get('created', 0)}")
            print(f"📈 Dashboards deployed: {results['dashboards'].get('created', 0)}")
            print(f"🚨 Alert policies created: {results['alert_policies'].get('created', 0)}")
            print(f"🔗 Integration status: {results['integration'].get('status', 'unknown')}")
            print(f"📝 Scripts updated: {len(results['script_updates'].get('updated_scripts', {}))}")
        else:
            print(f"\n❌ Transformer monitoring setup failed or incomplete: {results['status']}")
            if 'error' in results:
                print(f"Error: {results['error']}")
        
        # Save results to file
        results_file = Path(__file__).parent / 'transformer_monitoring_setup_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📋 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n💥 Setup failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())