#!/usr/bin/env python3
"""
Setup script for Google Cloud Monitoring integration.

This script automates the setup of Cloud Monitoring for the Shyvr RLTE trading system,
including custom metrics, alert policies, notification channels, and dashboards.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.cloud_monitoring import (
    CloudMonitoringClient,
    CloudMonitoringCollector,
    CloudMetricDescriptor,
    AlertPolicyCondition,
    NotificationChannel
)
from monitoring.base import MetricsRegistry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GCPMonitoringSetup:
    """
    Automated setup manager for GCP monitoring integration.
    
    This class handles the complete setup process including:
    - Creating custom metric descriptors
    - Setting up alert policies
    - Creating notification channels
    - Validating the setup
    """
    
    def __init__(self, project_id: str, config_path: Optional[str] = None):
        """
        Initialize the setup manager.
        
        Args:
            project_id: GCP project ID
            config_path: Path to monitoring configuration file
        """
        self.project_id = project_id
        self.config_path = config_path or "config/monitoring_config.json"
        self.client = CloudMonitoringClient(project_id)
        self.setup_results = {"metrics": [], "alerts": [], "channels": [], "errors": []}
        
        logger.info(f"Initialized GCP monitoring setup for project: {project_id}")
    
    def load_configuration(self) -> Dict:
        """Load monitoring configuration from file."""
        try:
            if not os.path.exists(self.config_path):
                logger.info(f"Config file not found at {self.config_path}, using defaults")
                return self._get_default_config()
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded monitoring configuration from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default monitoring configuration."""
        return {
            "metrics": [
                {
                    "name": "trading_pnl",
                    "display_name": "Trading P&L",
                    "description": "Total profit and loss from trading activities",
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "unit": "USD",
                    "labels": [
                        {"key": "currency", "description": "Currency pair"},
                        {"key": "timeframe", "description": "Time period"}
                    ]
                },
                {
                    "name": "trading_volume",
                    "display_name": "Trading Volume",
                    "description": "Volume of trades executed",
                    "metric_kind": "CUMULATIVE",
                    "value_type": "DOUBLE",
                    "unit": "USD",
                    "labels": [
                        {"key": "symbol", "description": "Trading symbol"},
                        {"key": "side", "description": "Buy or sell"}
                    ]
                },
                {
                    "name": "safety_risk_level",
                    "display_name": "Risk Level",
                    "description": "Overall system risk level (0-1)",
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "unit": "1",
                    "labels": [
                        {"key": "risk_type", "description": "Type of risk being measured"}
                    ]
                },
                {
                    "name": "safety_emergency_stops",
                    "display_name": "Emergency Stops",
                    "description": "Number of emergency stops triggered",
                    "metric_kind": "CUMULATIVE",
                    "value_type": "INT64",
                    "unit": "1",
                    "labels": [
                        {"key": "reason", "description": "Reason for emergency stop"}
                    ]
                },
                {
                    "name": "system_uptime",
                    "display_name": "System Uptime",
                    "description": "System uptime indicator (1 = up, 0 = down)",
                    "metric_kind": "GAUGE",
                    "value_type": "INT64",
                    "unit": "1",
                    "labels": []
                },
                {
                    "name": "ml_model_accuracy",
                    "display_name": "ML Model Accuracy",
                    "description": "Machine learning model prediction accuracy",
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "unit": "1",
                    "labels": [
                        {"key": "model_type", "description": "Type of ML model"},
                        {"key": "timeframe", "description": "Evaluation timeframe"}
                    ]
                },
                {
                    "name": "rl_agent_reward",
                    "display_name": "RL Agent Reward",
                    "description": "Reinforcement learning agent reward score",
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "unit": "1",
                    "labels": [
                        {"key": "agent_id", "description": "RL agent identifier"},
                        {"key": "episode", "description": "Training episode"}
                    ]
                }
            ],
            "notification_channels": [
                {
                    "type": "email",
                    "display_name": "Trading Alerts Email",
                    "labels": {
                        "email_address": os.getenv("ALERT_EMAIL", "alerts@example.com")
                    }
                }
            ],
            "alert_policies": [
                {
                    "display_name": "High Risk Level Alert",
                    "conditions": [
                        {
                            "filter": 'resource.type="gce_instance" AND metric.type="custom.googleapis.com/shyvr_rlte/safety_risk_level"',
                            "comparison": "COMPARISON_GT",
                            "threshold_value": 0.8,
                            "duration": "300s"
                        }
                    ],
                    "documentation": "Alert when risk level exceeds 80% for 5 minutes"
                },
                {
                    "display_name": "Trading System Down",
                    "conditions": [
                        {
                            "filter": 'resource.type="gce_instance" AND metric.type="custom.googleapis.com/shyvr_rlte/system_uptime"',
                            "comparison": "COMPARISON_LT",
                            "threshold_value": 1,
                            "duration": "60s"
                        }
                    ],
                    "documentation": "Alert when trading system is down for more than 1 minute"
                },
                {
                    "display_name": "Large Daily Loss Alert",
                    "conditions": [
                        {
                            "filter": 'resource.type="gce_instance" AND metric.type="custom.googleapis.com/shyvr_rlte/trading_pnl" AND metric.label.timeframe="daily"',
                            "comparison": "COMPARISON_LT",
                            "threshold_value": -500.0,
                            "duration": "0s"
                        }
                    ],
                    "documentation": "Immediate alert when daily loss exceeds $500"
                },
                {
                    "display_name": "Emergency Stop Triggered",
                    "conditions": [
                        {
                            "filter": 'resource.type="gce_instance" AND metric.type="custom.googleapis.com/shyvr_rlte/safety_emergency_stops"',
                            "comparison": "COMPARISON_GT",
                            "threshold_value": 0,
                            "duration": "0s",
                            "aggregation_per_series_aligner": "ALIGN_RATE"
                        }
                    ],
                    "documentation": "Immediate alert when emergency stop is triggered"
                },
                {
                    "display_name": "Low ML Model Accuracy",
                    "conditions": [
                        {
                            "filter": 'resource.type="gce_instance" AND metric.type="custom.googleapis.com/shyvr_rlte/ml_model_accuracy"',
                            "comparison": "COMPARISON_LT",
                            "threshold_value": 0.6,
                            "duration": "600s"
                        }
                    ],
                    "documentation": "Alert when ML model accuracy drops below 60% for 10 minutes"
                }
            ]
        }
    
    async def create_metric_descriptors(self, metrics_config: List[Dict]) -> List[str]:
        """
        Create custom metric descriptors.
        
        Args:
            metrics_config: List of metric configurations
            
        Returns:
            List of created metric names
        """
        created_metrics = []
        
        logger.info(f"Creating {len(metrics_config)} metric descriptors...")
        
        for metric_config in metrics_config:
            try:
                descriptor = CloudMetricDescriptor(
                    name=metric_config["name"],
                    display_name=metric_config["display_name"],
                    description=metric_config["description"],
                    metric_kind=metric_config["metric_kind"],
                    value_type=metric_config["value_type"],
                    unit=metric_config.get("unit", ""),
                    labels=metric_config.get("labels", [])
                )
                
                success = await self.client.create_metric_descriptor(descriptor)
                
                if success:
                    created_metrics.append(descriptor.name)
                    self.setup_results["metrics"].append({
                        "name": descriptor.name,
                        "status": "created"
                    })
                    logger.info(f"✓ Created metric: {descriptor.name}")
                else:
                    self.setup_results["errors"].append(f"Failed to create metric: {descriptor.name}")
                    logger.error(f"✗ Failed to create metric: {descriptor.name}")
                    
            except Exception as e:
                error_msg = f"Error creating metric {metric_config['name']}: {e}"
                self.setup_results["errors"].append(error_msg)
                logger.error(f"✗ {error_msg}")
        
        logger.info(f"Created {len(created_metrics)} metric descriptors")
        return created_metrics
    
    async def create_notification_channels(self, channels_config: List[Dict]) -> List[str]:
        """
        Create notification channels.
        
        Args:
            channels_config: List of notification channel configurations
            
        Returns:
            List of created channel IDs
        """
        created_channels = []
        
        logger.info(f"Creating {len(channels_config)} notification channels...")
        
        for channel_config in channels_config:
            try:
                channel = NotificationChannel(
                    type=channel_config["type"],
                    display_name=channel_config["display_name"],
                    labels=channel_config["labels"],
                    enabled=channel_config.get("enabled", True)
                )
                
                channel_id = await self.client.create_notification_channel(channel)
                
                if channel_id:
                    created_channels.append(channel_id)
                    self.setup_results["channels"].append({
                        "display_name": channel.display_name,
                        "id": channel_id,
                        "status": "created"
                    })
                    logger.info(f"✓ Created notification channel: {channel.display_name}")
                else:
                    error_msg = f"Failed to create notification channel: {channel.display_name}"
                    self.setup_results["errors"].append(error_msg)
                    logger.error(f"✗ {error_msg}")
                    
            except Exception as e:
                error_msg = f"Error creating notification channel {channel_config['display_name']}: {e}"
                self.setup_results["errors"].append(error_msg)
                logger.error(f"✗ {error_msg}")
        
        logger.info(f"Created {len(created_channels)} notification channels")
        return created_channels
    
    async def create_alert_policies(
        self, 
        alerts_config: List[Dict], 
        notification_channels: List[str]
    ) -> List[str]:
        """
        Create alert policies.
        
        Args:
            alerts_config: List of alert policy configurations
            notification_channels: List of notification channel IDs
            
        Returns:
            List of created policy IDs
        """
        created_policies = []
        
        logger.info(f"Creating {len(alerts_config)} alert policies...")
        
        for alert_config in alerts_config:
            try:
                conditions = []
                for condition_config in alert_config["conditions"]:
                    condition = AlertPolicyCondition(
                        filter=condition_config["filter"],
                        comparison=condition_config["comparison"],
                        threshold_value=condition_config["threshold_value"],
                        duration=condition_config.get("duration", "300s"),
                        aggregation_per_series_aligner=condition_config.get(
                            "aggregation_per_series_aligner", "ALIGN_RATE"
                        )
                    )
                    conditions.append(condition)
                
                policy_id = await self.client.create_alert_policy(
                    display_name=alert_config["display_name"],
                    conditions=conditions,
                    notification_channels=notification_channels,
                    documentation=alert_config.get("documentation", ""),
                    enabled=alert_config.get("enabled", True)
                )
                
                if policy_id:
                    created_policies.append(policy_id)
                    self.setup_results["alerts"].append({
                        "display_name": alert_config["display_name"],
                        "id": policy_id,
                        "status": "created"
                    })
                    logger.info(f"✓ Created alert policy: {alert_config['display_name']}")
                else:
                    error_msg = f"Failed to create alert policy: {alert_config['display_name']}"
                    self.setup_results["errors"].append(error_msg)
                    logger.error(f"✗ {error_msg}")
                    
            except Exception as e:
                error_msg = f"Error creating alert policy {alert_config['display_name']}: {e}"
                self.setup_results["errors"].append(error_msg)
                logger.error(f"✗ {error_msg}")
        
        logger.info(f"Created {len(created_policies)} alert policies")
        return created_policies
    
    async def validate_setup(self) -> Dict:
        """
        Validate the monitoring setup.
        
        Returns:
            Validation results
        """
        logger.info("Validating monitoring setup...")
        
        validation_results = {
            "status": "success",
            "checks": [],
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check if we can list alert policies (tests connection)
            policies = await self.client.list_alert_policies()
            validation_results["checks"].append({
                "check": "Cloud Monitoring connection",
                "status": "passed",
                "details": f"Found {len(policies)} alert policies"
            })
            
            # Check if our custom metrics exist (basic test)
            # This would require a more complex check in a real scenario
            validation_results["checks"].append({
                "check": "Custom metrics",
                "status": "assumed_passed",
                "details": "Metric descriptors were created (validation requires time series data)"
            })
            
            # Validate notification channels
            if self.setup_results["channels"]:
                validation_results["checks"].append({
                    "check": "Notification channels",
                    "status": "passed",
                    "details": f"Created {len(self.setup_results['channels'])} channels"
                })
            else:
                validation_results["warnings"].append("No notification channels were created")
            
            # Check for setup errors
            if self.setup_results["errors"]:
                validation_results["status"] = "partial_success"
                validation_results["errors"] = self.setup_results["errors"]
                
        except Exception as e:
            validation_results["status"] = "failed"
            validation_results["errors"].append(f"Validation failed: {e}")
        
        logger.info(f"Validation completed with status: {validation_results['status']}")
        return validation_results
    
    async def run_full_setup(self) -> Dict:
        """
        Run the complete monitoring setup process.
        
        Returns:
            Setup results summary
        """
        logger.info("Starting GCP monitoring setup...")
        
        try:
            # Load configuration
            config = self.load_configuration()
            
            # Create metric descriptors
            await self.create_metric_descriptors(config["metrics"])
            
            # Create notification channels
            notification_channels = await self.create_notification_channels(
                config["notification_channels"]
            )
            
            # Create alert policies
            await self.create_alert_policies(config["alert_policies"], notification_channels)
            
            # Validate setup
            validation_results = await self.validate_setup()
            
            # Compile final results
            setup_summary = {
                "project_id": self.project_id,
                "setup_time": "completed",
                "metrics_created": len(self.setup_results["metrics"]),
                "channels_created": len(self.setup_results["channels"]),
                "alerts_created": len(self.setup_results["alerts"]),
                "errors_count": len(self.setup_results["errors"]),
                "validation": validation_results,
                "details": self.setup_results
            }
            
            logger.info("GCP monitoring setup completed!")
            logger.info(f"Summary: {setup_summary['metrics_created']} metrics, "
                       f"{setup_summary['channels_created']} channels, "
                       f"{setup_summary['alerts_created']} alerts, "
                       f"{setup_summary['errors_count']} errors")
            
            return setup_summary
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return {
                "project_id": self.project_id,
                "status": "failed",
                "error": str(e),
                "details": self.setup_results
            }
    
    def save_results(self, results: Dict, output_path: str = "monitoring_setup_results.json"):
        """Save setup results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Setup results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")


async def main():
    """Main function for the setup script."""
    parser = argparse.ArgumentParser(description="Setup GCP monitoring for Shyvr RLTE")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--config", help="Path to monitoring configuration file")
    parser.add_argument("--output", default="monitoring_setup_results.json", 
                       help="Output file for results")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate existing setup")
    
    args = parser.parse_args()
    
    # Initialize setup manager
    setup_manager = GCPMonitoringSetup(args.project_id, args.config)
    
    try:
        if args.validate_only:
            # Run validation only
            results = await setup_manager.validate_setup()
            print(f"Validation status: {results['status']}")
        else:
            # Run full setup
            results = await setup_manager.run_full_setup()
            print(f"Setup completed for project: {args.project_id}")
        
        # Save results
        setup_manager.save_results(results, args.output)
        
        # Print summary
        if not args.validate_only:
            print(f"\nSetup Summary:")
            print(f"  Metrics created: {results.get('metrics_created', 0)}")
            print(f"  Channels created: {results.get('channels_created', 0)}")
            print(f"  Alerts created: {results.get('alerts_created', 0)}")
            print(f"  Errors: {results.get('errors_count', 0)}")
        
        # Exit with appropriate code
        if results.get('errors_count', 0) > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Setup script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())