#!/usr/bin/env python3
"""
Grafana GCP integration setup script.

This script sets up Grafana dashboards to work with Google Cloud Monitoring
for the Shyvr RLTE trading system.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GrafanaGCPIntegration:
    """
    Grafana integration manager for GCP monitoring.
    
    This class handles:
    - Creating GCP data sources in Grafana
    - Importing monitoring dashboards
    - Setting up alert rules
    - Configuring notification channels
    """
    
    def __init__(
        self, 
        grafana_url: str, 
        api_key: str, 
        project_id: str,
        service_account_path: Optional[str] = None
    ):
        """
        Initialize Grafana GCP integration.
        
        Args:
            grafana_url: Grafana instance URL
            api_key: Grafana API key with admin permissions
            project_id: GCP project ID
            service_account_path: Path to GCP service account JSON file
        """
        self.grafana_url = grafana_url.rstrip('/')
        self.api_key = api_key
        self.project_id = project_id
        self.service_account_path = service_account_path
        
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Initialized Grafana GCP integration for project: {project_id}")
    
    async def test_connection(self) -> bool:
        """Test connection to Grafana API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.grafana_url}/api/org",
                    headers=self.headers,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    org_info = response.json()
                    logger.info(f"Connected to Grafana org: {org_info.get('name', 'Unknown')}")
                    return True
                else:
                    logger.error(f"Grafana connection failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to connect to Grafana: {e}")
            return False
    
    async def create_gcp_datasource(self) -> Optional[Dict[str, Any]]:
        """Create Google Cloud Monitoring data source in Grafana."""
        try:
            # Check if data source already exists
            async with httpx.AsyncClient() as client:
                # List existing data sources
                response = await client.get(
                    f"{self.grafana_url}/api/datasources",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    datasources = response.json()
                    for ds in datasources:
                        if (ds.get('type') == 'stackdriver' and 
                            ds.get('jsonData', {}).get('defaultProject') == self.project_id):
                            logger.info(f"GCP data source already exists: {ds['name']}")
                            return ds
                
                # Create new data source
                datasource_config = {
                    "name": f"GCP Monitoring - {self.project_id}",
                    "type": "stackdriver",
                    "access": "proxy",
                    "url": "",
                    "jsonData": {
                        "defaultProject": self.project_id,
                        "authenticationType": "gce" if not self.service_account_path else "jwt"
                    },
                    "isDefault": False
                }
                
                # Add service account authentication if provided
                if self.service_account_path and os.path.exists(self.service_account_path):
                    with open(self.service_account_path, 'r') as f:
                        service_account = json.load(f)
                    
                    datasource_config["jsonData"]["clientEmail"] = service_account["client_email"]
                    datasource_config["jsonData"]["privateKey"] = service_account["private_key"]
                    datasource_config["jsonData"]["tokenUri"] = service_account["token_uri"]
                
                response = await client.post(
                    f"{self.grafana_url}/api/datasources",
                    headers=self.headers,
                    json=datasource_config
                )
                
                if response.status_code == 200:
                    datasource = response.json()
                    logger.info(f"Created GCP data source: {datasource['name']}")
                    return datasource
                else:
                    logger.error(f"Failed to create data source: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error creating GCP data source: {e}")
            return None
    
    async def import_dashboard(self, dashboard_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Import a dashboard into Grafana."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.grafana_url}/api/dashboards/db",
                    headers=self.headers,
                    json=dashboard_config
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Imported dashboard: {result['title']}")
                    return result
                else:
                    logger.error(f"Failed to import dashboard: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error importing dashboard: {e}")
            return None
    
    def generate_trading_dashboard(self, datasource_name: str) -> Dict[str, Any]:
        """Generate trading performance dashboard configuration."""
        return {
            "dashboard": {
                "id": None,
                "title": "Shyvr RLTE - Trading Performance",
                "tags": ["shyvr", "trading", "rlte"],
                "timezone": "utc",
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"]
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "System Status",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/system_uptime",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "green", "value": 1}
                                    ]
                                },
                                "mappings": [
                                    {"type": "value", "value": "1", "text": "Online"},
                                    {"type": "value", "value": "0", "text": "Offline"}
                                ]
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "Daily P&L",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 6, "y": 0},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/trading_daily_pnl",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "unit": "currencyUSD"
                            }
                        }
                    },
                    {
                        "id": 3,
                        "title": "Risk Level",
                        "type": "gauge",
                        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/safety_risk_level",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 0.7},
                                        {"color": "red", "value": 0.9}
                                    ]
                                },
                                "min": 0,
                                "max": 1,
                                "unit": "percentunit"
                            }
                        }
                    },
                    {
                        "id": 4,
                        "title": "Trading Success Rate",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/trading_success_rate",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "unit": "percentunit",
                                "min": 0,
                                "max": 1
                            }
                        }
                    },
                    {
                        "id": 5,
                        "title": "Active Positions",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/trading_position_count",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "value"},
                                "unit": "short"
                            }
                        }
                    },
                    {
                        "id": 6,
                        "title": "ML Model Accuracy",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/ml_model_accuracy",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "unit": "percentunit",
                                "min": 0,
                                "max": 1
                            }
                        }
                    },
                    {
                        "id": 7,
                        "title": "System Error Rate",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/system_error_rate",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "unit": "reqps"
                            }
                        }
                    }
                ]
            },
            "overwrite": True
        }
    
    def generate_risk_dashboard(self, datasource_name: str) -> Dict[str, Any]:
        """Generate risk management dashboard configuration."""
        return {
            "dashboard": {
                "id": None,
                "title": "Shyvr RLTE - Risk Management",
                "tags": ["shyvr", "risk", "safety"],
                "timezone": "utc",
                "refresh": "15s",
                "time": {
                    "from": "now-2h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "Overall Risk Level",
                        "type": "gauge",
                        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/safety_risk_level",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 0.6},
                                        {"color": "orange", "value": 0.8},
                                        {"color": "red", "value": 0.9}
                                    ]
                                },
                                "min": 0,
                                "max": 1,
                                "unit": "percentunit"
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "Portfolio Drawdown",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 16, "x": 8, "y": 0},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/safety_drawdown_percentage",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "unit": "percent"
                            }
                        }
                    },
                    {
                        "id": 3,
                        "title": "Emergency Stops",
                        "type": "stat",
                        "gridPos": {"h": 6, "w": 6, "x": 0, "y": 8},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/safety_emergency_stops",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "red", "value": 1}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 4,
                        "title": "Liquidations",
                        "type": "stat",
                        "gridPos": {"h": 6, "w": 6, "x": 6, "y": 8},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/safety_liquidations",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 1},
                                        {"color": "red", "value": 5}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 5,
                        "title": "Margin Ratio",
                        "type": "gauge",
                        "gridPos": {"h": 6, "w": 6, "x": 12, "y": 8},
                        "targets": [
                            {
                                "datasource": datasource_name,
                                "metricType": "custom.googleapis.com/shyvr_rlte/safety_margin_ratio",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 1.2},
                                        {"color": "green", "value": 2.0}
                                    ]
                                },
                                "min": 0,
                                "max": 5,
                                "unit": "short"
                            }
                        }
                    }
                ]
            },
            "overwrite": True
        }
    
    async def setup_dashboards(self, datasource_name: str) -> Dict[str, Any]:
        """Set up all monitoring dashboards."""
        results = {}
        
        # Import trading dashboard
        trading_dashboard = self.generate_trading_dashboard(datasource_name)
        trading_result = await self.import_dashboard(trading_dashboard)
        results['trading'] = trading_result
        
        # Import risk dashboard
        risk_dashboard = self.generate_risk_dashboard(datasource_name)
        risk_result = await self.import_dashboard(risk_dashboard)
        results['risk'] = risk_result
        
        return results
    
    async def create_notification_channel(
        self, 
        name: str, 
        channel_type: str, 
        settings: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a notification channel in Grafana."""
        try:
            notification_channel = {
                "name": name,
                "type": channel_type,
                "settings": settings,
                "isDefault": False,
                "sendReminder": True,
                "disableResolveMessage": False,
                "frequency": "10s"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.grafana_url}/api/alert-notifications",
                    headers=self.headers,
                    json=notification_channel
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Created notification channel: {name}")
                    return result
                else:
                    logger.error(f"Failed to create notification channel: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error creating notification channel: {e}")
            return None
    
    async def run_full_setup(self) -> Dict[str, Any]:
        """Run complete Grafana GCP integration setup."""
        results = {
            "status": "success",
            "datasource": None,
            "dashboards": {},
            "notification_channels": {},
            "errors": []
        }
        
        try:
            # Test connection
            if not await self.test_connection():
                results["status"] = "failed"
                results["errors"].append("Failed to connect to Grafana")
                return results
            
            # Create GCP data source
            datasource = await self.create_gcp_datasource()
            if not datasource:
                results["status"] = "failed"
                results["errors"].append("Failed to create GCP data source")
                return results
            
            results["datasource"] = datasource
            datasource_name = datasource["name"]
            
            # Set up dashboards
            dashboard_results = await self.setup_dashboards(datasource_name)
            results["dashboards"] = dashboard_results
            
            # Create email notification channel if configured
            email_settings = os.getenv("GRAFANA_EMAIL_SETTINGS")
            if email_settings:
                try:
                    email_config = json.loads(email_settings)
                    email_channel = await self.create_notification_channel(
                        "Shyvr RLTE Email Alerts",
                        "email",
                        email_config
                    )
                    if email_channel:
                        results["notification_channels"]["email"] = email_channel
                except Exception as e:
                    results["errors"].append(f"Failed to create email notification: {e}")
            
            logger.info("Grafana GCP setup completed successfully")
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"Setup failed: {e}")
            logger.error(f"Grafana setup failed: {e}")
        
        return results


async def main():
    """Main function for the Grafana GCP setup script."""
    parser = argparse.ArgumentParser(description="Setup Grafana GCP integration for Shyvr RLTE")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--grafana-url", default=os.getenv("GRAFANA_URL", "http://localhost:3000"),
                       help="Grafana URL")
    parser.add_argument("--api-key", default=os.getenv("GRAFANA_API_KEY"),
                       help="Grafana API key")
    parser.add_argument("--service-account", help="Path to GCP service account JSON")
    parser.add_argument("--output", default="grafana_setup_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    if not args.api_key:
        logger.error("Grafana API key is required (--api-key or GRAFANA_API_KEY env var)")
        sys.exit(1)
    
    try:
        # Initialize integration
        integration = GrafanaGCPIntegration(
            grafana_url=args.grafana_url,
            api_key=args.api_key,
            project_id=args.project_id,
            service_account_path=args.service_account
        )
        
        # Run setup
        results = await integration.run_full_setup()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print(f"Grafana GCP setup completed with status: {results['status']}")
        if results['errors']:
            print(f"Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
        
        # Exit with appropriate code
        sys.exit(0 if results['status'] == 'success' else 1)
        
    except Exception as e:
        logger.error(f"Setup script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())