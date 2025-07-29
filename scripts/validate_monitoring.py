#!/usr/bin/env python3
"""
Monitoring validation and health check script.

This script validates the complete monitoring setup for the Shyvr RLTE trading system,
including Cloud Monitoring, alert policies, notification channels, and dashboard connectivity.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import httpx
from google.cloud import monitoring_v3
from google.api_core import exceptions as gcp_exceptions

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring.cloud_monitoring import CloudMonitoringClient
from monitoring.notifications import NotificationManager, NotificationLevel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringValidator:
    """
    Comprehensive monitoring validation system.
    
    This class validates:
    - Cloud Monitoring connectivity and metrics
    - Alert policies and notification channels
    - Grafana dashboard connectivity
    - End-to-end notification flow
    """
    
    def __init__(self, project_id: str, config_path: Optional[str] = None):
        """
        Initialize monitoring validator.
        
        Args:
            project_id: GCP project ID
            config_path: Path to monitoring configuration file
        """
        self.project_id = project_id
        self.config_path = config_path or "config/monitoring_config.json"
        self.cloud_client = CloudMonitoringClient(project_id)
        
        self.validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": project_id,
            "overall_status": "unknown",
            "checks": [],
            "warnings": [],
            "errors": [],
            "summary": {}
        }
        
        logger.info(f"Initialized monitoring validator for project: {project_id}")
    
    def load_configuration(self) -> Dict:
        """Load monitoring configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    async def validate_gcp_connectivity(self) -> Dict[str, Any]:
        """Validate Google Cloud Monitoring connectivity."""
        check_result = {
            "check": "GCP Cloud Monitoring Connectivity",
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            # Test basic connectivity by listing alert policies
            policies = await self.cloud_client.list_alert_policies()
            
            check_result["status"] = "passed"
            check_result["details"]["alert_policies_count"] = len(policies)
            check_result["details"]["connection_test"] = "successful"
            
            logger.info(f"GCP connectivity validated - found {len(policies)} alert policies")
            
        except Exception as e:
            check_result["status"] = "failed"
            check_result["errors"].append(f"GCP connection failed: {e}")
            logger.error(f"GCP connectivity validation failed: {e}")
        
        return check_result
    
    async def validate_custom_metrics(self) -> Dict[str, Any]:
        """Validate custom metrics are properly configured."""
        check_result = {
            "check": "Custom Metrics Configuration",
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            config = self.load_configuration()
            expected_metrics = config.get("metrics", [])
            
            if not expected_metrics:
                check_result["status"] = "warning"
                check_result["errors"].append("No custom metrics defined in configuration")
                return check_result
            
            # List metric descriptors
            client = monitoring_v3.MetricServiceClient()
            project_name = f"projects/{self.project_id}"
            
            descriptors = []
            try:
                for descriptor in client.list_metric_descriptors(name=project_name):
                    if descriptor.type.startswith("custom.googleapis.com/shyvr_rlte/"):
                        descriptors.append(descriptor.type)
            except Exception as e:
                check_result["status"] = "failed"
                check_result["errors"].append(f"Failed to list metric descriptors: {e}")
                return check_result
            
            # Check if expected metrics exist
            expected_metric_names = [
                f"custom.googleapis.com/shyvr_rlte/{m['name']}" 
                for m in expected_metrics
            ]
            
            missing_metrics = [
                name for name in expected_metric_names 
                if name not in descriptors
            ]
            
            check_result["details"]["expected_metrics"] = len(expected_metric_names)
            check_result["details"]["found_metrics"] = len(descriptors)
            check_result["details"]["missing_metrics"] = missing_metrics
            
            if missing_metrics:
                check_result["status"] = "warning"
                check_result["errors"].append(f"Missing {len(missing_metrics)} metrics: {missing_metrics}")
            else:
                check_result["status"] = "passed"
            
            logger.info(f"Custom metrics validation: {len(descriptors)}/{len(expected_metric_names)} found")
            
        except Exception as e:
            check_result["status"] = "failed"
            check_result["errors"].append(f"Metrics validation failed: {e}")
            logger.error(f"Custom metrics validation failed: {e}")
        
        return check_result
    
    async def validate_alert_policies(self) -> Dict[str, Any]:
        """Validate alert policies are properly configured."""
        check_result = {
            "check": "Alert Policies Configuration",
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            config = self.load_configuration()
            expected_alerts = config.get("alert_policies", [])
            
            # Get actual alert policies
            policies = await self.cloud_client.list_alert_policies()
            
            # Filter for Shyvr RLTE policies
            shyvr_policies = [
                p for p in policies 
                if "shyvr" in p["display_name"].lower() or "rlte" in p["display_name"].lower()
            ]
            
            check_result["details"]["total_policies"] = len(policies)
            check_result["details"]["shyvr_policies"] = len(shyvr_policies)
            check_result["details"]["expected_policies"] = len(expected_alerts)
            
            # Check if critical policies exist
            critical_policy_names = [
                "trading system down",
                "critical risk level",
                "emergency stop"
            ]
            
            found_critical = []
            for policy in shyvr_policies:
                policy_name = policy["display_name"].lower()
                for critical in critical_policy_names:
                    if critical in policy_name:
                        found_critical.append(critical)
            
            check_result["details"]["critical_policies_found"] = found_critical
            missing_critical = [c for c in critical_policy_names if c not in found_critical]
            
            if missing_critical:
                check_result["status"] = "warning"
                check_result["errors"].append(f"Missing critical policies: {missing_critical}")
            elif len(shyvr_policies) >= len(expected_alerts) * 0.8:  # 80% threshold
                check_result["status"] = "passed"
            else:
                check_result["status"] = "warning"
                check_result["errors"].append("Fewer alert policies than expected")
            
            logger.info(f"Alert policies validation: {len(shyvr_policies)} Shyvr policies found")
            
        except Exception as e:
            check_result["status"] = "failed"
            check_result["errors"].append(f"Alert policies validation failed: {e}")
            logger.error(f"Alert policies validation failed: {e}")
        
        return check_result
    
    async def validate_notification_channels(self) -> Dict[str, Any]:
        """Validate notification channels configuration."""
        check_result = {
            "check": "Notification Channels",
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            # Test notification manager initialization
            notification_config = {
                "channels": {
                    "test_email": {
                        "type": "email",
                        "enabled": False,  # Disabled for testing
                        "smtp_host": "smtp.gmail.com",
                        "smtp_port": 587,
                        "smtp_user": "test@example.com",
                        "smtp_password": "test",
                        "from_address": "test@example.com",
                        "to_addresses": ["test@example.com"]
                    }
                }
            }
            
            # Try to initialize notification manager
            try:
                notification_manager = NotificationManager(notification_config)
                check_result["details"]["notification_manager"] = "initialized"
                
                # Get channel status
                channel_status = notification_manager.get_channel_status()
                check_result["details"]["channels"] = channel_status
                
                check_result["status"] = "passed"
                logger.info("Notification channels validation passed")
                
            except Exception as e:
                check_result["status"] = "failed"
                check_result["errors"].append(f"Notification manager initialization failed: {e}")
            
        except Exception as e:
            check_result["status"] = "failed"
            check_result["errors"].append(f"Notification channels validation failed: {e}")
            logger.error(f"Notification channels validation failed: {e}")
        
        return check_result
    
    async def validate_grafana_connectivity(self) -> Dict[str, Any]:
        """Validate Grafana dashboard connectivity."""
        check_result = {
            "check": "Grafana Dashboard Connectivity",
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        api_key = os.getenv("GRAFANA_API_KEY")
        
        if not api_key:
            check_result["status"] = "skipped"
            check_result["errors"].append("Grafana API key not configured")
            return check_result
        
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            async with httpx.AsyncClient() as client:
                # Test connection
                response = await client.get(
                    f"{grafana_url}/api/org",
                    headers=headers,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    org_info = response.json()
                    check_result["details"]["grafana_org"] = org_info.get("name")
                    
                    # Check for dashboards
                    dash_response = await client.get(
                        f"{grafana_url}/api/search?query=shyvr",
                        headers=headers,
                        timeout=10.0
                    )
                    
                    if dash_response.status_code == 200:
                        dashboards = dash_response.json()
                        check_result["details"]["shyvr_dashboards"] = len(dashboards)
                        
                        if len(dashboards) > 0:
                            check_result["status"] = "passed"
                        else:
                            check_result["status"] = "warning"
                            check_result["errors"].append("No Shyvr dashboards found")
                    else:
                        check_result["status"] = "warning"
                        check_result["errors"].append("Could not retrieve dashboards")
                else:
                    check_result["status"] = "failed"
                    check_result["errors"].append(f"Grafana connection failed: {response.status_code}")
            
            logger.info(f"Grafana connectivity validated - {check_result['status']}")
            
        except Exception as e:
            check_result["status"] = "failed"
            check_result["errors"].append(f"Grafana validation failed: {e}")
            logger.error(f"Grafana connectivity validation failed: {e}")
        
        return check_result
    
    async def test_metric_ingestion(self) -> Dict[str, Any]:
        """Test metric ingestion by sending a test metric."""
        check_result = {
            "check": "Metric Ingestion Test",
            "status": "unknown",
            "details": {},
            "errors": []
        }
        
        try:
            # Send a test metric
            test_metric_name = "system/test_metric"
            test_value = 1.0
            test_labels = {"test": "validation", "timestamp": str(int(datetime.utcnow().timestamp()))}
            
            success = await self.cloud_client.write_time_series(
                metric_type=test_metric_name,
                value=test_value,
                labels=test_labels
            )
            
            if success:
                check_result["status"] = "passed"
                check_result["details"]["test_metric_sent"] = True
                check_result["details"]["metric_name"] = f"custom.googleapis.com/shyvr_rlte/{test_metric_name}"
                logger.info("Test metric ingestion successful")
            else:
                check_result["status"] = "failed"
                check_result["errors"].append("Failed to send test metric")
            
        except Exception as e:
            check_result["status"] = "failed"
            check_result["errors"].append(f"Metric ingestion test failed: {e}")
            logger.error(f"Metric ingestion test failed: {e}")
        
        return check_result
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks."""
        logger.info("Starting comprehensive monitoring validation...")
        
        # Run all validation checks
        validation_checks = [
            self.validate_gcp_connectivity(),
            self.validate_custom_metrics(),
            self.validate_alert_policies(),
            self.validate_notification_channels(),
            self.validate_grafana_connectivity(),
            self.test_metric_ingestion()
        ]
        
        check_results = await asyncio.gather(*validation_checks, return_exceptions=True)
        
        # Process results
        passed_checks = 0
        warning_checks = 0
        failed_checks = 0
        skipped_checks = 0
        
        for i, result in enumerate(check_results):
            if isinstance(result, Exception):
                # Handle unexpected exceptions
                error_result = {
                    "check": f"Validation Check {i+1}",
                    "status": "failed",
                    "errors": [f"Unexpected error: {result}"]
                }
                self.validation_results["checks"].append(error_result)
                failed_checks += 1
            else:
                self.validation_results["checks"].append(result)
                
                if result["status"] == "passed":
                    passed_checks += 1
                elif result["status"] == "warning":
                    warning_checks += 1
                    self.validation_results["warnings"].extend(result.get("errors", []))
                elif result["status"] == "failed":
                    failed_checks += 1
                    self.validation_results["errors"].extend(result.get("errors", []))
                elif result["status"] == "skipped":
                    skipped_checks += 1
        
        # Determine overall status
        total_checks = len(check_results)
        if failed_checks == 0 and warning_checks == 0:
            self.validation_results["overall_status"] = "healthy"
        elif failed_checks == 0:
            self.validation_results["overall_status"] = "warning"
        elif failed_checks < total_checks / 2:
            self.validation_results["overall_status"] = "degraded"
        else:
            self.validation_results["overall_status"] = "unhealthy"
        
        # Update summary
        self.validation_results["summary"] = {
            "total_checks": total_checks,
            "passed": passed_checks,
            "warnings": warning_checks,
            "failed": failed_checks,
            "skipped": skipped_checks,
            "success_rate": round((passed_checks / max(total_checks - skipped_checks, 1)) * 100, 1)
        }
        
        logger.info(f"Validation completed - Status: {self.validation_results['overall_status']}")
        logger.info(f"Summary: {passed_checks} passed, {warning_checks} warnings, {failed_checks} failed, {skipped_checks} skipped")
        
        return self.validation_results
    
    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "degraded": "⚠️",
            "unhealthy": "❌"
        }
        
        check_emoji = {
            "passed": "✅",
            "warning": "⚠️",
            "failed": "❌",
            "skipped": "⏭️"
        }
        
        report = f"""
================================================================================
SHYVR RLTE MONITORING VALIDATION REPORT
================================================================================

Project ID: {self.validation_results['project_id']}
Validation Time: {self.validation_results['timestamp']}
Overall Status: {status_emoji.get(self.validation_results['overall_status'], '❓')} {self.validation_results['overall_status'].upper()}

SUMMARY:
  Total Checks: {self.validation_results['summary']['total_checks']}
  ✅ Passed: {self.validation_results['summary']['passed']}
  ⚠️ Warnings: {self.validation_results['summary']['warnings']}
  ❌ Failed: {self.validation_results['summary']['failed']}
  ⏭️ Skipped: {self.validation_results['summary']['skipped']}
  Success Rate: {self.validation_results['summary']['success_rate']}%

DETAILED RESULTS:
"""
        
        for check in self.validation_results['checks']:
            emoji = check_emoji.get(check['status'], '❓')
            report += f"\n{emoji} {check['check']}: {check['status'].upper()}\n"
            
            if check.get('details'):
                for key, value in check['details'].items():
                    report += f"    {key}: {value}\n"
            
            if check.get('errors'):
                for error in check['errors']:
                    report += f"    ⚠️ {error}\n"
        
        if self.validation_results['warnings']:
            report += "\nWARNINGS:\n"
            for warning in self.validation_results['warnings']:
                report += f"  ⚠️ {warning}\n"
        
        if self.validation_results['errors']:
            report += "\nERRORS:\n"
            for error in self.validation_results['errors']:
                report += f"  ❌ {error}\n"
        
        report += f"\n\nRECOMMENDATIONS:\n"
        
        if self.validation_results['overall_status'] == 'healthy':
            report += "  ✅ Monitoring system is healthy and fully operational.\n"
        elif self.validation_results['overall_status'] == 'warning':
            report += "  ⚠️ Monitoring system is functional but has minor issues.\n"
            report += "  📋 Review warnings and consider addressing them.\n"
        elif self.validation_results['overall_status'] == 'degraded':
            report += "  ⚠️ Monitoring system has significant issues that should be addressed.\n"
            report += "  🔧 Review failed checks and fix configuration issues.\n"
        else:
            report += "  ❌ Monitoring system has critical issues requiring immediate attention.\n"
            report += "  🚨 Review all failed checks and fix before using in production.\n"
        
        report += f"\n================================================================================\n"
        
        return report


async def main():
    """Main function for the validation script."""
    parser = argparse.ArgumentParser(description="Validate Shyvr RLTE monitoring setup")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--config", help="Path to monitoring configuration file")
    parser.add_argument("--output", default="monitoring_validation_results.json",
                       help="Output file for results")
    parser.add_argument("--report", default="monitoring_validation_report.txt",
                       help="Output file for human-readable report")
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = MonitoringValidator(args.project_id, args.config)
        
        # Run validation
        results = await validator.run_comprehensive_validation()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate and save report
        report = validator.generate_report()
        with open(args.report, 'w') as f:
            f.write(report)
        
        # Print summary
        print(f"Validation completed with status: {results['overall_status']}")
        print(f"Success rate: {results['summary']['success_rate']}%")
        print(f"Results saved to: {args.output}")
        print(f"Report saved to: {args.report}")
        
        # Also print the report to console
        print(report)
        
        # Exit with appropriate code
        if results['overall_status'] in ['healthy', 'warning']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())