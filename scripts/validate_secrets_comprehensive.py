#!/usr/bin/env python3
"""
Comprehensive Secret Availability Validation Script for Shyvr RLTE
Enhanced version of secret validation with detailed reporting and security checks
"""

import asyncio
import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from google.cloud import secretmanager
from google.cloud.exceptions import NotFound, PermissionDenied

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SecretValidationResult:
    """Secret validation result container"""
    secret_category: str
    secret_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    validation_time_ms: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

class ComprehensiveSecretValidator:
    """Comprehensive secret validation system"""
    
    def __init__(self, project_id: str = "shvyr-ai-bots"):
        self.project_id = project_id
        self.client = None
        self.results: List[SecretValidationResult] = []
        
        # Define secret categories with criticality levels
        self.secret_categories = {
            "critical": {
                "description": "Required for basic system operation",
                "secrets": [
                    ("TELEGRAM_TOKEN", "Telegram bot authentication token"),
                    ("WEBHOOK_SECRET", "Webhook security secret"),
                    ("DB_PASSWORD", "Database password"),
                    ("DATABASE_URL", "Complete database connection URL")
                ]
            },
            "database": {
                "description": "Database-specific secrets",
                "secrets": [
                    ("db-password-production", "Production database password"),
                    ("db-user-production", "Production database user"),
                    ("DATABASE_URL", "Database connection string")
                ]
            },
            "api_keys": {
                "description": "External API service keys",
                "secrets": [
                    ("HELIUS_API_KEY", "Helius Solana RPC service"),
                    ("BIRDEYE_API_KEY", "Birdeye market data service"),
                    ("ETHERSCAN_API_KEY", "Etherscan blockchain data"),
                    ("XAI_API_KEY", "AI/ML service API key"),
                    ("OPENAI_API_KEY", "OpenAI API key"),
                    ("LUNARCRUSH_API_KEY", "LunarCrush social data"),
                    ("COINGECKO_API_KEY", "CoinGecko price data")
                ]
            },
            "social_api": {
                "description": "Social media API credentials",
                "secrets": [
                    ("X_BEARER_TOKEN", "X (Twitter) API bearer token"),
                    ("X_API_KEY", "X (Twitter) API key"),
                    ("X_API_SECRET", "X (Twitter) API secret")
                ]
            },
            "trading_keys": {
                "description": "Trading wallet private keys (HIGH RISK)",
                "secrets": [
                    ("SOLANA_PRIVATE_KEY", "Solana wallet private key"),
                    ("ETHEREUM_PRIVATE_KEY", "Ethereum wallet private key"),
                    ("HYPERLIQUID_PRIVATE_KEY", "Hyperliquid trading key"),
                    ("ETH_PRIVATE_KEY", "Alternative Ethereum private key"),
                    ("BASE_PRIVATE_KEY", "Base chain private key")
                ]
            },
            "monitoring": {
                "description": "Monitoring and alerting credentials",
                "secrets": [
                    ("TELEGRAM_MONITORING_BOT_TOKEN", "Monitoring alerts bot token"),
                    ("GRAFANA_API_KEY", "Grafana dashboard API key"),
                    ("SENTRY_DSN", "Error tracking service DSN"),
                    ("PROMETHEUS_PUSH_GATEWAY", "Metrics push gateway URL")
                ]
            }
        }
    
    def add_result(self, result: SecretValidationResult):
        """Add validation result"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        time_info = f" ({result.validation_time_ms:.0f}ms)" if result.validation_time_ms else ""
        logger.info(f"{status} {result.secret_category}.{result.secret_name}{time_info}: {result.message}")
    
    async def initialize_client(self) -> SecretValidationResult:
        """Initialize Secret Manager client"""
        try:
            start_time = time.time()
            self.client = secretmanager.SecretManagerServiceClient()
            
            # Test client by listing first secret (to verify permissions)
            try:
                parent = f"projects/{self.project_id}"
                secrets = list(self.client.list_secrets(request={"parent": parent}, page_size=1))
                init_time = (time.time() - start_time) * 1000
                
                return SecretValidationResult(
                    secret_category="Client",
                    secret_name="Secret Manager Client",
                    passed=True,
                    message=f"Secret Manager client initialized successfully",
                    validation_time_ms=init_time,
                    details={"project_id": self.project_id, "can_list_secrets": True}
                )
                
            except PermissionDenied:
                init_time = (time.time() - start_time) * 1000
                return SecretValidationResult(
                    secret_category="Client",
                    secret_name="Secret Manager Client",
                    passed=False,
                    message="Permission denied - insufficient IAM permissions",
                    validation_time_ms=init_time,
                    details={"project_id": self.project_id, "error": "permission_denied"}
                )
                
        except Exception as e:
            init_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            return SecretValidationResult(
                secret_category="Client",
                secret_name="Secret Manager Client",
                passed=False,
                message=f"Failed to initialize Secret Manager client: {str(e)}",
                validation_time_ms=init_time,
                details={"error": str(e), "project_id": self.project_id}
            )
    
    async def get_secret_value(self, secret_name: str, version: str = "latest") -> tuple[bool, Optional[str], Dict[str, Any]]:
        """Get secret value with detailed error information"""
        if not self.client:
            return False, None, {"error": "Client not initialized"}
        
        try:
            start_time = time.time()
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
            response = self.client.access_secret_version(request={"name": name})
            access_time = (time.time() - start_time) * 1000
            
            secret_value = response.payload.data.decode("UTF-8")
            
            return True, secret_value, {
                "access_time_ms": access_time,
                "secret_length": len(secret_value),
                "version": version,
                "has_value": len(secret_value) > 0
            }
            
        except NotFound:
            access_time = (time.time() - start_time) * 1000
            return False, None, {
                "error": "not_found",
                "access_time_ms": access_time,
                "secret_name": secret_name
            }
            
        except PermissionDenied:
            access_time = (time.time() - start_time) * 1000
            return False, None, {
                "error": "permission_denied",
                "access_time_ms": access_time,
                "secret_name": secret_name
            }
            
        except Exception as e:
            access_time = (time.time() - start_time) * 1000
            return False, None, {
                "error": str(e),
                "access_time_ms": access_time,
                "secret_name": secret_name
            }
    
    async def validate_secret_category(self, category_name: str, category_info: Dict[str, Any]) -> List[SecretValidationResult]:
        """Validate all secrets in a category"""
        results = []
        
        logger.info(f"🔍 Validating {category_name} secrets: {category_info['description']}")
        
        for secret_name, description in category_info["secrets"]:
            success, value, details = await self.get_secret_value(secret_name)
            
            # Determine criticality based on category
            is_critical = category_name in ["critical", "database"]
            is_trading = category_name == "trading_keys"
            
            # Create validation result
            if success and value:
                # Additional validation for specific secret types
                additional_checks = await self.validate_secret_format(secret_name, value)
                
                # For trading keys, warn about security risk
                if is_trading:
                    message = f"🚨 TRADING KEY AVAILABLE - LIVE TRADING ENABLED ({description})"
                    passed = True  # Available but risky
                else:
                    message = f"Available and accessible ({description})"
                    passed = True
                
                details.update(additional_checks)
                
            else:
                if is_critical:
                    message = f"CRITICAL SECRET MISSING - {description}"
                    passed = False
                elif is_trading:
                    message = f"Trading key not configured (SAFE) - {description}"
                    passed = True  # Missing trading keys are safer
                else:
                    message = f"Optional secret not configured - {description}"
                    passed = True  # Optional secrets don't fail validation
            
            results.append(SecretValidationResult(
                secret_category=category_name,
                secret_name=secret_name,
                passed=passed,
                message=message,
                validation_time_ms=details.get("access_time_ms"),
                details=details
            ))
        
        return results
    
    async def validate_secret_format(self, secret_name: str, value: str) -> Dict[str, Any]:
        """Validate secret format and basic security characteristics"""
        checks = {}
        
        # Length checks
        checks["length"] = len(value)
        checks["is_empty"] = len(value) == 0
        
        # Format-specific checks
        if "API_KEY" in secret_name.upper():
            checks["format_type"] = "api_key"
            checks["looks_like_api_key"] = len(value) > 10 and any(c.isalnum() for c in value)
            
        elif "TOKEN" in secret_name.upper():
            checks["format_type"] = "token"
            checks["looks_like_token"] = len(value) > 20
            
        elif "PRIVATE_KEY" in secret_name.upper():
            checks["format_type"] = "private_key"
            checks["is_hex"] = all(c in '0123456789abcdefABCDEF' for c in value.replace('0x', ''))
            checks["has_0x_prefix"] = value.startswith('0x')
            checks["security_risk"] = True  # Private keys are high risk
            
        elif "PASSWORD" in secret_name.upper():
            checks["format_type"] = "password"
            checks["has_special_chars"] = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in value)
            checks["has_numbers"] = any(c.isdigit() for c in value)
            checks["has_uppercase"] = any(c.isupper() for c in value)
            
        elif "URL" in secret_name.upper():
            checks["format_type"] = "url"
            checks["looks_like_url"] = value.startswith(('http://', 'https://', 'postgres://'))
            
        else:
            checks["format_type"] = "unknown"
        
        # Security checks
        checks["entropy_score"] = len(set(value)) / len(value) if len(value) > 0 else 0
        checks["potentially_weak"] = len(value) < 12 and checks["entropy_score"] < 0.5
        
        return checks
    
    async def validate_secret_accessibility_performance(self) -> List[SecretValidationResult]:
        """Test secret access performance and reliability"""
        results = []
        
        # Test with a known secret (database password)
        test_secret = "DB_PASSWORD"
        
        # Multiple access test
        access_times = []
        for i in range(3):
            start_time = time.time()
            success, value, details = await self.get_secret_value(test_secret)
            access_time = (time.time() - start_time) * 1000
            
            if success:
                access_times.append(access_time)
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        if access_times:
            avg_access_time = sum(access_times) / len(access_times)
            max_access_time = max(access_times)
            min_access_time = min(access_times)
            
            # Consider performance good if average access time is under 1 second
            is_performant = avg_access_time < 1000
            
            results.append(SecretValidationResult(
                secret_category="Performance",
                secret_name="Secret Access Performance",
                passed=is_performant,
                message=f"Secret access avg: {avg_access_time:.0f}ms {'(good)' if is_performant else '(slow)'}",
                validation_time_ms=avg_access_time,
                details={
                    "avg_access_time_ms": avg_access_time,
                    "min_access_time_ms": min_access_time,
                    "max_access_time_ms": max_access_time,
                    "samples": len(access_times),
                    "all_access_times": access_times
                }
            ))
        else:
            results.append(SecretValidationResult(
                secret_category="Performance",
                secret_name="Secret Access Performance",
                passed=False,
                message="Could not test secret access performance",
                details={"test_secret": test_secret, "failed_attempts": 3}
            ))
        
        return results
    
    async def validate_secret_versions(self) -> List[SecretValidationResult]:
        """Validate secret versioning and history"""
        results = []
        
        if not self.client:
            results.append(SecretValidationResult(
                secret_category="Versioning",
                secret_name="Version Check",
                passed=False,
                message="Cannot check versions - client not available"
            ))
            return results
        
        # Check version management for critical secrets
        critical_secrets = ["DB_PASSWORD", "TELEGRAM_TOKEN"]
        
        for secret_name in critical_secrets:
            try:
                start_time = time.time()
                parent = f"projects/{self.project_id}/secrets/{secret_name}"
                versions = list(self.client.list_secret_versions(request={"parent": parent}))
                check_time = (time.time() - start_time) * 1000
                
                version_count = len(versions)
                has_versions = version_count > 0
                
                results.append(SecretValidationResult(
                    secret_category="Versioning",
                    secret_name=f"Versions {secret_name}",
                    passed=has_versions,
                    message=f"Secret {secret_name} has {version_count} version(s)",
                    validation_time_ms=check_time,
                    details={
                        "secret_name": secret_name,
                        "version_count": version_count,
                        "versions": [v.name.split('/')[-1] for v in versions[:5]]  # First 5 versions
                    }
                ))
                
            except NotFound:
                results.append(SecretValidationResult(
                    secret_category="Versioning",
                    secret_name=f"Versions {secret_name}",
                    passed=False,
                    message=f"Secret {secret_name} not found for version check",
                    details={"secret_name": secret_name, "error": "not_found"}
                ))
                
            except Exception as e:
                results.append(SecretValidationResult(
                    secret_category="Versioning",
                    secret_name=f"Versions {secret_name}",
                    passed=False,
                    message=f"Version check failed for {secret_name}: {str(e)}",
                    details={"secret_name": secret_name, "error": str(e)}
                ))
        
        return results
    
    async def generate_security_report(self) -> Dict[str, Any]:
        """Generate security-focused report on secret status"""
        security_summary = {
            "critical_secrets_available": 0,
            "trading_keys_configured": 0,
            "api_keys_available": 0,
            "weak_secrets_detected": 0,
            "high_risk_secrets": [],
            "missing_critical_secrets": [],
            "security_recommendations": []
        }
        
        for result in self.results:
            if result.secret_category == "critical" and result.passed:
                security_summary["critical_secrets_available"] += 1
            elif result.secret_category == "critical" and not result.passed:
                security_summary["missing_critical_secrets"].append(result.secret_name)
            
            if result.secret_category == "trading_keys" and result.passed:
                security_summary["trading_keys_configured"] += 1
                security_summary["high_risk_secrets"].append(result.secret_name)
            
            if result.secret_category == "api_keys" and result.passed:
                security_summary["api_keys_available"] += 1
            
            # Check for potentially weak secrets
            if result.details and result.details.get("potentially_weak", False):
                security_summary["weak_secrets_detected"] += 1
        
        # Generate recommendations
        if security_summary["trading_keys_configured"] > 0:
            security_summary["security_recommendations"].append(
                "🚨 Trading keys are configured - ensure live trading is intentional"
            )
        
        if security_summary["missing_critical_secrets"]:
            security_summary["security_recommendations"].append(
                f"⚠️ Critical secrets missing: {', '.join(security_summary['missing_critical_secrets'])}"
            )
        
        if security_summary["weak_secrets_detected"] > 0:
            security_summary["security_recommendations"].append(
                f"🔐 {security_summary['weak_secrets_detected']} potentially weak secrets detected"
            )
        
        return security_summary
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all secret validations"""
        logger.info(f"🔐 Starting comprehensive secret validation for project {self.project_id}")
        
        # Initialize client
        client_result = await self.initialize_client()
        self.add_result(client_result)
        
        if not client_result.passed:
            return {
                "validation_complete": False,
                "error": "Failed to initialize Secret Manager client",
                "timestamp": datetime.utcnow().isoformat(),
                "project_id": self.project_id
            }
        
        start_time = time.time()
        
        # Validate each category
        for category_name, category_info in self.secret_categories.items():
            category_results = await self.validate_secret_category(category_name, category_info)
            for result in category_results:
                self.add_result(result)
        
        # Performance tests
        performance_results = await self.validate_secret_accessibility_performance()
        for result in performance_results:
            self.add_result(result)
        
        # Version checks
        version_results = await self.validate_secret_versions()
        for result in version_results:
            self.add_result(result)
        
        total_time = time.time() - start_time
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Categorize results
        results_by_category = {}
        for result in self.results:
            if result.secret_category not in results_by_category:
                results_by_category[result.secret_category] = []
            results_by_category[result.secret_category].append(asdict(result))
        
        # Generate security report
        security_report = await self.generate_security_report()
        
        summary = {
            "validation_complete": True,
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": self.project_id,
            "execution_time_seconds": total_time,
            "secrets_ready": failed_tests == 0,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate_percent": round(success_rate, 2)
            },
            "security_report": security_report,
            "results_by_category": results_by_category,
            "critical_failures": [
                asdict(r) for r in self.results 
                if not r.passed and r.secret_category in ["critical", "Client"]
            ]
        }
        
        # Log summary
        logger.info(f"✅ Secret validation complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        logger.info(f"🔐 Security status: {security_report['critical_secrets_available']}/4 critical secrets available")
        
        if security_report["trading_keys_configured"] > 0:
            logger.warning(f"🚨 {security_report['trading_keys_configured']} trading keys configured - LIVE TRADING ENABLED")
        
        if failed_tests > 0:
            logger.error(f"❌ {failed_tests} secret tests failed")
        else:
            logger.info("🎉 All secret validations passed")
        
        return summary

async def main():
    """Main secret validation runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Shyvr RLTE secrets comprehensively")
    parser.add_argument("--project-id", default="shvyr-ai-bots", help="GCP project ID")
    parser.add_argument("--output-file", help="Output file for validation results")
    
    args = parser.parse_args()
    
    validator = ComprehensiveSecretValidator(project_id=args.project_id)
    
    try:
        results = await validator.run_all_validations()
        
        # Save results
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_file = Path(__file__).parent.parent / "reports" / f"secret_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📄 Secret validation report saved to: {output_file}")
        
        # Exit with appropriate code
        # Don't fail on missing optional secrets, only on critical failures
        critical_failures = len(results.get("critical_failures", []))
        sys.exit(0 if critical_failures == 0 else 1)
        
    except Exception as e:
        logger.error(f"💥 Secret validation failed with exception: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())